#!/usr/bin/env python3
"""Per-field G-code recoverability baseline using metadata-only shortcuts.

Phase-1 verification artifact. Read-only.

Parses each window's `gcode_text` into structured fields:

  - command:  G0 / G1 / G2 / G3 / other
  - has_x, has_y, has_z, has_i, has_j, has_r (axis presence flags)
  - x_sign, y_sign, z_sign  (sign of motion command, +/- or 0 if absent)
  - feed_rate F (regression target; nan if absent)
  - radius / depth proxies (regression: R, Z values)

Then trains lightweight metadata-only predictors (using window_index,
total_windows, operation_type, source_file_hash — the same fields the
decoder also sees) and reports per-field accuracy / MAE on val and test.

This establishes the FLOOR: any improvement the sensor pathway brings must
exceed these numbers. It does NOT use the actual V7 decoder — that's
deferred to Phase 5 retrain comparison.

Output: outputs/decoder20260511/audit/recoverability_baseline.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# -------- field parsing --------

CMD_RE = re.compile(r"\b(G0?[0-3])\b", re.IGNORECASE)
PARAM_RE = re.compile(r"([XYZIJKRF])(-?\d+\.?\d*)", re.IGNORECASE)


def parse_fields(text: str) -> dict[str, Any]:
    t = text.strip().upper()
    out: dict[str, Any] = {
        "command": "OTHER",
        "has_x": 0, "has_y": 0, "has_z": 0,
        "has_i": 0, "has_j": 0, "has_r": 0, "has_f": 0,
        "x_sign": 0, "y_sign": 0, "z_sign": 0,
        "x_val": np.nan, "y_val": np.nan, "z_val": np.nan,
        "i_val": np.nan, "j_val": np.nan, "r_val": np.nan, "f_val": np.nan,
    }
    if not t:
        return out
    m = CMD_RE.search(t)
    if m:
        cmd = m.group(1).upper()
        # Normalize G00 → G0, G01 → G1, etc.
        cmd = "G" + str(int(cmd[1:]))
        if cmd in ("G0", "G1", "G2", "G3"):
            out["command"] = cmd
    for addr, val in PARAM_RE.findall(t):
        a = addr.upper()
        try:
            v = float(val)
        except ValueError:
            continue
        key_has = f"has_{a.lower()}"
        key_val = f"{a.lower()}_val"
        if key_has in out:
            out[key_has] = 1
        if key_val in out:
            out[key_val] = v
        if a in ("X", "Y", "Z"):
            out[f"{a.lower()}_sign"] = int(np.sign(v))
    return out


def _hash_str(s: str, mod: int = 10_000) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % mod


def _load_split(npz_path: Path) -> dict[str, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    return {
        "gcode_texts": np.array([str(t) for t in d["gcode_texts"]]),
        "window_index": d["window_index"].astype(np.int64) if "window_index" in d.files else None,
        "total_windows": d["total_windows"].astype(np.int64) if "total_windows" in d.files else None,
        "operation_type": d["operation_type"].astype(np.int64) if "operation_type" in d.files else None,
        "source_file": np.array([str(s) for s in d["source_file"]]) if "source_file" in d.files else None,
    }


def _build_features(split: dict[str, np.ndarray], sf_cache: dict[str, int]) -> np.ndarray:
    n = len(split["gcode_texts"])
    wi = split["window_index"] if split["window_index"] is not None else np.zeros(n, dtype=np.int64)
    tw = split["total_windows"] if split["total_windows"] is not None else np.ones(n, dtype=np.int64)
    op = split["operation_type"] if split["operation_type"] is not None else np.zeros(n, dtype=np.int64)
    sf = split["source_file"]
    sf_hash = (
        np.array([sf_cache.setdefault(s, _hash_str(s)) for s in sf], dtype=np.float32)
        if sf is not None else np.zeros(n, dtype=np.float32)
    )
    feats = np.stack(
        [
            wi.astype(np.float32),
            tw.astype(np.float32),
            (wi.astype(np.float32) / np.maximum(tw.astype(np.float32), 1.0)),
            op.astype(np.float32),
            sf_hash,
        ],
        axis=1,
    )
    return feats


def parse_split(split: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    fields = [parse_fields(t) for t in split["gcode_texts"]]
    out = {k: np.array([f[k] for f in fields]) for k in fields[0].keys()}
    return out


def class_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true)
    if not mask.any():
        return float("nan")
    return float(np.abs(y_true[mask] - y_pred[mask]).mean())


def audit_fold(fold_dir: Path) -> dict[str, Any]:
    train = _load_split(fold_dir / "train_sequences.npz")
    val = _load_split(fold_dir / "val_sequences.npz")
    test = _load_split(fold_dir / "test_sequences.npz")

    tr_f = parse_split(train)
    va_f = parse_split(val)
    te_f = parse_split(test)

    sf_cache: dict[str, int] = {}
    X_tr = _build_features(train, sf_cache)
    X_va = _build_features(val, sf_cache)
    X_te = _build_features(test, sf_cache)

    report: dict[str, Any] = {
        "fold_dir": str(fold_dir),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "field_distributions": {},
        "metadata_baseline": {},
    }

    # 1) Field-level distributions across train/test
    for field in ["command", "has_x", "has_y", "has_z", "has_f", "has_r", "x_sign", "y_sign"]:
        u, c = np.unique(tr_f[field], return_counts=True)
        report["field_distributions"][f"{field}_train"] = {str(k): int(v) for k, v in zip(u, c)}
        u, c = np.unique(te_f[field], return_counts=True)
        report["field_distributions"][f"{field}_test"] = {str(k): int(v) for k, v in zip(u, c)}

    # 2) Categorical fields: metadata-only XGBoost baseline
    for field in ["command", "has_x", "has_y", "has_z", "has_r", "x_sign", "y_sign"]:
        y_tr = tr_f[field]
        y_va = va_f[field]
        y_te = te_f[field]
        if y_tr.dtype.kind == "U":  # strings → encode
            classes = sorted(set(y_tr.tolist()))
            label_to_id = {c: i for i, c in enumerate(classes)}
            unknown = -1
            y_tr_e = np.array([label_to_id.get(v, unknown) for v in y_tr])
            y_va_e = np.array([label_to_id.get(v, unknown) for v in y_va])
            y_te_e = np.array([label_to_id.get(v, unknown) for v in y_te])
            n_cls = len(classes)
        else:
            y_tr_e = y_tr.astype(np.int64)
            y_va_e = y_va.astype(np.int64)
            y_te_e = y_te.astype(np.int64)
            classes = sorted(set(y_tr_e.tolist()))
            n_cls = max(int(y_tr_e.max()) + 1, 2) if y_tr_e.size else 2

        # Majority-class baseline
        if y_tr_e.size:
            vals, counts = np.unique(y_tr_e[y_tr_e >= 0], return_counts=True)
            maj = int(vals[counts.argmax()]) if vals.size else 0
        else:
            maj = 0
        maj_val = class_acc(y_va_e, np.full_like(y_va_e, maj))
        maj_test = class_acc(y_te_e, np.full_like(y_te_e, maj))

        xgb_val = xgb_test = float("nan")
        if HAS_XGB and n_cls > 1 and y_tr_e.size:
            try:
                if n_cls == 2 and set(np.unique(y_tr_e).tolist()) <= {0, 1}:
                    clf = xgb.XGBClassifier(
                        n_estimators=200, max_depth=6, learning_rate=0.1,
                        objective="binary:logistic", tree_method="hist", verbosity=0, n_jobs=4,
                    )
                else:
                    clf = xgb.XGBClassifier(
                        n_estimators=200, max_depth=6, learning_rate=0.1,
                        objective="multi:softmax", num_class=n_cls, tree_method="hist", verbosity=0, n_jobs=4,
                    )
                clf.fit(X_tr, y_tr_e)
                xgb_val = class_acc(y_va_e, clf.predict(X_va))
                xgb_test = class_acc(y_te_e, clf.predict(X_te))
            except Exception:
                pass

        report["metadata_baseline"][field] = {
            "kind": "classification",
            "n_classes": int(n_cls),
            "majority_val_acc": maj_val,
            "majority_test_acc": maj_test,
            "xgb_val_acc": xgb_val,
            "xgb_test_acc": xgb_test,
        }

    # 3) Regression fields: x_val, y_val, z_val, f_val, r_val
    for field in ["x_val", "y_val", "z_val", "f_val", "r_val"]:
        y_tr = tr_f[field].astype(np.float64)
        y_va = va_f[field].astype(np.float64)
        y_te = te_f[field].astype(np.float64)

        mask_tr = ~np.isnan(y_tr)
        if mask_tr.sum() < 5:
            report["metadata_baseline"][field] = {"kind": "regression", "note": "too few non-nan train values"}
            continue

        # Mean-fill baseline
        mu = float(y_tr[mask_tr].mean())
        mae_mean_val = mae(y_va, np.full_like(y_va, mu))
        mae_mean_test = mae(y_te, np.full_like(y_te, mu))

        # XGBoost regressor on non-nan train
        xgb_mae_val = xgb_mae_test = float("nan")
        if HAS_XGB:
            try:
                reg = xgb.XGBRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    tree_method="hist", verbosity=0, n_jobs=4,
                )
                reg.fit(X_tr[mask_tr], y_tr[mask_tr])
                mask_va = ~np.isnan(y_va)
                mask_te = ~np.isnan(y_te)
                if mask_va.any():
                    xgb_mae_val = float(np.abs(reg.predict(X_va[mask_va]) - y_va[mask_va]).mean())
                if mask_te.any():
                    xgb_mae_test = float(np.abs(reg.predict(X_te[mask_te]) - y_te[mask_te]).mean())
            except Exception:
                pass

        report["metadata_baseline"][field] = {
            "kind": "regression",
            "n_train_non_nan": int(mask_tr.sum()),
            "mean_target_train": mu,
            "mae_mean_val": mae_mean_val,
            "mae_mean_test": mae_mean_test,
            "mae_xgb_val": xgb_mae_val,
            "mae_xgb_test": xgb_mae_test,
        }

    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Per-field recoverability baseline")
    p.add_argument("--preproc-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = p.parse_args()

    reports = []
    for fold in args.folds:
        fold_dir = args.preproc_root / f"fold_{fold}"
        if not fold_dir.exists():
            reports.append({"fold": fold, "error": f"missing {fold_dir}"})
            continue
        r = audit_fold(fold_dir)
        r["fold"] = fold
        reports.append(r)
        # Brief one-line summary
        cmd = r["metadata_baseline"].get("command", {})
        print(
            f"fold {fold}: cmd maj={cmd.get('majority_test_acc', float('nan')):.3f} "
            f"xgb={cmd.get('xgb_test_acc', float('nan')):.3f} "
            f"has_x_xgb={r['metadata_baseline'].get('has_x',{}).get('xgb_test_acc', float('nan')):.3f} "
            f"x_sign_xgb={r['metadata_baseline'].get('x_sign',{}).get('xgb_test_acc', float('nan')):.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"reports": reports}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
