#!/usr/bin/env python3
"""Audit positional / metadata shortcut leakage into the G-code label.

Phase-1 verification artifact. Read-only.

For each V7 fold, fit lightweight predictors that try to recover the window's
G-code label using ONLY metadata fields the decoder also has access to:

  - window_index
  - total_windows
  - normalized_position (window_index / total_windows)
  - operation_type
  - source_file (hashed to int)

If these are highly predictive on a held-out split, the decoder doesn't need
sensor input to score well — the sensor pathway is bypassable. This quantifies
the concern Stephen and Romesh raised in the 2026-04-28 meeting.

Models used:
  - Majority-class baseline (sanity floor)
  - Logistic regression on a few engineered features
  - XGBoost if available (richer combinations)

Output: outputs/decoder20260511/audit/shortcut_leakage.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def _hash_str(s: str, mod: int = 2**31 - 1) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % mod


def _load_split(npz_path: Path) -> dict[str, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    return {
        "gcode_texts": np.array([str(t).strip() for t in d["gcode_texts"]]),
        "window_index": d["window_index"].astype(np.int64) if "window_index" in d.files else None,
        "total_windows": d["total_windows"].astype(np.int64) if "total_windows" in d.files else None,
        "operation_type": d["operation_type"].astype(np.int64) if "operation_type" in d.files else None,
        "source_file": np.array([str(s) for s in d["source_file"]]) if "source_file" in d.files else None,
    }


def _build_features(split: dict[str, np.ndarray], source_file_hashes: dict[str, int]) -> np.ndarray:
    feats = []
    wi = split["window_index"]
    tw = split["total_windows"]
    op = split["operation_type"]
    sf = split["source_file"]
    n = len(split["gcode_texts"])
    if wi is None:
        wi = np.zeros(n, dtype=np.int64)
    if tw is None:
        tw = np.ones(n, dtype=np.int64)
    feats.append(wi.astype(np.float32))
    feats.append(tw.astype(np.float32))
    feats.append((wi.astype(np.float32) / np.maximum(tw.astype(np.float32), 1.0)))
    if op is not None:
        feats.append(op.astype(np.float32))
    else:
        feats.append(np.zeros(n, dtype=np.float32))
    if sf is not None:
        hashed = np.array(
            [source_file_hashes.setdefault(s, _hash_str(s) % 10_000) for s in sf],
            dtype=np.float32,
        )
        feats.append(hashed)
    else:
        feats.append(np.zeros(n, dtype=np.float32))
    return np.stack(feats, axis=1)


def audit_fold(fold_dir: Path) -> dict[str, Any]:
    train = _load_split(fold_dir / "train_sequences.npz")
    val = _load_split(fold_dir / "val_sequences.npz")
    test = _load_split(fold_dir / "test_sequences.npz")

    # Build a global label encoder over the union of train labels.
    label_to_id: dict[str, int] = {}
    for t in train["gcode_texts"]:
        if t not in label_to_id:
            label_to_id[t] = len(label_to_id)
    UNSEEN = -1

    def encode_labels(arr: np.ndarray) -> np.ndarray:
        return np.array([label_to_id.get(t, UNSEEN) for t in arr], dtype=np.int64)

    y_train = encode_labels(train["gcode_texts"])
    y_val = encode_labels(val["gcode_texts"])
    y_test = encode_labels(test["gcode_texts"])

    n_train = len(y_train)
    n_classes_train = len(label_to_id)
    val_unseen = int((y_val == UNSEEN).sum())
    test_unseen = int((y_test == UNSEEN).sum())

    # Source file hash registry shared across splits
    source_file_hashes: dict[str, int] = {}
    X_train = _build_features(train, source_file_hashes)
    X_val = _build_features(val, source_file_hashes)
    X_test = _build_features(test, source_file_hashes)

    # 1) Majority-class baseline
    unique, counts = np.unique(y_train, return_counts=True)
    majority_label = int(unique[counts.argmax()])
    majority_test_acc = float((y_test == majority_label).mean())
    majority_val_acc = float((y_val == majority_label).mean())

    # 2) Operation-type baseline (predict most-common label conditional on op type)
    op_to_label: dict[int, int] = {}
    if train["operation_type"] is not None:
        for op in np.unique(train["operation_type"]):
            mask = train["operation_type"] == op
            ys = y_train[mask]
            if ys.size:
                op_to_label[int(op)] = int(np.bincount(ys).argmax())
    def op_baseline(arr: np.ndarray, ops: np.ndarray) -> float:
        if ops is None:
            return float("nan")
        preds = np.array([op_to_label.get(int(o), majority_label) for o in ops], dtype=np.int64)
        return float((preds == arr).mean())
    op_val_acc = op_baseline(y_val, val["operation_type"])
    op_test_acc = op_baseline(y_test, test["operation_type"])

    # 3) (operation_type, window_index) lookup
    pair_to_label: dict[tuple[int, int], int] = {}
    if train["operation_type"] is not None and train["window_index"] is not None:
        for op, wi in zip(train["operation_type"], train["window_index"]):
            key = (int(op), int(wi))
            pair_to_label.setdefault(key, [])
            pair_to_label[key].append(label_to_id.get(train["gcode_texts"][len(pair_to_label[key]) - 1], -1))
    # Rebuild properly: for each (op, wi) pair, pick the modal label
    if train["operation_type"] is not None and train["window_index"] is not None:
        pair_lists: dict[tuple[int, int], list[int]] = {}
        for op, wi, y in zip(train["operation_type"], train["window_index"], y_train):
            pair_lists.setdefault((int(op), int(wi)), []).append(int(y))
        pair_modes = {k: int(np.bincount(v).argmax()) for k, v in pair_lists.items()}
    else:
        pair_modes = {}
    def pair_baseline(arr: np.ndarray, ops: np.ndarray, wis: np.ndarray) -> float:
        if ops is None or wis is None:
            return float("nan")
        preds = np.array(
            [pair_modes.get((int(o), int(w)), majority_label) for o, w in zip(ops, wis)],
            dtype=np.int64,
        )
        return float((preds == arr).mean())
    pair_val_acc = pair_baseline(y_val, val["operation_type"], val["window_index"])
    pair_test_acc = pair_baseline(y_test, test["operation_type"], test["window_index"])

    # 4) Logistic regression on engineered features
    lr_val_acc = lr_test_acc = float("nan")
    if HAS_SKLEARN and n_classes_train > 1:
        try:
            lr = LogisticRegression(max_iter=1000, multi_class="auto")
            lr.fit(X_train, y_train)
            mask_val = y_val != UNSEEN
            mask_test = y_test != UNSEEN
            if mask_val.any():
                lr_val_acc = float((lr.predict(X_val[mask_val]) == y_val[mask_val]).mean())
            if mask_test.any():
                lr_test_acc = float((lr.predict(X_test[mask_test]) == y_test[mask_test]).mean())
        except Exception as e:
            lr_val_acc = lr_test_acc = float("nan")
            lr_error = repr(e)
        else:
            lr_error = None
    else:
        lr_error = "sklearn not available or too few classes"

    # 5) XGBoost on the same features
    xgb_val_acc = xgb_test_acc = float("nan")
    xgb_feature_importance = None
    if HAS_XGB and n_classes_train > 1:
        try:
            clf = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective="multi:softmax",
                num_class=n_classes_train,
                tree_method="hist",
                verbosity=0,
                n_jobs=4,
            )
            clf.fit(X_train, y_train)
            mask_val = y_val != UNSEEN
            mask_test = y_test != UNSEEN
            if mask_val.any():
                xgb_val_acc = float((clf.predict(X_val[mask_val]) == y_val[mask_val]).mean())
            if mask_test.any():
                xgb_test_acc = float((clf.predict(X_test[mask_test]) == y_test[mask_test]).mean())
            xgb_feature_importance = [float(x) for x in clf.feature_importances_]
        except Exception as e:
            xgb_error = repr(e)
        else:
            xgb_error = None
    else:
        xgb_error = "xgboost not available or too few classes"

    return {
        "fold_dir": str(fold_dir),
        "n_train": int(n_train),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_classes_train": int(n_classes_train),
        "val_unseen_labels": val_unseen,
        "test_unseen_labels": test_unseen,
        "feature_names": ["window_index", "total_windows", "norm_position", "operation_type", "source_file_hash"],
        "majority_class": {
            "label_id": majority_label,
            "val_acc": majority_val_acc,
            "test_acc": majority_test_acc,
        },
        "operation_type_baseline": {
            "val_acc": op_val_acc,
            "test_acc": op_test_acc,
        },
        "operation_x_window_lookup": {
            "val_acc": pair_val_acc,
            "test_acc": pair_test_acc,
        },
        "logistic_regression": {
            "val_acc": lr_val_acc,
            "test_acc": lr_test_acc,
            "error": lr_error,
        },
        "xgboost": {
            "val_acc": xgb_val_acc,
            "test_acc": xgb_test_acc,
            "feature_importance": xgb_feature_importance,
            "error": xgb_error,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Shortcut-leakage audit")
    p.add_argument("--preproc-root", type=Path, required=True, help="e.g. outputs/decoder20260304/preprocessed_v7")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = p.parse_args()

    reports = []
    for fold in args.folds:
        fold_dir = args.preproc_root / f"fold_{fold}"
        if not fold_dir.exists():
            reports.append({"fold": fold, "error": f"missing {fold_dir}"})
            continue
        try:
            r = audit_fold(fold_dir)
            r["fold"] = fold
            reports.append(r)
            print(
                f"fold {fold}: classes={r['n_classes_train']:>2d} "
                f"maj_test={r['majority_class']['test_acc']:.3f} "
                f"op_test={r['operation_type_baseline']['test_acc']:.3f} "
                f"(op,wi)_test={r['operation_x_window_lookup']['test_acc']:.3f} "
                f"LR_test={r['logistic_regression']['val_acc']:.3f}/{r['logistic_regression']['test_acc']:.3f} "
                f"XGB_test={r['xgboost']['val_acc']:.3f}/{r['xgboost']['test_acc']:.3f}"
            )
        except Exception as e:
            reports.append({"fold": fold, "error": repr(e)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"reports": reports}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
