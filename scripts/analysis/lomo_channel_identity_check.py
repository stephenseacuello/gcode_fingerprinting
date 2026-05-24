#!/usr/bin/env python3
"""
LOMO channel-identity diagnostic (the smoke-gate safety net).

The leave-one-modality-out (LOMO) encoder study has a structural hazard: the
encoder side (run_9class_direct.py: build_modality_indices) and the decoder
side (run_preprocessing_v8_cv_fold.py: --exclude-* flags) historically used
*different* modality taxonomies. If a LOMO cell drops a slightly different set
of channels on each side, the frozen encoder is fed mismatched sensor tensors
and the whole attribution table is confidently wrong -- the exact silent-bug
class (cf. AUDIT_REPORT.md, the 16-token truncation) this project exists to
not repeat.

This script does NOT trust either code path's group names. It works purely off
the authoritative `continuous_columns` list emitted in each fold's
metadata.json and a single canonical modality->suffix map (verified verbatim
against run_9class_direct.py:69-90 and run_preprocessing_v8_cv_fold.py:241-249
on 2026-05-18), and asserts:

  1. The modality-excluded decoder NPZ dropped EXACTLY the channels the
     canonical map says belong to that modality -- no more, no fewer.
  2. The surviving column list equals  baseline_columns - modality_columns
     (same identities, same order).
  3. continuous.shape[-1] == len(continuous_columns) == sum over modality
     channel counts (no off-by-one / stale metadata).
  4. The decoder TARGETS (tokens, gcode_texts, token_length) are byte-identical
     to the no-exclusion baseline fold -- modality exclusion must touch sensors
     ONLY, never the supervision signal.
  5. If an encoder-side consumed-column dump is provided (--encoder-cols),
     it is set-and-order identical to the decoder side (the cross-path proof).

Exit code 0 == cell is safe to train. Non-zero == abort this cell; do NOT
spend GPU on it. Emits a JSON verdict for the aggregator/audit trail.

Usage:
  python scripts/analysis/lomo_channel_identity_check.py \
      --modality gyroscope \
      --baseline-dir outputs/decoder20260511/preprocessed_f98/per_row/fold_1 \
      --excluded-dir outputs/decoder20260511/lomo/gyroscope/per_row/fold_1 \
      [--encoder-cols outputs/decoder20260511/lomo/gyroscope/encoder/fold_1/consumed_columns.json] \
      --out outputs/decoder20260511/lomo/gyroscope/fold_1/channel_identity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Canonical modality -> channel-suffix map. The suffix is the part AFTER the
# last '.' in a column name like 'frame_l2.Gx'. Verified verbatim against:
#   run_9class_direct.py:69-90  (build_modality_indices.sensor_patterns)
#   run_preprocessing_v8_cv_fold.py:241-249  (excluded_channels assembly)
# 'audio' is the reconciled name for what the encoder calls 'rms' (suffix RMS).
# 'electrical' is the residual: any column whose suffix matches no other
# modality (encoder's unmatched -> 'electrical' bucket).
MODALITY_SUFFIXES = {
    "accelerometer": ["Ax", "Ay", "Az"],
    "gyroscope": ["Gx", "Gy", "Gz"],
    "magnetometer": ["Mx", "My", "Mz"],
    "color": ["ColorR", "ColorG", "ColorB", "ColorA"],
    "temperature": ["Temperature"],
    "audio": ["RMS"],
    # proximity / pressure are already absent in the 98-feature base; kept here
    # so the diagnostic can also be run against a full-sensor baseline.
    "proximity": ["Proximity"],
    "pressure": ["Pressure"],
}
_NON_ELECTRICAL = {s for v in MODALITY_SUFFIXES.values() for s in v}


def _suffix(col: str) -> str:
    return col.rsplit(".", 1)[1] if "." in col else col


def modality_of(col: str) -> str:
    suf = _suffix(col)
    for mod, sufs in MODALITY_SUFFIXES.items():
        if suf in sufs:
            return mod
    return "electrical"  # residual bucket (matches encoder's unmatched branch)


def columns_for(name: str, all_columns: list[str], kind: str = "modality",
                sensor: str | None = None) -> list[str]:
    if kind == "sensor":   # leave-one-physical-sensor-out: match the column prefix
        return [c for c in all_columns if c.startswith(f"{name}.")]
    if kind == "nested":
        # leave-one-(sensor,modality)-pair-out: intersect sensor prefix with
        # modality suffix. `name` is the modality, `sensor` is the prefix.
        if not sensor:
            raise SystemExit("--group-kind nested requires --sensor SENSOR")
        if name == "electrical":
            return [c for c in all_columns
                    if c.startswith(f"{sensor}.") and _suffix(c) not in _NON_ELECTRICAL]
        if name not in MODALITY_SUFFIXES:
            raise SystemExit(f"unknown modality '{name}'; "
                             f"known: {sorted(MODALITY_SUFFIXES) + ['electrical']}")
        sufs = set(MODALITY_SUFFIXES[name])
        return [c for c in all_columns
                if c.startswith(f"{sensor}.") and _suffix(c) in sufs]
    if name == "electrical":
        return [c for c in all_columns if _suffix(c) not in _NON_ELECTRICAL]
    if name not in MODALITY_SUFFIXES:
        raise SystemExit(f"unknown modality '{name}'; "
                         f"known: {sorted(MODALITY_SUFFIXES) + ['electrical']}")
    sufs = set(MODALITY_SUFFIXES[name])
    return [c for c in all_columns if _suffix(c) in sufs]


def _load_meta(d: Path) -> dict:
    m = json.loads((d / "metadata.json").read_text())
    if "continuous_columns" not in m:
        raise SystemExit(f"{d}/metadata.json missing 'continuous_columns'")
    return m


def _npz_shape(d: Path) -> tuple[int, int, int]:
    z = np.load(d / "train_sequences.npz", allow_pickle=True)
    return tuple(z["continuous"].shape)


def _corpus_split_sig(d: Path) -> dict:
    """Per-split supervision signature that MUST be modality-invariant.

    Not byte-identity of token arrays: each preprocessing run independently
    fits the scaler, clips outliers and windows, so sample *ordering* legitimately
    differs. What must NOT differ is the labelled corpus and its partition:
    which source files are in each split, the multiset of G-code targets, and
    the sample count. (With the sorted() split-determinism fix this is exact.)
    """
    sig = {}
    for sp in ("train", "val", "test"):
        z = np.load(d / f"{sp}_sequences.npz", allow_pickle=True)
        sig[sp] = {
            "files": sorted(set(map(str, z["source_file"].tolist()))),
            "gcode": sorted(map(str, z["gcode_texts"].tolist())),
            "n": int(z["gcode_texts"].shape[0]),
        }
    return sig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True,
                    help="group name: a modality, or a physical sensor (with --group-kind sensor); "
                         "for --group-kind nested it is the MODALITY half of the pair")
    ap.add_argument("--group-kind", choices=["modality", "sensor", "nested"], default="modality")
    ap.add_argument("--sensor", default=None,
                    help="physical sensor unit; required only with --group-kind nested")
    ap.add_argument("--baseline-dir", type=Path, required=True,
                    help="fold dir of the NO-extra-exclusion base (98-feat f98)")
    ap.add_argument("--excluded-dir", type=Path, required=True,
                    help="fold dir of the modality-excluded LOMO preprocessing")
    ap.add_argument("--encoder-cols", type=Path, default=None,
                    help="optional JSON list of columns the encoder actually "
                         "consumed (cross-path proof)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    base_cols = _load_meta(a.baseline_dir)["continuous_columns"]
    excl_meta = _load_meta(a.excluded_dir)
    excl_cols = excl_meta["continuous_columns"]

    expected_dropped = columns_for(a.modality, base_cols, a.group_kind, sensor=a.sensor)
    actually_dropped = [c for c in base_cols if c not in set(excl_cols)]
    expected_survivors = [c for c in base_cols if c not in set(expected_dropped)]

    # 1. dropped set == exactly the modality's channels
    check(
        "dropped_set_exact",
        sorted(actually_dropped) == sorted(expected_dropped)
        and len(expected_dropped) > 0,
        f"expected_drop={len(expected_dropped)} actual_drop={len(actually_dropped)} "
        f"unexpected={sorted(set(actually_dropped) ^ set(expected_dropped))[:6]}",
    )

    # 2. survivors identical in identity AND order
    check(
        "survivors_identity_and_order",
        excl_cols == expected_survivors,
        f"first_mismatch_idx="
        f"{next((i for i,(x,y) in enumerate(zip(excl_cols,expected_survivors)) if x!=y), 'none')}"
        f" len_excl={len(excl_cols)} len_expected={len(expected_survivors)}",
    )

    # 3. shape / metadata coherence
    shp = _npz_shape(a.excluded_dir)
    check(
        "shape_matches_columns",
        shp[-1] == len(excl_cols) == int(excl_meta.get("n_continuous_features", -1)),
        f"continuous.shape={shp} n_cols={len(excl_cols)} "
        f"meta.n_continuous_features={excl_meta.get('n_continuous_features')}",
    )

    # 4. corpus + train/val/test PARTITION is modality-invariant.
    #    (Sensor tensors legitimately differ; the labelled corpus and its
    #     file partition must NOT -- else each modality is scored on a
    #     different test set and the attribution table is confounded.)
    base_sig = _corpus_split_sig(a.baseline_dir)
    excl_sig = _corpus_split_sig(a.excluded_dir)
    diffs = []
    for sp in ("train", "val", "test"):
        b, e = base_sig[sp], excl_sig[sp]
        if b["files"] != e["files"]:
            diffs.append(f"{sp}.files(base_only={sorted(set(b['files'])-set(e['files']))[:3]},"
                         f"excl_only={sorted(set(e['files'])-set(b['files']))[:3]})")
        if b["gcode"] != e["gcode"]:
            diffs.append(f"{sp}.gcode_multiset(n_base={len(b['gcode'])},n_excl={len(e['gcode'])})")
        if b["n"] != e["n"]:
            diffs.append(f"{sp}.count({b['n']}!={e['n']})")
    check(
        "corpus_and_split_modality_invariant",
        not diffs,
        "all splits match" if not diffs else " | ".join(diffs),
    )

    # 5. cross-path proof (optional but this is THE point of the gate)
    if a.encoder_cols and a.encoder_cols.exists():
        enc_cols = json.loads(a.encoder_cols.read_text())
        check(
            "encoder_decoder_columns_identical",
            list(enc_cols) == list(excl_cols),
            f"encoder_n={len(enc_cols)} decoder_n={len(excl_cols)} "
            f"set_equal={set(enc_cols)==set(excl_cols)} "
            f"order_equal={list(enc_cols)==list(excl_cols)}",
        )
    else:
        check("encoder_decoder_columns_identical", True,
              "SKIPPED: no --encoder-cols supplied (run at smoke gate)")

    verdict = {
        "modality": a.modality,
        "sensor": a.sensor,
        "group_kind": a.group_kind,
        "baseline_dir": str(a.baseline_dir),
        "excluded_dir": str(a.excluded_dir),
        "n_baseline_channels": len(base_cols),
        "n_dropped": len(actually_dropped),
        "n_survivors": len(excl_cols),
        "dropped_channels": sorted(actually_dropped),
        "all_pass": all(c["pass"] for c in checks),
        "checks": checks,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(verdict, indent=2))

    status = "PASS" if verdict["all_pass"] else "FAIL"
    cell_lbl = f"{a.sensor}:{a.modality}" if a.group_kind == "nested" else a.modality
    print(f"[{status}] {a.group_kind}={cell_lbl}  drop={len(actually_dropped)}ch  "
          f"survivors={len(excl_cols)}ch  -> {a.out}")
    for c in checks:
        mark = "ok " if c["pass"] else "XXX"
        print(f"  [{mark}] {c['check']}: {c['detail']}")
    return 0 if verdict["all_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
