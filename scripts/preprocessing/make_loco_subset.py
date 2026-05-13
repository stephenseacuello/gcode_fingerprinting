#!/usr/bin/env python3
"""Phase B5 helper: derive LOCO (leave-one-class-out) NPZ subsets from V8 preprocessed data.

For a given operation class (e.g., "face"), build a subset where:
  - train: all samples whose operation_type_name != holdout
  - val:   same (subset of original val)
  - test:  ONLY samples whose operation_type_name == holdout

Uses the existing fold-1 NPZ as the source (so we get one LOCO setup per class
without 5x compute).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

OPERATION_CLASSES = [
    "adaptive", "adaptive150025", "face", "face150025",
    "pocket", "pocket150025",
    "damageadaptive", "damageface", "damagepocket",
]


def _filter_npz(src_npz: Path, dst_npz: Path, mask: np.ndarray) -> int:
    """Save a copy of src_npz keeping only rows where mask==True."""
    d = np.load(src_npz, allow_pickle=True)
    save_kwargs = {}
    n_kept = int(mask.sum())
    for k in d.files:
        v = d[k]
        if v.ndim == 0 or k in ("label_mode", "total_windows"):  # scalar fields preserved
            save_kwargs[k] = v
            continue
        if v.shape[0] != mask.shape[0]:
            save_kwargs[k] = v
            continue
        save_kwargs[k] = v[mask]
    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst_npz, **save_kwargs)
    return n_kept


def make_loco(src_dir: Path, dst_dir: Path, holdout_class: str) -> dict:
    train_src = src_dir / "train_sequences.npz"
    val_src = src_dir / "val_sequences.npz"
    test_src = src_dir / "test_sequences.npz"

    out = {"holdout_class": holdout_class, "src_dir": str(src_dir), "dst_dir": str(dst_dir)}

    # TRAIN: remove holdout
    d_tr = np.load(train_src, allow_pickle=True)
    op_names_tr = np.array([str(s) for s in d_tr["operation_type_names"]])
    train_mask = op_names_tr != holdout_class
    n_train_kept = _filter_npz(train_src, dst_dir / "train_sequences.npz", train_mask)
    n_train_drop = int(len(op_names_tr) - n_train_kept)

    # VAL: remove holdout
    d_va = np.load(val_src, allow_pickle=True)
    op_names_va = np.array([str(s) for s in d_va["operation_type_names"]])
    val_mask = op_names_va != holdout_class
    n_val_kept = _filter_npz(val_src, dst_dir / "val_sequences.npz", val_mask)
    n_val_drop = int(len(op_names_va) - n_val_kept)

    # TEST: KEEP ONLY holdout. But we want test samples not seen during training.
    # Pull holdout samples from ALL THREE splits (train+val+test) to maximize
    # the test population.
    d_te = np.load(test_src, allow_pickle=True)
    op_names_te = np.array([str(s) for s in d_te["operation_type_names"]])
    # Just use the test-split holdout samples to keep semantics clean.
    test_mask = op_names_te == holdout_class
    n_test_kept = _filter_npz(test_src, dst_dir / "test_sequences.npz", test_mask)

    # Copy metadata + scaler + file_split unchanged
    for name in ("metadata.json", "scaler_stats.json"):
        src = src_dir / name
        if src.exists():
            shutil.copy(src, dst_dir / name)
    fs_src = src_dir / "file_split.json"
    if fs_src.exists():
        fs = json.loads(fs_src.read_text())
        fs["loco_holdout_class"] = holdout_class
        (dst_dir / "file_split.json").write_text(json.dumps(fs, indent=2))

    out.update({
        "n_train_kept": n_train_kept,
        "n_train_dropped": n_train_drop,
        "n_val_kept": n_val_kept,
        "n_val_dropped": n_val_drop,
        "n_test_kept": n_test_kept,
    })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", type=Path, required=True,
                   help="Existing V8 NPZ fold directory, e.g. preprocessed_f98/per_row/fold_1")
    p.add_argument("--dst-root", type=Path, required=True,
                   help="Destination root; one subdir per holdout class will be created")
    p.add_argument("--holdouts", nargs="+", default=OPERATION_CLASSES)
    args = p.parse_args()

    log = []
    for cls in args.holdouts:
        dst = args.dst_root / f"holdout_{cls}"
        result = make_loco(args.src_dir, dst, cls)
        log.append(result)
        print(f"  {cls:>22s}: train={result['n_train_kept']} (-{result['n_train_dropped']}), "
              f"val={result['n_val_kept']} (-{result['n_val_dropped']}), test={result['n_test_kept']}")

    (args.dst_root / "loco_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nwrote LOCO subsets to {args.dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
