#!/usr/bin/env python3
"""Quantify train/test G-code line overlap.

For each of the 5 per_row folds, compute:
  - Distinct G-code lines in train, in test
  - Lines that appear verbatim in BOTH train and test
  - Lines that appear in test but never in train (zero-shot)
  - Per-fold and aggregate numbers
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _split_lines(arr):
    """For full_window mode, gcode_texts entries are multi-line strings joined by \\n.
    For per_row, each entry is a single line. We return a flat iterator of individual lines.
    """
    out = []
    for s in arr:
        for ln in str(s).split("\n"):
            ln = ln.strip()
            if ln:
                out.append(ln)
    return out


def fold_stats(fold: int, mode: str = "per_row"):
    base = REPO / "outputs" / "decoder20260511" / "preprocessed_f98" / mode / f"fold_{fold}"
    tr = np.load(base / "train_sequences.npz", allow_pickle=True)["gcode_texts"]
    te = np.load(base / "test_sequences.npz", allow_pickle=True)["gcode_texts"]

    # window-level overlap
    tr_set = set(tr.tolist())
    te_set = set(te.tolist())
    inter = tr_set & te_set
    te_only = te_set - tr_set
    te_in_train_rows = sum(1 for t in te if t in tr_set)

    # line-level overlap (matters for full_window: a window is a sequence of lines,
    # and the relevant generalization question is whether individual lines are novel)
    tr_lines = _split_lines(tr)
    te_lines = _split_lines(te)
    tr_lines_set = set(tr_lines)
    te_lines_set = set(te_lines)
    line_inter = tr_lines_set & te_lines_set
    te_lines_in_train_count = sum(1 for ln in te_lines if ln in tr_lines_set)

    return {
        "fold": fold,
        "n_train_rows": int(len(tr)),
        "n_test_rows": int(len(te)),
        "n_train_distinct": len(tr_set),
        "n_test_distinct": len(te_set),
        "n_test_distinct_in_train": len(inter),
        "n_test_distinct_only": len(te_only),
        "frac_test_distinct_in_train": len(inter) / max(len(te_set), 1),
        "n_test_rows_in_train": te_in_train_rows,
        "frac_test_rows_in_train": te_in_train_rows / max(len(te), 1),
        # line-level
        "n_train_distinct_lines": len(tr_lines_set),
        "n_test_distinct_lines": len(te_lines_set),
        "n_test_distinct_lines_in_train": len(line_inter),
        "frac_test_distinct_lines_in_train": len(line_inter) / max(len(te_lines_set), 1),
        "n_test_line_tokens": len(te_lines),
        "n_test_line_tokens_in_train": te_lines_in_train_count,
        "frac_test_line_tokens_in_train": te_lines_in_train_count / max(len(te_lines), 1),
    }


def main():
    rows = []
    for mode in ["per_row", "full_window"]:
        print(f"\n=== mode={mode} ===")
        for f in range(1, 6):
            try:
                r = fold_stats(f, mode=mode)
                r["mode"] = mode
                rows.append(r)
                print(f"  fold {f}: "
                      f"train_distinct={r['n_train_distinct']:>5}, "
                      f"test_distinct={r['n_test_distinct']:>5}, "
                      f"test∩train_distinct={r['n_test_distinct_in_train']:>5} "
                      f"({100*r['frac_test_distinct_in_train']:.1f}%), "
                      f"test_rows_in_train_frac={100*r['frac_test_rows_in_train']:.1f}%")
            except FileNotFoundError as e:
                print(f"  fold {f}: SKIP ({e.filename})")

    # Aggregate
    out = {"per_fold": rows}
    for mode in ["per_row", "full_window"]:
        rr = [r for r in rows if r["mode"] == mode]
        if not rr:
            continue
        out[f"{mode}_mean_frac_test_distinct_in_train"] = float(
            np.mean([r["frac_test_distinct_in_train"] for r in rr])
        )
        out[f"{mode}_mean_frac_test_rows_in_train"] = float(
            np.mean([r["frac_test_rows_in_train"] for r in rr])
        )
        out[f"{mode}_mean_test_distinct"] = float(
            np.mean([r["n_test_distinct"] for r in rr])
        )
        out[f"{mode}_mean_train_distinct"] = float(
            np.mean([r["n_train_distinct"] for r in rr])
        )
        # line-level
        out[f"{mode}_mean_frac_test_distinct_lines_in_train"] = float(
            np.mean([r["frac_test_distinct_lines_in_train"] for r in rr])
        )
        out[f"{mode}_mean_frac_test_line_tokens_in_train"] = float(
            np.mean([r["frac_test_line_tokens_in_train"] for r in rr])
        )
        out[f"{mode}_mean_test_distinct_lines"] = float(
            np.mean([r["n_test_distinct_lines"] for r in rr])
        )
        out[f"{mode}_mean_train_distinct_lines"] = float(
            np.mean([r["n_train_distinct_lines"] for r in rr])
        )

    out_path = REPO / "outputs" / "decoder20260511" / "audit" / "train_test_gcode_overlap.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    print("\n=== AGGREGATE ===")
    for k, v in out.items():
        if k != "per_fold":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
