#!/usr/bin/env python3
"""Per-field positive-support coverage over the V8 preprocessed NPZ corpus.

Reproduces the numbers in
``outputs/decoder20260511/decoder_paper_v2/tables/data_coverage.tex``
from first principles so the table has an executable provenance (previously
the table carried a "TODO: write this script" comment).

"Positive support" for an axis letter L is the fraction of samples whose
aligned G-code text contains an L address (the letter immediately followed by
an optional sign and a digit/decimal). For ``per_row`` each sample is one
G-code line; for ``full_window`` each sample is the multi-line G-code that
fired within the window.

Usage:
    python3 scripts/analysis/check_field_coverage.py \
        --root outputs/decoder20260511/preprocessed_f98 \
        --fold 1 \
        --output outputs/decoder20260511/audit/field_coverage.json

The detection regex is validated against the published per_row/test column
(X 88.5%, Y 86.6%, Z 31.4%, F 0.6%, S 0.0%, R 17.2%, I 0.0%, J 0.0%).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

FIELDS = ["X", "Y", "Z", "F", "S", "R", "I", "J"]
MODES = ["per_row", "full_window"]
SPLITS = ["train", "test"]


def _addr_re(letter: str) -> re.Pattern:
    # Letter not preceded by another letter (so it is an address, not part of
    # a word), followed by optional whitespace, optional sign, then a digit or
    # decimal point. Matches raw G-code text such as "X1.6269" or "X -1.2".
    return re.compile(r"(?<![A-Za-z])" + re.escape(letter) + r"\s*-?[0-9.]")


_PATTERNS = {f: _addr_re(f) for f in FIELDS}


def field_support(gcode_texts: np.ndarray) -> dict[str, float]:
    """Fraction of samples whose G-code text contains each axis address."""
    n = len(gcode_texts)
    if n == 0:
        return {f: 0.0 for f in FIELDS}
    counts = {f: 0 for f in FIELDS}
    for s in gcode_texts:
        s = str(s)
        for f in FIELDS:
            if _PATTERNS[f].search(s):
                counts[f] += 1
    return {f: counts[f] / n for f in FIELDS}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/decoder20260511/preprocessed_f98"),
        help="Directory containing <mode>/fold_<k>/<split>_sequences.npz",
    )
    p.add_argument("--fold", type=int, default=1)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/decoder20260511/audit/field_coverage.json"),
    )
    p.add_argument(
        "--emit-latex-rows",
        action="store_true",
        help="Also print the LaTeX body rows for data_coverage.tex.",
    )
    args = p.parse_args()

    result: dict[str, dict[str, dict[str, float]]] = {}
    for mode in MODES:
        result[mode] = {}
        for split in SPLITS:
            npz = args.root / mode / f"fold_{args.fold}" / f"{split}_sequences.npz"
            if not npz.exists():
                print(f"  skip {npz}: not found")
                result[mode][split] = {f: None for f in FIELDS}
                continue
            d = np.load(npz, allow_pickle=True)
            result[mode][split] = field_support(d["gcode_texts"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"fold": args.fold, "root": str(args.root), "support": result}, indent=2
        )
    )

    # Console table (percentages, matching data_coverage.tex layout).
    hdr = f"{'Field':<6}{'pr_train':>10}{'pr_test':>10}{'fw_train':>10}{'fw_test':>10}"
    print(f"\nPer-field positive support (fold {args.fold})\n{hdr}\n{'-'*46}")
    for f in FIELDS:
        def pct(mode, split):
            v = result[mode][split][f]
            return "--" if v is None else f"{100*v:.1f}%"
        print(
            f"{f:<6}{pct('per_row','train'):>10}{pct('per_row','test'):>10}"
            f"{pct('full_window','train'):>10}{pct('full_window','test'):>10}"
        )

    if args.emit_latex_rows:
        print("\n% data_coverage.tex body rows:")
        for f in FIELDS:
            def cell(mode, split):
                v = result[mode][split][f]
                if v is None:
                    return "---"
                s = f"{100*v:.1f}\\%"
                return f"\\mathbf{{{s}}}" if v < 0.01 else s
            print(
                f"{f}  & ${cell('per_row','train')}$ & ${cell('per_row','test')}$ "
                f"& ${cell('full_window','train')}$ & ${cell('full_window','test')}$ \\\\"
            )

    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
