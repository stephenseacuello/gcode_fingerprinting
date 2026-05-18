#!/usr/bin/env python3
"""Quantify AR output diversity (mode collapse) across the 5 baseline folds.

For each fold's beam_1 (autoregressive greedy) prediction set:
  - Total number of test samples (N)
  - Number of DISTINCT predicted token strings (U)
  - Frequency of the modal predicted string
  - Mean / median predicted-string length

Outputs audit/ar_output_diversity.json + prints a small table.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def fold_path(fold: int) -> Path:
    base = REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold" / f"fold_{fold}"
    cands = list(base.glob("*/results/beam_1_all_predictions.json"))
    # Exclude _fsm-suffixed run since we want the plain baseline AR
    plain = [c for c in cands if "_fsm" not in str(c)]
    if plain:
        return plain[0]
    if cands:
        return cands[0]
    raise FileNotFoundError(f"no beam_1_all_predictions.json in fold_{fold}")


def main():
    rows = []
    for f in range(1, 6):
        try:
            p = fold_path(f)
            preds = json.loads(p.read_text())
        except FileNotFoundError as e:
            print(f"fold {f}: {e}")
            continue

        N = len(preds)
        pred_strings = [item.get("pred", "") for item in preds]
        true_strings = [item.get("true", "") for item in preds]
        c = Counter(pred_strings)
        ct = Counter(true_strings)
        modal_pred, modal_count = c.most_common(1)[0]
        true_modal, true_modal_count = ct.most_common(1)[0]

        lengths = [len(s) for s in pred_strings]
        true_lengths = [len(s) for s in true_strings]

        out = {
            "fold": f,
            "n_test": N,
            "n_distinct_pred": len(c),
            "diversity_ratio_pred": len(c) / max(N, 1),
            "modal_pred_count": modal_count,
            "modal_pred_frac": modal_count / max(N, 1),
            "modal_pred_first120chars": modal_pred[:120],
            "n_distinct_true": len(ct),
            "diversity_ratio_true": len(ct) / max(N, 1),
            "modal_true_count": true_modal_count,
            "modal_true_frac": true_modal_count / max(N, 1),
            "mean_pred_len": sum(lengths) / max(len(lengths), 1),
            "mean_true_len": sum(true_lengths) / max(len(true_lengths), 1),
        }
        rows.append(out)
        print(f"fold {f}: N={N}, distinct_pred={len(c)} (ratio {len(c)/N:.3f}), "
              f"modal_pred_frac={out['modal_pred_frac']:.3f} | "
              f"distinct_true={len(ct)} (ratio {len(ct)/N:.3f}), "
              f"modal_true_frac={out['modal_true_frac']:.3f}")

    # Aggregate
    if rows:
        mean_div_pred = sum(r["diversity_ratio_pred"] for r in rows) / len(rows)
        mean_div_true = sum(r["diversity_ratio_true"] for r in rows) / len(rows)
        mean_modal_pred = sum(r["modal_pred_frac"] for r in rows) / len(rows)
        mean_modal_true = sum(r["modal_true_frac"] for r in rows) / len(rows)
        print()
        print(f"  AGGREGATE:")
        print(f"    mean diversity_ratio  pred={mean_div_pred:.3f}  true={mean_div_true:.3f}")
        print(f"    mean modal_frac       pred={mean_modal_pred:.3f}  true={mean_modal_true:.3f}")
        aggregate = {
            "mean_diversity_ratio_pred": mean_div_pred,
            "mean_diversity_ratio_true": mean_div_true,
            "mean_modal_pred_frac": mean_modal_pred,
            "mean_modal_true_frac": mean_modal_true,
            "n_folds": len(rows),
        }
    else:
        aggregate = {}

    out_path = REPO / "outputs" / "decoder20260511" / "audit" / "ar_output_diversity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"per_fold": rows, "aggregate": aggregate}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
