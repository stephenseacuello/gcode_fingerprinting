#!/usr/bin/env python3
"""Phase-C analysis: why is numeric accuracy stuck at ~0.60?

Loads the saved predictions.npz from a 5-fold sweep and breaks numeric
performance down along three axes:

  1. Per-digit-position (digits 0..5 of NUM_X_<6-digit-bin>)
  2. Per-axis (NUM_X vs NUM_Y vs NUM_Z separately)
  3. Distribution of digit-error magnitudes

Output: outputs/decoder20260511/audit/numeric_accuracy_diagnosis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

DIGIT_PAD = 10


def diagnose(predictions_npz: Path, vocab_path: Path) -> dict:
    d = np.load(predictions_npz, allow_pickle=True)
    if "digit_p" not in d.files or "digit_t" not in d.files:
        return {"error": "no digit predictions in this npz"}

    digit_p = d["digit_p"]   # [N, L, 6]
    digit_t = d["digit_t"]   # [N, L, 6]
    type_t = d.get("type_t", None)
    pt_t = d.get("pt_t", None)

    n_pos = digit_p.shape[-1]
    out: dict = {"n_samples": int(digit_p.shape[0]), "n_positions": n_pos}

    # Per-digit-position accuracy (excluding PAD)
    per_pos = []
    for pos in range(n_pos):
        dp = digit_p[..., pos].flatten()
        dt = digit_t[..., pos].flatten()
        mask = dt != DIGIT_PAD
        if mask.sum() == 0:
            per_pos.append(None)
            continue
        acc = float((dp[mask] == dt[mask]).mean())
        # Distance distribution
        errs = np.abs(dp[mask].astype(int) - dt[mask].astype(int))
        per_pos.append({
            "n_observed": int(mask.sum()),
            "accuracy": acc,
            "mean_abs_digit_error": float(errs.mean()),
            "p50_abs_digit_error": float(np.percentile(errs, 50)),
            "p95_abs_digit_error": float(np.percentile(errs, 95)),
        })
    out["per_position"] = per_pos

    # Per-axis (using param_type targets to filter by NUM_X / NUM_Y / NUM_Z)
    # NOTE: pt_t holds the PARAM type at TARGET positions; but NUM tokens
    # inherit the axis of their preceding PARAM token. So we approximate
    # axis attribution by treating consecutive positions: a numeric position
    # at index i likely belongs to the same param as the most recent PARAM
    # token at index <= i.
    if pt_t is not None and type_t is not None:
        TYPE_NUMERIC = 3
        PARAM_AXIS_IDS = {0: "X", 1: "Y", 2: "Z", 3: "I", 4: "J", 5: "K",
                          6: "R", 7: "F", 8: "S", 9: "P"}
        per_axis = defaultdict(lambda: {"n_observed": 0, "correct_digits": 0, "total_digits": 0,
                                        "full_value_correct": 0, "full_value_total": 0})
        N, L = type_t.shape
        for i in range(N):
            current_axis = None
            for j in range(L):
                pt = int(pt_t[i, j])
                ty = int(type_t[i, j])
                if pt in PARAM_AXIS_IDS:
                    current_axis = PARAM_AXIS_IDS[pt]
                if ty == TYPE_NUMERIC and current_axis is not None:
                    # Score this numeric position
                    n_correct = 0
                    n_total = 0
                    for pos in range(n_pos):
                        if int(digit_t[i, j, pos]) != DIGIT_PAD:
                            n_total += 1
                            if int(digit_p[i, j, pos]) == int(digit_t[i, j, pos]):
                                n_correct += 1
                    per_axis[current_axis]["correct_digits"] += n_correct
                    per_axis[current_axis]["total_digits"] += n_total
                    per_axis[current_axis]["full_value_total"] += 1
                    if n_correct == n_total and n_total > 0:
                        per_axis[current_axis]["full_value_correct"] += 1
        out["per_axis"] = {
            ax: {
                "digit_accuracy": (v["correct_digits"] / max(v["total_digits"], 1)),
                "full_value_correct_pct": (v["full_value_correct"] / max(v["full_value_total"], 1)),
                "n_value_positions": v["full_value_total"],
            }
            for ax, v in per_axis.items()
        }

    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", type=Path, default=REPO / "outputs" / "decoder20260511" / "checkpoints" / "per_row_5fold")
    p.add_argument("--vocab", type=Path, default=REPO / "data" / "gcode_vocab_v8.json")
    p.add_argument("--output", type=Path, default=REPO / "outputs" / "decoder20260511" / "audit" / "numeric_accuracy_diagnosis.json")
    args = p.parse_args()

    per_fold = []
    for F in [1, 2, 3, 4, 5]:
        npz = args.sweep_root / f"fold_{F}" / "results" / "predictions.npz"
        if not npz.exists():
            continue
        diag = diagnose(npz, args.vocab)
        diag["fold"] = F
        per_fold.append(diag)
        print(f"  fold {F}: per-position accuracies = {[round(p['accuracy'], 3) if p else None for p in diag.get('per_position', [])]}")
        if "per_axis" in diag:
            for ax, v in diag["per_axis"].items():
                print(f"    axis {ax}: digit_acc={v['digit_accuracy']:.3f} full_value_correct={v['full_value_correct_pct']:.3f} (n={v['n_value_positions']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"per_fold": per_fold}, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
