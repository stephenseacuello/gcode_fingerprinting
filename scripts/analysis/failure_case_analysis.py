#!/usr/bin/env python3
"""Phase-C: failure case analysis.

For each fold, find:
  - 20 worst-performing test samples (longest edit distance to ground truth)
  - The most-confused (predicted, true) token pairs
  - The G-code lines that never match

Output: outputs/decoder20260511/audit/failure_cases_v8.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _levenshtein(a, b):
    """Plain Levenshtein on lists of ints."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        current = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous[j + 1] + 1
            deletions = current[j] + 1
            substitutions = previous[j] + (c1 != c2)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]


def analyze_fold(pred_npz: Path, samples_json: Path) -> dict[str, Any]:
    d = np.load(pred_npz, allow_pickle=True)
    pred_tokens = d["pred_tokens"]
    target_tokens = d["target_tokens"]
    n = pred_tokens.shape[0]
    PAD = 0

    # Per-sample edit distance
    dists = []
    for i in range(n):
        t = [x for x in target_tokens[i].tolist() if x != PAD]
        p = pred_tokens[i, :len(t)].tolist()
        dists.append((i, _levenshtein(t, p), t, p))

    # Worst 20 (highest edit distance)
    worst = sorted(dists, key=lambda x: -x[1])[:20]

    # Token confusion (most common (pred, true) MISmatches)
    confusions: Counter = Counter()
    for i in range(n):
        t = target_tokens[i].tolist()
        p = pred_tokens[i].tolist()
        for tt, pp in zip(t, p):
            if tt == PAD:
                break
            if tt != pp:
                confusions[(int(pp), int(tt))] += 1

    samples = []
    if samples_json.exists():
        samples = json.loads(samples_json.read_text())

    worst_payload = []
    for idx, dist, t_ids, p_ids in worst:
        entry = {"sample_idx": int(idx), "edit_distance": int(dist),
                 "true_tokens": t_ids, "pred_tokens": p_ids}
        if idx < len(samples):
            entry["true_text"] = samples[idx].get("true", "")
            entry["pred_text"] = samples[idx].get("pred", "")
        worst_payload.append(entry)

    return {
        "n_samples": n,
        "n_exact_match": int(sum(1 for _, d, _, _ in dists if d == 0)),
        "mean_edit_distance": float(np.mean([d for _, d, _, _ in dists])),
        "median_edit_distance": float(np.median([d for _, d, _, _ in dists])),
        "worst_samples": worst_payload,
        "top_token_confusions": [
            {"predicted_id": int(p), "true_id": int(t), "count": int(c)}
            for (p, t), c in confusions.most_common(30)
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", type=Path, default=REPO / "outputs" / "decoder20260511" / "checkpoints" / "per_row_5fold")
    p.add_argument("--output", type=Path, default=REPO / "outputs" / "decoder20260511" / "audit" / "failure_cases_v8.json")
    args = p.parse_args()

    per_fold = []
    for F in [1, 2, 3, 4, 5]:
        # Handle wandb-subdir layout: prefer direct, fall back to <run_id>/
        direct_npz = args.sweep_root / f"fold_{F}" / "results" / "predictions.npz"
        if direct_npz.exists():
            pred_npz = direct_npz
            samples_json = args.sweep_root / f"fold_{F}" / "results" / "beam_0_all_predictions.json"
        else:
            cands = list((args.sweep_root / f"fold_{F}").glob("*/results/predictions.npz"))
            if not cands:
                continue
            cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            pred_npz = cands[0]
            samples_json = pred_npz.parent / "beam_0_all_predictions.json"
        if not pred_npz.exists():
            continue
        rep = analyze_fold(pred_npz, samples_json)
        rep["fold"] = F
        per_fold.append(rep)
        print(f"  fold {F}: n={rep['n_samples']}, exact match {rep['n_exact_match']} "
              f"({100*rep['n_exact_match']/rep['n_samples']:.1f}%), median edit {rep['median_edit_distance']:.1f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"per_fold": per_fold}, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
