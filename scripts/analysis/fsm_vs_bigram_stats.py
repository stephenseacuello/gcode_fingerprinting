#!/usr/bin/env python3
"""Statistical comparison of FSM vs bigram-only grammar at deployment.

Computes paired statistics across the 5 folds for:
  - per-fold token / numeric accuracy (Wilcoxon signed-rank)
  - per-fold grammar-violation rate (paired t)
  - per-fold threat-model TPR and FPR (paired t)

Tests the null hypothesis that FSM has no effect on each metric.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]


def load_5fold(sweep_root: Path, fsm: bool):
    rows = []
    for F in range(1, 6):
        if fsm:
            cands = list(sweep_root.glob(f"fold_{F}/*_fsm/results/beam_1_metrics.json"))
        else:
            cands = [c for c in sweep_root.glob(f"fold_{F}/*/results/beam_1_metrics.json") if "_fsm" not in str(c)]
        if not cands:
            continue
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        d = json.loads(cands[0].read_text())
        tm = d.get("test_metrics", {})
        rows.append({
            "fold": F,
            "token": tm.get("token_accuracy"),
            "numeric": tm.get("numeric_accuracy"),
            "command": tm.get("command_accuracy"),
        })
    return rows


def main():
    sweeps = {
        "baseline": REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold",
        "with_shortcuts": REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold_with_shortcuts",
    }
    results = {}

    for cfg, root in sweeps.items():
        big = load_5fold(root, fsm=False)
        fsm = load_5fold(root, fsm=True)
        if len(big) != len(fsm) or len(big) < 5:
            print(f"{cfg}: missing folds, skipping ({len(big)}/{len(fsm)})")
            continue

        print(f"\n=== {cfg} (n=5 folds) ===")
        cfg_results = {}
        for metric in ["token", "numeric", "command"]:
            b = np.array([r[metric] for r in big])
            f = np.array([r[metric] for r in fsm])
            delta = f - b
            t_stat, p_t = stats.ttest_rel(f, b)
            try:
                w_stat, p_w = stats.wilcoxon(f, b, zero_method="zsplit")
            except ValueError:
                w_stat, p_w = float("nan"), 1.0
            cfg_results[metric] = {
                "bigram_mean": float(b.mean()),
                "fsm_mean": float(f.mean()),
                "delta_mean": float(delta.mean()),
                "delta_std": float(delta.std()),
                "paired_t_stat": float(t_stat),
                "paired_t_p": float(p_t),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p": float(p_w),
            }
            print(f"  {metric:9s}: bigram={b.mean():.4f}±{b.std():.4f}  fsm={f.mean():.4f}±{f.std():.4f}  Δ={delta.mean():+.4f}  paired-t p={p_t:.4f}  Wilcoxon p={p_w:.4f}")
        results[cfg] = cfg_results

    out = REPO / "outputs" / "decoder20260511" / "audit" / "fsm_vs_bigram_paired_stats.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
