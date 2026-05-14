#!/usr/bin/env python3
"""Beam-width comparison plot.

Compares accuracy across beam widths {0=teacher-forced, 1=greedy AR,
3=beam-3, 5=beam-5} for token / numeric metrics. Uses the legacy
per_row 5-fold sweep that has all four beam widths cached, plus the
full_window 5-fold (TF + greedy AR available from prior runs).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
FIG_DIR = REPO / "outputs" / "decoder20260511" / "decoder_paper_v2" / "figures"


def collect_beam_metrics(sweep_root: Path, beam_widths=(0, 1, 3, 5)):
    """Returns {bw: {'token': [per-fold], 'numeric': [per-fold]}}."""
    out = {bw: {"token": [], "numeric": []} for bw in beam_widths}
    for F in range(1, 6):
        for bw in beam_widths:
            direct = sweep_root / f"fold_{F}" / "results" / f"beam_{bw}_metrics.json"
            nested = list(sweep_root.glob(f"fold_{F}/*/results/beam_{bw}_metrics.json"))
            nested = [n for n in nested if "_fsm" not in str(n)]
            cand = None
            if direct.exists():
                cand = direct
            elif nested:
                nested.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                cand = nested[0]
            if cand is None:
                continue
            d = json.loads(cand.read_text())
            tm = d.get("test_metrics", d)
            tok = tm.get("token_accuracy")
            num = tm.get("numeric_accuracy")
            if tok is not None:
                out[bw]["token"].append(tok)
            if num is not None:
                out[bw]["numeric"].append(num)
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    per_row_legacy = collect_beam_metrics(
        REPO / "outputs" / "decoder20260511" / "checkpoints" / "per_row_5fold_50ep_legacy",
        beam_widths=(0, 1, 3, 5),
    )
    full_window = collect_beam_metrics(
        REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold",
        beam_widths=(0, 1),
    )

    bws = [0, 1, 3, 5]
    bw_labels = ["TF\n(bw=0)", "AR-greedy\n(bw=1)", "Beam-3\n(bw=3)", "Beam-5\n(bw=5)"]
    x = np.arange(len(bws))

    fig, (ax_tok, ax_num) = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, metric, ylabel in [(ax_tok, "token", "Token accuracy"),
                                (ax_num, "numeric", "Numeric accuracy")]:
        # per_row legacy
        means_pr = []
        stds_pr = []
        for bw in bws:
            vals = per_row_legacy.get(bw, {}).get(metric, [])
            if vals:
                means_pr.append(np.mean(vals)); stds_pr.append(np.std(vals))
            else:
                means_pr.append(np.nan); stds_pr.append(0.0)
        # full_window
        means_fw = []
        stds_fw = []
        for bw in bws:
            vals = full_window.get(bw, {}).get(metric, [])
            if vals:
                means_fw.append(np.mean(vals)); stds_fw.append(np.std(vals))
            else:
                means_fw.append(np.nan); stds_fw.append(0.0)

        ax.errorbar(x - 0.08, means_pr, yerr=stds_pr, fmt="o-", color="#b33838",
                    label="per-row (legacy 5-fold)", capsize=4, linewidth=2, markersize=8)
        ax.errorbar(x + 0.08, means_fw, yerr=stds_fw, fmt="s-", color="#3866b3",
                    label="full-window (5-fold)", capsize=4, linewidth=2, markersize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(bw_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f"5-fold {ylabel}")
        ax.set_ylim(0, max(1.0, max([m for m in means_pr + means_fw if not np.isnan(m)]) * 1.1))
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle("Effect of decoding strategy on token and numeric accuracy", fontsize=11)
    plt.tight_layout()
    out = FIG_DIR / "beam_width_comparison.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
