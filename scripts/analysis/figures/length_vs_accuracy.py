#!/usr/bin/env python3
"""Length-vs-accuracy scatter plot.

For each test sample under autoregressive decoding, plot the TRUE sequence
length (in tokens) against the per-sample token accuracy. Tests whether
autoregressive mode collapse correlates with target sequence length.
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


def collect(sweep_root: Path):
    """Return (lengths, accuracies, fold_ids) for all test samples in a sweep."""
    lens, accs, folds = [], [], []
    for F in range(1, 6):
        cands = list(sweep_root.glob(f"fold_{F}/*/results/beam_1_all_predictions.json"))
        cands = [c for c in cands if "_fsm" not in str(c)]
        if not cands:
            continue
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        samples = json.loads(cands[0].read_text())
        for s in samples:
            t = s.get("true", "").split()
            p = s.get("pred", "").split()
            if not t:
                continue
            n = min(len(t), len(p))
            if n == 0:
                continue
            correct = sum(1 for i in range(n) if t[i] == p[i])
            acc = correct / len(t)
            lens.append(len(t))
            accs.append(acc)
            folds.append(F)
    return np.array(lens), np.array(accs), np.array(folds)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    base_root = REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold"
    sc_root = REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold_with_shortcuts"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, root, title in [(ax1, base_root, "Baseline (no shortcuts)"),
                            (ax2, sc_root, "+ positional metadata")]:
        L, A, F = collect(root)
        if len(L) == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            continue
        scatter = ax.scatter(L, A, c=F, cmap="viridis", alpha=0.5, s=14)
        # LOWESS-style smoothing via bins
        bins = np.linspace(L.min(), L.max(), 25)
        bin_means = []
        bin_centers = []
        for i in range(len(bins) - 1):
            mask = (L >= bins[i]) & (L < bins[i + 1])
            if mask.sum() > 0:
                bin_means.append(A[mask].mean())
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
        ax.plot(bin_centers, bin_means, "r-", linewidth=2, label="binned mean")

        ax.set_xlabel("TRUE sequence length (tokens)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        # Add overall correlation
        if len(L) > 1:
            r = np.corrcoef(L, A)[0, 1]
            ax.text(0.04, 0.96, f"Pearson r = {r:.3f}\nn = {len(L)} samples",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax1.set_ylabel("Per-sample token accuracy (AR, beam=1)")
    ax2.tick_params(axis="y", labelleft=True)

    fig.suptitle("Autoregressive token accuracy vs.\ TRUE-sequence length\n(test samples, 5-fold pooled, colour = fold)",
                 fontsize=11)
    plt.tight_layout()
    out = FIG_DIR / "length_vs_accuracy.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
