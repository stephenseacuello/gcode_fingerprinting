#!/usr/bin/env python3
"""Token-position accuracy curve.

Visual complement to Table tab:token_position_compare. Shows accuracy as
a function of output position, separately for per_row (legacy) and
full_window (5-fold) under both teacher-forced and autoregressive
evaluation.
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

# Hand-curated from audit/token_position_5fold_fullwindow.json + per_row pilot
# (audit/extended_rigor_audits.json rec8_token_position_failure)
DATA = {
    "per-row (fold 1, AR)": [0.761, 0.243, 0.781, 0.626, 0.514, 0.892, 0.900, 0.940, 0.928, 0.905],
    "full-window (5-fold, AR mean)": [0.738, 0.621, 0.776, 0.700, 0.610, 0.757, 0.900, 0.940, 0.928, 0.905],
}
# Std for full_window
FW_STD = [0.088, 0.030, 0.060, 0.080, 0.070, 0.042, 0.026, 0.036, 0.048, 0.062]
POSITIONS = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(POSITIONS))

    # per_row (single fold pilot)
    ax.plot(x, DATA["per-row (fold 1, AR)"], "o-", color="#b33838", alpha=0.8,
            label="per-row (fold 1)", linewidth=2, markersize=8)

    # full_window mean ± std band
    mu = np.array(DATA["full-window (5-fold, AR mean)"])
    sd = np.array(FW_STD)
    ax.plot(x, mu, "s-", color="#3866b3", alpha=0.85,
            label="full-window (5-fold mean ± std)", linewidth=2, markersize=8)
    ax.fill_between(x, mu - sd, mu + sd, color="#3866b3", alpha=0.18)

    # Annotate the position-1 sharp drop
    ax.annotate("position 1: first PARAM letter\nper-row drops to 0.24",
                xy=(1, 0.243), xytext=(2.8, 0.42),
                arrowprops=dict(arrowstyle="->", color="#b33838", lw=1.2),
                fontsize=9, color="#b33838")

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in POSITIONS])
    ax.set_xlabel("Output token position (0 = first emitted token)")
    ax.set_ylabel("Per-position token accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-token-position accuracy: per-row vs full-window targets")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out = FIG_DIR / "token_position_curve.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
