#!/usr/bin/env python3
"""K-spectrum figure: how does structural recovery vary with numeric vocab size?

Reads audit/k_spectrum_compare.json. Two panels:
  Left  : per-token structural accuracy (TF + AR) vs K, log-x.
  Right : per-window structural exact-match (TF + AR) vs K, log-x.

Four K points (4 vocab variants): 24 (placeholder), 69 (1-digit),
335 (2-digit), 2418 (4-digit, current).

The current K=2418 endpoint uses the headline training config (SS=0.5,
dw=1.0); the smaller-K variants share the Design-B style (SS=0, dw=0). The
caption notes the methodology caveat.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "outputs/decoder20260511/audit/k_spectrum_compare.json"
OUT = REPO / "outputs/decoder20260511/decoder_paper_v2/figures/k_spectrum.pdf"


def main() -> None:
    d = json.load(open(SRC))
    K_list = [v["K"] for v in d["variants"]]
    order = np.argsort(K_list)
    K_sorted = np.array(K_list)[order]

    def series(reg: str, metric: str):
        means = np.array([v[reg]["aggregate"][metric]["mean"] for v in d["variants"]])
        stds = np.array([v[reg]["aggregate"][metric]["std"] for v in d["variants"]])
        return means[order], stds[order]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    TF_COLOR = "#2E86AB"; AR_COLOR = "#E63946"

    # ----- panel A: per-token structural ------------------------------------
    ax = axes[0]
    for reg, marker, color, label in [("TF", "o", TF_COLOR, "Teacher-forced"),
                                      ("AR", "s", AR_COLOR, "Autoregressive")]:
        m, s = series(reg, "struct_token_acc")
        ax.errorbar(K_sorted, m, yerr=s, marker=marker, markersize=9,
                    capsize=4, color=color, label=label, linewidth=1.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"Numeric vocabulary size $K$")
    ax.set_ylabel("Structural token accuracy")
    ax.set_title("(a) Per-token structural recovery")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_sorted)
    ax.set_xticklabels([str(k) for k in K_sorted])
    ax.set_ylim(0.25, 1.02)

    # ----- panel B: per-window structural seq exact -------------------------
    ax = axes[1]
    for reg, marker, color, label in [("TF", "o", TF_COLOR, "Teacher-forced"),
                                      ("AR", "s", AR_COLOR, "Autoregressive")]:
        m, s = series(reg, "struct_seq_exact")
        ax.errorbar(K_sorted, m, yerr=s, marker=marker, markersize=9,
                    capsize=4, color=color, label=label, linewidth=1.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"Numeric vocabulary size $K$")
    ax.set_ylabel("Structural sequence exact-match")
    ax.set_title("(b) Per-window structural recovery")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_sorted)
    ax.set_xticklabels([str(k) for k in K_sorted])
    ax.set_ylim(0, 0.35)

    fig.suptitle(
        r"Structural recovery vs numeric vocabulary size $K$ "
        r"($K\!=\!24$: <NUM> placeholder; $K\!=\!69$: 1-digit; "
        r"$K\!=\!335$: 2-digit; $K\!=\!2{,}418$: 4-digit / current model)",
        fontsize=10, y=1.03,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    plt.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
