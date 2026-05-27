#!/usr/bin/env python3
"""K-spectrum figure: how does structural recovery vary with numeric vocab size?

Reads audit/k_spectrum_compare.json. Two panels:
  Left  : per-token structural accuracy (TF + AR) vs K, log-x.
  Right : per-window structural exact-match (TF + AR) vs K, log-x.

Five K points (T3.1 control added):
  - K=24  (placeholder, SS=0/dw=0)
  - K=69  (1-digit,    SS=0/dw=0)
  - K=335 (2-digit,    SS=0/dw=0)
  - K=2418 headline   (4-digit,  SS=0.5/dw=1.0)   <- open marker, dashed
  - K=2418 T3.1 ctrl   (4-digit,  SS=0/dw=0)       <- filled marker, solid

The T3.1 matched-methodology control resolves the earlier K-spectrum
methodology confound: the highest-performing variant is K=2418/SS=0/dw=0
(NOT K=335). The K=335 maximum in the 4-point sweep was the SS/dw
schedule effect, not vocabulary cardinality.
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
    # Five rows: 4 original variants + T3.1 control at K=2418
    # We need to plot them separately because two share K=2418.
    variants = d["variants"]
    # Build (label, K, methodology) tuples in display order
    rows = []
    for v in variants:
        is_headline = "SS=0.5" in v["name"]
        is_t3_1 = "T3.1" in v["name"]
        rows.append({"name": v["name"], "K": v["K"], "v": v,
                     "is_headline": is_headline, "is_t3_1": is_t3_1})

    # Matched-methodology points: K in {24, 69, 335} + T3.1 (K=2418 matched)
    matched = [r for r in rows if not r["is_headline"]]
    matched_sorted = sorted(matched, key=lambda r: r["K"])
    # Headline: the only cross-methodology point
    headline = [r for r in rows if r["is_headline"]][0]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    TF_COLOR = "#2E86AB"; AR_COLOR = "#E63946"

    def get_metric(row, reg, metric):
        agg = row["v"][reg]["aggregate"][metric]
        return agg["mean"], agg["std"]

    def plot_panel(ax, metric, ylim, ylabel, title, legend_loc):
        K_matched = np.array([r["K"] for r in matched_sorted])
        for reg, marker, color, label in [("TF", "o", TF_COLOR, "Teacher-forced"),
                                          ("AR", "s", AR_COLOR, "Autoregressive")]:
            m_m = np.array([get_metric(r, reg, metric)[0] for r in matched_sorted])
            s_m = np.array([get_metric(r, reg, metric)[1] for r in matched_sorted])
            # Solid line + filled markers across matched-methodology family
            ax.errorbar(K_matched, m_m, yerr=s_m, marker=marker, markersize=9,
                        capsize=4, color=color, label=label, linewidth=1.6,
                        linestyle="-", markerfacecolor=color, markeredgecolor=color)
            # Headline K=2418 point: open marker, slight x-jitter so it's visible next to T3.1
            m_h, s_h = get_metric(headline, reg, metric)
            x_h = headline["K"] * 1.18  # jitter right
            ax.errorbar([x_h], [m_h], yerr=[s_h], marker=marker, markersize=11,
                        capsize=4, color=color, linewidth=0,
                        markerfacecolor="white", markeredgecolor=color,
                        markeredgewidth=1.8)
            # Connect last matched point to headline with dashed line
            ax.plot([matched_sorted[-1]["K"], x_h],
                    [m_m[-1], m_h], color=color, linestyle="--",
                    linewidth=1.2, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(r"Numeric vocabulary size $K$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc=legend_loc, frameon=False)
        ax.grid(True, alpha=0.3)
        # x-ticks at the 4 distinct K values + jittered headline annotation
        xt = sorted({r["K"] for r in matched_sorted})
        ax.set_xticks(xt)
        ax.set_xticklabels([str(k) for k in xt])
        ax.set_ylim(*ylim)

    plot_panel(axes[0], "struct_token_acc", (0.25, 1.02),
               "Structural token accuracy", "(a) Per-token structural recovery",
               "lower right")
    plot_panel(axes[1], "struct_seq_exact", (0.0, 0.40),
               "Structural sequence exact-match", "(b) Per-window structural recovery",
               "upper left")

    # Annotation for T3.1 and headline distinction
    axes[0].annotate("$K\\!=\\!2{,}418$ headline\n(SS=0.5/dw=1.0)\n(cross-methodology)",
                     xy=(2418*1.18, 0.32), xytext=(550, 0.48),
                     fontsize=7, ha="center", color="dimgray",
                     arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.7))
    axes[0].annotate("$K\\!=\\!2{,}418$ T3.1 control\n(SS=0/dw=0, matched)",
                     xy=(2418, 0.47), xytext=(820, 0.78),
                     fontsize=7, ha="center", color="black",
                     arrowprops=dict(arrowstyle="->", color="black", lw=0.7))

    fig.suptitle(
        r"Structural recovery vs numeric vocabulary size $K$ "
        r"(5 variants; T3.1 = methodology-matched $K\!=\!2{,}418$ control)",
        fontsize=10, y=1.03,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    plt.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
