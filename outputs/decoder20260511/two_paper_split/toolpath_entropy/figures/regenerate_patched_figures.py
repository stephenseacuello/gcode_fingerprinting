#!/usr/bin/env python3
"""Regenerate the Paper B figures whose story changed under the float-epsilon
patch, reading directly from audit/patched_numeric_analysis.json.

Self-contained; writes ONLY into this (toolpath_entropy/figures) directory.

Figures regenerated:
  - per_digit_position_curve.pdf   (pooled bathtub, patched per-slot accuracy)
  - per_axis_bathtub_overlay.pdf   (per-axis X/Y/Z/R/F patched bathtub)
  - entropy_vs_accuracy_overlay.pdf(pooled per-slot accuracy vs source entropy)
  - entropy_accuracy_scatter.pdf   (n=30 per-(axis,slot) scatter, r=-0.83)
  - slot5_by_op_class.pdf          (slot-5 op-class accuracy: ~0-entropy / recovered)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
# HERE = .../decoder20260511/two_paper_split/toolpath_entropy/figures
# audit  = .../decoder20260511/audit
AUDIT = HERE.parents[2] / "audit" / "patched_numeric_analysis.json"
D = json.loads(AUDIT.read_text())

SLOT_TICK = ["$10^{1}$\n(tens)", "$10^{0}$\n(units)",
             "$10^{-1}$\n(tenths)", "$10^{-2}$\n(hund.)",
             "$10^{-3}$\n(thous.)", "$10^{-4}$\n(ten-th.)"]
SLOTS = list(range(6))


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(HERE / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


def bathtub_pooled():
    bt = D["pooled_per_slot_bathtub"]
    means = bt["patched_mean"]
    stds = bt["patched_std"]
    H = D["corrected_pooled_slot_entropy"]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.errorbar(SLOTS, means, yerr=stds, fmt="o-", color="#b33838",
                linewidth=2.0, markersize=8, capsize=4,
                label="Patched recovery accuracy")
    for s, m in zip(SLOTS, means):
        ax.annotate(f"{m:.2f}", (s, m), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Digit slot (model's fixed 6-slot grid)", fontsize=11)
    ax.set_ylabel("Per-slot accuracy (5-fold mean $\\pm$ std)", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(SLOTS)
    ax.set_xticklabels(SLOT_TICK, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Pooled per-digit-slot recovery (patched encoder): the "
                 "true precision floor is the thousandths ($10^{-3}$)",
                 fontsize=10.5)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
    save(fig, "per_digit_position_curve")


def bathtub_per_axis():
    pa = D["patched_per_axis_slot"]
    AXIS_ORDER = ["X", "Y", "Z", "R", "F"]
    AXIS_COLOR = {"X": "#b33838", "Y": "#bf8024", "Z": "#2c8d3a",
                  "R": "#4f7fb0", "F": "#7a4fa5"}
    AXIS_MARKER = {"X": "o", "Y": "s", "Z": "^", "R": "D", "F": "v"}
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for axis in AXIS_ORDER:
        means = pa[axis]
        alpha = 0.55 if axis == "F" else 1.0
        ls = "--" if axis == "F" else "-"
        ax.plot(SLOTS, means, AXIS_MARKER[axis] + ls,
                color=AXIS_COLOR[axis], linewidth=2.0, markersize=8,
                alpha=alpha, label=axis)
    ax.set_xlabel("Digit slot (model's fixed 6-slot grid)", fontsize=11)
    ax.set_ylabel("Per-slot accuracy (5-fold mean)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(SLOTS)
    ax.set_xticklabels(SLOT_TICK, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Per-axis per-slot accuracy (patched): Z recovers through the "
                 "hundredths; R is uniformly high; X/Y trough at the thousandths",
                 fontsize=10)
    ax.legend(loc="lower center", fontsize=10, framealpha=0.92, ncol=5,
              bbox_to_anchor=(0.5, -0.30))
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, "per_axis_bathtub_overlay")


def overlay():
    bt = D["pooled_per_slot_bathtub"]["patched_mean"]
    H = D["corrected_pooled_slot_entropy"]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
    ax1.plot(SLOTS, bt, "o-", color="#b33838", linewidth=2.2, markersize=8,
             label="Recovery accuracy (patched)")
    ax1.set_xlabel("Digit slot", fontsize=11)
    ax1.set_ylabel("Recovery accuracy", color="#b33838", fontsize=11)
    ax1.set_ylim(0, 1.08)
    ax1.tick_params(axis="y", labelcolor="#b33838")
    ax1.set_xticks(SLOTS)
    ax1.set_xticklabels(SLOT_TICK, fontsize=9)
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(SLOTS, H, "s--", color="#2452a8", linewidth=2.2, markersize=7,
             label="Tokenized entropy (bits)")
    ax2.axhline(np.log2(10), color="gray", linestyle=":", linewidth=1.3,
                label="$\\log_2 10\\approx3.32$ ceiling")
    ax2.set_ylabel("Tokenized entropy (bits)", color="#2452a8", fontsize=11)
    ax2.set_ylim(0, 3.7)
    ax2.tick_params(axis="y", labelcolor="#2452a8")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="center left", fontsize=8.5,
               framealpha=0.92)
    ax1.set_title("Recovery accuracy is the inverse of tokenized entropy "
                  "(pooled, $n=6$, illustrative)", fontsize=10)
    save(fig, "entropy_vs_accuracy_overlay")


def scatter_n30():
    pa = D["patched_per_axis_slot"]
    Hpa = D["corrected_per_axis_slot_entropy"]
    AXES = ["X", "Y", "Z", "F", "R"]
    AXIS_COLOR = {"X": "#b33838", "Y": "#bf8024", "Z": "#2c8d3a",
                  "R": "#4f7fb0", "F": "#7a4fa5"}
    AXIS_MARKER = {"X": "o", "Y": "s", "Z": "^", "R": "D", "F": "v"}
    xs, ys = [], []
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for axis in AXES:
        H = [max(h, 0.0) for h in Hpa[axis]]
        acc = pa[axis]
        ax.scatter(H, acc, s=70, c=AXIS_COLOR[axis], marker=AXIS_MARKER[axis],
                   edgecolors="black", linewidths=0.6, alpha=0.85,
                   label=axis, zorder=3)
        xs += H
        ys += acc
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    pr, pp = stats.pearsonr(xs, ys)
    sr, sp = stats.spearmanr(xs, ys)
    slope, intercept, _, _, _ = stats.linregress(xs, ys)
    xline = np.linspace(xs.min() - 0.2, xs.max() + 0.2, 100)
    ax.plot(xline, slope * xline + intercept, "--", color="gray",
            linewidth=1.5, alpha=0.8, zorder=2,
            label=f"OLS (slope={slope:+.3f})")
    ax.text(0.97, 0.97,
            f"Pearson r = {pr:+.3f}\nSpearman $\\rho$ = {sr:+.3f}\nn = {len(xs)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="0.4", alpha=0.92), zorder=5)
    ax.set_xlabel("Empirical tokenized entropy (bits)", fontsize=11)
    ax.set_ylabel("Recovery accuracy (TF, 5-fold mean)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=-0.1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92, ncol=2)
    ax.set_title("Per-(axis, slot) recovery vs. tokenized entropy ($n=30$, patched)",
                 fontsize=10.5)
    save(fig, "entropy_accuracy_scatter")
    print(f"  scatter n=30 pearson={pr:.3f} spearman={sr:.3f}")


def slot5_opclass():
    fam = D["slot5_by_opclass_family_patched"]
    order = ["face", "pocket", "adaptive"]
    vals = [fam[k] for k in order]
    colors = ["#2c8d3a", "#bf8024", "#b33838"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(order, vals, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Slot-5 ($10^{-4}$) recovery accuracy", fontsize=11)
    ax.set_xlabel("Toolpath family", fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Slot-5 ($10^{-4}$) recovered near-perfectly for every op-class:\n"
                 "the 0.001 tokenizer grid discards the near-uniform source digit",
                 fontsize=9.5)
    save(fig, "slot5_by_op_class")


if __name__ == "__main__":
    bathtub_pooled()
    bathtub_per_axis()
    overlay()
    scatter_n30()
    slot5_opclass()
    print("done")
