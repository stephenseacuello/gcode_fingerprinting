#!/usr/bin/env python3
"""Regenerate the Paper B figures driven by the corrected source-entropy
apparatus (the source-vs-tokenized reconciliation, R3 D-MAJOR/D6).

Reads ONLY from the authoritative audit JSONs:
  - audit/source_vs_tokenized_entropy.json  (per-(axis,slot) SOURCE vs TOKENIZED)
  - audit/raw_gcode_source_entropy.json      (per-op-class/family per-axis per-slot,
                                              per-axis digit-value distributions)
  - audit/patched_numeric_analysis.json      (patched per-axis-slot recovery accuracy)
  - audit/per_head_per_opclass_entropy_5fold.json (54-cell head x op-class entropy)
  - audit/entropy_accuracy_correlation_fixed.json (54-cell entropy vs accuracy)

Slot grid (index -> decimal place), shared by all panels:
  0:10^1 tens  1:10^0 units  2:10^-1 tenths  3:10^-2 hundredths
  4:10^-3 thousandths  5:10^-4 ten-thousandths

Writes ONLY into this (toolpath_entropy/figures) directory.

Figures regenerated:
  - bits_recoverable_per_axis.pdf
  - quantization_heatmap.pdf
  - digit_value_histograms.pdf
  - per_slot_per_opclass_entropy_heatmap.pdf
  - per_axis_per_slot_per_opclass_entropy_heatmap.pdf
  - entropy_accuracy_all_heads.pdf
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
AUDIT = HERE.parents[2] / "audit"
SVT = json.loads((AUDIT / "source_vs_tokenized_entropy.json").read_text())
RAW = json.loads((AUDIT / "raw_gcode_source_entropy.json").read_text())
PAT = json.loads((AUDIT / "patched_numeric_analysis.json").read_text())
H54 = json.loads((AUDIT / "entropy_accuracy_correlation_fixed.json").read_text())

SLOT_TICK = ["$10^{1}$\n(tens)", "$10^{0}$\n(units)",
             "$10^{-1}$\n(tenths)", "$10^{-2}$\n(hund.)",
             "$10^{-3}$\n(thous.)", "$10^{-4}$\n(ten-th.)"]
SLOT_SHORT = ["$10^{1}$", "$10^{0}$", "$10^{-1}$",
              "$10^{-2}$", "$10^{-3}$", "$10^{-4}$"]
SLOTS = list(range(6))
CEIL = np.log2(10)


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(HERE / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


def _clip(x):
    return max(float(x), 0.0)


def bits_recoverable_per_axis():
    """Per-axis bit budget: source entropy summed across slots, split per slot,
    shaded by recovered vs unrecovered (~H_s * a_s) from patched accuracy."""
    pa = SVT["per_axis_slot"]
    acc = PAT["patched_per_axis_slot"]
    axes = ["X", "Y", "Z", "R", "F"]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    rec_tot, tot_tot = {}, {}
    for i, axis in enumerate(axes):
        H = [_clip(h) for h in pa[axis]["source"]]
        a = acc[axis]
        bottom_rec = 0.0
        bottom_tot = 0.0
        for s in range(6):
            recov = H[s] * a[s]
            unrec = H[s] * (1.0 - a[s])
            ax.bar(i, recov, bottom=bottom_rec, width=0.62,
                   color="#2c8d3a", edgecolor="white", linewidth=0.4,
                   label="recovered" if (i == 0 and s == 0) else None)
            bottom_rec += recov
            ax.bar(i, unrec, bottom=bottom_rec, width=0.62,
                   color="#b33838", edgecolor="white", linewidth=0.4,
                   label="unrecovered" if (i == 0 and s == 0) else None)
            bottom_rec += unrec
            bottom_tot += H[s]
        rec_tot[axis] = sum(H[s] * a[s] for s in range(6))
        tot_tot[axis] = sum(H)
        ax.annotate(f"{rec_tot[axis]:.1f}/{tot_tot[axis]:.1f}",
                    (i, tot_tot[axis]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9)
    ax.set_xticks(range(len(axes)))
    ax.set_xticklabels(axes, fontsize=11)
    ax.set_ylabel("Source entropy summed over 6 slots (bits)", fontsize=11)
    ax.set_xlabel("Axis", fontsize=11)
    ax.set_title("Per-axis bit budget: recovered vs. unrecovered source bits\n"
                 "(SOURCE entropy x patched recovery accuracy)", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    ax.grid(alpha=0.3, axis="y")
    save(fig, "bits_recoverable_per_axis")
    print("  bit budgets:", {k: (round(rec_tot[k], 1), round(tot_tot[k], 1))
                             for k in axes})


def quantization_heatmap():
    """Per-(axis, slot) SOURCE entropy heatmap, X/Y/Z/R/F x 6 slots."""
    pa = SVT["per_axis_slot"]
    axes = ["X", "Y", "Z", "R", "F"]
    M = np.array([[_clip(h) for h in pa[ax]["source"]] for ax in axes])
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=CEIL)
    ax.set_xticks(range(6))
    ax.set_xticklabels(SLOT_SHORT, fontsize=10)
    ax.set_yticks(range(len(axes)))
    ax.set_yticklabels(axes, fontsize=11)
    for i in range(len(axes)):
        for j in range(6):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.6 * CEIL else "black", fontsize=8.5)
    ax.set_xlabel("Digit slot (tens $\\to$ ten-thousandths)", fontsize=11)
    ax.set_title("Per-(axis, slot) SOURCE entropy (bits); ceiling "
                 "$\\log_2 10\\approx3.32$", fontsize=10.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Source entropy (bits)", fontsize=9)
    save(fig, "quantization_heatmap")


def digit_value_histograms():
    """Per-axis (X/Y/Z) per-slot empirical digit distributions, source-side."""
    axes = ["X", "Y", "Z"]
    show_slots = [2, 3, 4, 5]  # tenths..ten-thousandths
    fig, axarr = plt.subplots(len(axes), len(show_slots),
                              figsize=(11.0, 5.6), sharey=True)
    for r, axis in enumerate(axes):
        slots = RAW["per_axis"][axis]["slots"]
        for c, s in enumerate(show_slots):
            a = axarr[r, c]
            dist = slots[str(s)]["dist"]
            Hbits = slots[str(s)]["entropy_bits"]
            a.bar(range(10), dist, color="#4f7fb0", edgecolor="black",
                  linewidth=0.4)
            a.axhline(0.1, color="gray", linestyle=":", linewidth=1.0)
            a.set_ylim(0, 1.0)
            a.set_xticks(range(0, 10, 2))
            a.tick_params(labelsize=8)
            a.annotate(f"H={Hbits:.2f}", (0.97, 0.92), xycoords="axes fraction",
                       ha="right", va="top", fontsize=8.5,
                       bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                 ec="0.5", alpha=0.9))
            if r == 0:
                a.set_title(SLOT_SHORT[s], fontsize=10)
            if c == 0:
                a.set_ylabel(f"{axis}\nP(digit)", fontsize=10)
    fig.suptitle("Per-axis SOURCE digit-value distributions (4-decimal corpus): "
                 "tenths$\\to$thousandths climb to the uniform ceiling; the "
                 "ten-thousandths is\nsource-uniform on X/Y/Z but discarded by "
                 "the 0.001 tokenizer grid (tokenized $H{=}0$)", fontsize=10.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "digit_value_histograms")


def per_slot_per_opclass_heatmap():
    """Pooled X/Y/Z per-slot per-op-class SOURCE entropy. 6 slots x 9 classes.
    Slots 0,1 are 0 for X/Y/Z; pooled n-weighted mean over X/Y/Z for slots 2-5."""
    opc = RAW["per_op_class_slots"]
    classes = [c for c in opc.keys() if c != "vocab_corpus"]
    # pretty order
    order = ["face", "face075", "face150", "pocket", "pocket075", "pocket150",
             "adaptive", "adaptive075", "adaptive150"]
    classes = [c for c in order if c in opc]
    M = np.zeros((6, len(classes)))
    for j, c in enumerate(classes):
        for s in range(6):
            num = 0.0
            den = 0.0
            for ax in ["X", "Y", "Z"]:
                cell = opc[c].get(ax, {}).get(str(s))
                if cell and cell.get("n", 0) > 0:
                    n = cell["n"]
                    num += cell["entropy_bits"] * n
                    den += n
            M[s, j] = num / den if den else 0.0
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=CEIL)
    ax.set_yticks(range(6))
    ax.set_yticklabels(SLOT_SHORT, fontsize=10)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8.5, rotation=40, ha="right")
    for s in range(6):
        for j in range(len(classes)):
            v = M[s, j]
            ax.text(j, s, f"{v:.1f}", ha="center", va="center",
                    color="white" if v < 0.55 * CEIL else "black", fontsize=7.5)
    ax.set_xlabel("Operation class", fontsize=11)
    ax.set_ylabel("Digit slot", fontsize=11)
    ax.set_title("Pooled X/Y/Z per-slot per-op-class SOURCE entropy (bits); "
                 "op-class split sharpest at the thousandths $10^{-3}$",
                 fontsize=10.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("bits", fontsize=9)
    save(fig, "per_slot_per_opclass_entropy_heatmap")


def per_axis_per_slot_per_opclass_heatmap():
    """Per-axis (X,Y,Z) x slot x op-FAMILY SOURCE entropy, 3 stacked heatmaps."""
    fam = RAW["per_op_family_slots"]
    families = ["face", "pocket", "adaptive"]
    axes = ["X", "Y", "Z"]
    fig, axarr = plt.subplots(3, 1, figsize=(7.2, 8.4))
    for r, axis in enumerate(axes):
        M = np.zeros((len(families), 6))
        for i, fm in enumerate(families):
            for s in range(6):
                cell = fam[fm].get(axis, {}).get(str(s))
                M[i, s] = cell["entropy_bits"] if cell and cell.get("n", 0) else 0.0
        a = axarr[r]
        im = a.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=CEIL)
        a.set_yticks(range(len(families)))
        a.set_yticklabels(families, fontsize=10)
        a.set_xticks(range(6))
        a.set_xticklabels(SLOT_SHORT, fontsize=9)
        for i in range(len(families)):
            for s in range(6):
                v = M[i, s]
                a.text(s, i, f"{v:.2f}", ha="center", va="center",
                       color="white" if v < 0.6 * CEIL else "black", fontsize=8)
        a.set_ylabel(f"{axis} axis", fontsize=11)
        if r == 0:
            a.set_title("Per-axis per-slot per-op-family SOURCE entropy (bits)\n"
                        "facing X drops at the thousandths $10^{-3}$; Y stays "
                        "near ceiling; Z carries cutting-depth structure",
                        fontsize=9.5)
        if r == len(axes) - 1:
            a.set_xlabel("Digit slot", fontsize=11)
        fig.colorbar(im, ax=a, fraction=0.03, pad=0.02)
    fig.tight_layout()
    save(fig, "per_axis_per_slot_per_opclass_entropy_heatmap")


def entropy_accuracy_all_heads():
    """54-cell per-(head, op-class): measured accuracy vs source entropy."""
    cells = H54["cells"]
    heads = sorted({c["head"] for c in cells})
    cmap = {"token": "#7a4fa5", "type": "#4f7fb0", "command": "#2c8d3a",
            "param_type": "#bf8024", "sign": "#b33838", "digit": "#111111"}
    mk = {"token": "o", "type": "s", "command": "^", "param_type": "D",
          "sign": "v", "digit": "P"}
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    xs, ys = [], []
    for h in heads:
        hx = [c["entropy_bits"] for c in cells if c["head"] == h]
        hy = [c["accuracy"] for c in cells if c["head"] == h]
        ax.scatter(hx, hy, s=55, c=cmap.get(h, "gray"), marker=mk.get(h, "o"),
                   edgecolors="black", linewidths=0.5, alpha=0.85, label=h,
                   zorder=3)
        xs += hx
        ys += hy
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    pr, _ = stats.pearsonr(xs, ys)
    sr, _ = stats.spearmanr(xs, ys)
    slope, intercept, _, _, _ = stats.linregress(xs, ys)
    xl = np.linspace(xs.min() - 0.2, xs.max() + 0.2, 100)
    ax.plot(xl, slope * xl + intercept, "--", color="gray", linewidth=1.4,
            alpha=0.8, zorder=2, label=f"OLS (slope={slope:+.3f})")
    ax.text(0.97, 0.03,
            f"Pearson r = {pr:+.3f}\nSpearman $\\rho$ = {sr:+.3f}\nn = {len(xs)}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="0.4", alpha=0.92), zorder=5)
    ax.set_xlabel("Empirical source entropy (bits)", fontsize=11)
    ax.set_ylabel("Recovery accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92, ncol=2)
    ax.set_title("Source-entropy ordering across heads x op-classes "
                 "($n=54$, cross-head)", fontsize=10.5)
    save(fig, "entropy_accuracy_all_heads")
    print(f"  all-heads n={len(xs)} pearson={pr:.3f} spearman={sr:.3f}")


if __name__ == "__main__":
    bits_recoverable_per_axis()
    quantization_heatmap()
    digit_value_histograms()
    per_slot_per_opclass_heatmap()
    per_axis_per_slot_per_opclass_heatmap()
    entropy_accuracy_all_heads()
    print("done")
