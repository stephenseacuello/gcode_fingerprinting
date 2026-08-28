#!/usr/bin/env python3
"""
Generate the unified entropy vs. accuracy scatter across all (head, op-class) cells.

Inputs:
  - audit/per_head_per_opclass_entropy.json: per-(head, op_class) empirical
    Shannon entropy (bits) over fold-1 training tokens (6 heads x 9 op-classes).
  - audit/per_op_class_v8.json: per-op-class per-head TF accuracies.
    If unavailable / errored, fall back to the headline 5-fold TF accuracy per
    head and broadcast across all 9 op-classes (each head plotted as a
    horizontal band at constant accuracy). This is less informative but still
    yields a publishable summary of the entropy ceiling.

Outputs:
  decoder_paper_v2/figures/entropy_accuracy_all_heads.{pdf,png}
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path("/home/seacuello/Documents/gcode_fingerprinting")
ENTROPY_JSON = ROOT / "outputs/decoder20260511/audit/per_head_per_opclass_entropy.json"
# Prefer the FIXED live per-(head, op-class) accuracies (TASK B2). The original
# per_op_class_v8.json errored on every fold (per-row vs full-window alignment
# bug), causing this script to fall back to hardcoded headline-TF constants
# broadcast as flat bands. The fixed file carries real per-cell accuracies.
PER_OP_CLASS_JSON_FIXED = ROOT / "outputs/decoder20260511/audit/per_op_class_v8_fixed.json"
PER_OP_CLASS_JSON = ROOT / "outputs/decoder20260511/audit/per_op_class_v8.json"
FIG_DIR = ROOT / "outputs/decoder20260511/decoder_paper_v2/figures"

# Headline 5-fold mean TF accuracies per head, from
# decoder_paper_v2/tables/headline_5fold.tex (TF column).
# The entropy JSON uses the head order: token, type, command, param_type, sign, digit.
HEADLINE_TF_ACC = {
    "token":      0.7811,
    "type":       0.9838,
    "command":    0.8875,
    "param_type": 0.9931,
    "sign":       0.9888,
    "digit":      0.5850,  # numeric / digit head pooled
}

# Colorblind-friendly palette (Wong, 2011) - 6 distinct colors.
HEAD_COLORS = {
    "token":      "#0072B2",  # blue
    "type":       "#E69F00",  # orange
    "command":    "#009E73",  # bluish green
    "param_type": "#CC79A7",  # reddish purple
    "sign":       "#D55E00",  # vermillion
    "digit":      "#56B4E9",  # sky blue
}

# Pretty labels for legend.
HEAD_LABELS = {
    "token":      "Token",
    "type":       "Type",
    "command":    "Command",
    "param_type": "Param-type",
    "sign":       "Sign",
    "digit":      "Digit (numeric)",
}


def setup_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def load_entropy():
    with open(ENTROPY_JSON) as f:
        data = json.load(f)
    return data


def load_per_op_accuracies():
    """Load LIVE per-op-class per-head TF accuracies.

    Returns dict[head][op_class] -> accuracy, or None on failure / missing.

    Prefers per_op_class_v8_fixed.json, which stores a plotting-friendly
    `aggregate_head_opclass` map: {head: {op_class: pooled_accuracy}}.
    Falls back to the legacy per_op_class_v8.json shape if needed.
    """
    # Preferred: the fixed file with a direct head->op_class->acc map.
    if PER_OP_CLASS_JSON_FIXED.exists():
        try:
            with open(PER_OP_CLASS_JSON_FIXED) as f:
                data = json.load(f)
            head_op = data.get("aggregate_head_opclass", {})
            result = {h: {} for h in HEADLINE_TF_ACC.keys()}
            found_any = False
            for head, by_op in head_op.items():
                if head not in result or not isinstance(by_op, dict):
                    continue
                for op_class, acc in by_op.items():
                    if isinstance(acc, (int, float)) and acc == acc:  # not nan
                        result[head][op_class] = float(acc)
                        found_any = True
            if found_any:
                return result
        except Exception:
            pass

    # Legacy fallback (original errored shape).
    if not PER_OP_CLASS_JSON.exists():
        return None
    try:
        with open(PER_OP_CLASS_JSON) as f:
            data = json.load(f)
    except Exception:
        return None

    agg = data.get("aggregate", {})
    if not agg:
        return None

    # aggregate[op_class][head] -> accuracy mean
    result = {h: {} for h in HEADLINE_TF_ACC.keys()}
    found_any = False
    for op_class, by_head in agg.items():
        if not isinstance(by_head, dict):
            continue
        for head, acc in by_head.items():
            if head in result and isinstance(acc, (int, float)):
                result[head][op_class] = float(acc)
                found_any = True
    return result if found_any else None


def build_points(entropy_data, per_op_acc):
    """Return list of (head, op_class, entropy_bits, accuracy)."""
    op_classes = entropy_data["op_classes"]
    heads = entropy_data["heads"]
    ent_matrix = entropy_data["entropy_matrix"]

    points = []
    for hi, head in enumerate(heads):
        for oi, op_class in enumerate(op_classes):
            ent = ent_matrix[hi][oi]
            if per_op_acc is not None and per_op_acc[head].get(op_class) is not None:
                acc = per_op_acc[head][op_class]
            else:
                acc = HEADLINE_TF_ACC[head]
            points.append((head, op_class, ent, acc))
    return points


def fit_and_stats(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    pearson_r, pearson_p = stats.pearsonr(xs, ys)
    spearman_rho, spearman_p = stats.spearmanr(xs, ys)
    slope, intercept, _, _, _ = stats.linregress(xs, ys)
    return pearson_r, pearson_p, spearman_rho, spearman_p, slope, intercept


def main():
    setup_style()
    entropy_data = load_entropy()
    per_op_acc = load_per_op_accuracies()
    used_per_op = per_op_acc is not None

    points = build_points(entropy_data, per_op_acc)
    xs = [p[2] for p in points]
    ys = [p[3] for p in points]

    pr, pp, sr, sp, slope, intercept = fit_and_stats(xs, ys)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Plot by head so legend groups colors cleanly.
    heads_order = entropy_data["heads"]
    for head in heads_order:
        head_pts = [(p[2], p[3]) for p in points if p[0] == head]
        hx = [p[0] for p in head_pts]
        hy = [p[1] for p in head_pts]
        ax.scatter(
            hx, hy,
            s=70,
            c=HEAD_COLORS[head],
            edgecolors="black",
            linewidths=0.6,
            alpha=0.85,
            label=HEAD_LABELS[head],
            zorder=3,
        )

    # Best-fit line.
    xline = np.linspace(min(xs) - 0.2, max(xs) + 0.2, 100)
    yline = slope * xline + intercept
    ax.plot(
        xline, yline,
        linestyle="--",
        color="gray",
        linewidth=1.5,
        alpha=0.8,
        zorder=2,
        label=f"OLS fit (slope={slope:+.3f})",
    )

    # Stats annotation box in upper right.
    stat_lines = [
        f"Pearson r = {pr:+.3f}  (p = {pp:.2g})",
        rf"Spearman $\rho$ = {sr:+.3f}  (p = {sp:.2g})",
        f"n = {len(xs)} (6 heads x 9 op-classes)",
    ]
    if not used_per_op:
        stat_lines.append("Accuracy: 5-fold TF mean per head")
    ax.text(
        0.97, 0.97,
        "\n".join(stat_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="0.4",
            alpha=0.92,
        ),
        zorder=5,
    )

    ax.set_xlabel("Empirical source entropy (bits)")
    ax.set_ylabel("Measured accuracy (TF, 5-fold mean)")
    ax.set_title("Source entropy bounds measured accuracy across heads and operation classes")

    # Sensible y-limits with headroom.
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(left=0.0)

    ax.legend(
        loc="lower left",
        framealpha=0.92,
        ncol=2,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / "entropy_accuracy_all_heads.pdf"
    png_path = FIG_DIR / "entropy_accuracy_all_heads.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    print(f"n_points={len(xs)} pearson_r={pr:+.4f} (p={pp:.3g}) "
          f"spearman_rho={sr:+.4f} (p={sp:.3g}) slope={slope:+.4f} "
          f"intercept={intercept:+.4f} used_per_op_class={used_per_op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
