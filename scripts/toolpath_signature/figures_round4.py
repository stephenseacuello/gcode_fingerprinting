"""Round 4 figures: noise robustness, bin importance, bootstrap CIs."""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolpath_signature import config as cfg

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
PALETTE = sns.color_palette("colorblind")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(cfg.FIGURES_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png/pdf")


def load_r4():
    with open(cfg.RESULTS_DIR / "round4_statistical.json") as f:
        return json.load(f)


def fig_noise_robustness(data):
    """Noise robustness curve."""
    print("Generating fig10_noise_robustness...")
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, iso in enumerate(["raw", "subtracted"]):
        if iso not in data["noise_robustness"]:
            continue
        nr = data["noise_robustness"][iso]
        levels = sorted([float(k) for k in nr.keys()])
        accs = [nr[str(l)]["accuracy_mean"] for l in levels]
        stds = [nr[str(l)]["accuracy_std"] for l in levels]

        ax.errorbar(levels, accs, yerr=stds, marker="o", capsize=4,
                   linewidth=2, markersize=7, label=iso.capitalize(), color=PALETTE[i])
        lower = np.clip(np.array(accs) - np.array(stds), 0, 1)
        upper = np.clip(np.array(accs) + np.array(stds), 0, 1)
        ax.fill_between(levels, lower, upper, alpha=0.1, color=PALETTE[i])

    ax.axhline(1/3, color="gray", linestyle="--", linewidth=1, label="Random baseline")
    ax.set_xlabel("Noise Level (× feature std)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Noise Robustness: Raw vs Subtracted Features")
    ax.set_ylim(0.25, 1.05)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig10_noise_robustness")


def fig_bin_importance(data):
    """Temporal bin importance bar chart."""
    print("Generating fig11_bin_importance...")
    if "bin_importance" not in data:
        print("  No bin importance data, skipping")
        return

    bi = data["bin_importance"]
    full_acc = bi["full"]["accuracy"]
    per_bin = bi["per_bin_removal"]

    bins = sorted([int(k) for k in per_bin.keys()])
    drops = [full_acc - per_bin[str(b)]["accuracy"] for b in bins]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PALETTE[0] if d > 0 else PALETTE[2] for d in drops]
    ax.bar(range(1, len(bins) + 1), drops, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Temporal Bin Removed")
    ax.set_ylabel("Accuracy Drop (full - ablated)")
    ax.set_title("Temporal Bin Importance: Accuracy Drop When Each Bin Is Removed")
    ax.set_xticks(range(1, len(bins) + 1))
    fig.tight_layout()
    save_fig(fig, "fig11_bin_importance")


def fig_bootstrap_ci(data):
    """Bootstrap CI comparison."""
    print("Generating fig12_bootstrap_ci...")
    if "bootstrap_ci" not in data:
        print("  No bootstrap data, skipping")
        return

    bi = data["bootstrap_ci"]
    methods = list(bi.keys())
    means = [bi[m]["mean"] for m in methods]
    ci_low = [bi[m]["ci_lower"] for m in methods]
    ci_high = [bi[m]["ci_upper"] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(methods))
    for i, m in enumerate(methods):
        ax.errorbar(i, means[i], yerr=[[means[i] - ci_low[i]], [ci_high[i] - means[i]]],
                   fmt="o", markersize=10, capsize=8, color=PALETTE[i], linewidth=2,
                   label=f"{m.capitalize()}: [{ci_low[i]:.3f}, {ci_high[i]:.3f}]")

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in methods])
    ax.set_ylabel("Accuracy")
    ax.set_title("Bootstrap 95% Confidence Intervals (RF, 1000 bootstrap samples)")
    ax.set_ylim(0.95, 1.005)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig12_bootstrap_ci")


if __name__ == "__main__":
    data = load_r4()
    fig_noise_robustness(data)
    fig_bin_importance(data)
    fig_bootstrap_ci(data)
    print("\nAll round 4 figures complete.")
