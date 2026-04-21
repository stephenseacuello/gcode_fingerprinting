#!/usr/bin/env python3
"""Generate publication-quality figures for the decoder paper.

Figure 1: PCA of encoder embeddings colored by 9 operation classes.
Figure 2: Raw sensor time series comparison across 3 operation types.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from pathlib import Path

# ── Publication settings ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "text.usetex": False,
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 0.8,
})

OUT_DIR = Path("outputs/decoder20260304/paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = Path("outputs/decoder20260304")

BEST_SEEDS = {1: 2024, 2: 123, 3: 456, 4: 2024, 5: 789}

# Pretty label mapping
LABEL_MAP = {
    "adaptive":        "Adaptive (cut)",
    "adaptive150025":  "Adaptive (air)",
    "damageadaptive":  "Adaptive (dmg)",
    "face":            "Face (cut)",
    "face150025":      "Face (air)",
    "damageface":      "Face (dmg)",
    "pocket":          "Pocket (cut)",
    "pocket150025":    "Pocket (air)",
    "damagepocket":    "Pocket (dmg)",
}

# Group-aware color palette: adaptive=blues, face=reds/oranges, pocket=greens
# Within each group: cut=saturated, air=lighter, dmg=dark/accented
COLOR_MAP = {
    "Adaptive (cut)": "#2166ac",
    "Adaptive (air)": "#92c5de",
    "Adaptive (dmg)": "#053061",
    "Face (cut)":     "#d6604d",
    "Face (air)":     "#f4a582",
    "Face (dmg)":     "#67001f",
    "Pocket (cut)":   "#1b7837",
    "Pocket (air)":   "#a6d96a",
    "Pocket (dmg)":   "#00441b",
}

# Marker per toolpath strategy group
MARKER_MAP = {
    "Adaptive (cut)": "o",
    "Adaptive (air)": "o",
    "Adaptive (dmg)": "o",
    "Face (cut)":     "s",
    "Face (air)":     "s",
    "Face (dmg)":     "s",
    "Pocket (cut)":   "^",
    "Pocket (air)":   "^",
    "Pocket (dmg)":   "^",
}

# Desired legend order (grouped by strategy)
LEGEND_ORDER = [
    "Adaptive (air)", "Adaptive (cut)", "Adaptive (dmg)",
    "Face (air)",     "Face (cut)",     "Face (dmg)",
    "Pocket (air)",   "Pocket (cut)",   "Pocket (dmg)",
]


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — PCA of encoder embeddings
# ════════════════════════════════════════════════════════════════════════════
def generate_pca_figure():
    print("Generating Figure 1: encoder_embedding_pca")

    all_embeddings = []
    all_labels = []

    for fold, seed in BEST_SEEDS.items():
        mem_path = (
            BASE / f"v7_best_5fold_multiseed/fold_{fold}_seed_{seed}"
            / "encoder_memory/test_memory.pt"
        )
        mem = torch.load(mem_path, map_location="cpu", weights_only=False)
        # mem shape: (N_test, 256, 256) — mean-pool over sequence dim
        emb = mem.mean(dim=1).numpy()  # (N_test, 256)
        all_embeddings.append(emb)

        npz_path = BASE / f"preprocessed_v7/fold_{fold}/test_sequences.npz"
        d = np.load(npz_path, allow_pickle=True)
        names = d["operation_type_names"]
        pretty = np.array([LABEL_MAP[n] for n in names])
        all_labels.append(pretty)

    X = np.concatenate(all_embeddings, axis=0)  # (552, 256)
    y = np.concatenate(all_labels, axis=0)       # (552,)

    print(f"  Combined: {X.shape[0]} points, {X.shape[1]} dims")

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_

    print(f"  Explained variance: PC1={ev[0]:.3f}, PC2={ev[1]:.3f}")

    # ── Plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 3.8))

    # Plot each class, in legend order, so legend is tidy
    for label in LEGEND_ORDER:
        mask = y == label
        if not mask.any():
            continue
        ax.scatter(
            X2[mask, 0], X2[mask, 1],
            c=COLOR_MAP[label],
            marker=MARKER_MAP[label],
            s=28,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.3,
            label=label,
            zorder=3,
        )

    ax.set_xlabel(f"PC 1 ({ev[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC 2 ({ev[1]*100:.1f}% variance)")

    # Legend outside plot area, to the right
    leg = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        handletextpad=0.4,
        columnspacing=0.8,
        markerscale=1.1,
    )
    leg.get_frame().set_linewidth(0.4)

    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = OUT_DIR / f"encoder_embedding_pca.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")

    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Raw sensor time series comparison
# ════════════════════════════════════════════════════════════════════════════
def generate_timeseries_figure():
    print("Generating Figure 2: sensor_timeseries_comparison")

    npz_path = BASE / "preprocessed_v7/fold_1/test_sequences.npz"
    d = np.load(npz_path, allow_pickle=True)
    continuous = d["continuous"]    # (132, 256, 110)
    names = d["operation_type_names"]

    # Hand-picked windows with highest signal dynamics for visual contrast:
    # adaptive idx=107: high accel+gyro variability (direction changes)
    # face idx=23: high audio variance (step-over transitions visible)
    # pocket idx=69: high gyro variance (plunge-clear alternation)
    picks = {"adaptive": 107, "face": 23, "pocket": 69}

    print(f"  Selected windows: {picks}")

    # Channel indices (board 1):
    # 0-2: Ax, Ay, Az  |  3-5: Gx, Gy, Gz  |  16: audio RMS
    # These are the first board's channels in the 110-column layout.

    titles = {
        "adaptive": "Adaptive Clearing",
        "face":     "Face Milling",
        "pocket":   "Pocket Machining",
    }

    channel_colors = ["#2166ac", "#d6604d", "#7a7a7a"]
    channel_labels = ["Accel. magnitude", "Gyro. magnitude", "Audio RMS"]

    fig, axes = plt.subplots(3, 1, figsize=(5.5, 5.0), sharex=True)

    timesteps = np.arange(256)

    for row_idx, op_name in enumerate(["adaptive", "face", "pocket"]):
        ax = axes[row_idx]
        win_idx = picks[op_name]
        data = continuous[win_idx]  # (256, 110)

        # Accelerometer magnitude (columns 0-2)
        accel = np.sqrt(data[:, 0]**2 + data[:, 1]**2 + data[:, 2]**2)
        # Gyroscope magnitude (columns 3-5)
        gyro = np.sqrt(data[:, 3]**2 + data[:, 4]**2 + data[:, 5]**2)
        # Audio RMS (column 16)
        audio = data[:, 16]

        # Normalize each signal to [0, 1] for visual comparison
        def norm(x):
            mn, mx = x.min(), x.max()
            if mx - mn < 1e-8:
                return x - mn
            return (x - mn) / (mx - mn)

        accel_n = norm(accel)
        gyro_n = norm(gyro)
        audio_n = norm(audio)

        ax.plot(timesteps, accel_n, color=channel_colors[0],
                linewidth=0.7, alpha=0.85, label=channel_labels[0])
        ax.plot(timesteps, gyro_n, color=channel_colors[1],
                linewidth=0.7, alpha=0.85, label=channel_labels[1])
        ax.plot(timesteps, audio_n, color=channel_colors[2],
                linewidth=0.7, alpha=0.85, label=channel_labels[2])

        ax.set_title(titles[op_name], fontweight="bold", pad=4)
        ax.set_ylabel("Normalized\namplitude")
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([0, 0.5, 1.0])

        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.set_axisbelow(True)

        if row_idx == 0:
            ax.legend(
                loc="upper right",
                frameon=True,
                framealpha=0.9,
                edgecolor="#cccccc",
                ncol=3,
                handlelength=1.5,
                columnspacing=1.0,
            )

    axes[-1].set_xlabel("Timestep")
    axes[-1].set_xlim(0, 255)

    fig.tight_layout(h_pad=0.6)

    for ext in ("pdf", "png"):
        path = OUT_DIR / f"sensor_timeseries_comparison.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")

    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    generate_pca_figure()
    generate_timeseries_figure()
    print("\nDone.")
