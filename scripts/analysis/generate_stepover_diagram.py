#!/usr/bin/env python3
"""
Generate a publication-quality step-over diagram showing why Y-axis
lateral passes produce indistinguishable sensor signatures.

Replaces the TikZ version that had overlapping labels and truncated values.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

# ---------- style ----------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 10,
    "text.usetex": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
})

# ---------- layout constants -----------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))

# Workpiece rectangle (data coordinates)
wp_left, wp_bottom = 1.0, 0.5
wp_width, wp_height = 5.0, 3.5
wp_right = wp_left + wp_width
wp_top = wp_bottom + wp_height

# Draw workpiece
workpiece = mpatches.FancyBboxPatch(
    (wp_left, wp_bottom), wp_width, wp_height,
    boxstyle="round,pad=0.05",
    facecolor="#d9d9d9", edgecolor="black", linewidth=1.2, zorder=1
)
ax.add_patch(workpiece)

# Title above workpiece
ax.text(
    wp_left + wp_width / 2, wp_top + 0.20,
    "Workpiece (top view)",
    ha="center", va="bottom", fontsize=11, fontweight="bold",
    fontstyle="italic",
)

# ---------- pass definitions -----------------------------------------------
# Map real Y values to visual positions inside the workpiece
# We space them evenly for clarity but label with true values.
passes = [
    {"y_val": 0.1755, "vis_y": wp_bottom + 0.65, "color": "#1f77b4", "dir": 1},   # right
    {"y_val": 0.4489, "vis_y": wp_bottom + 1.75, "color": "#d62728", "dir": -1},   # left
    {"y_val": 2.9132, "vis_y": wp_bottom + 2.85, "color": "#2ca02c", "dir": 1},    # right
]

arrow_pad = 0.30  # padding from workpiece edges
arrow_lw = 2.5
head_len = 0.20
head_wid = 0.12

for p in passes:
    y = p["vis_y"]
    if p["dir"] == 1:  # pointing right
        x_start = wp_left + arrow_pad
        x_end = wp_right - arrow_pad
    else:  # pointing left
        x_start = wp_right - arrow_pad
        x_end = wp_left + arrow_pad

    arrow = FancyArrowPatch(
        (x_start, y), (x_end, y),
        arrowstyle="->,head_length=8,head_width=5",
        color=p["color"], linewidth=arrow_lw, zorder=3,
        mutation_scale=1,
    )
    ax.add_patch(arrow)

    # Y-value label to the left of the workpiece
    ax.text(
        wp_left - 0.15, y,
        f"Y = {p['y_val']:.4f}",
        ha="right", va="center", fontsize=9, color=p["color"],
        fontweight="bold",
    )

# ---------- step-over annotation (right side) ------------------------------
# Double-headed arrow between bottom and top pass
so_x = wp_right + 0.45
y_bot = passes[0]["vis_y"]
y_top = passes[2]["vis_y"]

ax.annotate(
    "", xy=(so_x, y_top), xytext=(so_x, y_bot),
    arrowprops=dict(
        arrowstyle="<->", color="black", lw=1.3,
        shrinkA=0, shrinkB=0,
    ),
    zorder=4,
)
ax.text(
    so_x + 0.12, (y_bot + y_top) / 2,
    "step-over\n(Y axis)",
    ha="left", va="center", fontsize=9, rotation=90,
    fontstyle="italic",
)

# ---------- coordinate axes indicator (bottom-left) ------------------------
origin_x, origin_y = 0.25, 0.15
axis_len = 0.55

# X arrow
ax.annotate(
    "", xy=(origin_x + axis_len, origin_y), xytext=(origin_x, origin_y),
    arrowprops=dict(arrowstyle="->,head_width=4,head_length=4",
                    color="black", lw=1.2),
    zorder=5,
)
ax.text(origin_x + axis_len + 0.08, origin_y, "X", fontsize=10,
        fontweight="bold", va="center", ha="left")

# Y arrow
ax.annotate(
    "", xy=(origin_x, origin_y + axis_len), xytext=(origin_x, origin_y),
    arrowprops=dict(arrowstyle="->,head_width=4,head_length=4",
                    color="black", lw=1.2),
    zorder=5,
)
ax.text(origin_x, origin_y + axis_len + 0.08, "Y", fontsize=10,
        fontweight="bold", va="bottom", ha="center")

# Z label
ax.text(origin_x + 0.05, origin_y - 0.22, "Z into page", fontsize=8,
        va="top", ha="left", fontstyle="italic", color="#555555")

# ---------- explanation text box at bottom ---------------------------------
caption = (
    "Same depth (Z), feed (F), and radial engagement "
    r"$\rightarrow$ near-identical sensor signatures"
)
ax.text(
    wp_left + wp_width / 2, -0.15,
    caption,
    ha="center", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffcc",
              edgecolor="#999999", linewidth=0.8),
    zorder=5,
)

# ---------- axis cleanup ---------------------------------------------------
ax.set_xlim(-0.6, wp_right + 1.4)
ax.set_ylim(-0.65, wp_top + 0.55)
ax.set_aspect("equal")
ax.axis("off")

# ---------- save -----------------------------------------------------------
out_dir = "/home/seacuello/Documents/gcode_fingerprinting/outputs/decoder20260304/paper/figures"
os.makedirs(out_dir, exist_ok=True)

for ext in ("pdf", "png"):
    path = os.path.join(out_dir, f"stepover_diagram.{ext}")
    fig.savefig(path, bbox_inches="tight", dpi=300, facecolor="white")
    print(f"Saved: {path}")

plt.close(fig)
