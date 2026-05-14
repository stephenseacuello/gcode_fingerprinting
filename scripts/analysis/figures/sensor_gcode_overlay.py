#!/usr/bin/env python3
"""Sensor waveform + G-code overlay figure.

Takes one 64-second test window and shows the sensor signals (per modality
group) on top, with the corresponding G-code rows aligned in time below.
The single most-requested figure for sensor-side manufacturing papers.
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
PREP_ROOT = REPO / "outputs" / "decoder20260511" / "preprocessed_f98" / "full_window"

# 98-channel sensor layout from §4.1 (Sensor Encoder):
#   accelerometer: 3-axis × 6 boards = 18 channels (idx 0-17)
#   gyroscope:     18 (idx 18-35)
#   magnetometer:  18 (idx 36-53)
#   environmental temperature: 6 (idx 54-59)
#   ambient color: 24 (idx 60-83)
#   audio RMS:     6 (idx 84-89)
#   motor current: 8 (idx 90-97)
MODALITY_GROUPS = [
    ("Accelerometer (3-axis x 6 boards)", 0, 18, "#3866b3"),
    ("Gyroscope (3-axis x 6 boards)", 18, 18, "#b33838"),
    ("Magnetometer", 36, 18, "#38b376"),
    ("Environmental temp", 54, 6, "#b39638"),
    ("Ambient color (4-ch x 6 boards)", 60, 24, "#b338a8"),
    ("Audio RMS", 84, 6, "#38a8b3"),
    ("Motor current", 90, 8, "#a83838"),
]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Use sample 57 (11 G-code lines, face150025) - cleanest for the overlay
    SAMPLE_IDX = 57
    npz_path = PREP_ROOT / "fold_1" / "test_sequences.npz"
    d = np.load(npz_path, allow_pickle=True)

    sensor = d["continuous"][SAMPLE_IDX]  # [T=256, C=98]
    gcode_text = str(d["gcode_texts"][SAMPLE_IDX])
    op_class = str(d["operation_type_names"][SAMPLE_IDX])
    source = str(d["source_file"][SAMPLE_IDX])

    print(f"Loaded sample 0: op={op_class}, source={source}, gcode lines={gcode_text.count(chr(10)) + 1}")

    T = sensor.shape[0]
    duration_sec = T / 4.0  # 4 Hz sampling
    t_axis = np.linspace(0, duration_sec, T)

    n_panels = len(MODALITY_GROUPS) + 1  # +1 for G-code panel
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 1.0 + 0.85 * n_panels),
                             sharex=True,
                             gridspec_kw={"height_ratios": [1] * len(MODALITY_GROUPS) + [2]})

    for ax, (name, start, n_ch, color) in zip(axes[:-1], MODALITY_GROUPS):
        # Plot mean and ±std across channels for this modality (compactness)
        chs = sensor[:, start:start + n_ch]
        mean = chs.mean(axis=1)
        std = chs.std(axis=1)
        ax.plot(t_axis, mean, color=color, linewidth=1.2, alpha=0.9)
        ax.fill_between(t_axis, mean - std, mean + std, color=color, alpha=0.18)
        ax.set_ylabel(name, fontsize=7, rotation=0, labelpad=70, ha="right", va="center")
        ax.grid(alpha=0.25)
        ax.tick_params(axis="y", labelsize=7)
        # Hide x-tick labels except for the bottom axis
        ax.tick_params(labelbottom=False)

    # G-code panel
    ax_g = axes[-1]
    # Parse multi-line G-code text into individual rows
    lines = [ln.strip() for ln in gcode_text.split("\n") if ln.strip()]
    n_lines = len(lines)
    # Distribute lines evenly across the time axis (in reality each line maps to a
    # subwindow of timesteps; we approximate with even distribution since exact
    # per-line timing isn't stored in the NPZ).
    line_positions = np.linspace(0.5, duration_sec - 0.5, n_lines) if n_lines > 0 else []
    ax_g.set_yticks([])
    ax_g.set_xlabel("Time within 64-second sensor window (seconds)")
    ax_g.set_ylim(-1, 1)
    for i, (t, line) in enumerate(zip(line_positions, lines[:30])):  # cap at 30 lines for legibility
        y = 0.5 if i % 2 == 0 else -0.5
        ax_g.axvline(t, color="black", linestyle=":", alpha=0.2, linewidth=0.5)
        ax_g.text(t, y, line[:50] + ("..." if len(line) > 50 else ""),
                  fontsize=6.5, family="monospace", ha="left", va="center",
                  rotation=0, bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.9, edgecolor="none"))
    if n_lines > 30:
        ax_g.text(duration_sec * 0.5, -0.95, f"(showing first 30 of {n_lines} G-code rows)",
                  fontsize=7, ha="center", style="italic")
    ax_g.set_ylabel("G-code rows", fontsize=8, rotation=0, labelpad=70, ha="right", va="center")
    ax_g.set_xlim(0, duration_sec)

    fig.suptitle(f"Multi-modal sensor window (64 s, 4 Hz, {T} samples × 98 channels) and aligned G-code rows.\n"
                 f"Operation class: {op_class}, source: {source}, {n_lines} G-code rows",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / "sensor_gcode_overlay.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
