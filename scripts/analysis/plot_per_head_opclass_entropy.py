#!/usr/bin/env python3
"""Plot per-(head, op-class) source entropy heatmap.

Reads per_head_per_opclass_entropy.json and emits a 6x9 heatmap (heads x op-classes)
colored by empirical Shannon entropy in bits with a sequential colormap (viridis).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AUDIT_JSON = Path(
    "/home/seacuello/Documents/gcode_fingerprinting/outputs/decoder20260511/audit/per_head_per_opclass_entropy.json"
)
OUT_DIR = Path(
    "/home/seacuello/Documents/gcode_fingerprinting/outputs/decoder20260511/decoder_paper_v2/figures"
)
OUT_BASE = OUT_DIR / "per_head_opclass_entropy_heatmap"


def main() -> None:
    with AUDIT_JSON.open("r") as f:
        data = json.load(f)

    op_classes = data["op_classes"]
    heads = data["heads"]
    entropy = np.array(data["entropy_matrix"], dtype=float)  # shape (6, 9)

    assert entropy.shape == (len(heads), len(op_classes)), (
        f"Unexpected matrix shape {entropy.shape}; expected {(len(heads), len(op_classes))}"
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.0))

    im = ax.imshow(
        entropy,
        aspect="auto",
        cmap="viridis",
        vmin=float(np.nanmin(entropy)),
        vmax=float(np.nanmax(entropy)),
    )

    # Tick labels
    ax.set_xticks(np.arange(len(op_classes)))
    ax.set_xticklabels(op_classes, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(heads)))
    ax.set_yticklabels(heads, fontsize=10)

    # Annotate cells
    vmax = float(np.nanmax(entropy))
    vmin = float(np.nanmin(entropy))
    threshold = vmin + 0.55 * (vmax - vmin)
    for i in range(entropy.shape[0]):
        for j in range(entropy.shape[1]):
            value = entropy[i, j]
            text_color = "white" if value < threshold else "black"
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    ax.set_xlabel("Operation class", fontsize=11)
    ax.set_ylabel("Decoder head", fontsize=11)
    ax.set_title("Per-head per-operation-class source entropy (bits)", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Empirical entropy (bits)", fontsize=10)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_BASE.with_suffix(".pdf")
    png_path = OUT_BASE.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
