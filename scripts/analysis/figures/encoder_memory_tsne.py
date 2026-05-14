#!/usr/bin/env python3
"""t-SNE / UMAP of encoder memory colored by operation class.

The frozen encoder maps 64-second sensor windows to a [256, 256] memory
tensor. We mean-pool over the time dimension to get a [N, 256] embedding
per window, then run t-SNE to 2D.

If the encoder embedding cleanly separates the 9 operation classes,
expect compact clusters; if classes overlap heavily the encoder is not
class-discriminative and downstream decoder accuracy is bounded
accordingly.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
FIG_DIR = REPO / "outputs" / "decoder20260511" / "decoder_paper_v2" / "figures"
PREP_ROOT = REPO / "outputs" / "decoder20260511" / "preprocessed_f98" / "full_window"
MEMORY_ROOT = REPO / "outputs" / "decoder20260511" / "checkpoints" / "full_window_5fold"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Pool encoder memory across all 5 folds' test splits
    all_embed = []
    all_ops = []
    for F in range(1, 6):
        mem_files = list((MEMORY_ROOT / f"fold_{F}").glob("*/encoder_memory/test_memory.pt"))
        mem_files = [m for m in mem_files if "_fsm" not in str(m)]
        if not mem_files:
            print(f"fold {F}: no memory found")
            continue
        mem_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        mem = torch.load(mem_files[0], weights_only=False, map_location="cpu").numpy()  # [N, T=256, d=256]
        npz = np.load(PREP_ROOT / f"fold_{F}" / "test_sequences.npz", allow_pickle=True)
        ops = [str(x) for x in npz["operation_type_names"]]
        if len(ops) != mem.shape[0]:
            print(f"fold {F}: mismatch n_mem={mem.shape[0]} vs n_ops={len(ops)}")
            continue
        # Mean-pool over time dimension
        pooled = mem.mean(axis=1)  # [N, 256]
        all_embed.append(pooled)
        all_ops.extend(ops)

    X = np.concatenate(all_embed, axis=0)
    y = np.array(all_ops)
    print(f"Collected encoder embeddings: {X.shape}, n_op_classes={len(set(y))}")

    # Run t-SNE
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn missing, bailing")
        return
    print("Running t-SNE (perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    Z = tsne.fit_transform(X)

    # Plot
    classes = sorted(set(y))
    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for i, c in enumerate(classes):
        m = (y == c)
        ax.scatter(Z[m, 0], Z[m, 1], s=22, alpha=0.7, label=f"{c} (n={m.sum()})",
                   color=cmap(i / len(classes)), edgecolor="white", linewidth=0.3)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title(f"t-SNE of frozen encoder memory (mean-pooled over time),\n"
                 f"coloured by operation class — {X.shape[0]} test samples, 5 folds")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "encoder_memory_tsne.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
