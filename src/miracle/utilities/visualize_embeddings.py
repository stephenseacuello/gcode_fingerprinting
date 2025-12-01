#!/usr/bin/env python3
"""
Visualize learned embeddings from inference results.

Creates 2D visualizations of high-dimensional embeddings using:
- PCA (Principal Component Analysis)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)

Usage:
    python src/miracle/utilities/visualize_embeddings.py outputs/inference/test_results.npz
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_embeddings(results_path: Path, output_dir: Path = None):
    """Load embeddings and create 2D visualizations."""

    # Load results
    print(f"Loading results from {results_path}...")
    data = np.load(results_path)
    embeddings = data['embeddings']  # (N, 128)
    labels = data['cls_labels']      # (N,)
    recon_loss = data['recon_loss']  # (N,)

    n_samples, n_dims = embeddings.shape
    print(f"✓ Loaded {n_samples} samples with {n_dims}-dimensional embeddings")

    # Setup output directory
    if output_dir is None:
        output_dir = results_path.parent / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # === PCA Visualization ===
    print("\nRunning PCA (2 components)...")
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)

    explained_var = pca.explained_variance_ratio_
    print(f"  Explained variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}")
    print(f"  Total: {explained_var.sum():.2%}")

    ax = axes[0]
    scatter = ax.scatter(
        embeddings_pca[:, 0],
        embeddings_pca[:, 1],
        c=labels,
        cmap='tab10',
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5
    )

    # Add sample numbers
    for i, (x, y) in enumerate(embeddings_pca):
        ax.annotate(
            f'{i}',
            (x, y),
            fontsize=10,
            fontweight='bold',
            ha='center',
            va='center'
        )

    ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12)
    ax.set_title('PCA: Linear Dimensionality Reduction', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Colorbar for labels
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('G-code Line', fontsize=11)

    # === t-SNE Visualization ===
    if n_samples >= 4:  # t-SNE needs at least perplexity+1 samples
        print("\nRunning t-SNE (2 components)...")
        perplexity = min(3, n_samples - 1)  # perplexity < n_samples
        print(f"  Using perplexity={perplexity}")

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=1000,
            random_state=42,
            init='pca'
        )
        embeddings_tsne = tsne.fit_transform(embeddings)

        ax = axes[1]
        scatter = ax.scatter(
            embeddings_tsne[:, 0],
            embeddings_tsne[:, 1],
            c=labels,
            cmap='tab10',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5
        )

        # Add sample numbers
        for i, (x, y) in enumerate(embeddings_tsne):
            ax.annotate(
                f'{i}',
                (x, y),
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='center'
            )

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('t-SNE: Non-linear Dimensionality Reduction', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Colorbar for labels
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('G-code Line', fontsize=11)
    else:
        ax = axes[1]
        ax.text(
            0.5, 0.5,
            f'Not enough samples for t-SNE\n(need at least 4, have {n_samples})',
            ha='center',
            va='center',
            fontsize=12,
            transform=ax.transAxes
        )
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(
        f'Embedding Visualization ({n_samples} samples, {n_dims}D → 2D)',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()

    # Save plot
    output_file = output_dir / 'embeddings_2d.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_file}")

    # === Create reconstruction loss visualization ===
    fig, ax = plt.subplots(figsize=(8, 6))

    # Bar plot of reconstruction loss per sample
    bars = ax.bar(
        range(n_samples),
        recon_loss,
        color=['red' if labels[i] == -1 else f'C{int(labels[i])}' for i in range(n_samples)],
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5
    )

    ax.axhline(
        y=recon_loss.mean(),
        color='black',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {recon_loss.mean():.4f}'
    )

    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Reconstruction Loss', fontsize=12)
    ax.set_title('Per-Sample Reconstruction Loss', fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_samples))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot
    output_file = output_dir / 'reconstruction_loss.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved reconstruction loss plot to {output_file}")

    # === Summary Statistics ===
    print("\n" + "="*70)
    print("EMBEDDING ANALYSIS SUMMARY")
    print("="*70)

    print(f"\nEmbedding Statistics:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std:  {embeddings.std():.4f}")
    print(f"  Min:  {embeddings.min():.4f}")
    print(f"  Max:  {embeddings.max():.4f}")

    print(f"\nPCA Analysis:")
    print(f"  First 2 components explain {explained_var.sum():.2%} of variance")
    print(f"  Embeddings have some linear structure")

    print(f"\nReconstruction Loss:")
    print(f"  Mean: {recon_loss.mean():.4f}")
    print(f"  Std:  {recon_loss.std():.4f}")
    print(f"  Range: [{recon_loss.min():.4f}, {recon_loss.max():.4f}]")

    # Check if samples cluster by label
    unique_labels = np.unique(labels[labels >= 0])
    if len(unique_labels) > 1:
        print(f"\nLabel Distribution:")
        for label in unique_labels:
            count = (labels == label).sum()
            print(f"  G-code line {int(label)}: {count} samples")
    else:
        print(f"\nAll samples have the same label (no clustering expected)")

    print("\n" + "="*70)
    print("\nInterpretation:")
    print("="*70)

    if n_samples < 10:
        print("""
⚠️  Very small sample size ({n_samples} samples)
  • Hard to draw conclusions about clustering
  • PCA/t-SNE visualizations may not be meaningful
  • Need more data for robust analysis

Recommendation:
  • Collect more test samples (aim for 20+)
  • Run inference on training set to see learned patterns
  • Compare embeddings across different data splits
""".format(n_samples=n_samples))
    else:
        print("""
✓ Sufficient samples for basic analysis

Look for:
  • Do samples with same label cluster together?
  • Are there clear separations between different operations?
  • Any outliers far from other samples?

If no clear clustering:
  • Labels may not have strong signal in sensor data
  • Model may need more training
  • Feature engineering may be needed
""")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_embeddings.py <results.npz>")
        print("Example: python visualize_embeddings.py outputs/inference/test_results.npz")
        sys.exit(1)

    results_path = Path(sys.argv[1])

    if not results_path.exists():
        print(f"Error: {results_path} does not exist")
        sys.exit(1)

    visualize_embeddings(results_path)
