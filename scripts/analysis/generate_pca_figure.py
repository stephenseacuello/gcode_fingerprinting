#!/usr/bin/env python3
"""Generate PCA analysis figure from fingerprint embeddings.

Loads the pre-computed t-SNE data (which contains the raw fingerprint embeddings
via model checkpoints) or extracts fingerprints directly, runs PCA, and generates:
  1. 2D scatter of PC1 vs PC2 colored by class
  2. Cumulative explained variance curve (inset or side panel)

Usage:
    python scripts/analysis/generate_pca_figure.py \
        --experiments-dir outputs/experiments_2026_02_25 \
        --output-dir outputs/experiments_2026_02_25/paper/figures
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

# ── Constants (match generate_training_figures.py) ──

CLASS_NAMES_9 = [
    'adaptive', 'adaptive150025', 'face', 'face150025',
    'pocket', 'pocket150025', 'damageadaptive', 'damageface', 'damagepocket',
]

CLASS_DISPLAY = {
    'adaptive': 'Adaptive', 'adaptive150025': 'Adaptive-HF',
    'face': 'Face', 'face150025': 'Face-HF',
    'pocket': 'Pocket', 'pocket150025': 'Pocket-HF',
    'damageadaptive': 'Dmg-Adaptive', 'damageface': 'Dmg-Face',
    'damagepocket': 'Dmg-Pocket',
}

CLASS_COLORS = {
    'adaptive': '#1f77b4', 'adaptive150025': '#17becf',
    'face': '#2ca02c', 'face150025': '#7f7f7f',
    'pocket': '#ff7f0e', 'pocket150025': '#bcbd22',
    'damageadaptive': '#d62728', 'damageface': '#e377c2',
    'damagepocket': '#9467bd',
}

CLASS_MARKERS = {
    'adaptive': 'o', 'adaptive150025': 's', 'face': '^', 'face150025': 'D',
    'pocket': 'v', 'pocket150025': 'P', 'damageadaptive': 'X',
    'damageface': '*', 'damagepocket': 'h',
}

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 8, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
})


def extract_all_fingerprints(experiment_dir, n_folds=5, device='cpu'):
    """Extract 128-d fingerprint embeddings from all folds."""
    # Reuse the extraction logic from generate_training_figures
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_training_figures import extract_fingerprints

    all_fps, all_labels, all_folds = [], [], []
    for fold in range(1, n_folds + 1):
        fps, labels = extract_fingerprints(experiment_dir, fold, device)
        if fps is not None:
            all_fps.append(fps)
            all_labels.append(labels)
            all_folds.append(np.full(len(labels), fold))
            print(f"  Fold {fold}: {len(labels)} samples, dim={fps.shape[1]}")

    fps = np.concatenate(all_fps, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    folds = np.concatenate(all_folds, axis=0)
    return fps, labels, folds


def main():
    parser = argparse.ArgumentParser(description='PCA analysis of fingerprint embeddings')
    parser.add_argument('--experiments-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=str,
                        default='no_proximity_no_pressure_w64_s16_cv')
    parser.add_argument('--n-components', type=int, default=20,
                        help='Number of PCA components to compute')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = args.experiments_dir / args.config

    print(f"Extracting fingerprints from {experiment_dir}...")
    fps, labels, folds = extract_all_fingerprints(experiment_dir, device=args.device)
    print(f"Total: {len(labels)} samples, embedding dim={fps.shape[1]}")

    # Run PCA
    n_comp = min(args.n_components, fps.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(fps)
    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)

    # Find 95% threshold
    n95 = np.searchsorted(cumvar, 0.95) + 1
    print(f"\nPCA Results:")
    print(f"  PC1: {evr[0]*100:.1f}% variance")
    print(f"  PC2: {evr[1]*100:.1f}% variance")
    print(f"  PC1+PC2: {cumvar[1]*100:.1f}% variance")
    print(f"  Components for 95% variance: {n95}")
    print(f"  Top 5 explained variance: {[f'{v*100:.1f}%' for v in evr[:5]]}")

    # ── Figure: 2-panel (scatter + explained variance) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={'width_ratios': [3, 2]})

    # Panel (a): PC1 vs PC2 scatter
    for cls_idx, cls_name in enumerate(CLASS_NAMES_9):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        ax1.scatter(coords[mask, 0], coords[mask, 1],
                    c=CLASS_COLORS[cls_name],
                    marker=CLASS_MARKERS[cls_name],
                    s=25, alpha=0.7, edgecolors='white', linewidths=0.3,
                    label=CLASS_DISPLAY[cls_name])

    ax1.set_xlabel(f'PC1 ({evr[0]*100:.1f}% variance)')
    ax1.set_ylabel(f'PC2 ({evr[1]*100:.1f}% variance)')
    ax1.set_title('(a) PCA of Fingerprint Embeddings')
    ax1.legend(loc='best', markerscale=1.5, framealpha=0.9, fontsize=7)

    # Panel (b): Cumulative explained variance
    components = np.arange(1, n_comp + 1)
    ax2.bar(components, evr * 100, color='#4CAF50', alpha=0.6, label='Individual')
    ax2.plot(components, cumvar * 100, 'o-', color='#1565C0', markersize=4,
             linewidth=1.5, label='Cumulative')
    ax2.axhline(y=95, color='#F44336', linestyle='--', linewidth=1, alpha=0.7,
                label='95% threshold')
    ax2.axvline(x=n95, color='#F44336', linestyle=':', linewidth=1, alpha=0.5)
    ax2.annotate(f'{n95} PCs', xy=(n95, 95), xytext=(n95 + 1, 85),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='#F44336'),
                 color='#F44336')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance (%)')
    ax2.set_title('(b) Cumulative Explained Variance')
    ax2.set_xlim(0.5, n_comp + 0.5)
    ax2.set_ylim(0, 105)
    ax2.legend(loc='center right', fontsize=8)

    fig.tight_layout()
    out_path = args.output_dir / f'fig_pca_{args.config}.pdf'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nSaved figure: {out_path}")

    # Save PCA data for reproducibility
    np.savez(args.output_dir / f'pca_data_{args.config}.npz',
             coords=coords, labels=labels, folds=folds,
             explained_variance_ratio=evr,
             cumulative_variance=cumvar,
             n_components_95pct=n95,
             class_names=CLASS_NAMES_9)
    print(f"Saved PCA data: pca_data_{args.config}.npz")

    # Print summary table
    print(f"\n{'Component':<12} {'Var (%)':<10} {'Cumul (%)':<10}")
    print('-' * 32)
    for i in range(min(10, n_comp)):
        print(f"PC{i+1:<10} {evr[i]*100:>8.2f}  {cumvar[i]*100:>8.2f}")


if __name__ == '__main__':
    main()
