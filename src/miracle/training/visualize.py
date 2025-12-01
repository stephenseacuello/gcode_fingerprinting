"""
Post-training visualization utilities.

Provides functions to load and visualize training metrics from
completed training runs.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(run_dir: Path | str, format: str = 'json') -> List[Dict]:
    """
    Load training metrics from a run directory.

    Args:
        run_dir: Path to training run directory
        format: 'json' or 'csv'

    Returns:
        List of metric dictionaries, one per epoch
    """
    run_dir = Path(run_dir)

    if format == 'json':
        metrics_file = run_dir / 'metrics.json'
        with open(metrics_file, 'r') as f:
            return json.load(f)

    elif format == 'csv':
        metrics_file = run_dir / 'metrics.csv'
        with open(metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            return [
                {k: float(v) if k != 'epoch' else int(v)
                 for k, v in row.items()}
                for row in reader
            ]

    else:
        raise ValueError(f"Unknown format: {format}")


def plot_training_summary(
    run_dir: Path | str,
    output_path: Optional[Path | str] = None,
    figsize: tuple = (15, 10)
):
    """
    Create comprehensive training summary plot.

    Args:
        run_dir: Path to training run directory
        output_path: Where to save plot (default: run_dir/plots/summary.png)
        figsize: Figure size (width, height)
    """
    run_dir = Path(run_dir)
    metrics = load_metrics(run_dir)

    if not metrics:
        print("No metrics found")
        return

    epochs = [m['epoch'] for m in metrics]

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Training Summary: {run_dir.name}', fontsize=16, fontweight='bold')

    # 1. Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, [m['train_total'] for m in metrics],
            label='Train', marker='o', markersize=4, linewidth=2)
    ax.plot(epochs, [m['val_total'] for m in metrics],
            label='Val', marker='s', markersize=4, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Total Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 2. Reconstruction Loss
    ax = axes[0, 1]
    ax.plot(epochs, [m['train_recon'] for m in metrics],
            label='Train', marker='o', markersize=4, linewidth=2, color='tab:orange')
    ax.plot(epochs, [m['val_recon'] for m in metrics],
            label='Val', marker='s', markersize=4, linewidth=2, color='tab:red')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 3. Classification Loss
    ax = axes[1, 0]
    ax.plot(epochs, [m['train_cls'] for m in metrics],
            label='Train', marker='o', markersize=4, linewidth=2, color='tab:green')
    ax.plot(epochs, [m['val_cls'] for m in metrics],
            label='Val', marker='s', markersize=4, linewidth=2, color='tab:olive')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Classification Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 4. Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, [m['learning_rate'] for m in metrics],
            marker='o', markersize=4, linewidth=2, color='tab:purple')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    # Save plot
    if output_path is None:
        output_path = run_dir / 'plots' / 'training_summary.png'

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary plot to {output_path}")

    plt.close()


def compare_runs(
    run_dirs: List[Path | str],
    metric: str = 'val_total',
    output_path: Optional[Path | str] = None,
    figsize: tuple = (12, 6)
):
    """
    Compare multiple training runs on a single plot.

    Args:
        run_dirs: List of run directory paths
        metric: Metric to compare (e.g., 'val_total', 'train_cls')
        output_path: Where to save plot
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        metrics = load_metrics(run_dir)

        if not metrics:
            print(f"⚠️  No metrics found in {run_dir}")
            continue

        epochs = [m['epoch'] for m in metrics]
        values = [m[metric] for m in metrics]

        ax.plot(epochs, values, marker='o', markersize=3,
                label=run_dir.name, linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Comparison: {metric}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {output_path}")
    else:
        plt.show()

    plt.close()


def print_best_metrics(run_dir: Path | str):
    """
    Print best metrics achieved during training.

    Args:
        run_dir: Path to training run directory
    """
    run_dir = Path(run_dir)
    metrics = load_metrics(run_dir)

    if not metrics:
        print("No metrics found")
        return

    # Find best epoch for each metric
    best_train = min(metrics, key=lambda m: m['train_total'])
    best_val = min(metrics, key=lambda m: m['val_total'])

    print(f"\n{'='*60}")
    print(f"Best Metrics: {run_dir.name}")
    print(f"{'='*60}")
    print(f"\nBest Training Loss:")
    print(f"  Epoch {best_train['epoch']}: {best_train['train_total']:.4f}")
    print(f"    Recon: {best_train['train_recon']:.4f}")
    print(f"    Cls:   {best_train['train_cls']:.4f}")

    print(f"\nBest Validation Loss:")
    print(f"  Epoch {best_val['epoch']}: {best_val['val_total']:.4f}")
    print(f"    Recon: {best_val['val_recon']:.4f}")
    print(f"    Cls:   {best_val['val_cls']:.4f}")

    print(f"\nFinal Metrics (Epoch {metrics[-1]['epoch']}):")
    print(f"  Train: {metrics[-1]['train_total']:.4f}")
    print(f"  Val:   {metrics[-1]['val_total']:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    """
    Example usage for standalone visualization.

    Usage:
        python src/miracle/training/visualize.py outputs/training/run_20250116_073000
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualize.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        sys.exit(1)

    print(f"Analyzing: {run_dir}")
    print_best_metrics(run_dir)
    plot_training_summary(run_dir)
