#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for G-code Fingerprinting Paper.

This script generates:
1. Training curves (loss, accuracy vs epoch)
2. Confusion matrices (per-operation)
3. Ablation bar charts
4. Model comparison plots
5. Attention visualization heatmaps
6. Error analysis plots
7. Multi-seed box plots with confidence intervals

Usage:
    python scripts/generate_paper_figures.py --output-dir figures/
    python scripts/generate_paper_figures.py --results outputs/multiseed --style paper

Author: Claude Code
Date: December 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np

# Plotting imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# Try seaborn for prettier plots
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("seaborn not found, using matplotlib defaults")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def setup_style(style: str = 'paper'):
    """Configure matplotlib style for publication."""

    if style == 'paper':
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 10,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.figsize': (6, 4),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
        if HAS_SEABORN:
            sns.set_palette('colorblind')
    elif style == 'presentation':
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'figure.figsize': (10, 6),
            'figure.dpi': 150,
        })
        if HAS_SEABORN:
            sns.set_style('whitegrid')


def load_training_history(results_dir: str) -> Dict:
    """Load training history from results directory."""
    results_dir = Path(results_dir)

    # Try different possible locations
    for filename in ['training_history.json', 'history.json', 'results.json']:
        path = results_dir / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)

    return {}


def plot_training_curves(
    history: Dict,
    output_path: str,
    title: str = "Training Progress"
):
    """Plot training and validation curves."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Loss curves
    ax1 = axes[0]
    if 'train_loss' in history:
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
    if 'val_loss' in history:
        epochs = range(1, len(history['val_loss']) + 1)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Accuracy curves
    ax2 = axes[1]
    if 'train_token_acc' in history:
        epochs = range(1, len(history['train_token_acc']) + 1)
        ax2.plot(epochs, [a * 100 for a in history['train_token_acc']],
                 'b-', label='Train Acc', linewidth=1.5)
    if 'val_token_acc' in history:
        epochs = range(1, len(history['val_token_acc']) + 1)
        ax2.plot(epochs, [a * 100 for a in history['val_token_acc']],
                 'r-', label='Val Acc', linewidth=1.5)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Token Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True
):
    """Plot confusion matrix heatmap."""

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(8, 6))

    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                    cmap='Blues', xticklabels=class_names,
                    yticklabels=class_names, ax=ax)
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]:.2f}' if normalize else f'{cm[i, j]}',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_ablation_bars(
    ablation_results: Dict,
    output_path: str,
    title: str = "Ablation Study Results"
):
    """Plot ablation study as grouped bar chart."""

    ablations = []
    baseline_accs = []
    ablated_accs = []

    for name, data in ablation_results.items():
        if isinstance(data, dict) and 'baseline' in data:
            ablations.append(name.replace('_', '\n'))
            baseline_accs.append(data['baseline'])
            ablated_accs.append(data['ablated'])

    if not ablations:
        print("No ablation data to plot")
        return

    x = np.arange(len(ablations))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, baseline_accs, width, label='With Component', color='#2ecc71')
    bars2 = ax.bar(x + width/2, ablated_accs, width, label='Without Component', color='#e74c3c')

    ax.set_xlabel('Component')
    ax.set_ylabel('Token Accuracy (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(ablations)
    ax.legend()

    # Add value labels
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_comparison(
    results: Dict[str, Dict],
    output_path: str,
    metric: str = 'token_accuracy',
    title: str = "Model Comparison"
):
    """Plot model comparison bar chart with error bars."""

    models = []
    means = []
    errors = []

    for name, data in results.items():
        if name == 'metadata':
            continue

        if isinstance(data, dict):
            if metric in data:
                metric_data = data[metric]
                if isinstance(metric_data, dict):
                    mean = metric_data.get('mean', 0) * 100
                    ci = metric_data.get('ci_95', 0) * 100
                else:
                    mean = metric_data * 100
                    ci = 0
            else:
                mean = data.get(f'val_{metric}', data.get(metric, 0)) * 100
                ci = 0

            models.append(name)
            means.append(mean)
            errors.append(ci)

    if not models:
        print("No model data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(models))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))

    bars = ax.bar(x, means, yerr=errors, capsize=5, color=colors, edgecolor='black')

    ax.set_xlabel('Model')
    ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')

    # Add value labels
    for bar, mean, err in zip(bars, means, errors):
        ax.annotate(f'{mean:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_multiseed_boxplot(
    results: Dict,
    output_path: str,
    title: str = "Multi-Seed Results"
):
    """Plot box plot of multi-seed results."""

    fig, ax = plt.subplots(figsize=(6, 5))

    data = []
    labels = []

    if 'token_accuracy' in results and 'values' in results['token_accuracy']:
        data.append([v * 100 for v in results['token_accuracy']['values']])
        labels.append('Token Acc')

    if 'sequence_accuracy' in results and 'values' in results['sequence_accuracy']:
        data.append([v * 100 for v in results['sequence_accuracy']['values']])
        labels.append('Seq Acc')

    if not data:
        print("No multi-seed data to plot")
        return

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    colors = ['#3498db', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)

    # Add individual points
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.6, color='black', s=30, zorder=5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_learning_rate_schedule(
    n_epochs: int,
    warmup_epochs: int = 5,
    base_lr: float = 0.0002,
    output_path: str = "lr_schedule.pdf"
):
    """Plot learning rate schedule visualization."""

    epochs = np.arange(1, n_epochs + 1)

    # Warmup + cosine decay
    lrs = []
    for e in epochs:
        if e <= warmup_epochs:
            lr = base_lr * e / warmup_epochs
        else:
            progress = (e - warmup_epochs) / (n_epochs - warmup_epochs)
            lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        lrs.append(lr)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, lrs, 'b-', linewidth=2)
    ax.axvline(x=warmup_epochs, color='r', linestyle='--', alpha=0.7, label='Warmup End')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (Warmup + Cosine Decay)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_operation_accuracy(
    results: Dict,
    output_path: str,
    title: str = "Per-Operation Accuracy"
):
    """Plot accuracy breakdown by operation type."""

    operations = []
    accuracies = []

    # Try to extract per-operation data
    op_results = results.get('per_operation', results.get('classification_report', {}))

    for op_name, op_data in op_results.items():
        if isinstance(op_data, dict) and 'precision' in op_data:
            operations.append(op_name)
            # Use f1-score as the main metric
            accuracies.append(op_data.get('f1-score', op_data.get('precision', 0)) * 100)

    if not operations:
        print("No per-operation data to plot")
        return

    # Sort by accuracy
    sorted_pairs = sorted(zip(accuracies, operations), reverse=True)
    accuracies, operations = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(8, 6))

    y = np.arange(len(operations))
    colors = plt.cm.RdYlGn(np.array(accuracies) / 100)

    bars = ax.barh(y, accuracies, color=colors)

    ax.set_yticks(y)
    ax.set_yticklabels(operations)
    ax.set_xlabel('F1-Score (%)')
    ax.set_title(title)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_figures(args):
    """Generate all paper figures."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style(args.style)

    # Load results
    if args.results_dir:
        results_dir = Path(args.results_dir)

        # Training history
        history = load_training_history(results_dir)
        if history:
            plot_training_curves(
                history,
                str(output_dir / 'training_curves.pdf'),
                title="Training Progress"
            )

        # Multi-seed results
        stats_path = results_dir / 'statistics.json'
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            plot_multiseed_boxplot(
                stats,
                str(output_dir / 'multiseed_boxplot.pdf'),
                title="Multi-Seed Results Distribution"
            )

    # Ablation results
    if args.ablations_dir:
        ablation_path = Path(args.ablations_dir) / 'ablation_summary.json'
        if ablation_path.exists():
            with open(ablation_path) as f:
                ablation_data = json.load(f)
            plot_ablation_bars(
                ablation_data,
                str(output_dir / 'ablation_study.pdf'),
                title="Ablation Study: Component Contributions"
            )

    # Baseline comparison
    if args.baseline_dir:
        baseline_path = Path(args.baseline_dir) / 'baseline_summary.json'
        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline_data = json.load(f)
            plot_model_comparison(
                baseline_data,
                str(output_dir / 'baseline_comparison.pdf'),
                title="Baseline Model Comparison"
            )

    # Learning rate schedule
    plot_learning_rate_schedule(
        n_epochs=100,
        warmup_epochs=5,
        base_lr=0.0002,
        output_path=str(output_dir / 'lr_schedule.pdf')
    )

    print(f"\n{'='*60}")
    print(f"All figures saved to: {output_dir}")
    print('='*60)


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--output-dir', type=str, default='figures',
                        help='Output directory for figures')
    parser.add_argument('--results-dir', type=str,
                        help='Directory with training results')
    parser.add_argument('--ablations-dir', type=str,
                        help='Directory with ablation results')
    parser.add_argument('--baseline-dir', type=str,
                        help='Directory with baseline results')
    parser.add_argument('--style', choices=['paper', 'presentation'],
                        default='paper', help='Figure style')
    parser.add_argument('--format', choices=['pdf', 'png', 'svg'],
                        default='pdf', help='Figure format')
    args = parser.parse_args()

    generate_all_figures(args)


if __name__ == '__main__':
    main()
