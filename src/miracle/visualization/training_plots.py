"""
Training visualization utilities.

Provides functions for plotting training metrics, learning curves, and loss curves.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union


def plot_learning_curves(
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    output_path: Union[str, Path],
    title: str = "Learning Curves",
    figsize: tuple = (12, 8)
):
    """
    Plot training and validation metrics over epochs.

    Args:
        train_metrics: Dictionary of training metrics {metric_name: [values]}
        val_metrics: Dictionary of validation metrics {metric_name: [values]}
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size (width, height)
    """
    metrics = list(train_metrics.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=figsize)
    if n_metrics == 1:
        axes = np.array([axes]).flatten()
    else:
        axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        epochs = range(1, len(train_metrics[metric]) + 1)
        ax.plot(epochs, train_metrics[metric], label=f'Train {metric}', marker='o')
        ax.plot(epochs, val_metrics[metric], label=f'Val {metric}', marker='s')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Over Time')
        ax.legend()
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    output_path: Union[str, Path],
    title: str = "Training and Validation Loss",
    figsize: tuple = (10, 6)
):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training loss values per epoch
        val_losses: List of validation loss values per epoch
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size (width, height)
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(epochs, train_losses, label='Training Loss', marker='o', linewidth=2)
    ax.plot(epochs, val_losses, label='Validation Loss', marker='s', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Highlight best validation epoch
    best_epoch = np.argmin(val_losses) + 1
    best_loss = val_losses[best_epoch - 1]
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Val (epoch {best_epoch})')
    ax.scatter([best_epoch], [best_loss], color='red', s=100, zorder=5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric_comparison(
    metrics: Dict[str, float],
    output_path: Union[str, Path],
    title: str = "Model Performance Metrics",
    figsize: tuple = (10, 6),
    color: str = 'steelblue'
):
    """
    Plot bar chart comparing different metrics.

    Args:
        metrics: Dictionary of metric names and values
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size (width, height)
        color: Bar color
    """
    fig, ax = plt.subplots(figsize=figsize)

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    bars = ax.barh(metric_names, metric_values, color=color, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2,
                f'{width:.3f}',
                ha='left', va='center', fontweight='bold', fontsize=10)

    ax.set_xlabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
