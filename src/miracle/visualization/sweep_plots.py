"""
Hyperparameter sweep visualization utilities.

Provides functions for analyzing and visualizing W&B sweep results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List


def plot_parameter_importance(
    df: pd.DataFrame,
    metric: str,
    output_path: Union[str, Path],
    param_cols: List[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot parameter importance using correlation analysis.

    Args:
        df: DataFrame with hyperparameters and metrics
        metric: Target metric column name
        output_path: Path to save the figure
        param_cols: List of parameter column names (auto-detected if None)
        figsize: Figure size (width, height)
    """
    if param_cols is None:
        # Auto-detect hyperparameter columns
        param_cols = ['learning-rate', 'batch-size', 'hidden-dim', 'num-heads',
                      'num-layers', 'weight-decay', 'grad-clip', 'command-weight']
        param_cols = [c for c in param_cols if c in df.columns]

    # Calculate correlations
    correlations = df[param_cols + [metric]].corr()[metric].drop(metric)
    correlations = correlations.abs().sort_values(ascending=True)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    correlations.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel(f'Absolute Correlation with {metric}', fontsize=12)
    ax.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add correlation values on bars
    for i, (param, corr) in enumerate(correlations.items()):
        ax.text(corr, i, f'  {corr:.3f}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return correlations


def plot_parallel_coordinates(
    df: pd.DataFrame,
    metric: str,
    output_path: Union[str, Path],
    param_cols: List[str] = None,
    top_n: int = 20,
    figsize: tuple = (14, 6)
):
    """
    Plot parallel coordinates for top performing runs.

    Args:
        df: DataFrame with hyperparameters and metrics
        metric: Target metric column name
        output_path: Path to save the figure
        param_cols: List of parameter column names (auto-detected if None)
        top_n: Number of top runs to visualize
        figsize: Figure size (width, height)
    """
    if param_cols is None:
        param_cols = ['learning-rate', 'batch-size', 'hidden-dim', 'num-heads',
                      'num-layers', 'weight-decay']
        param_cols = [c for c in param_cols if c in df.columns]

    # Get top runs
    df_sorted = df.sort_values(metric, ascending=False).head(top_n)

    # Normalize parameters
    df_norm = df_sorted[param_cols + [metric]].copy()
    for col in param_cols:
        col_min = df_norm[col].min()
        col_max = df_norm[col].max()
        df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min + 1e-8)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate alpha based on metric performance
    metric_min = df_sorted[metric].min()
    metric_max = df_sorted[metric].max()

    for idx, row in df_norm.iterrows():
        # Higher performing runs get higher alpha
        alpha = 0.3 + 0.7 * (row[metric] - metric_min) / (metric_max - metric_min + 1e-8)
        ax.plot(range(len(param_cols)), row[param_cols], alpha=alpha, linewidth=2)

    ax.set_xticks(range(len(param_cols)))
    ax.set_xticklabels(param_cols, rotation=45, ha='right')
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title(f'Parallel Coordinates Plot (Top {top_n} Runs)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric_distributions(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    metrics: List[str] = None,
    figsize: tuple = (12, 10)
):
    """
    Plot distributions of multiple metrics.

    Args:
        df: DataFrame with metrics
        output_path: Path to save the figure
        metrics: List of metric column names (auto-detected if None)
        figsize: Figure size (width, height)
    """
    if metrics is None:
        metrics = ['val/overall_acc', 'val/command_acc', 'val/param_type_acc', 'val/param_value_acc']
        metrics = [m for m in metrics if m in df.columns]

    if not metrics:
        raise ValueError("No metrics found for distribution plot")

    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for idx, metric in enumerate(metrics):
        if idx < len(axes):
            ax = axes[idx]
            metric_data = df[metric].dropna()

            # Plot histogram
            ax.hist(metric_data, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel(metric, fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f'Distribution: {metric}', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # Add mean line
            mean_val = metric_data.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.3f}')
            ax.legend()

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
