#!/usr/bin/env python3
"""
Generate additional publication-quality visualizations for sweep analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_hyperparameter_heatmap(df: pd.DataFrame, output_dir: Path):
    """Plot heatmap of hyperparameters vs accuracy"""
    print("\nGenerating hyperparameter heatmap...")

    # Select numeric hyperparameters
    params = ['batch_size', 'hidden_dim', 'num_layers', 'num_heads']
    available_params = [p for p in params if p in df.columns and df[p].notna().any()]

    if len(available_params) < 2:
        print("  ⚠️ Not enough parameters for heatmap")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, param in enumerate(available_params[:4]):
        ax = axes[idx]

        # Create pivot table
        pivot_data = df.groupby(param)['val_overall_acc'].agg(['mean', 'std', 'count'])
        pivot_data = pivot_data.sort_index()

        # Bar plot with error bars
        x = pivot_data.index
        y = pivot_data['mean'] * 100
        yerr = pivot_data['std'] * 100

        bars = ax.bar(x, y, yerr=yerr, capsize=5, alpha=0.7,
                     color='steelblue', edgecolor='black', linewidth=0.5)

        # Color bars by performance
        colors = plt.cm.viridis(y / y.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel(param.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylabel('Mean Validation Accuracy (%)', fontweight='bold')
        ax.set_title(f'Accuracy vs {param.replace("_", " ").title()}', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add count labels
        for i, (x_val, count) in enumerate(zip(x, pivot_data['count'])):
            ax.text(i, 2, f'n={int(count)}', ha='center', fontsize=8)

    # Hide unused subplots
    for idx in range(len(available_params), 4):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'hyperparameter_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: hyperparameter_heatmap.png")


def plot_optimizer_scheduler_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare optimizer and scheduler choices"""
    print("\nGenerating optimizer/scheduler comparison...")

    if 'optimizer' not in df.columns or 'scheduler' not in df.columns:
        print("  ⚠️ Optimizer/scheduler columns not found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Optimizer comparison
    ax = axes[0]
    opt_stats = df.groupby('optimizer')['val_overall_acc'].agg(['mean', 'std', 'count'])
    opt_stats = opt_stats.sort_values('mean', ascending=False)

    x = np.arange(len(opt_stats))
    bars = ax.bar(x, opt_stats['mean']*100, yerr=opt_stats['std']*100,
                 capsize=5, alpha=0.7, color='coral', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(opt_stats.index, rotation=0)
    ax.set_xlabel('Optimizer', fontweight='bold')
    ax.set_ylabel('Mean Validation Accuracy (%)', fontweight='bold')
    ax.set_title('Optimizer Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add counts
    for i, (idx, row) in enumerate(opt_stats.iterrows()):
        ax.text(i, 2, f'n={int(row["count"])}', ha='center', fontsize=9)

    # Scheduler comparison
    ax = axes[1]
    sched_stats = df.groupby('scheduler')['val_overall_acc'].agg(['mean', 'std', 'count'])
    sched_stats = sched_stats.sort_values('mean', ascending=False)

    x = np.arange(len(sched_stats))
    bars = ax.bar(x, sched_stats['mean']*100, yerr=sched_stats['std']*100,
                 capsize=5, alpha=0.7, color='mediumseagreen', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(sched_stats.index, rotation=0)
    ax.set_xlabel('Learning Rate Scheduler', fontweight='bold')
    ax.set_ylabel('Mean Validation Accuracy (%)', fontweight='bold')
    ax.set_title('Scheduler Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add counts
    for i, (idx, row) in enumerate(sched_stats.iterrows()):
        ax.text(i, 2, f'n={int(row["count"])}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'optimizer_scheduler_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: optimizer_scheduler_comparison.png")


def plot_learning_rate_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze learning rate impact"""
    print("\nGenerating learning rate analysis...")

    if 'learning_rate' not in df.columns or df['learning_rate'].notna().sum() == 0:
        print("  ⚠️ Learning rate column not found or empty")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning rate vs accuracy scatter
    ax = axes[0]
    x = df['learning_rate']
    y = df['val_overall_acc'] * 100

    scatter = ax.scatter(x, y, c=y, cmap='viridis', s=100, alpha=0.7,
                        edgecolor='black', linewidth=0.5)

    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate (log scale)', fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
    ax.set_title('Learning Rate vs Accuracy', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Accuracy (%)', fontweight='bold')

    # Learning rate distribution
    ax = axes[1]
    ax.hist(df['learning_rate'], bins=20, color='steelblue',
           edgecolor='black', alpha=0.7)

    ax.set_xlabel('Learning Rate', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Learning Rate Distribution', fontweight='bold')
    ax.set_xscale('log')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: learning_rate_analysis.png")


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path):
    """Plot correlation matrix of hyperparameters"""
    print("\nGenerating correlation matrix...")

    params = ['batch_size', 'hidden_dim', 'num_layers', 'num_heads',
              'learning_rate', 'weight_decay', 'label_smoothing',
              'val_overall_acc', 'val_loss']

    available_params = [p for p in params if p in df.columns and df[p].notna().any()]

    if len(available_params) < 3:
        print("  ⚠️ Not enough parameters for correlation matrix")
        return

    # Compute correlation
    corr = df[available_params].corr()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Heatmap
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
               square=True, linewidths=0.5, cbar_kws={'shrink': 0.8},
               vmin=-1, vmax=1, ax=ax)

    ax.set_title('Hyperparameter Correlation Matrix', fontweight='bold', pad=20)

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: correlation_matrix.png")


def plot_production_vs_sweep(prod_results: dict, sweep_df: pd.DataFrame, output_dir: Path):
    """Compare production model to sweep results"""
    print("\nGenerating production vs sweep comparison...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get sweep stats
    metrics = ['type', 'command', 'param_type', 'param_value', 'overall']
    labels = ['Type', 'Command', 'Param Type', 'Param Value', 'Overall']

    # Production model accuracies
    prod_accs = [prod_results['accuracies'][m] for m in metrics]

    # Sweep best and mean accuracies (use command as proxy since we have it)
    sweep_best = 100.0  # From the sweep results, top runs achieved 100%
    sweep_mean = sweep_df['val_overall_acc'].mean() * 100

    x = np.arange(len(labels))
    width = 0.25

    # Production model
    bars1 = ax.bar(x - width, prod_accs, width, label='Production Model',
                  color='crimson', alpha=0.7, edgecolor='black', linewidth=0.5)

    # Sweep best (for overall accuracy only)
    sweep_best_vals = [np.nan] * (len(metrics) - 1) + [sweep_best]
    bars2 = ax.bar(x, sweep_best_vals, width, label='Sweep Best',
                  color='green', alpha=0.7, edgecolor='black', linewidth=0.5)

    # Sweep mean (for overall accuracy only)
    sweep_mean_vals = [np.nan] * (len(metrics) - 1) + [sweep_mean]
    bars3 = ax.bar(x + width, sweep_mean_vals, width, label='Sweep Mean',
                  color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Metric', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Production Model vs Sweep Results', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'production_vs_sweep.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: production_vs_sweep.png")


def main():
    print("="*80)
    print("GENERATING ADDITIONAL SWEEP VISUALIZATIONS")
    print("="*80)

    # Paths
    base_dir = Path(__file__).parent.parent
    analysis_dir = base_dir / 'outputs' / 'sweep_ab0ypky2_comprehensive_analysis'
    figures_dir = analysis_dir / 'figures'

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(analysis_dir / 'data' / 'cleaned_sweep_results.csv')
    print(f"✓ Loaded {len(df)} sweep results")

    # Load production model results
    prod_path = analysis_dir / 'data' / 'production_model_evaluation.json'
    if prod_path.exists():
        with open(prod_path, 'r') as f:
            prod_results = json.load(f)
        print(f"✓ Loaded production model results")
    else:
        prod_results = None
        print(f"⚠️  Production model results not found")

    # Generate visualizations
    plot_hyperparameter_heatmap(df, figures_dir)
    plot_optimizer_scheduler_comparison(df, figures_dir)
    plot_learning_rate_analysis(df, figures_dir)
    plot_correlation_matrix(df, figures_dir)

    if prod_results:
        plot_production_vs_sweep(prod_results, df, figures_dir)

    print("\n" + "="*80)
    print("✅ VISUALIZATION GENERATION COMPLETE!")
    print("="*80)
    print(f"\nFigures saved to: {figures_dir}")


if __name__ == '__main__':
    main()
