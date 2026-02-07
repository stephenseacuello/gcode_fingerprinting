#!/usr/bin/env python3
"""Generate Publication-Quality Figures for Encoder Ablation Paper.

This script reads ANOVA analysis results and generates figures suitable for
publication in IEEE/ACM venues.

Usage:
    python scripts/analysis/generate_paper_figures.py \
        --results-dir outputs/ablation_study_2026_02_06/analysis \
        --output-dir outputs/ablation_study_2026_02_06/paper/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

# Publication-quality settings
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.05

# Color palette for consistency
COLORS = {
    'baseline': '#2ecc71',
    'loo': '#e74c3c',
    'loi': '#3498db',
    'architecture': '#9b59b6',
    'ml_baseline': '#f39c12',
    'nn_baseline': '#1abc9c',
}


def load_results(results_dir: Path) -> dict:
    """Load all analysis results."""
    results = {}

    # Load main results CSV
    csv_path = results_dir / 'all_results.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)

        # Parse config_name to extract ablation_type, removed, included, model
        def parse_config(row):
            config = row.get('config_name', '')
            if pd.isna(config):
                config = ''
            config = str(config)

            if config.startswith('loo_'):
                return pd.Series({
                    'ablation_type': 'loo',
                    'removed': config[4:],  # Remove 'loo_' prefix
                    'included': None,
                    'model': None
                })
            elif config.startswith('loi_'):
                return pd.Series({
                    'ablation_type': 'loi',
                    'removed': None,
                    'included': config[4:],  # Remove 'loi_' prefix
                    'model': None
                })
            elif config == 'baseline':
                return pd.Series({
                    'ablation_type': 'baseline',
                    'removed': None,
                    'included': None,
                    'model': None
                })
            elif row.get('study') == 'baseline':
                # For baseline study, config_name IS the model name
                return pd.Series({
                    'ablation_type': 'model',
                    'removed': None,
                    'included': None,
                    'model': config
                })
            else:
                # Architecture or other configs
                return pd.Series({
                    'ablation_type': 'config',
                    'removed': None,
                    'included': None,
                    'model': None
                })

        # Apply parsing
        parsed = df.apply(parse_config, axis=1)
        df = pd.concat([df, parsed], axis=1)
        results['df'] = df

    # Load ANOVA results
    anova_path = results_dir / 'anova_results.json'
    if anova_path.exists():
        with open(anova_path) as f:
            results['anova'] = json.load(f)

    # Load importance rankings
    importance_path = results_dir / 'importance_rankings.json'
    if importance_path.exists():
        with open(importance_path) as f:
            results['importance'] = json.load(f)

    return results


def plot_modality_loo_importance(df: pd.DataFrame, output_dir: Path):
    """Bar chart showing accuracy drop when each modality is removed."""
    # Filter to modality_grouped LOO experiments
    loo_df = df[(df['study'] == 'modality_grouped') & (df['ablation_type'] == 'loo')]

    if loo_df.empty:
        print("No modality LOO data found, skipping figure")
        return

    # Calculate mean and std per modality
    stats = loo_df.groupby('removed')['test_accuracy'].agg(['mean', 'std']).reset_index()

    # Get baseline accuracy
    baseline_df = df[(df['study'] == 'modality_grouped') & (df['ablation_type'] == 'baseline')]
    baseline_acc = baseline_df['test_accuracy'].mean() if not baseline_df.empty else 1.0

    # Calculate accuracy drop
    stats['drop'] = baseline_acc - stats['mean']
    stats = stats.sort_values('drop', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(stats))
    bars = ax.bar(x, stats['drop'] * 100, yerr=stats['std'] * 100,
                  capsize=3, color=COLORS['loo'], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Removed Modality')
    ax.set_ylabel('Accuracy Drop (%)')
    ax.set_title('Modality Importance: Leave-One-Out Accuracy Drop')
    ax.set_xticks(x)
    ax.set_xticklabels(stats['removed'], rotation=45, ha='right')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, stats['drop'] * 100):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_modality_loo_importance.pdf')
    plt.savefig(output_dir / 'fig_modality_loo_importance.png')
    plt.close()
    print(f"Saved: fig_modality_loo_importance.pdf/png")


def plot_modality_loi_performance(df: pd.DataFrame, output_dir: Path):
    """Bar chart showing single-modality performance."""
    # Filter to modality_grouped LOI experiments
    loi_df = df[(df['study'] == 'modality_grouped') & (df['ablation_type'] == 'loi')]

    if loi_df.empty:
        print("No modality LOI data found, skipping figure")
        return

    # Calculate mean and std per modality
    stats = loi_df.groupby('included')['test_accuracy'].agg(['mean', 'std']).reset_index()
    stats = stats.sort_values('mean', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(stats))
    bars = ax.bar(x, stats['mean'] * 100, yerr=stats['std'] * 100,
                  capsize=3, color=COLORS['loi'], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Single Modality')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Single Modality Performance (Leave-One-In)')
    ax.set_xticks(x)
    ax.set_xticklabels(stats['included'], rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.axhline(y=60, color='red', linestyle='--', linewidth=1, label='H5 threshold (60%)')
    ax.legend(loc='upper right')

    # Add value labels on bars
    for bar, val in zip(bars, stats['mean'] * 100):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_modality_loi_performance.pdf')
    plt.savefig(output_dir / 'fig_modality_loi_performance.png')
    plt.close()
    print(f"Saved: fig_modality_loi_performance.pdf/png")


def plot_sensor_loo_importance(df: pd.DataFrame, output_dir: Path):
    """Bar chart showing accuracy drop when each sensor is removed."""
    # Filter to sensor_individual LOO experiments
    loo_df = df[(df['study'] == 'sensor_individual') & (df['ablation_type'] == 'loo')]

    if loo_df.empty:
        print("No sensor LOO data found, skipping figure")
        return

    # Calculate mean and std per sensor
    stats = loo_df.groupby('removed')['test_accuracy'].agg(['mean', 'std']).reset_index()

    # Get baseline accuracy
    baseline_df = df[(df['study'] == 'sensor_individual') & (df['ablation_type'] == 'baseline')]
    baseline_acc = baseline_df['test_accuracy'].mean() if not baseline_df.empty else 1.0

    # Calculate accuracy drop
    stats['drop'] = baseline_acc - stats['mean']
    stats = stats.sort_values('drop', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))

    x = np.arange(len(stats))
    bars = ax.bar(x, stats['drop'] * 100, yerr=stats['std'] * 100,
                  capsize=2, color=COLORS['loo'], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Removed Sensor')
    ax.set_ylabel('Accuracy Drop (%)')
    ax.set_title('Sensor Importance: Leave-One-Out Accuracy Drop')
    ax.set_xticks(x)
    ax.set_xticklabels(stats['removed'], rotation=45, ha='right', fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(y=3, color='orange', linestyle='--', linewidth=1, label='H13 threshold (3%)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_sensor_loo_importance.pdf')
    plt.savefig(output_dir / 'fig_sensor_loo_importance.png')
    plt.close()
    print(f"Saved: fig_sensor_loo_importance.pdf/png")


def plot_sensor_group_comparison(df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart comparing sensor group LOO and LOI performance."""
    # Filter to sensor_group experiments
    group_df = df[df['study'] == 'sensor_group']

    if group_df.empty:
        print("No sensor group data found, skipping figure")
        return

    # Get baseline
    baseline_df = group_df[group_df['ablation_type'] == 'baseline']
    baseline_acc = baseline_df['test_accuracy'].mean() if not baseline_df.empty else 1.0

    # LOO stats
    loo_df = group_df[group_df['ablation_type'] == 'loo']
    loo_stats = loo_df.groupby('removed')['test_accuracy'].agg(['mean', 'std']).reset_index()
    loo_stats.columns = ['group', 'loo_mean', 'loo_std']
    loo_stats['loo_drop'] = baseline_acc - loo_stats['loo_mean']

    # LOI stats
    loi_df = group_df[group_df['ablation_type'] == 'loi']
    loi_stats = loi_df.groupby('included')['test_accuracy'].agg(['mean', 'std']).reset_index()
    loi_stats.columns = ['group', 'loi_mean', 'loi_std']

    # Merge
    stats = loo_stats.merge(loi_stats, on='group', how='outer')
    stats = stats.sort_values('loo_drop', ascending=False)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(len(stats))
    width = 0.6

    # LOO subplot
    axes[0].bar(x, stats['loo_drop'] * 100, yerr=stats['loo_std'] * 100,
               capsize=3, color=COLORS['loo'], alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Removed Sensor Group')
    axes[0].set_ylabel('Accuracy Drop (%)')
    axes[0].set_title('Sensor Group LOO (Importance)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stats['group'], rotation=45, ha='right')

    # LOI subplot
    axes[1].bar(x, stats['loi_mean'] * 100, yerr=stats['loi_std'] * 100,
               capsize=3, color=COLORS['loi'], alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Single Sensor Group')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Sensor Group LOI (Standalone)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stats['group'], rotation=45, ha='right')
    axes[1].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_sensor_group_comparison.pdf')
    plt.savefig(output_dir / 'fig_sensor_group_comparison.png')
    plt.close()
    print(f"Saved: fig_sensor_group_comparison.pdf/png")


def plot_architecture_comparison(df: pd.DataFrame, output_dir: Path):
    """Box plot comparing architecture ablation results."""
    # Filter to architecture experiments
    arch_df = df[df['study'] == 'architecture']

    if arch_df.empty:
        print("No architecture data found, skipping figure")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))

    # Create box plot
    configs = arch_df['config_name'].unique()
    data = [arch_df[arch_df['config_name'] == cfg]['test_accuracy'].values * 100 for cfg in configs]

    bp = ax.boxplot(data, labels=configs, patch_artist=True)

    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['architecture'])
        patch.set_alpha(0.7)

    ax.set_xlabel('Architecture Configuration')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Architecture Ablation Comparison')
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=100, color='green', linestyle='--', linewidth=1, label='Baseline (100%)')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_architecture_comparison.pdf')
    plt.savefig(output_dir / 'fig_architecture_comparison.png')
    plt.close()
    print(f"Saved: fig_architecture_comparison.pdf/png")


def plot_baseline_comparison(df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing all baseline models."""
    # Filter to baseline experiments
    baseline_df = df[df['study'] == 'baseline'].copy()

    if baseline_df.empty:
        print("No baseline data found, skipping figure")
        return

    # Use 'model' column if available, otherwise use 'config_name'
    model_col = 'model' if 'model' in baseline_df.columns and baseline_df['model'].notna().any() else 'config_name'

    # Calculate mean and std per model
    stats = baseline_df.groupby(model_col)['test_accuracy'].agg(['mean', 'std']).reset_index()
    stats = stats.rename(columns={model_col: 'model'})
    stats = stats.sort_values('mean', ascending=True)

    # Define model types
    ml_models = ['xgboost', 'random_forest', 'svm_rbf', 'logistic_regression', 'knn']
    nn_models = ['mlp', 'cnn_1d', 'lstm_simple', 'transformer', 'tcn']

    # Assign colors
    colors = []
    for model in stats['model']:
        if model.lower() in ml_models:
            colors.append(COLORS['ml_baseline'])
        elif model.lower() in nn_models:
            colors.append(COLORS['nn_baseline'])
        else:
            colors.append(COLORS['baseline'])

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    y = np.arange(len(stats))
    bars = ax.barh(y, stats['mean'] * 100, xerr=stats['std'] * 100,
                   capsize=3, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(stats['model'])
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('Baseline Model Comparison')
    ax.set_xlim(0, 105)
    ax.axvline(x=100, color='green', linestyle='--', linewidth=1, label='MM-DTAE-LSTM')

    # Add legend for model types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['ml_baseline'], label='Traditional ML'),
        Patch(facecolor=COLORS['nn_baseline'], label='Neural Network'),
        Patch(facecolor=COLORS['baseline'], label='MM-DTAE-LSTM'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Add value labels
    for bar, val in zip(bars, stats['mean'] * 100):
        width = bar.get_width()
        ax.annotate(f'{val:.1f}%', xy=(width, bar.get_y() + bar.get_height()/2),
                   xytext=(3, 0), textcoords='offset points', ha='left', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_baseline_comparison.pdf')
    plt.savefig(output_dir / 'fig_baseline_comparison.png')
    plt.close()
    print(f"Saved: fig_baseline_comparison.pdf/png")


def plot_cross_study_summary(df: pd.DataFrame, output_dir: Path):
    """Summary heatmap showing accuracy across all studies."""
    # Get summary statistics per study
    summary = df.groupby(['study', 'ablation_type'])['test_accuracy'].agg(['mean', 'std', 'count']).reset_index()

    if summary.empty:
        print("No data found for cross-study summary, skipping figure")
        return

    # Pivot for heatmap
    pivot = summary.pivot(index='study', columns='ablation_type', values='mean')

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Test Accuracy (%)'})

    ax.set_title('Cross-Study Accuracy Summary')
    ax.set_xlabel('Ablation Type')
    ax.set_ylabel('Study')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_cross_study_summary.pdf')
    plt.savefig(output_dir / 'fig_cross_study_summary.png')
    plt.close()
    print(f"Saved: fig_cross_study_summary.pdf/png")


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to analysis results directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for figures')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(results_dir)

    if 'df' not in results:
        print(f"ERROR: No results CSV found at {results_dir / 'all_results.csv'}")
        print("Run the ANOVA analysis first to generate results.")
        return

    df = results['df']
    print(f"Loaded {len(df)} experiment results")

    # Generate all figures
    print("\nGenerating figures...")
    plot_modality_loo_importance(df, output_dir)
    plot_modality_loi_performance(df, output_dir)
    plot_sensor_loo_importance(df, output_dir)
    plot_sensor_group_comparison(df, output_dir)
    plot_architecture_comparison(df, output_dir)
    plot_baseline_comparison(df, output_dir)
    plot_cross_study_summary(df, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
