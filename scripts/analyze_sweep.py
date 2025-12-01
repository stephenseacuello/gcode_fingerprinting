#!/usr/bin/env python3
"""
Analyze and visualize W&B sweep results.

Usage:
    python scripts/analyze_sweep.py --sweep-id YOUR_SWEEP_ID --output outputs/sweep_analysis/
"""

import argparse
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def fetch_sweep_results(sweep_id: str, entity: str = None, project: str = 'gcode-fingerprinting'):
    """Fetch all runs from a W&B sweep."""
    try:
        api = wandb.Api()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize W&B API: {e}\nMake sure you're logged in: wandb login")

    if entity:
        sweep_path = f"{entity}/{project}/{sweep_id}"
    else:
        sweep_path = f"{project}/{sweep_id}"

    try:
        sweep = api.sweep(sweep_path)
    except wandb.errors.CommError as e:
        raise ValueError(f"Sweep not found: {sweep_path}\n"
                        f"Please check:\n"
                        f"  - Sweep ID is correct: {sweep_id}\n"
                        f"  - Entity is correct: {entity}\n"
                        f"  - Project is correct: {project}\n"
                        f"Original error: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch sweep: {e}")

    runs = []
    for run in sweep.runs:
        try:
            # Get config and summary
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            summary = {k: v for k, v in run.summary.items() if not k.startswith('_')}

            runs.append({
                'run_id': run.id,
                'name': run.name,
                'state': run.state,
                **config,
                **summary
            })
        except Exception as e:
            print(f"Warning: Failed to process run {run.id}: {e}")
            continue

    if len(runs) == 0:
        raise ValueError(f"No runs found for sweep: {sweep_id}\n"
                        f"The sweep exists but has no runs yet. Wait for at least one run to complete.")

    return pd.DataFrame(runs)

def plot_parameter_importance(df: pd.DataFrame, metric: str, output_dir: Path):
    """Plot parameter importance using correlation."""
    # Get hyperparameter columns (W&B uses underscores)
    param_cols = ['learning_rate', 'batch_size', 'hidden_dim', 'num_heads',
                  'num_layers', 'weight_decay', 'grad_clip', 'command_weight']
    param_cols = [c for c in param_cols if c in df.columns]

    # Calculate correlations
    correlations = df[param_cols + [metric]].corr()[metric].drop(metric)
    correlations = correlations.abs().sort_values(ascending=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    correlations.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Absolute Correlation with ' + metric)
    ax.set_title('Hyperparameter Importance')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    return correlations

def plot_parallel_coordinates(df: pd.DataFrame, metric: str, output_dir: Path, top_n: int = 20):
    """Plot parallel coordinates for top runs."""
    # Get top runs
    df_sorted = df.sort_values(metric, ascending=False).head(top_n)

    # Select columns (W&B uses underscores)
    param_cols = ['learning_rate', 'batch_size', 'hidden_dim', 'num_heads',
                  'num_layers', 'weight_decay']
    param_cols = [c for c in param_cols if c in df_sorted.columns]

    # Normalize parameters
    df_norm = df_sorted[param_cols + [metric]].copy()
    for col in param_cols:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min() + 1e-8)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, row in df_norm.iterrows():
        alpha = 0.3 + 0.7 * (row[metric] - df_sorted[metric].min()) / (df_sorted[metric].max() - df_sorted[metric].min() + 1e-8)
        ax.plot(range(len(param_cols)), row[param_cols], alpha=alpha, linewidth=2)

    ax.set_xticks(range(len(param_cols)))
    ax.set_xticklabels(param_cols, rotation=45, ha='right')
    ax.set_ylabel('Normalized Value')
    ax.set_title(f'Parallel Coordinates Plot (Top {top_n} Runs)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'parallel_coordinates.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metric_distributions(df: pd.DataFrame, output_dir: Path):
    """Plot distributions of key metrics."""
    metrics = ['val/overall_acc', 'val/command_acc', 'val/param_type_acc', 'val/param_value_acc']
    metrics = [m for m in metrics if m in df.columns]

    if not metrics:
        print("Warning: No metrics found for distribution plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        if idx < len(axes):
            ax = axes[idx]
            ax.hist(df[metric].dropna(), bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel(metric)
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution: {metric}')
            ax.grid(axis='y', alpha=0.3)

            # Add mean line
            mean_val = df[metric].mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'metric_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_curves(df: pd.DataFrame, metric: str, output_dir: Path, top_n: int = 5):
    """Plot learning curves for top runs."""
    # This would require fetching history from W&B API
    # For now, just show best vs worst comparison
    df_sorted = df.sort_values(metric, ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot best vs worst
    best_run = df_sorted.iloc[0]
    worst_run = df_sorted.iloc[-1]

    ax.bar(['Best Run', 'Worst Run'], [best_run[metric], worst_run[metric]],
           color=['green', 'red'], alpha=0.7)
    ax.set_ylabel(metric)
    ax.set_title(f'Best vs Worst Run ({metric})')
    ax.grid(axis='y', alpha=0.3)

    # Add values on bars
    for i, (label, val) in enumerate([(best_run[metric], 'Best'), (worst_run[metric], 'Worst')]):
        ax.text(i, label, f'{label:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'best_vs_worst.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(df: pd.DataFrame, metric: str, output_dir: Path):
    """Generate a summary report."""
    report = []
    report.append("=" * 80)
    report.append("SWEEP SUMMARY REPORT")
    report.append("=" * 80)
    report.append(f"\nTotal Runs: {len(df)}")
    report.append(f"Completed Runs: {len(df[df['state'] == 'finished'])}")
    report.append(f"Failed Runs: {len(df[df['state'] == 'failed'])}")

    report.append(f"\n{metric} Statistics:")
    report.append(f"  Mean: {df[metric].mean():.4f}")
    report.append(f"  Std:  {df[metric].std():.4f}")
    report.append(f"  Min:  {df[metric].min():.4f}")
    report.append(f"  Max:  {df[metric].max():.4f}")

    # Best run
    best_run = df.loc[df[metric].idxmax()]
    report.append(f"\nBest Run: {best_run['name']}")
    report.append(f"  {metric}: {best_run[metric]:.4f}")
    report.append(f"  Config:")
    for key in ['learning_rate', 'batch_size', 'hidden_dim', 'num_heads', 'num_layers']:
        if key in best_run:
            report.append(f"    {key}: {best_run[key]}")

    report.append("\n" + "=" * 80)

    # Save report
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write('\n'.join(report))

    print('\n'.join(report))

def main():
    parser = argparse.ArgumentParser(description='Analyze W&B sweep results')
    parser.add_argument('--sweep-id', type=str, required=True, help='W&B sweep ID')
    parser.add_argument('--entity', type=str, default=None, help='W&B entity/username')
    parser.add_argument('--project', type=str, default='gcode-fingerprinting', help='W&B project name')
    parser.add_argument('--metric', type=str, default='val/overall_acc', help='Metric to optimize')
    parser.add_argument('--output', type=str, default='outputs/sweep_analysis', help='Output directory')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Fetching sweep results for: {args.sweep_id}")
        df = fetch_sweep_results(args.sweep_id, args.entity, args.project)

        print(f"Found {len(df)} runs")

        # Save raw data
        df.to_csv(output_dir / 'sweep_results.csv', index=False)
        print(f"✓ Saved: {output_dir / 'sweep_results.csv'}")

        # Generate visualizations
        print("\nGenerating visualizations...")

        if args.metric not in df.columns:
            print(f"❌ Error: Metric '{args.metric}' not found in results")
            print(f"Available metrics: {[col for col in df.columns if 'val/' in col or 'train/' in col]}")
            return

        # Filter completed runs
        df_complete = df[df['state'] == 'finished'].copy()

        if len(df_complete) == 0:
            print("⚠️ Warning: No completed runs found")
            print(f"Run states: {df['state'].value_counts().to_dict()}")
            print("Wait for at least one run to complete before generating visualizations.")
            return

        print(f"Analyzing {len(df_complete)} completed runs...")

        # Try each visualization, continuing on failure
        try:
            plot_parameter_importance(df_complete, args.metric, output_dir)
            print(f"✓ Saved: {output_dir / 'parameter_importance.png'}")
        except Exception as e:
            print(f"⚠️ Failed to generate parameter_importance.png: {e}")

        try:
            plot_parallel_coordinates(df_complete, args.metric, output_dir)
            print(f"✓ Saved: {output_dir / 'parallel_coordinates.png'}")
        except Exception as e:
            print(f"⚠️ Failed to generate parallel_coordinates.png: {e}")

        try:
            plot_metric_distributions(df_complete, output_dir)
            print(f"✓ Saved: {output_dir / 'metric_distributions.png'}")
        except Exception as e:
            print(f"⚠️ Failed to generate metric_distributions.png: {e}")

        try:
            plot_learning_curves(df_complete, args.metric, output_dir)
            print(f"✓ Saved: {output_dir / 'best_vs_worst.png'}")
        except Exception as e:
            print(f"⚠️ Failed to generate best_vs_worst.png: {e}")

        try:
            generate_summary_report(df_complete, args.metric, output_dir)
            print(f"✓ Saved: {output_dir / 'summary_report.txt'}")
        except Exception as e:
            print(f"⚠️ Failed to generate summary_report.txt: {e}")

        print(f"\n✅ Analysis complete! Results saved to: {output_dir}")

    except (ValueError, RuntimeError) as e:
        print(f"\n❌ Error: {e}")
        return
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == '__main__':
    main()
