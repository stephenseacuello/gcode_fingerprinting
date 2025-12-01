#!/usr/bin/env python3
"""
Comprehensive Sweep Analysis Script

This script performs a complete analysis of the hyperparameter sweep including:
1. Data extraction and consolidation
2. Parameter importance analysis
3. Configuration comparison
4. Production model evaluation
5. Visualization generation
6. Summary report creation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
import argparse
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

class SweepAnalyzer:
    """Comprehensive sweep analysis"""

    def __init__(self, base_dir: Path, output_dir: Path, sweep_id: str):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.sweep_id = sweep_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / 'data').mkdir(exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)

        self.df = None
        self.df_clean = None
        self.best_runs = None

    def load_sweep_data(self, csv_path: Path) -> pd.DataFrame:
        """Load and clean sweep data"""
        print("\n" + "="*80)
        print("PHASE 1: DATA EXTRACTION & CONSOLIDATION")
        print("="*80)

        # Load CSV
        print(f"\nLoading sweep data from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} total runs")

        # Filter out failed runs (inf loss or 0 accuracy)
        df_clean = df[
            (df['val_loss'] != np.inf) &
            (df['val_loss'].notna()) &
            (df['val_overall_acc'] > 0)
        ].copy()

        print(f"✓ Filtered to {len(df_clean)} successful runs")
        print(f"  Failed/incomplete runs: {len(df) - len(df_clean)}")

        # Statistics
        print(f"\n📊 Performance Statistics:")
        print(f"  G-Command Accuracy:")
        print(f"    Mean: {df_clean['val_g_command_acc'].mean()*100:.2f}%")
        print(f"    Max:  {df_clean['val_g_command_acc'].max()*100:.2f}%")
        print(f"    Min:  {df_clean['val_g_command_acc'].min()*100:.2f}%")

        print(f"\n  Overall Accuracy:")
        print(f"    Mean: {df_clean['val_overall_acc'].mean()*100:.2f}%")
        print(f"    Max:  {df_clean['val_overall_acc'].max()*100:.2f}%")
        print(f"    Min:  {df_clean['val_overall_acc'].min()*100:.2f}%")

        print(f"\n  Validation Loss:")
        print(f"    Mean: {df_clean['val_loss'].mean():.4f}")
        print(f"    Min:  {df_clean['val_loss'].min():.4f}")
        print(f"    Max:  {df_clean['val_loss'].max():.4f}")

        # Identify top runs
        self.best_runs = df_clean.nlargest(10, 'val_overall_acc')

        print(f"\n🏆 Top 10 Runs by Overall Accuracy:")
        for idx, row in self.best_runs.iterrows():
            print(f"  {row['run_name']}")
            print(f"    Overall: {row['val_overall_acc']*100:.2f}%, "
                  f"G-Cmd: {row['val_g_command_acc']*100:.2f}%, "
                  f"Loss: {row['val_loss']:.4f}")

        # Save cleaned data
        output_csv = self.output_dir / 'data' / 'cleaned_sweep_results.csv'
        df_clean.to_csv(output_csv, index=False)
        print(f"\n✓ Saved cleaned data to: {output_csv}")

        self.df = df
        self.df_clean = df_clean
        return df_clean

    def analyze_parameter_importance(self):
        """Analyze hyperparameter importance"""
        print("\n" + "="*80)
        print("PHASE 2: PARAMETER IMPORTANCE ANALYSIS")
        print("="*80)

        # Define hyperparameters
        params = ['batch_size', 'hidden_dim', 'num_layers', 'num_heads',
                  'learning_rate', 'weight_decay', 'label_smoothing']

        # Filter params that exist in data
        available_params = [p for p in params if p in self.df_clean.columns and self.df_clean[p].notna().any()]

        print(f"\nAnalyzing {len(available_params)} hyperparameters:")
        for p in available_params:
            print(f"  - {p}")

        # Calculate correlations
        metric = 'val_overall_acc'
        correlations = {}

        for param in available_params:
            # Skip if all NaN
            if self.df_clean[param].notna().sum() == 0:
                continue

            corr = self.df_clean[[param, metric]].dropna().corr().iloc[0, 1]
            correlations[param] = abs(corr)
            print(f"\n  {param}:")
            print(f"    Correlation: {corr:.4f}")
            print(f"    Range: {self.df_clean[param].min()} - {self.df_clean[param].max()}")

        # Sort by importance
        sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

        print(f"\n📊 Parameter Importance Ranking:")
        for i, (param, importance) in enumerate(sorted_params, 1):
            print(f"  {i}. {param}: {importance:.4f}")

        # Plot parameter importance
        self._plot_parameter_importance(sorted_params)

        # Plot parallel coordinates
        self._plot_parallel_coordinates(available_params, metric)

        # Save results
        importance_df = pd.DataFrame(sorted_params, columns=['parameter', 'importance'])
        importance_df.to_csv(self.output_dir / 'data' / 'parameter_importance.csv', index=False)

        return sorted_params

    def _plot_parameter_importance(self, sorted_params: List[Tuple[str, float]]):
        """Plot parameter importance bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))

        params = [p[0] for p in sorted_params]
        importance = [p[1] for p in sorted_params]

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(params)))
        bars = ax.barh(params, importance, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Absolute Correlation with Validation Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Hyperparameter Importance Analysis', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance)):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'parameter_importance.png',
                   bbox_inches='tight', dpi=300)
        plt.close()
        print(f"\n✓ Saved: parameter_importance.png")

    def _plot_parallel_coordinates(self, params: List[str], metric: str, top_n: int = 20):
        """Plot parallel coordinates for top runs"""
        fig, ax = plt.subplots(figsize=(14, 6))

        # Get top runs
        df_top = self.df_clean.nlargest(top_n, metric)

        # Normalize parameters
        df_norm = df_top[params + [metric]].copy()
        for col in params:
            if df_norm[col].notna().sum() > 0:
                col_min = df_norm[col].min()
                col_max = df_norm[col].max()
                if col_max > col_min:
                    df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
                else:
                    df_norm[col] = 0.5

        # Plot
        for idx, row in df_norm.iterrows():
            # Color by performance
            perf = row[metric]
            alpha = 0.3 + 0.7 * (perf - df_top[metric].min()) / (df_top[metric].max() - df_top[metric].min() + 1e-8)
            color = plt.cm.viridis(perf)

            ax.plot(range(len(params)), row[params], alpha=alpha, linewidth=2, color=color)

        ax.set_xticks(range(len(params)))
        ax.set_xticklabels(params, rotation=45, ha='right')
        ax.set_ylabel('Normalized Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Parallel Coordinates Plot (Top {top_n} Runs)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(-0.05, 1.05)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                   norm=plt.Normalize(vmin=df_top[metric].min(),
                                                     vmax=df_top[metric].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Validation Accuracy')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'parallel_coordinates.png',
                   bbox_inches='tight', dpi=300)
        plt.close()
        print(f"✓ Saved: parallel_coordinates.png")

    def compare_configurations(self):
        """Compare top configurations"""
        print("\n" + "="*80)
        print("PHASE 3: CONFIGURATION COMPARISON")
        print("="*80)

        # Get top 5 runs
        top5 = self.df_clean.nlargest(5, 'val_overall_acc')

        print(f"\n🏆 Top 5 Configurations:")

        comparison_data = []
        for i, (idx, row) in enumerate(top5.iterrows(), 1):
            print(f"\n  Rank {i}: {row['run_name']}")
            print(f"    Overall Acc: {row['val_overall_acc']*100:.2f}%")
            print(f"    G-Cmd Acc:   {row['val_g_command_acc']*100:.2f}%")
            print(f"    Val Loss:    {row['val_loss']:.4f}")
            print(f"    Batch Size:  {row.get('batch_size', 'N/A')}")
            print(f"    Hidden Dim:  {row.get('hidden_dim', 'N/A')}")
            print(f"    Num Layers:  {row.get('num_layers', 'N/A')}")
            print(f"    Num Heads:   {row.get('num_heads', 'N/A')}")
            print(f"    Learn Rate:  {row.get('learning_rate', 'N/A')}")
            print(f"    Optimizer:   {row.get('optimizer', 'N/A')}")
            print(f"    Scheduler:   {row.get('scheduler', 'N/A')}")

            comparison_data.append({
                'rank': i,
                'run_name': row['run_name'],
                'val_overall_acc': row['val_overall_acc'],
                'val_g_command_acc': row['val_g_command_acc'],
                'val_loss': row['val_loss'],
                'batch_size': row.get('batch_size', np.nan),
                'hidden_dim': row.get('hidden_dim', np.nan),
                'num_layers': row.get('num_layers', np.nan),
                'num_heads': row.get('num_heads', np.nan),
                'learning_rate': row.get('learning_rate', np.nan),
                'optimizer': row.get('optimizer', ''),
                'scheduler': row.get('scheduler', '')
            })

        # Save comparison
        comp_df = pd.DataFrame(comparison_data)
        comp_df.to_csv(self.output_dir / 'data' / 'top5_comparison.csv', index=False)
        print(f"\n✓ Saved: top5_comparison.csv")

        # Plot comparison
        self._plot_configuration_comparison(comp_df)

        return comp_df

    def _plot_configuration_comparison(self, comp_df: pd.DataFrame):
        """Plot configuration comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Accuracy comparison
        ax = axes[0, 0]
        x = np.arange(len(comp_df))
        width = 0.35

        ax.bar(x - width/2, comp_df['val_overall_acc']*100, width,
               label='Overall Acc', color='steelblue', edgecolor='black')
        ax.bar(x + width/2, comp_df['val_g_command_acc']*100, width,
               label='G-Command Acc', color='orange', edgecolor='black')

        ax.set_xlabel('Configuration Rank', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title('Accuracy Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Rank {i+1}' for i in range(len(comp_df))])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 2. Loss comparison
        ax = axes[0, 1]
        ax.bar(x, comp_df['val_loss'], color='crimson', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Configuration Rank', fontweight='bold')
        ax.set_ylabel('Validation Loss', fontweight='bold')
        ax.set_title('Loss Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Rank {i+1}' for i in range(len(comp_df))])
        ax.grid(axis='y', alpha=0.3)

        # 3. Model size comparison
        ax = axes[1, 0]
        if comp_df['hidden_dim'].notna().any():
            ax.bar(x, comp_df['hidden_dim'], color='purple', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Configuration Rank', fontweight='bold')
            ax.set_ylabel('Hidden Dimension', fontweight='bold')
            ax.set_title('Model Size Comparison', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Rank {i+1}' for i in range(len(comp_df))])
            ax.grid(axis='y', alpha=0.3)

        # 4. Learning rate comparison
        ax = axes[1, 1]
        if comp_df['learning_rate'].notna().any():
            ax.bar(x, comp_df['learning_rate'], color='green', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Configuration Rank', fontweight='bold')
            ax.set_ylabel('Learning Rate', fontweight='bold')
            ax.set_title('Learning Rate Comparison', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Rank {i+1}' for i in range(len(comp_df))])
            ax.set_yscale('log')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'configuration_comparison.png',
                   bbox_inches='tight', dpi=300)
        plt.close()
        print(f"✓ Saved: configuration_comparison.png")

    def plot_metric_distributions(self):
        """Plot distributions of key metrics"""
        print("\n" + "="*80)
        print("PHASE 4: METRIC DISTRIBUTIONS")
        print("="*80)

        metrics = {
            'val_overall_acc': 'Overall Accuracy',
            'val_g_command_acc': 'G-Command Accuracy',
            'val_m_command_acc': 'M-Command Accuracy',
            'val_numeric_acc': 'Numeric Accuracy',
            'val_loss': 'Validation Loss'
        }

        # Filter available metrics
        available_metrics = {k: v for k, v in metrics.items() if k in self.df_clean.columns}

        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for idx, (metric, label) in enumerate(available_metrics.items()):
            ax = axes[idx]

            data = self.df_clean[metric].dropna()

            # Histogram
            n, bins, patches = ax.hist(data, bins=20, color='steelblue',
                                       edgecolor='black', alpha=0.7, density=False)

            # Mean line
            mean_val = data.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.3f}')

            # Median line
            median_val = data.median()
            ax.axvline(median_val, color='orange', linestyle='--', linewidth=2,
                      label=f'Median: {median_val:.3f}')

            ax.set_xlabel(label, fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_title(f'{label} Distribution', fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            print(f"\n{label}:")
            print(f"  Mean:   {mean_val:.4f}")
            print(f"  Median: {median_val:.4f}")
            print(f"  Std:    {data.std():.4f}")

        # Hide unused subplots
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'metric_distributions.png',
                   bbox_inches='tight', dpi=300)
        plt.close()
        print(f"\n✓ Saved: metric_distributions.png")

    def generate_summary_report(self, param_importance: List[Tuple[str, float]],
                               top5_configs: pd.DataFrame):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("PHASE 5: SUMMARY REPORT GENERATION")
        print("="*80)

        report = []
        report.append("=" * 80)
        report.append(f"COMPREHENSIVE SWEEP {self.sweep_id} ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Sweep ID: {self.sweep_id}")
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Runs: {len(self.df)}")
        report.append(f"Successful Runs: {len(self.df_clean)}")
        report.append(f"Failed Runs: {len(self.df) - len(self.df_clean)}")
        report.append("")

        # Performance Summary
        report.append("=" * 80)
        report.append("PERFORMANCE SUMMARY")
        report.append("=" * 80)
        report.append("")
        report.append("Overall Accuracy:")
        report.append(f"  Best:   {self.df_clean['val_overall_acc'].max()*100:.2f}%")
        report.append(f"  Mean:   {self.df_clean['val_overall_acc'].mean()*100:.2f}%")
        report.append(f"  Median: {self.df_clean['val_overall_acc'].median()*100:.2f}%")
        report.append(f"  Std:    {self.df_clean['val_overall_acc'].std()*100:.2f}%")
        report.append("")

        report.append("G-Command Accuracy:")
        report.append(f"  Best:   {self.df_clean['val_g_command_acc'].max()*100:.2f}%")
        report.append(f"  Mean:   {self.df_clean['val_g_command_acc'].mean()*100:.2f}%")
        report.append(f"  Median: {self.df_clean['val_g_command_acc'].median()*100:.2f}%")
        report.append("")

        report.append("Validation Loss:")
        report.append(f"  Best:   {self.df_clean['val_loss'].min():.4f}")
        report.append(f"  Mean:   {self.df_clean['val_loss'].mean():.4f}")
        report.append(f"  Median: {self.df_clean['val_loss'].median():.4f}")
        report.append("")

        # Parameter Importance
        report.append("=" * 80)
        report.append("HYPERPARAMETER IMPORTANCE")
        report.append("=" * 80)
        report.append("")
        for i, (param, importance) in enumerate(param_importance, 1):
            report.append(f"  {i}. {param:20s} {importance:.4f}")
        report.append("")

        # Top Configurations
        report.append("=" * 80)
        report.append("TOP 5 CONFIGURATIONS")
        report.append("=" * 80)
        report.append("")

        for idx, row in top5_configs.iterrows():
            report.append(f"Rank {row['rank']}: {row['run_name']}")
            report.append(f"  Overall Accuracy:   {row['val_overall_acc']*100:.2f}%")
            report.append(f"  G-Command Accuracy: {row['val_g_command_acc']*100:.2f}%")
            report.append(f"  Validation Loss:    {row['val_loss']:.4f}")
            report.append(f"  Configuration:")
            report.append(f"    Batch Size:       {row['batch_size']}")
            report.append(f"    Hidden Dimension: {row['hidden_dim']}")
            report.append(f"    Num Layers:       {row['num_layers']}")
            report.append(f"    Num Heads:        {row['num_heads']}")
            report.append(f"    Learning Rate:    {row['learning_rate']:.6f}")
            report.append(f"    Optimizer:        {row['optimizer']}")
            report.append(f"    Scheduler:        {row['scheduler']}")
            report.append("")

        # Recommendations
        report.append("=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("=" * 80)
        report.append("")
        report.append("Based on the sweep results:")
        report.append("")
        report.append("1. Optimal Hyperparameters:")
        best_run = top5_configs.iloc[0]
        report.append(f"   - Batch Size:       {best_run['batch_size']}")
        report.append(f"   - Hidden Dimension: {best_run['hidden_dim']}")
        report.append(f"   - Num Layers:       {best_run['num_layers']}")
        report.append(f"   - Num Heads:        {best_run['num_heads']}")
        report.append(f"   - Learning Rate:    {best_run['learning_rate']:.6f}")
        report.append(f"   - Optimizer:        {best_run['optimizer']}")
        report.append(f"   - Scheduler:        {best_run['scheduler']}")
        report.append("")

        report.append("2. Key Findings:")
        report.append(f"   - {len([1 for _, row in self.df_clean.iterrows() if row['val_g_command_acc'] == 1.0])} runs achieved 100% G-command accuracy")
        report.append(f"   - {len([1 for _, row in self.df_clean.iterrows() if row['val_overall_acc'] == 1.0])} runs achieved 100% overall accuracy")
        report.append(f"   - Most important hyperparameter: {param_importance[0][0]}")
        report.append("")

        report.append("3. Next Steps:")
        report.append("   - Deploy best configuration to production")
        report.append("   - Consider ensemble of top-3 models")
        report.append("   - Further tune learning rate schedule")
        report.append("")

        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        # Save report
        report_text = '\n'.join(report)
        report_path = self.output_dir / 'reports' / 'comprehensive_summary.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)

        print(f"\n✓ Saved: comprehensive_summary.txt")
        print("\n" + report_text)

        return report_text


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Sweep Analysis - Analyze W&B hyperparameter sweep results'
    )
    parser.add_argument(
        '--sweep-id',
        type=str,
        required=True,
        help='W&B sweep ID to analyze (e.g., 27v7pl9i)'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default=None,
        help='Path to sweep CSV file (default: outputs/wandb_sweep_<sweep-id>_analysis.csv)'
    )
    args = parser.parse_args()

    sweep_id = args.sweep_id

    print("=" * 80)
    print(f"COMPREHENSIVE SWEEP {sweep_id} ANALYSIS")
    print("=" * 80)

    # Paths
    base_dir = Path(__file__).parent.parent

    # Determine CSV path
    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        # Try sweep-specific CSV first, fallback to generic
        sweep_csv = base_dir / 'outputs' / f'wandb_sweep_{sweep_id}_analysis.csv'
        generic_csv = base_dir / 'outputs' / 'wandb_sweep_analysis.csv'

        if sweep_csv.exists():
            csv_path = sweep_csv
            print(f"Using sweep-specific CSV: {csv_path}")
        elif generic_csv.exists():
            csv_path = generic_csv
            print(f"Using generic CSV: {csv_path}")
        else:
            raise FileNotFoundError(
                f"No CSV file found. Tried:\n"
                f"  - {sweep_csv}\n"
                f"  - {generic_csv}\n"
                f"Please specify --csv-path or generate CSV with analyze_sweep.py"
            )

    output_dir = base_dir / 'outputs' / f'sweep_{sweep_id}_comprehensive_analysis'

    # Initialize analyzer
    analyzer = SweepAnalyzer(base_dir, output_dir, sweep_id)

    # Phase 1: Load data
    analyzer.load_sweep_data(csv_path)

    # Phase 2: Parameter importance
    param_importance = analyzer.analyze_parameter_importance()

    # Phase 3: Configuration comparison
    top5_configs = analyzer.compare_configurations()

    # Phase 4: Metric distributions
    analyzer.plot_metric_distributions()

    # Phase 5: Summary report
    analyzer.generate_summary_report(param_importance, top5_configs)

    print("\n" + "=" * 80)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  Data:")
    print("    - cleaned_sweep_results.csv")
    print("    - parameter_importance.csv")
    print("    - top5_comparison.csv")
    print("\n  Figures:")
    print("    - parameter_importance.png")
    print("    - parallel_coordinates.png")
    print("    - configuration_comparison.png")
    print("    - metric_distributions.png")
    print("\n  Reports:")
    print("    - comprehensive_summary.txt")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
