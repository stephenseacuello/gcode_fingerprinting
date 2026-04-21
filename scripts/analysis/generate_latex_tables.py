#!/usr/bin/env python3
"""Generate LaTeX Tables for Encoder Ablation Paper.

This script reads ANOVA analysis results and generates LaTeX table code
ready to paste into the paper.

Usage:
    python scripts/analysis/generate_latex_tables.py \
        --results-dir outputs/ablation_study_2026_02_06/analysis \
        --output-dir outputs/ablation_study_2026_02_06/paper/tables
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_results(results_dir: Path) -> dict:
    """Load all analysis results."""
    results = {}

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

    anova_path = results_dir / 'anova_results.json'
    if anova_path.exists():
        with open(anova_path) as f:
            results['anova'] = json.load(f)

    importance_path = results_dir / 'importance_rankings.json'
    if importance_path.exists():
        with open(importance_path) as f:
            results['importance'] = json.load(f)

    l5_path = results_dir / 'l5_threeway_anova.json'
    if l5_path.exists():
        with open(l5_path) as f:
            results['l5_anova'] = json.load(f)

    return results


def format_pvalue(p: float) -> str:
    """Format p-value for LaTeX."""
    if p < 0.001:
        return "$<$0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    elif p < 0.05:
        return f"{p:.2f}"
    else:
        return f"{p:.2f}"


def generate_modality_loo_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for modality LOO results."""
    # Filter to modality_grouped LOO
    loo_df = df[(df['study'] == 'modality_grouped') & (df['ablation_type'] == 'loo')]
    baseline_df = df[(df['study'] == 'modality_grouped') & (df['ablation_type'] == 'baseline')]

    if loo_df.empty:
        return "% No modality LOO data available\n"

    baseline_acc = baseline_df['test_accuracy'].mean() if not baseline_df.empty else 1.0

    # Calculate stats
    stats = loo_df.groupby('removed')['test_accuracy'].agg(['mean', 'std']).reset_index()
    stats['drop'] = baseline_acc - stats['mean']
    stats = stats.sort_values('drop', ascending=False)

    # Build LaTeX table
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Modality Leave-One-Out Results}",
        "\\label{tab:modality_loo}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "\\textbf{Removed Modality} & \\textbf{Test Acc (\\%)} & \\textbf{$\\Delta$ Acc} \\\\",
        "\\midrule",
        f"None (baseline) & {baseline_acc*100:.2f} $\\pm$ 0.00 & --- \\\\",
        "\\midrule",
    ]

    for _, row in stats.iterrows():
        modality = row['removed'].replace('_', '\\_')
        acc = f"{row['mean']*100:.2f} $\\pm$ {row['std']*100:.2f}"
        drop = f"$-${row['drop']*100:.2f}"
        lines.append(f"{modality} & {acc} & {drop} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return '\n'.join(lines)


def generate_modality_loi_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for modality LOI (single modality) results."""
    loi_df = df[(df['study'] == 'modality_grouped') & (df['ablation_type'] == 'loi')]

    if loi_df.empty:
        return "% No modality LOI data available\n"

    stats = loi_df.groupby('included')['test_accuracy'].agg(['mean', 'std']).reset_index()
    stats = stats.sort_values('mean', ascending=False)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Single Modality Performance (LOI)}",
        "\\label{tab:modality_loi}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "\\textbf{Single Modality} & \\textbf{Test Acc (\\%)} \\\\",
        "\\midrule",
    ]

    for _, row in stats.iterrows():
        modality = row['included'].replace('_', '\\_')
        acc = f"{row['mean']*100:.2f} $\\pm$ {row['std']*100:.2f}"
        lines.append(f"{modality} only & {acc} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return '\n'.join(lines)


def generate_sensor_group_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for sensor group ablation."""
    group_df = df[df['study'] == 'sensor_group']

    if group_df.empty:
        return "% No sensor group data available\n"

    baseline_df = group_df[group_df['ablation_type'] == 'baseline']
    baseline_acc = baseline_df['test_accuracy'].mean() if not baseline_df.empty else 1.0

    loo_stats = group_df[group_df['ablation_type'] == 'loo'].groupby('removed')['test_accuracy'].agg(['mean', 'std']).reset_index()
    loo_stats.columns = ['group', 'loo_mean', 'loo_std']

    loi_stats = group_df[group_df['ablation_type'] == 'loi'].groupby('included')['test_accuracy'].agg(['mean', 'std']).reset_index()
    loi_stats.columns = ['group', 'loi_mean', 'loi_std']

    stats = loo_stats.merge(loi_stats, on='group', how='outer')
    stats['loo_drop'] = baseline_acc - stats['loo_mean']
    stats = stats.sort_values('loo_drop', ascending=False)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Sensor Group Ablation}",
        "\\label{tab:sensor_group}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Group} & \\textbf{Sensors} & \\textbf{LOO Acc (\\%)} & \\textbf{LOI Acc (\\%)} \\\\",
        "\\midrule",
    ]

    sensor_counts = {'frame': 7, 'spindle': 2, 'motor': 1, 'bed': 4, 'gantry': 2}

    for _, row in stats.iterrows():
        group = row['group']
        count = sensor_counts.get(group, '?')
        loo = f"{row['loo_mean']*100:.2f}" if pd.notna(row['loo_mean']) else "---"
        loi = f"{row['loi_mean']*100:.2f}" if pd.notna(row['loi_mean']) else "---"
        lines.append(f"{group} & {count} & {loo} & {loi} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return '\n'.join(lines)


def generate_architecture_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for architecture ablation."""
    arch_df = df[df['study'] == 'architecture']

    if arch_df.empty:
        return "% No architecture data available\n"

    stats = arch_df.groupby('config_name')['test_accuracy'].agg(['mean', 'std']).reset_index()
    stats = stats.sort_values('mean', ascending=False)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Architecture Ablation Results}",
        "\\label{tab:architecture}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "\\textbf{Configuration} & \\textbf{Test Acc (\\%)} \\\\",
        "\\midrule",
    ]

    for _, row in stats.iterrows():
        config = row['config_name'].replace('_', '\\_')
        acc = f"{row['mean']*100:.2f} $\\pm$ {row['std']*100:.2f}"
        lines.append(f"{config} & {acc} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return '\n'.join(lines)


def generate_baseline_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for baseline comparison."""
    baseline_df = df[df['study'] == 'baseline'].copy()

    if baseline_df.empty:
        return "% No baseline data available\n"

    # Use 'model' column if available, otherwise use 'config_name'
    model_col = 'model' if 'model' in baseline_df.columns and baseline_df['model'].notna().any() else 'config_name'

    stats = baseline_df.groupby(model_col)['test_accuracy'].agg(['mean', 'std']).reset_index()
    stats = stats.rename(columns={model_col: 'model'})
    stats = stats.sort_values('mean', ascending=False)

    ml_models = ['xgboost', 'random_forest', 'svm_rbf', 'logistic_regression', 'knn']
    nn_models = ['mlp', 'cnn_1d', 'lstm_simple', 'transformer', 'tcn']

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Baseline Model Comparison}",
        "\\label{tab:baselines}",
        "\\begin{tabular}{llc}",
        "\\toprule",
        "\\textbf{Type} & \\textbf{Model} & \\textbf{Test Acc (\\%)} \\\\",
        "\\midrule",
    ]

    # Add Traditional ML section
    ml_stats = stats[stats['model'].str.lower().isin(ml_models)]
    if not ml_stats.empty:
        first = True
        for _, row in ml_stats.iterrows():
            prefix = "\\multirow{" + str(len(ml_stats)) + "}{*}{Traditional ML}" if first else ""
            first = False
            model = row['model'].replace('_', '\\_')
            acc = f"{row['mean']*100:.2f} $\\pm$ {row['std']*100:.2f}"
            lines.append(f"{prefix} & {model} & {acc} \\\\")
        lines.append("\\midrule")

    # Add Neural Network section
    nn_stats = stats[stats['model'].str.lower().isin(nn_models)]
    if not nn_stats.empty:
        first = True
        for _, row in nn_stats.iterrows():
            prefix = "\\multirow{" + str(len(nn_stats)) + "}{*}{Neural Networks}" if first else ""
            first = False
            model = row['model'].replace('_', '\\_')
            acc = f"{row['mean']*100:.2f} $\\pm$ {row['std']*100:.2f}"
            lines.append(f"{prefix} & {model} & {acc} \\\\")
        lines.append("\\midrule")

    # Add our model
    lines.append("& \\textbf{MM-DTAE-LSTM} & \\textbf{100.00} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return '\n'.join(lines)


def generate_anova_table(anova_results: dict) -> str:
    """Generate LaTeX table for ANOVA results."""
    if not anova_results:
        return "% No ANOVA results available\n"

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Two-Way ANOVA Summary (Configuration Effect)}",
        "\\label{tab:anova}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Study} & \\textbf{$F$} & \\textbf{$p$-value} & \\textbf{$\\eta^2$} \\\\",
        "\\midrule",
    ]

    for study, results in anova_results.items():
        if not isinstance(results, dict):
            continue

        study_name = study.replace('_', '\\_')

        # Handle different ANOVA result structures
        if 'oneway_anova' in results:
            f_val = f"{results['oneway_anova'].get('f_statistic', 0):.2f}"
            p_val = format_pvalue(results['oneway_anova'].get('p_value', 1.0))
        elif 'twoway_anova' in results:
            config_effect = results['twoway_anova'].get('config_effect', {})
            f_val = f"{config_effect.get('f_value', 0):.2f}"
            p_val = format_pvalue(config_effect.get('p_value', 1.0))
        else:
            f_val = f"{results.get('f_value', results.get('f_statistic', 0)):.2f}"
            p_val = format_pvalue(results.get('p_value', 1.0))

        # Handle effect size
        if 'effect_size' in results:
            eta = f"{results['effect_size'].get('eta_squared', 0):.3f}"
        else:
            eta = f"{results.get('eta_squared', 0):.3f}"

        lines.append(f"{study_name} & {f_val} & {p_val} & {eta} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return '\n'.join(lines)


def generate_crossed_l1_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for L1 sensor × modality top configs."""
    l1_df = df[df['study'] == 'crossed_L1_sensor_modality']

    if l1_df.empty:
        return "% No L1 sensor×modality data available\n"

    stats = l1_df.groupby('config_name')['test_accuracy'].agg(['mean', 'std', 'count']).reset_index()
    stats = stats.sort_values('mean', ascending=False)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Top-20 Single Sensor--Modality Configurations (L1)}",
        "\\label{tab:l1_sensor_modality}",
        "\\begin{tabular}{rlcc}",
        "\\toprule",
        "\\textbf{Rank} & \\textbf{Sensor -- Modality} & \\textbf{Test Acc (\\%)} & \\textbf{$n$} \\\\",
        "\\midrule",
    ]

    for rank, (_, row) in enumerate(stats.head(20).iterrows(), 1):
        config = str(row['config_name']).replace('_', '\\_')
        acc = f"{row['mean']*100:.2f} $\\pm$ {row['std']*100:.2f}"
        n = int(row['count'])
        lines.append(f"{rank} & {config} & {acc} & {n} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return '\n'.join(lines)


def generate_crossed_l2_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for L2 sensor group × modality top configs."""
    l2_df = df[df['study'] == 'crossed_L2_group_modality']

    if l2_df.empty:
        return "% No L2 group×modality data available\n"

    stats = l2_df.groupby('config_name')['test_accuracy'].agg(['mean', 'std', 'count']).reset_index()
    stats = stats.sort_values('mean', ascending=False)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Sensor Group $\\times$ Modality Configurations (L2)}",
        "\\label{tab:l2_group_modality}",
        "\\begin{tabular}{rlcc}",
        "\\toprule",
        "\\textbf{Rank} & \\textbf{Group -- Modality} & \\textbf{Test Acc (\\%)} & \\textbf{$n$} \\\\",
        "\\midrule",
    ]

    for rank, (_, row) in enumerate(stats.head(20).iterrows(), 1):
        config = str(row['config_name']).replace('_', '\\_')
        acc = f"{row['mean']*100:.2f} $\\pm$ {row['std']*100:.2f}"
        n = int(row['count'])
        lines.append(f"{rank} & {config} & {acc} & {n} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return '\n'.join(lines)


def generate_crossed_l3_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for L3 sensor pair top configs."""
    l3_df = df[df['study'] == 'crossed_L3_sensor_pairs']

    if l3_df.empty:
        return "% No L3 sensor pair data available\n"

    stats = l3_df.groupby('config_name')['test_accuracy'].agg(['mean', 'std', 'count']).reset_index()
    stats = stats.sort_values('mean', ascending=False)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Top-20 Sensor Pair Configurations (L3)}",
        "\\label{tab:l3_sensor_pairs}",
        "\\begin{tabular}{rlcc}",
        "\\toprule",
        "\\textbf{Rank} & \\textbf{Sensor Pair} & \\textbf{Test Acc (\\%)} & \\textbf{$n$} \\\\",
        "\\midrule",
    ]

    for rank, (_, row) in enumerate(stats.head(20).iterrows(), 1):
        config = str(row['config_name']).replace('+', ' + ').replace('_', '\\_')
        acc = f"{row['mean']*100:.2f} $\\pm$ {row['std']*100:.2f}"
        n = int(row['count'])
        lines.append(f"{rank} & {config} & {acc} & {n} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return '\n'.join(lines)


def generate_crossed_l4_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for L4 modality combination top configs."""
    l4_df = df[df['study'] == 'crossed_L4_modality_combos']

    if l4_df.empty:
        return "% No L4 modality combo data available\n"

    stats = l4_df.groupby('config_name')['test_accuracy'].agg(['mean', 'std', 'count']).reset_index()
    stats = stats.sort_values('mean', ascending=False)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Modality Combination Configurations (L4)}",
        "\\label{tab:l4_modality_combos}",
        "\\begin{tabular}{rlcc}",
        "\\toprule",
        "\\textbf{Rank} & \\textbf{Modality Combination} & \\textbf{Test Acc (\\%)} & \\textbf{$n$} \\\\",
        "\\midrule",
    ]

    for rank, (_, row) in enumerate(stats.iterrows(), 1):
        config = str(row['config_name']).replace('+', ' + ').replace('_', '\\_')
        acc = f"{row['mean']*100:.2f} $\\pm$ {row['std']*100:.2f}"
        n = int(row['count'])
        lines.append(f"{rank} & {config} & {acc} & {n} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return '\n'.join(lines)


def generate_l5_table(df: pd.DataFrame, l5_anova: dict = None) -> str:
    """Generate LaTeX table for L5 sensor × channel top configs."""
    l5_df = df[df['study'] == 'crossed_L5_sensor_channel']

    if l5_df.empty and not l5_anova:
        return "% No L5 sensor×channel data available\n"

    # Use L5 ANOVA top_20 if available, otherwise compute from df
    if l5_anova and 'top_20' in l5_anova:
        top20 = l5_anova['top_20']
        configs = list(top20['mean'].keys())
        rows = []
        for config in configs:
            rows.append({
                'config_name': config,
                'mean': top20['mean'][config],
                'std': top20['std'].get(config, 0),
                'count': top20['count'].get(config, 0),
            })
        top_stats = pd.DataFrame(rows)
    else:
        top_stats = l5_df.groupby('config_name')['test_accuracy'].agg(['mean', 'std', 'count']).reset_index()
        top_stats = top_stats.sort_values('mean', ascending=False).head(20)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Top-20 Sensor $\\times$ Channel Configurations (L5)}",
        "\\label{tab:l5_top_configs}",
        "\\begin{tabular}{rlcc}",
        "\\toprule",
        "\\textbf{Rank} & \\textbf{Sensor -- Channel} & \\textbf{Test Acc (\\%)} & \\textbf{$n$} \\\\",
        "\\midrule",
    ]

    for rank, (_, row) in enumerate(top_stats.iterrows(), 1):
        config = str(row['config_name']).replace('_', '\\_')
        acc = f"{row['mean']*100:.2f} $\\pm$ {row['std']*100:.2f}"
        n = int(row['count'])
        lines.append(f"{rank} & {config} & {acc} & {n} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    # Add L5 three-way ANOVA summary table if available
    if l5_anova and 'threeway_anova' in l5_anova:
        anova_data = l5_anova['threeway_anova']
        lines.extend([
            "",
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{L5 Three-Way ANOVA: Sensor $\\times$ Channel $\\times$ Data Split}",
            "\\label{tab:l5_anova}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "\\textbf{Factor} & \\textbf{df} & \\textbf{$F$} & \\textbf{$p$-value} & \\textbf{$\\eta^2_p$} \\\\",
            "\\midrule",
        ])

        factor_labels = {
            'C(sensor)': 'Sensor',
            'C(channel)': 'Channel',
            'C(sensor):C(channel)': 'Sensor $\\times$ Channel',
            'C(data_split_seed)': 'Data Split',
        }

        for factor_key, label in factor_labels.items():
            if factor_key in anova_data:
                fdata = anova_data[factor_key]
                df_val = int(fdata['df'])
                f_val = f"{fdata['F']:.2f}"
                p_val = format_pvalue(fdata['p_value'])
                eta = f"{fdata.get('partial_eta_squared', 0):.3f}"
                lines.append(f"{label} & {df_val} & {f_val} & {p_val} & {eta} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to analysis results directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for LaTeX files')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)

    if 'df' not in results:
        print(f"ERROR: No results CSV found at {results_dir / 'all_results.csv'}")
        return

    df = results['df']
    anova = results.get('anova', {})
    l5_anova = results.get('l5_anova', None)
    print(f"Loaded {len(df)} experiment results")

    # Generate all tables
    tables = {
        'table_modality_loo.tex': generate_modality_loo_table(df),
        'table_modality_loi.tex': generate_modality_loi_table(df),
        'table_sensor_group.tex': generate_sensor_group_table(df),
        'table_architecture.tex': generate_architecture_table(df),
        'table_baselines.tex': generate_baseline_table(df),
        'table_anova.tex': generate_anova_table(anova),
        'table_l1_sensor_modality.tex': generate_crossed_l1_table(df),
        'table_l2_group_modality.tex': generate_crossed_l2_table(df),
        'table_l3_sensor_pairs.tex': generate_crossed_l3_table(df),
        'table_l4_modality_combos.tex': generate_crossed_l4_table(df),
        'table_l5_top_configs.tex': generate_l5_table(df, l5_anova),
    }

    # Save all tables
    for filename, content in tables.items():
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Saved: {filename}")

    # Also save combined file
    combined_path = output_dir / 'all_tables.tex'
    with open(combined_path, 'w') as f:
        f.write("% Auto-generated LaTeX tables for encoder ablation paper\n")
        f.write(f"% Generated from: {results_dir}\n\n")
        for filename, content in tables.items():
            f.write(f"% === {filename} ===\n")
            f.write(content)
            f.write("\n\n")
    print(f"\nCombined tables saved to: {combined_path}")


if __name__ == '__main__':
    main()
