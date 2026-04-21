#!/usr/bin/env python3
"""Ablation Study ANOVA Analysis.

Performs statistical analysis on ablation study results:
- Two-way ANOVA (config × data_split) per study
- Post-hoc Tukey HSD for pairwise comparisons
- Effect size calculations (η²)
- Importance rankings
- Visualization generation

Usage:
    python scripts/analysis/run_ablation_anova.py \
        --results-dir outputs/ablation_study_2026_02_06 \
        --output-dir outputs/ablation_study_2026_02_06/analysis

    # Specific studies only
    python scripts/analysis/run_ablation_anova.py \
        --results-dir outputs/ablation_study_2026_02_06 \
        --output-dir outputs/ablation_study_2026_02_06/analysis \
        --studies modality_grouped,architecture
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Statistical imports
try:
    import scipy.stats as stats
    from scipy.stats import f_oneway, tukey_hsd
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def collect_results(results_dir, studies=None):
    """Collect all results.json files from ablation study."""
    results_dir = Path(results_dir)
    all_results = []

    # Walk through directory structure
    for results_file in results_dir.rglob('results.json'):
        try:
            with open(results_file) as f:
                result = json.load(f)

            # Extract experiment info from path
            rel_path = results_file.relative_to(results_dir)
            parts = rel_path.parts

            # Parse directory structure
            if 'crossed' in parts:
                # Crossed factorial studies (L1-L5)
                if 'L1_sensor_modality' in parts:
                    study = 'crossed_L1_sensor_modality'
                elif 'L2_group_modality' in parts:
                    study = 'crossed_L2_group_modality'
                elif 'L3_sensor_pairs' in parts:
                    study = 'crossed_L3_sensor_pairs'
                elif 'L4_modality_combos' in parts:
                    study = 'crossed_L4_modality_combos'
                elif 'L5_sensor_channel' in parts:
                    study = 'crossed_L5_sensor_channel'
                else:
                    continue
            elif 'modality' in parts:
                if 'grouped' in parts:
                    study = 'modality_grouped'
                else:
                    study = 'modality_individual'
            elif 'sensor' in parts:
                if 'group' in parts:
                    study = 'sensor_group'
                else:
                    study = 'sensor_individual'
            elif 'architecture' in parts:
                study = 'architecture'
            elif 'baseline' in parts:
                study = 'baseline'
            else:
                continue

            # Filter by studies if specified
            if studies and study not in studies:
                continue

            # Extract config info
            config = result.get('config', {})

            # Get accuracy metrics
            val_best_test_acc = result.get('val_best_test', {}).get('accuracy', None)
            if val_best_test_acc is None:
                # Try alternative keys
                val_best_test_acc = result.get('test', {}).get('accuracy', None)

            if val_best_test_acc is None:
                continue

            # Extract seeds from iteration name or config
            iteration = config.get('iteration', '')
            model_seed = config.get('seed', 42)

            # Parse data split seed from iteration name if present
            if '_ds' in iteration and '_ms' in iteration:
                try:
                    ds_part = iteration.split('_ds')[1].split('_')[0]
                    ms_part = iteration.split('_ms')[1].split('_')[0] if '_ms' in iteration else str(model_seed)
                    data_split_seed = int(ds_part)
                    model_seed = int(ms_part)
                except (IndexError, ValueError):
                    data_split_seed = 42
                    model_seed = 42
            else:
                data_split_seed = 42

            # Extract config name
            if 'ablation_type' in config:
                ablation_type = config['ablation_type']
                if ablation_type == 'baseline':
                    config_name = 'baseline'
                elif ablation_type == 'loo':
                    ablation_config = config.get('ablation_config', config)
                    if 'exclude_modalities' in ablation_config:
                        config_name = f"loo_{ablation_config['exclude_modalities'][0]}"
                    elif 'exclude_components' in ablation_config:
                        config_name = f"loo_{ablation_config['exclude_components'][0]}"
                    elif 'exclude_sensors' in ablation_config:
                        config_name = f"loo_{ablation_config['exclude_sensors'][0]}"
                    elif 'exclude_sensor_groups' in ablation_config:
                        config_name = f"loo_{ablation_config['exclude_sensor_groups'][0]}"
                    else:
                        config_name = 'loo_unknown'
                elif ablation_type == 'loi':
                    ablation_config = config.get('ablation_config', config)
                    if 'include_only_modalities' in ablation_config:
                        config_name = f"loi_{ablation_config['include_only_modalities'][0]}"
                    elif 'include_only_components' in ablation_config:
                        config_name = f"loi_{ablation_config['include_only_components'][0]}"
                    elif 'include_only_sensors' in ablation_config:
                        config_name = f"loi_{ablation_config['include_only_sensors'][0]}"
                    elif 'include_only_sensor_groups' in ablation_config:
                        config_name = f"loi_{ablation_config['include_only_sensor_groups'][0]}"
                    else:
                        config_name = 'loi_unknown'
                else:
                    config_name = iteration.split('_ds')[0] if '_ds' in iteration else iteration
            elif 'model' in config:
                # Baseline model
                config_name = config['model']
            else:
                config_name = iteration.split('_ds')[0] if '_ds' in iteration else 'unknown'

            # Get per-class accuracies
            per_class = result.get('val_best_test', {}).get('per_class', {})
            if not per_class:
                per_class = result.get('test', {}).get('per_class', {})

            damage_classes = ['damageadaptive', 'damageface', 'damagepocket']
            damage_accs = [per_class.get(c, 0) for c in damage_classes]
            damage_mean_acc = np.mean(damage_accs) if damage_accs else 0

            all_results.append({
                'study': study,
                'config_name': config_name,
                'data_split_seed': data_split_seed,
                'model_seed': model_seed,
                'test_accuracy': val_best_test_acc,
                'damage_mean_accuracy': damage_mean_acc,
                'file_path': str(results_file),
                **{f'acc_{c}': per_class.get(c, 0) for c in per_class},
            })

        except Exception as e:
            print(f"Warning: Could not parse {results_file}: {e}")
            continue

    return pd.DataFrame(all_results)


def run_anova_per_study(df, study_name):
    """Run two-way ANOVA for a single study."""
    study_df = df[df['study'] == study_name].copy()

    if len(study_df) == 0:
        return None

    # Need at least 2 configs and some replicates
    configs = study_df['config_name'].unique()
    if len(configs) < 2:
        return {'error': 'Need at least 2 configs for ANOVA'}

    results = {
        'study': study_name,
        'n_configs': len(configs),
        'n_observations': len(study_df),
        'configs': list(configs),
    }

    # Basic descriptive stats per config
    config_stats = study_df.groupby('config_name')['test_accuracy'].agg(['mean', 'std', 'count'])
    config_stats = config_stats.sort_values('mean', ascending=False)
    results['config_stats'] = config_stats.to_dict()

    # One-way ANOVA (config effect)
    if HAS_SCIPY:
        groups = [study_df[study_df['config_name'] == c]['test_accuracy'].values for c in configs]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) >= 2:
            f_stat, p_val = f_oneway(*groups)
            results['oneway_anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_val),
                'significant': p_val < 0.05,
            }

            # Effect size (eta squared)
            ss_between = sum(len(g) * (np.mean(g) - study_df['test_accuracy'].mean())**2 for g in groups)
            ss_total = sum((study_df['test_accuracy'] - study_df['test_accuracy'].mean())**2)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0
            results['effect_size'] = {
                'eta_squared': float(eta_sq),
                'interpretation': 'large' if eta_sq > 0.14 else ('medium' if eta_sq > 0.06 else 'small'),
            }

    # Two-way ANOVA if we have multiple data splits
    if HAS_STATSMODELS and len(study_df['data_split_seed'].unique()) > 1:
        try:
            # Create formula
            model = ols('test_accuracy ~ C(config_name) + C(data_split_seed) + C(config_name):C(data_split_seed)',
                       data=study_df).fit()
            anova_table = anova_lm(model, typ=2)

            results['twoway_anova'] = {
                'config_effect': {
                    'f_value': float(anova_table.loc['C(config_name)', 'F']),
                    'p_value': float(anova_table.loc['C(config_name)', 'PR(>F)']),
                },
                'data_split_effect': {
                    'f_value': float(anova_table.loc['C(data_split_seed)', 'F']),
                    'p_value': float(anova_table.loc['C(data_split_seed)', 'PR(>F)']),
                },
                'interaction_effect': {
                    'f_value': float(anova_table.loc['C(config_name):C(data_split_seed)', 'F']),
                    'p_value': float(anova_table.loc['C(config_name):C(data_split_seed)', 'PR(>F)']),
                },
            }
        except Exception as e:
            results['twoway_anova_error'] = str(e)

    # Post-hoc Tukey HSD
    if HAS_STATSMODELS and len(configs) > 2:
        try:
            tukey = pairwise_tukeyhsd(
                study_df['test_accuracy'],
                study_df['config_name'],
                alpha=0.05
            )
            # Get significant pairwise comparisons
            sig_pairs = []
            for i, (reject, group1, group2, meandiff, p_adj, lower, upper) in enumerate(
                zip(tukey.reject, tukey.groupsunique[tukey._multicomp.pairindices[0]],
                    tukey.groupsunique[tukey._multicomp.pairindices[1]],
                    tukey.meandiffs, tukey.pvalues, tukey.confint[:, 0], tukey.confint[:, 1])):
                if reject:
                    sig_pairs.append({
                        'group1': str(group1),
                        'group2': str(group2),
                        'mean_diff': float(meandiff),
                        'p_adj': float(p_adj),
                    })
            results['tukey_hsd'] = {
                'n_significant_pairs': len(sig_pairs),
                'significant_pairs': sig_pairs[:20],  # Limit output
            }
        except Exception as e:
            results['tukey_error'] = str(e)

    return results


def compute_importance_rankings(df, study_name):
    """Compute feature importance rankings for LOO experiments."""
    study_df = df[df['study'] == study_name].copy()

    if len(study_df) == 0:
        return None

    # Get baseline accuracy
    baseline_df = study_df[study_df['config_name'] == 'baseline']
    if len(baseline_df) == 0:
        return None

    baseline_acc = baseline_df['test_accuracy'].mean()

    # Get LOO configs
    loo_configs = [c for c in study_df['config_name'].unique() if c.startswith('loo_')]

    rankings = []
    for config in loo_configs:
        config_df = study_df[study_df['config_name'] == config]
        if len(config_df) == 0:
            continue

        removed_item = config.replace('loo_', '')
        config_acc = config_df['test_accuracy'].mean()
        config_std = config_df['test_accuracy'].std()

        # Importance = accuracy drop when removed
        importance = baseline_acc - config_acc

        rankings.append({
            'item': removed_item,
            'config': config,
            'mean_accuracy': config_acc,
            'std_accuracy': config_std,
            'accuracy_drop': importance,
            'relative_drop_pct': (importance / baseline_acc) * 100 if baseline_acc > 0 else 0,
        })

    # Sort by importance (largest drop = most important)
    rankings = sorted(rankings, key=lambda x: x['accuracy_drop'], reverse=True)

    return {
        'study': study_name,
        'baseline_accuracy': baseline_acc,
        'rankings': rankings,
    }


def run_l5_threeway_anova(df):
    """Run three-way ANOVA for L5: sensor × channel × data_split.

    Decomposes the config_name into sensor and channel factors for
    the 272 sensor×channel configs (excluding 8 electrical).
    """
    l5_df = df[df['study'] == 'crossed_L5_sensor_channel'].copy()
    if len(l5_df) == 0:
        return None

    # All 16 sensors and 17 channels
    ALL_SENSORS = [
        'frame_b1', 'frame_b2', 'frame_l1', 'frame_l2', 'frame_l3', 'frame_r1', 'frame_r2',
        'spindle1', 'spindle2', 'y_bed_1', 'y_bed_2', 'y_bed_3', 'y_bed_4',
        'z_gant_1', 'z_gant_2', 'xa_motor',
    ]
    ALL_CHANNELS = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz',
                    'Pressure', 'Temperature', 'Proximity',
                    'ColorR', 'ColorG', 'ColorB', 'ColorA', 'RMS']

    # Parse config_name into sensor and channel
    def parse_config(config_name):
        if config_name.startswith('electrical_'):
            return None, None
        for sensor in sorted(ALL_SENSORS, key=len, reverse=True):
            if config_name.startswith(sensor + '_'):
                channel = config_name[len(sensor) + 1:]
                if channel in ALL_CHANNELS:
                    return sensor, channel
        return None, None

    parsed = l5_df['config_name'].apply(lambda c: pd.Series(parse_config(c), index=['sensor', 'channel']))
    l5_df = pd.concat([l5_df, parsed], axis=1)

    # Separate sensor×channel from electrical
    sc_df = l5_df[l5_df['sensor'].notna()].copy()
    elec_df = l5_df[l5_df['sensor'].isna()].copy()

    results = {
        'n_sensor_channel': len(sc_df),
        'n_electrical': len(elec_df),
        'n_unique_sensor_channel_configs': sc_df['config_name'].nunique(),
        'n_unique_electrical_configs': elec_df['config_name'].nunique(),
    }

    # Three-way ANOVA on sensor×channel subset
    if HAS_STATSMODELS and len(sc_df) > 10:
        try:
            model = ols('test_accuracy ~ C(sensor) * C(channel) + C(data_split_seed)',
                       data=sc_df).fit()
            anova_table = anova_lm(model, typ=2)

            results['threeway_anova'] = {}
            for factor in ['C(sensor)', 'C(channel)', 'C(sensor):C(channel)', 'C(data_split_seed)']:
                if factor in anova_table.index:
                    results['threeway_anova'][factor] = {
                        'sum_sq': float(anova_table.loc[factor, 'sum_sq']),
                        'df': float(anova_table.loc[factor, 'df']),
                        'F': float(anova_table.loc[factor, 'F']),
                        'p_value': float(anova_table.loc[factor, 'PR(>F)']),
                    }

            # Partial eta-squared for each factor
            ss_error = float(anova_table.loc['Residual', 'sum_sq'])
            for factor in results['threeway_anova']:
                ss_factor = results['threeway_anova'][factor]['sum_sq']
                eta_sq = ss_factor / (ss_factor + ss_error)
                results['threeway_anova'][factor]['partial_eta_squared'] = eta_sq

        except Exception as e:
            results['threeway_anova_error'] = str(e)

    # Marginal means
    if len(sc_df) > 0:
        sensor_means = sc_df.groupby('sensor')['test_accuracy'].agg(['mean', 'std']).to_dict()
        channel_means = sc_df.groupby('channel')['test_accuracy'].agg(['mean', 'std']).to_dict()
        results['sensor_marginal_means'] = sensor_means
        results['channel_marginal_means'] = channel_means

    # Electrical results
    if len(elec_df) > 0:
        elec_stats = elec_df.groupby('config_name')['test_accuracy'].agg(['mean', 'std']).to_dict()
        results['electrical_stats'] = elec_stats

    # Top-20 configs
    config_means = l5_df.groupby('config_name')['test_accuracy'].agg(['mean', 'std', 'count'])
    config_means = config_means.sort_values('mean', ascending=False)
    results['top_20'] = config_means.head(20).to_dict()

    return results


def generate_visualizations(df, anova_results, importance_rankings, output_dir):
    """Generate visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Color palette
    colors = sns.color_palette('husl', 10)

    # 1. LOO accuracy drop bar plots per study
    for study in df['study'].unique():
        if study not in importance_rankings:
            continue

        rankings = importance_rankings[study]
        if not rankings or not rankings['rankings']:
            continue

        fig, ax = plt.subplots(figsize=(12, max(6, len(rankings['rankings']) * 0.3)))

        items = [r['item'] for r in rankings['rankings']]
        drops = [r['accuracy_drop'] * 100 for r in rankings['rankings']]
        stds = [r.get('std_accuracy', 0) * 100 for r in rankings['rankings']]

        y_pos = np.arange(len(items))
        bars = ax.barh(y_pos, drops, xerr=stds, capsize=3, color=colors[0], alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(items)
        ax.set_xlabel('Accuracy Drop (%)')
        ax.set_title(f'{study} - LOO Importance (Accuracy Drop When Removed)')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        # Add value labels
        for i, (v, s) in enumerate(zip(drops, stds)):
            ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / f'{study}_loo_importance.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 2. LOI accuracy bar plots
    for study in df['study'].unique():
        study_df = df[df['study'] == study]
        loi_configs = [c for c in study_df['config_name'].unique() if c.startswith('loi_')]

        if not loi_configs:
            continue

        loi_df = study_df[study_df['config_name'].isin(loi_configs)]
        if len(loi_df) == 0:
            continue

        config_stats = loi_df.groupby('config_name')['test_accuracy'].agg(['mean', 'std']).reset_index()
        config_stats['item'] = config_stats['config_name'].str.replace('loi_', '')
        config_stats = config_stats.sort_values('mean', ascending=True)

        fig, ax = plt.subplots(figsize=(12, max(6, len(config_stats) * 0.3)))

        y_pos = np.arange(len(config_stats))
        bars = ax.barh(y_pos, config_stats['mean'] * 100, xerr=config_stats['std'] * 100,
                      capsize=3, color=colors[2], alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(config_stats['item'])
        ax.set_xlabel('Test Accuracy (%)')
        ax.set_title(f'{study} - LOI Performance (Single Item Only)')
        ax.set_xlim(0, 105)

        for i, v in enumerate(config_stats['mean'] * 100):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / f'{study}_loi_performance.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Comparison across studies (best configs)
    best_per_study = []
    for study in df['study'].unique():
        study_df = df[df['study'] == study]
        config_means = study_df.groupby('config_name')['test_accuracy'].mean()
        best_config = config_means.idxmax()
        best_acc = config_means.max()
        best_per_study.append({
            'study': study,
            'best_config': best_config,
            'accuracy': best_acc,
        })

    if best_per_study:
        fig, ax = plt.subplots(figsize=(12, 6))
        best_df = pd.DataFrame(best_per_study).sort_values('accuracy', ascending=True)

        y_pos = np.arange(len(best_df))
        bars = ax.barh(y_pos, best_df['accuracy'] * 100, color=colors[4], alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['study']}\n({row['best_config']})" for _, row in best_df.iterrows()])
        ax.set_xlabel('Test Accuracy (%)')
        ax.set_title('Best Configuration per Study')
        ax.set_xlim(0, 105)

        for i, v in enumerate(best_df['accuracy'] * 100):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'cross_study_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Boxplot of all configs per study
    for study in df['study'].unique():
        study_df = df[df['study'] == study]
        n_configs = len(study_df['config_name'].unique())

        if n_configs > 20:
            # Too many configs, show only top/bottom
            config_means = study_df.groupby('config_name')['test_accuracy'].mean()
            top_configs = config_means.nlargest(10).index.tolist()
            bottom_configs = config_means.nsmallest(5).index.tolist()
            show_configs = top_configs + [c for c in bottom_configs if c not in top_configs]
            study_df = study_df[study_df['config_name'].isin(show_configs)]
            title_suffix = ' (Top 10 + Bottom 5)'
        else:
            title_suffix = ''

        fig, ax = plt.subplots(figsize=(14, 8))

        # Order by mean accuracy
        order = study_df.groupby('config_name')['test_accuracy'].mean().sort_values(ascending=False).index

        sns.boxplot(data=study_df, x='config_name', y='test_accuracy', order=order, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'{study} - Configuration Comparison{title_suffix}')
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(output_dir / f'{study}_boxplot.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Ablation Study ANOVA Analysis')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Path to ablation study results directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for analysis')
    parser.add_argument('--studies', type=str, default=None,
                        help='Comma-separated list of studies (default: all)')

    args = parser.parse_args()

    studies = None
    if args.studies:
        studies = set(s.strip() for s in args.studies.split(','))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting results from: {args.results_dir}")
    df = collect_results(args.results_dir, studies)

    if len(df) == 0:
        print("ERROR: No results found!")
        return

    print(f"Found {len(df)} experiment results")
    print(f"Studies: {df['study'].unique().tolist()}")

    # Save raw data
    df.to_csv(output_dir / 'all_results.csv', index=False)
    print(f"Saved raw data to: {output_dir / 'all_results.csv'}")

    # Run ANOVA per study
    print("\nRunning ANOVA analysis...")
    anova_results = {}
    for study in df['study'].unique():
        print(f"  {study}...")
        result = run_anova_per_study(df, study)
        if result:
            anova_results[study] = result

    # Save ANOVA results
    with open(output_dir / 'anova_results.json', 'w') as f:
        json.dump(anova_results, f, indent=2, default=str)

    # Compute importance rankings
    print("\nComputing importance rankings...")
    importance_rankings = {}
    for study in df['study'].unique():
        rankings = compute_importance_rankings(df, study)
        if rankings:
            importance_rankings[study] = rankings

    # Save importance rankings
    with open(output_dir / 'importance_rankings.json', 'w') as f:
        json.dump(importance_rankings, f, indent=2)

    # L5 three-way ANOVA
    if 'crossed_L5_sensor_channel' in df['study'].values:
        print("\nRunning L5 three-way ANOVA (sensor × channel × data_split)...")
        l5_anova = run_l5_threeway_anova(df)
        if l5_anova:
            anova_results['crossed_L5_threeway'] = l5_anova
            with open(output_dir / 'l5_threeway_anova.json', 'w') as f:
                json.dump(l5_anova, f, indent=2, default=str)
            print(f"  Saved L5 three-way ANOVA to: {output_dir / 'l5_threeway_anova.json'}")

            if 'threeway_anova' in l5_anova:
                for factor, stats in l5_anova['threeway_anova'].items():
                    sig = '***' if stats['p_value'] < 0.001 else ('**' if stats['p_value'] < 0.01 else '*')
                    print(f"  {factor}: F={stats['F']:.2f}, p={stats['p_value']:.4e}, "
                          f"η²p={stats.get('partial_eta_squared', 0):.3f} {sig}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    viz_dir = output_dir / 'visualizations'
    generate_visualizations(df, anova_results, importance_rankings, viz_dir)

    # Print summary
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}")

    for study, result in anova_results.items():
        print(f"\n{study}:")
        print(f"  Configs: {result['n_configs']}")
        print(f"  Observations: {result['n_observations']}")

        if 'oneway_anova' in result:
            anova = result['oneway_anova']
            sig = '***' if anova['p_value'] < 0.001 else ('**' if anova['p_value'] < 0.01 else ('*' if anova['p_value'] < 0.05 else ''))
            print(f"  One-way ANOVA: F={anova['f_statistic']:.2f}, p={anova['p_value']:.4f} {sig}")

        if 'effect_size' in result:
            es = result['effect_size']
            print(f"  Effect size: η²={es['eta_squared']:.3f} ({es['interpretation']})")

        if study in importance_rankings:
            rankings = importance_rankings[study]['rankings']
            if rankings:
                top3 = rankings[:3]
                print(f"  Most important (LOO):")
                for r in top3:
                    print(f"    - {r['item']}: {r['accuracy_drop']*100:.1f}% drop")

    print(f"\n{'='*70}")
    print(f"Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
