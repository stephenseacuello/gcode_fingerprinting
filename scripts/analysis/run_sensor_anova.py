#!/usr/bin/env python3
"""One-way ANOVA on per-file sensor statistics across 9 operation classes.

For each sensor channel, computes the mean and std across time rows per file,
then computes F = between-class variance / within-class variance across
the 9 operation classes.

Outputs:
  - Per-feature F-statistic ranking (CSV)
  - Per-modality summary (CSV)
  - Bar chart of median F by modality (PDF)
  - Heatmap of F by sensor × channel (PDF)

Usage:
    python scripts/analysis/run_sensor_anova.py \
        --data-dir data_clean/ \
        --output-dir outputs/experiments_2026_02_25/paper/figures/
"""
import argparse
import glob
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Constants ──

CLASS_ORDER = [
    'adaptive', 'adaptive150025', 'damageadaptive',
    'face', 'face150025', 'damageface',
    'pocket', 'pocket150025', 'damagepocket',
]

# Patterns matched in specificity order (longest prefix first)
CLASS_PATTERNS = [
    'adaptive150025', 'face150025', 'pocket150025',
    'damageadaptive', 'damageface', 'damagepocket',
    'adaptive', 'face', 'pocket',
]

CHANNEL_TO_MODALITY = {
    'Ax': 'Accelerometer', 'Ay': 'Accelerometer', 'Az': 'Accelerometer',
    'Gx': 'Gyroscope', 'Gy': 'Gyroscope', 'Gz': 'Gyroscope',
    'Mx': 'Magnetometer', 'My': 'Magnetometer', 'Mz': 'Magnetometer',
    'Pressure': 'Pressure', 'Temperature': 'Temperature',
    'Proximity': 'Proximity',
    'ColorR': 'Color', 'ColorG': 'Color', 'ColorB': 'Color', 'ColorA': 'Color',
    'RMS': 'RMS Audio',
}

ELECTRICAL_COLS = [
    'spindle', 'x_motor', 'y_motor', 'z_motor',
    'spindle_A', 'x_motor_A', 'y_motor_A', 'z_motor_A',
]

# Modalities removed in ablation (in order)
REMOVED_MODALITIES = {'Proximity', 'Pressure', 'Color', 'Magnetometer'}

MODALITY_ORDER = [
    'Pressure', 'Proximity', 'Color', 'Magnetometer',
    'Temperature', 'RMS Audio', 'Accelerometer', 'Gyroscope', 'Electrical',
]

# MDPI style
plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 8, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
})


def extract_class(filename):
    """Extract 9-class label from filename."""
    name = Path(filename).stem.replace('_aligned', '')
    for pat in CLASS_PATTERNS:
        if name.startswith(pat):
            return pat
    raise ValueError(f"Cannot extract class from: {filename}")


def get_modality(col):
    """Map column name to modality group."""
    if '.' in col:
        channel = col.split('.', 1)[1]
        return CHANNEL_TO_MODALITY.get(channel, 'Unknown')
    if col in ELECTRICAL_COLS:
        return 'Electrical'
    return 'Unknown'


def compute_eta_squared(groups):
    """Compute eta-squared (effect size) from grouped data."""
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_total = np.sum((all_data - grand_mean) ** 2)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    if ss_total == 0:
        return 0.0
    return ss_between / ss_total


def main():
    parser = argparse.ArgumentParser(description='One-way ANOVA on sensor features')
    parser.add_argument('--data-dir', type=Path, default=Path('data_clean'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('outputs/experiments_2026_02_25/paper/figures'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load all files ──
    csv_files = sorted(glob.glob(str(args.data_dir / '*_aligned.csv')))
    print(f"Found {len(csv_files)} CSV files")

    # Identify feature columns from first file
    df0 = pd.read_csv(csv_files[0])
    sensor_cols = [c for c in df0.columns if '.' in c]
    elec_cols = [c for c in ELECTRICAL_COLS if c in df0.columns]
    feature_cols = sensor_cols + elec_cols
    print(f"Features: {len(sensor_cols)} sensor + {len(elec_cols)} electrical = {len(feature_cols)}")

    # ── Step 2: Compute per-file mean and std ──
    records = []
    for f in csv_files:
        cls = extract_class(f)
        df = pd.read_csv(f)
        row = {'file': Path(f).stem, 'class': cls}
        for col in feature_cols:
            if col not in df.columns:
                row[f'{col}_mean'] = np.nan
                row[f'{col}_std'] = np.nan
                continue
            vals = df[col].dropna()
            row[f'{col}_mean'] = vals.mean() if len(vals) > 0 else np.nan
            row[f'{col}_std'] = vals.std() if len(vals) > 1 else 0.0
        records.append(row)

    summary_df = pd.DataFrame(records)
    print(f"Summary: {summary_df.shape[0]} files × {len(feature_cols)} features")

    # ── Step 3: One-way ANOVA per feature ──
    results = []
    for col in feature_cols:
        modality = get_modality(col)
        for stat_type in ['mean', 'std']:
            col_key = f'{col}_{stat_type}'
            groups = []
            for cls in CLASS_ORDER:
                vals = summary_df.loc[summary_df['class'] == cls, col_key].dropna().values
                if len(vals) > 0:
                    groups.append(vals)

            if len(groups) < 2:
                continue

            # Check if all values are identical (would cause F=0)
            all_vals = np.concatenate(groups)
            if np.std(all_vals) < 1e-12:
                f_stat, p_val, eta2 = 0.0, 1.0, 0.0
            else:
                f_stat, p_val = stats.f_oneway(*groups)
                eta2 = compute_eta_squared(groups)
                if np.isnan(f_stat):
                    f_stat, p_val, eta2 = 0.0, 1.0, 0.0

            results.append({
                'feature': col,
                'stat_type': stat_type,
                'modality': modality,
                'F_statistic': f_stat,
                'p_value': p_val,
                'eta_squared': eta2,
                'significant': p_val < 0.05,
            })

    results_df = pd.DataFrame(results)
    print(f"\nANOVA computed for {len(results_df)} feature×stat combinations")

    # ── Primary results: means ──
    means_df = results_df[results_df['stat_type'] == 'mean'].copy()
    means_df = means_df.sort_values('F_statistic', ascending=False).reset_index(drop=True)

    # Save full ranking
    means_df.to_csv(args.output_dir / 'anova_feature_ranking.csv', index=False)
    print(f"\nSaved: anova_feature_ranking.csv")

    # ── Step 4: Group by modality ──
    modality_stats = means_df.groupby('modality').agg(
        median_F=('F_statistic', 'median'),
        mean_F=('F_statistic', 'mean'),
        min_F=('F_statistic', 'min'),
        max_F=('F_statistic', 'max'),
        n_features=('F_statistic', 'count'),
        n_significant=('significant', 'sum'),
        median_eta2=('eta_squared', 'median'),
    ).reset_index()
    modality_stats = modality_stats.sort_values('median_F', ascending=False)
    modality_stats.to_csv(args.output_dir / 'anova_modality_summary.csv', index=False)
    print(f"Saved: anova_modality_summary.csv")

    # Print summary
    print(f"\n{'Modality':<16} {'Median F':>10} {'Mean F':>10} {'η²':>8} {'N feat':>7} {'N sig':>6}")
    print('-' * 60)
    for _, r in modality_stats.iterrows():
        print(f"{r['modality']:<16} {r['median_F']:>10.1f} {r['mean_F']:>10.1f} "
              f"{r['median_eta2']:>8.3f} {int(r['n_features']):>7} {int(r['n_significant']):>6}")

    print(f"\nTop 10 features (by F on means):")
    for _, r in means_df.head(10).iterrows():
        print(f"  {r['feature']:<30s} F={r['F_statistic']:>10.1f}  p={r['p_value']:.2e}  "
              f"η²={r['eta_squared']:.3f}  [{r['modality']}]")

    print(f"\nBottom 10 features:")
    for _, r in means_df.tail(10).iterrows():
        print(f"  {r['feature']:<30s} F={r['F_statistic']:>10.1f}  p={r['p_value']:.2e}  "
              f"η²={r['eta_squared']:.3f}  [{r['modality']}]")

    # ── Step 5: Generate figures ──

    # Figure 1: Bar chart — median F per modality
    fig, ax = plt.subplots(figsize=(10, 5))
    mod_order = [m for m in MODALITY_ORDER if m in modality_stats['modality'].values]
    mod_data = modality_stats.set_index('modality').loc[mod_order]

    colors = []
    for m in mod_order:
        if m in REMOVED_MODALITIES:
            colors.append('#E53935')  # red for removed
        else:
            colors.append('#1E88E5')  # blue for retained

    bars = ax.bar(range(len(mod_order)), mod_data['median_F'], color=colors, alpha=0.8,
                  edgecolor='white', linewidth=0.5)

    # Overlay individual features as scatter
    for i, mod in enumerate(mod_order):
        mod_features = means_df[means_df['modality'] == mod]['F_statistic'].values
        jitter = np.random.default_rng(42).uniform(-0.25, 0.25, len(mod_features))
        ax.scatter(i + jitter, mod_features, color='black', s=12, alpha=0.4, zorder=3)

    ax.set_xticks(range(len(mod_order)))
    ax.set_xticklabels(mod_order, rotation=35, ha='right')
    ax.set_ylabel('F-Statistic (one-way ANOVA on per-file means)')
    ax.set_title('Feature Discriminative Power by Modality Group')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E53935', alpha=0.8, label='Removed in ablation'),
        Patch(facecolor='#1E88E5', alpha=0.8, label='Retained in minimal config'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Add vertical separator
    n_removed = sum(1 for m in mod_order if m in REMOVED_MODALITIES)
    ax.axvline(x=n_removed - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    fig.tight_layout()
    out1 = args.output_dir / 'fig_anova_modality_fstats.pdf'
    fig.savefig(out1)
    plt.close(fig)
    print(f"\nSaved: {out1}")

    # Figure 2: Heatmap — F by sensor × channel
    sensors = sorted(set(c.split('.')[0] for c in sensor_cols))
    channels = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz',
                'Pressure', 'Temperature', 'Proximity',
                'ColorR', 'ColorG', 'ColorB', 'ColorA', 'RMS']

    heatmap_data = np.full((len(sensors), len(channels)), np.nan)
    f_lookup = means_df.set_index('feature')['F_statistic'].to_dict()

    for i, sensor in enumerate(sensors):
        for j, channel in enumerate(channels):
            col = f'{sensor}.{channel}'
            if col in f_lookup:
                heatmap_data[i, j] = f_lookup[col]

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd',
                   norm=mcolors.LogNorm(vmin=max(0.1, np.nanmin(heatmap_data[heatmap_data > 0])),
                                         vmax=np.nanmax(heatmap_data)))
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(sensors)))
    ax.set_yticklabels(sensors, fontsize=9)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Sensor Location')
    ax.set_title('ANOVA F-Statistic: Sensor Location × Channel Type')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('F-statistic (log scale)')

    # Add modality group separators
    group_boundaries = [3, 6, 9, 10, 11, 12, 16]  # after Az, Gz, Mz, Pres, Temp, Prox, ColorA
    for b in group_boundaries:
        ax.axvline(x=b - 0.5, color='white', linewidth=2)

    fig.tight_layout()
    out2 = args.output_dir / 'fig_anova_heatmap.pdf'
    fig.savefig(out2)
    plt.close(fig)
    print(f"Saved: {out2}")

    print("\nDone!")


if __name__ == '__main__':
    main()
