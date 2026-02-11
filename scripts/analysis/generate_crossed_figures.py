#!/usr/bin/env python3
"""Generate Missing Cross-Nested Ablation Figures for Encoder Ablation Paper.

Generates 8 figures analyzing the 4-level crossed factorial ablation design:
  1. L3 Sensor Pair Heatmap
  2. L4 Modality Combination Bar Chart
  3. Interaction Effect Magnitude Plot (L1 sensor x modality)
  4. Per-Class Accuracy Heatmap (L1)
  5. Minimal Configuration Path / Cost-Accuracy Frontier
  6. Cross-Level Comparison Figure
  7. Statistical Significance Heatmap (L1)
  8. Modality Requirement Staircase (L4)

Usage:
    python scripts/analysis/generate_crossed_figures.py
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from scipy import stats as scipy_stats

# Publication-quality settings (match existing paper figures)
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

MODALITIES = ['accelerometer', 'gyroscope', 'magnetometer', 'environmental', 'color', 'rms']

# Prettier display names
MODALITY_LABELS = {
    'accelerometer': 'Accel.',
    'gyroscope': 'Gyro.',
    'magnetometer': 'Mag.',
    'environmental': 'Env.',
    'color': 'Color',
    'rms': 'RMS',
}

CLASS_NAMES = [
    'adaptive', 'adaptive150025', 'face', 'face150025',
    'pocket', 'pocket150025', 'damageadaptive', 'damageface', 'damagepocket'
]

CLASS_LABELS = {
    'adaptive': 'Adaptive',
    'adaptive150025': 'Adaptive\n(150/025)',
    'face': 'Face',
    'face150025': 'Face\n(150/025)',
    'pocket': 'Pocket',
    'pocket150025': 'Pocket\n(150/025)',
    'damageadaptive': 'Damage\nAdaptive',
    'damageface': 'Damage\nFace',
    'damagepocket': 'Damage\nPocket',
}

# Sensor group mapping (16 real sensors verified against metadata.json)
SENSOR_GROUPS = {
    'frame_b1': 'Frame', 'frame_b2': 'Frame', 'frame_l1': 'Frame',
    'frame_l2': 'Frame', 'frame_l3': 'Frame', 'frame_r1': 'Frame', 'frame_r2': 'Frame',
    'spindle1': 'Spindle', 'spindle2': 'Spindle',
    'z_gant_1': 'Gantry', 'z_gant_2': 'Gantry',
    'y_bed_1': 'Bed', 'y_bed_2': 'Bed', 'y_bed_3': 'Bed', 'y_bed_4': 'Bed',
    'xa_motor': 'Motor',
}

# Individual channels for L5
ALL_CHANNELS = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz',
                'Pressure', 'Temperature', 'Proximity',
                'ColorR', 'ColorG', 'ColorB', 'ColorA', 'RMS']

CHANNEL_LABELS = {
    'Ax': 'Ax', 'Ay': 'Ay', 'Az': 'Az',
    'Gx': 'Gx', 'Gy': 'Gy', 'Gz': 'Gz',
    'Mx': 'Mx', 'My': 'My', 'Mz': 'Mz',
    'Pressure': 'Press.', 'Temperature': 'Temp.', 'Proximity': 'Prox.',
    'ColorR': 'R', 'ColorG': 'G', 'ColorB': 'B', 'ColorA': 'A',
    'RMS': 'RMS',
}

ALL_SENSORS = sorted(SENSOR_GROUPS.keys(), key=lambda s: (SENSOR_GROUPS[s], s))

BASE_DIR = Path(__file__).resolve().parent.parent.parent / 'outputs' / 'ablation_study_2026_02_06'
CROSSED_DIR = BASE_DIR / 'crossed'
OUTPUT_DIR = BASE_DIR / 'paper' / 'figures'


def parse_results(directory):
    """Parse all results.json files from experiment subdirectories."""
    results = []
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"  Directory not found: {directory}")
        return results

    for exp_dir in sorted(dir_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        results_file = exp_dir / 'results.json'
        if not results_file.exists():
            continue
        try:
            with open(results_file) as f:
                data = json.load(f)

            acc = None
            if 'checkpoint_info' in data:
                acc = data['checkpoint_info'].get('val_best_test_acc')
            if acc is None and 'val_best_test' in data:
                acc = data['val_best_test'].get('accuracy')
            if acc is None:
                continue

            # Per-class accuracies
            per_class = {}
            if 'val_best_test' in data and 'per_class' in data['val_best_test']:
                per_class = data['val_best_test']['per_class']

            config_name = exp_dir.name
            parts = config_name.rsplit('_ds', 1)
            base_config = parts[0] if len(parts) > 1 else config_name

            results.append({
                'config': base_config,
                'full_config': config_name,
                'test_accuracy': float(acc),
                'per_class': per_class,
            })
        except Exception as e:
            print(f"  Error parsing {results_file}: {e}")
    return results


def parse_sensor_modality(config):
    """Parse a L1 config name into (sensor, modality)."""
    for mod in MODALITIES:
        if config.endswith('_' + mod):
            sensor = config[:-(len(mod) + 1)]
            return sensor, mod
    return None, None


def aggregate_configs(results):
    """Group results by base config and compute mean/std."""
    grouped = defaultdict(list)
    grouped_perclass = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r['config']].append(r['test_accuracy'])
        for cls, val in r.get('per_class', {}).items():
            grouped_perclass[r['config']][cls].append(val)

    stats = {}
    for config, accs in grouped.items():
        stats[config] = {
            'mean': np.mean(accs),
            'std': np.std(accs),
            'n': len(accs),
            'per_class_mean': {
                cls: np.mean(vals) for cls, vals in grouped_perclass[config].items()
            },
        }
    return stats


def save_figure(fig, name):
    """Save figure as both PDF and PNG."""
    pdf_path = OUTPUT_DIR / f'{name}.pdf'
    png_path = OUTPUT_DIR / f'{name}.png'
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.pdf/png")


# ──────────────────────────────────────────────────────────────────────
# Figure 1: L3 Sensor Pair Heatmap
# ──────────────────────────────────────────────────────────────────────
def fig_sensor_pair_heatmap(l3_results):
    """16x16 symmetric heatmap of pairwise sensor combination accuracy."""
    print("\n[Fig 1] L3 Sensor Pair Heatmap")
    stats = aggregate_configs(l3_results)

    # Collect all sensors from pair configs
    all_sensors = set()
    pair_map = {}
    for config, st in stats.items():
        # Config like "frame_b1_spindle1" — need to split into two sensors
        # Sensors can have underscores, so we try all known sensors
        found = False
        for s1 in SENSOR_GROUPS:
            if config.startswith(s1 + '_'):
                remainder = config[len(s1) + 1:]
                if remainder in SENSOR_GROUPS:
                    all_sensors.add(s1)
                    all_sensors.add(remainder)
                    pair_map[(s1, remainder)] = st['mean']
                    pair_map[(remainder, s1)] = st['mean']
                    found = True
                    break
        if not found:
            # Try reverse: longer sensor names first
            sorted_sensors = sorted(SENSOR_GROUPS.keys(), key=len, reverse=True)
            for s1 in sorted_sensors:
                if config.startswith(s1 + '_'):
                    remainder = config[len(s1) + 1:]
                    for s2 in sorted_sensors:
                        if remainder == s2:
                            all_sensors.add(s1)
                            all_sensors.add(s2)
                            pair_map[(s1, s2)] = st['mean']
                            pair_map[(s2, s1)] = st['mean']
                            found = True
                            break
                    if found:
                        break

    sensors = sorted(all_sensors, key=lambda s: (SENSOR_GROUPS.get(s, 'ZZZ'), s))
    n = len(sensors)
    print(f"  Found {n} sensors, {len(pair_map)//2} unique pairs")

    matrix = np.full((n, n), np.nan)
    for i, s1 in enumerate(sensors):
        for j, s2 in enumerate(sensors):
            if (s1, s2) in pair_map:
                matrix[i, j] = pair_map[(s1, s2)] * 100

    # Group color bands
    group_list = [SENSOR_GROUPS.get(s, '?') for s in sensors]
    group_colors = {
        'Frame': '#e74c3c', 'Spindle': '#3498db', 'Gantry': '#2ecc71',
        'Bed': '#f39c12', 'Motor': '#9b59b6'
    }

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.eye(n, dtype=bool)  # mask diagonal (self-pairs don't exist)

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=np.nanmin(matrix[~mask]),
                   vmax=np.nanmax(matrix[~mask]), aspect='equal')
    # Mask diagonal
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    fill=True, facecolor='lightgray', edgecolor='white'))

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                color = 'white' if val < 70 else 'black'
                fontsize = 6 if n > 14 else 7
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=fontsize, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sensors, rotation=90, fontsize=7)
    ax.set_yticklabels(sensors, fontsize=7)

    # Color ticks by group
    for idx, s in enumerate(sensors):
        grp = SENSOR_GROUPS.get(s, '?')
        col = group_colors.get(grp, 'black')
        ax.get_xticklabels()[idx].set_color(col)
        ax.get_yticklabels()[idx].set_color(col)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mean Test Accuracy (%)', fontsize=9)

    ax.set_title('Sensor Pair Accuracy (L3 Crossed)', fontsize=12, pad=10)

    # Legend for groups
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, label=g) for g, c in group_colors.items()]
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.15, 1.0),
              fontsize=8, title='Sensor Group', title_fontsize=9)

    plt.tight_layout()
    save_figure(fig, 'fig_sensor_pair_heatmap')


# ──────────────────────────────────────────────────────────────────────
# Figure 2: L4 Modality Combination Bar Chart
# ──────────────────────────────────────────────────────────────────────
def fig_modality_combo_bars(l4_results):
    """Grouped bar chart comparing 2-modality and 3-modality combinations."""
    print("\n[Fig 2] L4 Modality Combination Bar Chart")
    stats = aggregate_configs(l4_results)

    keep2 = []
    keep3 = []
    for config, st in stats.items():
        if config.startswith('keep2_'):
            label = config[6:].replace('_', ' + ')
            keep2.append((label, st['mean'] * 100, st['std'] * 100))
        elif config.startswith('keep3_'):
            label = config[6:].replace('_', ' + ')
            keep3.append((label, st['mean'] * 100, st['std'] * 100))

    keep2.sort(key=lambda x: x[1], reverse=True)
    keep3.sort(key=lambda x: x[1], reverse=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [len(keep2), len(keep3)]})

    # --- Keep 2 panel ---
    ax = axes[0]
    labels2 = [k[0] for k in keep2]
    means2 = [k[1] for k in keep2]
    stds2 = [k[2] for k in keep2]
    y = np.arange(len(keep2))
    colors2 = plt.cm.RdYlGn(np.array(means2) / 100)
    bars = ax.barh(y, means2, xerr=stds2, capsize=2, color=colors2,
                   edgecolor='black', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels2, fontsize=8)
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('2-Modality Combinations', fontsize=11)
    ax.set_xlim(left=min(means2) - 10 if min(means2) > 20 else 0, right=105)
    ax.invert_yaxis()
    for bar, val in zip(bars, means2):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%',
                va='center', ha='left', fontsize=7)

    # --- Keep 3 panel ---
    ax = axes[1]
    labels3 = [k[0] for k in keep3]
    means3 = [k[1] for k in keep3]
    stds3 = [k[2] for k in keep3]
    y = np.arange(len(keep3))
    colors3 = plt.cm.RdYlGn(np.array(means3) / 100)
    bars = ax.barh(y, means3, xerr=stds3, capsize=2, color=colors3,
                   edgecolor='black', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels3, fontsize=8)
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('3-Modality Combinations', fontsize=11)
    ax.set_xlim(left=min(means3) - 10 if min(means3) > 20 else 0, right=105)
    ax.invert_yaxis()
    for bar, val in zip(bars, means3):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%',
                va='center', ha='left', fontsize=7)

    plt.tight_layout()
    save_figure(fig, 'fig_modality_combo_performance')


# ──────────────────────────────────────────────────────────────────────
# Figure 3: Interaction Effect Magnitude Plot
# ──────────────────────────────────────────────────────────────────────
def fig_interaction_effects(l1_results):
    """Heatmap showing the magnitude of sensor x modality interaction effects."""
    print("\n[Fig 3] Interaction Effect Magnitude Plot")
    stats = aggregate_configs(l1_results)

    # Build sensor x modality matrix
    sensors = set()
    modalities_found = set()
    cell_means = {}
    for config, st in stats.items():
        sensor, mod = parse_sensor_modality(config)
        if sensor and mod:
            sensors.add(sensor)
            modalities_found.add(mod)
            cell_means[(sensor, mod)] = st['mean']

    sensors = sorted(sensors, key=lambda s: (SENSOR_GROUPS.get(s, 'ZZZ'), s))
    mods = [m for m in MODALITIES if m in modalities_found]

    n_s, n_m = len(sensors), len(mods)

    # Build accuracy matrix
    acc_matrix = np.full((n_s, n_m), np.nan)
    for i, s in enumerate(sensors):
        for j, m in enumerate(mods):
            if (s, m) in cell_means:
                acc_matrix[i, j] = cell_means[(s, m)]

    # Compute marginal means
    sensor_means = np.nanmean(acc_matrix, axis=1)  # mean across modalities per sensor
    mod_means = np.nanmean(acc_matrix, axis=0)      # mean across sensors per modality
    grand_mean = np.nanmean(acc_matrix)

    # Interaction effect: cell - sensor_mean - mod_mean + grand_mean
    interaction = np.full_like(acc_matrix, np.nan)
    for i in range(n_s):
        for j in range(n_m):
            if not np.isnan(acc_matrix[i, j]):
                interaction[i, j] = acc_matrix[i, j] - sensor_means[i] - mod_means[j] + grand_mean

    interaction_pct = interaction * 100

    fig, ax = plt.subplots(figsize=(9, 8))

    # Symmetric colormap centered at 0
    vmax = np.nanmax(np.abs(interaction_pct))
    im = ax.imshow(interaction_pct, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    # Annotate
    for i in range(n_s):
        for j in range(n_m):
            if not np.isnan(interaction_pct[i, j]):
                val = interaction_pct[i, j]
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                        fontsize=7, color=color)

    ax.set_xticks(range(n_m))
    ax.set_yticks(range(n_s))
    ax.set_xticklabels([MODALITY_LABELS.get(m, m) for m in mods], fontsize=9)
    ax.set_yticklabels(sensors, fontsize=7)

    # Color y-tick labels by group
    group_colors = {
        'Frame': '#e74c3c', 'Spindle': '#3498db', 'Gantry': '#2ecc71',
        'Bed': '#f39c12', 'Motor': '#9b59b6'
    }
    for idx, s in enumerate(sensors):
        grp = SENSOR_GROUPS.get(s, '?')
        ax.get_yticklabels()[idx].set_color(group_colors.get(grp, 'black'))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Interaction Effect (pp)', fontsize=9)

    ax.set_title('Sensor $\\times$ Modality Interaction Effects (L1)', fontsize=12, pad=10)
    ax.set_xlabel('Modality')
    ax.set_ylabel('Sensor')

    plt.tight_layout()
    save_figure(fig, 'fig_interaction_effects')


# ──────────────────────────────────────────────────────────────────────
# Figure 4: Per-Class Accuracy Heatmap (L1)
# ──────────────────────────────────────────────────────────────────────
def fig_perclass_heatmap(l1_results):
    """Heatmap showing per-class accuracy for each modality, averaged across sensors."""
    print("\n[Fig 4] Per-Class Accuracy Heatmap (L1)")
    stats = aggregate_configs(l1_results)

    # Aggregate per-class accuracy by modality (average across sensors)
    mod_class_accs = defaultdict(lambda: defaultdict(list))
    for config, st in stats.items():
        sensor, mod = parse_sensor_modality(config)
        if mod and st.get('per_class_mean'):
            for cls, val in st['per_class_mean'].items():
                mod_class_accs[mod][cls].append(val)

    mods = [m for m in MODALITIES if m in mod_class_accs]
    classes = [c for c in CLASS_NAMES if c in mod_class_accs[mods[0]]]

    # Also aggregate by sensor group
    group_class_accs = defaultdict(lambda: defaultdict(list))
    for config, st in stats.items():
        sensor, mod = parse_sensor_modality(config)
        if sensor and st.get('per_class_mean'):
            grp = SENSOR_GROUPS.get(sensor, sensor)
            for cls, val in st['per_class_mean'].items():
                group_class_accs[grp][cls].append(val)

    groups = sorted(set(SENSOR_GROUPS.values()))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                              gridspec_kw={'width_ratios': [len(mods), len(groups)]})

    # --- Panel A: By Modality ---
    ax = axes[0]
    matrix_mod = np.full((len(classes), len(mods)), np.nan)
    for j, m in enumerate(mods):
        for i, c in enumerate(classes):
            vals = mod_class_accs[m].get(c, [])
            if vals:
                matrix_mod[i, j] = np.mean(vals) * 100

    im = ax.imshow(matrix_mod, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    for i in range(len(classes)):
        for j in range(len(mods)):
            if not np.isnan(matrix_mod[i, j]):
                val = matrix_mod[i, j]
                color = 'white' if val < 40 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8, color=color)

    ax.set_xticks(range(len(mods)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([MODALITY_LABELS.get(m, m) for m in mods], fontsize=9)
    ax.set_yticklabels([CLASS_LABELS.get(c, c) for c in classes], fontsize=8)
    ax.set_title('(a) Per-Class Accuracy by Modality', fontsize=11)
    ax.set_xlabel('Modality')
    ax.set_ylabel('Class')

    # --- Panel B: By Sensor Group ---
    ax = axes[1]
    matrix_grp = np.full((len(classes), len(groups)), np.nan)
    for j, g in enumerate(groups):
        for i, c in enumerate(classes):
            vals = group_class_accs[g].get(c, [])
            if vals:
                matrix_grp[i, j] = np.mean(vals) * 100

    im2 = ax.imshow(matrix_grp, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    for i in range(len(classes)):
        for j in range(len(groups)):
            if not np.isnan(matrix_grp[i, j]):
                val = matrix_grp[i, j]
                color = 'white' if val < 40 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8, color=color)

    ax.set_xticks(range(len(groups)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_yticklabels([CLASS_LABELS.get(c, c) for c in classes], fontsize=8)
    ax.set_title('(b) Per-Class Accuracy by Sensor Group', fontsize=11)
    ax.set_xlabel('Sensor Group')

    # Shared colorbar at the bottom
    cbar = fig.colorbar(im2, ax=axes, orientation='horizontal', shrink=0.6,
                        pad=0.15, aspect=40)
    cbar.set_label('Mean Accuracy (%)', fontsize=9)

    save_figure(fig, 'fig_perclass_heatmap')


# ──────────────────────────────────────────────────────────────────────
# Figure 5: Minimal Configuration Path / Cost-Accuracy Frontier
# ──────────────────────────────────────────────────────────────────────
def fig_cost_accuracy_frontier(l1_stats, l3_stats, l4_stats, l2_stats):
    """Progressive accuracy curve: best 1-sensor -> best pair -> best group -> all."""
    print("\n[Fig 5] Cost-Accuracy Frontier")

    # Gather best configs at each complexity level
    points = []

    # Level: single sensor + single modality (L1) — cheapest
    best_l1 = max(l1_stats.items(), key=lambda x: x[1]['mean'])
    points.append({
        'label': f'Best single\n({best_l1[0]})',
        'complexity': 1,
        'mean': best_l1[1]['mean'] * 100,
        'std': best_l1[1]['std'] * 100,
        'category': 'L1: Single sensor+modality',
    })

    # Top 5 L1 for scatter
    l1_sorted = sorted(l1_stats.items(), key=lambda x: x[1]['mean'], reverse=True)

    # Level: sensor pair (L3) — 2 sensors, all modalities
    best_l3 = max(l3_stats.items(), key=lambda x: x[1]['mean'])
    points.append({
        'label': f'Best pair\n({best_l3[0]})',
        'complexity': 2,
        'mean': best_l3[1]['mean'] * 100,
        'std': best_l3[1]['std'] * 100,
        'category': 'L3: Sensor pair',
    })

    # Level: sensor group (L2) — group of sensors, each modality
    best_l2 = max(l2_stats.items(), key=lambda x: x[1]['mean'])
    # Count sensors in best group
    best_group_name = best_l2[0].split('_')[0] if '_' in best_l2[0] else best_l2[0]
    group_sensor_count = sum(1 for s, g in SENSOR_GROUPS.items()
                            if g.lower() == best_group_name.lower())
    points.append({
        'label': f'Best group\n({best_l2[0]})',
        'complexity': max(3, group_sensor_count),
        'mean': best_l2[1]['mean'] * 100,
        'std': best_l2[1]['std'] * 100,
        'category': 'L2: Sensor group',
    })

    # Level: 2-modality combo (L4)
    keep2_stats = {k: v for k, v in l4_stats.items() if k.startswith('keep2_')}
    if keep2_stats:
        best_k2 = max(keep2_stats.items(), key=lambda x: x[1]['mean'])
        points.append({
            'label': f'Best 2-mod\n({best_k2[0][6:]})',
            'complexity': 2,
            'mean': best_k2[1]['mean'] * 100,
            'std': best_k2[1]['std'] * 100,
            'category': 'L4: 2-modality combo',
        })

    # Level: 3-modality combo (L4)
    keep3_stats = {k: v for k, v in l4_stats.items() if k.startswith('keep3_')}
    if keep3_stats:
        best_k3 = max(keep3_stats.items(), key=lambda x: x[1]['mean'])
        points.append({
            'label': f'Best 3-mod\n({best_k3[0][6:]})',
            'complexity': 3,
            'mean': best_k3[1]['mean'] * 100,
            'std': best_k3[1]['std'] * 100,
            'category': 'L4: 3-modality combo',
        })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel A: Sensor scaling (L1 -> L3 -> L2/all) ---
    ax = axes[0]
    cat_colors = {
        'L1: Single sensor+modality': '#e74c3c',
        'L3: Sensor pair': '#3498db',
        'L2: Sensor group': '#2ecc71',
    }

    # Scatter all L1 configs
    l1_means = [s['mean'] * 100 for _, s in l1_sorted]
    ax.scatter([1] * len(l1_means), l1_means, alpha=0.3, s=15, c='#e74c3c',
               zorder=1, label='All L1 configs')

    # Scatter all L3 configs
    l3_sorted = sorted(l3_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    l3_means = [s['mean'] * 100 for _, s in l3_sorted]
    ax.scatter([2] * len(l3_means), l3_means, alpha=0.3, s=15, c='#3498db',
               zorder=1, label='All L3 configs')

    # Scatter all L2 configs
    l2_sorted = sorted(l2_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    l2_means = [s['mean'] * 100 for _, s in l2_sorted]
    # Estimate sensor count per L2 config
    l2_complexities = []
    for cfg, _ in l2_sorted:
        grp_name = cfg.split('_')[0] if '_' in cfg else cfg
        cnt = sum(1 for s, g in SENSOR_GROUPS.items()
                  if g.lower() == grp_name.lower())
        l2_complexities.append(max(3, cnt))
    ax.scatter(l2_complexities, l2_means, alpha=0.3, s=15, c='#2ecc71',
               zorder=1, label='All L2 configs')

    # Connect best path
    sensor_points = [p for p in points if p['category'] in cat_colors]
    sensor_points.sort(key=lambda p: p['complexity'])
    path_x = [p['complexity'] for p in sensor_points]
    path_y = [p['mean'] for p in sensor_points]
    ax.plot(path_x, path_y, 'k--', linewidth=1.5, alpha=0.7, zorder=3)
    for p in sensor_points:
        ax.errorbar(p['complexity'], p['mean'], yerr=p['std'], fmt='o', markersize=8,
                    color=cat_colors[p['category']], zorder=4, capsize=4,
                    markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel('Number of Sensors')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('(a) Accuracy vs. Sensor Count', fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    # --- Panel B: Modality scaling (L4) ---
    ax = axes[1]
    mod_colors = {'L4: 2-modality combo': '#f39c12', 'L4: 3-modality combo': '#9b59b6'}

    # Scatter all keep2
    k2_means = [s['mean'] * 100 for _, s in keep2_stats.items()]
    ax.scatter([2] * len(k2_means), k2_means, alpha=0.4, s=20, c='#f39c12',
               zorder=1, label='All 2-mod combos')

    # Scatter all keep3
    k3_means = [s['mean'] * 100 for _, s in keep3_stats.items()]
    ax.scatter([3] * len(k3_means), k3_means, alpha=0.4, s=20, c='#9b59b6',
               zorder=1, label='All 3-mod combos')

    # Add baseline (all 6 modalities) reference
    ax.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.7,
               label='Baseline (all 6 modalities)')

    # Best path
    mod_points = [p for p in points if p['category'] in mod_colors]
    mod_points.sort(key=lambda p: p['complexity'])
    if mod_points:
        path_x = [p['complexity'] for p in mod_points]
        path_y = [p['mean'] for p in mod_points]
        ax.plot(path_x + [6], path_y + [100], 'k--', linewidth=1.5, alpha=0.7, zorder=3)
        for p in mod_points:
            ax.errorbar(p['complexity'], p['mean'], yerr=p['std'], fmt='o', markersize=8,
                        color=mod_colors[p['category']], zorder=4, capsize=4,
                        markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel('Number of Modalities')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('(b) Accuracy vs. Modality Count', fontsize=11)
    ax.set_xticks([2, 3, 6])
    ax.set_xticklabels(['2', '3', '6 (all)'])
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'fig_cost_accuracy_frontier')


# ──────────────────────────────────────────────────────────────────────
# Figure 6: Cross-Level Comparison
# ──────────────────────────────────────────────────────────────────────
def fig_cross_level_comparison(l1_stats, l2_stats, l3_stats, l4_stats):
    """Box/violin plot comparing accuracy distributions across L1-L4."""
    print("\n[Fig 6] Cross-Level Comparison")

    levels = {
        'L1\nSensor$\\times$Modality': l1_stats,
        'L2\nGroup$\\times$Modality': l2_stats,
        'L3\nSensor Pairs': l3_stats,
        'L4\nModality Combos': l4_stats,
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    data_list = []
    labels = []
    positions = []
    colors_list = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    for idx, (label, stats_dict) in enumerate(levels.items()):
        means = [s['mean'] * 100 for s in stats_dict.values()]
        data_list.append(means)
        labels.append(label)
        positions.append(idx)

    parts = ax.violinplot(data_list, positions=positions, showmedians=False,
                          showextrema=False)
    for idx, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_list[idx])
        pc.set_alpha(0.3)

    # Overlay box plots
    bp = ax.boxplot(data_list, positions=positions, widths=0.3,
                    patch_artist=True, showfliers=True,
                    flierprops={'markersize': 3, 'alpha': 0.5})
    for idx, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors_list[idx])
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')

    # Stats annotations
    for idx, (label, stats_dict) in enumerate(levels.items()):
        means = [s['mean'] * 100 for s in stats_dict.values()]
        ax.text(idx, max(means) + 2, f'n={len(means)}\n$\\mu$={np.mean(means):.1f}%',
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Mean Test Accuracy (%)')
    ax.set_title('Accuracy Distribution Across Ablation Levels', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'fig_cross_level_comparison')


# ──────────────────────────────────────────────────────────────────────
# Figure 7: Statistical Significance Heatmap (L1)
# ──────────────────────────────────────────────────────────────────────
def fig_significance_heatmap(l1_results):
    """Heatmap of p-values from t-tests comparing each L1 cell to grand mean."""
    print("\n[Fig 7] Statistical Significance Heatmap")

    # Group raw accuracies by (sensor, modality)
    cell_accs = defaultdict(list)
    for r in l1_results:
        sensor, mod = parse_sensor_modality(r['config'])
        if sensor and mod:
            cell_accs[(sensor, mod)].append(r['test_accuracy'])

    sensors = sorted(set(k[0] for k in cell_accs), key=lambda s: (SENSOR_GROUPS.get(s, 'ZZZ'), s))
    mods = [m for m in MODALITIES if any(k[1] == m for k in cell_accs)]

    # Grand mean and all accuracies
    all_accs = [a for accs in cell_accs.values() for a in accs]
    grand_mean = np.mean(all_accs)

    n_s, n_m = len(sensors), len(mods)
    pval_matrix = np.full((n_s, n_m), np.nan)
    effect_matrix = np.full((n_s, n_m), np.nan)  # Cohen's d

    for i, s in enumerate(sensors):
        for j, m in enumerate(mods):
            accs = cell_accs.get((s, m), [])
            if len(accs) >= 2:
                # One-sample t-test vs grand mean
                t_stat, p_val = scipy_stats.ttest_1samp(accs, grand_mean)
                pval_matrix[i, j] = p_val
                # Cohen's d
                d = (np.mean(accs) - grand_mean) / np.std(accs) if np.std(accs) > 0 else 0
                effect_matrix[i, j] = d

    # Transform p-values: -log10(p) signed by direction
    sig_matrix = np.full_like(pval_matrix, np.nan)
    for i in range(n_s):
        for j in range(n_m):
            if not np.isnan(pval_matrix[i, j]):
                log_p = -np.log10(max(pval_matrix[i, j], 1e-20))
                sign = np.sign(effect_matrix[i, j])
                sig_matrix[i, j] = sign * log_p

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- Panel A: Signed significance ---
    ax = axes[0]
    vmax = np.nanmax(np.abs(sig_matrix))
    im = ax.imshow(sig_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    # Mark significant cells
    for i in range(n_s):
        for j in range(n_m):
            if not np.isnan(pval_matrix[i, j]):
                p = pval_matrix[i, j]
                stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                color = 'white' if abs(sig_matrix[i, j]) > vmax * 0.5 else 'black'
                ax.text(j, i, stars, ha='center', va='center', fontsize=7,
                        color=color, fontweight='bold')

    ax.set_xticks(range(n_m))
    ax.set_yticks(range(n_s))
    ax.set_xticklabels([MODALITY_LABELS.get(m, m) for m in mods], fontsize=9)
    ax.set_yticklabels(sensors, fontsize=7)
    ax.set_title('(a) Statistical Significance\n($-\\mathrm{sign} \\cdot \\log_{10}(p)$)', fontsize=11)
    ax.set_xlabel('Modality')
    ax.set_ylabel('Sensor')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Signed $-\\log_{10}(p)$', fontsize=9)

    # Color y-tick labels by group
    group_colors = {
        'Frame': '#e74c3c', 'Spindle': '#3498db', 'Gantry': '#2ecc71',
        'Bed': '#f39c12', 'Motor': '#9b59b6'
    }
    for idx, s in enumerate(sensors):
        grp = SENSOR_GROUPS.get(s, '?')
        ax.get_yticklabels()[idx].set_color(group_colors.get(grp, 'black'))

    # --- Panel B: Effect size (Cohen's d) ---
    ax = axes[1]
    vmax_d = np.nanmax(np.abs(effect_matrix))
    im2 = ax.imshow(effect_matrix, cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d, aspect='auto')

    for i in range(n_s):
        for j in range(n_m):
            if not np.isnan(effect_matrix[i, j]):
                val = effect_matrix[i, j]
                color = 'white' if abs(val) > vmax_d * 0.5 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=7, color=color)

    ax.set_xticks(range(n_m))
    ax.set_yticks(range(n_s))
    ax.set_xticklabels([MODALITY_LABELS.get(m, m) for m in mods], fontsize=9)
    ax.set_yticklabels(sensors, fontsize=7)
    ax.set_title("(b) Effect Size (Cohen's $d$)", fontsize=11)
    ax.set_xlabel('Modality')

    cbar2 = fig.colorbar(im2, ax=ax, shrink=0.8, pad=0.02)
    cbar2.set_label("Cohen's $d$ (vs. grand mean)", fontsize=9)

    for idx, s in enumerate(sensors):
        grp = SENSOR_GROUPS.get(s, '?')
        ax.get_yticklabels()[idx].set_color(group_colors.get(grp, 'black'))

    plt.tight_layout()
    save_figure(fig, 'fig_significance_heatmap')


# ──────────────────────────────────────────────────────────────────────
# Figure 8: Modality Requirement Staircase
# ──────────────────────────────────────────────────────────────────────
def fig_modality_staircase(l4_stats, l1_stats):
    """Accuracy vs number of modalities — staircase/efficiency frontier."""
    print("\n[Fig 8] Modality Requirement Staircase")

    # Gather single-modality results from L1 (averaged across all sensors)
    single_mod_accs = defaultdict(list)
    for config, st in l1_stats.items():
        _, mod = parse_sensor_modality(config)
        if mod:
            single_mod_accs[mod].append(st['mean'])

    single_points = []
    for mod, accs in single_mod_accs.items():
        single_points.append({
            'label': MODALITY_LABELS.get(mod, mod),
            'n_modalities': 1,
            'mean': np.mean(accs) * 100,
            'std': np.std(accs) * 100,
            'config': mod,
        })

    # Keep2 and Keep3 from L4
    k2_points = []
    k3_points = []
    for config, st in l4_stats.items():
        if config.startswith('keep2_'):
            k2_points.append({
                'label': config[6:],
                'n_modalities': 2,
                'mean': st['mean'] * 100,
                'std': st['std'] * 100,
            })
        elif config.startswith('keep3_'):
            k3_points.append({
                'label': config[6:],
                'n_modalities': 3,
                'mean': st['mean'] * 100,
                'std': st['std'] * 100,
            })

    fig, ax = plt.subplots(figsize=(10, 5))

    # --- Scatter all points ---
    # Single modalities
    s1_means = [p['mean'] for p in single_points]
    ax.scatter([1] * len(s1_means), s1_means, c='#e74c3c', s=40, alpha=0.7,
               zorder=2, label='Single modality (L1 avg.)')

    # 2-modality
    k2_means = [p['mean'] for p in k2_points]
    ax.scatter([2] * len(k2_means), k2_means, c='#f39c12', s=40, alpha=0.7,
               zorder=2, label='2-modality combo (L4)')

    # 3-modality
    k3_means = [p['mean'] for p in k3_points]
    ax.scatter([3] * len(k3_means), k3_means, c='#9b59b6', s=40, alpha=0.7,
               zorder=2, label='3-modality combo (L4)')

    # 6-modality baseline
    ax.scatter([6], [100], c='#2ecc71', s=100, marker='D', zorder=3,
               label='All 6 modalities (baseline)')

    # --- Efficiency frontier (best at each level) ---
    frontier_x = [1, 2, 3, 6]
    frontier_y = [
        max(s1_means) if s1_means else 0,
        max(k2_means) if k2_means else 0,
        max(k3_means) if k3_means else 0,
        100,
    ]
    ax.plot(frontier_x, frontier_y, 'k-o', linewidth=2, markersize=6, zorder=4,
            alpha=0.8, label='Best at each level')

    # Annotate best configs
    if single_points:
        best_s1 = max(single_points, key=lambda p: p['mean'])
        ax.annotate(best_s1['label'], xy=(1, best_s1['mean']),
                    xytext=(1.15, best_s1['mean'] - 3), fontsize=8, ha='left',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    if k2_points:
        best_k2 = max(k2_points, key=lambda p: p['mean'])
        ax.annotate(best_k2['label'].replace('_', '+'), xy=(2, best_k2['mean']),
                    xytext=(2.15, best_k2['mean'] - 3), fontsize=8, ha='left',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    if k3_points:
        best_k3 = max(k3_points, key=lambda p: p['mean'])
        ax.annotate(best_k3['label'].replace('_', '+'), xy=(3, best_k3['mean']),
                    xytext=(3.15, best_k3['mean'] - 3), fontsize=8, ha='left',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    # Threshold lines
    for thresh, color, ls in [(95, '#e67e22', ':'), (99, '#27ae60', '--')]:
        ax.axhline(y=thresh, color=color, linestyle=ls, linewidth=1, alpha=0.6)
        ax.text(6.2, thresh, f'{thresh}%', fontsize=8, va='center', color=color)

    ax.set_xlabel('Number of Modalities')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Modality Requirement Staircase', fontsize=12)
    ax.set_xticks([1, 2, 3, 6])
    ax.set_xticklabels(['1', '2', '3', '6 (all)'])
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 6.8)

    plt.tight_layout()
    save_figure(fig, 'fig_modality_staircase')


# ──────────────────────────────────────────────────────────────────────
# Figure 9: L5 Sensor × Channel Heatmap (16 × 17)
# ──────────────────────────────────────────────────────────────────────
def parse_sensor_channel(config):
    """Parse an L5 config name into (sensor, channel) or (None, electrical_name)."""
    if config.startswith('electrical_'):
        return None, config
    # Try matching known sensors (longest first to avoid partial matches)
    for sensor in sorted(SENSOR_GROUPS.keys(), key=len, reverse=True):
        if config.startswith(sensor + '_'):
            channel = config[len(sensor) + 1:]
            if channel in ALL_CHANNELS:
                return sensor, channel
    return None, None


def fig_sensor_channel_heatmap(l5_results):
    """16×17 heatmap of individual sensor-channel pair accuracy."""
    print("\n[Fig 9] L5 Sensor × Channel Heatmap")
    stats = aggregate_configs(l5_results)

    sensors = ALL_SENSORS
    channels = ALL_CHANNELS
    n_s, n_c = len(sensors), len(channels)

    # Build accuracy matrix
    acc_matrix = np.full((n_s, n_c), np.nan)
    for config, st in stats.items():
        sensor, channel = parse_sensor_channel(config)
        if sensor and channel and sensor in sensors and channel in channels:
            i = sensors.index(sensor)
            j = channels.index(channel)
            acc_matrix[i, j] = st['mean'] * 100

    fig, ax = plt.subplots(figsize=(14, 9))

    # Use a diverging colormap anchored at chance level (11.1%)
    im = ax.imshow(acc_matrix, cmap='RdYlGn', vmin=0,
                   vmax=max(60, np.nanmax(acc_matrix)), aspect='auto')

    # Annotate cells
    for i in range(n_s):
        for j in range(n_c):
            if not np.isnan(acc_matrix[i, j]):
                val = acc_matrix[i, j]
                color = 'white' if val < 20 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=6, color=color)

    ax.set_xticks(range(n_c))
    ax.set_yticks(range(n_s))
    ax.set_xticklabels([CHANNEL_LABELS.get(c, c) for c in channels], fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(sensors, fontsize=7)

    # Color y-tick labels by sensor group
    group_colors = {
        'Frame': '#e74c3c', 'Spindle': '#3498db', 'Gantry': '#2ecc71',
        'Bed': '#f39c12', 'Motor': '#9b59b6'
    }
    for idx, s in enumerate(sensors):
        grp = SENSOR_GROUPS.get(s, '?')
        ax.get_yticklabels()[idx].set_color(group_colors.get(grp, 'black'))

    # Add channel group separators
    group_boundaries = [3, 6, 9, 12, 16]  # After Az, Gz, Mz, Prox, ColorA
    for b in group_boundaries:
        ax.axvline(x=b - 0.5, color='white', linewidth=2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mean Test Accuracy (%)', fontsize=9)

    ax.set_title('Sensor $\\times$ Individual Channel Accuracy (L5)', fontsize=12, pad=10)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Sensor')

    # Add marginal means as side annotations
    sensor_means = np.nanmean(acc_matrix, axis=1)
    channel_means = np.nanmean(acc_matrix, axis=0)

    # Sensor marginal means on right
    for i, mean_val in enumerate(sensor_means):
        if not np.isnan(mean_val):
            ax.text(n_c + 0.3, i, f'{mean_val:.1f}', ha='left', va='center',
                    fontsize=7, color='gray')

    # Channel marginal means on bottom
    for j, mean_val in enumerate(channel_means):
        if not np.isnan(mean_val):
            ax.text(j, n_s + 0.3, f'{mean_val:.0f}', ha='center', va='top',
                    fontsize=6, color='gray', rotation=45)

    plt.tight_layout()
    save_figure(fig, 'fig_sensor_channel_heatmap_L5')

    return acc_matrix, sensors, channels


# ──────────────────────────────────────────────────────────────────────
# Figure 10: L5 Top-K Ranking + Electrical Comparison
# ──────────────────────────────────────────────────────────────────────
def fig_l5_top_k_ranking(l5_results):
    """Top-20 sensor-channel pairs + electrical channels bar chart."""
    print("\n[Fig 10] L5 Top-K Ranking + Electrical")
    stats = aggregate_configs(l5_results)

    # Separate sensor×channel from electrical
    sensor_channel_stats = {}
    electrical_stats = {}
    for config, st in stats.items():
        if config.startswith('electrical_'):
            electrical_stats[config] = st
        else:
            sensor_channel_stats[config] = st

    # Sort by mean accuracy
    sorted_sc = sorted(sensor_channel_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    sorted_elec = sorted(electrical_stats.items(), key=lambda x: x[1]['mean'], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                              gridspec_kw={'width_ratios': [3, 1]})

    # --- Panel A: Top-20 sensor×channel pairs ---
    ax = axes[0]
    top_k = 20
    top_configs = sorted_sc[:top_k]
    labels = [c[0].replace('_', ' ') for c in top_configs]
    means = [c[1]['mean'] * 100 for c in top_configs]
    stds = [c[1]['std'] * 100 for c in top_configs]
    n_samples = [c[1]['n'] for c in top_configs]

    y_pos = np.arange(len(top_configs))
    colors = []
    for config, _ in top_configs:
        sensor, channel = parse_sensor_channel(config)
        grp = SENSOR_GROUPS.get(sensor, '?') if sensor else '?'
        group_colors = {
            'Frame': '#e74c3c', 'Spindle': '#3498db', 'Gantry': '#2ecc71',
            'Bed': '#f39c12', 'Motor': '#9b59b6'
        }
        colors.append(group_colors.get(grp, 'gray'))

    bars = ax.barh(y_pos, means, xerr=stds, capsize=3, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title(f'(a) Top-{top_k} Sensor$\\times$Channel Pairs', fontsize=11)
    ax.invert_yaxis()

    for i, (v, s, n) in enumerate(zip(means, stds, n_samples)):
        ax.text(v + s + 0.5, i, f'{v:.1f}% (n={n})', va='center', fontsize=7)

    # Chance level line
    ax.axvline(x=11.1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(11.5, len(top_configs) - 0.5, 'Chance', fontsize=7, color='red')

    # Legend for sensor groups
    from matplotlib.patches import Patch
    grp_colors = {'Frame': '#e74c3c', 'Spindle': '#3498db', 'Gantry': '#2ecc71',
                  'Bed': '#f39c12', 'Motor': '#9b59b6'}
    legend_patches = [Patch(facecolor=c, label=g) for g, c in grp_colors.items()]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8, title='Group')

    # --- Panel B: Electrical channels ---
    ax = axes[1]
    if sorted_elec:
        elec_labels = [c[0].replace('electrical_', '').replace('_', ' ') for c in sorted_elec]
        elec_means = [c[1]['mean'] * 100 for c in sorted_elec]
        elec_stds = [c[1]['std'] * 100 for c in sorted_elec]
        y_pos_e = np.arange(len(sorted_elec))

        ax.barh(y_pos_e, elec_means, xerr=elec_stds, capsize=3,
                color='#1abc9c', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos_e)
        ax.set_yticklabels(elec_labels, fontsize=8)
        ax.set_xlabel('Test Accuracy (%)')
        ax.set_title('(b) Electrical Channels', fontsize=11)
        ax.invert_yaxis()

        for i, (v, s) in enumerate(zip(elec_means, elec_stds)):
            ax.text(v + s + 0.5, i, f'{v:.1f}%', va='center', fontsize=7)

        ax.axvline(x=11.1, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    save_figure(fig, 'fig_L5_top_k_ranking')

    # Save CSV of full ranking
    ranking_csv = OUTPUT_DIR / 'L5_full_ranking.csv'
    with open(ranking_csv, 'w') as f:
        f.write('rank,config,sensor,channel,mean_accuracy,std_accuracy,n_seeds\n')
        for rank, (config, st) in enumerate(sorted_sc + sorted_elec, 1):
            sensor, channel = parse_sensor_channel(config)
            f.write(f"{rank},{config},{sensor or 'electrical'},{channel},"
                    f"{st['mean']:.6f},{st['std']:.6f},{st['n']}\n")
    print(f"  Saved ranking CSV: {ranking_csv}")


# ──────────────────────────────────────────────────────────────────────
# Figure 11: L5 Channel and Sensor Marginal Means
# ──────────────────────────────────────────────────────────────────────
def fig_l5_marginal_means(l5_results):
    """Bar charts of marginal means: channels averaged over sensors and vice versa."""
    print("\n[Fig 11] L5 Marginal Means")
    stats = aggregate_configs(l5_results)

    # Collect sensor×channel means (exclude electrical)
    sensor_accs = defaultdict(list)
    channel_accs = defaultdict(list)

    for config, st in stats.items():
        sensor, channel = parse_sensor_channel(config)
        if sensor and channel and sensor in SENSOR_GROUPS and channel in ALL_CHANNELS:
            sensor_accs[sensor].append(st['mean'] * 100)
            channel_accs[channel].append(st['mean'] * 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel A: Channel marginal means (avg over sensors) ---
    ax = axes[0]
    channel_means = {c: np.mean(accs) for c, accs in channel_accs.items()}
    channel_stds = {c: np.std(accs) for c, accs in channel_accs.items()}
    sorted_channels = sorted(channel_means.items(), key=lambda x: x[1], reverse=True)

    labels = [CHANNEL_LABELS.get(c, c) for c, _ in sorted_channels]
    means = [m for _, m in sorted_channels]
    stds = [channel_stds[c] for c, _ in sorted_channels]

    # Color by channel group
    ch_group_colors = {
        'Ax': '#e74c3c', 'Ay': '#e74c3c', 'Az': '#e74c3c',
        'Gx': '#3498db', 'Gy': '#3498db', 'Gz': '#3498db',
        'Mx': '#2ecc71', 'My': '#2ecc71', 'Mz': '#2ecc71',
        'Pressure': '#f39c12', 'Temperature': '#f39c12', 'Proximity': '#f39c12',
        'ColorR': '#9b59b6', 'ColorG': '#9b59b6', 'ColorB': '#9b59b6', 'ColorA': '#9b59b6',
        'RMS': '#1abc9c',
    }
    colors = [ch_group_colors.get(c, 'gray') for c, _ in sorted_channels]

    y_pos = np.arange(len(sorted_channels))
    ax.barh(y_pos, means, xerr=stds, capsize=3, color=colors, alpha=0.8,
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Mean Accuracy (%) Across All Sensors')
    ax.set_title('(a) Channel Marginal Means', fontsize=11)
    ax.invert_yaxis()
    ax.axvline(x=11.1, color='red', linestyle='--', linewidth=1, alpha=0.5)

    for i, (v, s) in enumerate(zip(means, stds)):
        ax.text(v + s + 0.5, i, f'{v:.1f}%', va='center', fontsize=7)

    # --- Panel B: Sensor marginal means (avg over channels) ---
    ax = axes[1]
    sensor_means = {s: np.mean(accs) for s, accs in sensor_accs.items()}
    sensor_stds = {s: np.std(accs) for s, accs in sensor_accs.items()}
    sorted_sensors = sorted(sensor_means.items(), key=lambda x: x[1], reverse=True)

    labels = [s for s, _ in sorted_sensors]
    means = [m for _, m in sorted_sensors]
    stds = [sensor_stds[s] for s, _ in sorted_sensors]

    grp_colors = {'Frame': '#e74c3c', 'Spindle': '#3498db', 'Gantry': '#2ecc71',
                  'Bed': '#f39c12', 'Motor': '#9b59b6'}
    colors = [grp_colors.get(SENSOR_GROUPS.get(s, '?'), 'gray') for s, _ in sorted_sensors]

    y_pos = np.arange(len(sorted_sensors))
    ax.barh(y_pos, means, xerr=stds, capsize=3, color=colors, alpha=0.8,
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Mean Accuracy (%) Across All Channels')
    ax.set_title('(b) Sensor Marginal Means', fontsize=11)
    ax.invert_yaxis()
    ax.axvline(x=11.1, color='red', linestyle='--', linewidth=1, alpha=0.5)

    for i, (v, s) in enumerate(zip(means, stds)):
        ax.text(v + s + 0.5, i, f'{v:.1f}%', va='center', fontsize=7)

    plt.tight_layout()
    save_figure(fig, 'fig_L5_marginal_means')


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading crossed factorial results...")
    print(f"  Base dir: {BASE_DIR}")
    print(f"  Output:   {OUTPUT_DIR}")

    # Parse all levels
    print("\nParsing L1: Sensor x Modality...")
    l1_results = parse_results(CROSSED_DIR / 'L1_sensor_modality')
    print(f"  {len(l1_results)} results loaded")

    print("Parsing L2: Group x Modality...")
    l2_results = parse_results(CROSSED_DIR / 'L2_group_modality')
    print(f"  {len(l2_results)} results loaded")

    print("Parsing L3: Sensor Pairs...")
    l3_results = parse_results(CROSSED_DIR / 'L3_sensor_pairs')
    print(f"  {len(l3_results)} results loaded")

    print("Parsing L4: Modality Combos...")
    l4_results = parse_results(CROSSED_DIR / 'L4_modality_combos')
    print(f"  {len(l4_results)} results loaded")

    print("Parsing L5: Sensor x Channel...")
    l5_results = parse_results(CROSSED_DIR / 'L5_sensor_channel')
    print(f"  {len(l5_results)} results loaded")

    # Aggregate stats
    l1_stats = aggregate_configs(l1_results)
    l2_stats = aggregate_configs(l2_results)
    l3_stats = aggregate_configs(l3_results)
    l4_stats = aggregate_configs(l4_results)
    l5_stats = aggregate_configs(l5_results)

    print(f"\nUnique configs: L1={len(l1_stats)}, L2={len(l2_stats)}, "
          f"L3={len(l3_stats)}, L4={len(l4_stats)}, L5={len(l5_stats)}")

    # Generate all figures
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    # L1-L4 figures (original 8)
    fig_sensor_pair_heatmap(l3_results)
    fig_modality_combo_bars(l4_results)
    fig_interaction_effects(l1_results)
    fig_perclass_heatmap(l1_results)
    fig_cost_accuracy_frontier(l1_stats, l3_stats, l4_stats, l2_stats)
    fig_cross_level_comparison(l1_stats, l2_stats, l3_stats, l4_stats)
    fig_significance_heatmap(l1_results)
    fig_modality_staircase(l4_stats, l1_stats)

    # L5 figures (new 3)
    if l5_results:
        fig_sensor_channel_heatmap(l5_results)
        fig_l5_top_k_ranking(l5_results)
        fig_l5_marginal_means(l5_results)
    else:
        print("\n  [SKIP] No L5 results found - L5 figures will be generated after experiments complete")

    print("\n" + "=" * 60)
    print("DONE — All figures saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
