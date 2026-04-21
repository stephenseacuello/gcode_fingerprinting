#!/usr/bin/env python3
"""
Generate cross-validation results: heatmaps, tables, line plots, and per-class breakdown.

Reads individual fold results directly (no cv_summary.json required) and produces:
  1. Summary heatmap  — all configs × all models (mean accuracy, color-coded)
  2. Feature ablation table — 5 feature configs at default window/stride
  3. Window/stride ablation tables — all window/stride combos at full AND minimal features
  4. Line plots — accuracy vs window size per model, for full and minimal feature configs
  5. Comparison plot — side-by-side full vs minimal features across temporal resolutions
  6. Degradation plot — accuracy drop (full → minimal) per model per window
  7. Per-class breakdowns — bar charts for best full-feature and minimal-feature configs
  8. LaTeX tables — publication-ready versions of Tables 1, 2, & 3

Usage:
    python3 scripts/analysis/generate_cv_results.py \
        --experiments-dir outputs/experiments_2026_02_24 \
        --output-dir outputs/cv_results_figures

    # Only generate specific outputs:
    python3 scripts/analysis/generate_cv_results.py \
        --experiments-dir outputs/experiments_2026_02_24 \
        --output-dir outputs/cv_results_figures \
        --only heatmap tables

    # Also generate cv_summary.json files in each experiment dir:
    python3 scripts/analysis/generate_cv_results.py \
        --experiments-dir outputs/experiments_2026_02_24 \
        --output-dir outputs/cv_results_figures \
        --save-summaries
"""
import json
import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# Optional imports — graceful degradation if not available
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NORMAL_CLASSES = ['adaptive', 'adaptive150025', 'face', 'face150025',
                  'pocket', 'pocket150025']
ANOMALY_CLASSES = ['damageadaptive', 'damageface', 'damagepocket']
ALL_CLASSES = NORMAL_CLASSES + ANOMALY_CLASSES

MODELS = [
    ('encoder',             'MM-LSTM-DAE'),
    ('xgboost',             'XGBoost'),
    ('random_forest',       'RandomForest'),
    ('logistic_regression', 'Logistic'),
    ('mlp',                 'MLP'),
    ('lstm_simple',         'SimpleLSTM'),
]
MODEL_KEYS = [k for k, _ in MODELS]
MODEL_LABELS = [l for _, l in MODELS]

# Feature-config display names (order matters for the table)
FEATURE_DISPLAY = {
    'full':                                              'Full (110)',
    'no_proximity':                                      'No Prox (104)',
    'no_proximity_no_pressure':                          'No Prox+Pres (98)',
    'no_proximity_no_pressure_no_color':                 'No Prox+Pres+Col (74)',
    'no_proximity_no_pressure_no_color_no_magnetometer': 'No Prox+Pres+Col+Mag (56)',
}

MINIMAL_CONFIG = 'no_proximity_no_pressure_no_color_no_magnetometer'

# Matplotlib style
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#795548']


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_experiment_dir(dirname: str):
    """
    Parse experiment directory name into (feature_config, window, stride).
    Examples:
        full_w64_s16_cv              -> ('full', 64, 16)
        no_proximity_no_color_w64_s16_cv -> ('no_proximity_no_color', 64, 16)
    """
    m = re.match(r'^(.+?)_w(\d+)_s(\d+)_cv$', dirname)
    if not m:
        return None, None, None
    feature_cfg = m.group(1)
    window = int(m.group(2))
    stride = int(m.group(3))
    return feature_cfg, window, stride


def load_encoder_result(fold_dir: Path):
    """Load encoder test accuracy from val_best checkpoint."""
    path = fold_dir / 'encoder' / 'results.json'
    if not path.exists():
        return None, {}
    with open(path) as f:
        data = json.load(f)
    result = data.get('val_best_test', {})
    return result.get('accuracy'), result.get('per_class', {})


def load_baseline_result(fold_dir: Path, model_key: str):
    """Load baseline test accuracy."""
    path = fold_dir / 'baselines' / model_key / 'results.json'
    if not path.exists():
        return None, {}
    with open(path) as f:
        data = json.load(f)
    result = data.get('test', {})
    return result.get('accuracy'), result.get('per_class', {})


def class_avg(per_class, class_list):
    vals = [per_class[c] for c in class_list if c in per_class]
    return float(np.mean(vals)) if vals else float('nan')


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def collect_all_results(experiments_dir: Path, n_folds: int = 5):
    """
    Scan all experiment directories and return a nested dict:
      results[experiment_name] = {
          'feature_cfg': str,
          'window': int,
          'stride': int,
          'models': {
              model_label: {
                  'per_fold_overall': [...],
                  'per_fold_normal': [...],
                  'per_fold_anomaly': [...],
                  'per_class': {class: [fold_accs]},
                  'mean': float,
                  'std': float,
              }
          }
      }
    """
    results = {}
    exp_dirs = sorted(d for d in experiments_dir.iterdir() if d.is_dir())

    for exp_dir in exp_dirs:
        feature_cfg, window, stride = parse_experiment_dir(exp_dir.name)
        if feature_cfg is None:
            continue

        exp_results = {
            'feature_cfg': feature_cfg,
            'window': window,
            'stride': stride,
            'models': {},
        }

        for model_key, model_label in MODELS:
            fold_overall = []
            fold_normal = []
            fold_anomaly = []
            per_class_folds = {c: [] for c in ALL_CLASSES}

            for fold in range(1, n_folds + 1):
                fold_dir = exp_dir / f'fold_{fold}'
                if model_key == 'encoder':
                    acc, pc = load_encoder_result(fold_dir)
                else:
                    acc, pc = load_baseline_result(fold_dir, model_key)

                fold_overall.append(acc)
                fold_normal.append(class_avg(pc, NORMAL_CLASSES) if pc else None)
                fold_anomaly.append(class_avg(pc, ANOMALY_CLASSES) if pc else None)
                for c in ALL_CLASSES:
                    per_class_folds[c].append(pc.get(c) if pc else None)

            valid = [v for v in fold_overall if v is not None]
            exp_results['models'][model_label] = {
                'per_fold_overall': fold_overall,
                'per_fold_normal': fold_normal,
                'per_fold_anomaly': fold_anomaly,
                'per_class': per_class_folds,
                'mean': float(np.mean(valid)) if valid else None,
                'std': float(np.std(valid)) if valid else None,
                'n_complete': len(valid),
            }

        results[exp_dir.name] = exp_results

    return results


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
def print_grand_summary(results):
    """Print a grand summary table to stdout."""
    print()
    sep = '=' * 120
    print(sep)
    print('GRAND SUMMARY — MEAN TEST ACCURACY ± STD ACROSS 5 FOLDS')
    print(sep)

    header = f"{'Experiment':<46}" + ''.join(f'{m:>12}' for m in MODEL_LABELS)
    print(header)
    print('-' * len(header))

    for exp_name in sorted(results.keys()):
        exp = results[exp_name]
        label = f"{exp['feature_cfg']} w{exp['window']}s{exp['stride']}"
        row = f"{label:<46}"
        for ml in MODEL_LABELS:
            m = exp['models'].get(ml, {})
            mean, std = m.get('mean'), m.get('std')
            if mean is not None:
                row += f"{mean*100:5.1f}±{std*100:4.1f}%".rjust(12)
            else:
                row += '  N/A'.rjust(12)
        print(row)

    print(sep)


def print_feature_ablation_table(results, default_window=64, default_stride=16):
    """Print Table 1: Feature ablation at default window/stride."""
    print()
    sep = '=' * 110
    print(sep)
    print(f'TABLE 1 — FEATURE ABLATION (window={default_window}, stride={default_stride})')
    print(sep)

    header = f"{'Feature Config':<28}" + ''.join(f'{m:>14}' for m in MODEL_LABELS)
    print(header)
    print('-' * len(header))

    for feat_key, feat_label in FEATURE_DISPLAY.items():
        exp_name = f"{feat_key}_w{default_window}_s{default_stride}_cv"
        exp = results.get(exp_name)
        if exp is None:
            continue
        row = f"{feat_label:<28}"
        for ml in MODEL_LABELS:
            m = exp['models'].get(ml, {})
            mean, std = m.get('mean'), m.get('std')
            if mean is not None:
                row += f"{mean*100:5.1f} ± {std*100:4.1f}%".rjust(14)
            else:
                row += '  N/A'.rjust(14)
        print(row)

    print(sep)


def print_window_ablation_table(results, feature_cfg='full'):
    """Print Table 2: Window/stride ablation at full features."""
    print()
    sep = '=' * 110
    print(sep)
    print(f'TABLE 2 — WINDOW/STRIDE ABLATION (features={feature_cfg})')
    print(sep)

    # Collect relevant experiments, sort by window then stride
    relevant = []
    for exp_name, exp in results.items():
        if exp['feature_cfg'] == feature_cfg:
            relevant.append((exp['window'], exp['stride'], exp_name, exp))
    relevant.sort()

    header = f"{'Window × Stride':<28}" + ''.join(f'{m:>14}' for m in MODEL_LABELS)
    print(header)
    print('-' * len(header))

    for window, stride, exp_name, exp in relevant:
        label = f"w{window} × s{stride} ({window // stride}:1)"
        row = f"{label:<28}"
        for ml in MODEL_LABELS:
            m = exp['models'].get(ml, {})
            mean, std = m.get('mean'), m.get('std')
            if mean is not None:
                row += f"{mean*100:5.1f} ± {std*100:4.1f}%".rjust(14)
            else:
                row += '  N/A'.rjust(14)
        print(row)

    print(sep)


# ---------------------------------------------------------------------------
# Matplotlib figures
# ---------------------------------------------------------------------------
def generate_heatmap(results, output_dir: Path):
    """Generate summary heatmap of all configs × all models."""
    if not HAS_MPL:
        print("[WARN] matplotlib not available — skipping heatmap")
        return

    # Build matrix
    exp_names = sorted(results.keys(),
                       key=lambda n: (results[n]['feature_cfg'],
                                      results[n]['window'],
                                      results[n]['stride']))
    n_exps = len(exp_names)
    n_models = len(MODEL_LABELS)

    matrix = np.full((n_exps, n_models), np.nan)
    row_labels = []

    for i, exp_name in enumerate(exp_names):
        exp = results[exp_name]
        row_labels.append(f"{exp['feature_cfg']}\nw{exp['window']}s{exp['stride']}")
        for j, ml in enumerate(MODEL_LABELS):
            m = exp['models'].get(ml, {})
            if m.get('mean') is not None:
                matrix[i, j] = m['mean'] * 100

    fig, ax = plt.subplots(figsize=(10, max(6, n_exps * 0.55)))

    # Color map: green=good, red=bad
    cmap = plt.cm.RdYlGn
    vmin = np.nanmin(matrix) - 2 if not np.all(np.isnan(matrix)) else 0
    vmax = 100

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    # Annotate cells
    for i in range(n_exps):
        for j in range(n_models):
            val = matrix[i, j]
            if not np.isnan(val):
                # Determine text color
                text_color = 'white' if val < (vmin + vmax) / 2 else 'black'
                std_val = results[exp_names[i]]['models'][MODEL_LABELS[j]].get('std', 0)
                std_pct = std_val * 100 if std_val else 0
                ax.text(j, i, f'{val:.1f}\n±{std_pct:.1f}',
                        ha='center', va='center', fontsize=7, color=text_color)
            else:
                ax.text(j, i, 'N/A', ha='center', va='center',
                        fontsize=7, color='gray')

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(MODEL_LABELS, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_exps))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title('5-Fold CV Test Accuracy (%) — All Configurations', fontsize=12, pad=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Accuracy (%)', fontsize=9)

    plt.tight_layout()
    path = output_dir / 'heatmap_all_configs.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_window_lineplot(results, output_dir: Path, feature_cfg='full'):
    """Line plot: accuracy vs window size per model, with error bands."""
    if not HAS_MPL:
        print("[WARN] matplotlib not available — skipping line plot")
        return

    # Collect data points
    relevant = []
    for exp_name, exp in results.items():
        if exp['feature_cfg'] == feature_cfg:
            relevant.append(exp)
    relevant.sort(key=lambda e: (e['window'], e['stride']))

    if not relevant:
        print(f"  [SKIP] No experiments for feature_cfg={feature_cfg}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, ml in enumerate(MODEL_LABELS):
        x_labels = []
        means = []
        stds = []
        for exp in relevant:
            m = exp['models'].get(ml, {})
            if m.get('mean') is not None:
                x_labels.append(f"w{exp['window']}s{exp['stride']}")
                means.append(m['mean'] * 100)
                stds.append((m.get('std', 0) or 0) * 100)

        if not means:
            continue

        x = range(len(means))
        means_arr = np.array(means)
        stds_arr = np.array(stds)

        color = COLORS[idx % len(COLORS)]
        ax.plot(x, means_arr, 'o-', color=color, label=ml, linewidth=2, markersize=6)
        ax.fill_between(x, means_arr - stds_arr, means_arr + stds_arr,
                        alpha=0.15, color=color)

    # X-axis labels
    x_tick_labels = [f"w{e['window']}s{e['stride']}" for e in relevant]
    ax.set_xticks(range(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Window × Stride', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title(f'Accuracy vs Window/Stride Configuration ({feature_cfg} features)',
                 fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 5))

    plt.tight_layout()
    suffix = feature_cfg if feature_cfg != 'full' else 'full'
    path = output_dir / f'lineplot_window_stride_{suffix}.pdf'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_comparison_lineplot(results, output_dir: Path,
                                  cfg_a='full', cfg_b=MINIMAL_CONFIG):
    """Side-by-side line plots comparing two feature configs across windows."""
    if not HAS_MPL:
        print("[WARN] matplotlib not available — skipping comparison plot")
        return

    def _collect(feature_cfg):
        relevant = [exp for exp in results.values()
                     if exp['feature_cfg'] == feature_cfg]
        relevant.sort(key=lambda e: (e['window'], e['stride']))
        return relevant

    data_a = _collect(cfg_a)
    data_b = _collect(cfg_b)

    if not data_a or not data_b:
        print(f"  [SKIP] Need both '{cfg_a}' and '{cfg_b}' for comparison")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    label_a = FEATURE_DISPLAY.get(cfg_a, cfg_a)
    label_b = FEATURE_DISPLAY.get(cfg_b, cfg_b)

    for ax, data, title in [(ax1, data_a, label_a), (ax2, data_b, label_b)]:
        for idx, ml in enumerate(MODEL_LABELS):
            x_labels = []
            means = []
            stds = []
            for exp in data:
                m = exp['models'].get(ml, {})
                if m.get('mean') is not None:
                    x_labels.append(f"w{exp['window']}s{exp['stride']}")
                    means.append(m['mean'] * 100)
                    stds.append((m.get('std', 0) or 0) * 100)

            if not means:
                continue

            x = range(len(means))
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            color = COLORS[idx % len(COLORS)]

            ax.plot(x, means_arr, 'o-', color=color, label=ml,
                    linewidth=2, markersize=5)
            ax.fill_between(x, means_arr - stds_arr, means_arr + stds_arr,
                            alpha=0.12, color=color)

        x_tick_labels = [f"w{e['window']}s{e['stride']}" for e in data]
        ax.set_xticks(range(len(x_tick_labels)))
        ax.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Window × Stride', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax1.legend(loc='lower right', fontsize=8)

    # Shared y-axis: find global range
    all_means = []
    for data in [data_a, data_b]:
        for exp in data:
            for ml in MODEL_LABELS:
                m = exp['models'].get(ml, {})
                if m.get('mean') is not None:
                    all_means.append(m['mean'] * 100)
    if all_means:
        ymin = min(all_means) - 5
        ymax = max(all_means) + 3
        ax1.set_ylim(max(0, ymin), min(100, ymax))

    fig.suptitle('Accuracy vs Temporal Resolution: Full Features vs Minimal Features',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = output_dir / 'comparison_full_vs_minimal.pdf'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_degradation_plot(results, output_dir: Path,
                               cfg_full='full', cfg_min=MINIMAL_CONFIG):
    """Bar chart showing accuracy degradation (full - minimal) per model per window."""
    if not HAS_MPL:
        print("[WARN] matplotlib not available — skipping degradation plot")
        return

    # Find window/stride combos that have BOTH configs
    full_by_ws = {}
    min_by_ws = {}
    for exp in results.values():
        ws = (exp['window'], exp['stride'])
        if exp['feature_cfg'] == cfg_full:
            full_by_ws[ws] = exp
        elif exp['feature_cfg'] == cfg_min:
            min_by_ws[ws] = exp

    shared_ws = sorted(set(full_by_ws.keys()) & set(min_by_ws.keys()))
    if not shared_ws:
        print("  [SKIP] No overlapping window/stride combos for degradation plot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(shared_ws))
    bar_width = 0.12

    for idx, ml in enumerate(MODEL_LABELS):
        deltas = []
        for ws in shared_ws:
            m_full = full_by_ws[ws]['models'].get(ml, {})
            m_min = min_by_ws[ws]['models'].get(ml, {})
            full_mean = m_full.get('mean')
            min_mean = m_min.get('mean')
            if full_mean is not None and min_mean is not None:
                deltas.append((full_mean - min_mean) * 100)  # positive = lost accuracy
            else:
                deltas.append(0)

        offset = (idx - len(MODEL_LABELS) / 2 + 0.5) * bar_width
        color = COLORS[idx % len(COLORS)]
        ax.bar(x + offset, deltas, bar_width, label=ml, color=color, alpha=0.85)

    x_labels = [f"w{w}s{s}" for w, s in shared_ws]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Window × Stride', fontsize=11)
    ax.set_ylabel('Accuracy Drop (pp)', fontsize=11)
    ax.set_title('Accuracy Degradation: Full (110) → Minimal (56) Features', fontsize=12)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    path = output_dir / 'degradation_full_vs_minimal.pdf'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_per_class_chart(results, output_dir: Path,
                              feature_cfg=None, window=None, stride=None):
    """Bar chart: per-class accuracy for a specific or best config.

    If feature_cfg/window/stride are given, use that specific experiment.
    Otherwise fall back to the config with highest encoder mean.
    """
    if not HAS_MPL:
        print("[WARN] matplotlib not available — skipping per-class chart")
        return

    best_exp = None

    # Try to use the explicitly requested config
    if feature_cfg is not None and window is not None and stride is not None:
        target_name = f"{feature_cfg}_w{window}_s{stride}_cv"
        if target_name in results:
            best_exp = target_name
        else:
            print(f"  [WARN] Requested config {target_name} not found, falling back to best")

    # Fallback: find the config with highest encoder mean
    if best_exp is None:
        best_mean = -1
        for ename, exp in results.items():
            enc = exp['models'].get('MM-LSTM-DAE', {})
            if enc.get('mean') is not None and enc['mean'] > best_mean:
                best_mean = enc['mean']
                best_exp = ename

    if best_exp is None:
        # Fall back to highest overall mean across any model
        best_mean = -1
        for ename, exp in results.items():
            for ml in MODEL_LABELS:
                m = exp['models'].get(ml, {})
                if m.get('mean') is not None and m['mean'] > best_mean:
                    best_mean = m['mean']
                    best_exp = ename

    if best_exp is None:
        print("  [SKIP] No results available for per-class chart")
        return

    exp = results[best_exp]
    exp_label = f"{exp['feature_cfg']} w{exp['window']}s{exp['stride']}"

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(ALL_CLASSES))
    bar_width = 0.12
    n_models = len(MODEL_LABELS)

    for idx, ml in enumerate(MODEL_LABELS):
        m = exp['models'].get(ml, {})
        class_means = []
        for c in ALL_CLASSES:
            vals = m.get('per_class', {}).get(c, [])
            valid = [v for v in vals if v is not None]
            class_means.append(np.mean(valid) * 100 if valid else 0)

        offset = (idx - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, class_means, bar_width,
                      label=ml, color=COLORS[idx % len(COLORS)], alpha=0.85)

    # Mark anomaly classes
    for i, c in enumerate(ALL_CLASSES):
        if c in ANOMALY_CLASSES:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.06, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_CLASSES, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'Per-Class Accuracy — Best Config: {exp_label}', fontsize=12)
    ax.legend(loc='lower left', fontsize=8, ncol=3)
    ax.set_ylim(0, 108)
    ax.grid(True, axis='y', alpha=0.3)

    # Add anomaly label
    anomaly_start = len(NORMAL_CLASSES)
    ax.annotate('Anomaly classes', xy=(anomaly_start + 0.5, 105),
                fontsize=8, color='red', ha='center', style='italic')

    plt.tight_layout()
    suffix = exp['feature_cfg']
    path = output_dir / f'per_class_breakdown_{suffix}.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_feature_ablation_chart(results, output_dir: Path,
                                    default_window=64, default_stride=16):
    """Grouped bar chart for feature ablation."""
    if not HAS_MPL:
        print("[WARN] matplotlib not available — skipping feature ablation chart")
        return

    feat_configs = []
    feat_labels = []
    for feat_key, feat_label in FEATURE_DISPLAY.items():
        exp_name = f"{feat_key}_w{default_window}_s{default_stride}_cv"
        if exp_name in results:
            feat_configs.append(exp_name)
            feat_labels.append(feat_label)

    if not feat_configs:
        print("  [SKIP] No feature ablation experiments found")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(feat_configs))
    bar_width = 0.12

    for idx, ml in enumerate(MODEL_LABELS):
        means = []
        stds = []
        for exp_name in feat_configs:
            m = results[exp_name]['models'].get(ml, {})
            means.append((m.get('mean', 0) or 0) * 100)
            stds.append((m.get('std', 0) or 0) * 100)

        offset = (idx - len(MODEL_LABELS) / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds,
               label=ml, color=COLORS[idx % len(COLORS)], alpha=0.85,
               capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels(feat_labels, fontsize=10)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Feature Ablation — Impact of Removing Sensor Groups', fontsize=12)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = output_dir / 'feature_ablation_bars.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------
def generate_latex_tables(results, output_dir: Path,
                          default_window=64, default_stride=16):
    """Generate publication-ready LaTeX tables."""
    lines = []

    # --- Table 1: Feature ablation ---
    lines.append('% Table 1: Feature Ablation')
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\caption{Feature ablation: 5-fold CV test accuracy '
                 f'(window={default_window}, stride={default_stride}).}}')
    lines.append('\\label{tab:feature_ablation}')
    n_cols = len(MODEL_LABELS) + 1
    lines.append('\\begin{tabular}{l' + 'c' * len(MODEL_LABELS) + '}')
    lines.append('\\toprule')
    lines.append('Features & ' + ' & '.join(MODEL_LABELS) + ' \\\\')
    lines.append('\\midrule')

    for feat_key, feat_label in FEATURE_DISPLAY.items():
        exp_name = f"{feat_key}_w{default_window}_s{default_stride}_cv"
        exp = results.get(exp_name)
        if exp is None:
            continue
        cells = [feat_label]
        best_val = -1
        for ml in MODEL_LABELS:
            m = exp['models'].get(ml, {})
            if m.get('mean') is not None and m['mean'] > best_val:
                best_val = m['mean']
        for ml in MODEL_LABELS:
            m = exp['models'].get(ml, {})
            mean, std = m.get('mean'), m.get('std')
            if mean is not None:
                cell = f'{mean*100:.1f} $\\pm$ {std*100:.1f}'
                if abs(mean - best_val) < 1e-6:
                    cell = '\\textbf{' + cell + '}'
                cells.append(cell)
            else:
                cells.append('--')
        lines.append(' & '.join(cells) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')

    # --- Table 2: Window/stride ablation ---
    lines.append('% Table 2: Window/Stride Ablation')
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\caption{Window/stride ablation: 5-fold CV test accuracy (full features).}')
    lines.append('\\label{tab:window_ablation}')
    lines.append('\\begin{tabular}{l' + 'c' * len(MODEL_LABELS) + '}')
    lines.append('\\toprule')
    lines.append('Window $\\times$ Stride & ' + ' & '.join(MODEL_LABELS) + ' \\\\')
    lines.append('\\midrule')

    relevant = [(exp['window'], exp['stride'], exp_name, exp)
                for exp_name, exp in results.items()
                if exp['feature_cfg'] == 'full']
    relevant.sort()

    for window, stride, exp_name, exp in relevant:
        label = f'$w{window} \\times s{stride}$'
        cells = [label]
        best_val = -1
        for ml in MODEL_LABELS:
            m = exp['models'].get(ml, {})
            if m.get('mean') is not None and m['mean'] > best_val:
                best_val = m['mean']
        for ml in MODEL_LABELS:
            m = exp['models'].get(ml, {})
            mean, std = m.get('mean'), m.get('std')
            if mean is not None:
                cell = f'{mean*100:.1f} $\\pm$ {std*100:.1f}'
                if abs(mean - best_val) < 1e-6:
                    cell = '\\textbf{' + cell + '}'
                cells.append(cell)
            else:
                cells.append('--')
        lines.append(' & '.join(cells) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')

    # --- Table 3: Window/stride ablation at minimal features ---
    minimal_label = FEATURE_DISPLAY.get(MINIMAL_CONFIG, MINIMAL_CONFIG)
    relevant_min = [(exp['window'], exp['stride'], ename, exp)
                    for ename, exp in results.items()
                    if exp['feature_cfg'] == MINIMAL_CONFIG]
    relevant_min.sort()

    if relevant_min:
        lines.append(f'% Table 3: Window/Stride Ablation — Minimal Features')
        lines.append('\\begin{table}[htbp]')
        lines.append('\\centering')
        lines.append('\\caption{Window/stride ablation: 5-fold CV test accuracy '
                     f'({minimal_label} features).}}')
        lines.append('\\label{tab:window_ablation_minimal}')
        lines.append('\\begin{tabular}{l' + 'c' * len(MODEL_LABELS) + '}')
        lines.append('\\toprule')
        lines.append('Window $\\times$ Stride & ' + ' & '.join(MODEL_LABELS) + ' \\\\')
        lines.append('\\midrule')

        for window, stride, ename, exp in relevant_min:
            label = f'$w{window} \\times s{stride}$'
            cells = [label]
            best_val = -1
            for ml in MODEL_LABELS:
                m = exp['models'].get(ml, {})
                if m.get('mean') is not None and m['mean'] > best_val:
                    best_val = m['mean']
            for ml in MODEL_LABELS:
                m = exp['models'].get(ml, {})
                mean, std = m.get('mean'), m.get('std')
                if mean is not None:
                    cell = f'{mean*100:.1f} $\\pm$ {std*100:.1f}'
                    if abs(mean - best_val) < 1e-6:
                        cell = '\\textbf{' + cell + '}'
                    cells.append(cell)
                else:
                    cells.append('--')
            lines.append(' & '.join(cells) + ' \\\\')

        lines.append('\\bottomrule')
        lines.append('\\end{tabular}')
        lines.append('\\end{table}')

    latex_path = output_dir / 'cv_tables.tex'
    latex_path.write_text('\n'.join(lines))
    print(f"  Saved: {latex_path}")


# ---------------------------------------------------------------------------
# Save cv_summary.json (optional)
# ---------------------------------------------------------------------------
def save_summaries(results, experiments_dir: Path):
    """Write cv_summary.json into each experiment directory."""
    for exp_name, exp in results.items():
        summary = {}
        for ml in MODEL_LABELS:
            m = exp['models'].get(ml, {})
            per_class_mean = {}
            for c in ALL_CLASSES:
                vals = [v for v in m.get('per_class', {}).get(c, [])
                        if v is not None]
                per_class_mean[c] = float(np.mean(vals)) if vals else None

            n_vals = [v for v in m.get('per_fold_normal', [])
                      if v is not None and not np.isnan(v)]
            a_vals = [v for v in m.get('per_fold_anomaly', [])
                      if v is not None and not np.isnan(v)]

            summary[ml] = {
                'per_fold_overall': m.get('per_fold_overall', []),
                'per_fold_normal':  m.get('per_fold_normal', []),
                'per_fold_anomaly': m.get('per_fold_anomaly', []),
                'mean_overall':  m.get('mean'),
                'std_overall':   m.get('std'),
                'mean_normal':   float(np.mean(n_vals)) if n_vals else None,
                'std_normal':    float(np.std(n_vals)) if n_vals else None,
                'mean_anomaly':  float(np.mean(a_vals)) if a_vals else None,
                'std_anomaly':   float(np.std(a_vals)) if a_vals else None,
                'per_class_mean': per_class_mean,
            }

        out_path = experiments_dir / exp_name / 'cv_summary.json'
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Generate cross-validation results visualizations')
    parser.add_argument('--experiments-dir', type=Path, required=True,
                        help='Directory containing experiment subdirs')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory for figures/tables '
                             '(default: <experiments-dir>/cv_results_figures)')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--default-window', type=int, default=64,
                        help='Window size for feature ablation table')
    parser.add_argument('--default-stride', type=int, default=16,
                        help='Stride for feature ablation table')
    parser.add_argument('--save-summaries', action='store_true',
                        help='Also write cv_summary.json into each experiment dir')
    parser.add_argument('--only', nargs='+', default=None,
                        choices=['heatmap', 'tables', 'lineplot',
                                 'perclass', 'feature_bars', 'latex',
                                 'console', 'comparison', 'degradation'],
                        help='Only generate specific outputs')
    args = parser.parse_args()

    output_dir = args.output_dir or (args.experiments_dir / 'cv_results_figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning experiments in: {args.experiments_dir}")
    results = collect_all_results(args.experiments_dir, args.n_folds)
    print(f"Found {len(results)} experiment configs\n")

    if not results:
        print("ERROR: No experiment directories found.")
        sys.exit(1)

    # Count completeness
    total_folds = 0
    complete_folds = 0
    for exp in results.values():
        for ml in MODEL_LABELS:
            m = exp['models'].get(ml, {})
            total_folds += args.n_folds
            complete_folds += m.get('n_complete', 0)
    print(f"Completeness: {complete_folds}/{total_folds} fold-model combos "
          f"({complete_folds/total_folds*100:.0f}%)\n")

    gen_all = args.only is None
    targets = set(args.only or [])

    # Console output (always shown unless --only excludes it)
    if gen_all or 'console' in targets:
        print_grand_summary(results)
        print_feature_ablation_table(results, args.default_window, args.default_stride)
        print_window_ablation_table(results, 'full')
        print_window_ablation_table(results, MINIMAL_CONFIG)

    # Figures
    if gen_all or 'heatmap' in targets:
        print("\nGenerating heatmap...")
        generate_heatmap(results, output_dir)

    if gen_all or 'lineplot' in targets:
        print("Generating window/stride line plots...")
        generate_window_lineplot(results, output_dir, 'full')
        generate_window_lineplot(results, output_dir, MINIMAL_CONFIG)

    if gen_all or 'comparison' in targets:
        print("Generating full vs minimal comparison plot...")
        generate_comparison_lineplot(results, output_dir)

    if gen_all or 'degradation' in targets:
        print("Generating degradation plot...")
        generate_degradation_plot(results, output_dir)

    if gen_all or 'perclass' in targets:
        print("Generating per-class breakdowns...")
        # Per-class for best full-feature config
        generate_per_class_chart(results, output_dir)
        # Per-class for best minimal-feature config at default window
        generate_per_class_chart(results, output_dir,
                                 feature_cfg=MINIMAL_CONFIG,
                                 window=args.default_window,
                                 stride=args.default_stride)

    if gen_all or 'feature_bars' in targets:
        print("Generating feature ablation bar chart...")
        generate_feature_ablation_chart(results, output_dir,
                                        args.default_window, args.default_stride)

    if gen_all or 'latex' in targets:
        print("Generating LaTeX tables...")
        generate_latex_tables(results, output_dir,
                              args.default_window, args.default_stride)

    if gen_all or 'tables' in targets:
        # Also save console tables to text file
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        print_grand_summary(results)
        print_feature_ablation_table(results, args.default_window, args.default_stride)
        print_window_ablation_table(results, 'full')
        print_window_ablation_table(results, MINIMAL_CONFIG)
        sys.stdout = old_stdout
        txt_path = output_dir / 'summary_tables.txt'
        txt_path.write_text(buf.getvalue())
        print(f"  Saved: {txt_path}")

    # Optional: save cv_summary.json files
    if args.save_summaries:
        print("\nSaving cv_summary.json files...")
        save_summaries(results, args.experiments_dir)

    print(f"\nDone! Outputs in: {output_dir}")


if __name__ == '__main__':
    main()
