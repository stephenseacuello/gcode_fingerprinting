#!/usr/bin/env python3
"""
Post-hoc analysis for cross-validation experiments.

Generates three publication-ready outputs from completed CV experiments:
  1. Aggregated confusion matrices (summed across folds) — heatmaps per model
  2. Statistical significance tests (paired t-test + Wilcoxon) — encoder vs baselines
  3. Training time comparison table — mean +/- std across folds

Reads from the same experiment directory structure used by run_cv_parallel.py
and generate_cv_results.py.

Usage:
    python3 scripts/analysis/posthoc_cv_analysis.py \
        --experiments-dir outputs/experiments_2026_02_25

    # Specific config/window/stride (default: best encoder config)
    python3 scripts/analysis/posthoc_cv_analysis.py \
        --experiments-dir outputs/experiments_2026_02_25 \
        --config full_w64_s16_cv

    # Only specific analyses
    python3 scripts/analysis/posthoc_cv_analysis.py \
        --experiments-dir outputs/experiments_2026_02_25 \
        --only confusion significance timing
"""
import json
import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ['adaptive', 'adaptive150025', 'face', 'face150025',
               'pocket', 'pocket150025',
               'damageadaptive', 'damageface', 'damagepocket']

SHORT_NAMES = ['adapt', 'adapt150', 'face', 'face150',
               'pocket', 'pocket150',
               'dmg-adapt', 'dmg-face', 'dmg-pocket']

MODELS = [
    ('encoder',             'MM-LSTM-DAE'),
    ('xgboost',             'XGBoost'),
    ('random_forest',       'RandomForest'),
    ('logistic_regression', 'Logistic'),
    ('mlp',                 'MLP'),
    ('lstm_simple',         'SimpleLSTM'),
]
MODEL_KEYS = [k for k, _ in MODELS]
MODEL_LABELS = {k: l for k, l in MODELS}

FEATURE_DISPLAY = {
    'full':                                              'Full (110)',
    'no_proximity':                                      'No Prox (104)',
    'no_proximity_no_pressure':                          'No Prox+Pres (98)',
    'no_proximity_no_pressure_no_color':                 'No Prox+Pres+Col (74)',
    'no_proximity_no_pressure_no_color_no_magnetometer': 'No Prox+Pres+Col+Mag (56)',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_experiment_dir(dirname: str):
    m = re.match(r'^(.+?)_w(\d+)_s(\d+)_cv$', dirname)
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), int(m.group(3))


def load_results_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_encoder_results(fold_dir: Path):
    """Return (accuracy, confusion_matrix, train_time) for encoder."""
    data = load_results_json(fold_dir / 'encoder' / 'results.json')
    if data is None:
        return None, None, None
    result = data.get('val_best_test', {})
    acc = result.get('accuracy')
    cm = result.get('confusion_matrix')
    tt = data.get('train_time_seconds')
    return acc, cm, tt


def get_baseline_results(fold_dir: Path, model_key: str):
    """Return (accuracy, confusion_matrix, train_time) for a baseline."""
    data = load_results_json(fold_dir / 'baselines' / model_key / 'results.json')
    if data is None:
        return None, None, None
    result = data.get('test', {})
    acc = result.get('accuracy')
    cm = result.get('confusion_matrix')
    tt = data.get('train_time_seconds')
    return acc, cm, tt


def find_best_encoder_config(experiments_dir: Path, n_folds: int = 5):
    """Find the experiment config with highest mean encoder accuracy."""
    best_name, best_acc = None, -1.0
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        feat, w, s = parse_experiment_dir(exp_dir.name)
        if feat is None:
            continue
        accs = []
        for fold in range(1, n_folds + 1):
            a, _, _ = get_encoder_results(exp_dir / f'fold_{fold}')
            if a is not None:
                accs.append(a)
        if len(accs) == n_folds:
            mean = np.mean(accs)
            if mean > best_acc:
                best_acc = mean
                best_name = exp_dir.name
    return best_name, best_acc


# ---------------------------------------------------------------------------
# 1. Aggregated Confusion Matrices
# ---------------------------------------------------------------------------
def generate_confusion_matrices(experiments_dir: Path, exp_name: str,
                                 output_dir: Path, n_folds: int = 5):
    """Sum confusion matrices across folds and plot per model."""
    if not HAS_MPL:
        print("  [SKIP] matplotlib not available")
        return

    exp_dir = experiments_dir / exp_name
    feat, w, s = parse_experiment_dir(exp_name)
    feat_label = FEATURE_DISPLAY.get(feat, feat)

    for model_key in MODEL_KEYS:
        label = MODEL_LABELS[model_key]
        agg_cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
        n_valid = 0

        for fold in range(1, n_folds + 1):
            fold_dir = exp_dir / f'fold_{fold}'
            if model_key == 'encoder':
                _, cm, _ = get_encoder_results(fold_dir)
            else:
                _, cm, _ = get_baseline_results(fold_dir, model_key)

            if cm is not None:
                agg_cm += np.array(cm, dtype=int)
                n_valid += 1

        if n_valid == 0:
            print(f"  [SKIP] {label}: no confusion matrices found")
            continue

        # Normalize by row (true class) for display
        row_sums = agg_cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid div-by-zero
        cm_norm = agg_cm / row_sums

        fig, axes = plt.subplots(1, 2, figsize=(22, 9))

        # Left: raw counts
        sns.heatmap(agg_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                    ax=axes[0])
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        axes[0].set_title(f'{label} — Raw Counts (summed over {n_valid} folds)')

        # Right: row-normalized (recall)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                    vmin=0, vmax=1, ax=axes[1])
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        axes[1].set_title(f'{label} — Per-Class Recall')

        fig.suptitle(f'{feat_label}  |  w={w} s={s}', fontsize=13)
        plt.tight_layout()
        out_path = output_dir / f'confusion_matrix_{model_key}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out_path.name}")

    # Also generate a single combined figure with all models (normalized only)
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    axes_flat = axes.flatten()

    for idx, model_key in enumerate(MODEL_KEYS):
        label = MODEL_LABELS[model_key]
        agg_cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)

        for fold in range(1, n_folds + 1):
            fold_dir = exp_dir / f'fold_{fold}'
            if model_key == 'encoder':
                _, cm, _ = get_encoder_results(fold_dir)
            else:
                _, cm, _ = get_baseline_results(fold_dir, model_key)
            if cm is not None:
                agg_cm += np.array(cm, dtype=int)

        row_sums = agg_cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = agg_cm / row_sums

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                    vmin=0, vmax=1, ax=axes_flat[idx], cbar=False)
        axes_flat[idx].set_title(label, fontsize=12, fontweight='bold')
        if idx >= 3:
            axes_flat[idx].set_xlabel('Predicted')
        if idx % 3 == 0:
            axes_flat[idx].set_ylabel('True')

    fig.suptitle(f'Confusion Matrices (Row-Normalized)  |  {feat_label}  |  w={w} s={s}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = output_dir / 'confusion_matrix_all_models.pdf'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# 2. Statistical Significance Tests
# ---------------------------------------------------------------------------
def generate_significance_tests(experiments_dir: Path, exp_name: str,
                                 output_dir: Path, n_folds: int = 5):
    """Paired t-test and Wilcoxon signed-rank: encoder vs each baseline."""
    if not HAS_SCIPY:
        print("  [SKIP] scipy not available")
        return

    exp_dir = experiments_dir / exp_name
    feat, w, s = parse_experiment_dir(exp_name)
    feat_label = FEATURE_DISPLAY.get(feat, feat)

    # Collect per-fold accuracies
    model_fold_accs = {}
    for model_key in MODEL_KEYS:
        accs = []
        for fold in range(1, n_folds + 1):
            fold_dir = exp_dir / f'fold_{fold}'
            if model_key == 'encoder':
                a, _, _ = get_encoder_results(fold_dir)
            else:
                a, _, _ = get_baseline_results(fold_dir, model_key)
            accs.append(a if a is not None else float('nan'))
        model_fold_accs[model_key] = np.array(accs)

    encoder_accs = model_fold_accs['encoder']
    baseline_keys = [k for k in MODEL_KEYS if k != 'encoder']

    print(f"\n  Config: {feat_label} | w={w} s={s}")
    print(f"  {'Model':<20s} {'Mean':>7s} {'Std':>7s} {'t-stat':>8s} {'t p-val':>8s} {'W-stat':>8s} {'W p-val':>8s}")
    print(f"  {'-'*72}")

    enc_mean = np.nanmean(encoder_accs)
    enc_std = np.nanstd(encoder_accs)
    print(f"  {'MM-LSTM-DAE':<20s} {enc_mean:>7.3f} {enc_std:>7.3f} {'---':>8s} {'---':>8s} {'---':>8s} {'---':>8s}")

    results = {
        'config': exp_name,
        'feature_cfg': feat,
        'window': w,
        'stride': s,
        'n_folds': n_folds,
        'encoder': {
            'per_fold': encoder_accs.tolist(),
            'mean': float(enc_mean),
            'std': float(enc_std),
        },
        'comparisons': {},
    }

    for bk in baseline_keys:
        bl_accs = model_fold_accs[bk]
        bl_label = MODEL_LABELS[bk]
        bl_mean = np.nanmean(bl_accs)
        bl_std = np.nanstd(bl_accs)

        # Paired t-test
        valid = ~(np.isnan(encoder_accs) | np.isnan(bl_accs))
        if valid.sum() >= 2:
            t_stat, t_pval = scipy_stats.ttest_rel(encoder_accs[valid], bl_accs[valid])
        else:
            t_stat, t_pval = float('nan'), float('nan')

        # Wilcoxon signed-rank (note: with n=5, min p ~0.0625)
        if valid.sum() >= 5:
            try:
                w_stat, w_pval = scipy_stats.wilcoxon(encoder_accs[valid], bl_accs[valid])
            except ValueError:
                w_stat, w_pval = float('nan'), float('nan')
        else:
            w_stat, w_pval = float('nan'), float('nan')

        sig = '***' if t_pval < 0.001 else '**' if t_pval < 0.01 else '*' if t_pval < 0.05 else ''
        print(f"  {bl_label:<20s} {bl_mean:>7.3f} {bl_std:>7.3f} {t_stat:>8.3f} {t_pval:>8.4f} "
              f"{w_stat:>8.1f} {w_pval:>8.4f} {sig}")

        results['comparisons'][bk] = {
            'label': bl_label,
            'per_fold': bl_accs.tolist(),
            'mean': float(bl_mean),
            'std': float(bl_std),
            'paired_ttest': {
                'statistic': float(t_stat) if not np.isnan(t_stat) else None,
                'p_value': float(t_pval) if not np.isnan(t_pval) else None,
            },
            'wilcoxon': {
                'statistic': float(w_stat) if not np.isnan(w_stat) else None,
                'p_value': float(w_pval) if not np.isnan(w_pval) else None,
            },
            'significant_005': bool(t_pval < 0.05) if not np.isnan(t_pval) else False,
        }

    # Save JSON
    out_path = output_dir / 'significance_tests.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {out_path.name}")

    # LaTeX table
    latex_lines = []
    latex_lines.append(r'\begin{table}[ht]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\caption{Statistical comparison of MM-LSTM-DAE against baselines (' + feat_label + r', $w$=' + str(w) + r', $s$=' + str(s) + r').}')
    latex_lines.append(r'\label{tab:significance}')
    latex_lines.append(r'\begin{tabular}{lcccc}')
    latex_lines.append(r'\toprule')
    latex_lines.append(r'\textbf{Model} & \textbf{Acc.\ (\%)} & \textbf{$\Delta$Enc.} & \textbf{$t$-test $p$} & \textbf{Wilcoxon $p$} \\')
    latex_lines.append(r'\midrule')
    latex_lines.append(f'MM-LSTM-DAE & {enc_mean*100:.1f} $\\pm$ {enc_std*100:.1f} & --- & --- & --- \\\\')

    for bk in baseline_keys:
        comp = results['comparisons'][bk]
        bl_label = comp['label']
        bl_mean = comp['mean']
        bl_std = comp['std']
        delta = enc_mean - bl_mean
        sign = '+' if delta >= 0 else ''
        tp = comp['paired_ttest']['p_value']
        wp = comp['wilcoxon']['p_value']
        t_str = f'{tp:.3f}' if tp is not None and tp >= 0.001 else '$<$0.001' if tp is not None else '---'
        w_str = f'{wp:.3f}' if wp is not None and wp >= 0.001 else '$<$0.001' if wp is not None else '---'
        latex_lines.append(f'{bl_label} & {bl_mean*100:.1f} $\\pm$ {bl_std*100:.1f} & {sign}{delta*100:.1f} & {t_str} & {w_str} \\\\')

    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table}')

    latex_str = '\n'.join(latex_lines)
    out_path = output_dir / 'significance_table.tex'
    with open(out_path, 'w') as f:
        f.write(latex_str)
    print(f"  Saved {out_path.name}")
    print(f"\n{latex_str}")


# ---------------------------------------------------------------------------
# 3. Training Time Comparison
# ---------------------------------------------------------------------------
def generate_timing_comparison(experiments_dir: Path, exp_name: str,
                                output_dir: Path, n_folds: int = 5):
    """Compare training time across models."""
    exp_dir = experiments_dir / exp_name
    feat, w, s = parse_experiment_dir(exp_name)
    feat_label = FEATURE_DISPLAY.get(feat, feat)

    print(f"\n  Config: {feat_label} | w={w} s={s}")
    print(f"  {'Model':<20s} {'Mean (s)':>10s} {'Std (s)':>10s} {'Mean (min)':>10s} {'Speedup':>10s}")
    print(f"  {'-'*64}")

    timing = {}
    encoder_mean = None

    for model_key in MODEL_KEYS:
        label = MODEL_LABELS[model_key]
        times = []
        for fold in range(1, n_folds + 1):
            fold_dir = exp_dir / f'fold_{fold}'
            if model_key == 'encoder':
                _, _, tt = get_encoder_results(fold_dir)
            else:
                _, _, tt = get_baseline_results(fold_dir, model_key)
            if tt is not None:
                times.append(tt)

        if not times:
            print(f"  {label:<20s} {'N/A':>10s}")
            continue

        mean_t = np.mean(times)
        std_t = np.std(times)
        mean_min = mean_t / 60.0

        if model_key == 'encoder':
            encoder_mean = mean_t
            speedup_str = '1.0x'
        elif encoder_mean is not None and mean_t > 0:
            speedup_str = f'{encoder_mean / mean_t:.1f}x'
        else:
            speedup_str = '---'

        print(f"  {label:<20s} {mean_t:>10.1f} {std_t:>10.1f} {mean_min:>10.2f} {speedup_str:>10s}")

        timing[model_key] = {
            'label': label,
            'per_fold_seconds': times,
            'mean_seconds': float(mean_t),
            'std_seconds': float(std_t),
            'mean_minutes': float(mean_min),
        }

    # Save JSON
    out_path = output_dir / 'training_times.json'
    with open(out_path, 'w') as f:
        json.dump({
            'config': exp_name,
            'feature_cfg': feat,
            'window': w,
            'stride': s,
            'models': timing,
        }, f, indent=2)
    print(f"\n  Saved {out_path.name}")

    # LaTeX table
    latex_lines = []
    latex_lines.append(r'\begin{table}[ht]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\caption{Training time comparison per fold (' + feat_label + r', $w$=' + str(w) + r', $s$=' + str(s) + r').}')
    latex_lines.append(r'\label{tab:timing}')
    latex_lines.append(r'\begin{tabular}{lcc}')
    latex_lines.append(r'\toprule')
    latex_lines.append(r'\textbf{Model} & \textbf{Time (s)} & \textbf{Rel.\ to Encoder} \\')
    latex_lines.append(r'\midrule')

    for model_key in MODEL_KEYS:
        if model_key not in timing:
            continue
        t = timing[model_key]
        label = t['label']
        mean_s = t['mean_seconds']
        std_s = t['std_seconds']
        if model_key == 'encoder':
            rel = '1.0$\\times$'
        elif encoder_mean is not None and mean_s > 0:
            rel = f'{encoder_mean / mean_s:.1f}$\\times$ faster'
        else:
            rel = '---'
        latex_lines.append(f'{label} & {mean_s:.1f} $\\pm$ {std_s:.1f} & {rel} \\\\')

    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table}')

    latex_str = '\n'.join(latex_lines)
    out_path = output_dir / 'timing_table.tex'
    with open(out_path, 'w') as f:
        f.write(latex_str)
    print(f"  Saved {out_path.name}")
    print(f"\n{latex_str}")

    # Bar chart
    if HAS_MPL and timing:
        labels = [timing[k]['label'] for k in MODEL_KEYS if k in timing]
        means = [timing[k]['mean_seconds'] for k in MODEL_KEYS if k in timing]
        stds = [timing[k]['std_seconds'] for k in MODEL_KEYS if k in timing]

        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#795548']
        colors = colors[:len(labels)]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title(f'Training Time per Fold  |  {feat_label}  |  w={w} s={s}')

        # Add value labels
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{m:.0f}s', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        out_path = output_dir / 'training_time_comparison.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Post-hoc analysis: confusion matrices, significance tests, timing')
    parser.add_argument('--experiments-dir', type=str, required=True,
                        help='Root directory of CV experiments')
    parser.add_argument('--config', type=str, default=None,
                        help='Specific experiment config name (e.g. full_w64_s16_cv). '
                             'Default: auto-select best encoder config.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: <experiments-dir>/posthoc_analysis)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--only', nargs='+',
                        choices=['confusion', 'significance', 'timing'],
                        default=None,
                        help='Only run specific analyses')
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.is_dir():
        print(f"ERROR: {experiments_dir} not found")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else experiments_dir / 'posthoc_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which experiment config to analyze
    if args.config:
        exp_name = args.config
    else:
        print("Auto-selecting best encoder config...")
        exp_name, best_acc = find_best_encoder_config(experiments_dir, args.n_folds)
        if exp_name is None:
            print("ERROR: No complete experiments found")
            sys.exit(1)
        print(f"  Best: {exp_name} (mean acc = {best_acc:.4f})")

    exp_path = experiments_dir / exp_name
    if not exp_path.is_dir():
        print(f"ERROR: {exp_path} not found")
        sys.exit(1)

    run_all = args.only is None

    # 1. Confusion matrices
    if run_all or 'confusion' in args.only:
        print(f"\n{'='*70}")
        print("1. AGGREGATED CONFUSION MATRICES")
        print(f"{'='*70}")
        generate_confusion_matrices(experiments_dir, exp_name, output_dir, args.n_folds)

    # 2. Statistical significance
    if run_all or 'significance' in args.only:
        print(f"\n{'='*70}")
        print("2. STATISTICAL SIGNIFICANCE TESTS")
        print(f"{'='*70}")
        generate_significance_tests(experiments_dir, exp_name, output_dir, args.n_folds)

    # 3. Training time
    if run_all or 'timing' in args.only:
        print(f"\n{'='*70}")
        print("3. TRAINING TIME COMPARISON")
        print(f"{'='*70}")
        generate_timing_comparison(experiments_dir, exp_name, output_dir, args.n_folds)

    print(f"\n{'='*70}")
    print(f"All outputs saved to {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
