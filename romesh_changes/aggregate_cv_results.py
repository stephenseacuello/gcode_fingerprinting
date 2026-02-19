#!/usr/bin/env python3
"""
Aggregate 5-fold cross-validation results and print a comparison table.

Reads results from:
  <cv_dir>/fold_{k}/encoder/results.json
  <cv_dir>/fold_{k}/baselines/{model}/results.json

Outputs:
  - Per-fold accuracy for each model
  - Mean ± std across folds
  - Normal-class avg vs anomaly-class avg breakdown
  - Saves summary to cv_summary.json

Usage:
    python romesh_changes/aggregate_cv_results.py \
        --cv-dir outputs/experiments/no_proximity_no_color_no_magnetometer_cv \
        --n-folds 5 \
        --output outputs/experiments/no_proximity_no_color_no_magnetometer_cv/cv_summary.json
"""
import json
import argparse
import numpy as np
from pathlib import Path

NORMAL_CLASSES = ['adaptive', 'adaptive150025', 'face', 'face150025', 'pocket', 'pocket150025']
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


def load_encoder_result(fold_dir):
    """
    Load encoder results for one fold.
    Returns (overall_acc, per_class_acc) from val_best checkpoint.
    """
    path = fold_dir / 'encoder' / 'results.json'
    if not path.exists():
        return None, None
    with open(path) as f:
        data = json.load(f)
    # Use val_best checkpoint (proper CV protocol — val is a held-out fold)
    result = data.get('val_best_test', {})
    return result.get('accuracy'), result.get('per_class', {})


def load_baseline_result(fold_dir, model_key):
    """
    Load baseline results for one fold.
    Returns (overall_acc, per_class_acc) from test split.
    """
    path = fold_dir / 'baselines' / model_key / 'results.json'
    if not path.exists():
        return None, None
    with open(path) as f:
        data = json.load(f)
    result = data.get('test', {})
    return result.get('accuracy'), result.get('per_class', {})


def class_avg(per_class, class_list):
    """Average accuracy over a subset of classes, ignoring missing ones."""
    vals = [per_class[c] for c in class_list if c in per_class]
    return np.mean(vals) if vals else float('nan')


def fmt(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '  N/A  '
    return f'{val*100:6.2f}%'


def fmt_mean_std(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return '  N/A ±  N/A '
    return f'{np.mean(vals)*100:6.2f}% ± {np.std(vals)*100:4.2f}%'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv-dir',  type=Path, required=True,
                        help='Base directory containing fold_1 ... fold_N subdirs')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--output',  type=Path, default=None,
                        help='Path to save cv_summary.json')
    args = parser.parse_args()

    folds = list(range(1, args.n_folds + 1))

    # ── Collect all results ──────────────────────────────────────────────────
    # results[model_key] = {
    #   'overall': [acc_fold1, acc_fold2, ...],
    #   'normal':  [acc_fold1, ...],
    #   'anomaly': [acc_fold1, ...],
    #   'per_class': {class_name: [acc_fold1, ...]}
    # }
    all_results = {}

    for model_key, model_label in MODELS:
        accs_overall = []
        accs_normal  = []
        accs_anomaly = []
        accs_per_cls = {c: [] for c in ALL_CLASSES}

        for fold in folds:
            fold_dir = args.cv_dir / f'fold_{fold}'
            if model_key == 'encoder':
                overall, per_class = load_encoder_result(fold_dir)
            else:
                overall, per_class = load_baseline_result(fold_dir, model_key)

            per_class = per_class or {}
            accs_overall.append(overall)
            accs_normal.append(class_avg(per_class, NORMAL_CLASSES))
            accs_anomaly.append(class_avg(per_class, ANOMALY_CLASSES))
            for c in ALL_CLASSES:
                accs_per_cls[c].append(per_class.get(c))

        all_results[model_key] = {
            'label':     model_label,
            'overall':   accs_overall,
            'normal':    accs_normal,
            'anomaly':   accs_anomaly,
            'per_class': accs_per_cls,
        }

    # ── Table 1: Overall accuracy per fold ──────────────────────────────────
    fold_headers = ''.join(f'  Fold {k}  ' for k in folds)
    col_w = 10
    sep = '=' * (24 + args.n_folds * col_w + 24)

    print()
    print(sep)
    print('CROSS-VALIDATION RESULTS — OVERALL TEST ACCURACY')
    print(sep)
    fold_cols = ''.join(('Fold ' + str(k)).rjust(col_w) for k in folds)
    header = f"{'Model':<22}" + fold_cols + f"  {'Mean':>8}  {'Std':>6}"
    print(header)
    print('-' * len(header))

    for model_key, model_label in MODELS:
        r = all_results[model_key]
        row = f"{model_label:<22}"
        for v in r['overall']:
            row += f"{fmt(v):>{col_w}}"
        row += f"  {fmt_mean_std(r['overall']):>16}"
        print(row)

    print(sep)

    # ── Table 2: Normal vs Anomaly breakdown ─────────────────────────────────
    print()
    print(sep)
    print('BREAKDOWN — NORMAL (6-class avg) vs ANOMALY (3-class avg)')
    print(sep)
    print(f"{'Model':<22}  {'Normal Mean':>13}  {'Normal Std':>10}  "
          f"{'Anomaly Mean':>13}  {'Anomaly Std':>10}  {'Gap (N-A)':>10}")
    print('-' * 82)

    for model_key, model_label in MODELS:
        r = all_results[model_key]
        n_vals = [v for v in r['normal']  if v is not None and not np.isnan(v)]
        a_vals = [v for v in r['anomaly'] if v is not None and not np.isnan(v)]
        n_mean = np.mean(n_vals) if n_vals else float('nan')
        a_mean = np.mean(a_vals) if a_vals else float('nan')
        n_std  = np.std(n_vals)  if n_vals else float('nan')
        a_std  = np.std(a_vals)  if a_vals else float('nan')
        gap    = n_mean - a_mean

        def pct(x): return f'{x*100:6.2f}%' if not np.isnan(x) else '  N/A  '

        print(f"{model_label:<22}  {pct(n_mean):>13}  {pct(n_std):>10}  "
              f"{pct(a_mean):>13}  {pct(a_std):>10}  {gap*100:>+9.2f}%")

    print(sep)

    # ── Table 3: Per-class accuracy (mean across folds) ──────────────────────
    print()
    print(sep)
    print('PER-CLASS MEAN ACCURACY (averaged across all folds)')
    print(sep)
    class_col_w = 15
    model_names = [label for _, label in MODELS]
    header2 = f"{'Class':<22}" + ''.join(f'{m:>{class_col_w}}' for m in model_names)
    print(header2)
    print('-' * len(header2))

    for c in ALL_CLASSES:
        marker = '  [anomaly]' if c in ANOMALY_CLASSES else ''
        row = f"{c + marker:<22}"
        for model_key, _ in MODELS:
            vals = [v for v in all_results[model_key]['per_class'][c]
                    if v is not None and not np.isnan(v)]
            mean_val = np.mean(vals) if vals else float('nan')
            row += f"{fmt(mean_val):>{class_col_w}}"
        print(row)

    print(sep)

    # ── Save summary JSON ────────────────────────────────────────────────────
    summary = {}
    for model_key, model_label in MODELS:
        r = all_results[model_key]
        n_vals = [v for v in r['normal']  if v is not None and not np.isnan(v)]
        a_vals = [v for v in r['anomaly'] if v is not None and not np.isnan(v)]
        o_vals = [v for v in r['overall'] if v is not None]
        summary[model_label] = {
            'per_fold_overall': r['overall'],
            'per_fold_normal':  r['normal'],
            'per_fold_anomaly': r['anomaly'],
            'mean_overall':  float(np.mean(o_vals)) if o_vals else None,
            'std_overall':   float(np.std(o_vals))  if o_vals else None,
            'mean_normal':   float(np.mean(n_vals)) if n_vals else None,
            'std_normal':    float(np.std(n_vals))  if n_vals else None,
            'mean_anomaly':  float(np.mean(a_vals)) if a_vals else None,
            'std_anomaly':   float(np.std(a_vals))  if a_vals else None,
            'per_class_mean': {
                c: float(np.mean([v for v in r['per_class'][c]
                                  if v is not None and not np.isnan(v)]) )
                if any(v is not None for v in r['per_class'][c]) else None
                for c in ALL_CLASSES
            },
        }

    output_path = args.output or (args.cv_dir / 'cv_summary.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {output_path}")


if __name__ == '__main__':
    main()
