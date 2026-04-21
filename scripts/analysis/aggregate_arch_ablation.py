#!/usr/bin/env python3
"""
Aggregate Architecture Ablation Results.

Collects results from 7 ablation conditions x 5 folds and produces:
  - Summary JSON with per-condition mean/std/per-fold accuracy
  - Printed comparison table
  - LaTeX table fragment for paper insertion

Usage:
    python scripts/analysis/aggregate_arch_ablation.py \
        --results-dir outputs/experiments_2026_03_18/arch_ablation_cv \
        --conditions "full,single_proj,no_gates,no_crossattn,no_denoise,no_dtae,minimal" \
        --n-folds 5 \
        --output outputs/experiments_2026_03_18/arch_ablation_cv/arch_ablation_summary.json
"""
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

CLASS_NAMES_9 = ['adaptive', 'adaptive150025', 'face', 'face150025',
                 'pocket', 'pocket150025', 'damageadaptive', 'damageface', 'damagepocket']

CONDITION_LABELS = {
    'full': 'Full MM-DTAE-LSTM',
    'single_proj': 'Single Shared Proj',
    'no_gates': 'Drop Modality Gates',
    'no_crossattn': 'Drop Cross-Modal Attn',
    'no_denoise': 'Drop Denoising',
    'no_dtae': 'Drop DTAE',
    'minimal': 'Minimal (Proj→LSTM)',
}


def load_fold_result(results_dir, condition, fold):
    """Load results.json for one condition/fold. Returns dict or None."""
    path = results_dir / condition / f'fold_{fold}' / 'results.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt_pct(val):
    if val is None or np.isnan(val):
        return '  N/A  '
    return f'{val * 100:.1f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--conditions', type=str,
                        default='full,single_proj,no_gates,no_crossattn,no_denoise,no_dtae,minimal')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    conditions = args.conditions.split(',')
    folds = list(range(1, args.n_folds + 1))

    summary = {}
    full_mean = None

    # ── Collect results ──────────────────────────────────────────────────────
    for cond in conditions:
        fold_accs = []
        fold_f1s = []
        per_class_accs = {c: [] for c in CLASS_NAMES_9}
        n_params = None

        for fold in folds:
            data = load_fold_result(args.results_dir, cond, fold)
            if data is None:
                print(f"WARNING: Missing results for {cond}/fold_{fold}")
                fold_accs.append(None)
                fold_f1s.append(None)
                continue

            # Use val_best checkpoint (proper CV protocol)
            result = data.get('val_best_test', {})
            acc = result.get('accuracy')
            fold_accs.append(acc)

            # Per-class accuracies
            pc = result.get('per_class', {})
            for c in CLASS_NAMES_9:
                per_class_accs[c].append(pc.get(c))

            # Compute macro F1 from confusion matrix if available
            cm = result.get('confusion_matrix')
            if cm is not None:
                cm = np.array(cm)
                y_true, y_pred = [], []
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        y_true.extend([i] * cm[i, j])
                        y_pred.extend([j] * cm[i, j])
                macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                fold_f1s.append(macro_f1)
            else:
                fold_f1s.append(None)

            # Parameter count (same across folds)
            if n_params is None:
                n_params = data.get('config', {}).get('n_params')

        valid_accs = [a for a in fold_accs if a is not None]
        valid_f1s = [f for f in fold_f1s if f is not None]

        entry = {
            'label': CONDITION_LABELS.get(cond, cond),
            'n_params': n_params,
            'per_fold_acc': fold_accs,
            'per_fold_f1': fold_f1s,
            'mean_acc': float(np.mean(valid_accs)) if valid_accs else None,
            'std_acc': float(np.std(valid_accs)) if valid_accs else None,
            'mean_f1': float(np.mean(valid_f1s)) if valid_f1s else None,
            'std_f1': float(np.std(valid_f1s)) if valid_f1s else None,
            'per_class_acc': {},
        }

        for c in CLASS_NAMES_9:
            vals = [v for v in per_class_accs[c] if v is not None]
            entry['per_class_acc'][c] = {
                'mean': float(np.mean(vals)) if vals else None,
                'std': float(np.std(vals)) if vals else None,
            }

        summary[cond] = entry

        if cond == 'full':
            full_mean = entry['mean_acc']

    # ── Print tables ─────────────────────────────────────────────────────────
    sep = '=' * 100
    print()
    print(sep)
    print('ARCHITECTURE ABLATION — ACCURACY (val-best checkpoint)')
    print(sep)

    # Table C: Per-fold accuracy
    fold_cols = ''.join(f'  Fold {k}  ' for k in folds)
    header = f"{'#':>2} {'Condition':<24}{fold_cols}  {'Mean±Std':>16}  {'Δ':>6}"
    print(header)
    print('-' * len(header))

    for i, cond in enumerate(conditions, 1):
        e = summary[cond]
        row = f"{i:>2} {e['label']:<24}"
        for acc in e['per_fold_acc']:
            row += f"  {fmt_pct(acc):>6}%"
        mean_str = f"{fmt_pct(e['mean_acc'])}±{fmt_pct(e['std_acc'])}%"
        delta = ''
        if full_mean is not None and e['mean_acc'] is not None and cond != 'full':
            d = (e['mean_acc'] - full_mean) * 100
            delta = f"{d:+.1f}"
        row += f"  {mean_str:>16}  {delta:>6}"
        print(row)

    print(sep)

    # Table D: Per-fold Macro F1
    print()
    print(sep)
    print('ARCHITECTURE ABLATION — MACRO F1 (val-best checkpoint)')
    print(sep)

    full_f1_mean = summary.get('full', {}).get('mean_f1')
    header = f"{'#':>2} {'Condition':<24}{fold_cols}  {'Mean±Std':>16}  {'Δ':>6}"
    print(header)
    print('-' * len(header))

    for i, cond in enumerate(conditions, 1):
        e = summary[cond]
        row = f"{i:>2} {e['label']:<24}"
        for f1 in e['per_fold_f1']:
            row += f"  {fmt_pct(f1):>6}%"
        mean_str = f"{fmt_pct(e['mean_f1'])}±{fmt_pct(e['std_f1'])}%"
        delta = ''
        if full_f1_mean is not None and e['mean_f1'] is not None and cond != 'full':
            d = (e['mean_f1'] - full_f1_mean) * 100
            delta = f"{d:+.1f}"
        row += f"  {mean_str:>16}  {delta:>6}"
        print(row)

    print(sep)

    # Parameter count summary
    print()
    print("PARAMETER COUNTS:")
    for cond in conditions:
        e = summary[cond]
        params = e.get('n_params')
        delta_p = ''
        if params and summary.get('full', {}).get('n_params'):
            dp = params - summary['full']['n_params']
            delta_p = f" ({dp:+,d})" if dp != 0 else ""
        print(f"  {e['label']:<24} {params:>12,d}{delta_p}" if params else
              f"  {e['label']:<24}          N/A")

    # ── Save JSON ────────────────────────────────────────────────────────────
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {args.output}")

    # ── LaTeX table ──────────────────────────────────────────────────────────
    latex_path = args.results_dir / 'arch_ablation_table.tex'
    with open(latex_path, 'w') as f:
        f.write('% Architecture Ablation Results — Auto-generated\n')
        f.write('\\begin{table}[htbp]\n')
        f.write('\\centering\n')
        f.write('\\caption{Architecture ablation: drop-one-out analysis of MM-DTAE-LSTM encoder components.}\n')
        f.write('\\label{tab:arch_ablation}\n')
        f.write('\\begin{tabular}{clcc}\n')
        f.write('\\toprule\n')
        f.write('\\# & Condition & Accuracy (\\%) & Macro F1 (\\%) \\\\\n')
        f.write('\\midrule\n')
        for i, cond in enumerate(conditions, 1):
            e = summary[cond]
            acc_str = f"${fmt_pct(e['mean_acc'])} \\pm {fmt_pct(e['std_acc'])}$" if e['mean_acc'] else 'N/A'
            f1_str = f"${fmt_pct(e['mean_f1'])} \\pm {fmt_pct(e['std_f1'])}$" if e['mean_f1'] else 'N/A'
            label = e['label'].replace('→', '$\\rightarrow$')
            f.write(f'{i} & {label} & {acc_str} & {f1_str} \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    print(f"Saved LaTeX table to {latex_path}")


if __name__ == '__main__':
    main()
