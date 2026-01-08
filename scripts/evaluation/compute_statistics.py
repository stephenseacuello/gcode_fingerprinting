#!/usr/bin/env python3
"""
Compute Statistical Analysis for G-code Fingerprinting Results.

This script computes:
1. Confidence intervals (95% CI) for multi-seed runs
2. Standard deviation and standard error
3. Statistical significance tests (paired t-test, McNemar's test)
4. Effect size (Cohen's d)
5. Summary tables in LaTeX and Markdown format

Usage:
    python scripts/compute_statistics.py --results-dir outputs/multiseed/run_20251229
    python scripts/compute_statistics.py --compare outputs/model_a outputs/model_b
    python scripts/compute_statistics.py --ablations outputs/ablations

Author: Claude Code
Date: December 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
from scipy import stats


def load_multiseed_results(results_dir: str) -> Dict:
    """Load results from multi-seed training runs."""
    results_dir = Path(results_dir)

    # Check for statistics.json (pre-computed)
    stats_path = results_dir / 'statistics.json'
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)

    # Otherwise, aggregate from seed directories
    seed_results = []
    for seed_dir in sorted(results_dir.glob('seed_*')):
        results_path = seed_dir / 'results.json'
        if results_path.exists():
            with open(results_path) as f:
                r = json.load(f)
                seed = int(seed_dir.name.replace('seed_', ''))
                seed_results.append({
                    'seed': seed,
                    'token_accuracy': r.get('val_token_accuracy', r.get('token_accuracy', 0)),
                    'sequence_accuracy': r.get('val_sequence_accuracy', r.get('sequence_accuracy', 0)),
                    'operation_accuracy': r.get('val_operation_accuracy', r.get('operation_accuracy', 0)),
                })

    return {'per_seed_results': seed_results}


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float, float]:
    """
    Compute mean, std, and confidence interval.

    Returns:
        mean, std, ci_lower, ci_upper
    """
    n = len(values)
    if n < 2:
        return np.mean(values), 0, np.mean(values), np.mean(values)

    mean = np.mean(values)
    std = np.std(values, ddof=1)
    se = std / np.sqrt(n)

    # t-distribution for small samples
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    ci = t_val * se

    return mean, std, mean - ci, mean + ci


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def paired_t_test(
    model1_results: List[float],
    model2_results: List[float],
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Perform paired t-test.

    Returns:
        t_statistic, p_value
    """
    if len(model1_results) != len(model2_results):
        raise ValueError("Results lists must have same length for paired t-test")

    t_stat, p_val = stats.ttest_rel(model1_results, model2_results, alternative=alternative)
    return t_stat, p_val


def mcnemar_test(
    contingency_table: np.ndarray
) -> Tuple[float, float]:
    """
    Perform McNemar's test for paired nominal data.

    Args:
        contingency_table: 2x2 table where:
            [0,0] = both correct
            [0,1] = model1 correct, model2 wrong
            [1,0] = model1 wrong, model2 correct
            [1,1] = both wrong

    Returns:
        chi2_statistic, p_value
    """
    b = contingency_table[0, 1]  # model1 correct, model2 wrong
    c = contingency_table[1, 0]  # model1 wrong, model2 correct

    # Use exact binomial test for small samples
    if b + c < 25:
        # Exact binomial test
        from scipy.stats import binom_test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                p_val = stats.binomtest(b, b + c, 0.5, alternative='two-sided').pvalue
            except AttributeError:
                # Older scipy
                p_val = stats.binom_test(b, b + c, 0.5, alternative='two-sided')
        return None, p_val
    else:
        # Chi-squared approximation
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_val = 1 - stats.chi2.cdf(chi2, 1)
        return chi2, p_val


def generate_latex_table(
    results: Dict[str, Dict],
    caption: str = "Model Comparison Results",
    label: str = "tab:results"
) -> str:
    """Generate LaTeX table from results."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Token Acc. (\%) & Seq. Acc. (\%) & 95\% CI & Params \\",
        r"\midrule",
    ]

    for name, data in results.items():
        if name == 'metadata':
            continue

        token_acc = data.get('token_accuracy', {})
        if isinstance(token_acc, dict):
            mean = token_acc.get('mean', 0) * 100
            ci = token_acc.get('ci_95', 0) * 100
            token_str = f"{mean:.2f} $\\pm$ {ci:.2f}"
        else:
            token_str = f"{token_acc * 100:.2f}"

        seq_acc = data.get('sequence_accuracy', {})
        if isinstance(seq_acc, dict):
            mean = seq_acc.get('mean', 0) * 100
            seq_str = f"{mean:.2f}"
        else:
            seq_str = f"{seq_acc * 100:.2f}" if seq_acc else "N/A"

        params = data.get('params', data.get('total_params', 'N/A'))
        if isinstance(params, int) and params > 1000:
            params_str = f"{params/1000:.1f}K"
        else:
            params_str = str(params)

        ci_str = f"$\\pm${ci:.2f}" if isinstance(token_acc, dict) else "—"

        lines.append(f"{name} & {token_str} & {seq_str} & {ci_str} & {params_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_markdown_table(results: Dict[str, Dict]) -> str:
    """Generate Markdown table from results."""

    lines = [
        "| Model | Token Acc. (%) | Seq. Acc. (%) | 95% CI | Params |",
        "|-------|----------------|---------------|--------|--------|",
    ]

    for name, data in results.items():
        if name == 'metadata':
            continue

        token_acc = data.get('token_accuracy', {})
        if isinstance(token_acc, dict):
            mean = token_acc.get('mean', 0) * 100
            ci = token_acc.get('ci_95', 0) * 100
            token_str = f"{mean:.2f}"
            ci_str = f"±{ci:.2f}"
        else:
            token_str = f"{token_acc * 100:.2f}"
            ci_str = "—"

        seq_acc = data.get('sequence_accuracy', {})
        if isinstance(seq_acc, dict):
            seq_str = f"{seq_acc.get('mean', 0) * 100:.2f}"
        else:
            seq_str = f"{seq_acc * 100:.2f}" if seq_acc else "N/A"

        params = data.get('params', data.get('total_params', 'N/A'))
        if isinstance(params, int) and params > 1000:
            params_str = f"{params/1000:.1f}K"
        else:
            params_str = str(params)

        lines.append(f"| {name} | {token_str} | {seq_str} | {ci_str} | {params_str} |")

    return "\n".join(lines)


def analyze_ablations(ablation_dir: str) -> Dict:
    """Analyze ablation study results."""

    ablation_dir = Path(ablation_dir)
    summary_path = ablation_dir / 'ablation_summary.json'

    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
    else:
        data = {}
        for subdir in ablation_dir.iterdir():
            if subdir.is_dir():
                results_path = subdir / 'results.json'
                if results_path.exists():
                    with open(results_path) as f:
                        data[subdir.name] = json.load(f)

    # Compute improvements/degradations
    analysis = {}

    for ablation_name, results in data.items():
        if ablation_name == 'metadata':
            continue

        if isinstance(results, dict):
            # Find baseline and ablated variants
            variants = list(results.keys())
            if len(variants) >= 2:
                # Assume first is baseline (with feature), second is ablated (without)
                baseline_key = variants[0]
                ablated_key = variants[1]

                baseline_acc = results[baseline_key].get('val_token_accuracy',
                              results[baseline_key].get('token_accuracy', 0))
                ablated_acc = results[ablated_key].get('val_token_accuracy',
                              results[ablated_key].get('token_accuracy', 0))

                if isinstance(baseline_acc, (int, float)) and isinstance(ablated_acc, (int, float)):
                    improvement = (baseline_acc - ablated_acc) * 100
                    analysis[ablation_name] = {
                        'baseline': baseline_acc * 100,
                        'ablated': ablated_acc * 100,
                        'improvement': improvement,
                        'pct_improvement': improvement / (ablated_acc * 100) * 100 if ablated_acc > 0 else 0,
                    }

    return analysis


def compare_models(model1_dir: str, model2_dir: str) -> Dict:
    """Compare two model results statistically."""

    results1 = load_multiseed_results(model1_dir)
    results2 = load_multiseed_results(model2_dir)

    # Extract token accuracies
    if 'per_seed_results' in results1:
        accs1 = [r['token_accuracy'] for r in results1['per_seed_results']]
    elif 'token_accuracy' in results1 and 'values' in results1['token_accuracy']:
        accs1 = results1['token_accuracy']['values']
    else:
        accs1 = [results1.get('token_accuracy', 0)]

    if 'per_seed_results' in results2:
        accs2 = [r['token_accuracy'] for r in results2['per_seed_results']]
    elif 'token_accuracy' in results2 and 'values' in results2['token_accuracy']:
        accs2 = results2['token_accuracy']['values']
    else:
        accs2 = [results2.get('token_accuracy', 0)]

    # Compute statistics
    mean1, std1, ci1_low, ci1_high = compute_confidence_interval(accs1)
    mean2, std2, ci2_low, ci2_high = compute_confidence_interval(accs2)

    comparison = {
        'model1': {
            'dir': model1_dir,
            'mean': mean1,
            'std': std1,
            'ci_95': (ci1_low, ci1_high),
            'n_seeds': len(accs1),
        },
        'model2': {
            'dir': model2_dir,
            'mean': mean2,
            'std': std2,
            'ci_95': (ci2_low, ci2_high),
            'n_seeds': len(accs2),
        },
        'difference': mean1 - mean2,
    }

    # Statistical tests (only if we have paired data)
    if len(accs1) == len(accs2) and len(accs1) >= 2:
        t_stat, p_val = paired_t_test(accs1, accs2)
        cohens_d = compute_cohens_d(accs1, accs2)

        comparison['t_test'] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'significant_05': p_val < 0.05,
            'significant_01': p_val < 0.01,
        }
        comparison['cohens_d'] = cohens_d

        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interpretation = 'negligible'
        elif abs(cohens_d) < 0.5:
            effect_interpretation = 'small'
        elif abs(cohens_d) < 0.8:
            effect_interpretation = 'medium'
        else:
            effect_interpretation = 'large'
        comparison['effect_interpretation'] = effect_interpretation

    return comparison


def main():
    parser = argparse.ArgumentParser(description='Compute statistical analysis')
    parser.add_argument('--results-dir', type=str,
                        help='Directory with multi-seed results')
    parser.add_argument('--compare', nargs=2, metavar=('MODEL1', 'MODEL2'),
                        help='Compare two model result directories')
    parser.add_argument('--ablations', type=str,
                        help='Directory with ablation results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for analysis')
    parser.add_argument('--format', choices=['json', 'latex', 'markdown', 'all'],
                        default='all', help='Output format')
    args = parser.parse_args()

    results = {}

    # Analyze multi-seed results
    if args.results_dir:
        print("="*60)
        print("MULTI-SEED ANALYSIS")
        print("="*60)

        data = load_multiseed_results(args.results_dir)

        if 'per_seed_results' in data:
            token_accs = [r['token_accuracy'] for r in data['per_seed_results']]
            seq_accs = [r['sequence_accuracy'] for r in data['per_seed_results']]

            mean, std, ci_low, ci_high = compute_confidence_interval(token_accs)

            print(f"\nToken Accuracy:")
            print(f"  Mean: {mean*100:.2f}%")
            print(f"  Std:  {std*100:.2f}%")
            print(f"  95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
            print(f"  N seeds: {len(token_accs)}")

            mean_seq, std_seq, _, _ = compute_confidence_interval(seq_accs)
            print(f"\nSequence Accuracy:")
            print(f"  Mean: {mean_seq*100:.2f}%")
            print(f"  Std:  {std_seq*100:.2f}%")

            results['multiseed'] = {
                'token_accuracy': {
                    'mean': mean,
                    'std': std,
                    'ci_95_low': ci_low,
                    'ci_95_high': ci_high,
                    'values': token_accs,
                },
                'sequence_accuracy': {
                    'mean': mean_seq,
                    'std': std_seq,
                    'values': seq_accs,
                },
                'n_seeds': len(token_accs),
            }

    # Compare two models
    if args.compare:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        comparison = compare_models(args.compare[0], args.compare[1])

        print(f"\nModel 1: {args.compare[0]}")
        print(f"  Mean: {comparison['model1']['mean']*100:.2f}%")
        print(f"  95% CI: [{comparison['model1']['ci_95'][0]*100:.2f}%, "
              f"{comparison['model1']['ci_95'][1]*100:.2f}%]")

        print(f"\nModel 2: {args.compare[1]}")
        print(f"  Mean: {comparison['model2']['mean']*100:.2f}%")
        print(f"  95% CI: [{comparison['model2']['ci_95'][0]*100:.2f}%, "
              f"{comparison['model2']['ci_95'][1]*100:.2f}%]")

        print(f"\nDifference: {comparison['difference']*100:+.2f}%")

        if 't_test' in comparison:
            print(f"\nStatistical Test:")
            print(f"  t-statistic: {comparison['t_test']['t_statistic']:.4f}")
            print(f"  p-value: {comparison['t_test']['p_value']:.4f}")
            print(f"  Significant (p<0.05): {comparison['t_test']['significant_05']}")
            print(f"  Cohen's d: {comparison['cohens_d']:.3f} ({comparison['effect_interpretation']})")

        results['comparison'] = comparison

    # Analyze ablations
    if args.ablations:
        print("\n" + "="*60)
        print("ABLATION ANALYSIS")
        print("="*60)

        ablation_analysis = analyze_ablations(args.ablations)

        print(f"\n{'Ablation':<30} {'Baseline':>10} {'Ablated':>10} {'Δ':>10}")
        print("-" * 65)

        for name, data in ablation_analysis.items():
            print(f"{name:<30} {data['baseline']:>9.2f}% {data['ablated']:>9.2f}% "
                  f"{data['improvement']:>+9.2f}%")

        results['ablations'] = ablation_analysis

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else Path('outputs/statistics')
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ['json', 'all']:
        with open(output_dir / 'statistics.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nJSON saved to: {output_dir / 'statistics.json'}")

    if args.format in ['latex', 'all'] and results:
        latex = generate_latex_table(results)
        with open(output_dir / 'table.tex', 'w') as f:
            f.write(latex)
        print(f"LaTeX saved to: {output_dir / 'table.tex'}")

    if args.format in ['markdown', 'all'] and results:
        md = generate_markdown_table(results)
        with open(output_dir / 'table.md', 'w') as f:
            f.write(md)
        print(f"Markdown saved to: {output_dir / 'table.md'}")


if __name__ == '__main__':
    main()
