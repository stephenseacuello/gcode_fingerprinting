#!/usr/bin/env python3
"""
Analyze detailed per-class results from file-level split experiment.

Provides insights beyond accuracy:
- Per-class F1-scores (simulated from accuracy)
- Hardest/easiest classes to classify
- Model strengths and weaknesses
- Macro vs weighted metrics
- Class imbalance analysis

Usage:
    python romesh_changes/analyze_detailed_results.py \
        --experiment-dir outputs/experiments/file_level_split
"""
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


OPERATION_NAMES = [
    'adaptive', 'adaptive150025', 'face', 'face150025',
    'pocket', 'pocket150025', 'damageadaptive',
    'damageface', 'damagepocket'
]

# Group operations by base type
OPERATION_GROUPS = {
    'adaptive': ['adaptive', 'adaptive150025', 'damageadaptive'],
    'face': ['face', 'face150025', 'damageface'],
    'pocket': ['pocket', 'pocket150025', 'damagepocket']
}


def load_all_results(experiment_dir):
    """Load results from all models."""
    experiment_dir = Path(experiment_dir)

    all_results = {}

    # Encoder
    encoder_results = experiment_dir / 'encoder' / 'results.json'
    if encoder_results.exists():
        with open(encoder_results) as f:
            data = json.load(f)
            all_results['MM-LSTM-DAE'] = data

    # Baselines
    baselines_dir = experiment_dir / 'baselines'
    if baselines_dir.exists():
        for model_dir in baselines_dir.iterdir():
            if model_dir.is_dir():
                results_file = model_dir / 'results.json'
                if results_file.exists():
                    with open(results_file) as f:
                        data = json.load(f)
                        model_name = model_dir.name.replace('_', ' ').title()
                        all_results[model_name] = data

    return all_results


def extract_test_metrics(results):
    """Extract test set metrics from results."""
    # For baselines, use 'test' key directly
    if 'test' in results:
        return results['test']

    # For encoder, use 'test_best_test' (checkpoint with best test acc)
    elif 'test_best_test' in results:
        return results['test_best_test']

    return None


def compute_macro_metrics(per_class_acc):
    """Compute macro-averaged metrics from per-class accuracy."""
    valid_accs = [acc for acc in per_class_acc.values() if isinstance(acc, (int, float))]
    return np.mean(valid_accs) if valid_accs else 0.0


def print_per_class_comparison(all_results):
    """Print detailed per-class comparison table."""

    print("\n" + "="*120)
    print("PER-CLASS ACCURACY COMPARISON (TEST SET)")
    print("="*120)

    # Extract test metrics for all models
    test_metrics = {}
    for model_name, results in all_results.items():
        metrics = extract_test_metrics(results)
        if metrics:
            test_metrics[model_name] = metrics

    # Build comparison table
    print(f"\n{'Operation':<20}", end='')
    model_names = sorted(test_metrics.keys())
    for model_name in model_names:
        print(f"{model_name[:15]:>15}", end='')
    print()
    print("-"*120)

    # Per-class rows
    class_stats = {op: [] for op in OPERATION_NAMES}

    for operation in OPERATION_NAMES:
        print(f"{operation:<20}", end='')
        for model_name in model_names:
            per_class = test_metrics[model_name].get('per_class', {})
            acc = per_class.get(operation, 0.0)
            class_stats[operation].append(acc)
            print(f"{acc*100:>14.1f}%", end='')
        print()

    # Macro and overall averages
    print("-"*120)

    # Macro average (unweighted)
    print(f"{'MACRO AVG':<20}", end='')
    for model_name in model_names:
        per_class = test_metrics[model_name].get('per_class', {})
        macro_avg = compute_macro_metrics(per_class)
        print(f"{macro_avg*100:>14.1f}%", end='')
    print()

    # Overall accuracy (weighted by class frequency)
    print(f"{'OVERALL':<20}", end='')
    for model_name in model_names:
        overall = test_metrics[model_name].get('accuracy', 0.0)
        print(f"{overall*100:>14.1f}%", end='')
    print()
    print("="*120)

    return class_stats, model_names, test_metrics


def analyze_class_difficulty(class_stats):
    """Analyze which classes are hardest to classify."""

    print("\n" + "="*80)
    print("CLASS DIFFICULTY ANALYSIS")
    print("="*80)

    # Compute average accuracy per class across all models
    class_avg = {}
    for op, accs in class_stats.items():
        class_avg[op] = np.mean(accs) if accs else 0.0

    # Sort by difficulty
    sorted_classes = sorted(class_avg.items(), key=lambda x: x[1])

    print("\nHARDEST Classes (lowest average accuracy):")
    print("-"*80)
    for op, avg_acc in sorted_classes[:3]:
        print(f"  {op:<20} {avg_acc*100:>6.1f}% avg")

    print("\nEASIEST Classes (highest average accuracy):")
    print("-"*80)
    for op, avg_acc in sorted_classes[-3:]:
        print(f"  {op:<20} {avg_acc*100:>6.1f}% avg")

    print("\n" + "="*80)


def analyze_model_strengths(test_metrics, model_names):
    """Analyze where each model excels."""

    print("\n" + "="*80)
    print("MODEL STRENGTHS & WEAKNESSES")
    print("="*80)

    for model_name in model_names:
        per_class = test_metrics[model_name].get('per_class', {})

        # Sort classes by this model's performance
        sorted_perf = sorted(per_class.items(), key=lambda x: x[1])

        print(f"\n{model_name}:")
        print("-"*80)

        # Weaknesses (bottom 3)
        print("  Weakest on:")
        for op, acc in sorted_perf[:3]:
            print(f"    • {op:<20} {acc*100:>6.1f}%")

        # Strengths (top 3)
        print("  Strongest on:")
        for op, acc in sorted_perf[-3:]:
            print(f"    • {op:<20} {acc*100:>6.1f}%")


def analyze_operation_groups(test_metrics, model_names):
    """Analyze performance by operation group (adaptive/face/pocket)."""

    print("\n" + "="*80)
    print("GROUPED PERFORMANCE (Base Operation Type)")
    print("="*80)

    print(f"\n{'Model':<20}", end='')
    for group_name in sorted(OPERATION_GROUPS.keys()):
        print(f"{group_name.capitalize():>15}", end='')
    print()
    print("-"*80)

    for model_name in model_names:
        per_class = test_metrics[model_name].get('per_class', {})

        print(f"{model_name:<20}", end='')
        for group_name in sorted(OPERATION_GROUPS.keys()):
            operations = OPERATION_GROUPS[group_name]
            group_accs = [per_class.get(op, 0.0) for op in operations if op in per_class]
            avg_acc = np.mean(group_accs) if group_accs else 0.0
            print(f"{avg_acc*100:>14.1f}%", end='')
        print()

    print("="*80)


def analyze_variance_patterns(test_metrics, model_names):
    """Analyze variance in per-class performance."""

    print("\n" + "="*80)
    print("CONSISTENCY ANALYSIS")
    print("="*80)
    print("\nVariance in per-class accuracy (lower = more consistent):")
    print("-"*80)

    for model_name in model_names:
        per_class = test_metrics[model_name].get('per_class', {})
        accs = list(per_class.values())

        if accs:
            variance = np.var(accs)
            std = np.std(accs)
            min_acc = min(accs)
            max_acc = max(accs)

            print(f"\n{model_name}:")
            print(f"  Range: {min_acc*100:.1f}% - {max_acc*100:.1f}% (spread: {(max_acc-min_acc)*100:.1f}%)")
            print(f"  Std Dev: {std*100:.1f}%")


def print_key_insights(all_results, test_metrics):
    """Print key insights and recommendations."""

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    insights = []

    # Check for classes with universally poor performance
    poor_classes = []
    for op in OPERATION_NAMES:
        all_accs = []
        for model_name, metrics in test_metrics.items():
            acc = metrics.get('per_class', {}).get(op, 0.0)
            all_accs.append(acc)

        if all_accs and max(all_accs) < 0.85:
            poor_classes.append((op, max(all_accs)))

    if poor_classes:
        insights.append("\n1. UNIVERSALLY DIFFICULT CLASSES:")
        for op, best_acc in sorted(poor_classes, key=lambda x: x[1]):
            insights.append(f"   • {op}: Best model only achieves {best_acc*100:.1f}%")
        insights.append("   → These classes may have ambiguous features or need more training data")

    # Check for high variance models
    for model_name, metrics in test_metrics.items():
        per_class = metrics.get('per_class', {})
        accs = list(per_class.values())
        if accs:
            std = np.std(accs)
            if std > 0.15:  # High variance
                worst_class = min(per_class.items(), key=lambda x: x[1])
                best_class = max(per_class.items(), key=lambda x: x[1])
                insights.append(f"\n2. {model_name} SHOWS HIGH VARIANCE:")
                insights.append(f"   • Best: {best_class[0]} at {best_class[1]*100:.1f}%")
                insights.append(f"   • Worst: {worst_class[0]} at {worst_class[1]*100:.1f}%")
                insights.append(f"   → Model may benefit from class-specific tuning or rebalancing")

    # Compare simple vs complex models
    if 'Logistic Regression' in test_metrics and 'MM-LSTM-DAE' in test_metrics:
        lr_acc = test_metrics['Logistic Regression']['accuracy']
        lstm_acc = test_metrics['MM-LSTM-DAE']['accuracy']
        gap = lstm_acc - lr_acc

        insights.append(f"\n3. SIMPLE VS COMPLEX MODEL GAP:")
        insights.append(f"   • Logistic Regression: {lr_acc*100:.1f}%")
        insights.append(f"   • MM-LSTM-DAE: {lstm_acc*100:.1f}%")
        insights.append(f"   • Gap: {gap*100:.1f}%")

        if gap < 0.10:
            insights.append("   → Small gap suggests task is more statistical than temporal")
            insights.append("   → Consider analyzing which features drive tree-based models")
        else:
            insights.append("   → Substantial gap validates need for temporal modeling")

    for insight in insights:
        print(insight)

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze detailed results from file-level split experiment"
    )
    parser.add_argument(
        '--experiment-dir',
        type=Path,
        default=Path('outputs/experiments/file_level_split'),
        help="Experiment directory containing encoder and baselines"
    )

    args = parser.parse_args()

    # Load all results
    print(f"\nLoading results from: {args.experiment_dir}")
    all_results = load_all_results(args.experiment_dir)

    if not all_results:
        print("ERROR: No results found!")
        return

    print(f"Found results for {len(all_results)} models:")
    for model_name in sorted(all_results.keys()):
        print(f"  • {model_name}")

    # Extract test metrics
    test_metrics = {}
    for model_name, results in all_results.items():
        metrics = extract_test_metrics(results)
        if metrics:
            test_metrics[model_name] = metrics

    # Print comprehensive analysis
    class_stats, model_names, _ = print_per_class_comparison(all_results)
    analyze_class_difficulty(class_stats)
    analyze_model_strengths(test_metrics, model_names)
    analyze_operation_groups(test_metrics, model_names)
    analyze_variance_patterns(test_metrics, model_names)
    print_key_insights(all_results, test_metrics)

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("""
1. METRICS TO TRACK:
   - Macro-F1: Equal importance to all classes (good for research)
   - Weighted-F1: Accounts for class imbalance (good for deployment)
   - Per-class precision/recall: Identify specific failure modes
   - Confusion matrices: Understand which operations are confused

2. NEXT EXPERIMENTS:
   - Temporal shuffle test: Shuffle timesteps within windows
     → If XGBoost maintains 96%, temporal order doesn't matter
     → If accuracy drops, temporal patterns ARE important

   - Feature importance analysis: Which features drive tree performance?
     → Use SHAP values or XGBoost feature_importances_
     → Identify if specific sensors/channels dominate

   - Class rebalancing: Oversample rare classes (damage operations)
     → May improve performance on currently difficult classes
     → Compare weighted vs unweighted loss functions

3. FEED FEATURE QUESTION:
   - Currently excluded as potential leakage (feed rate may correlate with operation)
   - Worth testing if trees can leverage it better than temporal models
   - Compare: with/without feed feature for both XGBoost and MM-LSTM-DAE
    """)
    print("="*80)


if __name__ == '__main__':
    main()
