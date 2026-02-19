#!/usr/bin/env python3
"""
Comprehensive evaluation with multiple metrics beyond accuracy.

Analyzes trained models with:
- Per-class precision, recall, F1-score
- Macro/weighted averages
- Confusion matrix analysis
- Top-K accuracy
- Classification report

Usage:
    python romesh_changes/evaluate_comprehensive_metrics.py \
        --data-dir outputs/experiments/file_level_split/preprocessed \
        --model-dirs outputs/experiments/file_level_split/baselines/* \
                     outputs/experiments/file_level_split/encoder
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score
)


# Operation type mapping
OPERATION_NAMES = [
    'adaptive', 'adaptive150025', 'damageadaptive',
    'face', 'face150025', 'damageface',
    'pocket', 'pocket150025', 'damagepocket'
]


def load_npz_data(npz_path):
    """Load preprocessed data from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)

    continuous = data['continuous']
    labels = data['labels']

    return continuous, labels


def load_predictions(model_dir):
    """Load predictions from model directory."""
    model_dir = Path(model_dir)

    # Try different prediction file names
    pred_files = [
        'test_predictions.npy',
        'predictions.npy',
        'test_pred.npy'
    ]

    for pred_file in pred_files:
        pred_path = model_dir / pred_file
        if pred_path.exists():
            return np.load(pred_path)

    return None


def compute_comprehensive_metrics(y_true, y_pred, y_pred_proba=None):
    """Compute comprehensive metrics beyond accuracy."""

    metrics = {}

    # Basic accuracy
    metrics['accuracy'] = np.mean(y_true == y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics['per_class'] = {}
    for i, class_name in enumerate(OPERATION_NAMES):
        if i < len(precision):
            metrics['per_class'][class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }

    # Macro averages (equal weight to each class)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['macro_avg'] = {
        'precision': float(p_macro),
        'recall': float(r_macro),
        'f1_score': float(f1_macro)
    }

    # Weighted averages (weight by class frequency)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['weighted_avg'] = {
        'precision': float(p_weighted),
        'recall': float(r_weighted),
        'f1_score': float(f1_weighted)
    }

    # Top-2 and Top-3 accuracy (if probabilities available)
    if y_pred_proba is not None and y_pred_proba.ndim == 2:
        try:
            metrics['top_2_accuracy'] = float(top_k_accuracy_score(y_true, y_pred_proba, k=2))
            metrics['top_3_accuracy'] = float(top_k_accuracy_score(y_true, y_pred_proba, k=3))
        except:
            pass

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics


def analyze_confusion_patterns(cm, class_names):
    """Analyze common confusion patterns."""
    patterns = []

    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                conf_rate = cm[i, j] / cm[i].sum() if cm[i].sum() > 0 else 0
                if conf_rate > 0.1:  # More than 10% confusion
                    patterns.append({
                        'true_class': class_names[i],
                        'predicted_class': class_names[j],
                        'count': int(cm[i, j]),
                        'rate': float(conf_rate)
                    })

    # Sort by confusion rate
    patterns.sort(key=lambda x: x['rate'], reverse=True)
    return patterns


def print_comparison_table(all_metrics, model_names):
    """Print comprehensive comparison table."""

    print("\n" + "="*100)
    print("COMPREHENSIVE METRICS COMPARISON")
    print("="*100)

    # Overall metrics
    print(f"\n{'Model':<25} {'Accuracy':>10} {'Macro-F1':>10} {'Weighted-F1':>10} {'Top-2':>10}")
    print("-"*100)

    for model_name in sorted(all_metrics.keys()):
        metrics = all_metrics[model_name]
        acc = metrics.get('accuracy', 0) * 100
        macro_f1 = metrics.get('macro_avg', {}).get('f1_score', 0) * 100
        weighted_f1 = metrics.get('weighted_avg', {}).get('f1_score', 0) * 100
        top2 = metrics.get('top_2_accuracy', 0) * 100 if 'top_2_accuracy' in metrics else 0

        if top2 > 0:
            print(f"{model_name:<25} {acc:>9.2f}% {macro_f1:>9.2f}% {weighted_f1:>9.2f}% {top2:>9.2f}%")
        else:
            print(f"{model_name:<25} {acc:>9.2f}% {macro_f1:>9.2f}% {weighted_f1:>9.2f}% {'N/A':>10}")

    # Per-class F1 scores
    print("\n" + "="*100)
    print("PER-CLASS F1-SCORES")
    print("="*100)

    # Get all operation types
    operation_types = set()
    for metrics in all_metrics.values():
        operation_types.update(metrics.get('per_class', {}).keys())
    operation_types = sorted(operation_types)

    # Header
    header = f"{'Model':<25}"
    for op in operation_types:
        header += f" {op[:12]:>12}"
    print(header)
    print("-"*100)

    # Data rows
    for model_name in sorted(all_metrics.keys()):
        metrics = all_metrics[model_name]
        row = f"{model_name:<25}"
        for op in operation_types:
            f1 = metrics.get('per_class', {}).get(op, {}).get('f1_score', 0) * 100
            row += f" {f1:>11.1f}%"
        print(row)

    # Class distribution
    print("\n" + "="*100)
    print("TEST SET CLASS DISTRIBUTION")
    print("="*100)

    # Use first model's metrics for support counts
    first_model = list(all_metrics.values())[0]
    per_class = first_model.get('per_class', {})

    total_samples = sum(info['support'] for info in per_class.values())

    for op in sorted(per_class.keys()):
        support = per_class[op]['support']
        percentage = (support / total_samples * 100) if total_samples > 0 else 0
        print(f"  {op:<20} {support:>5} samples ({percentage:>5.1f}%)")
    print(f"  {'TOTAL':<20} {total_samples:>5} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation with multiple metrics"
    )
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model-dirs', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)

    args = parser.parse_args()

    # Load test data
    test_path = args.data_dir / 'test_sequences.npz'
    print(f"\nLoading test data from: {test_path}")

    X_test, y_test = load_npz_data(test_path)
    print(f"Test set: {X_test.shape[0]} samples, {len(np.unique(y_test))} classes")

    # Evaluate each model
    all_metrics = {}

    for model_dir in args.model_dirs:
        model_name = model_dir.name

        # Try to load predictions
        y_pred = load_predictions(model_dir)

        if y_pred is None:
            print(f"\n⚠️  Skipping {model_name}: No predictions found")
            continue

        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*80}")

        # Compute metrics
        metrics = compute_comprehensive_metrics(y_test, y_pred)
        all_metrics[model_name] = metrics

        # Print classification report
        print(classification_report(
            y_test, y_pred,
            target_names=OPERATION_NAMES[:len(np.unique(y_test))],
            zero_division=0
        ))

        # Analyze confusion patterns
        cm = np.array(metrics['confusion_matrix'])
        patterns = analyze_confusion_patterns(cm, OPERATION_NAMES[:len(cm)])

        if patterns:
            print("\nMajor Confusion Patterns (>10%):")
            for pattern in patterns[:5]:
                print(f"  {pattern['true_class']} → {pattern['predicted_class']}: "
                      f"{pattern['count']} samples ({pattern['rate']*100:.1f}%)")

    # Print comparison table
    if len(all_metrics) > 1:
        print_comparison_table(all_metrics, list(all_metrics.keys()))

    # Save results
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        output_path = args.output_dir / 'comprehensive_metrics.json'
        with open(output_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n✅ Saved comprehensive metrics to: {output_path}")


if __name__ == '__main__':
    main()
