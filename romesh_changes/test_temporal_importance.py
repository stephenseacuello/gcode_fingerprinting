#!/usr/bin/env python3
"""
Test whether temporal order matters for classification performance.

Experiment: Shuffle timesteps within each window and retrain XGBoost.
- If accuracy stays high (>90%) → temporal order doesn't matter (statistical patterns)
- If accuracy drops significantly → temporal dynamics ARE important

This helps validate whether the task requires temporal modeling or just
co-occurrence pattern recognition.

Usage:
    python romesh_changes/test_temporal_importance.py \
        --data-dir outputs/experiments/file_level_split/preprocessed \
        --output-dir outputs/experiments/temporal_shuffle_test \
        --seed 42
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


OPERATION_NAMES = [
    'adaptive', 'adaptive150025', 'face', 'face150025',
    'pocket', 'pocket150025', 'damageadaptive',
    'damageface', 'damagepocket'
]


def load_npz_data(npz_path):
    """Load preprocessed data from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)

    continuous = data['continuous']  # Shape: (N, T, F)
    labels = data['operation_type']  # Operation type labels

    return continuous, labels


def shuffle_temporal_dimension(X, seed=42):
    """
    Shuffle the temporal dimension (timesteps) within each window.

    Original: X[i, :, :] has timesteps in order [t0, t1, t2, ..., t63]
    Shuffled: X[i, :, :] has timesteps in random order [t42, t5, t18, ...]

    If temporal ORDER matters, this should hurt performance.
    If only temporal CO-OCCURRENCE matters, performance should be similar.
    """
    rng = np.random.RandomState(seed)
    X_shuffled = X.copy()

    # Shuffle timesteps for each window independently
    for i in range(len(X_shuffled)):
        # Generate random permutation of timesteps
        perm = rng.permutation(X_shuffled.shape[1])
        X_shuffled[i] = X_shuffled[i, perm, :]

    return X_shuffled


def flatten_windows(X):
    """Flatten (N, T, F) to (N, T*F) for tree-based models."""
    return X.reshape(X.shape[0], -1)


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, model_name, seed=42):
    """Train a model and return metrics."""

    print(f"\nTraining {model_name}...")
    start_time = time.time()

    if model_name == 'xgboost':
        model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
    elif model_name == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=seed,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Metrics
    results = {
        'model': model_name,
        'train_time_seconds': train_time,
        'train_accuracy': float(accuracy_score(y_train, y_pred_train)),
        'val_accuracy': float(accuracy_score(y_val, y_pred_val)),
        'test_accuracy': float(accuracy_score(y_test, y_pred_test))
    }

    # Per-class accuracy
    results['test_per_class'] = {}
    for i, class_name in enumerate(OPERATION_NAMES):
        mask = y_test == i
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred_test[mask])
            results['test_per_class'][class_name] = float(acc)

    print(f"  Train: {results['train_accuracy']*100:.2f}%")
    print(f"  Val:   {results['val_accuracy']*100:.2f}%")
    print(f"  Test:  {results['test_accuracy']*100:.2f}%")

    return results, model


def main():
    parser = argparse.ArgumentParser(
        description="Test importance of temporal order via shuffling"
    )
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['xgboost', 'random_forest'])
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TEMPORAL ORDER IMPORTANCE TEST")
    print("="*80)
    print(f"\nExperiment: Train {args.model} on temporally-shuffled windows")
    print("Hypothesis: If temporal ORDER matters, shuffling should hurt performance")
    print("")

    # Load data
    print("Loading data...")
    X_train, y_train = load_npz_data(args.data_dir / 'train_sequences.npz')
    X_val, y_val = load_npz_data(args.data_dir / 'val_sequences.npz')
    X_test, y_test = load_npz_data(args.data_dir / 'test_sequences.npz')

    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    # Experiment 1: ORIGINAL (temporal order preserved)
    print("\n" + "="*80)
    print("EXPERIMENT 1: ORIGINAL DATA (Temporal Order Preserved)")
    print("="*80)

    X_train_flat = flatten_windows(X_train)
    X_val_flat = flatten_windows(X_val)
    X_test_flat = flatten_windows(X_test)

    results_original, model_original = train_and_evaluate(
        X_train_flat, y_train,
        X_val_flat, y_val,
        X_test_flat, y_test,
        args.model, args.seed
    )

    # Experiment 2: SHUFFLED (temporal order destroyed)
    print("\n" + "="*80)
    print("EXPERIMENT 2: SHUFFLED DATA (Temporal Order Destroyed)")
    print("="*80)
    print("\nShuffling timesteps within each window...")

    X_train_shuffled = shuffle_temporal_dimension(X_train, seed=args.seed)
    X_val_shuffled = shuffle_temporal_dimension(X_val, seed=args.seed)
    X_test_shuffled = shuffle_temporal_dimension(X_test, seed=args.seed)

    X_train_shuffled_flat = flatten_windows(X_train_shuffled)
    X_val_shuffled_flat = flatten_windows(X_val_shuffled)
    X_test_shuffled_flat = flatten_windows(X_test_shuffled)

    results_shuffled, model_shuffled = train_and_evaluate(
        X_train_shuffled_flat, y_train,
        X_val_shuffled_flat, y_val,
        X_test_shuffled_flat, y_test,
        args.model, args.seed
    )

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    print(f"\n{'Metric':<25} {'Original':>15} {'Shuffled':>15} {'Difference':>15}")
    print("-"*80)

    metrics = ['train_accuracy', 'val_accuracy', 'test_accuracy']
    for metric in metrics:
        orig = results_original[metric] * 100
        shuf = results_shuffled[metric] * 100
        diff = orig - shuf

        print(f"{metric:<25} {orig:>14.2f}% {shuf:>14.2f}% {diff:>+14.2f}%")

    # Per-class comparison
    print("\n" + "="*80)
    print("PER-CLASS ACCURACY COMPARISON")
    print("="*80)

    print(f"\n{'Class':<20} {'Original':>15} {'Shuffled':>15} {'Difference':>15}")
    print("-"*80)

    for class_name in OPERATION_NAMES:
        if class_name in results_original['test_per_class']:
            orig = results_original['test_per_class'][class_name] * 100
            shuf = results_shuffled['test_per_class'].get(class_name, 0.0) * 100
            diff = orig - shuf

            print(f"{class_name:<20} {orig:>14.1f}% {shuf:>14.1f}% {diff:>+14.1f}%")

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    test_acc_drop = (results_original['test_accuracy'] - results_shuffled['test_accuracy']) * 100

    print(f"\nTest accuracy drop: {test_acc_drop:+.2f}%")
    print("")

    if abs(test_acc_drop) < 2.0:
        print("✓ MINIMAL DROP (<2%)")
        print("  → Temporal ORDER does NOT matter significantly")
        print("  → Model learns co-occurrence patterns, not sequential dynamics")
        print("  → Task is fundamentally STATISTICAL, not TEMPORAL")
        print("  → Tree-based models are appropriate for this task")
    elif abs(test_acc_drop) < 5.0:
        print("~ SMALL DROP (2-5%)")
        print("  → Temporal order has MINOR importance")
        print("  → Both statistical and temporal patterns contribute")
        print("  → Hybrid models may be beneficial")
    else:
        print("✗ SIGNIFICANT DROP (>5%)")
        print("  → Temporal ORDER matters significantly!")
        print("  → Model relies on sequential dynamics")
        print("  → Task requires TEMPORAL modeling (LSTM, attention, etc.)")
        print("  → Tree-based models miss important temporal patterns")

    # Save results
    comparison = {
        'original': results_original,
        'shuffled': results_shuffled,
        'test_accuracy_drop_percent': float(test_acc_drop),
        'interpretation': 'statistical' if abs(test_acc_drop) < 2.0 else
                         ('hybrid' if abs(test_acc_drop) < 5.0 else 'temporal')
    }

    output_path = args.output_dir / 'temporal_shuffle_results.json'
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
