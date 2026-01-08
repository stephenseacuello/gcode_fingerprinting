#!/usr/bin/env python3
"""
Sklearn Baselines for G-code Fingerprinting.

This script trains traditional ML baselines:
1. SVM - Support Vector Machine for operation classification
2. Random Forest - With feature importance analysis
3. Logistic Regression - Linear baseline
4. k-NN - Nearest neighbor baseline

These focus on operation classification (9 classes) since traditional
ML cannot easily handle sequence-to-sequence prediction.

Usage:
    python scripts/train_sklearn_baselines.py --output-dir outputs/baselines/sklearn
    python scripts/train_sklearn_baselines.py --model svm --seed 42

Author: Claude Code
Date: December 2025
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import Counter
import warnings

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_data(split_dir: str, split: str = 'train'):
    """Load data from split files and prepare for sklearn."""
    data_path = Path(split_dir) / f'{split}_sequences.npz'
    data = np.load(data_path, allow_pickle=True)

    # Use continuous sensor features
    sensor_data = data['continuous']  # (N, T, F) = (N, 64, 155)
    operations = data['operation_type']  # (N,) operation labels

    # Flatten temporal dimension for traditional ML
    # Use multiple strategies and concatenate

    features_list = []

    for sensors in sensor_data:
        # sensors: (T, F)
        if len(sensors.shape) == 1:
            sensors = sensors.reshape(1, -1)

        # Statistical features
        feat = []

        # Mean across time
        feat.extend(np.mean(sensors, axis=0))

        # Std across time
        feat.extend(np.std(sensors, axis=0))

        # Min across time
        feat.extend(np.min(sensors, axis=0))

        # Max across time
        feat.extend(np.max(sensors, axis=0))

        # First timestep
        feat.extend(sensors[0])

        # Last timestep
        feat.extend(sensors[-1])

        # Delta (last - first)
        feat.extend(sensors[-1] - sensors[0])

        features_list.append(feat)

    X = np.array(features_list)
    y = np.array([op.item() if hasattr(op, 'item') else op for op in operations])

    # Convert operation names to numeric labels if needed
    if isinstance(y[0], str):
        unique_ops = sorted(set(y))
        op_to_id = {op: i for i, op in enumerate(unique_ops)}
        y = np.array([op_to_id[op] for op in y])

    print(f"Loaded {split}: X shape = {X.shape}, y shape = {y.shape}")
    print(f"  Classes: {np.unique(y)}")

    return X, y


def train_svm(X_train, y_train, X_val, y_val, seed=42):
    """Train SVM classifier."""
    print("\nTraining SVM...")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train SVM with RBF kernel
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=seed,
        class_weight='balanced',
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    val_acc = accuracy_score(y_val, model.predict(X_val_scaled))

    print(f"  Train accuracy: {train_acc*100:.2f}%")
    print(f"  Val accuracy: {val_acc*100:.2f}%")

    return model, scaler, {'train_acc': train_acc, 'val_acc': val_acc}


def train_random_forest(X_train, y_train, X_val, y_val, seed=42):
    """Train Random Forest classifier."""
    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=seed,
        class_weight='balanced',
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))

    print(f"  Train accuracy: {train_acc*100:.2f}%")
    print(f"  Val accuracy: {val_acc*100:.2f}%")

    # Feature importance
    importances = model.feature_importances_

    return model, None, {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'feature_importances': importances.tolist(),
    }


def train_logistic_regression(X_train, y_train, X_val, y_val, seed=42):
    """Train Logistic Regression classifier."""
    print("\nTraining Logistic Regression...")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        class_weight='balanced',
        multi_class='multinomial',
        solver='lbfgs',
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    val_acc = accuracy_score(y_val, model.predict(X_val_scaled))

    print(f"  Train accuracy: {train_acc*100:.2f}%")
    print(f"  Val accuracy: {val_acc*100:.2f}%")

    return model, scaler, {'train_acc': train_acc, 'val_acc': val_acc}


def train_knn(X_train, y_train, X_val, y_val, seed=42):
    """Train k-NN classifier."""
    print("\nTraining k-NN...")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='euclidean',
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    val_acc = accuracy_score(y_val, model.predict(X_val_scaled))

    print(f"  Train accuracy: {train_acc*100:.2f}%")
    print(f"  Val accuracy: {val_acc*100:.2f}%")

    return model, scaler, {'train_acc': train_acc, 'val_acc': val_acc}


def train_gradient_boosting(X_train, y_train, X_val, y_val, seed=42):
    """Train Gradient Boosting classifier."""
    print("\nTraining Gradient Boosting...")

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))

    print(f"  Train accuracy: {train_acc*100:.2f}%")
    print(f"  Val accuracy: {val_acc*100:.2f}%")

    return model, None, {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'feature_importances': model.feature_importances_.tolist(),
    }


def evaluate_on_test(model, scaler, X_test, y_test, class_names=None):
    """Evaluate model on test set with detailed metrics."""
    if scaler is not None:
        X_test = scaler.transform(X_test)

    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )

    # Per-class metrics
    per_class = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }


def main():
    parser = argparse.ArgumentParser(description='Train sklearn baselines')
    parser.add_argument('--split-dir', type=str, default='outputs/processed_v3',
                        help='Directory with train/val/test splits')
    parser.add_argument('--output-dir', type=str, default='outputs/baselines/sklearn',
                        help='Output directory')
    parser.add_argument('--model', type=str, default='all',
                        choices=['svm', 'rf', 'lr', 'knn', 'gb', 'all'],
                        help='Which model to train')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    X_train, y_train = load_data(args.split_dir, 'train')
    X_val, y_val = load_data(args.split_dir, 'val')
    X_test, y_test = load_data(args.split_dir, 'test')

    # Handle NaN/Inf values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Get class names
    unique_classes = sorted(set(y_train))
    n_classes = len(unique_classes)
    print(f"Number of classes: {n_classes}")

    # Class distribution
    print("\nClass distribution:")
    for c in unique_classes:
        count = (y_train == c).sum()
        print(f"  Class {c}: {count} samples ({count/len(y_train)*100:.1f}%)")

    # Model training functions
    model_funcs = {
        'svm': ('SVM', train_svm),
        'rf': ('Random Forest', train_random_forest),
        'lr': ('Logistic Regression', train_logistic_regression),
        'knn': ('k-NN', train_knn),
        'gb': ('Gradient Boosting', train_gradient_boosting),
    }

    # Determine which models to train
    if args.model == 'all':
        models_to_train = list(model_funcs.keys())
    else:
        models_to_train = [args.model]

    all_results = {}

    for model_key in models_to_train:
        model_name, train_func = model_funcs[model_key]

        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print('='*60)

        # Train
        model, scaler, train_metrics = train_func(
            X_train, y_train, X_val, y_val, seed=args.seed
        )

        # Evaluate on test
        test_metrics = evaluate_on_test(
            model, scaler, X_test, y_test,
            class_names=[str(c) for c in unique_classes]
        )

        print(f"\nTest Results for {model_name}:")
        print(f"  Accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"  F1 Score: {test_metrics['f1']*100:.2f}%")

        # Save model
        model_path = output_dir / f'{model_key}_model.joblib'
        joblib.dump({'model': model, 'scaler': scaler}, model_path)

        # Store results
        all_results[model_key] = {
            'name': model_name,
            'train_accuracy': train_metrics['train_acc'],
            'val_accuracy': train_metrics['val_acc'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'classification_report': test_metrics['classification_report'],
            'model_path': str(model_path),
        }

        # Add feature importances if available
        if 'feature_importances' in train_metrics:
            all_results[model_key]['feature_importances'] = train_metrics['feature_importances']

    # Summary
    print("\n" + "="*60)
    print("SKLEARN BASELINES SUMMARY")
    print("="*60)

    print(f"\n{'Model':<25} {'Train Acc':>12} {'Val Acc':>12} {'Test Acc':>12} {'Test F1':>12}")
    print("-" * 75)

    for model_key, results in all_results.items():
        print(f"{results['name']:<25} "
              f"{results['train_accuracy']*100:>11.2f}% "
              f"{results['val_accuracy']*100:>11.2f}% "
              f"{results['test_accuracy']*100:>11.2f}% "
              f"{results['test_f1']*100:>11.2f}%")

    print("-" * 75)

    # Save all results
    results_path = output_dir / 'sklearn_results.json'
    all_results['metadata'] = {
        'seed': args.seed,
        'split_dir': args.split_dir,
        'n_train': len(y_train),
        'n_val': len(y_val),
        'n_test': len(y_test),
        'n_features': X_train.shape[1],
        'n_classes': n_classes,
        'timestamp': datetime.now().isoformat(),
    }

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")

    # Note about limitations
    print("\n" + "="*60)
    print("NOTE: These baselines only evaluate OPERATION CLASSIFICATION")
    print("(9 classes) because traditional ML cannot easily handle")
    print("sequence-to-sequence token prediction. For fair comparison,")
    print("compare the 100% operation accuracy of our full model.")
    print("="*60)


if __name__ == '__main__':
    main()
