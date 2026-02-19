#!/usr/bin/env python3
"""
Train XGBoost and extract feature importances to identify shortcuts.
"""
import numpy as np
import xgboost as xgb
from pathlib import Path
import json


def load_data(data_dir):
    """Load preprocessed data."""
    train_data = np.load(data_dir / 'train_sequences.npz', allow_pickle=True)
    test_data = np.load(data_dir / 'test_sequences.npz', allow_pickle=True)
    metadata_path = data_dir / 'metadata.json'

    with open(metadata_path) as f:
        metadata = json.load(f)

    return train_data, test_data, metadata


def main():
    # Load data from no_proximity_no_color experiment
    data_dir = Path('outputs/experiments/no_proximity_no_color/preprocessed')

    print("="*100)
    print("EXTRACTING XGBOOST FEATURE IMPORTANCES")
    print("="*100)
    print()

    print("Loading data...")
    train_data, test_data, metadata = load_data(data_dir)

    # Extract features and labels
    X_train = train_data['continuous']  # Shape: (n_windows, window_size, n_features)
    y_train = train_data['operation_type']

    X_test = test_data['continuous']
    y_test = test_data['operation_type']

    # Flatten windows for XGBoost (can't handle 3D)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(n_test, -1)

    print(f"Train shape: {X_train_flat.shape}")
    print(f"Test shape: {X_test_flat.shape}")
    print()

    # Train XGBoost
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )

    model.fit(X_train_flat, y_train)

    # Get feature importances
    importances = model.feature_importances_

    print(f"✅ Model trained!")
    print()

    # Get feature names from metadata
    continuous_cols = metadata.get('continuous_columns', [])
    window_size = metadata.get('window_size', 64)

    # Create feature names: sensor.channel_timestep
    feature_names = []
    for col in continuous_cols:
        for t in range(window_size):
            feature_names.append(f"{col}_t{t}")

    print(f"Total features: {len(feature_names)}")
    print(f"Non-zero importances: {(importances > 0).sum()}")
    print()

    # Sort by importance
    importance_pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: -x[1]
    )

    # Top features
    print("="*100)
    print("TOP 50 MOST IMPORTANT FEATURES")
    print("="*100)
    print(f"{'Rank':<6} {'Feature':<50} {'Importance':<15} {'% of Total':<12}")
    print("-"*100)

    total_importance = importances.sum()
    cumulative = 0

    for i, (feat_name, importance) in enumerate(importance_pairs[:50], 1):
        pct = 100 * importance / total_importance
        cumulative += importance
        cumulative_pct = 100 * cumulative / total_importance

        print(f"{i:<6} {feat_name:<50} {importance:>13.6f} {pct:>10.2f}%  (cum: {cumulative_pct:.1f}%)")

    print("-"*100)
    print()

    # Analyze by sensor channel (aggregate across timesteps)
    print("="*100)
    print("FEATURE IMPORTANCE BY SENSOR CHANNEL (aggregated across time)")
    print("="*100)
    print()

    channel_importance = {}
    for feat_name, importance in zip(feature_names, importances):
        # Extract channel name (everything before _t<number>)
        channel = feat_name.rsplit('_t', 1)[0]

        if channel not in channel_importance:
            channel_importance[channel] = 0.0
        channel_importance[channel] += importance

    # Sort by total importance
    sorted_channels = sorted(
        channel_importance.items(),
        key=lambda x: -x[1]
    )

    print(f"{'Rank':<6} {'Sensor Channel':<40} {'Total Importance':<18} {'% of Total':<12}")
    print("-"*100)

    for i, (channel, importance) in enumerate(sorted_channels[:30], 1):
        pct = 100 * importance / total_importance
        print(f"{i:<6} {channel:<40} {importance:>16.6f} {pct:>10.2f}%")

    print("-"*100)
    print()

    # Identify most important sensor types
    print("="*100)
    print("MOST IMPORTANT SENSOR TYPES")
    print("="*100)
    print()

    sensor_type_importance = {}

    for channel, importance in channel_importance.items():
        # Extract sensor type (e.g., "Ax", "Gy", "Pressure", "Temperature")
        if '.' in channel:
            sensor_type = channel.split('.')[1]  # e.g., "frame_l2.Ax" -> "Ax"
        else:
            sensor_type = channel

        if sensor_type not in sensor_type_importance:
            sensor_type_importance[sensor_type] = 0.0
        sensor_type_importance[sensor_type] += importance

    sorted_types = sorted(
        sensor_type_importance.items(),
        key=lambda x: -x[1]
    )

    print(f"{'Sensor Type':<20} {'Total Importance':<18} {'% of Total':<12}")
    print("-"*80)

    for sensor_type, importance in sorted_types:
        pct = 100 * importance / total_importance
        print(f"{sensor_type:<20} {importance:>16.6f} {pct:>10.2f}%")

    print("-"*80)
    print()

    # Test accuracy
    y_pred = model.predict(X_test_flat)
    test_acc = (y_pred == y_test).mean()

    print("="*100)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print("="*100)
    print()

    # Save results
    output = {
        'test_accuracy': float(test_acc),
        'top_50_features': [
            {'rank': i+1, 'feature': feat, 'importance': float(imp)}
            for i, (feat, imp) in enumerate(importance_pairs[:50])
        ],
        'channel_importance': {
            ch: float(imp) for ch, imp in sorted_channels
        },
        'sensor_type_importance': {
            st: float(imp) for st, imp in sorted_types
        }
    }

    output_path = Path('outputs/xgboost_feature_importance.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ Results saved to: {output_path}")
    print()


if __name__ == '__main__':
    main()