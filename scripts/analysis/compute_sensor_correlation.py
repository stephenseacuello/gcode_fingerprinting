#!/usr/bin/env python3
"""
Compute sensor correlation matrix to identify redundant sensors.

This script computes pairwise correlations between sensor groups to help
identify which sensors contain redundant information.

Usage:
    python scripts/analysis/compute_sensor_correlation.py \
        --data-dir outputs/stratified_splits_2k_vocab \
        --output-dir outputs/analysis

Author: Claude Code
Date: January 2026
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Sensor IDs from the dataset (same as run_sensor_ablations.py)
SENSOR_IDS = [
    'xa_motor', 'frame_r1', 'frame_r2', 'frame_b2',
    'frame_l3', 'frame_l2', 'spindle1', 'spindle2',
    'y_bed__1', 'y_bed__2', 'y_bed__3', 'y_bed__4',
]

# Modalities per sensor
MODALITIES = [
    'Ax', 'Ay', 'Az',
    'Gx', 'Gy', 'Gz',
    'Mx', 'My', 'Mz',
    'Pressure', 'Temperature', 'Proximity',
    'ColorR', 'ColorG', 'ColorB', 'ColorA',
    'RMS',
]


def load_data(data_dir: str):
    """Load sensor features and metadata."""
    data_dir = Path(data_dir)

    # Load metadata
    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    columns = metadata['continuous_columns']

    # Load train data (use train for correlation analysis)
    train_data = np.load(data_dir / 'train_sequences.npz', allow_pickle=True)

    # Get sensor features - handle different possible keys
    if 'sensor_features' in train_data:
        features = train_data['sensor_features']
    elif 'continuous' in train_data:
        features = train_data['continuous']
    elif 'continuous_features' in train_data:
        features = train_data['continuous_features']
    else:
        # Try to find the right key
        print(f"Available keys: {list(train_data.keys())}")
        raise KeyError("Could not find sensor features in data")

    # Handle object arrays (list of variable-length sequences)
    if features.dtype == object:
        print(f"Data is stored as variable-length sequences")
        # Stack all sequences and compute mean per sequence
        # Each element is (seq_len, n_features)
        stacked = []
        for seq in features:
            if len(seq) > 0:
                # Take mean across time dimension to get (n_features,)
                stacked.append(np.mean(seq, axis=0))
        features = np.array(stacked)
        print(f"Aggregated to {features.shape[0]} samples with {features.shape[1]} features")
    elif len(features.shape) == 3:
        # Shape is (n_samples, seq_len, n_features)
        print(f"Loaded {len(features)} samples with shape {features.shape}")
        # Average across time dimension to get (n_samples, n_features)
        features = features.mean(axis=1)
        print(f"Averaged over time to shape {features.shape}")
    else:
        print(f"Loaded {len(features)} samples with shape {features.shape}")

    return features, columns


def get_sensor_feature_indices(columns: list) -> dict:
    """Get feature indices for each sensor."""
    sensor_indices = defaultdict(list)

    for i, col in enumerate(columns):
        parts = col.split('.')
        if len(parts) >= 2:
            sensor_id = parts[0]
            if sensor_id in SENSOR_IDS:
                sensor_indices[sensor_id].append(i)

    return dict(sensor_indices)


def compute_sensor_correlations(features: np.ndarray, columns: list):
    """
    Compute correlation matrix between sensor groups.

    For each sensor, we average all its features to get a single
    representative value per sample, then compute correlations.
    """
    sensor_indices = get_sensor_feature_indices(columns)

    # Extract mean feature per sensor for each sample
    n_samples = len(features)
    n_sensors = len(SENSOR_IDS)

    sensor_means = np.zeros((n_samples, n_sensors))

    for j, sensor_id in enumerate(SENSOR_IDS):
        if sensor_id in sensor_indices:
            indices = sensor_indices[sensor_id]
            # Average all modalities for this sensor
            sensor_means[:, j] = features[:, indices].mean(axis=1)

    # Compute correlation matrix
    corr_matrix = np.corrcoef(sensor_means.T)

    return corr_matrix, SENSOR_IDS


def compute_feature_level_correlations(features: np.ndarray, columns: list):
    """
    Compute detailed feature-level correlations between sensors.
    Returns correlation for each modality type across sensors.
    """
    sensor_indices = get_sensor_feature_indices(columns)

    results = {}

    for modality in MODALITIES:
        # Get this modality for each sensor
        modality_data = []
        sensor_names = []

        for sensor_id in SENSOR_IDS:
            if sensor_id in sensor_indices:
                # Find the column for this sensor+modality
                col_name = f"{sensor_id}.{modality}"
                if col_name in columns:
                    idx = columns.index(col_name)
                    modality_data.append(features[:, idx])
                    sensor_names.append(sensor_id)

        if len(modality_data) >= 2:
            modality_array = np.array(modality_data)
            corr = np.corrcoef(modality_array)
            results[modality] = {
                'correlation': corr,
                'sensors': sensor_names,
            }

    return results


def find_highly_correlated_pairs(corr_matrix: np.ndarray, sensor_names: list, threshold: float = 0.9):
    """Find sensor pairs with correlation above threshold."""
    pairs = []
    n = len(sensor_names)

    for i in range(n):
        for j in range(i + 1, n):
            corr = corr_matrix[i, j]
            if abs(corr) >= threshold:
                pairs.append({
                    'sensor1': sensor_names[i],
                    'sensor2': sensor_names[j],
                    'correlation': float(corr),
                })

    # Sort by correlation (highest first)
    pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    return pairs


def plot_correlation_heatmap(corr_matrix: np.ndarray, sensor_names: list, output_path: str):
    """Generate and save correlation heatmap."""
    plt.figure(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        xticklabels=sensor_names,
        yticklabels=sensor_names,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
    )

    plt.title('Sensor Correlation Matrix\n(Mean Feature Correlation)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute sensor correlation matrix')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to split data directory')
    parser.add_argument('--output-dir', type=str, default='outputs/analysis',
                        help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Correlation threshold for identifying redundant pairs')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SENSOR CORRELATION ANALYSIS")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    features, columns = load_data(args.data_dir)

    # Compute sensor-level correlations
    print("\nComputing sensor correlations...")
    corr_matrix, sensor_names = compute_sensor_correlations(features, columns)

    # Find highly correlated pairs
    print(f"\nFinding pairs with correlation >= {args.threshold}...")
    high_corr_pairs = find_highly_correlated_pairs(corr_matrix, sensor_names, args.threshold)

    # Print results
    print("\n" + "=" * 60)
    print("CORRELATION MATRIX")
    print("=" * 60)

    # Print matrix as table
    print(f"\n{'Sensor':<12}", end='')
    for name in sensor_names:
        print(f"{name[:8]:>10}", end='')
    print()

    for i, name in enumerate(sensor_names):
        print(f"{name:<12}", end='')
        for j in range(len(sensor_names)):
            print(f"{corr_matrix[i, j]:>10.2f}", end='')
        print()

    # Print highly correlated pairs
    print("\n" + "=" * 60)
    print(f"HIGHLY CORRELATED PAIRS (>= {args.threshold})")
    print("=" * 60)

    if high_corr_pairs:
        for pair in high_corr_pairs:
            print(f"  {pair['sensor1']} <-> {pair['sensor2']}: {pair['correlation']:.3f}")
    else:
        print(f"  No pairs found with correlation >= {args.threshold}")

    # Also check for moderately correlated pairs (0.7-0.8)
    moderate_pairs = find_highly_correlated_pairs(corr_matrix, sensor_names, 0.7)
    moderate_pairs = [p for p in moderate_pairs if abs(p['correlation']) < args.threshold]

    if moderate_pairs:
        print(f"\nMODERATELY CORRELATED PAIRS (0.7 - {args.threshold})")
        for pair in moderate_pairs:
            print(f"  {pair['sensor1']} <-> {pair['sensor2']}: {pair['correlation']:.3f}")

    # Generate heatmap
    print("\nGenerating heatmap...")
    heatmap_path = output_dir / 'sensor_correlation_heatmap.png'
    plot_correlation_heatmap(corr_matrix, sensor_names, str(heatmap_path))

    # Save results to JSON
    results = {
        'sensor_names': sensor_names,
        'correlation_matrix': corr_matrix.tolist(),
        'highly_correlated_pairs': high_corr_pairs,
        'threshold': args.threshold,
        'n_samples': len(features),
    }

    results_path = output_dir / 'sensor_correlation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {results_path}")

    # Print recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    if high_corr_pairs:
        redundant_sensors = set()
        for pair in high_corr_pairs:
            # Keep the one with better individual performance (if known)
            # For now, just flag both as potentially redundant
            redundant_sensors.add(pair['sensor2'])  # Keep first, remove second

        print(f"\nConsider removing these redundant sensors:")
        for s in redundant_sensors:
            print(f"  - {s}")
        print(f"\nThis would reduce from {len(sensor_names)} to {len(sensor_names) - len(redundant_sensors)} sensors")
    else:
        print("\nNo highly correlated sensor pairs found.")
        print("All sensors appear to provide unique information.")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
