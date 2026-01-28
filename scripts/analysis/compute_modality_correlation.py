#!/usr/bin/env python3
"""
Compute modality correlation matrix to identify redundant modalities.

This script computes pairwise correlations between modality types
(Ax, Ay, Az, Gx, etc.) across all sensors.

Usage:
    python scripts/analysis/compute_modality_correlation.py \
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

# Sensor IDs from the dataset
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

    # Load train data
    train_data = np.load(data_dir / 'train_sequences.npz', allow_pickle=True)

    # Get sensor features
    if 'sensor_features' in train_data:
        features = train_data['sensor_features']
    elif 'continuous' in train_data:
        features = train_data['continuous']
    elif 'continuous_features' in train_data:
        features = train_data['continuous_features']
    else:
        print(f"Available keys: {list(train_data.keys())}")
        raise KeyError("Could not find sensor features in data")

    # Handle 3D arrays
    if len(features.shape) == 3:
        print(f"Loaded {len(features)} samples with shape {features.shape}")
        features = features.mean(axis=1)
        print(f"Averaged over time to shape {features.shape}")
    else:
        print(f"Loaded {len(features)} samples with shape {features.shape}")

    return features, columns


def get_modality_feature_indices(columns: list) -> dict:
    """Get feature indices for each modality (across all sensors)."""
    modality_indices = defaultdict(list)

    for i, col in enumerate(columns):
        parts = col.split('.')
        if len(parts) >= 2:
            sensor_id = parts[0]
            modality = parts[1]
            if sensor_id in SENSOR_IDS and modality in MODALITIES:
                modality_indices[modality].append(i)

    return dict(modality_indices)


def compute_modality_correlations(features: np.ndarray, columns: list):
    """
    Compute correlation matrix between modality types.

    For each modality, we average across all sensors to get a single
    representative value per sample, then compute correlations.
    """
    modality_indices = get_modality_feature_indices(columns)

    n_samples = len(features)
    n_modalities = len(MODALITIES)

    modality_means = np.zeros((n_samples, n_modalities))

    for j, modality in enumerate(MODALITIES):
        if modality in modality_indices:
            indices = modality_indices[modality]
            # Average this modality across all sensors
            modality_means[:, j] = features[:, indices].mean(axis=1)

    # Compute correlation matrix
    corr_matrix = np.corrcoef(modality_means.T)

    return corr_matrix, MODALITIES


def find_highly_correlated_pairs(corr_matrix: np.ndarray, names: list, threshold: float = 0.9):
    """Find pairs with correlation above threshold."""
    pairs = []
    n = len(names)

    for i in range(n):
        for j in range(i + 1, n):
            corr = corr_matrix[i, j]
            if abs(corr) >= threshold:
                pairs.append({
                    'modality1': names[i],
                    'modality2': names[j],
                    'correlation': float(corr),
                })

    pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    return pairs


def plot_correlation_heatmap(corr_matrix: np.ndarray, names: list, output_path: str):
    """Generate and save correlation heatmap."""
    plt.figure(figsize=(14, 12))

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        xticklabels=names,
        yticklabels=names,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
    )

    plt.title('Modality Correlation Matrix\n(Averaged Across All Sensors)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute modality correlation matrix')
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
    print("MODALITY CORRELATION ANALYSIS")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    features, columns = load_data(args.data_dir)

    # Compute modality-level correlations
    print("\nComputing modality correlations...")
    corr_matrix, modality_names = compute_modality_correlations(features, columns)

    # Find highly correlated pairs
    print(f"\nFinding pairs with correlation >= {args.threshold}...")
    high_corr_pairs = find_highly_correlated_pairs(corr_matrix, modality_names, args.threshold)

    # Print results
    print("\n" + "=" * 60)
    print("CORRELATION MATRIX")
    print("=" * 60)

    # Print matrix as table
    print(f"\n{'Modality':<12}", end='')
    for name in modality_names:
        print(f"{name[:8]:>10}", end='')
    print()

    for i, name in enumerate(modality_names):
        print(f"{name:<12}", end='')
        for j in range(len(modality_names)):
            print(f"{corr_matrix[i, j]:>10.2f}", end='')
        print()

    # Print highly correlated pairs
    print("\n" + "=" * 60)
    print(f"HIGHLY CORRELATED PAIRS (>= {args.threshold})")
    print("=" * 60)

    if high_corr_pairs:
        for pair in high_corr_pairs:
            print(f"  {pair['modality1']} <-> {pair['modality2']}: {pair['correlation']:.3f}")
    else:
        print(f"  No pairs found with correlation >= {args.threshold}")

    # Check for moderate correlations
    moderate_pairs = find_highly_correlated_pairs(corr_matrix, modality_names, 0.7)
    moderate_pairs = [p for p in moderate_pairs if abs(p['correlation']) < args.threshold]

    if moderate_pairs:
        print(f"\nMODERATELY CORRELATED PAIRS (0.7 - {args.threshold})")
        for pair in moderate_pairs:
            print(f"  {pair['modality1']} <-> {pair['modality2']}: {pair['correlation']:.3f}")

    # Generate heatmap
    print("\nGenerating heatmap...")
    heatmap_path = output_dir / 'modality_correlation_heatmap.png'
    plot_correlation_heatmap(corr_matrix, modality_names, str(heatmap_path))

    # Save results to JSON
    results = {
        'modality_names': modality_names,
        'correlation_matrix': corr_matrix.tolist(),
        'highly_correlated_pairs': high_corr_pairs,
        'threshold': args.threshold,
        'n_samples': len(features),
    }

    results_path = output_dir / 'modality_correlation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {results_path}")

    # Print analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Group modalities by type
    imu_modalities = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz']
    color_modalities = ['ColorR', 'ColorG', 'ColorB', 'ColorA']
    env_modalities = ['Pressure', 'Temperature', 'Proximity']

    print("\nModality Groups:")
    print(f"  IMU (accelerometer, gyro, magnetometer): {imu_modalities}")
    print(f"  Color: {color_modalities}")
    print(f"  Environmental: {env_modalities}")
    print(f"  Computed: ['RMS']")

    # Check within-group correlations
    print("\nWithin-group correlation patterns:")

    # Accelerometer
    accel_idx = [modality_names.index(m) for m in ['Ax', 'Ay', 'Az']]
    accel_corr = corr_matrix[np.ix_(accel_idx, accel_idx)]
    print(f"  Accelerometer (Ax, Ay, Az): mean off-diagonal = {(accel_corr.sum() - 3) / 6:.2f}")

    # Gyroscope
    gyro_idx = [modality_names.index(m) for m in ['Gx', 'Gy', 'Gz']]
    gyro_corr = corr_matrix[np.ix_(gyro_idx, gyro_idx)]
    print(f"  Gyroscope (Gx, Gy, Gz): mean off-diagonal = {(gyro_corr.sum() - 3) / 6:.2f}")

    # Magnetometer
    mag_idx = [modality_names.index(m) for m in ['Mx', 'My', 'Mz']]
    mag_corr = corr_matrix[np.ix_(mag_idx, mag_idx)]
    print(f"  Magnetometer (Mx, My, Mz): mean off-diagonal = {(mag_corr.sum() - 3) / 6:.2f}")

    # Color
    color_idx = [modality_names.index(m) for m in color_modalities]
    color_corr = corr_matrix[np.ix_(color_idx, color_idx)]
    print(f"  Color (R, G, B, A): mean off-diagonal = {(color_corr.sum() - 4) / 12:.2f}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
