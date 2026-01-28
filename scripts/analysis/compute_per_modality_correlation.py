#!/usr/bin/env python3
"""
Compute per-modality sensor correlations.

This script computes correlations between sensors for EACH modality separately,
rather than averaging all modalities first. This reveals the true physical
correlations between sensors (e.g., frame_r2.Ax vs frame_r1.Ax).

Usage:
    python scripts/analysis/compute_per_modality_correlation.py \
        --data-dir outputs/stratified_splits_2k_vocab \
        --output-dir outputs/jan23_followup/correlations

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

    # Load train data (use train for correlation analysis)
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

    # Handle object arrays (list of variable-length sequences)
    if features.dtype == object:
        print(f"Data is stored as variable-length sequences")
        stacked = []
        for seq in features:
            if len(seq) > 0:
                stacked.append(np.mean(seq, axis=0))
        features = np.array(stacked)
        print(f"Aggregated to {features.shape[0]} samples with {features.shape[1]} features")
    elif len(features.shape) == 3:
        print(f"Loaded {len(features)} samples with shape {features.shape}")
        features = features.mean(axis=1)
        print(f"Averaged over time to shape {features.shape}")
    else:
        print(f"Loaded {len(features)} samples with shape {features.shape}")

    return features, columns


def compute_per_modality_correlations(features: np.ndarray, columns: list):
    """
    Compute correlations between sensors for each modality separately.

    Returns a dict: modality -> (correlation_matrix, sensor_names)
    """
    results = {}

    for modality in MODALITIES:
        # Get this modality for each sensor
        modality_data = []
        sensor_names = []

        for sensor_id in SENSOR_IDS:
            col_name = f"{sensor_id}.{modality}"
            if col_name in columns:
                idx = columns.index(col_name)
                modality_data.append(features[:, idx])
                sensor_names.append(sensor_id)

        if len(modality_data) >= 2:
            modality_array = np.array(modality_data)
            corr = np.corrcoef(modality_array)
            results[modality] = {
                'correlation_matrix': corr,
                'sensors': sensor_names,
            }

    return results


def find_high_correlations(results: dict, threshold: float = 0.8):
    """Find highly correlated sensor pairs per modality."""
    high_corr = []

    for modality, data in results.items():
        corr = data['correlation_matrix']
        sensors = data['sensors']
        n = len(sensors)

        for i in range(n):
            for j in range(i + 1, n):
                r = corr[i, j]
                if abs(r) >= threshold:
                    high_corr.append({
                        'modality': modality,
                        'sensor1': sensors[i],
                        'sensor2': sensors[j],
                        'correlation': float(r),
                    })

    # Sort by correlation
    high_corr.sort(key=lambda x: abs(x['correlation']), reverse=True)
    return high_corr


def plot_modality_heatmaps(results: dict, output_dir: Path):
    """Generate correlation heatmaps for each modality."""
    # Create a summary figure with key modalities
    key_modalities = ['RMS', 'Ax', 'Gx', 'Mx', 'Pressure', 'Temperature']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, modality in enumerate(key_modalities):
        if modality in results:
            data = results[modality]
            corr = data['correlation_matrix']
            sensors = data['sensors']

            # Shorten sensor names for display
            short_names = [s.replace('frame_', 'f_').replace('spindle', 'sp').replace('y_bed__', 'yb') for s in sensors]

            sns.heatmap(
                corr,
                xticklabels=short_names,
                yticklabels=short_names,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                ax=axes[idx],
                annot_kws={'size': 8}
            )
            axes[idx].set_title(f'{modality} Correlation', fontsize=12)

    plt.suptitle('Per-Modality Sensor Correlations\n(Same modality across different sensors)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_modality_correlation_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary heatmap to: {output_dir / 'per_modality_correlation_summary.png'}")

    # Also save individual heatmaps for each modality
    for modality, data in results.items():
        corr = data['correlation_matrix']
        sensors = data['sensors']

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr,
            xticklabels=sensors,
            yticklabels=sensors,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
        )
        plt.title(f'{modality}: Sensor Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / f'correlation_{modality}.png', dpi=100, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compute per-modality sensor correlations')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to split data directory')
    parser.add_argument('--output-dir', type=str, default='outputs/jan23_followup/correlations',
                        help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Correlation threshold for identifying high pairs')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PER-MODALITY SENSOR CORRELATION ANALYSIS")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    features, columns = load_data(args.data_dir)

    # Compute per-modality correlations
    print("\nComputing per-modality correlations...")
    results = compute_per_modality_correlations(features, columns)

    # Find high correlations
    print(f"\nFinding pairs with |r| >= {args.threshold}...")
    high_corr = find_high_correlations(results, args.threshold)

    # Print results
    print("\n" + "=" * 70)
    print("HIGH CORRELATIONS BY MODALITY")
    print("=" * 70)

    if high_corr:
        # Group by modality
        by_modality = defaultdict(list)
        for item in high_corr:
            by_modality[item['modality']].append(item)

        for modality in MODALITIES:
            if modality in by_modality:
                print(f"\n{modality}:")
                for item in by_modality[modality]:
                    print(f"  {item['sensor1']} <-> {item['sensor2']}: r = {item['correlation']:.3f}")
    else:
        print(f"No pairs found with |r| >= {args.threshold}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: MAX CORRELATIONS PER MODALITY")
    print("=" * 70)

    summary = {}
    for modality, data in results.items():
        corr = data['correlation_matrix']
        # Get max off-diagonal correlation
        n = corr.shape[0]
        max_corr = 0
        max_pair = ('', '')
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) > abs(max_corr):
                    max_corr = corr[i, j]
                    max_pair = (data['sensors'][i], data['sensors'][j])

        summary[modality] = {
            'max_correlation': float(max_corr),
            'max_pair': max_pair,
        }
        print(f"{modality:15} max |r| = {abs(max_corr):.3f}  ({max_pair[0]} <-> {max_pair[1]})")

    # Generate heatmaps
    print("\nGenerating heatmaps...")
    plot_modality_heatmaps(results, output_dir)

    # Save results to JSON
    results_json = {
        'per_modality': {
            mod: {
                'correlation_matrix': data['correlation_matrix'].tolist(),
                'sensors': data['sensors'],
            }
            for mod, data in results.items()
        },
        'high_correlations': high_corr,
        'summary': summary,
        'threshold': args.threshold,
        'n_samples': len(features),
    }

    results_path = output_dir / 'per_modality_correlation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Count high correlations
    n_high = len(high_corr)
    print(f"\nTotal high-correlation pairs (|r| >= {args.threshold}): {n_high}")

    # Check frame sensor correlations
    frame_sensors = ['frame_r1', 'frame_r2', 'frame_b2', 'frame_l2', 'frame_l3']
    frame_high = [h for h in high_corr if h['sensor1'] in frame_sensors and h['sensor2'] in frame_sensors]
    print(f"High correlations between frame sensors: {len(frame_high)}")

    if frame_high:
        print("\nFrame sensor pairs with high correlation:")
        for item in frame_high[:10]:  # Top 10
            print(f"  {item['modality']:12} {item['sensor1']:10} <-> {item['sensor2']:10}: r = {item['correlation']:.3f}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
