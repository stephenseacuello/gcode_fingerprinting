#!/usr/bin/env python3
"""
Run experiment without color channels.

This script tests the hypothesis that removing redundant color channels
(R, G, B, A which are >97% correlated) might improve performance.

Reduces features from 296 to ~248 by removing 4 color channels per sensor.

Usage:
    python scripts/experiments/run_no_color_ablation.py \
        --data-dir outputs/stratified_splits_2k_vocab \
        --output-dir outputs/jan23_followup/no_color

Author: Claude Code
Date: January 2026
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Color channel suffixes to remove
COLOR_SUFFIXES = ['.ColorR', '.ColorG', '.ColorB', '.ColorA']


def get_non_color_indices(columns: list) -> tuple:
    """Get indices of columns that are NOT color channels."""
    indices = []
    removed = []
    for i, col in enumerate(columns):
        is_color = any(col.endswith(suffix) for suffix in COLOR_SUFFIXES)
        if not is_color:
            indices.append(i)
        else:
            removed.append(col)
    return indices, removed


def create_filtered_data(data_dir: Path, output_dir: Path):
    """Create new data files with color channels removed."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    columns = metadata['continuous_columns']
    non_color_indices, removed_cols = get_non_color_indices(columns)

    # Filter columns
    new_columns = [columns[i] for i in non_color_indices]

    print(f"Original features: {len(columns)}")
    print(f"After removing color: {len(new_columns)}")
    print(f"Removed: {len(removed_cols)} color features")
    print(f"  Example removed: {removed_cols[:5]}...")

    # Process each split
    for split in ['train', 'val', 'test']:
        input_file = data_dir / f'{split}_sequences.npz'
        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping")
            continue

        print(f"\nProcessing {split}...")
        data = np.load(input_file, allow_pickle=True)

        # Get continuous features
        if 'continuous' in data:
            features = data['continuous']
        elif 'continuous_features' in data:
            features = data['continuous_features']
        else:
            print(f"Available keys: {list(data.keys())}")
            continue

        # Handle object arrays
        if features.dtype == object:
            # Filter each sequence
            filtered = []
            for seq in features:
                if len(seq) > 0:
                    filtered.append(seq[:, non_color_indices])
            filtered_features = np.array(filtered, dtype=object)
        else:
            filtered_features = features[:, :, non_color_indices] if len(features.shape) == 3 else features[:, non_color_indices]

        print(f"  Original shape: {features.shape if hasattr(features, 'shape') else 'object array'}")
        print(f"  Filtered shape: {filtered_features.shape if hasattr(filtered_features, 'shape') else 'object array'}")

        # Save filtered data
        output_file = output_dir / f'{split}_sequences.npz'

        # Copy all other arrays, replace continuous
        save_dict = {}
        for key in data.keys():
            if key in ['continuous', 'continuous_features']:
                save_dict['continuous'] = filtered_features
            else:
                save_dict[key] = data[key]

        np.savez(output_file, **save_dict)
        print(f"  Saved to: {output_file}")

    # Save updated metadata
    new_metadata = metadata.copy()
    new_metadata['continuous_columns'] = new_columns
    new_metadata['n_continuous_features'] = len(new_columns)
    new_metadata['color_removed'] = True
    new_metadata['removed_color_features'] = removed_cols

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(new_metadata, f, indent=2)
    print(f"\nSaved metadata to: {output_dir / 'metadata.json'}")

    return new_columns, removed_cols


def main():
    parser = argparse.ArgumentParser(description='Remove color channels from data')
    parser.add_argument('--data-dir', type=str, default='outputs/stratified_splits_2k_vocab',
                        help='Path to original split data directory')
    parser.add_argument('--output-dir', type=str, default='outputs/jan23_followup/no_color',
                        help='Output directory for filtered data')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("NO-COLOR CHANNEL ABLATION: Data Preparation")
    print("=" * 70)

    # Create filtered data
    new_columns, removed = create_filtered_data(data_dir, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original features: 296")
    print(f"Remaining features: {len(new_columns)}")
    print(f"Removed features: {len(removed)}")
    print(f"\nFiltered data saved to: {output_dir}")
    print("\nTo train on this data, run:")
    print(f"  python scripts/training/train_sensor_multihead.py \\")
    print(f"    --split-dir {output_dir} \\")
    print(f"    --config configs/best_lambda_2k_vocab.json \\")
    print(f"    --output-dir outputs/jan23_followup/no_color/training")

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
