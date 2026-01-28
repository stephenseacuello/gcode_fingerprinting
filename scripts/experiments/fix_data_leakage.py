#!/usr/bin/env python3
"""
Fix data leakage by fitting scaler on training data only.

This script addresses the data leakage issue where the scaler was fit on
combined train+val+test data before splitting. We re-process the data to:
1. Use the existing file-level splits
2. Fit the StandardScaler on TRAINING data only
3. Apply the same scaler to validation and test sets

Usage:
    python scripts/experiments/fix_data_leakage.py \
        --data-dir outputs/stratified_splits_2k_vocab \
        --output-dir outputs/jan23_followup/no_leakage

Author: Claude Code
Date: January 2026
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_raw_sequences(data_dir: Path, split: str):
    """Load sequences for a split."""
    file_path = data_dir / f'{split}_sequences.npz'
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = np.load(file_path, allow_pickle=True)

    # Get continuous features
    if 'continuous' in data:
        features = data['continuous']
    elif 'continuous_features' in data:
        features = data['continuous_features']
    else:
        raise KeyError(f"Could not find continuous features. Keys: {list(data.keys())}")

    return data, features


def flatten_for_scaler(features):
    """Flatten features for scaler fitting."""
    if features.dtype == object:
        # Variable-length sequences
        flattened = []
        for seq in features:
            if len(seq) > 0:
                flattened.append(seq)
        return np.vstack(flattened)
    elif len(features.shape) == 3:
        # (n_samples, seq_len, n_features) -> (n_samples * seq_len, n_features)
        n_samples, seq_len, n_features = features.shape
        return features.reshape(-1, n_features)
    else:
        return features


def apply_scaler_to_sequences(features, scaler):
    """Apply fitted scaler to sequences."""
    if features.dtype == object:
        # Variable-length sequences
        scaled = []
        for seq in features:
            if len(seq) > 0:
                scaled.append(scaler.transform(seq))
            else:
                scaled.append(seq)
        return np.array(scaled, dtype=object)
    elif len(features.shape) == 3:
        # (n_samples, seq_len, n_features)
        n_samples, seq_len, n_features = features.shape
        flat = features.reshape(-1, n_features)
        scaled_flat = scaler.transform(flat)
        return scaled_flat.reshape(n_samples, seq_len, n_features)
    else:
        return scaler.transform(features)


def main():
    parser = argparse.ArgumentParser(description='Fix data leakage in preprocessing')
    parser.add_argument('--data-dir', type=str, default='outputs/stratified_splits_2k_vocab',
                        help='Path to original split data directory')
    parser.add_argument('--output-dir', type=str, default='outputs/jan23_followup/no_leakage',
                        help='Output directory for corrected data')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FIXING DATA LEAKAGE: Train-Only Scaler Fitting")
    print("=" * 70)

    # Load metadata
    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    print(f"\nOriginal data:")
    print(f"  Features: {metadata['n_continuous_features']}")
    print(f"  Train samples: {metadata['n_train_samples']}")
    print(f"  Val samples: {metadata['n_val_samples']}")
    print(f"  Test samples: {metadata['n_test_samples']}")
    print(f"  Original scaler: {metadata.get('scaler_type', 'unknown')}")

    # Step 1: Load training data
    print("\n[STEP 1] Loading training data...")
    train_data, train_features = load_raw_sequences(data_dir, 'train')
    print(f"  Train features shape: {train_features.shape if hasattr(train_features, 'shape') else 'object array'}")

    # Step 2: Fit scaler on training data ONLY
    print("\n[STEP 2] Fitting StandardScaler on TRAINING data only...")
    scaler = StandardScaler()

    train_flat = flatten_for_scaler(train_features)
    print(f"  Flattened train data shape: {train_flat.shape}")

    # Handle NaN values
    nan_mask = np.isnan(train_flat)
    if nan_mask.any():
        print(f"  Warning: Found {nan_mask.sum()} NaN values, replacing with 0")
        train_flat = np.nan_to_num(train_flat, nan=0.0)

    scaler.fit(train_flat)
    print(f"  Scaler fitted on {train_flat.shape[0]} samples")
    print(f"  Feature means range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"  Feature stds range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")

    # Step 3: Apply scaler to all splits
    print("\n[STEP 3] Applying scaler to all splits...")

    for split in ['train', 'val', 'test']:
        print(f"\n  Processing {split}...")
        split_data, split_features = load_raw_sequences(data_dir, split)

        # Handle NaN values
        if split_features.dtype == object:
            for i, seq in enumerate(split_features):
                if len(seq) > 0 and np.isnan(seq).any():
                    split_features[i] = np.nan_to_num(seq, nan=0.0)
        else:
            split_features = np.nan_to_num(split_features, nan=0.0)

        # Apply scaler
        scaled_features = apply_scaler_to_sequences(split_features, scaler)
        print(f"    Scaled shape: {scaled_features.shape if hasattr(scaled_features, 'shape') else 'object array'}")

        # Verify scaling
        if scaled_features.dtype != object:
            flat_scaled = flatten_for_scaler(scaled_features)
            print(f"    Mean after scaling: {flat_scaled.mean():.6f} (should be ~0 for train)")
            print(f"    Std after scaling: {flat_scaled.std():.6f} (should be ~1 for train)")

        # Save scaled data
        output_file = output_dir / f'{split}_sequences.npz'

        save_dict = {}
        for key in split_data.keys():
            if key in ['continuous', 'continuous_features']:
                save_dict['continuous'] = scaled_features
            else:
                save_dict[key] = split_data[key]

        np.savez(output_file, **save_dict)
        print(f"    Saved to: {output_file}")

    # Save updated metadata
    new_metadata = metadata.copy()
    new_metadata['scaler_type'] = 'standard (train-only)'
    new_metadata['data_leakage_fixed'] = True
    new_metadata['scaler_fitted_on'] = 'train_only'

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(new_metadata, f, indent=2)
    print(f"\nSaved metadata to: {output_dir / 'metadata.json'}")

    # Save scaler statistics for reference
    scaler_stats = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'var': scaler.var_.tolist(),
        'n_samples_seen': int(scaler.n_samples_seen_),
    }
    with open(output_dir / 'scaler_stats.json', 'w') as f:
        json.dump(scaler_stats, f, indent=2)
    print(f"Saved scaler stats to: {output_dir / 'scaler_stats.json'}")

    print("\n" + "=" * 70)
    print("DATA LEAKAGE FIX COMPLETE")
    print("=" * 70)
    print(f"\nCorrected data saved to: {output_dir}")
    print("\nNext step: Rerun training with this corrected data to compare results.")


if __name__ == '__main__':
    main()
