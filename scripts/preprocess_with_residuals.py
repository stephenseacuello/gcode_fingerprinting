#!/usr/bin/env python3
"""
Preprocessing script with real residual extraction for hybrid bucketing.

This script:
1. Uses 1-digit bucketing for coarse classification (10 classes)
2. Extracts actual residual values for regression
3. Stores both in the preprocessed data
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import re
from typing import List, Dict, Tuple
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.utilities.gcode_tokenizer import GCodeTokenizer
from miracle.dataset.preprocessing import GCodePreprocessor, extract_operation_type
from sklearn.model_selection import train_test_split
import pandas as pd


# Pattern to extract numeric values from G-code
NUMERIC_PATTERN = re.compile(r'([A-Z])(-?\d+(?:\.\d+)?)')


def extract_residuals_from_gcode(gcode_text: str, tokenizer: GCodeTokenizer) -> Dict[str, float]:
    """
    Extract residual values from G-code text.

    For each numeric parameter (e.g., X34.7):
    1. Extract the raw value (34.7)
    2. Determine the coarse bucket (3 for 1-digit bucketing)
    3. Calculate residual: 34.7 - (3 * 10) = 4.7

    Returns dict mapping token position to residual value.
    """
    residuals = {}

    # Tokenize to get the bucketed tokens
    canon = tokenizer.canonicalize([gcode_text])
    tokens = tokenizer.tokenize_canonical(canon)

    # Parse the raw G-code to get actual values
    raw_values = {}
    for match in NUMERIC_PATTERN.finditer(gcode_text.upper()):
        param_type = match.group(1)  # e.g., 'X'
        value = float(match.group(2))  # e.g., 34.7
        raw_values[param_type] = value

    # Match tokens with raw values to compute residuals
    for i, token in enumerate(tokens):
        if token.startswith('NUM_'):
            # Parse token format: NUM_X_3
            parts = token.split('_')
            if len(parts) >= 3:
                param_type = parts[1]  # 'X'
                coarse_bucket = int(parts[2])  # 3

                # Get the raw value for this parameter
                if param_type in raw_values:
                    raw_value = raw_values[param_type]

                    # Compute residual
                    # For positive values: residual = raw_value - (coarse_bucket * 10)
                    # For negative values: need to handle sign appropriately
                    if raw_value >= 0:
                        residual = raw_value - (coarse_bucket * 10)
                    else:
                        # Negative numbers are bucketed differently
                        # -125 → bucket=-1, so residual = -125 - (-10) = -115
                        # But we want residuals in range [0, 9.9] for stability
                        # So we'll use absolute residual within bucket
                        abs_value = abs(raw_value)
                        abs_bucket = coarse_bucket * 10
                        residual = abs_value - abs_bucket
                        if residual < 0:
                            residual = 0  # Clamp to valid range

                    # Normalize residual to [0, 9.9] range
                    residual = max(0, min(9.9, abs(residual)))
                    residuals[i] = residual

    return residuals


def preprocess_with_residuals(
    data_dir: Path,
    output_dir: Path,
    vocab_path: Path,
    window_size: int = 64,
    stride: int = 16,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
):
    """
    Preprocess G-code data with residual extraction.
    """

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer from {vocab_path}")
    tokenizer = GCodeTokenizer.load(vocab_path)

    # Initialize preprocessor
    preprocessor = GCodePreprocessor(vocab_path, window_size=window_size, stride=stride)

    # Get all CSV files
    csv_files = sorted(data_dir.glob('**/*.csv'))
    print(f"Found {len(csv_files)} CSV files")

    # Split files for train/val/test using stratified split by operation type
    # Extract operation types from filenames
    file_operations = []
    for f in csv_files:
        op_type = extract_operation_type(f.name)
        file_operations.append((f, op_type))

    # Create DataFrame for stratified splitting
    df = pd.DataFrame(file_operations, columns=['file', 'operation'])

    # Filter out rare operation types (need at least 2 files per operation for stratification)
    min_samples = 2
    operation_counts = df['operation'].value_counts()
    rare_operations = operation_counts[operation_counts < min_samples].index.tolist()

    if rare_operations:
        print(f"\nWarning: Excluding {len(rare_operations)} rare operation types with <{min_samples} files:")
        for op in rare_operations:
            count = operation_counts[op]
            print(f"  - {op}: {count} file(s)")

        # Filter to only operations with enough samples
        valid_operations = operation_counts[operation_counts >= min_samples].index
        df = df[df['operation'].isin(valid_operations)]
        print(f"  Remaining files: {len(df)}")

    # Get split sizes from ratios
    train_ratio, val_ratio, test_ratio = split_ratios
    test_size = test_ratio
    val_size = val_ratio

    # First split: (train+val) vs test (stratified by operation)
    train_val_files, test_files = train_test_split(
        df,
        test_size=test_size,
        stratify=df['operation'],
        random_state=42
    )

    # Second split: train vs val (stratified by operation)
    val_ratio_adjusted = val_size / (1 - test_size)
    train_files, val_files = train_test_split(
        train_val_files,
        test_size=val_ratio_adjusted,
        stratify=train_val_files['operation'],
        random_state=42
    )

    # Convert back to file lists
    train_files = train_files['file'].tolist()
    val_files = val_files['file'].tolist()
    test_files = test_files['file'].tolist()

    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files,
    }

    print(f"Split sizes: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # Process each split
    for split_name, files in splits.items():
        print(f"\nProcessing {split_name} split ({len(files)} files)...")

        all_windows = []

        # First pass: collect all data for scaler fitting
        if split_name == 'train':
            print("  Fitting scaler on training data...")
            all_continuous = []
            for csv_file in files:
                df = preprocessor.load_csv(csv_file)
                continuous, _, _ = preprocessor.extract_features(df)
                all_continuous.append(continuous)

            # Fit scaler on all training data
            combined_continuous = np.vstack(all_continuous)
            preprocessor.fit_scaler(combined_continuous)

        # Second pass: create windows with residuals
        for csv_file in files:
            print(f"  Processing {csv_file.name}...")

            # Get operation type
            operation_type = extract_operation_type(csv_file.name)

            # Load and extract features
            df = preprocessor.load_csv(csv_file)
            continuous, categorical, gcode_texts = preprocessor.extract_features(df)

            # Transform continuous features
            if preprocessor.fitted:
                continuous = preprocessor.transform(continuous)

            # Create windows
            windows = preprocessor.create_windows(continuous, categorical, gcode_texts, operation_type)

            # Add residuals to each window
            for window in windows:
                gcode_text = window['gcode_text']
                if gcode_text:
                    # Extract residuals for this G-code
                    residuals = extract_residuals_from_gcode(gcode_text, tokenizer)

                    # Create residual array aligned with token sequence
                    token_ids = window['token_ids']
                    residual_array = np.zeros(len(token_ids), dtype=np.float32)

                    for token_idx, residual_value in residuals.items():
                        if token_idx < len(residual_array):
                            residual_array[token_idx] = residual_value

                    window['residuals'] = residual_array
                else:
                    # No G-code text, use zero residuals
                    window['residuals'] = np.zeros(len(window['token_ids']), dtype=np.float32)

            all_windows.extend(windows)

        # Convert to arrays for saving
        print(f"  Collected {len(all_windows)} windows")

        if all_windows:
            # Stack data
            continuous_data = np.stack([w['continuous'] for w in all_windows])
            categorical_data = np.stack([w['categorical'] for w in all_windows])

            # Token sequences (pad to max length)
            max_seq_len = max(len(w['token_ids']) for w in all_windows)
            token_data = np.zeros((len(all_windows), max_seq_len), dtype=np.int32)
            residual_data = np.zeros((len(all_windows), max_seq_len), dtype=np.float32)
            lengths = np.array([len(w['token_ids']) for w in all_windows], dtype=np.int32)

            for i, w in enumerate(all_windows):
                seq_len = len(w['token_ids'])
                token_data[i, :seq_len] = w['token_ids']
                residual_data[i, :seq_len] = w['residuals']

            # Operation types
            operation_map = {
                'unknown': 0, 'adaptive': 1, 'face': 2, 'pocket': 3,
                'slot': 4, 'damageface': 5, 'damagepocket': 6,
                'damageslot': 7, 'pocket150025': 8, 'slot025': 9
            }
            operation_types = np.array([operation_map.get(w['operation_type'], 0) for w in all_windows], dtype=np.int32)

            # Save to npz
            output_file = output_dir / f'{split_name}.npz'
            np.savez_compressed(
                output_file,
                continuous=continuous_data,
                categorical=categorical_data,
                tokens=token_data,
                residuals=residual_data,  # NEW: actual residual values
                lengths=lengths,
                operation_type=operation_types,
                gcode_texts=[w['gcode_text'] for w in all_windows],
            )

            print(f"  Saved to {output_file}")
            print(f"    Shapes: continuous={continuous_data.shape}, tokens={token_data.shape}")
            print(f"    Residuals: shape={residual_data.shape}, non-zero={np.count_nonzero(residual_data)}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess G-code data with residual extraction')
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory with CSV files')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory for preprocessed data')
    parser.add_argument('--vocab-path', type=Path, required=True, help='Path to vocabulary JSON (must use 1-digit bucketing)')
    parser.add_argument('--window-size', type=int, default=64, help='Window size')
    parser.add_argument('--stride', type=int, default=16, help='Stride for sliding window')

    args = parser.parse_args()

    # Verify vocabulary uses 1-digit bucketing
    with open(args.vocab_path, 'r') as f:
        vocab_data = json.load(f)

    config = vocab_data.get('config', {})
    bucket_digits = config.get('bucket_digits')

    if bucket_digits != 1:
        print(f"WARNING: Vocabulary uses {bucket_digits}-digit bucketing, but hybrid approach requires 1-digit!")
        print("For best results, use a vocabulary with bucket_digits=1")

    preprocess_with_residuals(
        args.data_dir,
        args.output_dir,
        args.vocab_path,
        args.window_size,
        args.stride,
    )

    print("\n✅ Preprocessing with residuals complete!")


if __name__ == '__main__':
    main()