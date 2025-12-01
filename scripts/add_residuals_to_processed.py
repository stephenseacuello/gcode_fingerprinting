#!/usr/bin/env python3
"""
Add residuals to already-processed data.

This script reads preprocessed .npz files that have gcode_texts,
extracts residuals from the G-code, and saves updated .npz files with residuals.
"""
import argparse
import numpy as np
import re
import sys
from pathlib import Path
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.utilities.gcode_tokenizer import GCodeTokenizer

# Pattern to extract numeric values from G-code
NUMERIC_PATTERN = re.compile(r'([A-Z])(-?\d+(?:\.\d+)?)')


def extract_residuals_from_gcode(gcode_text: str, tokenizer: GCodeTokenizer) -> Dict[int, float]:
    """
    Extract residual values from G-code text.

    For each numeric parameter (e.g., X34.7):
    1. Extract the raw value (34.7)
    2. Determine the coarse bucket (3 for 1-digit bucketing)
    3. Calculate residual: 34.7 - (3 * 10) = 4.7

    Returns dict mapping token position to residual value.
    """
    residuals = {}

    if not gcode_text or gcode_text == '':
        return residuals

    # Tokenize to get the bucketed tokens
    try:
        canon = tokenizer.canonicalize([gcode_text])
        tokens = tokenizer.tokenize_canonical(canon)
    except Exception as e:
        print(f"Warning: Failed to tokenize G-code: {e}")
        return residuals

    # Parse the raw G-code to get actual values
    raw_values = {}
    for match in NUMERIC_PATTERN.finditer(str(gcode_text).upper()):
        param_type = match.group(1)  # e.g., 'X'
        value = float(match.group(2))  # e.g., 34.7
        raw_values[param_type] = value

    # Match tokens with raw values to compute residuals
    for i, token in enumerate(tokens):
        if isinstance(token, str) and token.startswith('NUM_'):
            # Parse token format: NUM_X_3
            parts = token.split('_')
            if len(parts) >= 3:
                param_type = parts[1]  # 'X'
                try:
                    coarse_bucket = int(parts[2])  # 3
                except (ValueError, IndexError):
                    continue

                # Get the raw value for this parameter
                if param_type in raw_values:
                    raw_value = raw_values[param_type]

                    # Compute residual
                    # For positive values: residual = raw_value - (coarse_bucket * 10)
                    if raw_value >= 0:
                        residual = raw_value - (coarse_bucket * 10)
                    else:
                        # For negative values, use absolute residual
                        abs_value = abs(raw_value)
                        abs_bucket = coarse_bucket * 10
                        residual = abs_value - abs_bucket

                    # Normalize residual to [0, 9.9] range
                    residual = max(0.0, min(9.9, abs(residual)))
                    residuals[i] = float(residual)

    return residuals


def add_residuals_to_npz(input_file: Path, output_file: Path, tokenizer: GCodeTokenizer):
    """
    Add residuals to an existing .npz file.

    Args:
        input_file: Input .npz file path
        output_file: Output .npz file path (can be same as input to overwrite)
        tokenizer: GCodeTokenizer instance
    """
    print(f"\nProcessing {input_file.name}...")

    # Load existing data
    data = np.load(input_file, allow_pickle=True)

    # Extract arrays
    continuous = data['continuous']
    categorical = data['categorical']
    tokens = data['tokens']
    lengths = data['lengths']
    gcode_texts = data['gcode_texts']
    operation_type = data['operation_type']

    n_samples = len(tokens)
    max_seq_len = tokens.shape[1]

    print(f"  {n_samples} samples, max sequence length: {max_seq_len}")

    # Initialize residual array
    residual_data = np.zeros((n_samples, max_seq_len), dtype=np.float32)

    # Extract residuals for each sample
    non_zero_count = 0
    for i in range(n_samples):
        gcode_text = gcode_texts[i]
        if gcode_text and str(gcode_text) != '':
            residuals = extract_residuals_from_gcode(str(gcode_text), tokenizer)

            # Fill residual array
            for token_idx, residual_value in residuals.items():
                if token_idx < max_seq_len:
                    residual_data[i, token_idx] = residual_value
                    non_zero_count += 1

        if (i + 1) % 500 == 0:
            print(f"    Processed {i + 1}/{n_samples} samples...")

    print(f"  Extracted {non_zero_count} non-zero residuals")

    # Save updated data
    np.savez_compressed(
        output_file,
        continuous=continuous,
        categorical=categorical,
        tokens=tokens,
        residuals=residual_data,  # NEW!
        lengths=lengths,
        gcode_texts=gcode_texts,
        operation_type=operation_type,
    )

    print(f"  ✓ Saved to {output_file}")
    print(f"    Residuals shape: {residual_data.shape}")
    print(f"    Non-zero residuals: {non_zero_count} ({100*non_zero_count/(n_samples*max_seq_len):.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Add residuals to preprocessed data')
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directory with preprocessed .npz files')
    parser.add_argument('--vocab-path', type=Path, required=True,
                        help='Path to vocabulary JSON (must use 1-digit bucketing)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files (default: create backup)')

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.vocab_path}")
    tokenizer = GCodeTokenizer.load(args.vocab_path)

    # Find all .npz files
    npz_files = list(args.data_dir.glob('*_sequences.npz'))

    if not npz_files:
        print(f"ERROR: No *_sequences.npz files found in {args.data_dir}")
        return 1

    print(f"\nFound {len(npz_files)} files to process:")
    for f in npz_files:
        print(f"  - {f.name}")

    # Process each file
    for npz_file in npz_files:
        if args.overwrite:
            output_file = npz_file
        else:
            # Create backup
            backup_file = npz_file.with_suffix('.npz.backup')
            if not backup_file.exists():
                import shutil
                shutil.copy2(npz_file, backup_file)
                print(f"\n  Created backup: {backup_file.name}")
            output_file = npz_file

        add_residuals_to_npz(npz_file, output_file, tokenizer)

    print("\n✅ All files processed!")
    print(f"\nUpdated files in: {args.data_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
