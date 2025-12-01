#!/usr/bin/env python3
"""
Add raw numeric values to already-processed data for direct regression.

This script reads preprocessed .npz files that have gcode_texts,
extracts raw numeric values from the G-code, and saves updated .npz files.
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


def extract_raw_values_from_gcode(gcode_text: str, tokenizer: GCodeTokenizer) -> Dict[int, float]:
    """
    Extract raw numeric values from G-code text.

    For each numeric parameter (e.g., X34.7):
    1. Extract the raw value (34.7)
    2. Map it to the token position

    Returns dict mapping token position to raw numeric value.
    """
    raw_values_dict = {}

    if not gcode_text or gcode_text == '':
        return raw_values_dict

    # Tokenize to get token positions
    try:
        canon = tokenizer.canonicalize([gcode_text])
        tokens = tokenizer.tokenize_canonical(canon)
    except Exception as e:
        print(f"Warning: Failed to tokenize G-code: {e}")
        return raw_values_dict

    # Parse the raw G-code to get actual values
    raw_values = {}
    for match in NUMERIC_PATTERN.finditer(str(gcode_text).upper()):
        param_type = match.group(1)  # e.g., 'X'
        value = float(match.group(2))  # e.g., 34.7
        raw_values[param_type] = value

    # Match tokens with raw values
    for i, token in enumerate(tokens):
        if isinstance(token, str) and token.startswith('NUM_'):
            # Parse token format: NUM_X_3 or NUM_X_34 (depending on bucketing)
            parts = token.split('_')
            if len(parts) >= 2:
                param_type = parts[1]  # 'X'

                # Get the raw value for this parameter
                if param_type in raw_values:
                    raw_value = raw_values[param_type]
                    raw_values_dict[i] = float(raw_value)

    return raw_values_dict


def add_raw_values_to_npz(input_file: Path, output_file: Path, tokenizer: GCodeTokenizer):
    """
    Add raw numeric values to an existing .npz file.

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

    # Check if residuals exist (from previous hybrid bucketing attempts)
    residuals = data.get('residuals', None)

    n_samples = len(tokens)
    max_seq_len = tokens.shape[1]

    print(f"  {n_samples} samples, max sequence length: {max_seq_len}")

    # Initialize raw values array
    raw_values_data = np.zeros((n_samples, max_seq_len), dtype=np.float32)

    # Extract raw values for each sample
    non_zero_count = 0
    for i in range(n_samples):
        gcode_text = gcode_texts[i]
        if gcode_text and str(gcode_text) != '':
            raw_values = extract_raw_values_from_gcode(str(gcode_text), tokenizer)

            # Fill raw values array
            for token_idx, raw_value in raw_values.items():
                if token_idx < max_seq_len:
                    raw_values_data[i, token_idx] = raw_value
                    non_zero_count += 1

        if (i + 1) % 500 == 0:
            print(f"    Processed {i + 1}/{n_samples} samples...")

    print(f"  Extracted {non_zero_count} non-zero raw values")

    # Save updated data
    save_dict = {
        'continuous': continuous,
        'categorical': categorical,
        'tokens': tokens,
        'param_value_raw': raw_values_data,  # NEW! Raw values for direct regression
        'lengths': lengths,
        'gcode_texts': gcode_texts,
        'operation_type': operation_type,
    }

    # Include residuals if they exist (for backwards compatibility)
    if residuals is not None:
        save_dict['residuals'] = residuals

    np.savez_compressed(output_file, **save_dict)

    print(f"  ✓ Saved to {output_file}")
    print(f"    Raw values shape: {raw_values_data.shape}")
    print(f"    Non-zero values: {non_zero_count} ({100*non_zero_count/(n_samples*max_seq_len):.2f}%)")
    print(f"    Value range: [{raw_values_data[raw_values_data != 0].min():.2f}, {raw_values_data[raw_values_data != 0].max():.2f}]")


def main():
    parser = argparse.ArgumentParser(description='Add raw numeric values to preprocessed data')
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directory with preprocessed .npz files')
    parser.add_argument('--vocab-path', type=Path, required=True,
                        help='Path to vocabulary JSON')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files (default: create backup)')

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.vocab_path}")
    tokenizer = GCodeTokenizer.load(args.vocab_path)

    # Find all .npz files
    npz_files = list(args.data_dir.glob('*_sequences.npz'))

    if not npz_files:
        # Try without _sequences suffix
        npz_files = list(args.data_dir.glob('*.npz'))

    if not npz_files:
        print(f"ERROR: No .npz files found in {args.data_dir}")
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

        add_raw_values_to_npz(npz_file, output_file, tokenizer)

    print("\n✅ All files processed!")
    print(f"\nUpdated files in: {args.data_dir}")
    print("\nRaw values have been added as 'param_value_raw' field for direct regression training.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
