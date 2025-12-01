#!/usr/bin/env python3
"""
Analyze operation type distribution in the processed data.

This script checks if operation_type classification is more balanced
than raw command classification, making it potentially more useful
for machine fingerprinting.

Usage:
    python scripts/analyze_operation_distribution.py \
        --data-dir outputs/processed_hybrid
"""

import argparse
import numpy as np
import json
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from miracle.dataset.target_utils import TokenDecomposer


def analyze_dataset(data_path: Path, dataset_name: str, decomposer=None):
    """Analyze distribution of operation types and commands in a dataset."""
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} DATASET ANALYSIS")
    print(f"{'='*80}\n")

    # Load data
    data = np.load(data_path, allow_pickle=True)

    # Basic stats
    n_sequences = len(data['tokens'])
    print(f"Total sequences: {n_sequences}")

    # Operation type distribution
    if 'operation_type' in data:
        op_types = data['operation_type']
        op_counter = Counter(op_types)

        print(f"\n📊 Operation Type Distribution:")
        print(f"{'='*60}")
        total_ops = len(op_types)
        for op_type, count in sorted(op_counter.items(), key=lambda x: -x[1]):
            pct = count / total_ops * 100
            op_label = str(op_type)
            print(f"  {op_label:<20}: {count:5d} ({pct:5.2f}%)")

        # Calculate balance metrics
        op_values = list(op_counter.values())
        max_op = max(op_values)
        min_op = min(op_values)
        imbalance_ratio = max_op / min_op if min_op > 0 else float('inf')

        print(f"\n  Balance metrics:")
        print(f"    Max/Min ratio: {imbalance_ratio:.2f}x")
        print(f"    Gini coefficient: {calculate_gini(op_values):.3f} (0=perfect balance, 1=perfect imbalance)")
    else:
        print("\n⚠️  No operation_type found in data")

    # Command distribution (from tokens)
    if decomposer and 'tokens' in data:
        print(f"\n📊 Command Distribution:")
        print(f"{'='*60}")

        tokens = data['tokens']
        all_tokens = [t for seq in tokens for t in seq]

        command_counts = Counter()
        for token_id in all_tokens:
            if token_id == 0:
                continue
            token_type, command_id, _, _ = decomposer.decompose_token(token_id)
            if command_id is not None:
                command_counts[command_id] += 1

        total_commands = sum(command_counts.values())
        for cmd_id, count in sorted(command_counts.items(), key=lambda x: -x[1]):
            pct = count / total_commands * 100
            # Use command ID as label
            cmd_name = f"Command_{cmd_id}"
            print(f"  {cmd_name:20s} (ID {cmd_id:2d}): {count:6d} ({pct:5.2f}%)")

        # Balance metrics
        cmd_values = list(command_counts.values())
        max_cmd = max(cmd_values)
        min_cmd = min(cmd_values)
        cmd_imbalance_ratio = max_cmd / min_cmd if min_cmd > 0 else float('inf')

        print(f"\n  Balance metrics:")
        print(f"    Max/Min ratio: {cmd_imbalance_ratio:.2f}x")
        print(f"    Gini coefficient: {calculate_gini(cmd_values):.3f}")

    # File distribution
    if 'file_ids' in data:
        file_ids = data['file_ids']
        file_counter = Counter(file_ids)
        print(f"\nFile distribution:")
        print(f"  Unique files: {len(file_counter)}")
        print(f"  Sequences per file (avg): {n_sequences / len(file_counter):.1f}")

    return {
        'n_sequences': n_sequences,
        'operation_counter': op_counter if 'operation_type' in data else None,
        'command_counter': command_counts if decomposer else None,
    }


def calculate_gini(values):
    """Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)."""
    if not values:
        return 0.0

    values = sorted(values)
    n = len(values)
    cumsum = np.cumsum(values)

    # Gini = (2 * sum of (i * value_i)) / (n * sum(values)) - (n+1)/n
    gini = (2.0 * sum((i+1) * val for i, val in enumerate(values))) / (n * sum(values)) - (n + 1.0) / n
    return gini


def compare_balance(train_stats, val_stats, test_stats):
    """Compare operation vs command balance across splits."""
    print(f"\n{'='*80}")
    print(f"BALANCE COMPARISON: Operation Type vs Command")
    print(f"{'='*80}\n")

    # Operation type balance
    print("Operation Type Balance:")
    if train_stats['operation_counter']:
        train_op_vals = list(train_stats['operation_counter'].values())
        val_op_vals = list(val_stats['operation_counter'].values())
        test_op_vals = list(test_stats['operation_counter'].values())

        print(f"  Train Gini: {calculate_gini(train_op_vals):.3f}")
        print(f"  Val Gini:   {calculate_gini(val_op_vals):.3f}")
        print(f"  Test Gini:  {calculate_gini(test_op_vals):.3f}")

        train_max_min = max(train_op_vals) / min(train_op_vals)
        print(f"  Train Max/Min: {train_max_min:.2f}x")

    # Command balance
    print("\nCommand Balance:")
    if train_stats['command_counter']:
        train_cmd_vals = list(train_stats['command_counter'].values())
        val_cmd_vals = list(val_stats['command_counter'].values())
        test_cmd_vals = list(test_stats['command_counter'].values())

        print(f"  Train Gini: {calculate_gini(train_cmd_vals):.3f}")
        print(f"  Val Gini:   {calculate_gini(val_cmd_vals):.3f}")
        print(f"  Test Gini:  {calculate_gini(test_cmd_vals):.3f}")

        train_max_min = max(train_cmd_vals) / min(train_cmd_vals)
        print(f"  Train Max/Min: {train_max_min:.2f}x")

    # Recommendation
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION")
    print(f"{'='*80}\n")

    if train_stats['operation_counter'] and train_stats['command_counter']:
        op_gini = calculate_gini(list(train_stats['operation_counter'].values()))
        cmd_gini = calculate_gini(list(train_stats['command_counter'].values()))

        if op_gini < cmd_gini * 0.8:  # Significantly more balanced
            print("✅ Operation type classification is SIGNIFICANTLY more balanced!")
            print("   Recommended approach: Focus on operation_type head for evaluation.")
            print(f"   Operation Gini: {op_gini:.3f} vs Command Gini: {cmd_gini:.3f}")
        elif op_gini < cmd_gini:
            print("✅ Operation type classification is somewhat more balanced.")
            print("   Consider using operation_type as primary metric.")
        else:
            print("⚠️  Command classification is as balanced or more balanced than operations.")
            print("   Continue focusing on command-level improvements.")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze operation type and command distribution')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory with processed data')
    parser.add_argument('--vocab-path', type=str,
                       default='data/vocabulary_1digit_hybrid.json',
                       help='Path to vocabulary file')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load decomposer for command analysis
    decomposer = None
    if Path(args.vocab_path).exists():
        decomposer = TokenDecomposer(args.vocab_path)
        print(f"✅ Loaded TokenDecomposer from {args.vocab_path}")
    else:
        print(f"⚠️  Vocabulary not found at {args.vocab_path}")
        print(f"   Will skip command distribution analysis")

    # Analyze each split
    train_stats = analyze_dataset(
        data_dir / 'train_sequences.npz', 'train', decomposer)
    val_stats = analyze_dataset(
        data_dir / 'val_sequences.npz', 'validation', decomposer)
    test_stats = analyze_dataset(
        data_dir / 'test_sequences.npz', 'test', decomposer)

    # Compare balance
    compare_balance(train_stats, val_stats, test_stats)

    # Save results
    output_path = data_dir / 'distribution_analysis.json'
    results = {
        'train': {
            'n_sequences': train_stats['n_sequences'],
            'operation_distribution': dict(train_stats['operation_counter']) if train_stats['operation_counter'] else None,
            'command_distribution': dict(train_stats['command_counter']) if train_stats['command_counter'] else None,
        },
        'val': {
            'n_sequences': val_stats['n_sequences'],
            'operation_distribution': dict(val_stats['operation_counter']) if val_stats['operation_counter'] else None,
            'command_distribution': dict(val_stats['command_counter']) if val_stats['command_counter'] else None,
        },
        'test': {
            'n_sequences': test_stats['n_sequences'],
            'operation_distribution': dict(test_stats['operation_counter']) if test_stats['operation_counter'] else None,
            'command_distribution': dict(test_stats['command_counter']) if test_stats['command_counter'] else None,
        },
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
