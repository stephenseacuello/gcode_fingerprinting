#!/usr/bin/env python3
"""
Generate extreme class weights to combat severe class imbalance.

For severely imbalanced datasets (like G0 being 53% of tokens), we need
MUCH more aggressive weighting than standard inverse frequency.

This script generates weights with:
- Minimum weight: 0.001-0.01 (for majority class G0)
- Maximum weight: 50-100 (for rare classes)
- Ratio: 1000-10000x instead of typical 10x

Usage:
    python scripts/generate_extreme_class_weights.py \
        --data-dir outputs/processed_hybrid \
        --vocab-path data/vocabulary_1digit_hybrid.json \
        --output outputs/class_weights_extreme.json \
        --command-min-weight 0.001 \
        --command-max-weight 100.0 \
        --strategy sqrt
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from miracle.dataset.target_utils import TokenDecomposer


def compute_extreme_weights(frequencies, min_weight=0.001, max_weight=100.0, strategy='sqrt'):
    """
    Compute extreme class weights for highly imbalanced data.

    Args:
        frequencies: Dict of {class_id: count}
        min_weight: Minimum weight (for majority class)
        max_weight: Maximum weight (for rare classes)
        strategy: 'sqrt', 'log', 'linear', or 'quadratic'

    Returns:
        List of weights
    """
    total = sum(frequencies.values())
    n_classes = len(frequencies)

    # Compute base inverse frequencies
    inv_freqs = {}
    for cls_id, count in frequencies.items():
        freq = count / total
        inv_freqs[cls_id] = 1.0 / freq

    # Apply strategy
    if strategy == 'sqrt':
        # Square root dampens extreme values while keeping relative order
        transformed = {k: np.sqrt(v) for k, v in inv_freqs.items()}
    elif strategy == 'log':
        # Log dampens even more aggressively
        transformed = {k: np.log1p(v) for k, v in inv_freqs.items()}
    elif strategy == 'quadratic':
        # Square amplifies differences (very aggressive)
        transformed = {k: v ** 2 for k, v in inv_freqs.items()}
    else:  # linear
        transformed = inv_freqs

    # Normalize to [min_weight, max_weight] range
    min_val = min(transformed.values())
    max_val = max(transformed.values())

    weights = []
    for cls_id in sorted(frequencies.keys()):
        if cls_id in transformed:
            # Scale to target range
            normalized = (transformed[cls_id] - min_val) / (max_val - min_val)
            weight = min_weight + normalized * (max_weight - min_weight)
            weights.append(weight)
        else:
            weights.append(1.0)

    return weights


def main():
    parser = argparse.ArgumentParser(
        description='Generate extreme class weights for imbalanced data')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory with processed data')
    parser.add_argument('--vocab-path', type=str, required=True,
                       help='Path to vocabulary file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for class weights JSON')
    parser.add_argument('--command-min-weight', type=float, default=0.001,
                       help='Minimum weight for majority command (default: 0.001)')
    parser.add_argument('--command-max-weight', type=float, default=100.0,
                       help='Maximum weight for rare commands (default: 100.0)')
    parser.add_argument('--param-type-min-weight', type=float, default=0.1,
                       help='Minimum weight for param types (default: 0.1)')
    parser.add_argument('--param-type-max-weight', type=float, default=10.0,
                       help='Maximum weight for rare param types (default: 10.0)')
    parser.add_argument('--strategy', type=str, default='sqrt',
                       choices=['linear', 'sqrt', 'log', 'quadratic'],
                       help='Weighting strategy (default: sqrt)')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    print(f"\n{'='*80}")
    print(f"GENERATING EXTREME CLASS WEIGHTS")
    print(f"{'='*80}\n")

    # Load vocabulary and decomposer
    print(f"Loading vocabulary: {args.vocab_path}")
    decomposer = TokenDecomposer(args.vocab_path)

    # Load training data
    train_file = data_dir / 'train_sequences.npz'
    print(f"Loading training data: {train_file}")
    train_data = np.load(train_file, allow_pickle=True)
    tokens = train_data['tokens']

    print(f"  Total sequences: {len(tokens)}")

    # Flatten all tokens
    all_tokens = [t for seq in tokens for t in seq]
    print(f"  Total tokens: {len(all_tokens)}")

    # Decompose tokens to get command IDs, param type IDs, etc.
    print("\nDecomposing tokens...")
    command_counts = Counter()
    param_type_counts = Counter()
    param_value_counts = Counter()

    for token_id in all_tokens:
        if token_id == 0:  # PAD or special token
            continue

        # Get decomposition (returns tuple: type, command_id, param_type_id, param_value_id)
        token_type, command_id, param_type_id, param_value_id = decomposer.decompose_token(token_id)

        if command_id is not None:
            command_counts[command_id] += 1
        if param_type_id is not None:
            param_type_counts[param_type_id] += 1
        if param_value_id is not None:
            param_value_counts[param_value_id] += 1

    print(f"\nClass distributions:")
    print(f"  Commands: {len(command_counts)} classes")
    print(f"  Param types: {len(param_type_counts)} classes")
    print(f"  Param values: {len(param_value_counts)} classes")

    # Show distribution
    print(f"\nTop 5 commands:")
    for cmd_id, count in command_counts.most_common(5):
        pct = count / sum(command_counts.values()) * 100
        print(f"  Command {cmd_id}: {count:6d} ({pct:5.2f}%)")

    print(f"\nCommand distribution:")
    for cmd_id, count in sorted(command_counts.items()):
        pct = count / sum(command_counts.values()) * 100
        print(f"  Command {cmd_id}: {count:6d} ({pct:5.2f}%)")

    # Compute extreme weights for commands
    print(f"\n{'='*80}")
    print(f"COMPUTING EXTREME WEIGHTS")
    print(f"{'='*80}\n")
    print(f"Commands:")
    print(f"  Strategy: {args.strategy}")
    print(f"  Min weight: {args.command_min_weight}")
    print(f"  Max weight: {args.command_max_weight}")
    print(f"  Ratio: {args.command_max_weight / args.command_min_weight:.1f}x")

    command_weights = compute_extreme_weights(
        {i: command_counts.get(i, 1) for i in range(decomposer.n_commands)},
        min_weight=args.command_min_weight,
        max_weight=args.command_max_weight,
        strategy=args.strategy
    )

    print(f"\nParam types:")
    print(f"  Strategy: {args.strategy}")
    print(f"  Min weight: {args.param_type_min_weight}")
    print(f"  Max weight: {args.param_type_max_weight}")
    print(f"  Ratio: {args.param_type_max_weight / args.param_type_min_weight:.1f}x")

    param_type_weights = compute_extreme_weights(
        {i: param_type_counts.get(i, 1) for i in range(decomposer.n_param_types)},
        min_weight=args.param_type_min_weight,
        max_weight=args.param_type_max_weight,
        strategy=args.strategy
    )

    # For param values, use less extreme weights (already working reasonably)
    param_value_weights = compute_extreme_weights(
        {i: param_value_counts.get(i, 1) for i in range(decomposer.n_param_values)},
        min_weight=0.5,
        max_weight=5.0,
        strategy='sqrt'
    )

    # Create output
    weights_dict = {
        'command_weights': [float(w) for w in command_weights],
        'param_type_weights': [float(w) for w in param_type_weights],
        'param_value_weights': [float(w) for w in param_value_weights],
        'metadata': {
            'strategy': args.strategy,
            'command_min_weight': args.command_min_weight,
            'command_max_weight': args.command_max_weight,
            'param_type_min_weight': args.param_type_min_weight,
            'param_type_max_weight': args.param_type_max_weight,
            'total_tokens': len(all_tokens),
            'n_commands': decomposer.n_commands,
            'n_param_types': decomposer.n_param_types,
            'n_param_values': decomposer.n_param_values,
        }
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(weights_dict, f, indent=2)

    print(f"\n{'='*80}")
    print(f"WEIGHTS GENERATED")
    print(f"{'='*80}\n")
    print(f"Saved to: {output_path}\n")

    print(f"Command weights preview:")
    for i, w in enumerate(command_weights[:decomposer.n_commands]):
        count = command_counts.get(i, 0)
        pct = count / sum(command_counts.values()) * 100 if count > 0 else 0
        print(f"  Command {i}: weight={w:8.4f} (freq={pct:5.2f}%)")

    print(f"\nParam type weights preview:")
    for i, w in enumerate(param_type_weights[:decomposer.n_param_types]):
        count = param_type_counts.get(i, 0)
        pct = count / sum(param_type_counts.values()) * 100 if count > 0 else 0
        print(f"  Param type {i}: weight={w:8.4f} (freq={pct:5.2f}%)")

    print(f"\n✅ Extreme class weights generated successfully!")
    print(f"\nUsage in training:")
    print(f"  --class-weights-path={output_path}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
