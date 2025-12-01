#!/usr/bin/env python3
"""
Analyze class frequencies and generate class weights.

This script analyzes the training data to:
1. Compute class frequency distributions
2. Generate class weights to address imbalance
3. Save weights to JSON file for use in training

Usage:
    python scripts/analyze_class_frequencies.py \
        --data-dir outputs/processed_hybrid \
        --vocab-path data/vocabulary_1digit_hybrid.json \
        --output outputs/class_weights.json \
        --method inverse
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from torch.utils.data import DataLoader
from miracle.dataset.dataset import GCodeDataset, collate_fn
from miracle.dataset.target_utils import TokenDecomposer
from miracle.training.mode_collapse_prevention import (
    compute_class_frequencies,
    compute_class_weights,
    save_class_weights
)


def main():
    parser = argparse.ArgumentParser(description='Analyze class frequencies and generate weights')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to preprocessed data directory')
    parser.add_argument('--vocab-path', type=str, required=True,
                       help='Path to vocabulary file')
    parser.add_argument('--output', type=str, default='outputs/class_weights.json',
                       help='Output path for class weights JSON')
    parser.add_argument('--method', type=str, default='inverse',
                       choices=['inverse', 'sqrt_inverse'],
                       help='Weight computation method')
    parser.add_argument('--smooth', type=float, default=0.1,
                       help='Smoothing factor to prevent extreme weights')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for data loading')

    args = parser.parse_args()

    # Validate paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return 1

    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        print(f"❌ Vocabulary file not found: {vocab_path}")
        return 1

    print("\n" + "=" * 80)
    print("CLASS FREQUENCY ANALYSIS")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Vocabulary: {vocab_path}")
    print(f"Method: {args.method}")
    print(f"Smoothing: {args.smooth}")
    print("=" * 80 + "\n")

    # Create decomposer
    decomposer = TokenDecomposer(str(vocab_path))
    print(f"✓ Vocabulary loaded: {decomposer.vocab_size} tokens")
    print(f"  Commands: {decomposer.n_commands}")
    print(f"  Param types: {decomposer.n_param_types}")
    print(f"  Param values: {decomposer.n_param_values}")

    # Load training dataset
    train_path = data_dir / 'train_sequences.npz'
    if not train_path.exists():
        print(f"\n❌ Training data not found: {train_path}")
        return 1

    train_dataset = GCodeDataset(str(train_path))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"\n✓ Training dataset loaded: {len(train_dataset)} sequences")

    # Compute frequencies
    frequencies = compute_class_frequencies(train_loader, decomposer)

    # Compute weights
    weights = compute_class_weights(
        frequencies,
        method=args.method,
        smooth=args.smooth
    )

    # Save weights
    output_path = Path(args.output)
    save_class_weights(weights, output_path)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nTo use these weights in training, add:")
    print(f"  --class-weights-path {output_path}")
    print("\nThis will enable class-balanced loss to prevent mode collapse.")
    print("=" * 80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
