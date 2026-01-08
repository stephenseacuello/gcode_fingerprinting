#!/usr/bin/env python3
"""
Verify vocabulary coverage in processed data splits.

This script verifies that ALL tokens in the processed train/val/test splits
exist in the vocabulary (zero OOV tokens).

Usage:
    python scripts/verify_vocabulary_coverage.py \
        --vocab-path data/vocabulary_4digit_full.json \
        --data-dir outputs/processed_v3
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
import argparse
import numpy as np
from collections import Counter


def load_vocabulary(vocab_path: Path) -> dict:
    """Load vocabulary from JSON file."""
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    vocab = vocab_data.get('vocab', vocab_data.get('token2id', {}))
    return vocab


def verify_split(split_path: Path, vocab: dict, inv_vocab: dict) -> dict:
    """Verify vocabulary coverage for a single split."""
    data = np.load(split_path, allow_pickle=True)

    # Get token IDs
    tokens = data['tokens']

    # Count unique tokens and find OOV
    unique_token_ids = set()
    oov_ids = set()
    total_tokens = 0
    oov_occurrences = 0

    # Get vocab size
    vocab_size = len(vocab)

    for seq in tokens:
        for token_id in seq:
            token_id = int(token_id)
            if token_id == 0:  # PAD token, skip
                continue
            total_tokens += 1
            unique_token_ids.add(token_id)

            # Check if token ID is in vocabulary range
            if token_id >= vocab_size:
                oov_ids.add(token_id)
                oov_occurrences += 1

    # Try to decode OOV tokens if inv_vocab available
    oov_tokens = []
    for tid in oov_ids:
        if tid in inv_vocab:
            oov_tokens.append(f"{inv_vocab[tid]} (id={tid})")
        else:
            oov_tokens.append(f"<UNKNOWN_ID={tid}>")

    return {
        'num_sequences': len(tokens),
        'total_tokens': total_tokens,
        'unique_token_ids': len(unique_token_ids),
        'oov_count': len(oov_ids),
        'oov_occurrences': oov_occurrences,
        'oov_tokens': oov_tokens[:20],  # First 20 OOV tokens
        'coverage': 100.0 * (1 - len(oov_ids) / max(1, len(unique_token_ids)))
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify vocabulary coverage in processed splits"
    )
    parser.add_argument('--vocab-path', type=str,
                        default='data/vocabulary_4digit_full.json',
                        help='Path to vocabulary JSON file')
    parser.add_argument('--data-dir', type=str,
                        default='outputs/processed_v3',
                        help='Directory containing processed .npz files')

    args = parser.parse_args()

    vocab_path = Path(args.vocab_path)
    data_dir = Path(args.data_dir)

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}...")
    vocab = load_vocabulary(vocab_path)
    inv_vocab = {v: k for k, v in vocab.items()}
    print(f"  Vocabulary size: {len(vocab)} tokens")

    print(f"\n{'='*60}")
    print("VOCABULARY COVERAGE VERIFICATION")
    print(f"{'='*60}")

    all_passed = True
    splits = ['train', 'val', 'test']

    for split in splits:
        split_path = data_dir / f'{split}_sequences.npz'
        if not split_path.exists():
            print(f"\n{split.upper()} split: NOT FOUND")
            continue

        result = verify_split(split_path, vocab, inv_vocab)

        print(f"\n{split.upper()} split: {result['num_sequences']} sequences")
        print(f"  Total tokens: {result['total_tokens']:,}")
        print(f"  Unique token IDs: {result['unique_token_ids']}")
        print(f"  OOV token IDs: {result['oov_count']}", end="")

        if result['oov_count'] == 0:
            print(" ✓")
        else:
            print(f" ✗ ({result['oov_occurrences']:,} occurrences)")
            all_passed = False
            print(f"  OOV samples: {result['oov_tokens']}")

        print(f"  Coverage: {result['coverage']:.1f}%")

    print(f"\n{'='*60}")
    if all_passed:
        print("✓ VERIFICATION PASSED: Zero OOV tokens in all splits!")
        print("  The vocabulary covers 100% of tokens in the processed data.")
    else:
        print("✗ VERIFICATION FAILED: OOV tokens found!")
        print("  Recommendation: Rebuild vocabulary to include missing tokens.")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
