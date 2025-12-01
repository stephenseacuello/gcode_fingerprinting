#!/usr/bin/env python3
"""
Rebuild vocabulary from actual CSV data files.

This fixes the vocab/tokenization mismatch issue where the pre-built vocabulary
doesn't contain the NUM tokens that are generated during preprocessing.

Solution: Build vocabulary from the same data that will be preprocessed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import argparse
from collections import Counter
from miracle.utilities.gcode_tokenizer import GCodeTokenizer, TokenizerConfig


def extract_gcode_from_csvs(data_dir: Path):
    """Extract all G-code lines from CSV files."""
    print(f"\nExtracting G-code from CSV files in {data_dir}...")

    csv_files = sorted(data_dir.glob('*_aligned.csv'))
    print(f"Found {len(csv_files)} CSV files")

    all_gcode_lines = []
    for csv_file in csv_files:
        print(f"  Reading {csv_file.name}...", end=" ")
        df = pd.read_csv(csv_file)

        if 'gcode_string' in df.columns:
            gcode_lines = df['gcode_string'].dropna().tolist()
            all_gcode_lines.extend(gcode_lines)
            print(f"{len(gcode_lines)} lines")
        else:
            print("WARNING: No 'gcode_string' column found!")

    print(f"\nTotal G-code lines extracted: {len(all_gcode_lines)}")
    return all_gcode_lines


def build_vocabulary(
    gcode_lines: list,
    output_path: Path,
    mode: str = "hybrid",
    bucket_digits: int = 2,
    vocab_size: int = 5000,
    min_freq: int = 1
):
    """Build vocabulary from G-code lines."""
    print(f"\n{'='*80}")
    print(f"BUILDING VOCABULARY")
    print(f"{'='*80}")
    print(f"Mode: {mode}")
    print(f"Bucket digits: {bucket_digits}")
    print(f"Vocab size: {vocab_size}")
    print(f"Min frequency: {min_freq}")

    # Create tokenizer config
    config = TokenizerConfig(
        mode=mode,
        bucket_digits=bucket_digits,
        vocab_size=vocab_size,
        min_freq=min_freq,
        dynamic_numbers=False  # We're building the vocab, don't need dynamic
    )

    tokenizer = GCodeTokenizer(config)

    # Tokenize all lines and count frequencies
    print(f"\nTokenizing {len(gcode_lines)} G-code lines...")
    token_counts = Counter()

    for i, line in enumerate(gcode_lines):
        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i:,} lines...")

        # Canonicalize and tokenize
        canonical = tokenizer.canonicalize([line])
        tokens = tokenizer.tokenize_canonical(canonical)
        token_counts.update(tokens)

    print(f"  Processed {len(gcode_lines):,} lines")
    print(f"\nUnique tokens found: {len(token_counts)}")

    # Show token type distribution
    print(f"\nToken type distribution:")
    command_tokens = sum(1 for t in token_counts if t.startswith('G') or t.startswith('M'))
    param_tokens = sum(1 for t in token_counts if t in 'XYZIJKFRSABCPQE' and len(t) == 1)
    num_tokens = sum(1 for t in token_counts if t.startswith('NUM_'))
    special_tokens = sum(1 for t in token_counts if t in config.special)
    other_tokens = len(token_counts) - command_tokens - param_tokens - num_tokens - special_tokens

    print(f"  Commands (G0, M30, etc.): {command_tokens}")
    print(f"  Parameters (X, Y, Z, etc.): {param_tokens}")
    print(f"  NUM tokens (NUM_X_52, etc.): {num_tokens}")
    print(f"  Special (PAD, UNK, etc.): {special_tokens}")
    print(f"  Other: {other_tokens}")

    # Show most common NUM tokens
    num_token_counts = [(t, c) for t, c in token_counts.items() if t.startswith('NUM_')]
    num_token_counts.sort(key=lambda x: -x[1])

    print(f"\nTop 20 most common NUM tokens:")
    for tok, count in num_token_counts[:20]:
        print(f"  {tok}: {count:,}")

    # Build vocabulary (keep top vocab_size tokens)
    vocab = dict(config.special)

    # Sort by frequency (descending) then alphabetically
    items = sorted(
        [(tok, cnt) for tok, cnt in token_counts.items()
         if cnt >= min_freq and tok not in vocab],
        key=lambda x: (-x[1], x[0])
    )

    for tok, _ in items[:max(0, vocab_size - len(vocab))]:
        vocab[tok] = len(vocab)

    print(f"\nFinal vocabulary size: {len(vocab)}")

    # Update tokenizer vocab
    tokenizer.vocab = vocab
    tokenizer.inv_vocab = {i: t for t, i in vocab.items()}

    # Save vocabulary
    tokenizer.save_vocab(output_path)
    print(f"\n✓ Vocabulary saved to: {output_path}")

    # Verify the vocab contains NUM tokens
    num_tokens_in_vocab = [t for t in vocab if t.startswith('NUM_')]
    print(f"✓ NUM tokens in vocabulary: {len(num_tokens_in_vocab)}")

    return vocab


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild vocabulary from actual CSV data files"
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing *_aligned.csv files')
    parser.add_argument('--output-vocab', type=str, required=True,
                        help='Output path for vocabulary JSON file')
    parser.add_argument('--mode', type=str, default='hybrid',
                        choices=['literal', 'split', 'hybrid'],
                        help='Tokenization mode (default: hybrid)')
    parser.add_argument('--bucket-digits', type=int, default=2,
                        help='Number of digits for bucketing (default: 2)')
    parser.add_argument('--vocab-size', type=int, default=5000,
                        help='Maximum vocabulary size (default: 5000)')
    parser.add_argument('--min-freq', type=int, default=1,
                        help='Minimum token frequency (default: 1)')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output_vocab)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Extract G-code from CSV files
    gcode_lines = extract_gcode_from_csvs(data_dir)

    if len(gcode_lines) == 0:
        raise ValueError(f"No G-code lines found in {data_dir}")

    # Build vocabulary
    vocab = build_vocabulary(
        gcode_lines,
        output_path,
        mode=args.mode,
        bucket_digits=args.bucket_digits,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq
    )

    print(f"\n{'='*80}")
    print(f"✓ VOCABULARY REBUILD COMPLETE")
    print(f"{'='*80}")
    print(f"Output: {output_path}")
    print(f"Size: {len(vocab)} tokens")
    print(f"\nNext step: Re-run preprocessing with this vocabulary")


if __name__ == '__main__':
    main()
