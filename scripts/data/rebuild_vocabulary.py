#!/usr/bin/env python3
"""
Rebuild vocabulary from ALL CSV data files.

This fixes the vocab/tokenization mismatch issue where the pre-built vocabulary
doesn't contain all NUM tokens that appear in the data.

Key changes from original:
1. Scans ALL *_aligned.csv files (not just train split)
2. Uses min_freq=1 to capture every token seen
3. Increased vocab_size to 15,000 (safety margin)
4. Maintains same precision settings as existing vocabulary
5. Adds comprehensive verification output

Usage:
    python scripts/rebuild_vocabulary.py \
        --data-dir data \
        --output-vocab data/vocabulary_4digit_full.json \
        --mode hybrid \
        --bucket-digits 4 \
        --vocab-size 15000 \
        --min-freq 1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
import pandas as pd
import argparse
from collections import Counter
from src.miracle.utilities.gcode_tokenizer import GCodeTokenizer, TokenizerConfig


def extract_gcode_from_csvs(data_dir: Path):
    """Extract all G-code lines from CSV files."""
    print(f"\nExtracting G-code from CSV files in {data_dir}...")

    csv_files = sorted(data_dir.glob('*_aligned.csv'))
    print(f"Found {len(csv_files)} CSV files")

    all_gcode_lines = []
    file_stats = {}

    for csv_file in csv_files:
        print(f"  Reading {csv_file.name}...", end=" ")
        df = pd.read_csv(csv_file)

        if 'gcode_string' in df.columns:
            gcode_lines = df['gcode_string'].dropna().tolist()
            all_gcode_lines.extend(gcode_lines)
            file_stats[csv_file.name] = len(gcode_lines)
            print(f"{len(gcode_lines)} lines")
        else:
            print("WARNING: No 'gcode_string' column found!")
            file_stats[csv_file.name] = 0

    print(f"\nTotal G-code lines extracted: {len(all_gcode_lines)}")
    return all_gcode_lines, file_stats


def build_vocabulary(
    gcode_lines: list,
    output_path: Path,
    mode: str = "hybrid",
    bucket_digits: int = 4,
    vocab_size: int = 15000,
    min_freq: int = 1
):
    """Build vocabulary from G-code lines with full coverage."""
    print(f"\n{'='*80}")
    print(f"BUILDING VOCABULARY")
    print(f"{'='*80}")
    print(f"Mode: {mode}")
    print(f"Bucket digits: {bucket_digits}")
    print(f"Max vocab size: {vocab_size}")
    print(f"Min frequency: {min_freq}")

    # Create tokenizer config with same precision as original vocabulary
    config = TokenizerConfig(
        mode=mode,
        bucket_digits=bucket_digits,
        vocab_size=vocab_size,
        min_freq=min_freq,
        dynamic_numbers=False
    )

    # Use same precision settings as the original 4-digit vocabulary
    config.precision = {
        'X': 0.001, 'Y': 0.001, 'Z': 0.001,
        'A': 0.001, 'B': 0.001, 'C': 0.001,
        'I': 0.0001, 'J': 0.0001, 'K': 0.0001,
        'F': 1.0, 'S': 10.0,
        'R': 0.0001, 'P': 0.001, 'Q': 0.001, 'E': 0.0001
    }

    print(f"Precision settings: {config.precision}")

    tokenizer = GCodeTokenizer(config)

    # Tokenize all lines and count frequencies
    print(f"\nTokenizing {len(gcode_lines)} G-code lines...")
    token_counts = Counter()

    for i, line in enumerate(gcode_lines):
        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i:,} lines...")

        try:
            # Canonicalize and tokenize
            canonical = tokenizer.canonicalize([line])
            tokens = tokenizer.tokenize_canonical(canonical)
            token_counts.update(tokens)
        except Exception as e:
            # Skip problematic lines
            pass

    print(f"  Processed {len(gcode_lines):,} lines")
    print(f"\nUnique tokens found: {len(token_counts)}")

    # Show token type distribution
    print(f"\nToken type distribution:")
    command_tokens = sum(1 for t in token_counts if (t.startswith('G') or t.startswith('M')) and len(t) > 1)
    param_tokens = sum(1 for t in token_counts if t in 'XYZIJKFRSABCPQE' and len(t) == 1)
    num_x_tokens = sum(1 for t in token_counts if t.startswith('NUM_X_'))
    num_y_tokens = sum(1 for t in token_counts if t.startswith('NUM_Y_'))
    num_z_tokens = sum(1 for t in token_counts if t.startswith('NUM_Z_'))
    num_r_tokens = sum(1 for t in token_counts if t.startswith('NUM_R_'))
    num_f_tokens = sum(1 for t in token_counts if t.startswith('NUM_F_'))
    num_other_tokens = sum(1 for t in token_counts if t.startswith('NUM_') and not any(
        t.startswith(f'NUM_{p}_') for p in ['X', 'Y', 'Z', 'R', 'F']))
    special_tokens = sum(1 for t in token_counts if t in config.special)
    other_tokens = len(token_counts) - command_tokens - param_tokens - \
        num_x_tokens - num_y_tokens - num_z_tokens - num_r_tokens - num_f_tokens - \
        num_other_tokens - special_tokens

    print(f"  Commands (G*, M*): {command_tokens}")
    print(f"  Parameters (X, Y, Z, etc.): {param_tokens}")
    print(f"  NUM_X tokens: {num_x_tokens}")
    print(f"  NUM_Y tokens: {num_y_tokens}")
    print(f"  NUM_Z tokens: {num_z_tokens}")
    print(f"  NUM_R tokens: {num_r_tokens}")
    print(f"  NUM_F tokens: {num_f_tokens}")
    print(f"  NUM_other tokens: {num_other_tokens}")
    print(f"  Special (PAD, UNK, etc.): {special_tokens}")
    print(f"  Other: {other_tokens}")

    # Show most common NUM tokens
    num_token_counts = [(t, c) for t, c in token_counts.items() if t.startswith('NUM_')]
    num_token_counts.sort(key=lambda x: -x[1])

    print(f"\nTop 20 most common NUM tokens:")
    for tok, count in num_token_counts[:20]:
        print(f"  {tok}: {count:,}")

    # Build vocabulary (keep all tokens meeting min_freq, up to vocab_size)
    vocab = dict(config.special)

    # Sort by frequency (descending) then alphabetically
    items = sorted(
        [(tok, cnt) for tok, cnt in token_counts.items()
         if cnt >= min_freq and tok not in vocab],
        key=lambda x: (-x[1], x[0])
    )

    # Add all tokens that meet frequency threshold (up to vocab_size)
    tokens_added = 0
    for tok, _ in items:
        if len(vocab) >= vocab_size:
            print(f"\nWARNING: Reached vocab_size limit of {vocab_size}!")
            print(f"  {len(items) - tokens_added} tokens excluded")
            break
        vocab[tok] = len(vocab)
        tokens_added += 1

    print(f"\nFinal vocabulary size: {len(vocab)}")

    # Update tokenizer vocab
    tokenizer.vocab = vocab
    tokenizer.inv_vocab = {i: t for t, i in vocab.items()}

    # Save vocabulary with full config
    vocab_data = {
        'config': {
            'mode': mode,
            'bucket_digits': bucket_digits,
            'precision': config.precision,
            'clip_bins': {},
            'canonical_decimal_places': 4,
            'ignore_for_pid': ['F', 'S'],
            'special': config.special,
            'vocab_size': vocab_size,
            'min_freq': min_freq,
            'dynamic_numbers': False
        },
        'vocab': vocab
    }

    with open(output_path, 'w') as f:
        json.dump(vocab_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"VOCABULARY BUILD COMPLETE")
    print(f"{'='*80}")
    print(f"Output: {output_path}")
    print(f"Total tokens: {len(vocab)}")

    # Verify the vocab contains all token types
    num_tokens_in_vocab = [t for t in vocab if t.startswith('NUM_')]
    print(f"\nVocabulary Statistics:")
    print(f"  NUM_X tokens: {sum(1 for t in vocab if t.startswith('NUM_X_'))}")
    print(f"  NUM_Y tokens: {sum(1 for t in vocab if t.startswith('NUM_Y_'))}")
    print(f"  NUM_Z tokens: {sum(1 for t in vocab if t.startswith('NUM_Z_'))}")
    print(f"  NUM_R tokens: {sum(1 for t in vocab if t.startswith('NUM_R_'))}")
    print(f"  NUM_F tokens: {sum(1 for t in vocab if t.startswith('NUM_F_'))}")
    print(f"  Total NUM tokens: {len(num_tokens_in_vocab)}")

    return vocab, vocab_data


def verify_vocabulary(vocab: dict, gcode_lines: list, config: dict):
    """Verify that vocabulary covers all tokens in data."""
    print(f"\n{'='*80}")
    print(f"VERIFYING VOCABULARY COVERAGE")
    print(f"{'='*80}")

    # Create tokenizer with same config
    tokenizer_config = TokenizerConfig(
        mode=config.get('mode', 'hybrid'),
        bucket_digits=config.get('bucket_digits', 4),
        vocab_size=config.get('vocab_size', 15000),
        min_freq=1,
        dynamic_numbers=False
    )
    tokenizer_config.precision = config.get('precision', {})

    tokenizer = GCodeTokenizer(tokenizer_config)

    oov_tokens = set()
    total_tokens = 0
    oov_occurrences = 0

    for line in gcode_lines:
        try:
            canonical = tokenizer.canonicalize([line])
            tokens = tokenizer.tokenize_canonical(canonical)
            for token in tokens:
                total_tokens += 1
                if token not in vocab:
                    oov_tokens.add(token)
                    oov_occurrences += 1
        except Exception:
            pass

    if len(oov_tokens) == 0:
        print(f"\n✓ VERIFICATION PASSED: Zero OOV tokens!")
        print(f"  Total tokens checked: {total_tokens:,}")
        print(f"  All tokens are in vocabulary")
        return True
    else:
        print(f"\n✗ VERIFICATION FAILED: {len(oov_tokens)} OOV tokens found!")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  OOV occurrences: {oov_occurrences:,}")
        print(f"  OOV tokens: {list(oov_tokens)[:20]}...")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild vocabulary from ALL CSV data files"
    )
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing *_aligned.csv files')
    parser.add_argument('--output-vocab', type=str,
                        default='data/vocabulary_4digit_full.json',
                        help='Output path for vocabulary JSON file')
    parser.add_argument('--mode', type=str, default='hybrid',
                        choices=['literal', 'split', 'hybrid'],
                        help='Tokenization mode (default: hybrid)')
    parser.add_argument('--bucket-digits', type=int, default=4,
                        help='Number of digits for bucketing (default: 4)')
    parser.add_argument('--vocab-size', type=int, default=15000,
                        help='Maximum vocabulary size (default: 15000)')
    parser.add_argument('--min-freq', type=int, default=1,
                        help='Minimum token frequency (default: 1)')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify vocabulary coverage after building')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output_vocab)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Extract G-code from all CSV files
    gcode_lines, file_stats = extract_gcode_from_csvs(data_dir)

    if len(gcode_lines) == 0:
        raise ValueError(f"No G-code lines found in {data_dir}")

    # Build vocabulary
    vocab, vocab_data = build_vocabulary(
        gcode_lines,
        output_path,
        mode=args.mode,
        bucket_digits=args.bucket_digits,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq
    )

    # Verify coverage
    if args.verify:
        success = verify_vocabulary(vocab, gcode_lines, vocab_data['config'])
        if not success:
            print("\nWARNING: Vocabulary does not cover all tokens!")
            print("Consider increasing --vocab-size or checking data")

    print(f"\n{'='*80}")
    print(f"NEXT STEP: Reprocess data with new vocabulary")
    print(f"{'='*80}")
    print(f"""
python -m miracle.dataset.preprocessing \\
    --data-dir data \\
    --vocab-path {output_path} \\
    --output-dir outputs/processed_v3 \\
    --window-size 64 \\
    --stride 16 \\
    --train-ratio 0.7 \\
    --val-ratio 0.15 \\
    --test-ratio 0.15 \\
    --random-seed 42
""")


if __name__ == '__main__':
    main()
