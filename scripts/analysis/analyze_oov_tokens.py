#!/usr/bin/env python3
"""
Analyze Out-of-Vocabulary (OOV) tokens in the G-code dataset.

This script:
1. Loads all CSV files from the data directory
2. Tokenizes ALL G-code lines using the same tokenizer config as vocabulary
3. Compares tokens against current vocabulary
4. Reports OOV statistics by type (NUM_X, NUM_Y, etc.)

Usage:
    python scripts/analyze_oov_tokens.py \
        --data-dir data \
        --vocab-path data/vocabulary_4digit_hybrid.json \
        --output reports/oov_analysis.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
import argparse
from collections import Counter, defaultdict
import pandas as pd

from miracle.utilities.gcode_tokenizer import GCodeTokenizer, TokenizerConfig


def load_vocabulary(vocab_path: Path) -> tuple:
    """Load vocabulary and config from JSON file."""
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    vocab = vocab_data.get('vocab', vocab_data.get('token2id', {}))
    config = vocab_data.get('config', {})

    return vocab, config


def extract_gcode_from_csvs(data_dir: Path) -> tuple:
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


def categorize_token(token: str) -> str:
    """Categorize a token by type."""
    if token.startswith('NUM_X_'):
        return 'NUM_X'
    elif token.startswith('NUM_Y_'):
        return 'NUM_Y'
    elif token.startswith('NUM_Z_'):
        return 'NUM_Z'
    elif token.startswith('NUM_R_'):
        return 'NUM_R'
    elif token.startswith('NUM_I_'):
        return 'NUM_I'
    elif token.startswith('NUM_J_'):
        return 'NUM_J'
    elif token.startswith('NUM_K_'):
        return 'NUM_K'
    elif token.startswith('NUM_F_'):
        return 'NUM_F'
    elif token.startswith('NUM_S_'):
        return 'NUM_S'
    elif token.startswith('NUM_'):
        return 'NUM_OTHER'
    elif token.startswith('G') and len(token) > 1 and token[1:].replace('.', '').isdigit():
        return 'G_COMMAND'
    elif token.startswith('M') and len(token) > 1 and token[1:].isdigit():
        return 'M_COMMAND'
    elif token in ['X', 'Y', 'Z', 'I', 'J', 'K', 'R', 'F', 'S', 'A', 'B', 'C', 'P', 'Q', 'E']:
        return 'PARAM_LETTER'
    elif token in ['PAD', 'BOS', 'EOS', 'UNK', 'MASK']:
        return 'SPECIAL'
    else:
        return 'OTHER'


def analyze_oov(
    gcode_lines: list,
    vocab: dict,
    config: dict
) -> dict:
    """Analyze OOV tokens in G-code lines."""
    print(f"\n{'='*80}")
    print("ANALYZING OOV TOKENS")
    print(f"{'='*80}")

    # Create tokenizer with same config as vocabulary
    tokenizer_config = TokenizerConfig(
        mode=config.get('mode', 'hybrid'),
        bucket_digits=config.get('bucket_digits', 4),
        vocab_size=config.get('vocab_size', 10000),
        min_freq=1,
        dynamic_numbers=False
    )

    # Apply precision if available
    if 'precision' in config:
        tokenizer_config.precision = config['precision']

    tokenizer = GCodeTokenizer(tokenizer_config)

    print(f"Tokenizer config:")
    print(f"  Mode: {tokenizer_config.mode}")
    print(f"  Bucket digits: {tokenizer_config.bucket_digits}")
    print(f"  Precision: {tokenizer_config.precision}")

    # Tokenize all lines and track OOV
    all_tokens = Counter()
    oov_tokens = Counter()
    oov_by_category = defaultdict(Counter)
    in_vocab_by_category = defaultdict(Counter)

    print(f"\nTokenizing {len(gcode_lines)} G-code lines...")

    for i, line in enumerate(gcode_lines):
        if i % 50000 == 0 and i > 0:
            print(f"  Processed {i:,} lines...")

        try:
            # Canonicalize and tokenize
            canonical = tokenizer.canonicalize([line])
            tokens = tokenizer.tokenize_canonical(canonical)

            for token in tokens:
                all_tokens[token] += 1
                category = categorize_token(token)

                if token not in vocab:
                    oov_tokens[token] += 1
                    oov_by_category[category][token] += 1
                else:
                    in_vocab_by_category[category][token] += 1

        except Exception as e:
            # Skip problematic lines
            pass

    print(f"  Processed {len(gcode_lines):,} lines total")

    # Compute statistics
    total_unique = len(all_tokens)
    total_oov = len(oov_tokens)
    vocab_size = len(vocab)
    oov_percentage = (total_oov / total_unique * 100) if total_unique > 0 else 0

    # Count total token occurrences
    total_occurrences = sum(all_tokens.values())
    oov_occurrences = sum(oov_tokens.values())
    oov_occurrence_pct = (oov_occurrences / total_occurrences * 100) if total_occurrences > 0 else 0

    print(f"\n{'='*80}")
    print("OOV ANALYSIS REPORT")
    print(f"{'='*80}")
    print(f"\nToken Statistics:")
    print(f"  Total G-code lines: {len(gcode_lines):,}")
    print(f"  Total token occurrences: {total_occurrences:,}")
    print(f"  Unique tokens in data: {total_unique:,}")
    print(f"  Tokens in vocabulary: {vocab_size:,}")
    print(f"  OOV unique tokens: {total_oov:,} ({oov_percentage:.1f}% of unique)")
    print(f"  OOV token occurrences: {oov_occurrences:,} ({oov_occurrence_pct:.1f}% of total)")

    print(f"\nOOV by Category:")
    for category in sorted(oov_by_category.keys()):
        tokens = oov_by_category[category]
        occurrences = sum(tokens.values())
        print(f"  {category}: {len(tokens)} unique tokens, {occurrences:,} occurrences")

    print(f"\nIn-Vocabulary by Category:")
    for category in sorted(in_vocab_by_category.keys()):
        tokens = in_vocab_by_category[category]
        occurrences = sum(tokens.values())
        print(f"  {category}: {len(tokens)} unique tokens, {occurrences:,} occurrences")

    # Top OOV tokens
    print(f"\nTop 30 OOV tokens (by frequency):")
    for token, count in oov_tokens.most_common(30):
        category = categorize_token(token)
        print(f"  {token}: {count:,} ({category})")

    # Build report
    report = {
        'summary': {
            'total_gcode_lines': len(gcode_lines),
            'total_token_occurrences': total_occurrences,
            'unique_tokens_in_data': total_unique,
            'vocab_size': vocab_size,
            'oov_unique_tokens': total_oov,
            'oov_unique_percentage': oov_percentage,
            'oov_occurrences': oov_occurrences,
            'oov_occurrence_percentage': oov_occurrence_pct,
        },
        'oov_by_category': {
            cat: {
                'unique_count': len(tokens),
                'total_occurrences': sum(tokens.values()),
                'top_tokens': dict(tokens.most_common(20))
            }
            for cat, tokens in oov_by_category.items()
        },
        'in_vocab_by_category': {
            cat: {
                'unique_count': len(tokens),
                'total_occurrences': sum(tokens.values())
            }
            for cat, tokens in in_vocab_by_category.items()
        },
        'all_oov_tokens': dict(oov_tokens.most_common()),
        'vocab_config': config
    }

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Analyze OOV tokens in G-code dataset"
    )
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing *_aligned.csv files')
    parser.add_argument('--vocab-path', type=str,
                        default='data/vocabulary_4digit_hybrid.json',
                        help='Path to vocabulary JSON file')
    parser.add_argument('--output', type=str, default='reports/oov_analysis.json',
                        help='Output path for analysis JSON')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    vocab_path = Path(args.vocab_path)
    output_path = Path(args.output)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}...")
    vocab, config = load_vocabulary(vocab_path)
    print(f"  Vocabulary size: {len(vocab)} tokens")
    print(f"  Config: {config}")

    # Extract G-code from CSV files
    gcode_lines, file_stats = extract_gcode_from_csvs(data_dir)

    if len(gcode_lines) == 0:
        raise ValueError(f"No G-code lines found in {data_dir}")

    # Analyze OOV
    report = analyze_oov(gcode_lines, vocab, config)
    report['file_stats'] = file_stats

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*80}")
    print(f"OOV ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Report saved to: {output_path}")

    # Print recommendation
    oov_pct = report['summary']['oov_unique_percentage']
    if oov_pct > 0:
        print(f"\n⚠️  WARNING: {oov_pct:.1f}% of tokens are OOV!")
        print("   Recommendation: Rebuild vocabulary from ALL data with min_freq=1")
    else:
        print("\n✓ All tokens are in vocabulary!")


if __name__ == '__main__':
    main()
