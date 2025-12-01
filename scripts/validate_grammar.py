#!/usr/bin/env python3
"""
Validate G-code generation quality by checking grammar violations.

Usage:
    python scripts/validate_grammar.py \
        --checkpoint outputs/hybrid_1digit/checkpoint_best.pt \
        --vocab data/vocabulary_1digit_hybrid.json \
        --data-dir outputs/processed_hybrid \
        --n-samples 100
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.training.grammar_constraints import GCodeGrammarConstraints
from miracle.utilities.gcode_tokenizer import GCodeTokenizer


def load_model_and_vocab(checkpoint_path, vocab_path):
    """Load model checkpoint and vocabulary."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"Loading vocabulary from {vocab_path}")
    tokenizer = GCodeTokenizer.load(vocab_path)
    vocab = tokenizer.vocab

    # Create inverse vocab
    vocab_inv = {v: k for k, v in vocab.items()}

    return checkpoint, vocab, vocab_inv, tokenizer


def load_generated_sequences(data_dir, n_samples=100):
    """Load test sequences from processed data."""
    test_file = Path(data_dir) / 'test_sequences.npz'

    if not test_file.exists():
        print(f"ERROR: Test file not found: {test_file}")
        return None

    print(f"Loading test sequences from {test_file}")
    data = np.load(test_file, allow_pickle=True)

    tokens = data['tokens'][:n_samples]
    gcode_texts = data['gcode_texts'][:n_samples]

    print(f"Loaded {len(tokens)} sequences")
    return tokens, gcode_texts


def validate_sequences(sequences, vocab, vocab_inv):
    """
    Validate generated sequences and count grammar violations.

    Args:
        sequences: np.ndarray of token IDs [n_samples, seq_len]
        vocab: Vocabulary dict
        vocab_inv: Inverse vocabulary dict

    Returns:
        Validation report dict
    """
    print("\n" + "="*60)
    print("G-CODE GRAMMAR VALIDATION REPORT")
    print("="*60)

    grammar = GCodeGrammarConstraints(vocab, device='cpu')

    total_violations = {
        'arc_without_radius': 0,
        'rapid_with_feed': 0,
        'invalid_ordering': 0,
        'total': 0,
    }

    sequence_violations = []

    for i, seq in enumerate(sequences):
        # Convert to tensor
        seq_tensor = torch.from_numpy(seq).long()

        # Validate
        violations = grammar.validate_sequence(seq_tensor, vocab_inv)

        # Accumulate
        for key in total_violations:
            total_violations[key] += violations[key]

        sequence_violations.append(violations)

        # Print examples of violating sequences
        if violations['total'] > 0 and i < 5:
            print(f"\nExample violation #{i+1}:")
            print(f"  Sequence: {_format_sequence(seq, vocab_inv, max_tokens=20)}")
            print(f"  Violations: {violations}")

    # Summary statistics
    n_sequences = len(sequences)
    n_violating = sum(1 for v in sequence_violations if v['total'] > 0)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total sequences validated: {n_sequences}")
    print(f"Sequences with violations: {n_violating} ({100*n_violating/n_sequences:.1f}%)")
    print(f"Sequences without violations: {n_sequences - n_violating} ({100*(n_sequences - n_violating)/n_sequences:.1f}%)")

    print("\nViolation Breakdown:")
    print(f"  Arc without radius (G2/G3 missing R):  {total_violations['arc_without_radius']}")
    print(f"  Rapid with feed rate (G0 with F):      {total_violations['rapid_with_feed']}")
    print(f"  Invalid token ordering:                 {total_violations['invalid_ordering']}")
    print(f"  Total violations:                       {total_violations['total']}")

    if n_sequences > 0:
        print(f"\nAverage violations per sequence:        {total_violations['total']/n_sequences:.2f}")

    # Quality score (percentage of sequences without violations)
    quality_score = 100 * (n_sequences - n_violating) / n_sequences
    print(f"\nGrammar Quality Score:                  {quality_score:.1f}%")

    if quality_score >= 90:
        print("  ✅ EXCELLENT - Model produces valid G-code")
    elif quality_score >= 70:
        print("  ⚠️  GOOD - Some violations but mostly valid")
    elif quality_score >= 50:
        print("  ⚠️  FAIR - Many violations, needs improvement")
    else:
        print("  ❌ POOR - Most sequences have grammar errors")

    return {
        'total_sequences': n_sequences,
        'violating_sequences': n_violating,
        'quality_score': quality_score,
        'violations': total_violations,
    }


def _format_sequence(seq, vocab_inv, max_tokens=20):
    """Format sequence for display."""
    tokens = []
    for i, token_id in enumerate(seq):
        if i >= max_tokens:
            tokens.append('...')
            break
        token = vocab_inv.get(token_id, f'<UNK:{token_id}>')
        if token == 'PAD':
            break
        tokens.append(token)
    return ' '.join(tokens)


def analyze_value_diversity(sequences, vocab_inv):
    """
    Analyze whether model predicts diverse numeric values or just repeats.

    Args:
        sequences: np.ndarray of token IDs
        vocab_inv: Inverse vocabulary dict
    """
    print("\n" + "="*60)
    print("VALUE DIVERSITY ANALYSIS")
    print("="*60)

    # Count unique numeric tokens
    numeric_tokens = []
    for seq in sequences:
        for token_id in seq:
            token = vocab_inv.get(token_id, '')
            if token.startswith('NUM_'):
                numeric_tokens.append(token)

    if not numeric_tokens:
        print("No numeric tokens found in sequences")
        return

    unique_numeric = set(numeric_tokens)
    total_numeric = len(numeric_tokens)

    print(f"Total numeric tokens:  {total_numeric}")
    print(f"Unique numeric tokens: {len(unique_numeric)}")
    print(f"Diversity ratio:       {len(unique_numeric)/total_numeric:.2%}")

    # Top 10 most common
    from collections import Counter
    counts = Counter(numeric_tokens)

    print("\nTop 10 most common numeric tokens:")
    for token, count in counts.most_common(10):
        print(f"  {token}: {count} ({100*count/total_numeric:.1f}%)")

    # Warning if model collapsed to single value
    if len(unique_numeric) < 5:
        print("\n❌ WARNING: Very low diversity - model may have collapsed!")
    elif counts.most_common(1)[0][1] / total_numeric > 0.5:
        print(f"\n⚠️  WARNING: Single token '{counts.most_common(1)[0][0]}' represents " +
              f"{100*counts.most_common(1)[0][1]/total_numeric:.1f}% of predictions")
    else:
        print("\n✅ Good diversity in numeric predictions")


def main():
    parser = argparse.ArgumentParser(description='Validate G-code grammar in generated sequences')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=Path, required=True,
                        help='Path to vocabulary JSON')
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directory with processed test data')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of sequences to validate')

    args = parser.parse_args()

    # Load model and vocab
    checkpoint, vocab, vocab_inv, tokenizer = load_model_and_vocab(args.checkpoint, args.vocab)

    # Load test sequences (ground truth from dataset)
    sequences, gcode_texts = load_generated_sequences(args.data_dir, args.n_samples)

    if sequences is None:
        return 1

    # Validate grammar
    report = validate_sequences(sequences, vocab, vocab_inv)

    # Analyze diversity
    analyze_value_diversity(sequences, vocab_inv)

    # Print recommendation
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if report['quality_score'] < 90:
        print("To improve grammar quality:")
        print("1. Add grammar constraint losses during training")
        print("2. Use inference-time constraint masking")
        print("3. Implement post-processing to fix violations")
        print("\nSee: src/miracle/training/grammar_constraints.py")

    if report['quality_score'] >= 90:
        print("Grammar quality is good!")
        print("Next steps:")
        print("1. Test model generation (not just validation on ground truth)")
        print("2. Evaluate printer identification accuracy")
        print("3. Analyze attention patterns for interpretability")

    print("\n✅ Validation complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
