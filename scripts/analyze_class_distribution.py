"""
Analyze command class distribution across train/val/test splits.

This script investigates why G1/G2/G3/F have 0% recall by examining:
1. Command class distribution in each split
2. Whether validation set contains rare classes
3. Class imbalance ratios
"""

import numpy as np
import json
from pathlib import Path
from collections import Counter
import sys

def load_vocab(vocab_path: str) -> dict:
    """Load vocabulary and return token2id mapping."""
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    return vocab_data.get('vocab', vocab_data.get('token2id', {}))

def analyze_split(npz_path: str, vocab: dict) -> dict:
    """Analyze command distribution in a single split."""
    data = np.load(npz_path, allow_pickle=True)

    tokens = data['tokens']  # Shape: (N, seq_len)
    gcode_texts = data['gcode_texts']  # Shape: (N,)

    # Reverse vocab mapping
    id2token = {v: k for k, v in vocab.items()}

    # Count commands from gcode_texts (more reliable)
    command_counts = Counter()
    param_counts = Counter()

    for text in gcode_texts:
        text = str(text).strip()
        parts = text.split()
        for part in parts:
            if part.startswith('G') and len(part) > 1:
                try:
                    int(part[1:])  # Verify it's a valid G-code
                    command_counts[part] += 1
                except ValueError:
                    pass
            elif part.startswith('M') and len(part) > 1:
                try:
                    int(part[1:])
                    command_counts[part] += 1
                except ValueError:
                    pass
            elif len(part) >= 2 and part[0] in 'XYZFRS':
                param_counts[part[0]] += 1

    # Count from token IDs for verification
    token_command_counts = Counter()
    for seq in tokens:
        for tid in seq:
            token = id2token.get(tid, '<UNK>')
            if token.startswith('G') and len(token) > 1 and token[1:].isdigit():
                token_command_counts[token] += 1
            elif token.startswith('M') and len(token) > 1 and token[1:].isdigit():
                token_command_counts[token] += 1

    return {
        'n_samples': len(tokens),
        'command_counts': dict(command_counts),
        'param_counts': dict(param_counts),
        'token_command_counts': dict(token_command_counts),
        'unique_commands': set(command_counts.keys()),
        'gcode_samples': list(gcode_texts[:10])  # First 10 samples
    }

def main():
    # Paths
    data_dir = Path('outputs/processed_hybrid')
    vocab_path = Path('data/vocabulary_1digit_hybrid.json')

    if not vocab_path.exists():
        print(f"Vocabulary not found: {vocab_path}")
        sys.exit(1)

    vocab = load_vocab(str(vocab_path))
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Command tokens: {[k for k in vocab if k.startswith('G') or k.startswith('M')]}")
    print()

    # Analyze each split
    splits = ['train', 'val', 'test']
    results = {}

    for split in splits:
        npz_path = data_dir / f'{split}_sequences.npz'
        if not npz_path.exists():
            print(f"Split not found: {npz_path}")
            continue

        print(f"=" * 60)
        print(f"Analyzing {split.upper()} split: {npz_path}")
        print(f"=" * 60)

        result = analyze_split(str(npz_path), vocab)
        results[split] = result

        print(f"\nSamples: {result['n_samples']}")

        print(f"\nCommand distribution (from gcode_texts):")
        for cmd, count in sorted(result['command_counts'].items(), key=lambda x: -x[1]):
            pct = 100 * count / sum(result['command_counts'].values()) if result['command_counts'] else 0
            print(f"  {cmd:6s}: {count:6d} ({pct:5.2f}%)")

        print(f"\nParameter distribution:")
        for param, count in sorted(result['param_counts'].items(), key=lambda x: -x[1]):
            print(f"  {param}: {count:6d}")

        print(f"\nSample G-code strings:")
        for i, text in enumerate(result['gcode_samples']):
            print(f"  {i+1}. {text}")
        print()

    # Cross-split comparison
    if len(results) == 3:
        print("=" * 60)
        print("CROSS-SPLIT COMPARISON")
        print("=" * 60)

        # Get all commands
        all_commands = set()
        for r in results.values():
            all_commands.update(r['unique_commands'])

        print(f"\n{'Command':<8} {'Train':>10} {'Val':>10} {'Test':>10} {'In Val?':>10}")
        print("-" * 50)

        for cmd in sorted(all_commands):
            train_count = results.get('train', {}).get('command_counts', {}).get(cmd, 0)
            val_count = results.get('val', {}).get('command_counts', {}).get(cmd, 0)
            test_count = results.get('test', {}).get('command_counts', {}).get(cmd, 0)
            in_val = "YES" if val_count > 0 else "NO !!!"

            print(f"{cmd:<8} {train_count:>10} {val_count:>10} {test_count:>10} {in_val:>10}")

        # Class imbalance analysis
        print("\n" + "=" * 60)
        print("CLASS IMBALANCE ANALYSIS (Training Set)")
        print("=" * 60)

        train_cmds = results.get('train', {}).get('command_counts', {})
        if train_cmds:
            total = sum(train_cmds.values())
            max_count = max(train_cmds.values())
            min_count = min(train_cmds.values())

            print(f"\nTotal commands: {total}")
            print(f"Max class count: {max_count}")
            print(f"Min class count: {min_count}")
            print(f"Imbalance ratio: {max_count / min_count:.1f}x")

            print("\nClass weights needed for balancing:")
            for cmd, count in sorted(train_cmds.items(), key=lambda x: -x[1]):
                weight = max_count / count
                print(f"  {cmd}: {weight:.2f}x (count: {count})")

        # Critical finding
        print("\n" + "=" * 60)
        print("CRITICAL FINDINGS")
        print("=" * 60)

        val_cmds = results.get('val', {}).get('command_counts', {})
        missing_in_val = []
        for cmd in ['G1', 'G2', 'G3']:
            if val_cmds.get(cmd, 0) == 0:
                missing_in_val.append(cmd)

        if missing_in_val:
            print(f"\n!!! PROBLEM: Commands {missing_in_val} are MISSING from validation set!")
            print("    This explains 0% recall - model cannot be evaluated on these classes")
            print("    Solution: Re-split data with stratification by command type")
        else:
            print("\nAll commands present in validation set.")
            print("Issue may be class imbalance or model capacity.")

if __name__ == '__main__':
    main()
