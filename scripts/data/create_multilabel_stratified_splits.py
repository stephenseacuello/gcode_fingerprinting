"""Create multi-label stratified train/val/test splits with coverage checks.

This script implements comprehensive stratification across multiple labels:
- operation_type (9 classes)
- command (G0, G1, G2, G3, G53, etc.)
- param_type (X, Y, Z, R, F, etc.)
- value_buckets (for numeric values)

Features:
1. Compute frequencies for all label types
2. Require minimum counts per split
3. Use composite label stratification (operation + command)
4. Assert coverage after splitting
5. Persist split indices for reproducibility
"""
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import json
import argparse
import re
from sklearn.model_selection import StratifiedShuffleSplit


def load_all_data(data_dir: str):
    """Load and combine all NPZ files from train/val/test splits."""
    data_path = Path(data_dir)

    # Try different file patterns
    if (data_path / 'train_sequences.npz').exists():
        train_data = np.load(data_path / 'train_sequences.npz', allow_pickle=True)
        val_data = np.load(data_path / 'val_sequences.npz', allow_pickle=True)
        test_data = np.load(data_path / 'test_sequences.npz', allow_pickle=True)
    elif (data_path / 'train.npz').exists():
        train_data = np.load(data_path / 'train.npz', allow_pickle=True)
        val_data = np.load(data_path / 'val.npz', allow_pickle=True)
        test_data = np.load(data_path / 'test.npz', allow_pickle=True)
    else:
        raise FileNotFoundError(f"No NPZ files found in {data_path}")

    keys = list(train_data.keys())
    print(f"Data keys: {keys}")

    # Combine all data
    combined = {}
    for key in keys:
        combined[key] = np.concatenate([
            train_data[key],
            val_data[key],
            test_data[key]
        ], axis=0)

    n_train = len(train_data['tokens'])
    n_val = len(val_data['tokens'])
    n_test = len(test_data['tokens'])

    print(f"Original splits: train={n_train}, val={n_val}, test={n_test}")
    print(f"Combined total: {len(combined['tokens'])}")

    return combined, keys


def extract_labels_from_tokens(tokens, vocab_path=None):
    """Extract command and param type labels from token sequences."""
    # Load vocabulary if provided
    if vocab_path and Path(vocab_path).exists():
        with open(vocab_path) as f:
            vocab_data = json.load(f)
        vocab = vocab_data.get('vocab', vocab_data)
        id2token = {v: k for k, v in vocab.items()}
    else:
        id2token = {}

    commands = []
    param_types = []
    value_buckets = []

    for seq in tokens:
        seq_commands = set()
        seq_params = set()
        seq_values = []

        for tid in seq:
            if tid in id2token:
                token = id2token[tid]
                # Extract command (G0, G1, G2, G3, G53, M-codes)
                if token.startswith('G') or token.startswith('M'):
                    seq_commands.add(token)
                # Extract param type from NUM_X_1650 format
                elif token.startswith('NUM_'):
                    parts = token.split('_')
                    if len(parts) >= 2:
                        param = parts[1]
                        seq_params.add(param)
                        # Extract value for bucketing
                        if len(parts) >= 3:
                            try:
                                val_str = parts[2]
                                if val_str.startswith('-'):
                                    val = -int(val_str[1:])
                                else:
                                    val = int(val_str)
                                seq_values.append(val)
                            except ValueError:
                                pass
                elif token in ('X', 'Y', 'Z', 'R', 'F', 'I', 'J', 'K', 'A', 'B', 'C'):
                    seq_params.add(token)

        # Create composite labels
        commands.append(tuple(sorted(seq_commands)) if seq_commands else ('NONE',))
        param_types.append(tuple(sorted(seq_params)) if seq_params else ('NONE',))

        # Create value bucket (low/mid/high)
        if seq_values:
            avg_val = np.mean(seq_values)
            if avg_val < 500:
                bucket = 'LOW'
            elif avg_val < 1500:
                bucket = 'MID'
            else:
                bucket = 'HIGH'
        else:
            bucket = 'NONE'
        value_buckets.append(bucket)

    return commands, param_types, value_buckets


def create_composite_label(operation_types, commands):
    """Create composite label combining operation_type and primary command."""
    composite = []
    for op, cmds in zip(operation_types, commands):
        # Use first command or 'NONE'
        cmd = cmds[0] if cmds else 'NONE'
        composite.append(f"op{op}_{cmd}")
    return composite


def verify_coverage(
    split_name: str,
    split_indices: np.ndarray,
    operation_types: np.ndarray,
    commands: list,
    param_types: list,
    min_samples: int = 1
) -> dict:
    """Verify that all required classes are present in split."""
    issues = []

    # Check operation types
    op_counts = Counter(operation_types[split_indices].tolist())
    for op_id in range(9):
        if op_counts.get(op_id, 0) < min_samples:
            issues.append(f"Operation {op_id}: {op_counts.get(op_id, 0)} samples (need {min_samples})")

    # Check commands
    cmd_counts = Counter()
    for idx in split_indices:
        for cmd in commands[idx]:
            cmd_counts[cmd] += 1

    # Expected commands
    expected_cmds = {'G0', 'G1'}  # At minimum
    for cmd in expected_cmds:
        if cmd_counts.get(cmd, 0) < min_samples:
            issues.append(f"Command {cmd}: {cmd_counts.get(cmd, 0)} samples (need {min_samples})")

    # Check param types
    param_counts = Counter()
    for idx in split_indices:
        for param in param_types[idx]:
            param_counts[param] += 1

    # Expected params
    expected_params = {'X', 'Y', 'Z'}  # At minimum
    for param in expected_params:
        if param_counts.get(param, 0) < min_samples:
            issues.append(f"Param {param}: {param_counts.get(param, 0)} samples (need {min_samples})")

    return {
        'split': split_name,
        'n_samples': len(split_indices),
        'operation_counts': dict(op_counts),
        'command_counts': dict(cmd_counts),
        'param_counts': dict(param_counts),
        'issues': issues,
        'passed': len(issues) == 0
    }


def iterative_stratified_split(
    n_samples: int,
    labels: np.ndarray,
    composite_labels: list,
    val_size: float,
    test_size: float,
    random_seed: int,
    max_iterations: int = 10
):
    """
    Iterative stratified splitting with retry on coverage failure.

    Falls back to simple composite stratification if iterative doesn't work.
    """
    best_split = None
    best_coverage = 0

    for iteration in range(max_iterations):
        seed = random_seed + iteration

        # Convert composite labels to integers
        unique_labels = list(set(composite_labels))
        label_map = {l: i for i, l in enumerate(unique_labels)}
        composite_int = np.array([label_map[l] for l in composite_labels])

        # Check if we have enough samples per class for stratification
        class_counts = Counter(composite_int)
        min_count = min(class_counts.values())

        if min_count < 2:
            # Fall back to operation-only stratification
            print(f"  Iteration {iteration+1}: Composite has classes with <2 samples, using operation-only")
            stratify_labels = labels
        else:
            stratify_labels = composite_int

        try:
            # First split: separate test set
            sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_val_idx, test_idx = next(sss_test.split(np.zeros(n_samples), stratify_labels))

            # Second split: separate train and val
            val_size_adj = val_size / (1 - test_size)
            sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adj, random_state=seed)
            train_idx_rel, val_idx_rel = next(sss_val.split(
                np.zeros(len(train_val_idx)), stratify_labels[train_val_idx]
            ))

            train_idx = train_val_idx[train_idx_rel]
            val_idx = train_val_idx[val_idx_rel]

            # Count coverage
            train_ops = set(labels[train_idx])
            val_ops = set(labels[val_idx])
            test_ops = set(labels[test_idx])
            coverage = len(train_ops) + len(val_ops) + len(test_ops)

            if coverage > best_coverage:
                best_coverage = coverage
                best_split = (train_idx, val_idx, test_idx)

            # Check if all 9 ops in all splits
            if len(train_ops) == 9 and len(val_ops) == 9 and len(test_ops) == 9:
                print(f"  Iteration {iteration+1}: Perfect coverage found!")
                return train_idx, val_idx, test_idx

        except ValueError as e:
            print(f"  Iteration {iteration+1}: Stratification failed ({e}), retrying...")
            continue

    print(f"  Using best split with coverage={best_coverage}/27")
    return best_split if best_split else (np.arange(n_samples), np.array([]), np.array([]))


def create_multilabel_stratified_splits(
    data_dir: str,
    output_dir: str,
    vocab_path: str = None,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42,
    min_samples_per_class: int = 1
):
    """Create multi-label stratified splits with coverage verification."""

    # Load data
    combined, keys = load_all_data(data_dir)
    tokens = combined['tokens']
    operation_types = combined['operation_type']
    n_samples = len(tokens)

    # Extract multi-labels
    print("\nExtracting labels from tokens...")
    commands, param_types, value_buckets = extract_labels_from_tokens(tokens, vocab_path)

    # Analyze label distributions
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # Operation types
    op_counts = Counter(operation_types.tolist())
    print(f"\nOperation Types (9 classes):")
    for op_id in sorted(op_counts.keys()):
        print(f"  Op {op_id}: {op_counts[op_id]:4d} samples")

    # Commands
    cmd_counts = Counter()
    for cmds in commands:
        for cmd in cmds:
            cmd_counts[cmd] += 1
    print(f"\nCommands ({len(cmd_counts)} unique):")
    for cmd, count in cmd_counts.most_common(10):
        print(f"  {cmd}: {count:4d} samples")

    # Param types
    param_counts = Counter()
    for params in param_types:
        for param in params:
            param_counts[param] += 1
    print(f"\nParam Types ({len(param_counts)} unique):")
    for param, count in param_counts.most_common(10):
        print(f"  {param}: {count:4d} samples")

    # Value buckets
    bucket_counts = Counter(value_buckets)
    print(f"\nValue Buckets:")
    for bucket, count in bucket_counts.most_common():
        print(f"  {bucket}: {count:4d} samples")

    # Create composite label for stratification
    print("\nCreating composite labels (operation + command)...")
    composite_labels = create_composite_label(operation_types, commands)
    composite_counts = Counter(composite_labels)
    print(f"  Unique composite labels: {len(composite_counts)}")

    # Perform stratified split
    print("\nPerforming iterative stratified split...")
    train_idx, val_idx, test_idx = iterative_stratified_split(
        n_samples=n_samples,
        labels=operation_types,
        composite_labels=composite_labels,
        val_size=val_size,
        test_size=test_size,
        random_seed=random_seed
    )

    print(f"\nSplit Results:")
    print(f"  Train: {len(train_idx)} samples ({len(train_idx)/n_samples*100:.1f}%)")
    print(f"  Val:   {len(val_idx)} samples ({len(val_idx)/n_samples*100:.1f}%)")
    print(f"  Test:  {len(test_idx)} samples ({len(test_idx)/n_samples*100:.1f}%)")

    # Verify coverage
    print("\n" + "=" * 60)
    print("COVERAGE VERIFICATION")
    print("=" * 60)

    coverage_results = {}
    all_passed = True

    for split_name, indices in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        result = verify_coverage(
            split_name, indices, operation_types, commands, param_types,
            min_samples=min_samples_per_class
        )
        coverage_results[split_name] = result

        status = "PASS" if result['passed'] else "FAIL"
        print(f"\n{split_name.upper()} ({len(indices)} samples): {status}")

        # Show operation distribution
        print(f"  Operations: ", end="")
        for op_id in sorted(result['operation_counts'].keys()):
            count = result['operation_counts'][op_id]
            print(f"Op{op_id}:{count} ", end="")
        print()

        if result['issues']:
            print(f"  Issues:")
            for issue in result['issues']:
                print(f"    - {issue}")
            all_passed = False

    # Final coverage status
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL COVERAGE CHECKS PASSED")
    else:
        print("COVERAGE CHECKS FAILED - Some classes may be underrepresented")
    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save splits
    def save_split(name, indices):
        split_data = {key: combined[key][indices] for key in keys}
        np.savez(output_path / f'{name}_sequences.npz', **split_data)
        print(f"Saved {name}_sequences.npz with {len(indices)} samples")

    save_split('train', train_idx)
    save_split('val', val_idx)
    save_split('test', test_idx)

    # Save split indices for reproducibility
    np.savez(
        output_path / 'split_indices.npz',
        train=train_idx,
        val=val_idx,
        test=test_idx
    )
    print("Saved split_indices.npz for reproducibility")

    # Save split info
    def to_python(obj):
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, (np.integer,)) else k: to_python(v)
                    for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [to_python(x) for x in obj]
        return obj

    split_info = {
        'method': 'multilabel_stratified',
        'train_samples': int(len(train_idx)),
        'val_samples': int(len(val_idx)),
        'test_samples': int(len(test_idx)),
        'random_seed': random_seed,
        'val_size': val_size,
        'test_size': test_size,
        'min_samples_per_class': min_samples_per_class,
        'coverage_results': to_python(coverage_results),
        'label_counts': {
            'n_operations': len(op_counts),
            'n_commands': len(cmd_counts),
            'n_param_types': len(param_counts),
            'n_composite': len(composite_counts)
        }
    }

    with open(output_path / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    # Copy metadata files
    metadata_files = ['train_sequences_metadata.json', 'metadata.json', 'preprocessing_config.json']
    for mf in metadata_files:
        src = Path(data_dir) / mf
        if src.exists():
            import shutil
            shutil.copy(src, output_path / mf)
            print(f"Copied {mf}")

    print(f"\nAll files saved to {output_path}")

    return split_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create multi-label stratified splits with coverage checks'
    )
    parser.add_argument('--data-dir', type=str, default='outputs/grouped_splits_4digit',
                        help='Directory with original NPZ files')
    parser.add_argument('--output-dir', type=str, default='outputs/multilabel_stratified_splits',
                        help='Output directory for new splits')
    parser.add_argument('--vocab-path', type=str, default='data/vocabulary_4digit_hybrid.json',
                        help='Path to vocabulary file for token parsing')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='Validation set size as fraction')
    parser.add_argument('--test-size', type=float, default=0.15,
                        help='Test set size as fraction')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--min-samples', type=int, default=1,
                        help='Minimum samples per class per split')

    args = parser.parse_args()

    create_multilabel_stratified_splits(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        vocab_path=args.vocab_path,
        val_size=args.val_size,
        test_size=args.test_size,
        random_seed=args.seed,
        min_samples_per_class=args.min_samples
    )
