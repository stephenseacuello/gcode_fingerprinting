#!/usr/bin/env python3
"""Create stratified data splits with different random seeds.

Takes existing pre-split data (train/val/test_sequences.npz), recombines it,
and re-splits using a specified seed. Preserves stratification by operation_type.

This enables proper multi-seed ANOVA designs where data_split_seed varies the
train/val/test partition rather than just being a label.

Usage:
    # Create all 3 splits for the ablation study
    python scripts/data/create_data_splits.py \
        --source-dir outputs/7class_cascade_to_9class/9class_moddropout_final/data \
        --output-base outputs/ablation_study_2026_02_06/data_splits \
        --seeds 42 123 456

    # Create a single split
    python scripts/data/create_data_splits.py \
        --source-dir outputs/7class_cascade_to_9class/9class_moddropout_final/data \
        --output-base outputs/ablation_study_2026_02_06/data_splits \
        --seeds 42
"""
import argparse
import json
import shutil
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit


def load_and_combine(source_dir):
    """Load train/val/test NPZs and recombine into single dataset."""
    source_dir = Path(source_dir)

    splits = {}
    for split in ['train', 'val', 'test']:
        npz_path = source_dir / f'{split}_sequences.npz'
        splits[split] = np.load(npz_path, allow_pickle=True)

    keys = list(splits['train'].keys())
    print(f"Data keys: {keys}")

    combined = {}
    for key in keys:
        combined[key] = np.concatenate(
            [splits[s][key] for s in ['train', 'val', 'test']], axis=0
        )

    n_train = len(splits['train']['operation_type'])
    n_val = len(splits['val']['operation_type'])
    n_test = len(splits['test']['operation_type'])
    n_total = n_train + n_val + n_test

    print(f"Original: train={n_train}, val={n_val}, test={n_test}, total={n_total}")

    return combined, keys


def create_split(combined, keys, seed, val_frac=0.15, test_frac=0.15):
    """Create a stratified split with given seed."""
    labels = combined['operation_type']
    n = len(labels)

    # Split 1: separate test set
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(np.zeros(n), labels))

    # Split 2: separate val from train
    val_frac_adj = val_frac / (1 - test_frac)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_adj, random_state=seed)
    train_idx_rel, val_idx_rel = next(sss_val.split(
        np.zeros(len(train_val_idx)), labels[train_val_idx]
    ))

    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    return train_idx, val_idx, test_idx


def save_split(combined, keys, indices, output_dir, split_name):
    """Save a single split as NPZ."""
    data = {key: combined[key][indices] for key in keys}
    np.savez(Path(output_dir) / f'{split_name}_sequences.npz', **data)


def main():
    parser = argparse.ArgumentParser(description='Create stratified data splits')
    parser.add_argument('--source-dir', type=str, required=True,
                        help='Directory with original train/val/test_sequences.npz')
    parser.add_argument('--output-base', type=str, required=True,
                        help='Base output directory (splits go in split_<seed>/ subdirs)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seeds for splitting')
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--test-size', type=float, default=0.15)

    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    output_base = Path(args.output_base)

    # Load and combine all data
    print("Loading and combining data...")
    combined, keys = load_and_combine(source_dir)
    labels = combined['operation_type']
    n_total = len(labels)

    print(f"\nOverall class distribution:")
    for cls_id, count in sorted(Counter(labels).items()):
        print(f"  Class {cls_id}: {count} ({count/n_total*100:.1f}%)")

    # Create splits for each seed
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Creating split with seed={seed}")
        print(f"{'='*60}")

        train_idx, val_idx, test_idx = create_split(
            combined, keys, seed,
            val_frac=args.val_size, test_frac=args.test_size
        )

        split_dir = output_base / f'split_{seed}'
        split_dir.mkdir(parents=True, exist_ok=True)

        # Save NPZ files
        for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            save_split(combined, keys, idx, split_dir, name)
            split_labels = labels[idx]
            class_dist = Counter(split_labels)
            print(f"  {name}: {len(idx)} samples | classes: {dict(sorted(class_dist.items()))}")

        # Save split indices for reproducibility
        np.savez(split_dir / 'split_indices.npz',
                 train=train_idx, val=val_idx, test=test_idx)

        # Copy metadata and scaler files from source
        for fname in ['metadata.json', 'scaler_stats.json']:
            src = source_dir / fname
            if src.exists():
                shutil.copy(src, split_dir / fname)

        # Save split info
        split_info = {
            'seed': seed,
            'source_dir': str(source_dir),
            'val_size': args.val_size,
            'test_size': args.test_size,
            'n_total': n_total,
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_test': len(test_idx),
            'train_class_dist': {int(k): int(v) for k, v in Counter(labels[train_idx]).items()},
            'val_class_dist': {int(k): int(v) for k, v in Counter(labels[val_idx]).items()},
            'test_class_dist': {int(k): int(v) for k, v in Counter(labels[test_idx]).items()},
        }
        with open(split_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)

        print(f"  Saved to: {split_dir}")

    print(f"\nAll {len(args.seeds)} splits created in {output_base}")


if __name__ == '__main__':
    main()
