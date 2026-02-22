#!/usr/bin/env python3
"""
Create a clean dataset containing only:
  - Runs where ALL required positions are present
  - Columns belonging to those positions only

Runs a dry-run by default (shows what would be written).
Pass --execute to actually write files.

Usage:
    # Dry run — see what survives
    python scripts/analysis/create_clean_dataset.py --data-dir data/ --output-dir data_clean/

    # Write the clean dataset
    python scripts/analysis/create_clean_dataset.py --data-dir data/ --output-dir data_clean/ --execute
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# The 6 positions identified as giving 120 clean runs within class constraints
DEFAULT_POSITIONS = [
    'frame_r2',
    'frame_l2',
    'frame_l3',
    'spindle2',
    'y_bed__3',
    'y_bed__4',
]


def parse_filename(filepath):
    m = re.match(r'^(.+?)_(\d+)_', os.path.basename(filepath))
    if m:
        return m.group(1), m.group(2)
    return None, None


def has_all_positions(columns, required):
    present = {col.split('.', 1)[0] for col in columns if '.' in col}
    return all(pos in present for pos in required)


def filter_columns(df, required):
    """Keep only columns belonging to required positions, plus any non-position columns."""
    keep = [
        col for col in df.columns
        if '.' not in col or col.split('.', 1)[0] in required
    ]
    return df[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/')
    parser.add_argument('--output-dir', default='data_clean/')
    parser.add_argument('--positions', nargs='+', default=DEFAULT_POSITIONS,
                        help='Required positions (space-separated)')
    parser.add_argument('--execute', action='store_true',
                        help='Write files (default: dry run only)')
    args = parser.parse_args()

    required = set(args.positions)

    csv_files = glob.glob(os.path.join(args.data_dir, '**/*.csv'), recursive=True)
    csv_files = [f for f in csv_files if 'aligned' in f]

    kept = []
    skipped_by_class = defaultdict(list)

    for filepath in tqdm(csv_files, desc='Scanning'):
        cls, run = parse_filename(filepath)
        if cls is None:
            continue
        header = pd.read_csv(filepath, nrows=0)
        if has_all_positions(header.columns, required):
            kept.append(filepath)
        else:
            skipped_by_class[cls].append(run)

    # Summary
    class_counts = defaultdict(int)
    for f in kept:
        cls, _ = parse_filename(f)
        class_counts[cls] += 1

    print(f'\n{"="*60}')
    print(f'  Positions required ({len(required)}):')
    for p in sorted(required):
        print(f'    {p}')
    print(f'{"="*60}')
    print(f'  Runs kept:    {len(kept)} / {len(csv_files)}')
    print(f'  Runs dropped: {len(csv_files) - len(kept)} / {len(csv_files)}')
    print(f'\n  Per-class breakdown:')
    for cls in sorted(class_counts):
        dropped = sorted(skipped_by_class[cls])
        dropped_str = f'  dropped: {", ".join(dropped)}' if dropped else ''
        print(f'    {cls:<22} {class_counts[cls]:>3} runs{dropped_str}')

    if not args.execute:
        print(f'\n  [DRY RUN] No files written.')
        print(f'  Run with --execute to write clean files to: {args.output_dir}')
        return

    # Write filtered files
    print(f'\n  Writing filtered CSVs to: {args.output_dir}')
    for filepath in tqdm(kept, desc='Writing'):
        df = pd.read_csv(filepath)
        df_clean = filter_columns(df, required)

        rel = os.path.relpath(filepath, args.data_dir)
        out_path = os.path.join(args.output_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_clean.to_csv(out_path, index=False)

    print(f'\n  Done. {len(kept)} files written to {args.output_dir}')


if __name__ == '__main__':
    main()
