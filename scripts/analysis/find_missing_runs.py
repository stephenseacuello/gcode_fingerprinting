#!/usr/bin/env python3
"""
For each (position, class) pair where data is not present in every run,
report the exact run numbers that are missing.

Usage:
    python scripts/analysis/find_missing_runs.py --data-dir data/
"""
import argparse
import glob
import os
import re
from collections import defaultdict

from tqdm import tqdm


def parse_filename(filepath):
    """Return (class, run_number) from an aligned CSV filename."""
    m = re.match(r'^(.+?)_(\d+)_', os.path.basename(filepath))
    if m:
        return m.group(1), m.group(2)
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/')
    args = parser.parse_args()

    csv_files = glob.glob(os.path.join(args.data_dir, '**/*.csv'), recursive=True)
    csv_files = [f for f in csv_files if 'aligned' in f]

    # all_runs[class] = sorted list of run numbers seen
    all_runs = defaultdict(set)
    # runs_with_pos[position][class] = set of run numbers that have that position
    runs_with_pos = defaultdict(lambda: defaultdict(set))

    for filepath in tqdm(csv_files, desc='Scanning'):
        cls, run = parse_filename(filepath)
        if cls is None:
            continue
        all_runs[cls].add(run)

        # Read only the header to find which positions are present
        with open(filepath, 'r') as f:
            header = f.readline().strip().split(',')

        positions_in_file = set()
        for col in header:
            if '.' in col:
                pos = col.split('.', 1)[0]
                positions_in_file.add(pos)

        for pos in positions_in_file:
            runs_with_pos[pos][cls].add(run)

    classes = sorted(all_runs.keys())
    all_positions = sorted(runs_with_pos.keys())

    print(f'\n{"="*70}')
    print('  MISSING RUNS PER POSITION × CLASS')
    print(f'{"="*70}')
    print('  Only showing (position, class) pairs with at least one missing run.\n')

    any_missing = False
    for pos in all_positions:
        pos_printed = False
        for cls in classes:
            total = all_runs[cls]
            present = runs_with_pos[pos][cls]
            missing = sorted(total - present)
            if missing:
                if not pos_printed:
                    print(f'  {pos}')
                    pos_printed = True
                    any_missing = True
                print(f'    {cls:<20}  missing {len(missing)}/{len(total)} runs:  {", ".join(missing)}')
        if pos_printed:
            print()

    if not any_missing:
        print('  No missing runs found — all positions present in all runs.')

    # Summary: unique run numbers that have ANY missing position
    print(f'{"="*70}')
    print('  SUMMARY: runs that are incomplete (missing ≥1 position)')
    print(f'{"="*70}')
    incomplete = defaultdict(set)  # class -> set of run numbers
    for pos in all_positions:
        for cls in classes:
            missing = all_runs[cls] - runs_with_pos[pos][cls]
            for run in missing:
                incomplete[cls].add(run)

    for cls in classes:
        runs = sorted(incomplete[cls])
        if runs:
            print(f'  {cls:<22}  {len(runs)} incomplete run(s):  {", ".join(runs)}')
        else:
            print(f'  {cls:<22}  all runs complete')


if __name__ == '__main__':
    main()
