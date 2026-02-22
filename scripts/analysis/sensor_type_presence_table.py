#!/usr/bin/env python3
"""
Simple presence table: for each sensor_type × position × class,
count how many runs have that sensor type's columns in the CSV.

Usage:
    python scripts/analysis/sensor_type_presence_table.py --data-dir data/
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

CHANNEL_TO_TYPE = {
    'Ax': 'Accelerometer', 'Ay': 'Accelerometer', 'Az': 'Accelerometer',
    'Gx': 'Gyroscope',     'Gy': 'Gyroscope',     'Gz': 'Gyroscope',
    'Mx': 'Magnetometer',  'My': 'Magnetometer',  'Mz': 'Magnetometer',
    'Pressure':    'Pressure',
    'Temperature': 'Temperature',
    'Proximity':   'Proximity',
    'ColorR': 'Color', 'ColorG': 'Color', 'ColorB': 'Color', 'ColorA': 'Color',
    'RMS': 'RMS',
}


def get_class(filepath):
    m = re.match(r'^(.+?)_\d+_', os.path.basename(filepath))
    return m.group(1) if m else 'unknown'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/')
    args = parser.parse_args()

    csv_files = glob.glob(os.path.join(args.data_dir, '**/*.csv'), recursive=True)
    csv_files = [f for f in csv_files if 'aligned' in f]

    # presence[sensor_type][position][class] = count of runs where columns exist
    presence = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    class_totals = defaultdict(int)

    for filepath in tqdm(csv_files, desc='Scanning'):
        cls = get_class(filepath)
        class_totals[cls] += 1

        # Only read the header row — no need to load data
        df = pd.read_csv(filepath, nrows=0)

        seen = set()  # avoid double-counting channels of same type at same position
        for col in df.columns:
            if '.' not in col:
                continue
            pos, ch = col.split('.', 1)
            stype = CHANNEL_TO_TYPE.get(ch)
            if stype and (pos, stype) not in seen:
                presence[stype][pos][cls] += 1
                seen.add((pos, stype))

    classes = sorted(class_totals.keys())
    sensor_types = sorted(presence.keys())
    positions = sorted({pos for st in presence.values() for pos in st},
                       key=lambda p: sum(presence[st][p].get(cls, 0)
                                         for st in presence for cls in classes),
                       reverse=True)

    total_runs = sum(class_totals.values())
    col_w = max(len(c) for c in classes) + 3

    for stype in sensor_types:
        print(f'\n{"="*80}')
        print(f'  {stype}')
        print(f'{"="*80}')
        header = f"  {'Position':<15}" + ''.join(f'{c:>{col_w}}' for c in classes) + f'{"TOTAL":>10}'
        print(header)
        print('  ' + '-' * (len(header) - 2))

        for pos in positions:
            row = f'  {pos:<15}'
            pos_total = 0
            for cls in classes:
                cnt = presence[stype][pos].get(cls, 0)
                n = class_totals[cls]
                pos_total += cnt
                cell = f'{cnt}/{n}'
                row += f'{cell:>{col_w}}'
            row += f'{pos_total:>6}/{total_runs}'
            print(row)

    # Summary: which (sensor_type, position) pairs are present in every single run?
    print(f'\n{"="*80}')
    print('  SUMMARY: present in ALL runs across ALL classes?')
    print(f'{"="*80}')
    for stype in sensor_types:
        always = [p for p in positions
                  if all(presence[stype][p].get(cls, 0) == class_totals[cls] for cls in classes)]
        partial = [p for p in positions
                   if not all(presence[stype][p].get(cls, 0) == class_totals[cls] for cls in classes)
                   and any(presence[stype][p].get(cls, 0) > 0 for cls in classes)]
        print(f'\n  {stype}')
        print(f'    Always present ({len(always)}): {", ".join(always) if always else "none"}')
        print(f'    Sometimes:      ({len(partial)}): {", ".join(partial) if partial else "none"}')


if __name__ == '__main__':
    main()
