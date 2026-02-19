#!/usr/bin/env python3
"""
Analyze zero values in the 5 consistent sensors across all raw CSV files.

For each sensor channel, calculate:
- Percentage of zero values
- Mean, min, max
- This distinguishes truly dead sensors from sensors that cross zero naturally
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

# The 5 sensors with >=95% activity
SENSORS = ['frame_l2', 'frame_r2', 'spindle2', 'y_bed__3', 'y_bed__4']

# 17 channels per sensor
CHANNELS = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz',
            'Pressure', 'Temperature', 'Proximity',
            'ColorR', 'ColorG', 'ColorB', 'ColorA', 'RMS']


def extract_operation_type(filename):
    """Extract operation type from filename."""
    fname_lower = filename.lower()

    if 'adaptive150025' in fname_lower:
        return 'adaptive150025'
    elif 'face150025' in fname_lower:
        return 'face150025'
    elif 'pocket150025' in fname_lower:
        return 'pocket150025'
    elif 'damageadaptive' in fname_lower:
        return 'damageadaptive'
    elif 'damageface' in fname_lower:
        return 'damageface'
    elif 'damagepocket' in fname_lower:
        return 'damagepocket'
    elif 'adaptive' in fname_lower:
        return 'adaptive'
    elif 'face' in fname_lower:
        return 'face'
    elif 'pocket' in fname_lower:
        return 'pocket'
    else:
        return 'unknown'


def analyze_sensor_channel(data_dir):
    """Analyze zero patterns in sensor channels across all files."""

    csv_files = sorted(Path(data_dir).glob("*_aligned.csv"))
    if not csv_files:
        csv_files = sorted(Path(data_dir).glob("*.csv"))

    print(f"Found {len(csv_files)} CSV files\n")
    print("="*100)
    print("ANALYZING ZERO PATTERNS IN 5 CONSISTENT SENSORS")
    print("="*100)

    # Collect stats for each sensor.channel
    stats = defaultdict(lambda: {
        'values': [],
        'files_with_sensor': 0,
        'files_all_zeros': 0,
        'total_values': 0,
        'zero_values': 0,
        'by_operation': defaultdict(lambda: {
            'files': 0,
            'all_zeros': 0,
            'total_values': 0,
            'zero_values': 0,
        })
    })

    # Process each file
    for csv_file in csv_files:
        operation = extract_operation_type(csv_file.name)
        df = pd.read_csv(csv_file)

        # Check each sensor.channel
        for sensor in SENSORS:
            for channel in CHANNELS:
                col_name = f"{sensor}.{channel}"

                if col_name in df.columns:
                    values = df[col_name].values

                    # Overall stats
                    stats[col_name]['files_with_sensor'] += 1
                    stats[col_name]['values'].extend(values)
                    stats[col_name]['total_values'] += len(values)

                    # Count zeros (exactly 0)
                    zero_count = (values == 0).sum()
                    stats[col_name]['zero_values'] += zero_count

                    # Check if ALL values are zero
                    if np.all(values == 0):
                        stats[col_name]['files_all_zeros'] += 1

                    # Per-operation stats
                    op_stats = stats[col_name]['by_operation'][operation]
                    op_stats['files'] += 1
                    op_stats['total_values'] += len(values)
                    op_stats['zero_values'] += zero_count
                    if np.all(values == 0):
                        op_stats['all_zeros'] += 1

    # Print results
    print("\n" + "="*100)
    print("SENSOR CHANNEL ANALYSIS")
    print("="*100)

    # Group by sensor
    for sensor in SENSORS:
        print(f"\n{'='*100}")
        print(f"SENSOR: {sensor}")
        print('='*100)
        print(f"{'Channel':<15} {'Files':<8} {'Dead Files':<12} {'Zero %':<10} "
              f"{'Mean':<12} {'Min':<12} {'Max':<12} {'Status':<20}")
        print('-'*100)

        for channel in CHANNELS:
            col_name = f"{sensor}.{channel}"
            s = stats[col_name]

            if s['files_with_sensor'] == 0:
                continue

            # Calculate statistics (ignoring NaNs)
            values = np.array(s['values'])
            zero_pct = 100 * s['zero_values'] / s['total_values']
            dead_files = s['files_all_zeros']

            # Use nanmean, nanmin, nanmax to handle NaN values
            mean_val = np.nanmean(values)
            min_val = np.nanmin(values)
            max_val = np.nanmax(values)

            # Classify
            if dead_files > 0:
                if dead_files >= s['files_with_sensor'] * 0.5:
                    status = "🔴 DEAD (50%+ files)"
                else:
                    status = f"⚠️  DEAD in {dead_files} files"
            elif zero_pct > 80:
                status = "⚠️  Mostly zeros"
            elif min_val < 0 and max_val > 0:
                status = "✅ Normal (crosses zero)"
            elif zero_pct < 1:
                status = "✅ Normal (rarely zero)"
            else:
                status = "⚠️  Check manually"

            print(f"{channel:<15} {s['files_with_sensor']:<8} {dead_files:<12} "
                  f"{zero_pct:>8.1f}% {mean_val:>11.3f} {min_val:>11.3f} "
                  f"{max_val:>11.3f} {status:<20}")

        print()

    # Print problematic channels by operation
    print("\n" + "="*100)
    print("PROBLEMATIC CHANNELS BY OPERATION TYPE")
    print("="*100)

    for sensor in SENSORS:
        for channel in CHANNELS:
            col_name = f"{sensor}.{channel}"
            s = stats[col_name]

            # Check if any operations have this channel completely dead
            dead_operations = []
            for op, op_stats in s['by_operation'].items():
                if op_stats['all_zeros'] > 0:
                    pct = 100 * op_stats['all_zeros'] / op_stats['files']
                    if pct >= 50:  # Dead in 50%+ of files for this operation
                        dead_operations.append((op, op_stats['all_zeros'], op_stats['files'], pct))

            if dead_operations:
                print(f"\n{col_name}:")
                for op, dead, total, pct in dead_operations:
                    print(f"  🔴 {op:<20} - ALL ZEROS in {dead}/{total} files ({pct:.0f}%)")

    # Summary
    print("\n" + "="*100)
    print("SUMMARY: Channels to Consider Excluding")
    print("="*100)

    exclude_candidates = []

    for sensor in SENSORS:
        for channel in CHANNELS:
            col_name = f"{sensor}.{channel}"
            s = stats[col_name]

            if s['files_all_zeros'] > 0:
                dead_pct = 100 * s['files_all_zeros'] / s['files_with_sensor']
                if dead_pct >= 20:  # Dead in 20%+ of files
                    exclude_candidates.append((col_name, dead_pct, s['files_all_zeros'], s['files_with_sensor']))

    if exclude_candidates:
        print("\nChannels that are ALL ZEROS in significant number of files:\n")
        exclude_candidates.sort(key=lambda x: -x[1])  # Sort by dead percentage

        for col_name, dead_pct, dead_files, total_files in exclude_candidates:
            print(f"  ❌ {col_name:<30} - Dead in {dead_files}/{total_files} files ({dead_pct:.1f}%)")
    else:
        print("\n✅ No channels are consistently dead across files!")

    print("\n" + "="*100)


if __name__ == '__main__':
    data_dir = Path('data')
    analyze_sensor_channel(data_dir)
