#!/usr/bin/env python3
"""
Identify sensors that are consistently active across all experimental runs.

This script analyzes all CSV files to determine:
1. Which sensor locations are present in each file
2. Which sensors actually have signal activity (non-zero variance)
3. Coverage percentage for each sensor across all files
4. Recommends sensors for consistent-length data without padding
5. Per-class sensor activity breakdown to reveal class-specific sensor behaviour

Usage:
    python scripts/analysis/identify_consistent_sensors.py --data-dir data/ --threshold 95
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Maps individual channel names to their physical sensor type
CHANNEL_TO_SENSOR_TYPE = {
    'Ax': 'Accelerometer', 'Ay': 'Accelerometer', 'Az': 'Accelerometer',
    'Gx': 'Gyroscope',     'Gy': 'Gyroscope',     'Gz': 'Gyroscope',
    'Mx': 'Magnetometer',  'My': 'Magnetometer',  'Mz': 'Magnetometer',
    'Pressure':    'Pressure',
    'Temperature': 'Temperature',
    'Proximity':   'Proximity',
    'ColorR': 'Color', 'ColorG': 'Color', 'ColorB': 'Color', 'ColorA': 'Color',
    'RMS': 'RMS',
}


def extract_class_from_filename(filepath):
    """
    Extract the class label from a filename like 'adaptive150025_012_aligned.csv'.

    The convention is: {class}_{run_number}_{suffix}.csv
    e.g. 'face_017_aligned.csv'  -> 'face'
         'pocket150025_020_aligned.csv' -> 'pocket150025'
    """
    basename = os.path.basename(filepath)
    match = re.match(r'^(.+?)_(\d+)_', basename)
    if match:
        return match.group(1)
    return 'unknown'


def get_sensor_channels(df_columns):
    """
    Extract sensor locations from DataFrame columns.

    Sensor columns follow pattern: {sensor_name}.{channel_name}
    e.g., 'frame_r2.Ax', 'spindle1.Temperature'

    Returns:
        dict: {sensor_name: [list of channel columns]}
    """
    sensors = defaultdict(list)

    for col in df_columns:
        if '.' in col:  # Sensor column format: sensor.channel
            sensor_name, channel = col.split('.', 1)
            sensors[sensor_name].append(col)

    return dict(sensors)


def check_sensor_activity(df, sensor_columns, activity_threshold=1e-6):
    """
    Check if a sensor has actual signal activity (not just zeros or constant values).

    Args:
        df: DataFrame with sensor data
        sensor_columns: List of column names for this sensor
        activity_threshold: Minimum std deviation to consider active

    Returns:
        bool: True if sensor shows activity, False if constant/zeros
    """
    # Check each channel for non-zero variance
    active_channels = 0

    for col in sensor_columns:
        if col in df.columns:
            values = df[col].values
            # Remove NaN values
            values = values[~np.isnan(values)]

            if len(values) > 0:
                std = np.std(values)
                if std > activity_threshold:
                    active_channels += 1

    # Sensor is active if at least half its channels show activity
    return active_channels >= len(sensor_columns) / 2


def check_sensor_type_activity(df, type_columns, activity_threshold=1e-6):
    """
    Check if a specific sensor type (e.g. Accelerometer) at one position is active.

    Args:
        df: DataFrame with sensor data
        type_columns: List of column names belonging to this sensor type at this position
                      e.g. ['frame_r1.Ax', 'frame_r1.Ay', 'frame_r1.Az']
        activity_threshold: Minimum std deviation to consider active

    Returns:
        bool: True if at least half the channels of this type show activity
    """
    active = 0
    for col in type_columns:
        if col in df.columns:
            values = df[col].values
            values = values[~np.isnan(values)]
            if len(values) > 0 and np.std(values) > activity_threshold:
                active += 1
    return active >= len(type_columns) / 2


def analyze_sensor_consistency(data_dir, activity_threshold=1e-6, verbose=True):
    """
    Analyze sensor presence and activity across all CSV files.

    Args:
        data_dir: Directory containing CSV files
        activity_threshold: Minimum std deviation for activity
        verbose: Print progress information

    Returns:
        dict: Analysis results
    """
    # Find all CSV files
    csv_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)

    # Filter out aligned files if they exist alongside raw files
    raw_files = [f for f in csv_files if 'aligned' not in f]
    if len(raw_files) > 0:
        csv_files = raw_files

    if verbose:
        print(f"Found {len(csv_files)} CSV files in {data_dir}")

    # Track sensor presence and activity — global
    sensor_presence = defaultdict(int)
    sensor_activity = defaultdict(int)
    sensor_channels = {}
    total_files = len(csv_files)

    # Track per-class sensor presence and activity
    class_file_counts = defaultdict(int)
    class_sensor_presence = defaultdict(lambda: defaultdict(int))
    class_sensor_activity = defaultdict(lambda: defaultdict(int))

    # Track sensor-type activity at (type, position, class) granularity
    # sensor_type_data[type][position][cls] = {'active': N, 'total': M}
    sensor_type_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {'active': 0, 'total': 0}))
    )

    # Track per-file sensor counts for statistics
    file_sensor_counts = []

    # Analyze each file
    for csv_path in tqdm(csv_files, desc="Analyzing files", disable=not verbose):
        try:
            # Read CSV
            df = pd.read_csv(csv_path, low_memory=False)

            # Determine class from filename
            cls = extract_class_from_filename(csv_path)
            class_file_counts[cls] += 1

            # Get sensors in this file
            file_sensors = get_sensor_channels(df.columns)
            file_sensor_counts.append(len(file_sensors))

            # Update presence and activity counts
            for sensor_name, columns in file_sensors.items():
                # Track channel names (should be consistent)
                if sensor_name not in sensor_channels:
                    channel_names = [col.split('.', 1)[1] for col in columns]
                    sensor_channels[sensor_name] = sorted(channel_names)

                # Global
                sensor_presence[sensor_name] += 1
                is_active = check_sensor_activity(df, columns, activity_threshold)
                if is_active:
                    sensor_activity[sensor_name] += 1

                # Per-class
                class_sensor_presence[cls][sensor_name] += 1
                if is_active:
                    class_sensor_activity[cls][sensor_name] += 1

                # Per sensor-type within this position
                type_groups = defaultdict(list)
                for col in columns:
                    channel = col.split('.', 1)[1]
                    stype = CHANNEL_TO_SENSOR_TYPE.get(channel, 'Unknown')
                    type_groups[stype].append(col)

                for stype, type_cols in type_groups.items():
                    sensor_type_data[stype][sensor_name][cls]['total'] += 1
                    if check_sensor_type_activity(df, type_cols, activity_threshold):
                        sensor_type_data[stype][sensor_name][cls]['active'] += 1

        except Exception as e:
            if verbose:
                print(f"Error processing {csv_path}: {e}")
            continue

    # Build global sensor results
    results = {
        'total_files': total_files,
        'sensors': {},
        'statistics': {
            'min_sensors_per_file': min(file_sensor_counts) if file_sensor_counts else 0,
            'max_sensors_per_file': max(file_sensor_counts) if file_sensor_counts else 0,
            'mean_sensors_per_file': np.mean(file_sensor_counts) if file_sensor_counts else 0,
        },
        'classes': {},
    }

    for sensor_name in sorted(sensor_presence.keys()):
        presence_count = sensor_presence[sensor_name]
        activity_count = sensor_activity[sensor_name]
        presence_pct = (presence_count / total_files) * 100
        activity_pct = (activity_count / total_files) * 100

        results['sensors'][sensor_name] = {
            'presence_count': presence_count,
            'presence_percentage': presence_pct,
            'activity_count': activity_count,
            'activity_percentage': activity_pct,
            'channels': sensor_channels.get(sensor_name, []),
            'num_channels': len(sensor_channels.get(sensor_name, [])),
        }

    # Build per-class sensor results
    for cls in sorted(class_file_counts.keys()):
        n_files = class_file_counts[cls]
        results['classes'][cls] = {
            'total_files': n_files,
            'sensors': {},
        }
        all_sensors_for_class = set(class_sensor_presence[cls].keys())
        for sensor_name in sorted(all_sensors_for_class):
            presence_count = class_sensor_presence[cls][sensor_name]
            activity_count = class_sensor_activity[cls][sensor_name]
            results['classes'][cls]['sensors'][sensor_name] = {
                'presence_count': presence_count,
                'presence_percentage': (presence_count / n_files) * 100,
                'activity_count': activity_count,
                'activity_percentage': (activity_count / n_files) * 100,
            }

    # Build sensor-type results
    results['sensor_types'] = {}
    for stype in sorted(sensor_type_data.keys()):
        positions = sorted(sensor_type_data[stype].keys())
        type_entry = {'positions': {}, 'classes': {}}

        # Aggregate across all (position, class) combinations
        total_active_all = 0
        total_obs_all = 0

        for pos in positions:
            cls_data = sensor_type_data[stype][pos]
            pos_active = 0
            pos_total = 0
            pos_classes = {}
            for cls, counts in sorted(cls_data.items()):
                pct = (counts['active'] / counts['total'] * 100) if counts['total'] > 0 else 0.0
                pos_classes[cls] = {
                    'active': counts['active'],
                    'total': counts['total'],
                    'activity_percentage': pct,
                }
                pos_active += counts['active']
                pos_total += counts['total']
            mean_pct = (pos_active / pos_total * 100) if pos_total > 0 else 0.0
            type_entry['positions'][pos] = {
                'classes': pos_classes,
                'mean_activity_percentage': mean_pct,
            }
            total_active_all += pos_active
            total_obs_all += pos_total

        # Per-class summary (averaged across positions)
        all_classes = sorted({cls for pos in positions for cls in sensor_type_data[stype][pos]})
        for cls in all_classes:
            cls_active = sum(
                sensor_type_data[stype][pos][cls]['active']
                for pos in positions if cls in sensor_type_data[stype][pos]
            )
            cls_total = sum(
                sensor_type_data[stype][pos][cls]['total']
                for pos in positions if cls in sensor_type_data[stype][pos]
            )
            type_entry['classes'][cls] = {
                'activity_percentage': (cls_active / cls_total * 100) if cls_total > 0 else 0.0
            }

        type_entry['overall_activity_percentage'] = (
            (total_active_all / total_obs_all * 100) if total_obs_all > 0 else 0.0
        )
        results['sensor_types'][stype] = type_entry

    return results


def print_results(results, threshold=95.0):
    """Print analysis results in a readable format."""

    print("\n" + "="*80)
    print("SENSOR CONSISTENCY ANALYSIS")
    print("="*80)

    print(f"\nTotal files analyzed: {results['total_files']}")
    print(f"Sensors per file: {results['statistics']['min_sensors_per_file']:.0f} - "
          f"{results['statistics']['max_sensors_per_file']:.0f} "
          f"(mean: {results['statistics']['mean_sensors_per_file']:.1f})")

    print("\n" + "-"*80)
    print("ALL SENSORS (sorted by activity %)")
    print("-"*80)
    print(f"{'Sensor':<15} {'Present':>8} {'Pres%':>6} {'Active':>8} {'Act%':>6} {'Channels':>8}")
    print("-"*80)

    # Sort by activity percentage
    sorted_sensors = sorted(
        results['sensors'].items(),
        key=lambda x: x[1]['activity_percentage'],
        reverse=True
    )

    for sensor_name, info in sorted_sensors:
        print(f"{sensor_name:<15} "
              f"{info['presence_count']:>8} "
              f"{info['presence_percentage']:>5.1f}% "
              f"{info['activity_count']:>8} "
              f"{info['activity_percentage']:>5.1f}% "
              f"{info['num_channels']:>8}")

    # Identify consistent sensors
    print("\n" + "="*80)
    print(f"CONSISTENT SENSORS (≥{threshold}% activity)")
    print("="*80)

    consistent_sensors = {
        name: info for name, info in results['sensors'].items()
        if info['activity_percentage'] >= threshold
    }

    if consistent_sensors:
        print(f"\nFound {len(consistent_sensors)} sensors with ≥{threshold}% activity:\n")

        total_channels = 0
        for sensor_name, info in sorted(consistent_sensors.items(),
                                       key=lambda x: x[1]['activity_percentage'],
                                       reverse=True):
            print(f"  • {sensor_name:<15} - {info['activity_percentage']:>5.1f}% active "
                  f"({info['num_channels']} channels)")
            total_channels += info['num_channels']

        print(f"\nTotal features from consistent sensors:")
        print(f"  • Sensor features: {total_channels}")
        print(f"  • Electrical features: 8")
        print(f"  • TOTAL: {total_channels + 8} features")

        print(f"\n✅ These {len(consistent_sensors)} sensors can be used WITHOUT zero-padding!")
        print(f"   All files have active signal from these sensors.\n")
    else:
        print(f"\n⚠️  No sensors found with ≥{threshold}% activity.")
        print(f"   Try lowering the threshold.\n")

    # Show borderline sensors
    borderline_sensors = {
        name: info for name, info in results['sensors'].items()
        if 90 <= info['activity_percentage'] < threshold
    }

    if borderline_sensors:
        print("-"*80)
        print(f"BORDERLINE SENSORS (90-{threshold}% activity)")
        print("-"*80)
        print("\nThese sensors are mostly consistent but missing in some files:\n")

        for sensor_name, info in sorted(borderline_sensors.items(),
                                       key=lambda x: x[1]['activity_percentage'],
                                       reverse=True):
            missing = results['total_files'] - info['activity_count']
            print(f"  • {sensor_name:<15} - {info['activity_percentage']:>5.1f}% active "
                  f"(missing in {missing} files)")

    # Show low-activity sensors
    low_activity_sensors = {
        name: info for name, info in results['sensors'].items()
        if info['activity_percentage'] < 50
    }

    if low_activity_sensors:
        print("\n" + "-"*80)
        print("LOW ACTIVITY SENSORS (<50% activity)")
        print("-"*80)
        print("\n⚠️  These sensors rarely show activity and should be excluded:\n")

        for sensor_name, info in sorted(low_activity_sensors.items(),
                                       key=lambda x: x[1]['activity_percentage']):
            print(f"  • {sensor_name:<15} - {info['activity_percentage']:>5.1f}% active")

    print("\n" + "="*80 + "\n")


def print_class_results(results, threshold=95.0):
    """
    Print per-class sensor activity analysis:
      1. Activity matrix  — sensors × classes (activity %)
      2. Cross-class consistent sensors — active in every class above threshold
      3. Per-class sensor ranking
    """
    classes = sorted(results['classes'].keys())
    all_sensors = sorted(results['sensors'].keys())

    if not classes:
        print("No class information found in results.")
        return

    print("\n" + "="*80)
    print("PER-CLASS SENSOR ACTIVITY ANALYSIS")
    print("="*80)

    # ── 1. Activity matrix ──────────────────────────────────────────────────
    print("\n" + "-"*80)
    print("SENSOR × CLASS ACTIVITY MATRIX  (activity %)")
    print("-"*80)

    col_w = 10  # width per class column
    header = f"{'Sensor':<15}" + "".join(f"{cls:>{col_w}}" for cls in classes) + f"{'GLOBAL':>{col_w}}"
    print(header)
    print("-" * len(header))

    for sensor_name in all_sensors:
        global_pct = results['sensors'][sensor_name]['activity_percentage']
        row = f"{sensor_name:<15}"
        for cls in classes:
            cls_sensors = results['classes'][cls]['sensors']
            if sensor_name in cls_sensors:
                pct = cls_sensors[sensor_name]['activity_percentage']
                row += f"{pct:>{col_w - 1}.1f}%"
            else:
                row += f"{'—':>{col_w}}"
        row += f"{global_pct:>{col_w - 1}.1f}%"
        print(row)

    # ── 2. Cross-class consistent sensors ──────────────────────────────────
    print("\n" + "-"*80)
    print(f"CROSS-CLASS CONSISTENT SENSORS  (≥{threshold}% active in EVERY class)")
    print("-"*80)

    cross_consistent = []
    for sensor_name in all_sensors:
        active_in_all = all(
            results['classes'][cls]['sensors'].get(sensor_name, {}).get('activity_percentage', 0.0) >= threshold
            for cls in classes
        )
        if active_in_all:
            cross_consistent.append(sensor_name)

    if cross_consistent:
        print(f"\nFound {len(cross_consistent)} sensors active in ALL {len(classes)} classes:\n")
        for s in cross_consistent:
            per_class = ", ".join(
                f"{cls}: {results['classes'][cls]['sensors'].get(s, {}).get('activity_percentage', 0):.1f}%"
                for cls in classes
            )
            print(f"  • {s:<15}  [{per_class}]")
    else:
        print(f"\n  None — no sensor meets ≥{threshold}% in every class.")
        print("  Consider lowering --threshold or inspecting per-class rankings below.")

    # Partially consistent: meets threshold in some but not all classes
    partial = [
        s for s in all_sensors
        if s not in cross_consistent
        and any(
            results['classes'][cls]['sensors'].get(s, {}).get('activity_percentage', 0.0) >= threshold
            for cls in classes
        )
    ]
    if partial:
        print(f"\n  Sensors meeting threshold in SOME (not all) classes:")
        for s in partial:
            class_hits = [
                cls for cls in classes
                if results['classes'][cls]['sensors'].get(s, {}).get('activity_percentage', 0.0) >= threshold
            ]
            print(f"    • {s:<15}  ({len(class_hits)}/{len(classes)} classes: {', '.join(class_hits)})")

    # ── 3. Per-class sensor ranking ────────────────────────────────────────
    print("\n" + "-"*80)
    print("PER-CLASS SENSOR RANKING  (sorted by activity %)")
    print("-"*80)

    for cls in classes:
        n_files = results['classes'][cls]['total_files']
        cls_sensors = results['classes'][cls]['sensors']
        ranked = sorted(cls_sensors.items(), key=lambda x: x[1]['activity_percentage'], reverse=True)

        print(f"\n  [{cls}]  ({n_files} runs)")
        print(f"  {'Rank':<6} {'Sensor':<15} {'Active':>8} {'Act%':>7}")
        print(f"  {'-'*40}")
        for rank, (sensor_name, info) in enumerate(ranked, 1):
            bar_len = int(info['activity_percentage'] / 5)  # one block per 5%
            bar = '█' * bar_len
            print(f"  {rank:<6} {sensor_name:<15} "
                  f"{info['activity_count']:>4}/{n_files:<3} "
                  f"{info['activity_percentage']:>5.1f}%  {bar}")

    print("\n" + "="*80 + "\n")


def print_presence_results(results, threshold=95.0):
    """
    Print sensor presence analysis — which Arduinos were physically logging.

    Presence = the sensor's columns exist in the CSV.
    This is independent of whether the data varied (activity).

    Sections:
      1. Presence matrix — sensors × classes (presence %)
      2. Sensors present in ALL classes above threshold
      3. Per-class presence ranking (how many sensors were plugged in per class)
    """
    classes = sorted(results['classes'].keys())
    all_sensors = sorted(results['sensors'].keys())

    if not classes:
        print("No class information found.")
        return

    print("\n" + "="*80)
    print("SENSOR PRESENCE ANALYSIS  (which Arduinos were physically logging)")
    print("="*80)
    print("  'Present' = sensor columns exist in the CSV file, regardless of signal quality.")
    print("  'Active'  = present AND producing non-constant signal (shown in activity tables).")

    # ── 1. Presence matrix ───────────────────────────────────────────────────
    print("\n" + "-"*80)
    print("SENSOR × CLASS PRESENCE MATRIX  (% of runs where sensor was logging)")
    print("-"*80)

    col_w = 10
    header = f"{'Sensor':<15}" + "".join(f"{cls:>{col_w}}" for cls in classes) + f"{'GLOBAL':>{col_w}}"
    print(header)
    print("-" * len(header))

    # Sort sensors by global presence % descending
    sorted_sensors = sorted(
        all_sensors,
        key=lambda s: results['sensors'][s]['presence_percentage'],
        reverse=True,
    )

    for sensor_name in sorted_sensors:
        global_pct = results['sensors'][sensor_name]['presence_percentage']
        row = f"{sensor_name:<15}"
        for cls in classes:
            cls_info = results['classes'][cls]['sensors'].get(sensor_name)
            if cls_info is None:
                row += f"{'—':>{col_w}}"
            else:
                pct = cls_info['presence_percentage']
                row += f"{pct:>{col_w - 1}.1f}%"
        row += f"{global_pct:>{col_w - 1}.1f}%"
        print(row)

    # ── 2. Sensors present in ALL classes ────────────────────────────────────
    print("\n" + "-"*80)
    print(f"SENSORS PRESENT IN ALL CLASSES  (≥{threshold}% presence in every class)")
    print("-"*80)

    universal_present = []
    partial_present = []   # meets threshold in some but not all classes
    low_present = []       # never meets threshold in any class

    for sensor_name in sorted_sensors:
        per_class_pcts = {
            cls: results['classes'][cls]['sensors'].get(sensor_name, {}).get('presence_percentage', 0.0)
            for cls in classes
        }
        n_classes_ok = sum(1 for p in per_class_pcts.values() if p >= threshold)

        if n_classes_ok == len(classes):
            universal_present.append((sensor_name, per_class_pcts))
        elif n_classes_ok > 0:
            partial_present.append((sensor_name, per_class_pcts, n_classes_ok))
        else:
            low_present.append((sensor_name, per_class_pcts))

    if universal_present:
        print(f"\n  Reliably present everywhere ({len(universal_present)} sensors):\n")
        for sensor_name, pcts in universal_present:
            min_pct = min(pcts.values())
            print(f"    • {sensor_name:<15}  min across classes: {min_pct:.1f}%")
    else:
        print(f"\n  No sensor meets ≥{threshold}% presence in ALL 9 classes.")

    if partial_present:
        print(f"\n  Present in SOME classes ({len(partial_present)} sensors):\n")
        for sensor_name, pcts, n_ok in partial_present:
            missing_classes = [cls for cls, p in pcts.items() if p < threshold]
            print(f"    • {sensor_name:<15}  {n_ok}/{len(classes)} classes  — weak in: {', '.join(missing_classes)}")

    if low_present:
        print(f"\n  Rarely/never present (<{threshold}% in any class — consider excluding):\n")
        for sensor_name, pcts in low_present:
            best_cls = max(pcts, key=pcts.get)
            print(f"    • {sensor_name:<15}  best class: {best_cls} ({pcts[best_cls]:.1f}%)")

    # ── 3. Per-class presence summary ────────────────────────────────────────
    print("\n" + "-"*80)
    print("PER-CLASS PRESENCE SUMMARY  (how many sensors were logging per class)")
    print("-"*80)

    for cls in classes:
        n_files = results['classes'][cls]['total_files']
        cls_sensors = results['classes'][cls]['sensors']

        # Count sensors that were present in ALL runs of this class
        always_present = [s for s, info in cls_sensors.items() if info['presence_percentage'] == 100.0]
        sometimes_present = [s for s, info in cls_sensors.items() if 0 < info['presence_percentage'] < 100.0]
        total_seen = len(cls_sensors)

        print(f"\n  [{cls}]  ({n_files} runs,  {total_seen} sensors seen)")
        print(f"    Always present (100%): {len(always_present):>2}  — {', '.join(sorted(always_present))}")
        if sometimes_present:
            print(f"    Sometimes present:     {len(sometimes_present):>2}  — "
                  + ", ".join(f"{s}({results['classes'][cls]['sensors'][s]['presence_percentage']:.0f}%)"
                               for s in sorted(sometimes_present)))

    print("\n" + "="*80 + "\n")


def print_sensor_type_results(results, threshold=95.0):
    """
    Print sensor-type analysis across positions and classes.

    Sections:
      1. Overall ranking  — sensor types by activity % across all positions × classes
      2. Sensor type × class matrix
      3. Sensor type × position matrix  (mean across classes)
      4. Universally common types       — above threshold in every class AND position
    """
    sensor_types = results.get('sensor_types', {})
    if not sensor_types:
        print("No sensor-type information found.")
        return

    classes = sorted({
        cls
        for st in sensor_types.values()
        for cls in st['classes']
    })
    positions = sorted({
        pos
        for st in sensor_types.values()
        for pos in st['positions']
    })

    # Sort sensor types by overall activity (best first)
    ranked_types = sorted(
        sensor_types.items(),
        key=lambda x: x[1]['overall_activity_percentage'],
        reverse=True
    )

    print("\n" + "="*80)
    print("SENSOR TYPE ANALYSIS  (across positions & classes)")
    print("="*80)

    # ── 1. Overall ranking ──────────────────────────────────────────────────
    print("\n" + "-"*80)
    print("OVERALL SENSOR TYPE RANKING  (activity % across all positions × classes)")
    print("-"*80)
    print(f"  {'Rank':<5} {'Sensor Type':<15} {'Activity':>9}  {'Bar'}")
    print(f"  {'-'*55}")
    for rank, (stype, info) in enumerate(ranked_types, 1):
        pct = info['overall_activity_percentage']
        bar = '█' * int(pct / 5)
        print(f"  {rank:<5} {stype:<15} {pct:>8.1f}%  {bar}")

    # ── 2. Sensor type × class matrix ───────────────────────────────────────
    print("\n" + "-"*80)
    print("SENSOR TYPE × CLASS MATRIX  (activity % averaged across positions)")
    print("-"*80)

    col_w = 11
    header = f"  {'Sensor Type':<15}" + "".join(f"{cls:>{col_w}}" for cls in classes) + f"{'MEAN':>{col_w}}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for stype, info in ranked_types:
        row = f"  {stype:<15}"
        for cls in classes:
            pct = info['classes'].get(cls, {}).get('activity_percentage', None)
            if pct is None:
                row += f"{'—':>{col_w}}"
            else:
                row += f"{pct:>{col_w - 1}.1f}%"
        mean_pct = info['overall_activity_percentage']
        row += f"{mean_pct:>{col_w - 1}.1f}%"
        print(row)

    # ── 3. Sensor type × position matrix ────────────────────────────────────
    print("\n" + "-"*80)
    print("SENSOR TYPE × POSITION MATRIX  (mean activity % across classes)")
    print("-"*80)

    pos_col_w = 12
    header2 = f"  {'Sensor Type':<15}" + "".join(f"{pos:>{pos_col_w}}" for pos in positions)
    print(header2)
    print("  " + "-" * (len(header2) - 2))

    for stype, info in ranked_types:
        row = f"  {stype:<15}"
        for pos in positions:
            pos_info = info['positions'].get(pos, None)
            if pos_info is None:
                row += f"{'—':>{pos_col_w}}"
            else:
                pct = pos_info['mean_activity_percentage']
                row += f"{pct:>{pos_col_w - 1}.1f}%"
        print(row)

    # ── 4. Common sensor types ───────────────────────────────────────────────
    print("\n" + "-"*80)
    print(f"COMMON SENSOR TYPES  (≥{threshold}% in EVERY class AND EVERY position)")
    print("-"*80)

    universal = []
    partial_class = []   # meets threshold in all positions, some classes
    partial_pos = []     # meets threshold in all classes, some positions

    for stype, info in ranked_types:
        # Check every class (averaged across positions)
        class_ok = all(
            info['classes'].get(cls, {}).get('activity_percentage', 0.0) >= threshold
            for cls in classes
        )
        # Check every position (averaged across classes)
        pos_ok = all(
            info['positions'].get(pos, {}).get('mean_activity_percentage', 0.0) >= threshold
            for pos in positions
        )

        if class_ok and pos_ok:
            universal.append(stype)
        elif class_ok:
            partial_pos.append(stype)
        elif pos_ok:
            partial_class.append(stype)

    if universal:
        print(f"\n  Universally reliable ({len(universal)} types — safe for all classes & positions):")
        for stype in universal:
            print(f"    • {stype:<15}  overall: {sensor_types[stype]['overall_activity_percentage']:.1f}%")
    else:
        print(f"\n  None meet ≥{threshold}% in EVERY class AND position simultaneously.")

    if partial_class:
        print(f"\n  Active in all positions but NOT all classes ({len(partial_class)} types):")
        for stype in partial_class:
            weak_classes = [
                cls for cls in classes
                if sensor_types[stype]['classes'].get(cls, {}).get('activity_percentage', 0.0) < threshold
            ]
            print(f"    • {stype:<15}  weak in: {', '.join(weak_classes)}")

    if partial_pos:
        print(f"\n  Active in all classes but NOT all positions ({len(partial_pos)} types):")
        for stype in partial_pos:
            weak_pos = [
                pos for pos in positions
                if sensor_types[stype]['positions'].get(pos, {}).get('mean_activity_percentage', 0.0) < threshold
            ]
            print(f"    • {stype:<15}  weak at: {', '.join(weak_pos)}")

    print("\n" + "="*80 + "\n")


def print_type_position_reliability(results, threshold=95.0):
    """
    Answer: which (sensor_type, position) combinations are safe to use
    as features across ALL 9 classes?

    Key metric: minimum activity % across all 9 classes for each (type, position) pair.
    A min of 0% means that combination is completely missing in at least one class.

    Sections:
      1. Reliability matrix  — sensor_type × position, cell = min(activity%) across classes
      2. Recommended feature combinations  — (type, pos) pairs above threshold in every class
      3. Sensor type ranking by number of reliable positions
    """
    sensor_types = results.get('sensor_types', {})
    if not sensor_types:
        return

    classes = sorted(results['classes'].keys())
    all_positions = sorted({
        pos
        for st in sensor_types.values()
        for pos in st['positions']
    })
    ranked_types = sorted(
        sensor_types.items(),
        key=lambda x: x[1]['overall_activity_percentage'],
        reverse=True,
    )

    # Compute min activity % across all classes for each (type, pos) pair.
    # If a class has no data for (type, pos) → 0.0 (position never appeared).
    def min_pct(stype, pos):
        pos_info = sensor_types[stype]['positions'].get(pos)
        if pos_info is None:
            return 0.0
        return min(
            pos_info['classes'].get(cls, {}).get('activity_percentage', 0.0)
            for cls in classes
        )

    print("\n" + "="*80)
    print("SENSOR TYPE × POSITION RELIABILITY")
    print("  Each cell = MINIMUM activity % across all 9 classes.")
    print("  0% means that combination is unavailable in at least one class.")
    print("="*80)

    # ── 1. Reliability matrix ────────────────────────────────────────────────
    print("\n" + "-"*80)
    print(f"RELIABILITY MATRIX  (min activity % across all {len(classes)} classes)")
    print("-"*80)

    pos_col_w = 12
    header = f"  {'Sensor Type':<15}" + "".join(f"{pos:>{pos_col_w}}" for pos in all_positions)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for stype, _ in ranked_types:
        row = f"  {stype:<15}"
        for pos in all_positions:
            m = min_pct(stype, pos)
            if sensor_types[stype]['positions'].get(pos) is None:
                row += f"{'—':>{pos_col_w}}"
            elif m == 0.0:
                # Present at this position in some classes but 0% in at least one
                row += f"{'✗ 0%':>{pos_col_w}}"
            else:
                row += f"{m:>{pos_col_w - 1}.1f}%"
        print(row)

    print(f"\n  Legend:  ✗ 0% = unavailable in ≥1 class   —  = position never seen with this type")

    # ── 2. Recommended (type, pos) combinations ──────────────────────────────
    print("\n" + "-"*80)
    print(f"RECOMMENDED (type, position) COMBINATIONS  (≥{threshold}% in ALL {len(classes)} classes)")
    print("-"*80)

    recommended = {}   # stype → list of reliable positions
    for stype, _ in ranked_types:
        reliable_pos = [pos for pos in all_positions if min_pct(stype, pos) >= threshold]
        recommended[stype] = reliable_pos

    any_found = any(v for v in recommended.values())
    if any_found:
        print(f"\n  These feature combinations have ≥{threshold}% signal in every class:\n")
        for stype, reliable_pos in recommended.items():
            if reliable_pos:
                n_channels = {'Accelerometer': 3, 'Gyroscope': 3, 'Magnetometer': 3,
                              'Color': 4, 'Pressure': 1, 'Temperature': 1,
                              'Proximity': 1, 'RMS': 1}.get(stype, 1)
                total_features = len(reliable_pos) * n_channels
                print(f"    {stype:<15} → {len(reliable_pos)} positions × {n_channels} ch = {total_features} features")
                for pos in reliable_pos:
                    m = min_pct(stype, pos)
                    print(f"        • {pos:<15}  (min: {m:.1f}%)")
    else:
        print(f"\n  No (type, position) combination meets ≥{threshold}% in all classes.")
        print("  Try lowering --threshold.")

    # ── 3. Sensor type ranking by coverage ───────────────────────────────────
    print("\n" + "-"*80)
    print(f"SENSOR TYPE RANKING BY POSITION COVERAGE  (how many positions meet ≥{threshold}%)")
    print("-"*80)
    print(f"\n  {'Rank':<5} {'Sensor Type':<15} {'Reliable positions':>20}  {'Positions'}")
    print(f"  {'-'*70}")
    for rank, (stype, _) in enumerate(ranked_types, 1):
        reliable = recommended[stype]
        bar = '█' * len(reliable)
        print(f"  {rank:<5} {stype:<15} {len(reliable):>20}  {bar}  {', '.join(reliable) if reliable else '—'}")

    # Total unique features from all reliable (type, pos) combinations
    ch_map = {'Accelerometer': 3, 'Gyroscope': 3, 'Magnetometer': 3,
              'Color': 4, 'Pressure': 1, 'Temperature': 1, 'Proximity': 1, 'RMS': 1}
    total = sum(len(pos_list) * ch_map.get(stype, 1) for stype, pos_list in recommended.items())
    print(f"\n  Total reliable features across all types: {total} channels")
    print(f"  (before adding electrical/environment signals)")

    print("\n" + "="*80 + "\n")


def print_sensor_type_coverage(results, min_runs=100):
    """
    Given positions that meet a minimum presence threshold, shows which sensor
    types are reliably active at ALL of those positions across all classes.

    Cell = active_runs / present_runs  (across all classes for that position).

    This answers: "within my chosen positions, which sensor types can I use?"
    """
    total_runs = results['total_files']
    classes = sorted(results['classes'].keys())
    sensor_types = results.get('sensor_types', {})

    # Filter positions by minimum presence count
    selected = sorted(
        [p for p, info in results['sensors'].items()
         if info['presence_count'] >= min_runs],
        key=lambda p: results['sensors'][p]['presence_count'],
        reverse=True,
    )

    if not selected:
        print(f"No positions with ≥{min_runs} runs. Lower --min-runs.")
        return

    ranked_types = sorted(
        sensor_types.items(),
        key=lambda x: x[1]['overall_activity_percentage'],
        reverse=True,
    )

    print("\n" + "="*80)
    print(f"SENSOR TYPE COVERAGE  (positions with ≥{min_runs}/{total_runs} runs present)")
    print("  Cell = active_runs / present_runs  (summed across all 9 classes).")
    print("  'ALL?' = sensor type is reliably active at EVERY selected position.")
    print("="*80)

    print(f"\n  {len(selected)} selected positions:")
    for pos in selected:
        print(f"    • {pos:<15}  {results['sensors'][pos]['presence_count']}/{total_runs} runs")

    # ── Activity table ───────────────────────────────────────────────────────
    print()
    pos_col_w = 14
    header = f"  {'Sensor Type':<15}" + "".join(f"{p:>{pos_col_w}}" for p in selected) + f"  ALL?"
    print(header)
    print("  " + "-" * len(header))

    ACTIVITY_THRESHOLD = 90.0  # % of present runs that must be active to count as reliable

    def cell_stats(stype, pos):
        """Returns (active_total, present_total) summed across all classes."""
        pos_info = sensor_types.get(stype, {}).get('positions', {}).get(pos)
        if pos_info is None:
            return None, None
        active = sum(pos_info['classes'].get(cls, {}).get('active', 0) for cls in classes)
        present = sum(pos_info['classes'].get(cls, {}).get('total', 0) for cls in classes)
        return active, present

    reliable_types = []

    for stype, _ in ranked_types:
        row = f"  {stype:<15}"
        type_ok = True
        for pos in selected:
            active, present = cell_stats(stype, pos)
            if present is None or present == 0:
                row += f"{'—':>{pos_col_w}}"
                type_ok = False
            else:
                pct = active / present * 100
                cell = f"{active}/{present}"
                if pct < ACTIVITY_THRESHOLD:
                    type_ok = False
                row += f"{cell:>{pos_col_w}}"
        row += f"  {'✓' if type_ok else '✗'}"
        print(row)
        if type_ok:
            reliable_types.append(stype)

    # ── Recommendation ───────────────────────────────────────────────────────
    ch_map = {'Accelerometer': 3, 'Gyroscope': 3, 'Magnetometer': 3,
              'Color': 4, 'Temperature': 1, 'Proximity': 1, 'Pressure': 1, 'RMS': 1}

    unreliable = [t for t, _ in ranked_types if t not in reliable_types]

    print("\n" + "-"*80)
    print(f"RECOMMENDATION  (threshold: ≥{ACTIVITY_THRESHOLD:.0f}% active when present)")
    print("-"*80)

    if reliable_types:
        print(f"\n  USE — reliable at all {len(selected)} positions:\n")
        total_features = 0
        for t in reliable_types:
            n_ch = ch_map.get(t, 1)
            n_feat = n_ch * len(selected)
            total_features += n_feat
            print(f"    ✓ {t:<15}  {n_ch} ch × {len(selected)} positions = {n_feat:>4} features")
        print(f"\n    Subtotal: {total_features} sensor features")
        print(f"    + electrical/environment signals if included")

    if unreliable:
        print(f"\n  AVOID — not reliable across all selected positions:\n")
        for t in unreliable:
            weak_positions = []
            for pos in selected:
                active, present = cell_stats(t, pos)
                if present is None or present == 0:
                    weak_positions.append(f"{pos}(absent)")
                elif active / present * 100 < ACTIVITY_THRESHOLD:
                    weak_positions.append(f"{pos}({active}/{present})")
            print(f"    ✗ {t:<15}  weak at: {', '.join(weak_positions)}")

    print("\n" + "="*80 + "\n")


def print_availability_matrix(results):
    """
    Simple answer: for each class × position, how many runs is that position present?

    When a position (Arduino) is present it always carries all 8 sensor types:
      Accelerometer (3ch)  Gyroscope (3ch)  Magnetometer (3ch)
      Color (4ch)  Temperature (1ch)  Proximity (1ch)  Pressure (1ch)  RMS (1ch)
      = 17 channels total

    So presence count = sensor type availability count.
    """
    classes = sorted(results['classes'].keys())
    # Sort positions by total presence count descending
    all_positions = sorted(
        results['sensors'].keys(),
        key=lambda p: results['sensors'][p]['presence_count'],
        reverse=True,
    )

    print("\n" + "="*80)
    print("SENSOR AVAILABILITY MATRIX")
    print("  Cell = runs present / total runs for that class.")
    print("  When a position is present → all 8 sensor types are available (17 channels).")
    print("="*80)

    # Column widths: class names can be long
    col_w = max(len(c) for c in classes) + 2  # at least 2 padding

    header = f"{'Position':<15}" + "".join(f"{c:>{col_w}}" for c in classes) + f"{'TOTAL':>10}"
    print("\n" + header)
    print("-" * len(header))

    total_runs = results['total_files']
    total_per_class = {cls: results['classes'][cls]['total_files'] for cls in classes}

    for pos in all_positions:
        global_count = results['sensors'][pos]['presence_count']
        row = f"{pos:<15}"
        for cls in classes:
            cls_info = results['classes'][cls]['sensors'].get(pos)
            n_total = total_per_class[cls]
            if cls_info is None:
                count = 0
            else:
                count = cls_info['presence_count']
            cell = f"{count}/{n_total}"
            row += f"{cell:>{col_w}}"
        row += f"{global_count:>5}/{total_runs:<4}"
        print(row)

    print()

    # Summary: classify positions into safe / partial / rare
    safe, partial, rare = [], [], []
    for pos in all_positions:
        per_class_counts = {
            cls: results['classes'][cls]['sensors'].get(pos, {}).get('presence_count', 0)
            for cls in classes
        }
        # Safe = present in every run of every class
        if all(per_class_counts[cls] == total_per_class[cls] for cls in classes):
            safe.append(pos)
        # Rare = never present in some classes (0 runs in ≥1 class)
        elif any(per_class_counts[cls] == 0 for cls in classes):
            rare.append(pos)
        else:
            partial.append(pos)

    print("-"*80)
    print("SUMMARY")
    print("-"*80)

    ch_per_type = {'Accelerometer': 3, 'Gyroscope': 3, 'Magnetometer': 3,
                   'Color': 4, 'Temperature': 1, 'Proximity': 1, 'Pressure': 1, 'RMS': 1}
    channels_per_position = sum(ch_per_type.values())   # 17

    if safe:
        print(f"\n  SAFE — present in every run of every class ({len(safe)} positions):")
        for pos in safe:
            print(f"    • {pos}  →  {channels_per_position} channels always available")
        print(f"\n    Combined: {len(safe)} × {channels_per_position} = {len(safe)*channels_per_position} channels")

    if partial:
        print(f"\n  PARTIAL — present in all classes but not all runs ({len(partial)} positions):")
        for pos in partial:
            global_count = results['sensors'][pos]['presence_count']
            worst_cls = min(classes,
                key=lambda c: results['classes'][c]['sensors'].get(pos, {}).get('presence_count', 0))
            worst_n = results['classes'][worst_cls]['sensors'].get(pos, {}).get('presence_count', 0)
            worst_total = total_per_class[worst_cls]
            print(f"    • {pos:<15}  {global_count}/{total_runs} globally   "
                  f"worst class: {worst_cls} ({worst_n}/{worst_total})")

    if rare:
        print(f"\n  RARE — completely absent from ≥1 class ({len(rare)} positions — avoid for cross-class models):")
        for pos in rare:
            missing = [c for c in classes
                       if results['classes'][c]['sensors'].get(pos, {}).get('presence_count', 0) == 0]
            print(f"    • {pos:<15}  absent in: {', '.join(missing)}")

    print("\n" + "="*80 + "\n")


def save_results(results, output_path):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Identify sensors that are consistently active across all runs"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/',
        help='Directory containing CSV files (default: data/)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=95.0,
        help='Minimum activity percentage to consider sensor consistent (default: 95.0)'
    )
    parser.add_argument(
        '--activity-threshold',
        type=float,
        default=1e-6,
        help='Minimum std deviation to consider sensor active (default: 1e-6)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/sensor_consistency_report.json',
        help='Output JSON file path (default: outputs/sensor_consistency_report.json)'
    )
    parser.add_argument(
        '--min-runs',
        type=int,
        default=100,
        help='Minimum total runs a position must be present to be included in '
             'sensor type coverage analysis (default: 100)'
    )

    args = parser.parse_args()

    # Analyze sensor consistency
    results = analyze_sensor_consistency(
        args.data_dir,
        activity_threshold=args.activity_threshold,
        verbose=True
    )

    # Print results
    print_availability_matrix(results)
    print_sensor_type_coverage(results, min_runs=args.min_runs)
    print_results(results, threshold=args.threshold)
    print_presence_results(results, threshold=args.threshold)
    print_class_results(results, threshold=args.threshold)
    print_sensor_type_results(results, threshold=args.threshold)
    print_type_position_reliability(results, threshold=args.threshold)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_results(results, args.output)

    # Print recommendation
    consistent_sensors = [
        name for name, info in results['sensors'].items()
        if info['activity_percentage'] >= args.threshold
    ]

    print("RECOMMENDATION:")
    print("="*80)
    print("\nFor preprocessing without zero-padding, use these sensors:")
    print(f"\nSENSORS = {consistent_sensors}")
    print(f"\n# This gives {sum(results['sensors'][s]['num_channels'] for s in consistent_sensors)} sensor features + 8 electrical = "
          f"{sum(results['sensors'][s]['num_channels'] for s in consistent_sensors) + 8} total features")
    print("\nAll files have active signal from these sensors, eliminating the need for padding.")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()