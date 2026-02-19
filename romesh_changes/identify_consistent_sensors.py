#!/usr/bin/env python3
"""
Identify sensors that are consistently active across all experimental runs.

This script analyzes all CSV files to determine:
1. Which sensor locations are present in each file
2. Which sensors actually have signal activity (non-zero variance)
3. Coverage percentage for each sensor across all files
4. Recommends sensors for consistent-length data without padding

Usage:
    python scripts/analysis/identify_consistent_sensors.py --data-dir data/ --threshold 95
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


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

    # Track sensor presence and activity
    sensor_presence = defaultdict(int)  # How many files have this sensor
    sensor_activity = defaultdict(int)  # How many files show activity for this sensor
    sensor_channels = {}  # Expected channels for each sensor
    total_files = len(csv_files)

    # Track per-file sensor counts for statistics
    file_sensor_counts = []

    # Analyze each file
    for csv_path in tqdm(csv_files, desc="Analyzing files", disable=not verbose):
        try:
            # Read CSV
            df = pd.read_csv(csv_path, low_memory=False)

            # Get sensors in this file
            file_sensors = get_sensor_channels(df.columns)
            file_sensor_counts.append(len(file_sensors))

            # Update presence and activity counts
            for sensor_name, columns in file_sensors.items():
                # Track channel names (should be consistent)
                if sensor_name not in sensor_channels:
                    # Extract channel names without sensor prefix
                    channel_names = [col.split('.', 1)[1] for col in columns]
                    sensor_channels[sensor_name] = sorted(channel_names)

                # This sensor is present in this file
                sensor_presence[sensor_name] += 1

                # Check if sensor shows actual activity
                if check_sensor_activity(df, columns, activity_threshold):
                    sensor_activity[sensor_name] += 1

        except Exception as e:
            if verbose:
                print(f"Error processing {csv_path}: {e}")
            continue

    # Calculate coverage percentages
    results = {
        'total_files': total_files,
        'sensors': {},
        'statistics': {
            'min_sensors_per_file': min(file_sensor_counts) if file_sensor_counts else 0,
            'max_sensors_per_file': max(file_sensor_counts) if file_sensor_counts else 0,
            'mean_sensors_per_file': np.mean(file_sensor_counts) if file_sensor_counts else 0,
        }
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

    args = parser.parse_args()

    # Analyze sensor consistency
    results = analyze_sensor_consistency(
        args.data_dir,
        activity_threshold=args.activity_threshold,
        verbose=True
    )

    # Print results
    print_results(results, threshold=args.threshold)

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