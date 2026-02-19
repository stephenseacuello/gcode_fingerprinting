#!/usr/bin/env python3
"""
Analyze electrical features to check if they are operation-specific.

This checks if the 8 electrical features have operation-specific patterns
that would allow tree models to easily classify operations.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json


# Electrical features to analyze
ELECTRICAL_FEATURES = [
    'spindle', 'x_motor', 'y_motor', 'z_motor',
    'spindle_A', 'x_motor_A', 'y_motor_A', 'z_motor_A'
]


def extract_operation_type(filename):
    """Extract operation type from filename."""
    fname_lower = filename.lower()

    # Check specific patterns first
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


def analyze_electrical_features(data_dir):
    """Analyze electrical features across operations."""

    csv_files = sorted(Path(data_dir).glob("*_aligned.csv"))
    if not csv_files:
        csv_files = sorted(Path(data_dir).glob("*.csv"))

    print(f"Found {len(csv_files)} CSV files\n")

    # Collect statistics per operation
    stats_by_operation = defaultdict(lambda: {
        feat: {'values': [], 'files': 0}
        for feat in ELECTRICAL_FEATURES
    })

    # Process each file
    for csv_file in csv_files:
        operation = extract_operation_type(csv_file.name)
        df = pd.read_csv(csv_file)

        for feat in ELECTRICAL_FEATURES:
            if feat in df.columns:
                values = df[feat].values
                # Remove NaN values
                values = values[~np.isnan(values)]

                if len(values) > 0:
                    stats_by_operation[operation][feat]['values'].extend(values)
                    stats_by_operation[operation][feat]['files'] += 1

    # Calculate statistics for each operation + feature
    print("=" * 120)
    print("ELECTRICAL FEATURES BY OPERATION TYPE")
    print("=" * 120)
    print()

    for feat in ELECTRICAL_FEATURES:
        print(f"\n{'='*120}")
        print(f"FEATURE: {feat}")
        print('='*120)
        print(f"{'Operation':<20} {'Files':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Range':<12}")
        print('-'*120)

        feature_stats = {}

        for operation in sorted(stats_by_operation.keys()):
            values = np.array(stats_by_operation[operation][feat]['values'])
            files = stats_by_operation[operation][feat]['files']

            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                range_val = max_val - min_val

                feature_stats[operation] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'range': range_val,
                    'files': files
                }

                print(f"{operation:<20} {files:<8} {mean_val:>11.3f} {std_val:>11.3f} "
                      f"{min_val:>11.3f} {max_val:>11.3f} {range_val:>11.3f}")

        # Check if means are separable
        if len(feature_stats) > 1:
            means = [s['mean'] for s in feature_stats.values()]
            mean_range = max(means) - min(means)
            avg_std = np.mean([s['std'] for s in feature_stats.values()])

            # If mean range >> average std, means are well-separated
            separation_ratio = mean_range / avg_std if avg_std > 0 else 0

            print('-'*120)
            print(f"Mean separation: {mean_range:.3f} | Avg std: {avg_std:.3f} | Ratio: {separation_ratio:.2f}")

            if separation_ratio > 2.0:
                print(f"⚠️  HIGHLY SEPARABLE - Trees can easily distinguish operations using {feat}!")
            elif separation_ratio > 1.0:
                print(f"⚠️  MODERATELY SEPARABLE - {feat} provides discrimination")
            else:
                print(f"✅ Low separation - {feat} similar across operations")

    # Summary
    print("\n" + "="*120)
    print("SUMMARY: Which electrical features are operation-specific?")
    print("="*120)
    print()

    problematic_features = []

    for feat in ELECTRICAL_FEATURES:
        feature_stats = {}

        for operation in stats_by_operation.keys():
            values = np.array(stats_by_operation[operation][feat]['values'])
            if len(values) > 0:
                feature_stats[operation] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }

        if len(feature_stats) > 1:
            means = [s['mean'] for s in feature_stats.values()]
            mean_range = max(means) - min(means)
            avg_std = np.mean([s['std'] for s in feature_stats.values()])
            separation_ratio = mean_range / avg_std if avg_std > 0 else 0

            if separation_ratio > 2.0:
                problematic_features.append((feat, separation_ratio))

    if problematic_features:
        print("⚠️  PROBLEMATIC ELECTRICAL FEATURES (operation-specific):\n")
        problematic_features.sort(key=lambda x: -x[1])

        for feat, ratio in problematic_features:
            print(f"  • {feat:<15} - Separation ratio: {ratio:.2f}")

        print(f"\n{len(problematic_features)}/{len(ELECTRICAL_FEATURES)} electrical features are operation-specific!")
        print("\nThese features allow tree models to perfectly separate operations,")
        print("explaining the 100% accuracy on 7/9 classes.")
    else:
        print("✅ No electrical features are strongly operation-specific")

    print("\n" + "="*120)

    # Save detailed stats
    output_file = Path('outputs/electrical_features_by_operation.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable_stats = {}
    for operation, feats in stats_by_operation.items():
        serializable_stats[operation] = {}
        for feat, data in feats.items():
            values = np.array(data['values'])
            if len(values) > 0:
                serializable_stats[operation][feat] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'files': data['files']
                }

    with open(output_file, 'w') as f:
        json.dump(serializable_stats, f, indent=2)

    print(f"\nDetailed statistics saved to: {output_file}")
    print()


if __name__ == '__main__':
    data_dir = Path('data')
    analyze_electrical_features(data_dir)
