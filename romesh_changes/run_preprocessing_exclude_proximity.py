#!/usr/bin/env python3
"""
Preprocess data with FILE-LEVEL splitting + EXCLUDE y_bed__3.Proximity.

This tests the hypothesis that y_bed__3.Proximity is the main feature
allowing trees to achieve 96% accuracy.

Key changes from run_preprocessing_file_level_split.py:
- Excludes y_bed__3.Proximity channel (100% constant in 4 operations)
- Optionally exclude other problematic channels

Usage:
    python romesh_changes/run_preprocessing_exclude_proximity.py \
        --data-dir data/ \
        --output-dir outputs/experiments/file_level_split_no_proximity/preprocessed \
        --vocab-path outputs/vocabulary/gcode_vocabulary_v2.json
"""
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.miracle.dataset.preprocessing import GCodePreprocessor
from src.miracle.config.preprocessing_config import PreprocessingConfig

# Sensors with ≥95% activity
CONSISTENT_SENSORS = ['frame_l2', 'frame_r2', 'spindle2', 'y_bed__3', 'y_bed__4']

# Electrical features
ELECTRICAL_FEATURES = [
    'spindle', 'x_motor', 'y_motor', 'z_motor',
    'spindle_A', 'x_motor_A', 'y_motor_A', 'z_motor_A'
]

# PROBLEMATIC CHANNELS TO EXCLUDE

# All Proximity sensors (dead in many files)
ALL_PROXIMITY_CHANNELS = [
    'frame_l2.Proximity',
    'frame_r2.Proximity',
    'spindle2.Proximity',
    'y_bed__3.Proximity',
    'y_bed__4.Proximity',
]

# All Color sensors (RGBA - not core motion/vibration sensors)
ALL_COLOR_CHANNELS = [
    'frame_l2.ColorR', 'frame_l2.ColorG', 'frame_l2.ColorB', 'frame_l2.ColorA',
    'frame_r2.ColorR', 'frame_r2.ColorG', 'frame_r2.ColorB', 'frame_r2.ColorA',
    'spindle2.ColorR', 'spindle2.ColorG', 'spindle2.ColorB', 'spindle2.ColorA',
    'y_bed__3.ColorR', 'y_bed__3.ColorG', 'y_bed__3.ColorB', 'y_bed__3.ColorA',
    'y_bed__4.ColorR', 'y_bed__4.ColorG', 'y_bed__4.ColorB', 'y_bed__4.ColorA',
]

# All Magnetometer sensors (Mx, My, Mz - identified as operation-specific shortcuts)
ALL_MAGNETOMETER_CHANNELS = [
    'frame_l2.Mx', 'frame_l2.My', 'frame_l2.Mz',
    'frame_r2.Mx', 'frame_r2.My', 'frame_r2.Mz',
    'spindle2.Mx', 'spindle2.My', 'spindle2.Mz',
    'y_bed__3.Mx', 'y_bed__3.My', 'y_bed__3.Mz',
    'y_bed__4.Mx', 'y_bed__4.My', 'y_bed__4.Mz',
]

# Additional problematic Pressure sensors (optional)
ADDITIONAL_PROBLEMATIC = [
    'y_bed__3.Pressure',   # 89.8% constant in face150025
    'frame_l2.Pressure',   # 79.2% constant in face150025
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


def stratified_file_split(csv_files, train_frac=0.7, val_frac=0.15, seed=42):
    """
    Split files by operation type (stratified) to maintain class balance.

    Returns:
        train_files, val_files, test_files
    """
    # Group files by operation type
    files_by_operation = defaultdict(list)
    for csv_file in csv_files:
        operation = extract_operation_type(csv_file.name)
        files_by_operation[operation].append(csv_file)

    train_files = []
    val_files = []
    test_files = []

    print("\n" + "="*80)
    print("FILE-LEVEL STRATIFIED SPLIT")
    print("="*80)
    print(f"Total files: {len(csv_files)}")
    print(f"Split: {train_frac:.0%} train, {val_frac:.0%} val, {1-train_frac-val_frac:.0%} test")
    print("\nPer-operation split:")
    print("-"*80)

    for operation in sorted(files_by_operation.keys()):
        files = sorted(files_by_operation[operation])
        n_files = len(files)

        if n_files == 0:
            continue

        if n_files == 1:
            train_files.extend(files)
            print(f"{operation:<20} {n_files:>3} files  →  Train: 1  Val: 0  Test: 0  (too few files)")
            continue

        if n_files < 5:
            n_train = max(1, int(n_files * train_frac))
            n_val = 0
            train, test = train_test_split(
                files, train_size=n_train, random_state=seed, shuffle=True
            )
            val = []
        else:
            n_train = max(1, int(n_files * train_frac))
            n_val = max(1, int(n_files * val_frac))
            train, temp = train_test_split(
                files, train_size=n_train, random_state=seed, shuffle=True
            )
            val, test = train_test_split(
                temp, train_size=n_val, random_state=seed, shuffle=True
            )

        train_files.extend(train)
        val_files.extend(val)
        test_files.extend(test)

        print(f"{operation:<20} {n_files:>3} files  →  "
              f"Train: {len(train):<2}  Val: {len(val):<2}  Test: {len(test):<2}")

    print("-"*80)
    print(f"{'TOTAL':<20} {len(csv_files):>3} files  →  "
          f"Train: {len(train_files):<2}  Val: {len(val_files):<2}  Test: {len(test_files):<2}")
    print("="*80 + "\n")

    return train_files, val_files, test_files


def filter_master_columns_to_consistent_sensors(master_columns, consistent_sensors=None,
                                                  exclude_channels=None):
    """Filter master column list to only include consistent sensors and exclude problematic channels."""
    if consistent_sensors is None:
        consistent_sensors = CONSISTENT_SENSORS

    if exclude_channels is None:
        exclude_channels = []

    filtered_columns = []

    for col in master_columns:
        # Skip if in exclude list
        if col in exclude_channels:
            continue

        if col in ELECTRICAL_FEATURES:
            filtered_columns.append(col)
            continue

        if '.' in col:
            sensor_name = col.split('.', 1)[0]
            if sensor_name in consistent_sensors:
                filtered_columns.append(col)

    return filtered_columns


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess with file-level split + exclude problematic channels"
    )
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--vocab-path', type=Path, required=True)
    parser.add_argument('--sensor-report', type=Path,
                       default='outputs/sensor_consistency_report.json')
    parser.add_argument('--threshold', type=float, default=95.0)
    parser.add_argument('--window-size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--train-frac', type=float, default=0.7)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exclude-proximity', action='store_true',
                       help='Exclude all 5 Proximity channels')
    parser.add_argument('--exclude-color', action='store_true',
                       help='Exclude all Color (RGBA) channels')
    parser.add_argument('--exclude-magnetometer', action='store_true',
                       help='Exclude all Magnetometer (Mx, My, Mz) channels')
    parser.add_argument('--exclude-additional', action='store_true',
                       help='Also exclude additional problematic channels (Pressure sensors)')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which channels to exclude
    exclude_channels = []

    if args.exclude_proximity:
        exclude_channels.extend(ALL_PROXIMITY_CHANNELS)

    if args.exclude_color:
        exclude_channels.extend(ALL_COLOR_CHANNELS)

    if args.exclude_magnetometer:
        exclude_channels.extend(ALL_MAGNETOMETER_CHANNELS)

    if args.exclude_additional:
        exclude_channels.extend(ADDITIONAL_PROBLEMATIC)

    print("\n" + "="*80)
    print("EXCLUDED CHANNELS (Testing Hypothesis)")
    print("="*80)

    if args.exclude_proximity:
        print(f"\n❌ ALL PROXIMITY CHANNELS ({len(ALL_PROXIMITY_CHANNELS)} channels):")
        for ch in ALL_PROXIMITY_CHANNELS:
            print(f"   • {ch}")

    if args.exclude_color:
        print(f"\n❌ ALL COLOR CHANNELS ({len(ALL_COLOR_CHANNELS)} channels):")
        for ch in ALL_COLOR_CHANNELS:
            print(f"   • {ch}")

    if args.exclude_magnetometer:
        print(f"\n❌ ALL MAGNETOMETER CHANNELS ({len(ALL_MAGNETOMETER_CHANNELS)} channels):")
        for ch in ALL_MAGNETOMETER_CHANNELS:
            print(f"   • {ch}")

    if args.exclude_additional:
        print(f"\n❌ ADDITIONAL PROBLEMATIC ({len(ADDITIONAL_PROBLEMATIC)} channels):")
        for ch in ADDITIONAL_PROBLEMATIC:
            print(f"   • {ch}")

    print(f"\n📊 Total excluded: {len(exclude_channels)} channels")
    print(f"   Remaining features: 93 - {len(exclude_channels)} = {93 - len(exclude_channels)}")
    print("="*80 + "\n")

    # Load sensor consistency report
    consistent_sensors = CONSISTENT_SENSORS

    if args.sensor_report.exists():
        print(f"Loading sensor consistency report from: {args.sensor_report}")
        with open(args.sensor_report, 'r') as f:
            report = json.load(f)

        consistent_sensors = [
            name for name, info in report['sensors'].items()
            if info['activity_percentage'] >= args.threshold
        ]

        print(f"\nUsing {len(consistent_sensors)} sensors with ≥{args.threshold}% activity:")
        for sensor_name in sorted(consistent_sensors):
            info = report['sensors'][sensor_name]
            print(f"  • {sensor_name:<15} - {info['activity_percentage']:>5.1f}% active "
                  f"({info['num_channels']} channels)")

        total_channels = sum(report['sensors'][s]['num_channels'] for s in consistent_sensors)
        print(f"\nTotal: {total_channels} sensor features + 8 electrical = {total_channels + 8}")
        print(f"After exclusions: {total_channels + 8 - len(exclude_channels)} features\n")

    # Find CSV files
    csv_files = sorted(args.data_dir.glob("*_aligned.csv"))
    if not csv_files:
        csv_files = sorted(args.data_dir.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {args.data_dir}")

    print(f"Found {len(csv_files)} CSV files\n")

    # FILE-LEVEL SPLIT (KEY FIX FOR ISSUE #2)
    train_files, val_files, test_files = stratified_file_split(
        csv_files,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed
    )

    # Save file split info
    split_info = {
        'train_files': [f.name for f in train_files],
        'val_files': [f.name for f in val_files],
        'test_files': [f.name for f in test_files],
    }
    split_info_path = args.output_dir / 'file_split.json'
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved file split info to: {split_info_path}\n")

    # Create preprocessing config
    config = PreprocessingConfig(
        window_size=args.window_size,
        stride=args.stride,
        scaler_type='robust',
        nan_strategy='forward_fill',
        outlier_method='clip',
        remove_zero_variance=True,
        correlation_threshold=0.95,
        random_seed=args.seed,
    )

    # Build master column list
    print("="*80)
    print("BUILDING MASTER COLUMN LIST")
    print("="*80)

    all_continuous_cols = set()
    exclude_cols = config.exclude_features
    cat_cols_names = config.categorical_features

    for csv_path in csv_files:
        df = pd.read_csv(csv_path, nrows=1)
        for col in df.columns:
            if col not in exclude_cols and col not in cat_cols_names:
                if pd.api.types.is_numeric_dtype(df[col]):
                    all_continuous_cols.add(col)

    master_columns = sorted(list(all_continuous_cols))
    print(f"  Found {len(master_columns)} total features")

    # Filter to consistent sensors AND exclude problematic channels
    master_columns_filtered = filter_master_columns_to_consistent_sensors(
        master_columns, consistent_sensors, exclude_channels
    )
    print(f"  After filtering to consistent sensors: {len(master_columns_filtered)} features")
    print(f"  ✅ {len(exclude_channels)} problematic channels excluded\n")

    # Create preprocessor
    preprocessor = GCodePreprocessor(
        args.vocab_path,
        config=config,
        master_columns=master_columns_filtered
    )

    # FIT SCALER ONLY ON TRAINING FILES (KEY FIX)
    print("="*80)
    print("FITTING SCALER ON TRAINING DATA ONLY")
    print("="*80)

    train_continuous_data = []
    for csv_path in train_files:
        df = preprocessor.load_csv(csv_path)
        continuous, _, _ = preprocessor.extract_features(df)
        train_continuous_data.append(continuous)

    train_combined = np.vstack(train_continuous_data)
    preprocessor.fit_scaler(train_combined)
    print(f"  Scaler fitted on {train_combined.shape} training data points")
    print("  ✅ Test data NOT used in scaler fitting!\n")

    # Process each file set separately
    def process_file_set(files, set_name):
        print(f"Processing {set_name} files ({len(files)} files)...")
        windows = []
        for csv_path in files:
            file_windows = preprocessor.process_file(csv_path, fit_scaler=False)
            windows.extend(file_windows)
        print(f"  Total {set_name} windows: {len(windows)}\n")
        return windows

    print("="*80)
    print("CREATING WINDOWS FROM EACH FILE SET")
    print("="*80 + "\n")

    train_windows = process_file_set(train_files, "TRAIN")
    val_windows = process_file_set(val_files, "VAL")
    test_windows = process_file_set(test_files, "TEST")

    # Metadata
    continuous_shape = train_windows[0]['continuous'].shape
    categorical_shape = train_windows[0]['categorical'].shape

    metadata = {
        'n_continuous_features': continuous_shape[1],
        'n_categorical_features': categorical_shape[1],
        'window_size': args.window_size,
        'stride': args.stride,
        'vocab_size': len(preprocessor.vocabulary),
        'n_train': len(train_windows),
        'n_val': len(val_windows),
        'n_test': len(test_windows),
        'n_train_files': len(train_files),
        'n_val_files': len(val_files),
        'n_test_files': len(test_files),
        'master_columns': master_columns_filtered,
        'continuous_columns': master_columns_filtered,
        'consistent_sensors': consistent_sensors,
        'threshold': args.threshold,
        'split_method': 'file_level_stratified',
        'excluded_columns': config.exclude_features,
        'excluded_channels': exclude_channels,  # NEW: Track excluded channels
        'categorical_columns': config.categorical_features,
        'preprocessing_config': {
            'scaler_type': config.scaler_type,
            'nan_strategy': config.nan_strategy,
            'outlier_method': config.outlier_method,
        }
    }

    # Save splits using preprocessor's built-in method
    print("="*80)
    print("SAVING PREPROCESSED DATA")
    print("="*80 + "\n")

    preprocessor.save_processed(
        train_windows,
        args.output_dir / 'train_sequences.npz',
        metadata
    )
    print(f"Saved train: {len(train_windows)} windows from {len(train_files)} files")

    preprocessor.save_processed(
        val_windows,
        args.output_dir / 'val_sequences.npz',
        metadata
    )
    print(f"Saved val: {len(val_windows)} windows from {len(val_files)} files")

    preprocessor.save_processed(
        test_windows,
        args.output_dir / 'test_sequences.npz',
        metadata
    )
    print(f"Saved test: {len(test_windows)} windows from {len(test_files)} files\n")

    # Save overall metadata
    metadata_path = args.output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        metadata_json = {}
        for k, v in metadata.items():
            if isinstance(v, (np.integer, np.floating)):
                metadata_json[k] = int(v) if isinstance(v, np.integer) else float(v)
            else:
                metadata_json[k] = v
        json.dump(metadata_json, f, indent=2)

    # Save scaler stats
    scaler_stats = {
        'mean': preprocessor.continuous_scaler.center_.tolist(),
        'scale': preprocessor.continuous_scaler.scale_.tolist(),
        'scaler_type': config.scaler_type,
    }
    scaler_path = args.output_dir / 'scaler_stats.json'
    with open(scaler_path, 'w') as f:
        json.dump(scaler_stats, f, indent=2)

    # Summary
    print("="*80)
    print("✅ PREPROCESSING COMPLETE (FILE-LEVEL SPLIT - NO PROXIMITY)")
    print("="*80)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Features: {metadata['n_continuous_features']} continuous + "
          f"{metadata['n_categorical_features']} categorical")
    print(f"  (Original: 93 features → After exclusions: {metadata['n_continuous_features']} features)")
    print(f"\nWindows: {metadata['n_train']} train, {metadata['n_val']} val, "
          f"{metadata['n_test']} test")
    print(f"Files: {metadata['n_train_files']} train, {metadata['n_val_files']} val, "
          f"{metadata['n_test_files']} test")
    print(f"\n🎯 Excluded {len(exclude_channels)} problematic channels:")
    for ch in exclude_channels:
        print(f"   ❌ {ch}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
