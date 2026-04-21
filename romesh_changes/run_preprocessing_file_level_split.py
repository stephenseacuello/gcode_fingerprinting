#!/usr/bin/env python3
"""
Preprocess data with FILE-LEVEL splitting (fixes Issue #2).

This addresses both Issue #1 (zero-padding) and Issue #2 (window-level split):
- Uses only 95% consistent sensors (no zero-padding)
- Splits FILES first, then creates windows (prevents file-level leakage)

Key difference from run_preprocessing_consistent_sensors.py:
- OLD: Create all windows → shuffle → split
- NEW: Split files → create windows from each file set

Usage:
    python romesh_changes/run_preprocessing_file_level_split.py \
        --data-dir data/ \
        --output-dir outputs/experiments/file_level_split/preprocessed \
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

from miracle.dataset.preprocessing import GCodePreprocessor
from miracle.config.preprocessing_config import PreprocessingConfig

# Sensors with ≥95% activity
CONSISTENT_SENSORS = ['frame_l2', 'frame_r2', 'spindle2', 'y_bed__3', 'y_bed__4']

# Electrical features
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

        # For very small classes (damage), ensure at least 1 in test
        if n_files <= 5:
            # Put at least 1 in test, 1 in val if possible
            n_test = max(1, int(n_files * (1 - train_frac - val_frac)))
            n_val = max(1, int(n_files * val_frac)) if n_files > 2 else 0
            n_train = n_files - n_test - n_val

            test_files.extend(files[:n_test])
            val_files.extend(files[n_test:n_test+n_val])
            train_files.extend(files[n_test+n_val:])
        else:
            # Standard stratified split for larger classes
            # First split: train vs (val+test)
            train, temp = train_test_split(
                files,
                test_size=(1 - train_frac),
                random_state=seed
            )

            # Second split: val vs test
            val_fraction = val_frac / (1 - train_frac)
            val, test = train_test_split(
                temp,
                test_size=(1 - val_fraction),
                random_state=seed
            )

            train_files.extend(train)
            val_files.extend(val)
            test_files.extend(test)

        print(f"  {operation:<20} {n_files:>3} files → "
              f"train={len([f for f in train_files if operation in f.name]):>2}, "
              f"val={len([f for f in val_files if operation in f.name]):>2}, "
              f"test={len([f for f in test_files if operation in f.name]):>2}")

    print("-"*80)
    print(f"{'TOTAL':<20} {len(csv_files):>3} files → "
          f"train={len(train_files):>2}, "
          f"val={len(val_files):>2}, "
          f"test={len(test_files):>2}")
    print("="*80 + "\n")

    return train_files, val_files, test_files


def filter_master_columns_to_consistent_sensors(master_columns, consistent_sensors=None):
    """Filter master column list to only include consistent sensors."""
    if consistent_sensors is None:
        consistent_sensors = CONSISTENT_SENSORS

    filtered_columns = []

    for col in master_columns:
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
        description="Preprocess with file-level split (fixes Issue #2)"
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

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load sensor consistency report
    consistent_sensors = CONSISTENT_SENSORS

    if args.sensor_report.exists():
        print(f"\nLoading sensor consistency report from: {args.sensor_report}")
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
        print("✅ NO ZERO-PADDING\n")

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

    # Filter to consistent sensors
    master_columns_filtered = filter_master_columns_to_consistent_sensors(
        master_columns, consistent_sensors
    )
    print(f"  After filtering: {len(master_columns_filtered)} features\n")

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
            print(f"  {csv_path.name}: {len(file_windows)} windows")
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
        'split_method': 'file_level_stratified',  # Important marker!
        'excluded_columns': list(exclude_cols),
        'categorical_columns': cat_cols_names,
        'preprocessing_config': {
            'scaler_type': config.scaler_type,
            'nan_strategy': config.nan_strategy,
            'outlier_method': config.outlier_method,
        }
    }

    # Save splits
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

    # Save metadata
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
    scaler_stats_path = args.output_dir / 'scaler_stats.json'
    scaler_stats = {
        'mean': preprocessor.continuous_scaler.center_.tolist(),
        'scale': preprocessor.continuous_scaler.scale_.tolist(),
        'scaler_type': config.scaler_type,
    }
    with open(scaler_stats_path, 'w') as f:
        json.dump(scaler_stats, f, indent=2)

    # Summary
    print("="*80)
    print("✅ PREPROCESSING COMPLETE (FILE-LEVEL SPLIT)")
    print("="*80)
    print(f"\nOutput: {args.output_dir}")
    print(f"Features: {continuous_shape[1]} (NO zero-padding)")
    print(f"\nFiles split:")
    print(f"  Train: {len(train_files)} files → {len(train_windows)} windows")
    print(f"  Val:   {len(val_files)} files → {len(val_windows)} windows")
    print(f"  Test:  {len(test_files)} files → {len(test_windows)} windows")
    print(f"\n✅ No file appears in multiple splits!")
    print(f"✅ Scaler fitted only on training data!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
