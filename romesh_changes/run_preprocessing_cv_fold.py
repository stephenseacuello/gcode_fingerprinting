#!/usr/bin/env python3
"""
Preprocess data for a single fold of 5-fold cross-validation.

Files are assigned to folds using round-robin within each class (sorted order),
so the assignment is deterministic and reproducible without a random seed.

Fold assignment:
  - Normal classes (20 files each): 4 files per fold
  - Damage classes (5 files each):  1 file per fold

For fold k (0-indexed):
  - test  = files assigned to fold k
  - val   = files assigned to fold (k+1) % n_folds
  - train = all remaining files

Usage:
    python romesh_changes/run_preprocessing_cv_fold.py \
        --data-dir data/ \
        --output-dir outputs/experiments/no_prox_no_color_no_mag_cv/fold_1/preprocessed \
        --vocab-path outputs/vocabulary/gcode_vocabulary_v2.json \
        --fold 1 \
        --n-folds 5 \
        --no-proximity --no-color --no-magnetometer
"""
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np

from src.miracle.dataset.preprocessing import GCodePreprocessor
from src.miracle.config.preprocessing_config import PreprocessingConfig

# Sensors with >=95% activity
CONSISTENT_SENSORS = ['frame_l2', 'frame_r2', 'spindle2', 'y_bed__3', 'y_bed__4']

ELECTRICAL_FEATURES = [
    'spindle', 'x_motor', 'y_motor', 'z_motor',
    'spindle_A', 'x_motor_A', 'y_motor_A', 'z_motor_A'
]

ALL_PROXIMITY_CHANNELS = [
    'frame_l2.Proximity', 'frame_r2.Proximity', 'spindle2.Proximity',
    'y_bed__3.Proximity', 'y_bed__4.Proximity',
]
ALL_COLOR_CHANNELS = [
    'frame_l2.ColorR', 'frame_l2.ColorG', 'frame_l2.ColorB', 'frame_l2.ColorA',
    'frame_r2.ColorR', 'frame_r2.ColorG', 'frame_r2.ColorB', 'frame_r2.ColorA',
    'spindle2.ColorR', 'spindle2.ColorG', 'spindle2.ColorB', 'spindle2.ColorA',
    'y_bed__3.ColorR', 'y_bed__3.ColorG', 'y_bed__3.ColorB', 'y_bed__3.ColorA',
    'y_bed__4.ColorR', 'y_bed__4.ColorG', 'y_bed__4.ColorB', 'y_bed__4.ColorA',
]
ALL_MAGNETOMETER_CHANNELS = [
    'frame_l2.Mx', 'frame_l2.My', 'frame_l2.Mz',
    'frame_r2.Mx', 'frame_r2.My', 'frame_r2.Mz',
    'spindle2.Mx', 'spindle2.My', 'spindle2.Mz',
    'y_bed__3.Mx', 'y_bed__3.My', 'y_bed__3.Mz',
    'y_bed__4.Mx', 'y_bed__4.My', 'y_bed__4.Mz',
]


def extract_operation_type(filename):
    fname = filename.lower()
    if 'adaptive150025' in fname:   return 'adaptive150025'
    if 'face150025'     in fname:   return 'face150025'
    if 'pocket150025'   in fname:   return 'pocket150025'
    if 'damageadaptive' in fname:   return 'damageadaptive'
    if 'damageface'     in fname:   return 'damageface'
    if 'damagepocket'   in fname:   return 'damagepocket'
    if 'adaptive'       in fname:   return 'adaptive'
    if 'face'           in fname:   return 'face'
    if 'pocket'         in fname:   return 'pocket'
    return 'unknown'


def stratified_cv_fold_split(csv_files, fold, n_folds=5):
    """
    Assign files to folds via round-robin within each class (sorted order),
    then partition into test / val / train for the given fold.

      test  = fold k
      val   = fold (k+1) % n_folds
      train = all other folds
    """
    files_by_operation = defaultdict(list)
    for f in csv_files:
        files_by_operation[extract_operation_type(f.name)].append(f)

    train_files, val_files, test_files = [], [], []

    print("\n" + "=" * 80)
    print(f"CV FOLD SPLIT  (fold={fold+1}/{n_folds})")
    print("=" * 80)
    print(f"{'Class':<22} {'Total':>5}  {'Train':>5}  {'Val':>4}  {'Test':>4}")
    print("-" * 80)

    val_fold = (fold + 1) % n_folds

    for operation in sorted(files_by_operation.keys()):
        files = sorted(files_by_operation[operation])
        t, v, te = [], [], []
        for i, f in enumerate(files):
            file_fold = i % n_folds
            if file_fold == fold:
                te.append(f)
            elif file_fold == val_fold:
                v.append(f)
            else:
                t.append(f)
        train_files.extend(t)
        val_files.extend(v)
        test_files.extend(te)
        print(f"  {operation:<20} {len(files):>5}  {len(t):>5}  {len(v):>4}  {len(te):>4}")

    print("-" * 80)
    print(f"  {'TOTAL':<20} {len(csv_files):>5}  {len(train_files):>5}  "
          f"{len(val_files):>4}  {len(test_files):>4}")
    print("=" * 80 + "\n")

    return train_files, val_files, test_files


def filter_columns(master_columns, consistent_sensors, exclude_channels):
    filtered = []
    for col in master_columns:
        if col in exclude_channels:
            continue
        if col in ELECTRICAL_FEATURES:
            filtered.append(col)
            continue
        if '.' in col:
            sensor = col.split('.', 1)[0]
            if sensor in consistent_sensors:
                filtered.append(col)
    return filtered


def main():
    parser = argparse.ArgumentParser(description='Preprocess one CV fold')
    parser.add_argument('--data-dir',    type=Path, required=True)
    parser.add_argument('--output-dir',  type=Path, required=True)
    parser.add_argument('--vocab-path',  type=Path, required=True)
    parser.add_argument('--sensor-report', type=Path,
                        default='outputs/sensor_consistency_report.json')
    parser.add_argument('--threshold',   type=float, default=95.0)
    parser.add_argument('--fold',        type=int,   required=True,
                        help='Fold index, 1-indexed (1 to n-folds)')
    parser.add_argument('--n-folds',     type=int,   default=5)
    parser.add_argument('--window-size', type=int,   default=64)
    parser.add_argument('--stride',      type=int,   default=16)
    parser.add_argument('--exclude-proximity',    action='store_true')
    parser.add_argument('--exclude-color',        action='store_true')
    parser.add_argument('--exclude-magnetometer', action='store_true')
    args = parser.parse_args()

    # Convert to 0-indexed fold
    fold_0 = args.fold - 1
    if not (0 <= fold_0 < args.n_folds):
        raise ValueError(f"--fold must be between 1 and {args.n_folds}, got {args.fold}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Channels to exclude
    exclude_channels = []
    if args.exclude_proximity:    exclude_channels.extend(ALL_PROXIMITY_CHANNELS)
    if args.exclude_color:        exclude_channels.extend(ALL_COLOR_CHANNELS)
    if args.exclude_magnetometer: exclude_channels.extend(ALL_MAGNETOMETER_CHANNELS)

    # Load sensor consistency report
    consistent_sensors = CONSISTENT_SENSORS
    if args.sensor_report.exists():
        with open(args.sensor_report) as f:
            report = json.load(f)
        consistent_sensors = [
            name for name, info in report['sensors'].items()
            if info['activity_percentage'] >= args.threshold
        ]
        print(f"Using {len(consistent_sensors)} sensors with >={args.threshold}% activity: "
              f"{consistent_sensors}")

    # Find CSV files
    csv_files = sorted(args.data_dir.glob('*_aligned.csv'))
    if not csv_files:
        csv_files = sorted(args.data_dir.glob('*.csv'))
    if not csv_files:
        raise ValueError(f"No CSV files found in {args.data_dir}")
    print(f"Found {len(csv_files)} CSV files")

    # CV fold split
    train_files, val_files, test_files = stratified_cv_fold_split(
        csv_files, fold=fold_0, n_folds=args.n_folds
    )

    # Save split info
    split_info = {
        'fold': args.fold,
        'n_folds': args.n_folds,
        'val_fold': (fold_0 + 1) % args.n_folds + 1,
        'train_files': [f.name for f in train_files],
        'val_files':   [f.name for f in val_files],
        'test_files':  [f.name for f in test_files],
    }
    with open(args.output_dir / 'file_split.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    # Preprocessing config
    config = PreprocessingConfig(
        window_size=args.window_size,
        stride=args.stride,
        scaler_type='robust',
        nan_strategy='forward_fill',
        outlier_method='clip',
        remove_zero_variance=True,
        correlation_threshold=0.95,
        random_seed=42,
    )

    # Build master column list from all files
    import pandas as pd
    all_continuous_cols = set()
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, nrows=1)
        for col in df.columns:
            if col not in config.exclude_features and col not in config.categorical_features:
                if pd.api.types.is_numeric_dtype(df[col]):
                    all_continuous_cols.add(col)

    master_columns = filter_columns(
        sorted(all_continuous_cols), consistent_sensors, exclude_channels
    )
    print(f"Features after filtering: {len(master_columns)}")

    # Preprocessor
    preprocessor = GCodePreprocessor(
        args.vocab_path, config=config, master_columns=master_columns
    )

    # Fit scaler on training files only
    print("Fitting scaler on training data...")
    train_continuous = []
    for csv_path in train_files:
        df = preprocessor.load_csv(csv_path)
        continuous, _, _ = preprocessor.extract_features(df)
        train_continuous.append(continuous)
    preprocessor.fit_scaler(np.vstack(train_continuous))
    print(f"  Scaler fitted on {sum(c.shape[0] for c in train_continuous)} rows")

    # Create windows for each split
    def process_set(files, name):
        print(f"Windowing {name} ({len(files)} files)...")
        windows = []
        for csv_path in files:
            windows.extend(preprocessor.process_file(csv_path, fit_scaler=False))
        print(f"  {name}: {len(windows)} windows")
        return windows

    train_windows = process_set(train_files, 'TRAIN')
    val_windows   = process_set(val_files,   'VAL')
    test_windows  = process_set(test_files,  'TEST')

    # Build metadata
    cont_shape = train_windows[0]['continuous'].shape
    cat_shape  = train_windows[0]['categorical'].shape
    metadata = {
        'n_continuous_features': cont_shape[1],
        'n_categorical_features': cat_shape[1],
        'window_size': args.window_size,
        'stride': args.stride,
        'vocab_size': len(preprocessor.vocabulary),
        'n_train': len(train_windows),
        'n_val':   len(val_windows),
        'n_test':  len(test_windows),
        'n_train_files': len(train_files),
        'n_val_files':   len(val_files),
        'n_test_files':  len(test_files),
        'master_columns': master_columns,
        'continuous_columns': master_columns,
        'consistent_sensors': consistent_sensors,
        'threshold': args.threshold,
        'fold': args.fold,
        'n_folds': args.n_folds,
        'split_method': 'cv_file_level_stratified',
        'excluded_channels': exclude_channels,
        'categorical_columns': config.categorical_features,
        'preprocessing_config': {
            'scaler_type': config.scaler_type,
            'nan_strategy': config.nan_strategy,
            'outlier_method': config.outlier_method,
        },
    }

    # Save NPZ and metadata
    print("Saving preprocessed splits...")
    preprocessor.save_processed(train_windows, args.output_dir / 'train_sequences.npz', metadata)
    preprocessor.save_processed(val_windows,   args.output_dir / 'val_sequences.npz',   metadata)
    preprocessor.save_processed(test_windows,  args.output_dir / 'test_sequences.npz',  metadata)

    def to_json_safe(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, (np.integer,)):    out[k] = int(v)
            elif isinstance(v, (np.floating,)): out[k] = float(v)
            else:                               out[k] = v
        return out

    with open(args.output_dir / 'metadata.json', 'w') as f:
        json.dump(to_json_safe(metadata), f, indent=2)

    scaler_stats = {
        'mean':  preprocessor.continuous_scaler.center_.tolist(),
        'scale': preprocessor.continuous_scaler.scale_.tolist(),
        'scaler_type': config.scaler_type,
    }
    with open(args.output_dir / 'scaler_stats.json', 'w') as f:
        json.dump(scaler_stats, f, indent=2)

    # Per-split metadata
    for windows, name in [(train_windows, 'train'), (val_windows, 'val'), (test_windows, 'test')]:
        split_meta = dict(metadata)
        split_meta.update({'n_samples': len(windows), 'split': name})
        with open(args.output_dir / f'{name}_sequences_metadata.json', 'w') as f:
            json.dump(to_json_safe(split_meta), f, indent=2)

    print(f"\nFold {args.fold}/{args.n_folds} preprocessed -> {args.output_dir}")
    print(f"  Train: {len(train_windows)} windows ({len(train_files)} files)")
    print(f"  Val:   {len(val_windows)} windows ({len(val_files)} files)")
    print(f"  Test:  {len(test_windows)} windows ({len(test_files)} files)")


if __name__ == '__main__':
    main()
