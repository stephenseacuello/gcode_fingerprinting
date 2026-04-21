#!/usr/bin/env python3
"""
Preprocess OOD leave-one-toolpath-out splits for the decoder revision.

For each heldout toolpath in {adaptive, face, pocket}:
  - train = 80% of files from the OTHER two toolpaths (incl. 150025/damage variants)
  - val   = 20% of files from the OTHER two toolpaths
  - test  = all files from the heldout toolpath

Reuses romesh_changes/run_preprocessing_cv_fold.py machinery for scaler
fitting, windowing, and NPZ output. Scaler is refit on the OOD train split.
"""
import sys
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
import pandas as pd

from miracle.dataset.preprocessing import GCodePreprocessor
from miracle.config.preprocessing_config import PreprocessingConfig

# Reuse constants from the fold-CV preprocessor
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'romesh_changes'))
from run_preprocessing_cv_fold import (  # noqa: E402
    CONSISTENT_SENSORS,
    filter_columns,
    extract_operation_type,
)


TOOLPATH_MAP = {
    'adaptive': {'adaptive', 'adaptive150025', 'damageadaptive'},
    'face':     {'face', 'face150025', 'damageface'},
    'pocket':   {'pocket', 'pocket150025', 'damagepocket'},
}


def toolpath_of(fname: str) -> str:
    op = extract_operation_type(fname)
    for tp, ops in TOOLPATH_MAP.items():
        if op in ops:
            return tp
    return 'other'


def build_ood_split(csv_files, heldout: str, val_frac: float = 0.2, seed: int = 42):
    """
    Train + val = files from the two non-heldout toolpaths.
    Test        = all files from the heldout toolpath.
    val_frac of the non-heldout files (stratified by operation type) goes to val.
    """
    rng = random.Random(seed)
    train, val, test = [], [], []
    by_op = defaultdict(list)
    for f in csv_files:
        by_op[extract_operation_type(f.name)].append(f)

    for op in sorted(by_op.keys()):
        files = sorted(by_op[op])
        files_toolpath = toolpath_of(files[0].name)
        if files_toolpath == heldout:
            test.extend(files)
        else:
            shuffled = files[:]
            rng.shuffle(shuffled)
            n_val = max(1, int(round(len(shuffled) * val_frac)))
            val.extend(shuffled[:n_val])
            train.extend(shuffled[n_val:])

    return sorted(train), sorted(val), sorted(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir',   type=Path, default=Path('data_clean'))
    ap.add_argument('--output-dir', type=Path, required=True,
                    help='Base dir; writes heldout_{adaptive,face,pocket}/ subdirs')
    ap.add_argument('--vocab-path', type=Path, default=Path('data/gcode_vocab_712.json'))
    ap.add_argument('--sensor-report', type=Path,
                    default=Path('outputs/sensor_consistency_report.json'))
    ap.add_argument('--threshold', type=float, default=93.0)
    ap.add_argument('--window-size', type=int, default=256)
    ap.add_argument('--stride', type=int, default=64)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--heldout', choices=['adaptive', 'face', 'pocket', 'all'],
                    default='all')
    args = ap.parse_args()

    heldouts = ['adaptive', 'face', 'pocket'] if args.heldout == 'all' else [args.heldout]

    # Resolve CSV files and consistent sensors once
    csv_files = sorted(args.data_dir.glob('*_aligned.csv'))
    if not csv_files:
        raise SystemExit(f"No CSVs found in {args.data_dir}")

    consistent_sensors = CONSISTENT_SENSORS
    if args.sensor_report.exists():
        report = json.load(open(args.sensor_report))
        consistent_sensors = [name for name, info in report['sensors'].items()
                              if info['activity_percentage'] >= args.threshold]
    print(f"Consistent sensors ({len(consistent_sensors)}): {consistent_sensors}")

    # Filter files to those with all required sensors
    valid_files = []
    for p in csv_files:
        hdr = pd.read_csv(p, nrows=0)
        sensors_in_file = {c.split('.', 1)[0] for c in hdr.columns if '.' in c}
        if all(s in sensors_in_file for s in consistent_sensors):
            valid_files.append(p)
    csv_files = valid_files
    print(f"Files with all required sensors: {len(csv_files)}")

    # Build master column list from first file
    config = PreprocessingConfig(
        window_size=args.window_size, stride=args.stride,
        scaler_type='robust', nan_strategy='forward_fill',
        outlier_method='clip', remove_zero_variance=True,
        correlation_threshold=0.95, random_seed=42,
    )
    all_cols = set()
    for p in csv_files:
        df = pd.read_csv(p, nrows=1)
        for c in df.columns:
            if c not in config.exclude_features and c not in config.categorical_features:
                if pd.api.types.is_numeric_dtype(df[c]):
                    all_cols.add(c)
    master_columns = filter_columns(sorted(all_cols), consistent_sensors, [])
    print(f"Features: {len(master_columns)}")

    for heldout in heldouts:
        print("\n" + "=" * 80)
        print(f"OOD SPLIT: heldout = {heldout}")
        print("=" * 80)
        out_dir = args.output_dir / f'heldout_{heldout}'
        out_dir.mkdir(parents=True, exist_ok=True)

        train_files, val_files, test_files = build_ood_split(
            csv_files, heldout, args.val_frac, args.seed)

        print(f"train: {len(train_files)} files ({sum(1 for f in train_files if toolpath_of(f.name)==heldout)} heldout leak)")
        print(f"val  : {len(val_files)}   files ({sum(1 for f in val_files   if toolpath_of(f.name)==heldout)} heldout leak)")
        print(f"test : {len(test_files)} files")

        split_info = {
            'heldout_toolpath': heldout,
            'val_frac': args.val_frac,
            'seed': args.seed,
            'train_files': [f.name for f in train_files],
            'val_files':   [f.name for f in val_files],
            'test_files':  [f.name for f in test_files],
        }
        with open(out_dir / 'file_split.json', 'w') as f:
            json.dump(split_info, f, indent=2)

        preprocessor = GCodePreprocessor(
            args.vocab_path, config=config, master_columns=master_columns)

        # Fit scaler on OOD train pool
        print("Fitting scaler on OOD train pool...")
        train_continuous = []
        for p in train_files:
            df = preprocessor.load_csv(p)
            continuous, _, _ = preprocessor.extract_features(df)
            train_continuous.append(continuous)
        preprocessor.fit_scaler(np.vstack(train_continuous))

        # Window each split
        def process_set(files, name):
            print(f"  {name}: windowing {len(files)} files...")
            windows = []
            for p in files:
                windows.extend(preprocessor.process_file(p, fit_scaler=False))
            print(f"    → {len(windows)} windows")
            return windows

        train_w = process_set(train_files, 'TRAIN')
        val_w   = process_set(val_files,   'VAL')
        test_w  = process_set(test_files,  'TEST')

        # Metadata
        metadata = {
            'n_continuous_features': train_w[0]['continuous'].shape[1],
            'n_categorical_features': train_w[0]['categorical'].shape[1],
            'window_size': args.window_size,
            'stride': args.stride,
            'vocab_size': len(preprocessor.vocabulary),
            'n_train': len(train_w),
            'n_val': len(val_w),
            'n_test': len(test_w),
            'n_train_files': len(train_files),
            'n_val_files': len(val_files),
            'n_test_files': len(test_files),
            'master_columns': master_columns,
            'continuous_columns': master_columns,
            'consistent_sensors': consistent_sensors,
            'split_method': 'ood_leave_one_toolpath_out',
            'heldout_toolpath': heldout,
            'categorical_columns': config.categorical_features,
        }
        preprocessor.save_processed(train_w, out_dir / 'train_sequences.npz', metadata)
        preprocessor.save_processed(val_w,   out_dir / 'val_sequences.npz',   metadata)
        preprocessor.save_processed(test_w,  out_dir / 'test_sequences.npz',  metadata)

        with open(out_dir / 'metadata.json', 'w') as f:
            json.dump({k: (int(v) if isinstance(v, np.integer) else
                           float(v) if isinstance(v, np.floating) else v)
                       for k, v in metadata.items()}, f, indent=2)
        with open(out_dir / 'scaler_stats.json', 'w') as f:
            json.dump({
                'mean':  preprocessor.continuous_scaler.center_.tolist(),
                'scale': preprocessor.continuous_scaler.scale_.tolist(),
                'scaler_type': config.scaler_type,
            }, f, indent=2)
        print(f"Saved → {out_dir}")


if __name__ == '__main__':
    main()
