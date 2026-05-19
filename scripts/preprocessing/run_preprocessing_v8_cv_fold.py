#!/usr/bin/env python3
"""Preprocess one CV fold for the decoder20260511 (V8) pipeline.

Fork of `romesh_changes/run_preprocessing_cv_fold.py`. Do NOT modify the
romesh script in place — it is the V7 reference. This entry point adds:

  - `--label-mode {full_window,per_row}` to drive Phase-2 dual emission.
  - V7-matching window/stride defaults (256/64) so the new NPZs are
    like-for-like comparable to `outputs/decoder20260304/preprocessed_v7/`.
  - Output schema with `token_length` and `window_length` fields written
    explicitly (the V7 schema's `lengths` field was the sensor length —
    see `outputs/decoder20260511/AUDIT_REPORT.md` Priority 1).

Usage:
    python scripts/preprocessing/run_preprocessing_v8_cv_fold.py \\
        --data-dir data_clean \\
        --output-dir outputs/decoder20260511/preprocessed/full_window/fold_1 \\
        --vocab-path data/gcode_vocab_712.json \\
        --fold 1 --n-folds 5 \\
        --window-size 256 --stride 64 \\
        --label-mode full_window
"""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from miracle.dataset.preprocessing import GCodePreprocessor, VALID_LABEL_MODES
from miracle.config.preprocessing_config import PreprocessingConfig

# 6 consistent sensors per Romesh's analysis (>=93% activity across files)
CONSISTENT_SENSORS = ['frame_l2', 'frame_l3', 'frame_r2', 'spindle2', 'y_bed__3', 'y_bed__4']

ELECTRICAL_FEATURES = [
    'spindle', 'x_motor', 'y_motor', 'z_motor',
    'spindle_A', 'x_motor_A', 'y_motor_A', 'z_motor_A',
]


def extract_operation_type(filename: str) -> str:
    fname = filename.lower()
    if 'adaptive150025' in fname: return 'adaptive150025'
    if 'face150025' in fname:     return 'face150025'
    if 'pocket150025' in fname:   return 'pocket150025'
    if 'damageadaptive' in fname: return 'damageadaptive'
    if 'damageface' in fname:     return 'damageface'
    if 'damagepocket' in fname:   return 'damagepocket'
    if 'adaptive' in fname:       return 'adaptive'
    if 'face' in fname:           return 'face'
    if 'pocket' in fname:         return 'pocket'
    return 'unknown'


def stratified_cv_fold_split(csv_files, fold, n_folds=5):
    """Round-robin file-level CV split (same as V7 reference)."""
    files_by_op = defaultdict(list)
    for f in csv_files:
        files_by_op[extract_operation_type(f.name)].append(f)
    train_fs, val_fs, test_fs = [], [], []
    val_fold = (fold + 1) % n_folds
    print(f"\nCV fold split (fold={fold+1}/{n_folds}):")
    print(f"  {'class':<22} {'total':>5} {'train':>5} {'val':>4} {'test':>4}")
    for op in sorted(files_by_op):
        files = sorted(files_by_op[op])
        t, v, te = [], [], []
        for i, f in enumerate(files):
            file_fold = i % n_folds
            if file_fold == fold:
                te.append(f)
            elif file_fold == val_fold:
                v.append(f)
            else:
                t.append(f)
        train_fs.extend(t); val_fs.extend(v); test_fs.extend(te)
        print(f"  {op:<22} {len(files):>5} {len(t):>5} {len(v):>4} {len(te):>4}")
    return train_fs, val_fs, test_fs


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


def get_per_file_sequences(files, preprocessor):
    out = {}
    for csv_path in files:
        df = preprocessor.load_csv(csv_path)
        _, _, gcode_texts = preprocessor.extract_features(df)
        out[csv_path] = set(g for g in (gcode_texts or []) if g and isinstance(g, str) and g.strip())
    return out


def repair_sequence_coverage(train_fs, val_fs, test_fs, preprocessor):
    """Swap files so that every G-code line in test/val also appears in train."""
    per_file_train = get_per_file_sequences(train_fs, preprocessor)
    per_file_val = get_per_file_sequences(val_fs, preprocessor)
    per_file_test = get_per_file_sequences(test_fs, preprocessor)

    train_seqs: set = set()
    for s in per_file_train.values():
        train_seqs.update(s)
    val_seqs: set = set()
    for s in per_file_val.values():
        val_seqs.update(s)
    test_seqs: set = set()
    for s in per_file_test.values():
        test_seqs.update(s)
    missing = (val_seqs | test_seqs) - train_seqs

    swaps = 0
    if not missing:
        return train_fs, val_fs, test_fs, swaps

    # sorted(), not list(): `missing` is a set, so list() iteration order is
    # process-randomized (string hash seed) -> the coverage swaps land
    # differently every run -> the val/test partition is nondeterministic.
    # Its contents are channel-independent (pure G-code corpus + deterministic
    # stratified_cv_fold_split), so sorting makes the whole split reproducible
    # AND identical across sensor-modality exclusions -- required for the LOMO
    # study to be a valid paired comparison (same test set per modality).
    for seq in sorted(missing):
        # Skip if already in train (from previous swap)
        if any(seq in per_file_train.get(f, set()) for f in train_fs):
            continue
        # Find source (val/test) file containing this seq
        source_file = None; source_list = None
        for f in val_fs:
            if seq in per_file_val.get(f, set()):
                source_file = f; source_list = val_fs; break
        if source_file is None:
            for f in test_fs:
                if seq in per_file_test.get(f, set()):
                    source_file = f; source_list = test_fs; break
        if source_file is None:
            continue
        op = extract_operation_type(source_file.name)
        cand = next((f for f in train_fs if extract_operation_type(f.name) == op), None)
        if cand:
            train_fs.remove(cand); source_list.remove(source_file)
            train_fs.append(source_file); source_list.append(cand)
            per_file_train[source_file] = per_file_val.get(source_file, set()) | per_file_test.get(source_file, set())
            swaps += 1
            print(f"    Swapped {cand.name} <-> {source_file.name} (op={op}) for seq: {seq[:50]}...")
    return train_fs, val_fs, test_fs, swaps


def main() -> int:
    p = argparse.ArgumentParser(description='V8 preprocessing for one CV fold')
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--vocab-path', type=Path, required=True)
    p.add_argument('--sensor-report', type=Path, default=REPO / 'outputs' / 'sensor_consistency_report.json')
    p.add_argument('--threshold', type=float, default=93.0)
    p.add_argument('--fold', type=int, required=True, help='1-indexed (1..n_folds)')
    p.add_argument('--n-folds', type=int, default=5)
    p.add_argument('--window-size', type=int, default=256, help='V7 default: 256')
    p.add_argument('--stride', type=int, default=64, help='V7 default: 64')
    p.add_argument('--label-mode', choices=list(VALID_LABEL_MODES), default='full_window')
    # Channel-exclusion flags so the V8 preprocessing can be aligned with a
    # specific frozen encoder. The f98 encoder available in this repo
    # (`outputs/experiments_2026_02_25/no_proximity_no_pressure_w256_s64_cv/`)
    # requires proximity + pressure excluded.
    p.add_argument('--exclude-proximity', action='store_true')
    p.add_argument('--exclude-pressure', action='store_true')
    p.add_argument('--exclude-color', action='store_true')
    p.add_argument('--exclude-magnetometer', action='store_true')
    p.add_argument('--exclude-accelerometer', action='store_true')
    p.add_argument('--exclude-gyroscope', action='store_true')
    p.add_argument('--exclude-temperature', action='store_true')
    p.add_argument('--exclude-audio', action='store_true')
    p.add_argument('--exclude-electrical', action='store_true')
    args = p.parse_args()

    fold_0 = args.fold - 1
    if not (0 <= fold_0 < args.n_folds):
        raise ValueError(f"--fold must be 1..{args.n_folds}, got {args.fold}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    consistent_sensors = CONSISTENT_SENSORS
    if args.sensor_report.exists():
        rep = json.load(open(args.sensor_report))
        consistent_sensors = [
            name for name, info in rep['sensors'].items()
            if info['activity_percentage'] >= args.threshold
        ]
        print(f"Using {len(consistent_sensors)} sensors (>= {args.threshold}% activity)")

    csv_files = sorted(args.data_dir.glob('*_aligned.csv'))
    if not csv_files:
        csv_files = sorted(args.data_dir.glob('*.csv'))
    if not csv_files:
        raise ValueError(f"No CSV files in {args.data_dir}")

    # Filter to files with all consistent sensors present
    valid = []
    for csv in csv_files:
        sensors_in_file = {c.split('.')[0] for c in pd.read_csv(csv, nrows=0).columns if '.' in c}
        if all(s in sensors_in_file for s in consistent_sensors):
            valid.append(csv)
    n_dropped = len(csv_files) - len(valid)
    if n_dropped:
        print(f"Filtered to {len(valid)} files with all required sensors (dropped {n_dropped})")
    csv_files = valid

    train_fs, val_fs, test_fs = stratified_cv_fold_split(csv_files, fold=fold_0, n_folds=args.n_folds)

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

    # Master column list
    all_cont_cols = set()
    for csv in csv_files:
        df = pd.read_csv(csv, nrows=1)
        for col in df.columns:
            if col not in config.exclude_features and col not in config.categorical_features:
                if pd.api.types.is_numeric_dtype(df[col]):
                    all_cont_cols.add(col)
    # Build channel-exclusion list from flags (same channel sets as romesh script)
    excluded_channels: list = []
    sensors = consistent_sensors
    if args.exclude_proximity:    excluded_channels += [f'{s}.Proximity' for s in sensors]
    if args.exclude_pressure:     excluded_channels += [f'{s}.Pressure' for s in sensors]
    if args.exclude_color:        excluded_channels += [f'{s}.{c}' for s in sensors for c in ['ColorR','ColorG','ColorB','ColorA']]
    if args.exclude_magnetometer: excluded_channels += [f'{s}.{c}' for s in sensors for c in ['Mx','My','Mz']]
    if args.exclude_accelerometer:excluded_channels += [f'{s}.{c}' for s in sensors for c in ['Ax','Ay','Az']]
    if args.exclude_gyroscope:    excluded_channels += [f'{s}.{c}' for s in sensors for c in ['Gx','Gy','Gz']]
    if args.exclude_temperature:  excluded_channels += [f'{s}.Temperature' for s in sensors]
    if args.exclude_audio:        excluded_channels += [f'{s}.RMS' for s in sensors]
    if args.exclude_electrical:   excluded_channels += list(ELECTRICAL_FEATURES)

    master_columns = filter_columns(sorted(all_cont_cols), consistent_sensors, exclude_channels=excluded_channels)
    print(f"Master columns after filtering: {len(master_columns)} (excluded {len(excluded_channels)} channels)")

    preprocessor = GCodePreprocessor(
        args.vocab_path, config=config,
        master_columns=master_columns,
        label_mode=args.label_mode,
    )

    print("Verifying sequence coverage (and repairing if needed)...")
    train_fs, val_fs, test_fs, n_swaps = repair_sequence_coverage(train_fs, val_fs, test_fs, preprocessor)
    if n_swaps:
        print(f"  {n_swaps} file swaps performed for coverage")
    else:
        print("  coverage OK without swaps")

    json.dump(
        {
            'fold': args.fold,
            'n_folds': args.n_folds,
            'val_fold': (fold_0 + 1) % args.n_folds + 1,
            'label_mode': args.label_mode,
            'train_files': [f.name for f in train_fs],
            'val_files': [f.name for f in val_fs],
            'test_files': [f.name for f in test_fs],
        },
        open(args.output_dir / 'file_split.json', 'w'), indent=2,
    )

    # Fit scaler on training rows
    print("Fitting scaler on training data...")
    train_continuous = []
    for csv in train_fs:
        df = preprocessor.load_csv(csv)
        cont, _, _ = preprocessor.extract_features(df)
        train_continuous.append(cont)
    preprocessor.fit_scaler(np.vstack(train_continuous))
    print(f"  Scaler fitted on {sum(c.shape[0] for c in train_continuous)} rows")

    def process_set(files, name):
        print(f"Windowing {name} ({len(files)} files, label_mode={args.label_mode})...")
        windows = []
        for csv in files:
            windows.extend(preprocessor.process_file(csv, fit_scaler=False))
        print(f"  {name}: {len(windows)} samples")
        return windows

    train_windows = process_set(train_fs, 'TRAIN')
    val_windows = process_set(val_fs, 'VAL')
    test_windows = process_set(test_fs, 'TEST')

    cont_shape = train_windows[0]['continuous'].shape
    cat_shape = train_windows[0]['categorical'].shape
    metadata = {
        'n_continuous_features': cont_shape[1],
        'n_categorical_features': cat_shape[1],
        'window_size': args.window_size,
        'stride': args.stride,
        'vocab_size': len(preprocessor.vocabulary),
        'n_train': len(train_windows),
        'n_val': len(val_windows),
        'n_test': len(test_windows),
        'n_train_files': len(train_fs),
        'n_val_files': len(val_fs),
        'n_test_files': len(test_fs),
        'master_columns': master_columns,
        'continuous_columns': master_columns,
        'consistent_sensors': consistent_sensors,
        'threshold': args.threshold,
        'fold': args.fold,
        'n_folds': args.n_folds,
        'label_mode': args.label_mode,
        'split_method': 'cv_file_level_stratified',
        'preprocessing_config': {
            'scaler_type': config.scaler_type,
            'nan_strategy': config.nan_strategy,
            'outlier_method': config.outlier_method,
        },
        'v8_schema_notes': {
            'lengths': 'token sequence length (V7 had sensor length here - bug, see AUDIT_REPORT.md)',
            'token_length': 'explicit alias of lengths',
            'window_length': 'sensor window length (always = window_size)',
            'line_in_window_index': 'per_row: position of this G-code line within window; full_window: always 0',
            'n_lines_in_window': 'number of distinct G-code lines that fired in this window',
        },
    }

    print("Saving NPZ + metadata...")
    preprocessor.save_processed(train_windows, args.output_dir / 'train_sequences.npz', metadata)
    preprocessor.save_processed(val_windows, args.output_dir / 'val_sequences.npz', metadata)
    preprocessor.save_processed(test_windows, args.output_dir / 'test_sequences.npz', metadata)

    def jsafe(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, np.integer): out[k] = int(v)
            elif isinstance(v, np.floating): out[k] = float(v)
            else: out[k] = v
        return out

    json.dump(jsafe(metadata), open(args.output_dir / 'metadata.json', 'w'), indent=2)
    json.dump(
        {
            'mean': preprocessor.continuous_scaler.center_.tolist(),
            'scale': preprocessor.continuous_scaler.scale_.tolist(),
            'scaler_type': config.scaler_type,
        },
        open(args.output_dir / 'scaler_stats.json', 'w'), indent=2,
    )
    for windows, name in [(train_windows, 'train'), (val_windows, 'val'), (test_windows, 'test')]:
        sm = dict(metadata); sm.update({'n_samples': len(windows), 'split': name})
        json.dump(jsafe(sm), open(args.output_dir / f'{name}_sequences_metadata.json', 'w'), indent=2)

    print(f"\nFold {args.fold}/{args.n_folds} ({args.label_mode}) -> {args.output_dir}")
    print(f"  Train: {len(train_windows)} samples ({len(train_fs)} files)")
    print(f"  Val:   {len(val_windows)} samples ({len(val_fs)} files)")
    print(f"  Test:  {len(test_windows)} samples ({len(test_fs)} files)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
