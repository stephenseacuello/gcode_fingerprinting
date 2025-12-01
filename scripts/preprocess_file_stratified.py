#!/usr/bin/env python3
"""
File-level stratified preprocessing to prevent data leakage.

CRITICAL FIX: Splits files FIRST by operation type, THEN generates windows.
This prevents overlapping windows from the same file appearing in train/val/test.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import json
import argparse
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

from miracle.dataset.preprocessing import (
    GCodePreprocessor,
    extract_operation_type,
    load_vocabulary
)
from miracle.config.preprocessing_config import get_default_config


def stratified_file_split(
    csv_files: List[Path],
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split files (not windows) into train/val/test with operation-type stratification.

    This is CRITICAL to prevent data leakage - ensures no G-code file appears
    in multiple splits.

    Args:
        csv_files: List of CSV file paths
        test_size: Fraction for test set (default 0.15 = 15%)
        val_size: Fraction for validation set (default 0.15 = 15%)
        random_state: Random seed for reproducibility

    Returns:
        train_files, val_files, test_files
    """
    print("\n" + "="*80)
    print("FILE-LEVEL STRATIFIED SPLITTING (prevents data leakage)")
    print("="*80)

    # Extract operation types from filenames
    file_operations = []
    for f in csv_files:
        op_type = extract_operation_type(f.name)
        file_operations.append((f, op_type))

    # Create DataFrame for stratified splitting
    df = pd.DataFrame(file_operations, columns=['file', 'operation'])

    print(f"\nTotal files: {len(df)}")
    print(f"Operation distribution:")
    print(df['operation'].value_counts())

    # First split: (train+val) vs test (stratified by operation)
    train_val_files, test_files = train_test_split(
        df,
        test_size=test_size,
        stratify=df['operation'],
        random_state=random_state
    )

    # Second split: train vs val (stratified by operation)
    val_ratio = val_size / (1 - test_size)
    train_files, val_files = train_test_split(
        train_val_files,
        test_size=val_ratio,
        stratify=train_val_files['operation'],
        random_state=random_state
    )

    # Convert back to file lists
    train_list = train_files['file'].tolist()
    val_list = val_files['file'].tolist()
    test_list = test_files['file'].tolist()

    # Verify no overlap
    train_names = set(f.name for f in train_list)
    val_names = set(f.name for f in val_list)
    test_names = set(f.name for f in test_list)

    assert len(train_names & val_names) == 0, "Train/val overlap detected!"
    assert len(train_names & test_names) == 0, "Train/test overlap detected!"
    assert len(val_names & test_names) == 0, "Val/test overlap detected!"

    print(f"\n✓ Split verification:")
    print(f"  Train files: {len(train_list)} ({len(train_list)/len(csv_files)*100:.1f}%)")
    print(f"  Val files:   {len(val_list)} ({len(val_list)/len(csv_files)*100:.1f}%)")
    print(f"  Test files:  {len(test_list)} ({len(test_list)/len(csv_files)*100:.1f}%)")
    print(f"  No overlap: ✓")

    # Show operation distribution per split
    print(f"\n✓ Operation distribution per split:")
    for split_name, split_files in [("Train", train_files), ("Val", val_files), ("Test", test_files)]:
        print(f"  {split_name}:")
        counts = split_files['operation'].value_counts()
        for op, count in counts.items():
            print(f"    {op}: {count} files ({count/len(split_files)*100:.1f}%)")

    return train_list, val_list, test_list


def process_file_split(
    files: List[Path],
    preprocessor: GCodePreprocessor,
    split_name: str
) -> List[Dict]:
    """
    Process a list of files into windows.

    Args:
        files: List of CSV files for this split
        preprocessor: Fitted preprocessor
        split_name: Name of split (for logging)

    Returns:
        List of window dictionaries
    """
    print(f"\nProcessing {split_name} split ({len(files)} files)...")
    all_windows = []

    for csv_path in files:
        print(f"  {csv_path.name}...", end=" ")
        windows = preprocessor.process_file(csv_path, fit_scaler=False)
        all_windows.extend(windows)
        print(f"{len(windows)} windows")

    print(f"  Total: {len(all_windows)} windows from {len(files)} files")
    return all_windows


def save_split_metadata(
    output_dir: Path,
    train_files: List[Path],
    val_files: List[Path],
    test_files: List[Path]
):
    """Save file split information for verification and reproducibility."""
    metadata = {
        'train_files': [str(f) for f in train_files],
        'val_files': [str(f) for f in val_files],
        'test_files': [str(f) for f in test_files],
        'train_operations': [extract_operation_type(f.name) for f in train_files],
        'val_operations': [extract_operation_type(f.name) for f in val_files],
        'test_operations': [extract_operation_type(f.name) for f in test_files],
    }

    split_file = output_dir / 'file_splits.json'
    with open(split_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ File split metadata saved to: {split_file}")


def preprocess_dataset_stratified(
    data_dir: Path,
    output_dir: Path,
    vocab_path: Path,
    window_size: int = 64,
    stride: int = 16,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    verify_no_leakage: bool = True
):
    """
    Preprocess dataset with file-level stratified splitting.

    This prevents data leakage by:
    1. Splitting files FIRST (not windows)
    2. Stratifying by operation type
    3. Processing each split separately
    4. Saving file lists for verification

    Args:
        data_dir: Directory containing CSV files
        output_dir: Output directory for processed data
        vocab_path: Path to vocabulary file
        window_size: Window size for sequences
        stride: Stride for sliding windows
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed
        verify_no_leakage: Run leakage verification checks
    """
    print("\n" + "="*80)
    print("FILE-LEVEL STRATIFIED PREPROCESSING")
    print("="*80)
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Vocabulary: {vocab_path}")
    print(f"Window size: {window_size}, Stride: {stride}")
    print(f"Split: {100*(1-test_size-val_size):.0f}% train / {val_size*100:.0f}% val / {test_size*100:.0f}% test")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSV files
    input_files = sorted(data_dir.glob('*_aligned.csv'))
    print(f"\nFound {len(input_files)} CSV files")

    if len(input_files) == 0:
        raise ValueError(f"No *_aligned.csv files found in {data_dir}")

    # STEP 1: FILE-LEVEL STRATIFIED SPLIT (CRITICAL!)
    train_files, val_files, test_files = stratified_file_split(
        input_files,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )

    # Save split metadata for reproducibility
    save_split_metadata(output_dir, train_files, val_files, test_files)

    # STEP 2: Initialize preprocessor with config
    config = get_default_config()
    config.window_size = window_size
    config.stride = stride

    # Determine master columns from first file
    first_df = pd.read_csv(input_files[0])
    exclude_cols = config.exclude_features
    master_columns = [col for col in first_df.columns if col not in exclude_cols]

    preprocessor = GCodePreprocessor(
        vocab_path,
        config=config,
        master_columns=master_columns
    )

    # STEP 3: Fit scaler on TRAIN files only (prevent test leakage)
    print("\nFitting scaler on TRAIN files only...")
    train_continuous_data = []
    for csv_path in train_files:
        df = preprocessor.load_csv(csv_path)
        continuous, _, _ = preprocessor.extract_features(df)
        train_continuous_data.append(continuous)

    combined_train = np.vstack(train_continuous_data)
    preprocessor.fit_scaler(combined_train)
    print(f"  Scaler fitted on {combined_train.shape} training data points")

    # STEP 4: Process each split separately
    train_windows = process_file_split(train_files, preprocessor, "train")
    val_windows = process_file_split(val_files, preprocessor, "val")
    test_windows = process_file_split(test_files, preprocessor, "test")

    # Metadata
    continuous_shape = train_windows[0]['continuous'].shape
    categorical_shape = train_windows[0]['categorical'].shape

    metadata = {
        'n_continuous_features': continuous_shape[1],
        'n_categorical_features': categorical_shape[1],
        'window_size': window_size,
        'stride': stride,
        'vocab_size': len(preprocessor.vocabulary),
        'n_train': len(train_windows),
        'n_val': len(val_windows),
        'n_test': len(test_windows),
        'n_train_files': len(train_files),
        'n_val_files': len(val_files),
        'n_test_files': len(test_files),
        'master_columns': master_columns,
        'stratified_by': 'operation_type',
        'file_level_split': True,  # Mark as proper splitting
    }

    # Save splits
    print(f"\n✓ Saving train set ({len(train_windows)} windows from {len(train_files)} files)...")
    preprocessor.save_processed(train_windows, output_dir / 'train_sequences.npz', metadata)

    print(f"✓ Saving val set ({len(val_windows)} windows from {len(val_files)} files)...")
    preprocessor.save_processed(val_windows, output_dir / 'val_sequences.npz', metadata)

    print(f"✓ Saving test set ({len(test_windows)} windows from {len(test_files)} files)...")
    preprocessor.save_processed(test_windows, output_dir / 'test_sequences.npz', metadata)

    print("\n" + "="*80)
    print("✓ PREPROCESSING COMPLETE - NO DATA LEAKAGE")
    print("="*80)
    print(f"Output: {output_dir}")
    print(f"  train_sequences.npz: {len(train_windows)} windows")
    print(f"  val_sequences.npz: {len(val_windows)} windows")
    print(f"  test_sequences.npz: {len(test_windows)} windows")
    print(f"  file_splits.json: File list metadata")

    if verify_no_leakage:
        verify_splits(output_dir)


def verify_splits(output_dir: Path):
    """Verify no data leakage in splits."""
    print("\n" + "="*80)
    print("VERIFYING NO DATA LEAKAGE")
    print("="*80)

    # Load split metadata
    with open(output_dir / 'file_splits.json') as f:
        splits = json.load(f)

    train_files = set(Path(f).name for f in splits['train_files'])
    val_files = set(Path(f).name for f in splits['val_files'])
    test_files = set(Path(f).name for f in splits['test_files'])

    # Check overlaps
    train_val_overlap = train_files & val_files
    train_test_overlap = train_files & test_files
    val_test_overlap = val_files & test_files

    if train_val_overlap:
        print(f"❌ LEAKAGE: Train/Val overlap: {train_val_overlap}")
        return False

    if train_test_overlap:
        print(f"❌ LEAKAGE: Train/Test overlap: {train_test_overlap}")
        return False

    if val_test_overlap:
        print(f"❌ LEAKAGE: Val/Test overlap: {val_test_overlap}")
        return False

    print("✓ No file overlap between splits")
    print(f"  Train files: {len(train_files)}")
    print(f"  Val files: {len(val_files)}")
    print(f"  Test files: {len(test_files)}")
    print(f"  Total unique: {len(train_files) + len(val_files) + len(test_files)}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="File-level stratified preprocessing (prevents data leakage)"
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing CSV files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--vocab-path', type=str, required=True,
                        help='Path to vocabulary JSON file')
    parser.add_argument('--window-size', type=int, default=64,
                        help='Window size for sequences (default: 64)')
    parser.add_argument('--stride', type=int, default=16,
                        help='Stride for sliding windows (default: 16)')
    parser.add_argument('--test-size', type=float, default=0.15,
                        help='Test set fraction (default: 0.15)')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='Validation set fraction (default: 0.15)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip leakage verification')

    args = parser.parse_args()

    preprocess_dataset_stratified(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        vocab_path=Path(args.vocab_path),
        window_size=args.window_size,
        stride=args.stride,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        verify_no_leakage=not args.no_verify
    )


if __name__ == '__main__':
    main()
