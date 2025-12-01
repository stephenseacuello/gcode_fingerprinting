"""
Preprocessing for G-code sensor data with advanced tokenizer.

Converts aligned CSV files into sequences for training:
- Tokenizes G-code commands using advanced tokenizer
- Creates sliding windows from continuous sensor data
- Splits into train/val/test sets
- Saves processed data with metadata
- Flexible preprocessing strategies based on configuration
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import argparse

# Import the advanced tokenizer
from miracle.utilities.gcode_tokenizer import (
    GCodeTokenizer as AdvancedGCodeTokenizer,
    TokenizerConfig,
)

# Import configuration and utilities
from miracle.config.preprocessing_config import PreprocessingConfig, get_default_config
from miracle.dataset.preprocessing_utils import (
    get_scaler,
    handle_missing_values,
    clip_outliers,
    remove_zero_variance_features,
    remove_high_correlation_features,
    remove_high_missing_features,
    apply_log_transform_skewed,
    add_velocity_features,
    add_rolling_statistics,
    get_column_statistics,
)

__all__ = ["GCodePreprocessor", "load_vocabulary", "preprocess_dataset", "extract_operation_type"]


def extract_operation_type(filename: str) -> str:
    """
    Extract operation type from G-code filename.

    Args:
        filename: Name of G-code or CSV file

    Returns:
        Operation type (one of 9 classes + unknown):
        'adaptive', 'adaptive150025', 'face', 'face150025', 'pocket', 'pocket150025',
        'damageadaptive', 'damageface', 'damagepocket', or 'unknown'
    """
    fname_lower = filename.lower()

    # CRITICAL: Check most specific patterns FIRST to avoid greedy matching
    # Order: {damage}{operation}{150025} variants before base {operation}

    # 150025 variants (6 types)
    if 'adaptive150025' in fname_lower or 'adaptive_150025' in fname_lower:
        return 'adaptive150025'
    elif 'face150025' in fname_lower or 'face_150025' in fname_lower:
        return 'face150025'
    elif 'pocket150025' in fname_lower or 'pocket_150025' in fname_lower:
        return 'pocket150025'

    # Damage variants (3 types)
    elif 'damageadaptive' in fname_lower or 'damage_adaptive' in fname_lower:
        return 'damageadaptive'
    elif 'damageface' in fname_lower or 'damage_face' in fname_lower:
        return 'damageface'
    elif 'damagepocket' in fname_lower or 'damage_pocket' in fname_lower:
        return 'damagepocket'

    # Base types (3 types) - MUST come last
    elif 'adaptive' in fname_lower:
        return 'adaptive'
    elif 'face' in fname_lower:
        return 'face'
    elif 'pocket' in fname_lower:
        return 'pocket'

    else:
        return 'unknown'


def load_vocabulary(vocab_path: Path) -> Dict[str, int]:
    """Load G-code vocabulary from JSON."""
    with open(vocab_path) as f:
        data = json.load(f)
    return data['vocab']


class GCodePreprocessor:
    """Preprocess G-code sensor data into training sequences with flexible configuration."""

    def __init__(
        self,
        vocab_path: Path,
        config: Optional[PreprocessingConfig] = None,
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
        master_columns: Optional[List[str]] = None,
    ):
        """
        Initialize preprocessor with configuration.

        Args:
            vocab_path: Path to G-code vocabulary
            config: PreprocessingConfig instance (if None, uses default)
            window_size: Override config window_size (deprecated, use config instead)
            stride: Override config stride (deprecated, use config instead)
            master_columns: Master column list for consistent dimensions
        """
        # Load or create config
        if config is None:
            config = get_default_config()
        self.config = config

        # Override config if legacy parameters provided
        if window_size is not None:
            self.config.window_size = window_size
        if stride is not None:
            self.config.stride = stride

        self.window_size = self.config.window_size
        self.stride = self.config.stride
        self.vocab_path = vocab_path

        # Load the advanced tokenizer
        self.tokenizer = AdvancedGCodeTokenizer.load(vocab_path)
        self.vocabulary = self.tokenizer.vocab

        # Feature scaler (configurable)
        self.continuous_scaler = get_scaler(
            self.config.scaler_type,
            output_distribution=self.config.quantile_output_distribution
        )
        self.fitted = False
        self.column_stats = None  # For consistent NaN handling across splits

        # Master column list for consistent feature dimensions across all files
        self.master_columns = master_columns  # Will be set during preprocessing
        self.selected_columns = None  # Columns after feature selection

        # Print config summary
        print(self.config.get_preprocessing_summary())

    def load_csv(self, csv_path: Path) -> pd.DataFrame:
        """Load aligned CSV file."""
        df = pd.read_csv(csv_path)
        return df

    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract continuous and categorical features from DataFrame with config-based preprocessing.

        Returns:
            continuous_features: [T, n_continuous]
            categorical_features: [T, n_categorical]
            gcode_texts: List of G-code strings per timestep
        """
        # Use configured exclusion list
        exclude_cols = self.config.exclude_features

        # Categorical features from config
        cat_cols = [col for col in df.columns if col in self.config.categorical_features]
        if cat_cols:
            categorical_features = df[cat_cols].values.astype(np.int64)
        else:
            # Create dummy categorical if none found
            categorical_features = np.zeros((len(df), 1), dtype=np.int64)

        # Continuous features: Use master column list if available, otherwise auto-detect
        if self.master_columns is not None:
            # Use the master column list for consistent dimensions across all files
            cont_cols = self.master_columns

            # Create feature matrix with zeros for missing columns
            continuous_features = np.zeros((len(df), len(cont_cols)), dtype=np.float32)

            # Fill in values for columns that exist in this file
            for i, col in enumerate(cont_cols):
                if col in df.columns:
                    continuous_features[:, i] = df[col].values.astype(np.float32)
                # else: remains zero (padding for missing sensor)
        else:
            # Auto-detect mode (for initial scan or single-file processing)
            core_features = [
                # Machine positions (measured)
                'mpox', 'mpoy', 'mpoz',
                # Motor currents
                'spindle', 'x_motor', 'y_motor', 'z_motor',
                'spindle_A', 'x_motor_A', 'y_motor_A', 'z_motor_A'
            ]

            cont_cols = []
            for col in core_features:
                if col in df.columns:
                    cont_cols.append(col)

            # Add any sensor columns that exist (IMU data, etc.)
            for col in df.columns:
                if col not in exclude_cols and col not in cat_cols and col not in cont_cols:
                    # Check if column is numeric
                    if pd.api.types.is_numeric_dtype(df[col]):
                        cont_cols.append(col)

            # Sort columns for consistency
            cont_cols = sorted(cont_cols)

            # Extract columns that exist in this file
            continuous_features = df[cont_cols].values.astype(np.float32)

        # Handle NaN values using configured strategy
        nan_count = np.isnan(continuous_features).sum()
        if nan_count > 0:
            print(f"    Found {nan_count} NaN values, handling with '{self.config.nan_strategy}' strategy")
            continuous_features = handle_missing_values(
                continuous_features,
                strategy=self.config.nan_strategy,
                column_stats=self.column_stats  # Will be None on first pass, set after fitting
            )

        # G-code text (check both possible column names)
        if 'gcode_text' in df.columns:
            gcode_texts = df['gcode_text'].tolist()
        elif 'gcode_string' in df.columns:
            gcode_texts = df['gcode_string'].tolist()
        else:
            gcode_texts = None

        return continuous_features, categorical_features, gcode_texts

    def create_windows(
        self,
        continuous: np.ndarray,
        categorical: np.ndarray,
        gcode_texts: List[str],
        operation_type: str = 'unknown',
    ) -> List[Dict]:
        """
        Create sliding windows from data.

        Args:
            continuous: Continuous features
            categorical: Categorical features
            gcode_texts: G-code text per timestep
            operation_type: Operation type for all windows from this file

        Returns:
            List of dictionaries with windowed data
        """
        T = len(continuous)
        windows = []

        for start_idx in range(0, T - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size

            # Extract window
            cont_window = continuous[start_idx:end_idx]
            cat_window = categorical[start_idx:end_idx]
            gcode_window = gcode_texts[start_idx:end_idx] if gcode_texts else None

            # Get unique G-code commands in this window (for label)
            if gcode_window:
                unique_gcodes = list(set(gcode_window))
                # Use the most frequent one as the label
                gcode_label = max(set(gcode_window), key=gcode_window.count)

                # Tokenize using advanced tokenizer
                token_ids = self.tokenizer.encode([gcode_label], add_bos_eos=False)
            else:
                gcode_label = ""
                token_ids = []

            windows.append({
                'continuous': cont_window,
                'categorical': cat_window,
                'gcode_text': gcode_label,
                'token_ids': token_ids,
                'length': self.window_size,
                'operation_type': operation_type,
            })

        return windows

    def fit_scaler(self, continuous_data: np.ndarray):
        """Fit scaler on continuous features with outlier handling."""
        # Compute column statistics for consistent NaN handling across splits
        self.column_stats = get_column_statistics(continuous_data)

        # Apply outlier clipping if configured (before fitting scaler)
        if self.config.outlier_method == 'clip':
            print(f"  Clipping outliers (threshold={self.config.outlier_threshold}*IQR)")
            continuous_data = clip_outliers(continuous_data, threshold=self.config.outlier_threshold)

        # Fit scaler if not 'none'
        if self.continuous_scaler is not None:
            self.continuous_scaler.fit(continuous_data)

        self.fitted = True

    def transform(self, continuous_data: np.ndarray) -> np.ndarray:
        """Normalize continuous features with outlier handling."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")

        # Apply outlier clipping if configured (use same bounds as training)
        if self.config.outlier_method == 'clip':
            continuous_data = clip_outliers(continuous_data, threshold=self.config.outlier_threshold)

        # Transform if scaler exists
        if self.continuous_scaler is not None:
            return self.continuous_scaler.transform(continuous_data)
        else:
            return continuous_data  # No scaling

    def process_file(self, csv_path: Path, fit_scaler: bool = False) -> List[Dict]:
        """Process a single CSV file into windows."""
        # Load data
        df = self.load_csv(csv_path)

        # Extract operation type from filename
        operation_type = extract_operation_type(csv_path.name)

        # Extract features
        continuous, categorical, gcode_texts = self.extract_features(df)

        # Fit scaler if needed
        if fit_scaler:
            self.fit_scaler(continuous)

        # Normalize
        if self.fitted:
            continuous = self.transform(continuous)

        # Create windows
        windows = self.create_windows(continuous, categorical, gcode_texts, operation_type)

        return windows

    def save_processed(
        self,
        windows: List[Dict],
        output_path: Path,
        metadata: Optional[Dict] = None,
    ):
        """Save processed windows to .npz file."""
        # Stack windows
        continuous_data = np.stack([w['continuous'] for w in windows])
        categorical_data = np.stack([w['categorical'] for w in windows])

        # Pad token sequences to same length
        max_token_len = max(len(w['token_ids']) for w in windows)
        token_data = []
        pad_token_id = self.tokenizer.cfg.special.get('PAD', 0)
        for w in windows:
            tokens = w['token_ids']
            padded = tokens + [pad_token_id] * (max_token_len - len(tokens))
            token_data.append(padded)
        token_data = np.array(token_data, dtype=np.int64)

        lengths = np.array([w['length'] for w in windows], dtype=np.int64)
        gcode_texts = [w['gcode_text'] for w in windows]

        # Operation type mapping and extraction
        # Fixed order to ensure consistency across train/val/test
        # All 9 operation types + unknown
        operation_type_mapping = {
            'adaptive': 0,
            'adaptive150025': 1,
            'face': 2,
            'face150025': 3,
            'pocket': 4,
            'pocket150025': 5,
            'damageadaptive': 6,
            'damageface': 7,
            'damagepocket': 8,
            'unknown': 9,
        }
        operation_types = [w['operation_type'] for w in windows]
        operation_type_ids = np.array([operation_type_mapping[op] for op in operation_types], dtype=np.int64)

        # Save
        np.savez(
            output_path,
            continuous=continuous_data,
            categorical=categorical_data,
            tokens=token_data,
            lengths=lengths,
            gcode_texts=np.array(gcode_texts, dtype=object),
            operation_type=operation_type_ids,
            operation_type_names=np.array(operation_types, dtype=object),
        )

        # Save metadata
        if metadata:
            metadata_path = output_path.parent / (output_path.stem + '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)


def preprocess_dataset(
    input_files: List[Path],
    output_dir: Path,
    vocab_path: Path,
    config: Optional[PreprocessingConfig] = None,
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
    train_frac: Optional[float] = None,
    val_frac: Optional[float] = None,
):
    """
    Preprocess multiple CSV files into train/val/test splits with flexible configuration.

    Args:
        input_files: List of CSV file paths
        output_dir: Directory to save processed data
        vocab_path: Path to vocabulary JSON
        config: PreprocessingConfig (if None, uses default)
        window_size: Override config window_size (deprecated)
        stride: Override config stride (deprecated)
        train_frac: Override config train_frac (deprecated)
        val_frac: Override config val_frac (deprecated)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create config
    if config is None:
        config = get_default_config()

    # Override config with legacy parameters if provided
    if window_size is not None:
        config.window_size = window_size
    if stride is not None:
        config.stride = stride
    if train_frac is not None:
        config.train_frac = train_frac
    if val_frac is not None:
        config.val_frac = val_frac

    print("\n" + "="*60)
    print("PREPROCESSING CONFIGURATION")
    print("="*60)
    print(config.get_preprocessing_summary())
    print("="*60 + "\n")

    # STEP 1: Scan all files to build master column list
    print("STEP 1: Scanning files to build master feature list...")
    all_continuous_cols = set()

    exclude_cols = config.exclude_features
    cat_cols_names = config.categorical_features

    for csv_path in input_files:
        df = pd.read_csv(csv_path, nrows=1)  # Just read header
        for col in df.columns:
            if col not in exclude_cols and col not in cat_cols_names:
                if pd.api.types.is_numeric_dtype(df[col]):
                    all_continuous_cols.add(col)

    master_columns = sorted(list(all_continuous_cols))
    print(f"  Found {len(master_columns)} total continuous features across all files")

    # STEP 1.5: Apply feature selection on metadata (before loading all data)
    print("\nSTEP 1.5: Applying feature selection...")

    # Load first file to check for high-missing and zero-variance features
    first_df = pd.read_csv(input_files[0])

    # Remove high-missing features
    if config.max_missing_pct < 100:
        first_df_cont = first_df[[col for col in master_columns if col in first_df.columns]]
        # Only check columns that exist in this file
        cols_in_file = [col for col in master_columns if col in first_df.columns]
        first_df_cont, kept_cols, removed_cols = remove_high_missing_features(
            first_df_cont, cols_in_file, threshold_pct=config.max_missing_pct
        )
        master_columns = kept_cols

    # Remove zero/low-variance features
    if config.remove_zero_variance:
        first_df_cont = first_df[[col for col in master_columns if col in first_df.columns]]
        # Only check columns that exist in this file
        cols_in_file = [col for col in master_columns if col in first_df.columns]
        first_df_cont, kept_cols, removed_cols = remove_zero_variance_features(
            first_df_cont, cols_in_file, threshold=config.variance_threshold
        )
        master_columns = kept_cols

    # Remove highly correlated features
    if config.correlation_threshold < 1.0:
        # Load more data for better correlation estimation
        sample_df = pd.concat([pd.read_csv(f, nrows=1000) for f in input_files[:min(5, len(input_files))]])
        sample_df_cont = sample_df[[col for col in master_columns if col in sample_df.columns]]
        # Only check columns that exist in the sample data
        cols_in_sample = [col for col in master_columns if col in sample_df.columns]
        sample_df_cont, kept_cols, removed_cols = remove_high_correlation_features(
            sample_df_cont, cols_in_sample, threshold=config.correlation_threshold
        )
        master_columns = kept_cols

    print(f"  After feature selection: {len(master_columns)} features remaining")

    # STEP 2: Initialize preprocessor with master column list and config
    preprocessor = GCodePreprocessor(vocab_path, config=config, master_columns=master_columns)

    # STEP 3: Fit scaler on ALL files (to handle different sensor ranges)
    print("Fitting scaler on all files...")
    all_continuous_data = []
    for csv_path in input_files:
        df = preprocessor.load_csv(csv_path)
        continuous, _, _ = preprocessor.extract_features(df)
        all_continuous_data.append(continuous)

    # Concatenate and fit scaler
    combined_data = np.vstack(all_continuous_data)
    preprocessor.fit_scaler(combined_data)
    print(f"  Scaler fitted on {combined_data.shape} total data points")

    # STEP 4: Process all files with consistent dimensions and normalization
    all_windows = []
    for csv_path in input_files:
        print(f"Processing {csv_path.name}...")
        windows = preprocessor.process_file(csv_path, fit_scaler=False)  # Already fitted above
        all_windows.extend(windows)

    print(f"Created {len(all_windows)} windows from {len(input_files)} files")

    # Split into train/val/test
    n_total = len(all_windows)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    # Shuffle
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_windows = [all_windows[i] for i in train_indices]
    val_windows = [all_windows[i] for i in val_indices]
    test_windows = [all_windows[i] for i in test_indices]

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
        'master_columns': master_columns,  # Save for reference
        'excluded_columns': list(exclude_cols),  # Document what was removed
        'categorical_columns': cat_cols_names,
    }

    # Save splits
    print(f"Saving train set ({len(train_windows)} samples)...")
    preprocessor.save_processed(train_windows, output_dir / 'train_sequences.npz', metadata)

    print(f"Saving val set ({len(val_windows)} samples)...")
    preprocessor.save_processed(val_windows, output_dir / 'val_sequences.npz', metadata)

    print(f"Saving test set ({len(test_windows)} samples)...")
    preprocessor.save_processed(test_windows, output_dir / 'test_sequences.npz', metadata)

    print(f"\n✅ Preprocessing complete!")
    print(f"Output directory: {output_dir}")
    print(f"Train: {len(train_windows)}, Val: {len(val_windows)}, Test: {len(test_windows)}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess G-code sensor data")
    parser.add_argument('--data-dir', type=Path, required=True, help="Directory with CSV files")
    parser.add_argument('--output-dir', type=Path, required=True, help="Output directory")
    parser.add_argument('--vocab-path', type=Path, required=True, help="Vocabulary JSON file")
    parser.add_argument('--window-size', type=int, default=64, help="Window size")
    parser.add_argument('--stride', type=int, default=16, help="Stride for sliding window")
    parser.add_argument('--train-frac', type=float, default=0.7, help="Training fraction")
    parser.add_argument('--val-frac', type=float, default=0.15, help="Validation fraction")

    args = parser.parse_args()

    # Find all CSV files
    csv_files = sorted(args.data_dir.glob("*_aligned.csv"))
    print(f"Found {len(csv_files)} CSV files")

    # Preprocess
    preprocess_dataset(
        input_files=csv_files,
        output_dir=args.output_dir,
        vocab_path=args.vocab_path,
        window_size=args.window_size,
        stride=args.stride,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()
