"""
Statistical exploration and validation of raw G-code sensor data.

This module provides comprehensive analysis of RAW CSV data BEFORE preprocessing:
- Statistical summaries per sensor
- Missing value and outlier detection
- Feature correlation analysis
- Distribution analysis (normality, skewness, kurtosis)
- Temporal pattern detection
- Data quality checks
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from collections import Counter
import warnings

__all__ = [
    "RawDataExplorer",
    "compute_sensor_statistics",
    "detect_outliers",
    "analyze_correlations",
    "analyze_distributions",
    "detect_temporal_patterns",
]


class RawDataExplorer:
    """Comprehensive explorer for raw G-code sensor data."""

    def __init__(self, csv_files: List[Path]):
        """
        Initialize explorer with list of CSV files.

        Args:
            csv_files: List of paths to aligned CSV files
        """
        self.csv_files = csv_files
        self.data_frames = {}
        self.combined_df = None
        self.statistics = {}

        # Define column categories (same as preprocessing.py)
        self.exclude_cols = [
            'time', 'gcode_line_num', 'gcode_text', 'gcode_tokens',
            't_console', 'gcode_line', 'gcode_string', 'raw_json',
            'vel', 'plane', 'line', 'posx', 'posy', 'posz', 'feed', 'momo'
        ]
        self.cat_cols = ['stat', 'unit', 'dist', 'coor']

    def load_all_data(self, sample_rate: Optional[float] = None) -> None:
        """
        Load all CSV files into memory.

        Args:
            sample_rate: Optional fraction to sample (e.g., 0.1 for 10%)
        """
        print(f"Loading {len(self.csv_files)} CSV files...")

        for csv_path in self.csv_files:
            df = pd.read_csv(csv_path)

            # Optional sampling for large datasets
            if sample_rate and sample_rate < 1.0:
                df = df.sample(frac=sample_rate, random_state=42)

            self.data_frames[csv_path.stem] = df

        # Combine all dataframes
        self.combined_df = pd.concat(self.data_frames.values(), ignore_index=True)
        print(f"  Loaded {len(self.combined_df):,} total rows from {len(self.csv_files)} files")

    def identify_feature_columns(self) -> Dict[str, List[str]]:
        """
        Identify and categorize feature columns.

        Returns:
            Dictionary with 'continuous', 'categorical', 'gcode', 'excluded' lists
        """
        if self.combined_df is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")

        all_cols = set(self.combined_df.columns)

        # Categorical features
        categorical = [col for col in self.cat_cols if col in all_cols]

        # G-code column
        gcode_col = None
        if 'gcode_text' in all_cols:
            gcode_col = 'gcode_text'
        elif 'gcode_string' in all_cols:
            gcode_col = 'gcode_string'

        # Continuous features (everything else that's numeric and not excluded)
        continuous = []
        for col in all_cols:
            if col in self.exclude_cols or col in categorical or col == gcode_col:
                continue
            if pd.api.types.is_numeric_dtype(self.combined_df[col]):
                continuous.append(col)

        continuous = sorted(continuous)

        # Excluded columns
        excluded = [col for col in self.exclude_cols if col in all_cols]

        return {
            'continuous': continuous,
            'categorical': categorical,
            'gcode': gcode_col,
            'excluded': excluded,
        }

    def compute_sensor_statistics(self) -> pd.DataFrame:
        """
        Compute comprehensive statistics for all sensor columns.

        Returns:
            DataFrame with statistics per sensor
        """
        feature_cols = self.identify_feature_columns()
        continuous_cols = feature_cols['continuous']

        stats_list = []

        for col in continuous_cols:
            data = self.combined_df[col].dropna()

            if len(data) == 0:
                continue

            stats_dict = {
                'sensor': col,
                'count': len(data),
                'missing_count': self.combined_df[col].isna().sum(),
                'missing_pct': 100 * self.combined_df[col].isna().sum() / len(self.combined_df),
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'q25': data.quantile(0.25),
                'median': data.median(),
                'q75': data.quantile(0.75),
                'max': data.max(),
                'range': data.max() - data.min(),
                'iqr': data.quantile(0.75) - data.quantile(0.25),
                'cv': data.std() / data.mean() if data.mean() != 0 else np.nan,  # Coefficient of variation
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
            }
            stats_list.append(stats_dict)

        stats_df = pd.DataFrame(stats_list)
        self.statistics['sensor_stats'] = stats_df

        return stats_df

    def detect_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, np.ndarray]:
        """
        Detect outliers in sensor data.

        Args:
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier (1.5) or Z-score threshold (3.0)

        Returns:
            Dictionary mapping sensor names to boolean outlier masks
        """
        feature_cols = self.identify_feature_columns()
        continuous_cols = feature_cols['continuous']

        outlier_masks = {}
        outlier_counts = {}

        for col in continuous_cols:
            data = self.combined_df[col].dropna()

            if len(data) == 0:
                continue

            if method == 'iqr':
                q25 = data.quantile(0.25)
                q75 = data.quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - threshold * iqr
                upper_bound = q75 + threshold * iqr
                mask = (self.combined_df[col] < lower_bound) | (self.combined_df[col] > upper_bound)

            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                mask = pd.Series(False, index=self.combined_df.index)
                mask.loc[data.index] = z_scores > threshold

            else:
                raise ValueError(f"Unknown method: {method}")

            outlier_masks[col] = mask
            outlier_counts[col] = mask.sum()

        self.statistics['outlier_counts'] = outlier_counts

        return outlier_masks

    def analyze_correlations(self, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
        """
        Analyze feature correlations and identify highly correlated pairs.

        Args:
            threshold: Correlation threshold for identifying redundant features

        Returns:
            correlation_matrix: DataFrame with pairwise correlations
            high_corr_pairs: List of (sensor1, sensor2) pairs with |corr| > threshold
        """
        feature_cols = self.identify_feature_columns()
        continuous_cols = feature_cols['continuous']

        # Compute correlation matrix
        corr_matrix = self.combined_df[continuous_cols].corr()

        # Find highly correlated pairs
        high_corr_pairs = []
        n_features = len(continuous_cols)

        for i in range(n_features):
            for j in range(i+1, n_features):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr_pairs.append((
                        continuous_cols[i],
                        continuous_cols[j],
                        corr_val
                    ))

        self.statistics['correlation_matrix'] = corr_matrix
        self.statistics['high_correlations'] = high_corr_pairs

        return corr_matrix, high_corr_pairs

    def analyze_distributions(self) -> pd.DataFrame:
        """
        Analyze statistical distributions of each sensor.

        Tests for normality and computes distribution metrics.

        Returns:
            DataFrame with distribution analysis per sensor
        """
        feature_cols = self.identify_feature_columns()
        continuous_cols = feature_cols['continuous']

        dist_list = []

        for col in continuous_cols:
            data = self.combined_df[col].dropna()

            if len(data) < 3:
                continue

            # Shapiro-Wilk test for normality (use sample if too large)
            if len(data) > 5000:
                sample_data = data.sample(n=5000, random_state=42)
            else:
                sample_data = data

            try:
                shapiro_stat, shapiro_p = stats.shapiro(sample_data)
            except:
                shapiro_stat, shapiro_p = np.nan, np.nan

            # Kolmogorov-Smirnov test against normal distribution
            try:
                ks_stat, ks_p = stats.kstest(sample_data, 'norm', args=(data.mean(), data.std()))
            except:
                ks_stat, ks_p = np.nan, np.nan

            dist_dict = {
                'sensor': col,
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'is_normal_shapiro': shapiro_p > 0.05 if not np.isnan(shapiro_p) else None,
                'ks_stat': ks_stat,
                'ks_p': ks_p,
                'is_normal_ks': ks_p > 0.05 if not np.isnan(ks_p) else None,
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'is_skewed': abs(stats.skew(data)) > 1.0,
            }
            dist_list.append(dist_dict)

        dist_df = pd.DataFrame(dist_list)
        self.statistics['distributions'] = dist_df

        return dist_df

    def detect_temporal_patterns(self, max_lag: int = 50) -> Dict[str, np.ndarray]:
        """
        Detect temporal patterns using autocorrelation analysis.

        Args:
            max_lag: Maximum lag for autocorrelation computation

        Returns:
            Dictionary mapping sensor names to autocorrelation arrays
        """
        feature_cols = self.identify_feature_columns()
        continuous_cols = feature_cols['continuous']

        autocorr_results = {}

        for col in continuous_cols:
            data = self.combined_df[col].dropna()

            if len(data) < max_lag * 2:
                continue

            # Compute autocorrelation
            autocorr = np.correlate(data - data.mean(), data - data.mean(), mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
            autocorr = autocorr / autocorr[0]  # Normalize
            autocorr = autocorr[:max_lag+1]

            autocorr_results[col] = autocorr

        self.statistics['autocorrelation'] = autocorr_results

        return autocorr_results

    def check_data_quality(self) -> Dict[str, any]:
        """
        Comprehensive data quality check.

        Returns:
            Dictionary with quality metrics and warnings
        """
        quality_report = {
            'total_rows': len(self.combined_df),
            'total_files': len(self.data_frames),
            'warnings': [],
            'recommendations': [],
        }

        feature_cols = self.identify_feature_columns()
        continuous_cols = feature_cols['continuous']

        # Check for missing values
        missing_pct = self.combined_df[continuous_cols].isna().sum() / len(self.combined_df) * 100
        high_missing = missing_pct[missing_pct > 10]

        if len(high_missing) > 0:
            quality_report['warnings'].append(
                f"{len(high_missing)} sensors have >10% missing values: {high_missing.to_dict()}"
            )
            quality_report['recommendations'].append(
                "Consider imputation or removal of high-missing sensors"
            )

        # Check for zero-variance features
        zero_var_cols = []
        for col in continuous_cols:
            if self.combined_df[col].std() == 0:
                zero_var_cols.append(col)

        if zero_var_cols:
            quality_report['warnings'].append(
                f"{len(zero_var_cols)} sensors have zero variance: {zero_var_cols}"
            )
            quality_report['recommendations'].append(
                "Remove zero-variance features (they provide no information)"
            )

        # Check for high correlations
        if 'high_correlations' in self.statistics:
            high_corr = self.statistics['high_correlations']
            if high_corr:
                quality_report['warnings'].append(
                    f"Found {len(high_corr)} highly correlated sensor pairs (|r| > 0.95)"
                )
                quality_report['recommendations'].append(
                    "Consider removing redundant sensors to reduce multicollinearity"
                )

        # Check for severe outliers
        if 'outlier_counts' in self.statistics:
            outlier_counts = self.statistics['outlier_counts']
            severe_outliers = {k: v for k, v in outlier_counts.items() if v > len(self.combined_df) * 0.05}
            if severe_outliers:
                quality_report['warnings'].append(
                    f"{len(severe_outliers)} sensors have >5% outliers"
                )
                quality_report['recommendations'].append(
                    "Investigate outliers - may indicate sensor malfunction or interesting events"
                )

        # Check file size balance
        file_sizes = {name: len(df) for name, df in self.data_frames.items()}
        size_std = np.std(list(file_sizes.values()))
        size_mean = np.mean(list(file_sizes.values()))

        if size_std / size_mean > 0.5:
            quality_report['warnings'].append(
                f"File sizes are imbalanced (CV={size_std/size_mean:.2f})"
            )
            quality_report['file_sizes'] = file_sizes

        return quality_report

    def analyze_per_file_statistics(self) -> pd.DataFrame:
        """
        Compute statistics for each individual file.

        Returns:
            DataFrame with per-file statistics
        """
        file_stats = []

        for file_name, df in self.data_frames.items():
            feature_cols = self.identify_feature_columns()
            continuous_cols = [col for col in feature_cols['continuous'] if col in df.columns]

            # Basic stats
            stats_dict = {
                'file': file_name,
                'num_rows': len(df),
                'num_sensors': len(continuous_cols),
            }

            # Missing values
            if continuous_cols:
                stats_dict['total_missing'] = df[continuous_cols].isna().sum().sum()
                stats_dict['missing_pct'] = 100 * stats_dict['total_missing'] / (len(df) * len(continuous_cols))

            # G-code commands
            gcode_col = feature_cols['gcode']
            if gcode_col and gcode_col in df.columns:
                stats_dict['num_unique_gcodes'] = df[gcode_col].nunique()
                stats_dict['most_common_gcode'] = df[gcode_col].mode()[0] if len(df) > 0 else None

            # Temporal info (if time column exists)
            if 'time' in df.columns:
                time_diffs = df['time'].diff().dropna()
                stats_dict['mean_time_step'] = time_diffs.mean()
                stats_dict['std_time_step'] = time_diffs.std()

            file_stats.append(stats_dict)

        file_stats_df = pd.DataFrame(file_stats)
        self.statistics['per_file_stats'] = file_stats_df

        return file_stats_df


def compute_sensor_statistics(data: pd.DataFrame, sensor_col: str) -> Dict:
    """
    Compute comprehensive statistics for a single sensor.

    Args:
        data: DataFrame containing sensor data
        sensor_col: Name of sensor column

    Returns:
        Dictionary with statistics
    """
    series = data[sensor_col].dropna()

    return {
        'count': len(series),
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'q25': series.quantile(0.25),
        'median': series.median(),
        'q75': series.quantile(0.75),
        'max': series.max(),
        'skewness': stats.skew(series),
        'kurtosis': stats.kurtosis(series),
    }


def detect_outliers(data: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers in a single column.

    Args:
        data: DataFrame
        column: Column name
        method: 'iqr' or 'zscore'
        threshold: Detection threshold

    Returns:
        Boolean mask indicating outliers
    """
    series = data[column].dropna()

    if method == 'iqr':
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        lower = q25 - threshold * iqr
        upper = q75 + threshold * iqr
        mask = (data[column] < lower) | (data[column] > upper)
    elif method == 'zscore':
        z = np.abs(stats.zscore(series))
        mask = pd.Series(False, index=data.index)
        mask.loc[series.index] = z > threshold
    else:
        raise ValueError(f"Unknown method: {method}")

    return mask.values


def analyze_correlations(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Compute correlation matrix for specified columns.

    Args:
        data: DataFrame
        columns: List of column names

    Returns:
        Correlation matrix
    """
    return data[columns].corr()


def analyze_distributions(data: pd.DataFrame, column: str) -> Dict:
    """
    Analyze distribution of a single column.

    Args:
        data: DataFrame
        column: Column name

    Returns:
        Dictionary with distribution metrics
    """
    series = data[column].dropna()

    # Sample if too large
    if len(series) > 5000:
        sample = series.sample(n=5000, random_state=42)
    else:
        sample = series

    try:
        shapiro_stat, shapiro_p = stats.shapiro(sample)
    except:
        shapiro_stat, shapiro_p = np.nan, np.nan

    return {
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'is_normal': shapiro_p > 0.05 if not np.isnan(shapiro_p) else None,
        'skewness': stats.skew(series),
        'kurtosis': stats.kurtosis(series),
    }


def detect_temporal_patterns(data: pd.DataFrame, column: str, max_lag: int = 50) -> np.ndarray:
    """
    Compute autocorrelation for temporal pattern detection.

    Args:
        data: DataFrame
        column: Column name
        max_lag: Maximum lag

    Returns:
        Autocorrelation array
    """
    series = data[column].dropna()

    # Compute autocorrelation
    autocorr = np.correlate(series - series.mean(), series - series.mean(), mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]
    autocorr = autocorr[:max_lag+1]

    return autocorr
