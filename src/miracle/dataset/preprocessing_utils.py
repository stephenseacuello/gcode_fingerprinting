"""
Utility functions for data preprocessing.

Provides flexible preprocessing strategies based on configuration.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Set
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
    MinMaxScaler,
)
from scipy import stats


def get_scaler(scaler_type: str, **kwargs):
    """
    Get scaler instance based on type.

    Args:
        scaler_type: One of 'standard', 'robust', 'quantile', 'minmax', 'none'
        **kwargs: Additional arguments for the scaler

    Returns:
        Scaler instance or None if scaler_type='none'
    """
    if scaler_type == 'standard':
        return StandardScaler()
    elif scaler_type == 'robust':
        return RobustScaler()
    elif scaler_type == 'quantile':
        output_dist = kwargs.get('output_distribution', 'uniform')
        return QuantileTransformer(output_distribution=output_dist, random_state=42)
    elif scaler_type == 'minmax':
        return MinMaxScaler()
    elif scaler_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")


def handle_missing_values(
    data: np.ndarray,
    strategy: str = 'zero',
    column_stats: dict = None
) -> np.ndarray:
    """
    Handle missing values in data.

    Args:
        data: Data array [T, D]
        strategy: One of 'zero', 'forward_fill', 'interpolate', 'median', 'mean'
        column_stats: Dict with 'median' and 'mean' per column (for median/mean strategies)

    Returns:
        Data with NaN values handled
    """
    data = data.copy()

    if strategy == 'zero':
        data = np.nan_to_num(data, nan=0.0)

    elif strategy == 'forward_fill':
        # Forward fill per column
        df = pd.DataFrame(data)
        df = df.fillna(method='ffill')
        # If still NaN at start, backfill
        df = df.fillna(method='bfill')
        # If still NaN, fill with 0
        df = df.fillna(0.0)
        data = df.values

    elif strategy == 'interpolate':
        df = pd.DataFrame(data)
        df = df.interpolate(method='linear', limit_direction='both')
        df = df.fillna(0.0)  # Fallback for remaining NaNs
        data = df.values

    elif strategy == 'median':
        if column_stats is None:
            # Compute median per column
            medians = np.nanmedian(data, axis=0)
        else:
            medians = column_stats['median']

        # Replace NaN with median
        for col_idx in range(data.shape[1]):
            col_mask = np.isnan(data[:, col_idx])
            data[col_mask, col_idx] = medians[col_idx]

    elif strategy == 'mean':
        if column_stats is None:
            means = np.nanmean(data, axis=0)
        else:
            means = column_stats['mean']

        for col_idx in range(data.shape[1]):
            col_mask = np.isnan(data[:, col_idx])
            data[col_mask, col_idx] = means[col_idx]

    else:
        raise ValueError(f"Unknown NaN strategy: {strategy}")

    return data


def detect_outliers_iqr(data: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers using IQR method.

    Args:
        data: Data array [T, D]
        threshold: IQR multiplier (default: 1.5 for standard, 3.0 for lenient)

    Returns:
        Boolean mask [T, D] where True indicates outlier
    """
    q25 = np.nanpercentile(data, 25, axis=0)
    q75 = np.nanpercentile(data, 75, axis=0)
    iqr = q75 - q25

    lower_bound = q25 - threshold * iqr
    upper_bound = q75 + threshold * iqr

    outlier_mask = (data < lower_bound) | (data > upper_bound)
    return outlier_mask


def clip_outliers(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Clip outliers to [Q1 - k*IQR, Q3 + k*IQR].

    Args:
        data: Data array [T, D]
        threshold: IQR multiplier

    Returns:
        Clipped data
    """
    data = data.copy()

    q25 = np.nanpercentile(data, 25, axis=0)
    q75 = np.nanpercentile(data, 75, axis=0)
    iqr = q75 - q25

    lower_bound = q25 - threshold * iqr
    upper_bound = q75 + threshold * iqr

    data = np.clip(data, lower_bound, upper_bound)
    return data


def remove_zero_variance_features(
    data: pd.DataFrame,
    columns: List[str],
    threshold: float = 0.0
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Remove features with variance below threshold.

    Args:
        data: DataFrame
        columns: Column names to check
        threshold: Variance threshold

    Returns:
        filtered_data, kept_columns, removed_columns
    """
    variances = data[columns].var()
    keep_mask = variances > threshold

    kept_columns = variances[keep_mask].index.tolist()
    removed_columns = variances[~keep_mask].index.tolist()

    if removed_columns:
        print(f"  Removed {len(removed_columns)} zero/low-variance features: {removed_columns[:5]}...")

    return data[kept_columns], kept_columns, removed_columns


def remove_high_correlation_features(
    data: pd.DataFrame,
    columns: List[str],
    threshold: float = 0.95
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Remove one feature from each highly correlated pair.

    Args:
        data: DataFrame
        columns: Column names to check
        threshold: Correlation threshold (default: 0.95)

    Returns:
        filtered_data, kept_columns, removed_columns
    """
    corr_matrix = data[columns].corr().abs()

    # Get upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation > threshold
    to_drop = set()
    for column in upper_triangle.columns:
        correlated = upper_triangle[column][upper_triangle[column] > threshold].index.tolist()
        if correlated:
            # Drop the feature with lower variance (keep more informative one)
            variances = data[columns].var()
            for corr_col in correlated:
                if column not in to_drop and corr_col not in to_drop:
                    # Drop the one with lower variance
                    if variances[column] < variances[corr_col]:
                        to_drop.add(column)
                    else:
                        to_drop.add(corr_col)

    kept_columns = [col for col in columns if col not in to_drop]
    removed_columns = list(to_drop)

    if removed_columns:
        print(f"  Removed {len(removed_columns)} highly correlated features (|r|>{threshold})")

    return data[kept_columns], kept_columns, removed_columns


def remove_high_missing_features(
    data: pd.DataFrame,
    columns: List[str],
    threshold_pct: float = 50.0
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Remove features with too many missing values.

    Args:
        data: DataFrame
        columns: Column names to check
        threshold_pct: Maximum missing percentage (default: 50%)

    Returns:
        filtered_data, kept_columns, removed_columns
    """
    missing_pct = data[columns].isna().sum() / len(data) * 100

    kept_columns = missing_pct[missing_pct <= threshold_pct].index.tolist()
    removed_columns = missing_pct[missing_pct > threshold_pct].index.tolist()

    if removed_columns:
        print(f"  Removed {len(removed_columns)} high-missing features (>{threshold_pct}%)")

    return data[kept_columns], kept_columns, removed_columns


def apply_log_transform_skewed(
    data: np.ndarray,
    column_names: List[str],
    threshold: float = 3.0
) -> Tuple[np.ndarray, List[str]]:
    """
    Apply log(1+x) transform to highly skewed features.

    Args:
        data: Data array [T, D]
        column_names: Names of columns
        threshold: Skewness threshold (default: 3.0)

    Returns:
        transformed_data, list of transformed column names
    """
    data = data.copy()
    transformed_cols = []

    for col_idx in range(data.shape[1]):
        col_data = data[:, col_idx]
        col_data_clean = col_data[~np.isnan(col_data)]

        if len(col_data_clean) > 0:
            skewness = stats.skew(col_data_clean)

            if abs(skewness) > threshold:
                # Apply log(1+x) transform (handles negative values)
                # Shift data to be positive first if needed
                min_val = np.nanmin(col_data)
                if min_val < 0:
                    col_data = col_data - min_val + 1

                data[:, col_idx] = np.log1p(col_data)
                transformed_cols.append(column_names[col_idx])

    if transformed_cols:
        print(f"  Applied log transform to {len(transformed_cols)} skewed features (|skew|>{threshold})")

    return data, transformed_cols


def add_velocity_features(data: np.ndarray) -> np.ndarray:
    """
    Add time derivatives (velocities) of sensor readings.

    Args:
        data: Data array [T, D]

    Returns:
        Augmented data [T, 2*D] with original and velocity features
    """
    # Compute differences (velocities)
    velocities = np.diff(data, axis=0, prepend=data[0:1])

    # Concatenate original and velocities
    augmented = np.concatenate([data, velocities], axis=1)

    return augmented


def add_rolling_statistics(
    data: np.ndarray,
    window: int = 5
) -> np.ndarray:
    """
    Add rolling mean and std over small windows.

    Args:
        data: Data array [T, D]
        window: Rolling window size

    Returns:
        Augmented data [T, 3*D] with original, rolling mean, and rolling std
    """
    df = pd.DataFrame(data)

    # Compute rolling statistics
    rolling_mean = df.rolling(window=window, center=True, min_periods=1).mean().values
    rolling_std = df.rolling(window=window, center=True, min_periods=1).std().fillna(0).values

    # Concatenate
    augmented = np.concatenate([data, rolling_mean, rolling_std], axis=1)

    return augmented


def get_column_statistics(data: np.ndarray) -> dict:
    """
    Compute statistics for each column (for consistent NaN handling across splits).

    Args:
        data: Data array [T, D]

    Returns:
        Dictionary with 'median', 'mean', 'std' per column
    """
    return {
        'median': np.nanmedian(data, axis=0),
        'mean': np.nanmean(data, axis=0),
        'std': np.nanstd(data, axis=0),
    }
