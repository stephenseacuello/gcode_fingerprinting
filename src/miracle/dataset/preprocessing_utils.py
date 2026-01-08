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


# ============================================================================
# G-code Value Clamping and Denoising
# ============================================================================

class GCodeValueDenoiser:
    """
    Clamp and denoise raw G-code parameter values before bucketing.

    This helps prevent:
    1. Out-of-vocabulary buckets from extreme values
    2. Noise-induced bucket boundary crossings
    3. Inconsistent bucket assignments for near-boundary values

    Usage:
        denoiser = GCodeValueDenoiser()
        clean_values = denoiser.denoise(raw_values, param_type='X')
    """

    # Default bounds per parameter (based on typical CNC machine limits)
    DEFAULT_BOUNDS = {
        'X': (-100.0, 100.0),
        'Y': (-100.0, 100.0),
        'Z': (-50.0, 50.0),
        'R': (-50.0, 50.0),
        'I': (-100.0, 100.0),
        'J': (-100.0, 100.0),
        'K': (-50.0, 50.0),
        'F': (0.0, 10000.0),
        'S': (0.0, 30000.0),
        'A': (-360.0, 360.0),
        'B': (-360.0, 360.0),
        'C': (-360.0, 360.0),
    }

    # Default precisions per parameter (for rounding/quantization)
    DEFAULT_PRECISION = {
        'X': 0.001,
        'Y': 0.001,
        'Z': 0.001,
        'R': 0.0001,
        'I': 0.001,
        'J': 0.001,
        'K': 0.001,
        'F': 1.0,
        'S': 10.0,
        'A': 0.001,
        'B': 0.001,
        'C': 0.001,
    }

    def __init__(
        self,
        bounds: dict = None,
        precision: dict = None,
        denoise_sigma: float = 0.0,
        snap_threshold: float = 0.0,
        use_median_filter: bool = False,
        median_window: int = 3,
    ):
        """
        Initialize the denoiser.

        Args:
            bounds: Dict mapping param letter to (min, max) bounds.
                    Uses DEFAULT_BOUNDS if None.
            precision: Dict mapping param letter to quantization precision.
                       Uses DEFAULT_PRECISION if None.
            denoise_sigma: Gaussian smoothing sigma. 0 = no smoothing.
            snap_threshold: Snap values within this threshold of bucket boundaries
                           to the boundary. Helps reduce bucket ambiguity.
            use_median_filter: Apply median filter to remove impulse noise.
            median_window: Window size for median filter.
        """
        self.bounds = {**self.DEFAULT_BOUNDS, **(bounds or {})}
        self.precision = {**self.DEFAULT_PRECISION, **(precision or {})}
        self.denoise_sigma = denoise_sigma
        self.snap_threshold = snap_threshold
        self.use_median_filter = use_median_filter
        self.median_window = median_window

    def clamp(self, value: float, param_type: str) -> float:
        """
        Clamp value to valid range for parameter type.

        Args:
            value: Raw parameter value
            param_type: Parameter letter (e.g., 'X', 'Y', 'Z')

        Returns:
            Clamped value
        """
        if param_type not in self.bounds:
            return value  # Unknown param, pass through

        min_val, max_val = self.bounds[param_type]
        return max(min_val, min(max_val, value))

    def quantize(self, value: float, param_type: str) -> float:
        """
        Quantize value to precision for parameter type.

        Args:
            value: Raw parameter value
            param_type: Parameter letter

        Returns:
            Quantized value
        """
        if param_type not in self.precision:
            return value

        prec = self.precision[param_type]
        if prec > 0:
            return round(value / prec) * prec
        return value

    def snap_to_bucket_boundary(
        self,
        value: float,
        param_type: str,
        bucket_digits: int = 4,
    ) -> float:
        """
        Snap value to nearest bucket boundary if within threshold.

        This reduces bucket ambiguity for values near boundaries.

        Args:
            value: Raw parameter value
            param_type: Parameter letter
            bucket_digits: Number of digits in bucket (for computing step size)

        Returns:
            Snapped value (or original if not near boundary)
        """
        if self.snap_threshold <= 0:
            return value

        # Compute bucket step size based on precision and digits
        prec = self.precision.get(param_type, 0.001)
        bucket_step = prec * (10 ** (4 - bucket_digits))

        if bucket_step <= 0:
            return value

        # Find nearest bucket boundary
        bucket_idx = round(value / bucket_step)
        bucket_boundary = bucket_idx * bucket_step

        # Snap if within threshold
        if abs(value - bucket_boundary) < self.snap_threshold * bucket_step:
            return bucket_boundary

        return value

    def denoise_sequence(
        self,
        values: np.ndarray,
        param_type: str,
    ) -> np.ndarray:
        """
        Denoise a sequence of values for a single parameter.

        Args:
            values: Array of values [N]
            param_type: Parameter letter

        Returns:
            Denoised values [N]
        """
        if len(values) == 0:
            return values

        values = values.copy().astype(np.float64)

        # 1. Median filter for impulse noise
        if self.use_median_filter and len(values) >= self.median_window:
            from scipy.ndimage import median_filter
            values = median_filter(values, size=self.median_window, mode='nearest')

        # 2. Gaussian smoothing
        if self.denoise_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            values = gaussian_filter1d(values, sigma=self.denoise_sigma, mode='nearest')

        # 3. Clamp to bounds
        min_val, max_val = self.bounds.get(param_type, (-np.inf, np.inf))
        values = np.clip(values, min_val, max_val)

        # 4. Quantize
        prec = self.precision.get(param_type, 0.001)
        if prec > 0:
            values = np.round(values / prec) * prec

        return values

    def denoise(
        self,
        value: float,
        param_type: str,
        bucket_digits: int = 4,
    ) -> float:
        """
        Denoise a single raw G-code value.

        Args:
            value: Raw parameter value
            param_type: Parameter letter (e.g., 'X', 'Y', 'Z')
            bucket_digits: Number of digits for bucketing

        Returns:
            Cleaned value
        """
        # 1. Clamp to valid range
        value = self.clamp(value, param_type)

        # 2. Quantize to precision
        value = self.quantize(value, param_type)

        # 3. Snap to bucket boundary if near
        value = self.snap_to_bucket_boundary(value, param_type, bucket_digits)

        return value

    def denoise_batch(
        self,
        values: np.ndarray,
        param_types: list,
        bucket_digits: int = 4,
    ) -> np.ndarray:
        """
        Denoise a batch of G-code values.

        Args:
            values: Array of values [N]
            param_types: List of parameter letters [N]
            bucket_digits: Number of digits for bucketing

        Returns:
            Cleaned values [N]
        """
        cleaned = np.zeros_like(values, dtype=np.float64)

        for i, (val, ptype) in enumerate(zip(values, param_types)):
            cleaned[i] = self.denoise(val, ptype, bucket_digits)

        return cleaned


def clamp_gcode_values(
    values: np.ndarray,
    param_types: list,
    bounds: dict = None,
) -> np.ndarray:
    """
    Simple function to clamp G-code values to valid ranges.

    Args:
        values: Array of values [N]
        param_types: List of parameter letters [N]
        bounds: Optional custom bounds dict

    Returns:
        Clamped values [N]
    """
    denoiser = GCodeValueDenoiser(bounds=bounds)
    return np.array([
        denoiser.clamp(v, pt) for v, pt in zip(values, param_types)
    ])


def denoise_gcode_values(
    values: np.ndarray,
    param_types: list,
    denoise_sigma: float = 0.5,
    snap_threshold: float = 0.1,
    bucket_digits: int = 4,
) -> np.ndarray:
    """
    Denoise G-code values with Gaussian smoothing and boundary snapping.

    Args:
        values: Array of values [N]
        param_types: List of parameter letters [N]
        denoise_sigma: Gaussian smoothing sigma
        snap_threshold: Bucket boundary snap threshold
        bucket_digits: Number of digits for bucketing

    Returns:
        Denoised values [N]
    """
    denoiser = GCodeValueDenoiser(
        denoise_sigma=denoise_sigma,
        snap_threshold=snap_threshold,
    )
    return denoiser.denoise_batch(values, param_types, bucket_digits)


def adaptive_denoise_by_operation(
    values: np.ndarray,
    param_types: list,
    operation_type: str,
    bucket_digits: int = 4,
) -> np.ndarray:
    """
    Apply operation-specific denoising settings.

    Different machining operations have different noise characteristics:
    - Adaptive clearing: higher vibration, needs more smoothing
    - Face milling: lower vibration, less smoothing needed
    - Pocket: moderate smoothing

    Args:
        values: Array of values [N]
        param_types: List of parameter letters [N]
        operation_type: One of 'adaptive', 'face', 'pocket', etc.
        bucket_digits: Number of digits for bucketing

    Returns:
        Denoised values [N]
    """
    # Operation-specific settings
    op_settings = {
        'adaptive': {'denoise_sigma': 0.8, 'snap_threshold': 0.15},
        'adaptive150025': {'denoise_sigma': 0.8, 'snap_threshold': 0.15},
        'face': {'denoise_sigma': 0.3, 'snap_threshold': 0.05},
        'face150025': {'denoise_sigma': 0.3, 'snap_threshold': 0.05},
        'pocket': {'denoise_sigma': 0.5, 'snap_threshold': 0.1},
        'pocket150025': {'denoise_sigma': 0.5, 'snap_threshold': 0.1},
        'damageadaptive': {'denoise_sigma': 1.0, 'snap_threshold': 0.2},
        'damageface': {'denoise_sigma': 0.6, 'snap_threshold': 0.1},
        'damagepocket': {'denoise_sigma': 0.7, 'snap_threshold': 0.12},
    }

    settings = op_settings.get(operation_type, {'denoise_sigma': 0.5, 'snap_threshold': 0.1})

    return denoise_gcode_values(
        values, param_types,
        denoise_sigma=settings['denoise_sigma'],
        snap_threshold=settings['snap_threshold'],
        bucket_digits=bucket_digits,
    )
