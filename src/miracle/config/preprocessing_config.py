"""
Preprocessing configuration for G-code sensor data.

This module provides flexible preprocessing options based on data characteristics.
Defaults are set based on statistical analysis of the raw data.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pathlib import Path
import json


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing pipeline."""

    # Windowing parameters
    window_size: int = 64
    stride: int = 16

    # Train/val/test split
    train_frac: float = 0.7
    val_frac: float = 0.15
    # test_frac automatically = 1 - train_frac - val_frac

    # Normalization strategy
    scaler_type: Literal['standard', 'robust', 'quantile', 'minmax', 'none'] = 'robust'
    """
    Scaler type for continuous features:
    - 'robust': RobustScaler - uses median and IQR, good for outliers and non-normal data (RECOMMENDED)
    - 'quantile': QuantileTransformer - transforms to uniform/normal distribution
    - 'standard': StandardScaler - assumes normal distribution (NOT recommended for this dataset)
    - 'minmax': MinMaxScaler - scales to [0, 1] range
    - 'none': No scaling

    **Default:** 'robust' (based on data analysis showing 231/232 features are non-normal)
    """

    quantile_output_distribution: Literal['uniform', 'normal'] = 'uniform'
    """Output distribution for QuantileTransformer"""

    # Missing value handling
    nan_strategy: Literal['zero', 'forward_fill', 'interpolate', 'median', 'mean', 'drop'] = 'forward_fill'
    """
    Strategy for handling NaN values:
    - 'zero': Replace with 0 (simple but may introduce bias)
    - 'forward_fill': Propagate last valid observation forward (good for time series)
    - 'interpolate': Linear interpolation between valid values
    - 'median': Replace with median of sensor
    - 'mean': Replace with mean of sensor
    - 'drop': Drop samples with NaN (may lose significant data)

    **Default:** 'forward_fill' (appropriate for time series sensor data)
    """

    max_missing_pct: float = 50.0
    """Drop sensors with >this% missing values (default: 50%)"""

    # Outlier handling
    outlier_method: Literal['clip', 'remove', 'none'] = 'clip'
    """
    Outlier handling method:
    - 'clip': Clip to [Q1 - k*IQR, Q3 + k*IQR]
    - 'remove': Remove samples with outliers
    - 'none': No outlier handling

    **Default:** 'clip' (67 sensors have >5% outliers in this dataset)
    """

    outlier_threshold: float = 3.0
    """IQR multiplier for outlier detection (default: 3.0)"""

    # Feature selection
    remove_zero_variance: bool = True
    """Remove features with zero variance (default: True)"""

    correlation_threshold: float = 0.95
    """Remove one feature from pairs with |correlation| > threshold (default: 0.95)"""

    variance_threshold: float = 0.0
    """Remove features with variance < threshold (0.0 = only zero variance)"""

    # Feature transformation
    log_transform_skewed: bool = True
    """Apply log transform to highly skewed features (default: True)"""

    skewness_threshold: float = 3.0
    """Apply log transform to features with abs(skewness) > threshold"""

    # Feature engineering
    add_derivatives: bool = False
    """Add time derivatives (velocity) of sensor readings"""

    add_rolling_stats: bool = False
    """Add rolling mean/std over small windows"""

    rolling_window: int = 5
    """Window size for rolling statistics"""

    # Categorical feature handling
    categorical_features: List[str] = field(default_factory=lambda: ['stat', 'unit', 'dist', 'coor'])
    """List of categorical feature names"""

    # Features to exclude (data leakage prevention)
    exclude_features: List[str] = field(default_factory=lambda: [
        'time', 'gcode_line_num', 'gcode_text', 'gcode_tokens',
        't_console', 'gcode_line', 'gcode_string', 'raw_json',
        'vel', 'plane',  # NaN columns
        'line', 'posx', 'posy', 'posz', 'feed', 'momo'  # Data leakage
    ])
    """Features to exclude from processing"""

    # G-code column
    gcode_column: str = 'gcode_text'
    """Name of column containing G-code text"""

    # Validation
    validate_shapes: bool = True
    """Validate that all files have consistent shapes after processing"""

    random_seed: int = 42
    """Random seed for reproducibility"""

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        config_dict = self.__dict__.copy()
        # Convert Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'PreprocessingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    @classmethod
    def from_data_analysis(cls,
                          high_missing_sensors: int = 85,
                          high_outlier_sensors: int = 67,
                          high_correlation_pairs: int = 123,
                          non_normal_features: int = 231) -> 'PreprocessingConfig':
        """
        Create config based on data analysis findings.

        This uses the actual statistics from your raw data analysis:
        - 85 sensors with >10% missing values
        - 67 sensors with >5% outliers
        - 123 highly correlated pairs
        - 231/232 features non-normally distributed
        """
        return cls(
            scaler_type='robust',  # Non-normal data
            nan_strategy='forward_fill',  # Time series
            max_missing_pct=50.0,  # Drop sensors with >50% missing
            outlier_method='clip',  # Handle 67 sensors with outliers
            outlier_threshold=3.0,
            remove_zero_variance=True,
            correlation_threshold=0.95,  # Remove 123 redundant pairs
            log_transform_skewed=True,  # Handle extreme skewness
            skewness_threshold=3.0,
        )

    def get_preprocessing_summary(self) -> str:
        """Get human-readable summary of preprocessing configuration."""
        summary = [
            "Preprocessing Configuration:",
            f"  Window: {self.window_size} (stride={self.stride})",
            f"  Scaler: {self.scaler_type.upper()}",
            f"  NaN handling: {self.nan_strategy}",
            f"  Outlier handling: {self.outlier_method}",
            f"  Feature selection:",
            f"    - Remove zero variance: {self.remove_zero_variance}",
            f"    - Correlation threshold: {self.correlation_threshold}",
            f"    - Max missing %: {self.max_missing_pct}%",
            f"  Transformations:",
            f"    - Log transform skewed: {self.log_transform_skewed} (>={self.skewness_threshold})",
            f"    - Add derivatives: {self.add_derivatives}",
            f"    - Add rolling stats: {self.add_rolling_stats}",
        ]
        return "\n".join(summary)


# Predefined configurations

def get_default_config() -> PreprocessingConfig:
    """Get default preprocessing configuration (based on data analysis)."""
    return PreprocessingConfig.from_data_analysis()


def get_fast_config() -> PreprocessingConfig:
    """Get configuration optimized for fast preprocessing (minimal transforms)."""
    return PreprocessingConfig(
        scaler_type='robust',
        nan_strategy='zero',  # Fastest
        outlier_method='none',  # Skip
        remove_zero_variance=True,
        correlation_threshold=0.99,  # Less aggressive
        log_transform_skewed=False,  # Skip
        add_derivatives=False,
        add_rolling_stats=False,
    )


def get_thorough_config() -> PreprocessingConfig:
    """Get configuration for thorough preprocessing (all features)."""
    return PreprocessingConfig(
        scaler_type='quantile',  # Most aggressive normalization
        quantile_output_distribution='normal',
        nan_strategy='interpolate',
        max_missing_pct=30.0,  # More aggressive dropping
        outlier_method='clip',
        outlier_threshold=2.5,  # More aggressive clipping
        remove_zero_variance=True,
        correlation_threshold=0.90,  # More aggressive correlation removal
        log_transform_skewed=True,
        skewness_threshold=2.0,  # Lower threshold
        add_derivatives=True,  # Add velocity features
        add_rolling_stats=True,  # Add windowed stats
    )


def get_minimal_config() -> PreprocessingConfig:
    """Get minimal preprocessing (for debugging)."""
    return PreprocessingConfig(
        scaler_type='standard',  # Simple standard scaling
        nan_strategy='zero',
        outlier_method='none',
        remove_zero_variance=False,
        correlation_threshold=1.0,  # No correlation removal
        log_transform_skewed=False,
        add_derivatives=False,
        add_rolling_stats=False,
    )
