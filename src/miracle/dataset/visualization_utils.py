"""
Reusable visualization utilities for G-code sensor data analysis.

This module consolidates common plotting functions used across analysis scripts:
- Distribution plots (histograms, box plots, violin plots)
- Correlation heatmaps
- Time series plots
- Dimensionality reduction visualizations (PCA, t-SNE)
- Publication-quality figure formatting
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Try to import optional dependencies
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

__all__ = [
    "setup_publication_style",
    "plot_sensor_distributions",
    "plot_correlation_heatmap",
    "plot_time_series",
    "plot_pca_2d",
    "plot_tsne_2d",
    "plot_outliers",
    "plot_token_frequencies",
    "plot_cooccurrence_matrix",
    "plot_missing_values",
    "plot_autocorrelation",
    "save_figure",
    "get_operation_color_map",
    "detect_operation_type",
]


# Color scheme for operation types
OPERATION_COLORS = {
    'pocket': '#1f77b4',    # Blue
    'face': '#2ca02c',       # Green
    'adaptive': '#ff7f0e',   # Orange
    'contour': '#d62728',    # Red
    'drill': '#9467bd',      # Purple
    'slot': '#8c564b',       # Brown
    'engrave': '#e377c2',    # Pink
    'unknown': '#7f7f7f',    # Gray
}


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use('seaborn-v0_8-darkgrid')

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
    })


def detect_operation_type(filename: str) -> str:
    """
    Detect operation type from filename.

    Args:
        filename: Name of the file (e.g., 'pocket_001_aligned.csv')

    Returns:
        Operation type: 'pocket', 'face', 'adaptive', 'contour', 'drill', 'slot', 'engrave', or 'unknown'
    """
    filename_lower = filename.lower()

    # Check for operation types in filename
    if 'pocket' in filename_lower:
        return 'pocket'
    elif 'face' in filename_lower:
        return 'face'
    elif 'adaptive' in filename_lower:
        return 'adaptive'
    elif 'contour' in filename_lower:
        return 'contour'
    elif 'drill' in filename_lower:
        return 'drill'
    elif 'slot' in filename_lower:
        return 'slot'
    elif 'engrave' in filename_lower or 'engraving' in filename_lower:
        return 'engrave'
    else:
        return 'unknown'


def get_operation_color_map(operation_types: List[str]) -> Dict[str, str]:
    """
    Get color mapping for operation types.

    Args:
        operation_types: List of operation type strings

    Returns:
        Dictionary mapping operation type to hex color
    """
    unique_types = set(operation_types)
    color_map = {}

    for op_type in unique_types:
        color_map[op_type] = OPERATION_COLORS.get(op_type, OPERATION_COLORS['unknown'])

    return color_map


def plot_sensor_distributions(
    data: pd.DataFrame,
    sensors: List[str],
    figsize: Tuple[int, int] = (15, 10),
    plot_type: str = 'hist'
) -> Figure:
    """
    Plot distributions for multiple sensors.

    Args:
        data: DataFrame with sensor data
        sensors: List of sensor column names
        figsize: Figure size
        plot_type: 'hist', 'box', or 'violin'

    Returns:
        matplotlib Figure
    """
    n_sensors = len(sensors)
    n_cols = min(4, n_sensors)
    n_rows = (n_sensors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for idx, sensor in enumerate(sensors):
        ax = axes[idx]
        sensor_data = data[sensor].dropna()

        if plot_type == 'hist':
            ax.hist(sensor_data, bins=50, edgecolor='black', alpha=0.7)
            ax.set_ylabel('Frequency')
        elif plot_type == 'box':
            ax.boxplot(sensor_data, vert=True)
            ax.set_ylabel('Value')
        elif plot_type == 'violin':
            parts = ax.violinplot([sensor_data], vert=True, showmeans=True, showmedians=True)
            ax.set_ylabel('Value')

        ax.set_title(f'{sensor}\n(n={len(sensor_data):,})', fontsize=10)
        ax.set_xlabel('Value' if plot_type == 'hist' else '')
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for idx in range(n_sensors, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Sensor Distributions ({plot_type.title()} Plot)', fontsize=14, y=1.00)
    plt.tight_layout()

    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'RdBu_r',
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot: bool = False
) -> Figure:
    """
    Plot correlation heatmap.

    Args:
        corr_matrix: Correlation matrix DataFrame
        figsize: Figure size
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        annot: Whether to annotate cells

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        annot=annot,
        fmt='.2f' if annot else '',
        cbar_kws={'label': 'Correlation Coefficient'},
        ax=ax
    )

    ax.set_title('Feature Correlation Matrix', fontsize=14, pad=15)
    plt.tight_layout()

    return fig


def plot_time_series(
    data: pd.DataFrame,
    sensors: List[str],
    time_col: Optional[str] = None,
    max_points: int = 10000,
    figsize: Tuple[int, int] = (15, 8)
) -> Figure:
    """
    Plot time series for multiple sensors.

    Args:
        data: DataFrame with sensor data
        sensors: List of sensor column names
        time_col: Optional time column name (uses index if None)
        max_points: Maximum points to plot (downsamples if exceeded)
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Downsample if too many points
    if len(data) > max_points:
        step = len(data) // max_points
        plot_data = data.iloc[::step].copy()
    else:
        plot_data = data.copy()

    # Get time axis
    if time_col and time_col in plot_data.columns:
        time_axis = plot_data[time_col].values
        xlabel = time_col
    else:
        time_axis = np.arange(len(plot_data))
        xlabel = 'Sample Index'

    # Create subplots
    n_sensors = len(sensors)
    fig, axes = plt.subplots(n_sensors, 1, figsize=figsize, sharex=True)

    if n_sensors == 1:
        axes = [axes]

    for idx, sensor in enumerate(sensors):
        ax = axes[idx]
        sensor_data = plot_data[sensor].values

        ax.plot(time_axis, sensor_data, linewidth=0.8, alpha=0.8)
        ax.set_ylabel(sensor, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add statistics annotation
        mean_val = np.nanmean(sensor_data)
        std_val = np.nanstd(sensor_data)
        ax.text(0.02, 0.95, f'μ={mean_val:.3f}, σ={std_val:.3f}',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[-1].set_xlabel(xlabel, fontsize=11)
    fig.suptitle('Sensor Time Series', fontsize=14, y=0.995)
    plt.tight_layout()

    return fig


def plot_pca_2d(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[Dict[int, str]] = None,
    color_map: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'PCA 2D Projection'
) -> Tuple[Figure, np.ndarray]:
    """
    Plot 2D PCA projection of features.

    Args:
        features: Feature matrix [N, D]
        labels: Optional labels for coloring [N]
        label_names: Optional dict mapping label indices to names
        color_map: Optional dict mapping label names to colors (hex codes)
        figsize: Figure size
        title: Plot title

    Returns:
        Figure and transformed features
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn is required for PCA. Install with: pip install scikit-learn")

    # Perform PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = np.unique(labels)
        for label_idx in unique_labels:
            mask = labels == label_idx
            label_name = label_names[label_idx] if label_names else f'Class {label_idx}'

            # Get color for this label
            if color_map and label_name in color_map:
                color = color_map[label_name]
            else:
                color = None  # Use default matplotlib colors

            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                label=label_name.capitalize(),
                color=color,
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidths=0.5
            )
        ax.legend(loc='best', framealpha=0.9)
    else:
        ax.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=30)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)

    # Add total variance explained
    total_var = pca.explained_variance_ratio_[:2].sum()
    ax.text(0.02, 0.98, f'Total variance: {total_var:.1%}',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    return fig, features_2d


def plot_tsne_2d(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[Dict[int, str]] = None,
    color_map: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    perplexity: int = 30,
    title: str = 't-SNE 2D Projection'
) -> Tuple[Figure, np.ndarray]:
    """
    Plot 2D t-SNE projection of features.

    Args:
        features: Feature matrix [N, D]
        labels: Optional labels for coloring [N]
        label_names: Optional dict mapping label indices to names
        color_map: Optional dict mapping label names to colors (hex codes)
        figsize: Figure size
        perplexity: t-SNE perplexity parameter
        title: Plot title

    Returns:
        Figure and transformed features
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn is required for t-SNE. Install with: pip install scikit-learn")

    # Adjust perplexity if needed
    n_samples = features.shape[0]
    if perplexity >= n_samples:
        perplexity = max(5, n_samples // 3)
        print(f"  Adjusted perplexity to {perplexity} (n_samples={n_samples})")

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = np.unique(labels)
        for label_idx in unique_labels:
            mask = labels == label_idx
            label_name = label_names[label_idx] if label_names else f'Class {label_idx}'

            # Get color for this label
            if color_map and label_name in color_map:
                color = color_map[label_name]
            else:
                color = None  # Use default matplotlib colors

            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                label=label_name.capitalize(),
                color=color,
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidths=0.5
            )
        ax.legend(loc='best', framealpha=0.9)
    else:
        ax.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=30)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.set_title(f'{title} (perplexity={perplexity})', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, features_2d


def plot_outliers(
    data: pd.DataFrame,
    sensor: str,
    outlier_mask: np.ndarray,
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Visualize outliers in sensor data.

    Args:
        data: DataFrame with sensor data
        sensor: Sensor column name
        outlier_mask: Boolean mask indicating outliers
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    sensor_data = data[sensor].values
    normal_data = sensor_data[~outlier_mask]
    outlier_data = sensor_data[outlier_mask]

    # Time series plot with outliers highlighted
    ax1.plot(sensor_data, linewidth=0.5, alpha=0.5, label='Normal', color='blue')
    outlier_indices = np.where(outlier_mask)[0]
    ax1.scatter(outlier_indices, outlier_data, color='red', s=20, alpha=0.7, label='Outliers', zorder=5)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel(sensor)
    ax1.set_title(f'{sensor} Time Series with Outliers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot comparison
    ax2.boxplot([normal_data, outlier_data], labels=['Normal', 'Outliers'])
    ax2.set_ylabel(sensor)
    ax2.set_title(f'{sensor} Distribution')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics
    outlier_pct = 100 * outlier_mask.sum() / len(outlier_mask)
    fig.suptitle(f'Outlier Analysis: {sensor} ({outlier_pct:.2f}% outliers)', fontsize=13)

    plt.tight_layout()

    return fig


def plot_token_frequencies(
    token_counts: Dict[str, int],
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Plot token frequency distribution.

    Args:
        token_counts: Dictionary or Counter with token frequencies
        top_n: Number of top tokens to show
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Get top N tokens
    if hasattr(token_counts, 'most_common'):
        top_tokens = token_counts.most_common(top_n)
    else:
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        top_tokens = sorted_tokens[:top_n]

    tokens, counts = zip(*top_tokens)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(range(len(tokens)), counts, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=9)
    ax.set_xlabel('Frequency', fontsize=11)
    ax.set_title(f'Top {top_n} G-code Token Frequencies', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')

    # Add count labels
    for i, (token, count) in enumerate(top_tokens):
        ax.text(count, i, f' {count:,}', va='center', fontsize=8)

    ax.invert_yaxis()
    plt.tight_layout()

    return fig


def plot_cooccurrence_matrix(
    cooccur_matrix: np.ndarray,
    token_labels: List[str],
    figsize: Tuple[int, int] = (12, 10)
) -> Figure:
    """
    Plot token co-occurrence matrix as heatmap.

    Args:
        cooccur_matrix: Co-occurrence matrix [N, N]
        token_labels: Token names
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cooccur_matrix,
        xticklabels=token_labels,
        yticklabels=token_labels,
        cmap='YlOrRd',
        cbar_kws={'label': 'Co-occurrence Count'},
        ax=ax,
        square=True
    )

    ax.set_title('G-code Token Co-occurrence Matrix', fontsize=13, pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    return fig


def plot_missing_values(
    data: pd.DataFrame,
    columns: List[str],
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Visualize missing value patterns.

    Args:
        data: DataFrame
        columns: Columns to analyze
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Compute missing percentages
    missing_pct = data[columns].isna().sum() / len(data) * 100
    missing_pct = missing_pct.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(range(len(missing_pct)), missing_pct.values, color='coral', alpha=0.8)
    ax.set_yticks(range(len(missing_pct)))
    ax.set_yticklabels(missing_pct.index, fontsize=9)
    ax.set_xlabel('Missing Values (%)', fontsize=11)
    ax.set_title('Missing Value Analysis', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')

    # Add percentage labels
    for i, pct in enumerate(missing_pct.values):
        if pct > 0:
            ax.text(pct, i, f' {pct:.1f}%', va='center', fontsize=8)

    # Add warning line at 10%
    ax.axvline(x=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='10% threshold')
    ax.legend()

    ax.invert_yaxis()
    plt.tight_layout()

    return fig


def plot_autocorrelation(
    autocorr_results: Dict[str, np.ndarray],
    sensors: Optional[List[str]] = None,
    max_sensors: int = 12,
    figsize: Tuple[int, int] = (15, 10)
) -> Figure:
    """
    Plot autocorrelation functions for multiple sensors.

    Args:
        autocorr_results: Dictionary mapping sensor names to autocorr arrays
        sensors: Optional list of specific sensors to plot
        max_sensors: Maximum number of sensors to plot
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Select sensors
    if sensors is None:
        sensors = list(autocorr_results.keys())[:max_sensors]
    else:
        sensors = [s for s in sensors if s in autocorr_results]

    n_sensors = len(sensors)
    n_cols = min(4, n_sensors)
    n_rows = (n_sensors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for idx, sensor in enumerate(sensors):
        ax = axes[idx]
        autocorr = autocorr_results[sensor]
        lags = np.arange(len(autocorr))

        ax.plot(lags, autocorr, linewidth=1.5, color='steelblue')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Lag', fontsize=9)
        ax.set_ylabel('Autocorrelation', fontsize=9)
        ax.set_title(sensor, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1, 1)

    # Hide extra subplots
    for idx in range(n_sensors, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Autocorrelation Analysis', fontsize=14, y=1.00)
    plt.tight_layout()

    return fig


def save_figure(
    fig: Figure,
    output_path: Union[str, Path],
    dpi: int = 300,
    formats: List[str] = ['png']
) -> List[Path]:
    """
    Save figure in multiple formats.

    Args:
        fig: matplotlib Figure
        output_path: Output file path (without extension)
        dpi: Resolution for raster formats
        formats: List of formats ('png', 'pdf', 'svg', etc.)

    Returns:
        List of saved file paths
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for fmt in formats:
        save_path = output_path.parent / f"{output_path.stem}.{fmt}"
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        saved_paths.append(save_path)

    return saved_paths
