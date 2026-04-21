"""Shared utilities for the toolpath signature pipeline."""

import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from . import config as cfg


# ── File Discovery ─────────────────────────────────────────────────────────

def discover_runs(prefix: str, data_dir: Path = cfg.DATA_DIR) -> List[Path]:
    """Find all CSV files matching {prefix}_*_aligned.csv, sorted by run number."""
    pattern = f"{prefix}_*_aligned.csv"
    files = sorted(data_dir.glob(pattern))
    # Exclude damage files
    files = [f for f in files if not f.name.startswith("damage")]
    return files


def extract_run_id(path: Path) -> str:
    """Extract the run number from a filename like adaptive150025_003_aligned.csv -> '003'."""
    name = path.stem  # e.g. adaptive150025_003_aligned
    parts = name.replace("_aligned", "").split("_")
    return parts[-1]


# ── Data Loading ───────────────────────────────────────────────────────────

def load_run(path: Path, sensor_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Load a CSV, keep only sensor + metadata columns.

    Missing sensor columns are filled with 0.0 to ensure consistent dimensions.
    """
    if sensor_cols is None:
        sensor_cols = cfg.SENSOR_COLS

    df = pd.read_csv(path)

    keep = [c for c in sensor_cols if c in df.columns]
    keep += [c for c in cfg.METADATA_COLS if c in df.columns]
    df = df[keep].copy()

    # Ensure sensor columns are numeric
    for c in keep:
        if c not in cfg.METADATA_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill missing sensor columns with 0.0 for consistent dimensions
    for c in sensor_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Reorder so sensor columns are in canonical order
    meta = [c for c in cfg.METADATA_COLS if c in df.columns]
    df = df[sensor_cols + meta]

    return df


# ── Statistics ─────────────────────────────────────────────────────────────

def compute_rms(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Root mean square along an axis."""
    return np.sqrt(np.mean(arr ** 2, axis=axis))


def compute_block_stats(
    group: pd.DataFrame, sensor_cols: List[str]
) -> np.ndarray:
    """Compute 7 summary statistics for a group of rows.

    Returns a 1-D array of length len(sensor_cols) * 7.
    Order per sensor: mean, rms, std, skewness, kurtosis, min, max.
    """
    data = group[sensor_cols].values.astype(np.float64)
    n = data.shape[0]

    means = np.nanmean(data, axis=0)
    rms = compute_rms(data, axis=0)

    if n < 2:
        stds = np.zeros(data.shape[1])
        skews = np.zeros(data.shape[1])
        kurts = np.zeros(data.shape[1])
    else:
        stds = np.nanstd(data, axis=0, ddof=1)
        with np.errstate(invalid="ignore"):
            skews = sp_stats.skew(data, axis=0, nan_policy="omit", bias=False)
            kurts = sp_stats.kurtosis(data, axis=0, nan_policy="omit", bias=False)
        skews = np.nan_to_num(skews, nan=0.0)
        kurts = np.nan_to_num(kurts, nan=0.0)

    mins = np.nanmin(data, axis=0)
    maxs = np.nanmax(data, axis=0)

    # Stack: (7, n_sensors) then flatten in order
    return np.concatenate([means, rms, stds, skews, kurts, mins, maxs])


def make_stat_column_names(sensor_cols: List[str]) -> List[str]:
    """Generate column names for the 7-stat feature vector."""
    names = []
    for stat in cfg.BLOCK_STATS:
        for col in sensor_cols:
            names.append(f"{col}_{stat}")
    return names


# ── Temporal Binning ───────────────────────────────────────────────────────

def temporal_bin(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    n_bins: int = cfg.N_BINS,
) -> pd.DataFrame:
    """Compute per-temporal-bin statistics for a single run.

    Returns DataFrame of shape (n_bins, len(sensor_cols) * 7).
    """
    if sensor_cols is None:
        sensor_cols = cfg.SENSOR_COLS

    # Filter to available sensor columns
    available = [c for c in sensor_cols if c in df.columns]
    n_rows = len(df)

    # Assign bin ids
    t_norm = np.arange(n_rows) / max(n_rows - 1, 1)
    bin_ids = np.clip((t_norm * n_bins).astype(int), 0, n_bins - 1)

    col_names = make_stat_column_names(available)
    results = []

    for b in range(n_bins):
        mask = bin_ids == b
        group = df.iloc[mask]
        if len(group) == 0:
            results.append(np.zeros(len(available) * 7))
        else:
            results.append(compute_block_stats(group, available))

    return pd.DataFrame(results, columns=col_names, index=range(n_bins))


def compute_whole_run_stats(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute 7 summary statistics over the entire run. Returns 770-element vector."""
    if sensor_cols is None:
        sensor_cols = cfg.SENSOR_COLS
    available = [c for c in sensor_cols if c in df.columns]
    return compute_block_stats(df, available)


# ── Run-Level Aggregation ──────────────────────────────────────────────────

def aggregate_bins_to_run(bin_stats_df: pd.DataFrame, method: str = "summary") -> np.ndarray:
    """Aggregate bin-level stats to a single run vector.

    method="summary": mean + std across bins → 770*2 = 1540 features.
    method="concat": flatten all bins → 770*20 = 15400 features.
    """
    data = bin_stats_df.values  # (n_bins, 770)
    if method == "concat":
        return data.flatten()
    elif method == "summary":
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0, ddof=1) if data.shape[0] > 1 else np.zeros(data.shape[1])
        return np.concatenate([means, stds])
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def make_agg_column_names(base_cols: List[str], method: str = "summary", n_bins: int = cfg.N_BINS) -> List[str]:
    """Generate column names for the aggregated run-level vector."""
    if method == "concat":
        names = []
        for b in range(n_bins):
            for c in base_cols:
                names.append(f"bin{b:02d}_{c}")
        return names
    elif method == "summary":
        names = []
        for prefix in ["binmean", "binstd"]:
            for c in base_cols:
                names.append(f"{prefix}_{c}")
        return names
    else:
        raise ValueError(f"Unknown method: {method}")


# ── Safe Math ──────────────────────────────────────────────────────────────

def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-10, clip: float = 100.0) -> np.ndarray:
    """Element-wise division with epsilon floor and clipping."""
    b_safe = np.where(np.abs(b) < eps, eps * np.sign(b + eps), b)
    result = a / b_safe
    return np.clip(result, -clip, clip)


def safe_zscore(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score with small-sigma guard."""
    safe_sigma = np.where(sigma < eps, 1.0, sigma)  # avoid div by 0
    result = (x - mu) / safe_sigma
    result = np.where(sigma < eps, 0.0, result)  # zero out where sigma was too small
    return result


# ── I/O Helpers ────────────────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as parquet with auto-mkdir."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file."""
    return pd.read_parquet(path)


def save_numpy(arr: np.ndarray, path: Path, **kwargs) -> None:
    """Save a numpy array with auto-mkdir."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, data=arr, **kwargs)


def load_numpy(path: Path) -> dict:
    """Load a numpy npz file."""
    return dict(np.load(path, allow_pickle=True))
