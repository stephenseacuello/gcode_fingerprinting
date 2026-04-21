"""Phase 2: Apply 4 isolation methods, aggregate to run-level vectors."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolpath_signature import config as cfg
from toolpath_signature import utils


def isolate_bins(active_bins: np.ndarray, air_mean: np.ndarray, air_std: np.ndarray,
                 method: str) -> np.ndarray:
    """Apply an isolation method to binned data.

    active_bins: (n_bins, n_features)
    air_mean:    (n_bins, n_features)
    air_std:     (n_bins, n_features)
    Returns:     (n_bins, n_features)
    """
    if method == "raw":
        return active_bins.copy()
    elif method == "subtracted":
        return active_bins - air_mean
    elif method == "ratio":
        return utils.safe_divide(active_bins, air_mean, eps=1e-10, clip=100.0)
    elif method == "zscore":
        return utils.safe_zscore(active_bins, air_mean, air_std, eps=1e-6)
    else:
        raise ValueError(f"Unknown isolation method: {method}")


def isolate_wholerun(active_wr: np.ndarray, air_mean: np.ndarray, air_std: np.ndarray,
                     method: str) -> np.ndarray:
    """Apply isolation to whole-run vectors (1-D)."""
    if method == "raw":
        return active_wr.copy()
    elif method == "subtracted":
        return active_wr - air_mean
    elif method == "ratio":
        return utils.safe_divide(active_wr, air_mean, eps=1e-10, clip=100.0)
    elif method == "zscore":
        return utils.safe_zscore(active_wr, air_mean, air_std, eps=1e-6)
    else:
        raise ValueError(f"Unknown method: {method}")


def run_phase2():
    print("=" * 70)
    print("PHASE 2: Feature Isolation")
    print("=" * 70)

    # Load air references
    air_refs = {}
    for toolpath in cfg.TOOLPATH_TYPES:
        air_refs[toolpath] = {
            "bin_mean": utils.load_parquet(cfg.AIR_REF_DIR / f"{toolpath}_bin_mean.parquet").values,
            "bin_std": utils.load_parquet(cfg.AIR_REF_DIR / f"{toolpath}_bin_std.parquet").values,
            "wr_mean": utils.load_parquet(cfg.AIR_REF_DIR / f"{toolpath}_wholerun_mean.parquet").values[0],
            "wr_std": utils.load_parquet(cfg.AIR_REF_DIR / f"{toolpath}_wholerun_std.parquet").values[0],
        }

    # Get column names from a reference file
    sample_prefix = cfg.ACTIVE_PREFIX["adaptive"]
    sample_files = utils.discover_runs(sample_prefix)
    sample_run_id = utils.extract_run_id(sample_files[0])
    bin_col_names = utils.load_parquet(
        cfg.BIN_STATS_DIR / f"{sample_prefix}_{sample_run_id}_bins.parquet"
    ).columns.tolist()
    wr_col_names = utils.load_parquet(
        cfg.BIN_STATS_DIR / f"{sample_prefix}_{sample_run_id}_wholerun.parquet"
    ).columns.tolist()

    # Aggregated column names
    agg_summary_cols = utils.make_agg_column_names(bin_col_names, method="summary")
    meta_cols = ["toolpath", "run_id", "prefix"]

    for iso_method in cfg.ISOLATION_METHODS:
        print(f"\n── Isolation: {iso_method} ──")

        # Collect run vectors
        temporal_rows = []
        wholerun_rows = []
        meta_rows = []

        for toolpath in cfg.TOOLPATH_TYPES:
            prefix = cfg.ACTIVE_PREFIX[toolpath]
            files = utils.discover_runs(prefix)
            ref = air_refs[toolpath]

            for fpath in files:
                run_id = utils.extract_run_id(fpath)

                # Load bin stats
                bin_path = cfg.BIN_STATS_DIR / f"{prefix}_{run_id}_bins.parquet"
                active_bins = utils.load_parquet(bin_path).values  # (20, 770)

                # Apply isolation to bins
                isolated_bins = isolate_bins(
                    active_bins, ref["bin_mean"], ref["bin_std"], iso_method
                )

                # Save isolated bins (for temporal fingerprint visualization)
                iso_bin_df = pd.DataFrame(isolated_bins, columns=bin_col_names)
                iso_path = cfg.ISOLATED_DIR / iso_method / f"{prefix}_{run_id}_bins.parquet"
                utils.save_parquet(iso_bin_df, iso_path)

                # Aggregate to run-level: summary (mean+std across bins)
                run_vec = utils.aggregate_bins_to_run(
                    pd.DataFrame(isolated_bins, columns=bin_col_names), method="summary"
                )
                temporal_rows.append(run_vec)

                # Whole-run isolation
                wr_path = cfg.BIN_STATS_DIR / f"{prefix}_{run_id}_wholerun.parquet"
                active_wr = utils.load_parquet(wr_path).values[0]
                isolated_wr = isolate_wholerun(
                    active_wr, ref["wr_mean"], ref["wr_std"], iso_method
                )
                wholerun_rows.append(isolated_wr)

                meta_rows.append({"toolpath": toolpath, "run_id": run_id, "prefix": prefix})

        # Build DataFrames
        meta_df = pd.DataFrame(meta_rows)

        # Temporal summary vectors
        temporal_arr = np.stack(temporal_rows)
        temporal_df = pd.concat([
            meta_df.reset_index(drop=True),
            pd.DataFrame(temporal_arr, columns=agg_summary_cols),
        ], axis=1)
        out_path = cfg.RUN_VECTORS_DIR / f"{iso_method}_temporal_summary.parquet"
        utils.save_parquet(temporal_df, out_path)

        # Whole-run vectors
        wholerun_arr = np.stack(wholerun_rows)
        wholerun_df = pd.concat([
            meta_df.reset_index(drop=True),
            pd.DataFrame(wholerun_arr, columns=wr_col_names),
        ], axis=1)
        wr_out_path = cfg.RUN_VECTORS_DIR / f"{iso_method}_wholerun.parquet"
        utils.save_parquet(wholerun_df, wr_out_path)

        # Validation
        assert not np.any(np.isnan(temporal_arr)), f"NaN in {iso_method} temporal"
        assert not np.any(np.isinf(temporal_arr)), f"Inf in {iso_method} temporal"
        assert not np.any(np.isnan(wholerun_arr)), f"NaN in {iso_method} wholerun"
        assert not np.any(np.isinf(wholerun_arr)), f"Inf in {iso_method} wholerun"

        print(f"  Temporal summary: {temporal_df.shape} (51 runs × {temporal_arr.shape[1]} features)")
        print(f"  Whole-run:        {wholerun_df.shape} (51 runs × {wholerun_arr.shape[1]} features)")

        if iso_method == "subtracted":
            print(f"  Subtracted mean ≈ {np.mean(temporal_arr):.4f} (should be near 0)")
        elif iso_method == "ratio":
            print(f"  Ratio mean ≈ {np.mean(temporal_arr):.4f} (should be near 1)")

    print("\nPhase 2 complete.")


if __name__ == "__main__":
    run_phase2()
