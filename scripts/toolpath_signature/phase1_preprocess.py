"""Phase 1: Load CSVs, compute temporal bin statistics, build air references."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolpath_signature import config as cfg
from toolpath_signature import utils


def run_phase1():
    print("=" * 70)
    print("PHASE 1: Preprocessing & Temporal Binning")
    print("=" * 70)

    manifest = {}

    for toolpath in cfg.TOOLPATH_TYPES:
        for condition, prefix_map in [("air", cfg.AIR_PREFIX), ("active", cfg.ACTIVE_PREFIX)]:
            prefix = prefix_map[toolpath]
            files = utils.discover_runs(prefix)
            label = f"{toolpath}_{condition}"
            manifest[label] = {"files": [], "run_ids": [], "row_counts": [], "n_bins": cfg.N_BINS}

            print(f"\n── {label} ({len(files)} files) ──")

            for fpath in files:
                run_id = utils.extract_run_id(fpath)
                df = utils.load_run(fpath)
                n_rows = len(df)

                # Compute temporal bin stats
                bin_stats = utils.temporal_bin(df, n_bins=cfg.N_BINS)
                out_path = cfg.BIN_STATS_DIR / f"{prefix}_{run_id}_bins.parquet"
                utils.save_parquet(bin_stats, out_path)

                # Compute whole-run stats
                wr_stats = utils.compute_whole_run_stats(df)
                col_names = utils.make_stat_column_names(
                    [c for c in cfg.SENSOR_COLS if c in df.columns]
                )
                wr_df = pd.DataFrame([wr_stats], columns=col_names)
                wr_path = cfg.BIN_STATS_DIR / f"{prefix}_{run_id}_wholerun.parquet"
                utils.save_parquet(wr_df, wr_path)

                manifest[label]["files"].append(str(fpath.name))
                manifest[label]["run_ids"].append(run_id)
                manifest[label]["row_counts"].append(n_rows)

                print(f"  {fpath.name}: {n_rows} rows → {bin_stats.shape} bin stats")

            # Print summary
            counts = manifest[label]["row_counts"]
            print(f"  Summary: {len(files)} runs, avg {np.mean(counts):.0f} rows/run "
                  f"(min={min(counts)}, max={max(counts)})")

    # ── Build air reference profiles ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("Building air reference profiles...")

    for toolpath in cfg.TOOLPATH_TYPES:
        prefix = cfg.AIR_PREFIX[toolpath]
        files = utils.discover_runs(prefix)

        # Collect all bin stats
        all_bins = []
        all_wholerun = []
        for fpath in files:
            run_id = utils.extract_run_id(fpath)
            bin_path = cfg.BIN_STATS_DIR / f"{prefix}_{run_id}_bins.parquet"
            all_bins.append(utils.load_parquet(bin_path).values)

            wr_path = cfg.BIN_STATS_DIR / f"{prefix}_{run_id}_wholerun.parquet"
            all_wholerun.append(utils.load_parquet(wr_path).values[0])

        all_bins = np.stack(all_bins)  # (n_runs, n_bins, n_features)
        all_wholerun = np.stack(all_wholerun)  # (n_runs, n_features)

        # Temporal bin reference: mean and std across runs
        bin_mean = np.mean(all_bins, axis=0)
        bin_std = np.std(all_bins, axis=0, ddof=1) if len(files) > 1 else np.zeros_like(bin_mean)

        col_names = utils.load_parquet(
            cfg.BIN_STATS_DIR / f"{prefix}_{utils.extract_run_id(files[0])}_bins.parquet"
        ).columns.tolist()

        utils.save_parquet(
            pd.DataFrame(bin_mean, columns=col_names),
            cfg.AIR_REF_DIR / f"{toolpath}_bin_mean.parquet"
        )
        utils.save_parquet(
            pd.DataFrame(bin_std, columns=col_names),
            cfg.AIR_REF_DIR / f"{toolpath}_bin_std.parquet"
        )

        # Whole-run reference
        wr_mean = np.mean(all_wholerun, axis=0)
        wr_std = np.std(all_wholerun, axis=0, ddof=1) if len(files) > 1 else np.zeros_like(wr_mean)

        wr_col_names = utils.load_parquet(
            cfg.BIN_STATS_DIR / f"{prefix}_{utils.extract_run_id(files[0])}_wholerun.parquet"
        ).columns.tolist()

        utils.save_parquet(
            pd.DataFrame([wr_mean], columns=wr_col_names),
            cfg.AIR_REF_DIR / f"{toolpath}_wholerun_mean.parquet"
        )
        utils.save_parquet(
            pd.DataFrame([wr_std], columns=wr_col_names),
            cfg.AIR_REF_DIR / f"{toolpath}_wholerun_std.parquet"
        )

        print(f"  {toolpath}: air ref from {len(files)} runs, "
              f"bins shape={bin_mean.shape}, features={bin_mean.shape[1]}")

    # Save manifest
    manifest_path = cfg.INTERMEDIATE_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=convert)

    print(f"\nManifest saved to {manifest_path}")
    print("Phase 1 complete.")


if __name__ == "__main__":
    run_phase1()
