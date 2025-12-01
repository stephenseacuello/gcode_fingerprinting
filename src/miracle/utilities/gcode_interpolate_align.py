#!/usr/bin/env python3
"""
gcode_interpolate_align.py

Combined pipeline that:
1. Interpolates MCC and sensor logs onto SR timestamps
2. Labels with G-code strings based on line numbers in SR file

The script expects the SR file to contain G-code line numbers in a column.
It will auto-detect common column names like: 'line', 'gcode_line', 'ln', etc.

Usage Option 1 (with basename):
  python3 gcode_interpolate_align.py --basename test_001

  Expects files:
    - test_001_sr.csv      (status report with timestamps and line numbers)
    - test_001_mcc.csv     (MCCDAQ log)
    - test_001_sensor.csv  (sensor log)
    - test_001.gcode       (G-code file)
  Output:
    - test_001_aligned.csv (fully interpolated and labeled)

Usage Option 2 (with explicit paths):
  python3 gcode_interpolate_align.py \
    --sr data/status_report.csv \
    --mcc data/mccdaq_log.csv \
    --sensor data/sensors.csv \
    --gcode programs/mill_part.gcode \
    --out results/aligned_output.csv

Usage Option 3 (custom line column name):
  python3 gcode_interpolate_align.py \
    --basename test_001 \
    --sr_line_col my_custom_line_col
"""

import argparse
import re
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd


# ===========================================================================
# INTERPOLATION FUNCTIONS (from interpolate_to_sr.py)
# ===========================================================================

def parse_any_time(series):
    """Parse timestamp-like strings or HH:MM:SS(.fff) into pandas datetime"""
    dt = pd.to_datetime(series, errors="coerce")
    mask = dt.isna() & series.notna()
    if mask.any():
        def parse_hms(s):
            s = str(s).strip().replace(",", ".")
            try:
                return pd.to_datetime("1970-01-01 " + s, errors="coerce")
            except Exception:
                return pd.NaT
        dt.loc[mask] = series[mask].map(parse_hms)
    mask = dt.isna() & series.notna()
    if mask.any():
        def parse_epoch(x):
            try:
                return pd.to_datetime(float(x), unit="s")
            except Exception:
                return pd.NaT
        dt.loc[mask] = series[mask].map(parse_epoch)
    return dt


def to_seconds_of_day(dt_series):
    """Convert datetime series to seconds of day"""
    dt_series = pd.to_datetime(dt_series, errors="coerce")
    return (dt_series - dt_series.dt.normalize()).dt.total_seconds()


def load_numeric(df, drop_cols):
    """Extract numeric columns excluding specified columns"""
    num_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    return df[num_cols].copy()


def interpolate_log_to_targets(log_df, log_time_col, target_times, prefix):
    """Interpolate log data onto target timestamps"""
    if log_df.empty:
        return pd.DataFrame(index=range(len(target_times)))
    dt = parse_any_time(log_df[log_time_col])
    s = to_seconds_of_day(dt)
    tmp = log_df.copy()
    tmp["_sec"] = s
    tmp = tmp.dropna(subset=["_sec"]).sort_values("_sec").drop_duplicates("_sec", keep="last")
    if tmp.empty:
        return pd.DataFrame(index=range(len(target_times)))
    num = load_numeric(tmp, drop_cols=[log_time_col, "_sec"])
    res = pd.DataFrame(index=np.arange(len(target_times)))
    x = tmp["_sec"].to_numpy()
    for col in num.columns:
        y = tmp[col].to_numpy(dtype=float)
        mask = ~np.isnan(y) & ~np.isnan(x)
        xcol = x[mask]; ycol = y[mask]
        if len(xcol) == 0:
            res[prefix + col] = np.full_like(target_times, np.nan, dtype=float)
            continue
        y_interp = np.interp(target_times, xcol, ycol, left=ycol[0], right=ycol[-1])
        res[prefix + col] = y_interp
    return res


# ===========================================================================
# G-CODE PARSING FUNCTIONS
# ===========================================================================

def parse_gcode_file(path: str) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Parse a .gcode file and track evolving X/Y/Z, feed, spindle, motion mode"""
    gcode_lines: List[Dict[str, float]] = []
    line_to_cmd: Dict[int, str] = {}

    x = y = z = 0.0
    feed_rate = 0.0
    spindle_speed = 0.0
    motion_mode = None  # 0=G0, 1=G1, 2=G2, 3=G3

    with open(path, "r") as f:
        for line_num, raw_line in enumerate(f, 1):
            full_text = raw_line.rstrip("\n")
            line_to_cmd[line_num] = full_text

            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(("(", ";", "//")):
                continue

            # Spindle (M3 + S####)
            if re.search(r"\bM3\b", line):
                sm = re.search(r"S(\d+)", line)
                if sm:
                    spindle_speed = float(sm.group(1))

            # Motion mode
            if re.search(r"\bG0\b", line):
                motion_mode = 0
            elif re.search(r"\bG1\b", line):
                motion_mode = 1
            elif re.search(r"\bG2\b", line):
                motion_mode = 2
            elif re.search(r"\bG3\b", line):
                motion_mode = 3

            # Axes and feed
            xm = re.search(r"X([-\d.]+)", line)
            ym = re.search(r"Y([-\d.]+)", line)
            zm = re.search(r"Z([-\d.]+)", line)
            fm = re.search(r"F([-\d.]+)", line)

            if xm:
                x = float(xm.group(1))
            if ym:
                y = float(ym.group(1))
            if zm:
                z = float(zm.group(1))
            if fm:
                feed_rate = float(fm.group(1))

            if xm or ym or zm or re.match(r"^(G0|G1|G2|G3)", line):
                gcode_lines.append(
                    {
                        "line": line_num,
                        "cmd": line,
                        "x": x,
                        "y": y,
                        "z": z,
                        "feed": feed_rate,
                        "spindle": spindle_speed,
                        "momo": motion_mode,
                    }
                )

    gcode_motion = pd.DataFrame(gcode_lines)
    if gcode_motion.empty:
        raise SystemExit(f"ERROR: No motion lines found in G-code file: {path}")

    return gcode_motion, line_to_cmd


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Combined interpolation and G-code labeling pipeline for CNC logs"
    )
    
    # File path arguments
    parser.add_argument(
        "--basename",
        help="Base name for files (e.g., 'test_001' for test_001_sr.csv, etc.). "
             "If provided, will construct file paths automatically unless overridden by specific file arguments."
    )
    parser.add_argument(
        "--sr",
        help="Path to SR (status report) CSV file"
    )
    parser.add_argument(
        "--mcc",
        help="Path to MCC (MCCDAQ) CSV file"
    )
    parser.add_argument(
        "--sensor",
        help="Path to sensor CSV file"
    )
    parser.add_argument(
        "--gcode",
        help="Path to G-code file"
    )
    parser.add_argument(
        "--out",
        help="Output path for aligned CSV file"
    )
    
    # Time column arguments
    parser.add_argument(
        "--sr_timecol",
        default="t_console",
        help="Time column name in SR file (default: t_console)"
    )
    parser.add_argument(
        "--mcc_timecol",
        default="timestamp",
        help="Time column name in MCC file (default: timestamp)"
    )
    parser.add_argument(
        "--sensor_timecol",
        default="ts_ms",
        help="Time column name in sensor file (default: ts_ms)"
    )
    
    # Prefix arguments
    parser.add_argument(
        "--mcc_prefix",
        default="",
        help="Prefix for MCC columns in output (default: none)"
    )
    parser.add_argument(
        "--sensor_prefix",
        default="",
        help="Prefix for sensor columns in output (default: none)"
    )
    
    # G-code line column
    parser.add_argument(
        "--sr_line_col",
        default=None,
        help="Column name in SR file containing G-code line numbers. "
             "If not specified, will auto-detect from common names: 'line', 'gcode_line', 'ln', etc."
    )

    args = parser.parse_args()

    # Build file paths - either from basename or explicit arguments
    if args.basename:
        sr_file = args.sr or f"{args.basename}_sr.csv"
        mcc_file = args.mcc or f"{args.basename}_mcc.csv"
        sensor_file = args.sensor or f"{args.basename}_sensor.csv"
        gcode_file = args.gcode or f"{args.basename}.gcode"
        output_file = args.out or f"{args.basename}_aligned.csv"
    else:
        # If no basename, all file paths must be explicit
        if not all([args.sr, args.mcc, args.sensor, args.gcode, args.out]):
            parser.error(
                "Either --basename must be provided, or all of "
                "--sr, --mcc, --sensor, --gcode, and --out must be specified"
            )
        sr_file = args.sr
        mcc_file = args.mcc
        sensor_file = args.sensor
        gcode_file = args.gcode
        output_file = args.out

    print("=" * 70)
    print("CNC ALIGNMENT PIPELINE")
    print("=" * 70)

    # Check that files exist
    for fname in [sr_file, mcc_file, sensor_file, gcode_file]:
        if not os.path.exists(fname):
            raise SystemExit(f"ERROR: File not found: {fname}")

    # ===========================================================================
    # PART 1: INTERPOLATION
    # ===========================================================================
    print(f"\n[STEP 1] Loading SR file: {sr_file}")
    sr = pd.read_csv(sr_file)
    if args.sr_timecol not in sr.columns:
        raise SystemExit(f"SR time column '{args.sr_timecol}' not found in {sr_file}")
    print(f"Loaded {len(sr)} rows with {len(sr.columns)} columns.")

    print(f"\n[STEP 1b] Parsing SR timestamps...")
    sr_dt = parse_any_time(sr[args.sr_timecol])
    sr_sec = to_seconds_of_day(sr_dt).to_numpy()

    print(f"\n[STEP 2a] Loading MCC file: {mcc_file}")
    mcc = pd.read_csv(mcc_file)
    print(f"Loaded {len(mcc)} rows with {len(mcc.columns)} columns.")

    print(f"\n[STEP 2b] Loading sensor file: {sensor_file}")
    sensor = pd.read_csv(sensor_file)
    print(f"Loaded {len(sensor)} rows with {len(sensor.columns)} columns.")

    print(f"\n[STEP 2c] Interpolating MCC data onto SR timestamps...")
    part_mcc = interpolate_log_to_targets(mcc, args.mcc_timecol, sr_sec, prefix=args.mcc_prefix)

    print(f"\n[STEP 2d] Interpolating sensor data onto SR timestamps...")
    part_sensor = interpolate_log_to_targets(sensor, args.sensor_timecol, sr_sec, prefix=args.sensor_prefix)

    print(f"\n[STEP 2e] Combining interpolated data...")
    interpolated_df = sr.copy().reset_index(drop=True)
    interpolated_df = pd.concat([interpolated_df, part_mcc, part_sensor], axis=1)
    print(f"Combined dataframe has {len(interpolated_df)} rows with {len(interpolated_df.columns)} columns.")

    # ===========================================================================
    # PART 2: G-CODE LABELING
    # ===========================================================================
    print(f"\n[STEP 3a] Parsing G-code file: {gcode_file}")
    gcode_motion, line_to_cmd = parse_gcode_file(gcode_file)
    print(f"  Parsed {len(gcode_motion)} motion lines from G-code.")

    print(f"\n[STEP 3b] Finding G-code line column in SR file...")
    
    # Auto-detect line column if not specified
    if args.sr_line_col:
        line_col = args.sr_line_col
        if line_col not in interpolated_df.columns:
            raise SystemExit(
                f"ERROR: Specified SR line column '{line_col}' not found in SR file.\n"
                f"Available columns: {list(interpolated_df.columns)}"
            )
    else:
        # Try common line column names
        line_candidates = ["line", "gcode_line", "ln", "sr_line", "gcline", "g_line"]
        line_col = None
        for candidate in line_candidates:
            if candidate in interpolated_df.columns:
                line_col = candidate
                break
        
        if line_col is None:
            raise SystemExit(
                f"ERROR: Could not find G-code line column in SR file.\n"
                f"Tried: {line_candidates}\n"
                f"Available columns: {list(interpolated_df.columns)}\n"
                f"Please specify the column name with --sr_line_col"
            )
    
    print(f"  Using line numbers from SR column: {line_col}")
    
    # Use line numbers from SR file directly
    interpolated_df["gcode_line"] = interpolated_df[line_col].fillna(0).astype(int)
    
    # Attach G-code strings
    def lookup_cmd(ln: int) -> str:
        return line_to_cmd.get(int(ln), "")
    
    interpolated_df["gcode_string"] = interpolated_df["gcode_line"].apply(lookup_cmd)
    
    print(f"  Assigned G-code strings for {len(interpolated_df)} rows.")
    unique_lines = interpolated_df["gcode_line"].nunique()
    print(f"  Found {unique_lines} unique G-code lines in the data.")

    # ===========================================================================
    # SAVE OUTPUT
    # ===========================================================================
    print(f"\n[STEP 4] Saving aligned output: {output_file}")
    interpolated_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    print("\nDone! Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()