#!/usr/bin/env python3
"""Phase-7 (decoder20260511): per-row label alignment for the summer DOE dataset.

When the Tormach arrives and we collect new data, each CSV will pair sensor
samples with line-by-line G-code execution. The alignment problem: a sensor
row at timestamp t needs to be labeled with WHICH G-code line was active at t.

This module implements the alignment used by the existing `data_clean/` CSVs
but parameterised so the same approach scales to the DOE runs.

Two strategies:
  - "timestamp": Match sensor row timestamps to G-code line start/end times
    (requires the CNC controller to log timestamps with each line).
  - "interpolate": Treat the G-code lines as a uniform sequence and use the
    sensor row count to subdivide. Coarser but works when timestamps are absent.

Input: raw CSV with sensor columns + a G-code source program (.gcode file).
Output: aligned CSV with `gcode_line` + `gcode_string` columns matching the
schema consumed by `src/miracle/dataset/preprocessing.py`.

This is a small utility — full alignment was already implemented in
`src/miracle/utilities/gcode_interpolate_align.py`. This module is a thin
wrapper that handles the DOE-specific manifest format.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


def align_doe_run(sensor_csv: Path, gcode_file: Path, output_csv: Path,
                  timestamp_col: Optional[str] = None) -> dict:
    """Align a DOE run's sensor CSV with its G-code program."""
    df = pd.read_csv(sensor_csv)
    gcode_lines = [l.strip() for l in gcode_file.read_text().splitlines() if l.strip() and not l.strip().startswith("(")]

    n_rows = len(df)
    n_lines = len(gcode_lines)

    if timestamp_col and timestamp_col in df.columns:
        # Timestamp-based alignment requires the CNC log to include a per-line
        # timestamp. Implementation deferred to when we have a sample CNC log.
        # Falls through to interpolation as a stub.
        pass

    # Interpolation alignment: each sensor row is assigned the G-code line
    # active at its proportional position in the run.
    line_idx_per_row = [
        min(int(i * n_lines / max(n_rows, 1)), n_lines - 1)
        for i in range(n_rows)
    ]
    df["gcode_line"] = line_idx_per_row
    df["gcode_string"] = [gcode_lines[idx] for idx in line_idx_per_row]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return {
        "sensor_csv": str(sensor_csv),
        "gcode_file": str(gcode_file),
        "output_csv": str(output_csv),
        "n_rows": n_rows,
        "n_gcode_lines": n_lines,
        "alignment": "interpolated",
    }


def main() -> int:
    p = argparse.ArgumentParser(description="DOE sensor + G-code alignment")
    p.add_argument("--manifest", type=Path, required=True,
                   help="manifest.json from generate_single_line_gcode.py")
    p.add_argument("--sensor-dir", type=Path, required=True,
                   help="Directory with raw sensor CSVs named {run_id}.csv")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--timestamp-col", default=None)
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    align_log = []
    for run_id, info in manifest["programs"].items():
        sensor_csv = args.sensor_dir / f"{run_id}.csv"
        gcode_file = Path(info["path"])
        output_csv = args.output_dir / f"{run_id}_aligned.csv"

        if not sensor_csv.exists():
            print(f"  skip {run_id}: sensor CSV not found at {sensor_csv}")
            continue

        try:
            rec = align_doe_run(sensor_csv, gcode_file, output_csv,
                                timestamp_col=args.timestamp_col)
            rec["run_id"] = run_id
            rec["factors"] = info["factors"]
            align_log.append(rec)
            print(f"  aligned {run_id} -> {output_csv.name} ({rec['n_rows']} rows, {rec['n_gcode_lines']} lines)")
        except Exception as e:
            print(f"  FAILED {run_id}: {e!r}")

    (args.output_dir / "alignment_log.json").write_text(json.dumps(align_log, indent=2))
    print(f"\nAligned {len(align_log)} runs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
