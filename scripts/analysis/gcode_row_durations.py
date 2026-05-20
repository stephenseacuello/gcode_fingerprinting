#!/usr/bin/env python3
"""G-code row execution-duration distribution at the 4 Hz sample rate.

Referee R2 M2/R2: the central physical caveat of the paper is that a 4 Hz
sensor stack cannot resolve sub-second G-code instructions. This converts
that caveat from a qualitative assertion into a measured statement: across
the 120-file aligned corpus, how long (in 4 Hz samples / seconds) does each
G-code row actually execute, and what fraction of row-execution instances
occupy <= 1 sample period (0.25 s) -- i.e. are at or below the Nyquist
limit and therefore temporally aliased / unresolvable as transients?

A "row-execution instance" is a maximal contiguous run of 4 Hz samples
sharing the same aligned G-code line (the `line` column; blank/NaN
gcode_string rows excluded). Duration = run_length / 4 Hz.

Outputs audit/gcode_row_durations.json and prints a compact table.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FS_HZ = 4.0
DT = 1.0 / FS_HZ  # 0.25 s


def run_lengths(line_ids: np.ndarray) -> list[int]:
    """Lengths of maximal contiguous constant runs."""
    if len(line_ids) == 0:
        return []
    change = np.flatnonzero(np.diff(line_ids) != 0) + 1
    bounds = np.concatenate(([0], change, [len(line_ids)]))
    return np.diff(bounds).tolist()


def main() -> int:
    files = sorted(glob.glob(str(REPO / "data_clean" / "*_aligned.csv")))
    if not files:
        print("no aligned CSVs found")
        return 1

    all_runs_samples: list[int] = []
    per_file = {}
    for f in files:
        df = pd.read_csv(f, usecols=lambda c: c in ("line", "gcode_string"))
        if "line" not in df.columns:
            continue
        # Keep only samples that carry an actual G-code line.
        mask = df["gcode_string"].notna() & (df["gcode_string"].astype(str).str.strip() != "")
        sub = df.loc[mask, "line"].to_numpy()
        rl = run_lengths(sub)
        all_runs_samples.extend(rl)
        per_file[Path(f).stem] = {
            "n_row_instances": len(rl),
            "median_samples": float(np.median(rl)) if rl else 0.0,
        }

    a = np.asarray(all_runs_samples, dtype=float)
    n = len(a)
    secs = a * DT

    def frac_le(samples_thresh: int) -> float:
        return float(np.mean(a <= samples_thresh))

    summary = {
        "sample_rate_hz": FS_HZ,
        "sample_period_s": DT,
        "n_row_execution_instances": int(n),
        "n_files": len(per_file),
        "duration_samples": {
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
            "max": float(a.max()),
        },
        "duration_seconds": {
            "mean": float(secs.mean()),
            "median": float(np.median(secs)),
            "p90": float(np.percentile(secs, 90)),
        },
        # The Nyquist-relevant fractions:
        "frac_le_1_sample_0p25s": frac_le(1),
        "frac_le_2_samples_0p50s": frac_le(2),
        "frac_le_4_samples_1p00s": frac_le(4),
        "frac_ge_256_samples_64s": float(np.mean(a >= 256)),
    }
    out = REPO / "outputs/decoder20260511/audit/gcode_row_durations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_file": per_file}, indent=2))

    s = summary
    print("\n=== G-code row execution durations (4 Hz aligned corpus) ===")
    print(f"files={s['n_files']}  row-execution instances={s['n_row_execution_instances']}")
    print(f"duration: median {s['duration_samples']['median']:.0f} samples "
          f"({s['duration_seconds']['median']:.2f} s), "
          f"mean {s['duration_seconds']['mean']:.2f} s, "
          f"p90 {s['duration_seconds']['p90']:.2f} s")
    print(f"FRACTION <= 1 sample (0.25 s, sub-Nyquist): {100*s['frac_le_1_sample_0p25s']:.1f}%")
    print(f"FRACTION <= 2 samples (0.50 s):             {100*s['frac_le_2_samples_0p50s']:.1f}%")
    print(f"FRACTION <= 4 samples (1.00 s):             {100*s['frac_le_4_samples_1p00s']:.1f}%")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
