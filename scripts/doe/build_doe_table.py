#!/usr/bin/env python3
"""Phase-7 (decoder20260511): build a designed-experiment (DOE) factor table.

Generates a factorial or fractional-factorial table of CNC experiments to be
run on the new Tormach when it arrives. Each row of the output describes one
distinct experiment (one short G-code program) parameterised by:

    motion_type:   {straight, arc_cw, arc_ccw}
    direction:     {x_pos, x_neg, y_pos, y_neg, xy_diag}   (NA for arcs)
    feed_rate:     mm/min, levels configurable
    spindle_speed: rpm, levels configurable
    depth_of_cut:  mm, levels configurable
    material:      {aluminum, steel, plastic, ...}
    tool_diameter: mm, levels configurable
    air_cut:       bool

Output: CSV + JSON. The JSON also embeds a metadata schema that the
preprocessing pipeline can consume during data ingestion.

Usage:
    python scripts/doe/build_doe_table.py --output outputs/decoder20260511/DOE/doe_v1.json

By default emits a fractional factorial. Set --full-factorial for the full grid.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

DEFAULT_LEVELS = {
    "motion_type": ["straight", "arc_cw", "arc_ccw"],
    "direction":   ["x_pos", "x_neg", "y_pos", "y_neg", "xy_diag"],
    "feed_rate":   [50.0, 100.0, 200.0, 400.0],      # mm/min
    "spindle_speed":[3000, 6000, 9000],              # rpm
    "depth_of_cut":[0.10, 0.25, 0.50, 1.00],         # mm
    "material":    ["aluminum_6061", "steel_1018", "delrin"],
    "tool_diameter":[3.0, 6.0, 9.5],                 # mm
    "air_cut":     [False, True],
}


def _full_factorial(levels: dict[str, list]) -> list[dict[str, Any]]:
    keys = list(levels.keys())
    rows = []
    for vals in itertools.product(*[levels[k] for k in keys]):
        rows.append(dict(zip(keys, vals)))
    return rows


def _fractional_factorial(levels: dict[str, list], target_n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Latin-hypercube-style fractional design: sample ~target_n cells.

    For each factor with K levels, ensure each level appears roughly target_n/K
    times across the selected rows. This is NOT a formal DOE design, but it
    gives reasonable coverage for an exploratory dataset.
    """
    import random
    random.seed(seed)

    full = _full_factorial(levels)
    random.shuffle(full)
    return full[:target_n]


def filter_invalid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop rows whose factor combinations are physically nonsensical."""
    out = []
    for r in rows:
        # Direction is only meaningful for straight motion; collapse to "n/a" for arcs.
        if r["motion_type"] != "straight":
            r = dict(r); r["direction"] = "n/a"
        # Air cuts: depth_of_cut is 0 by definition; force consistency.
        if r["air_cut"]:
            r = dict(r); r["depth_of_cut"] = 0.0
        out.append(r)
    # Dedup
    seen = set()
    unique = []
    for r in out:
        key = tuple(sorted(r.items()))
        if key not in seen:
            seen.add(key); unique.append(r)
    return unique


def main() -> int:
    p = argparse.ArgumentParser(description="DOE table generator for the summer dataset")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--full-factorial", action="store_true",
                   help="Emit full factorial (large). Default: sampled subset.")
    p.add_argument("--target-n", type=int, default=300,
                   help="Target rows for fractional design (default 300).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.full_factorial:
        rows = _full_factorial(DEFAULT_LEVELS)
    else:
        rows = _fractional_factorial(DEFAULT_LEVELS, args.target_n, args.seed)

    rows = filter_invalid(rows)

    # Assign run_id and metadata schema
    for i, r in enumerate(rows):
        r["run_id"] = f"DOE_{i:04d}"

    output = {
        "schema_version": "decoder20260511-doe-v1",
        "factor_levels": DEFAULT_LEVELS,
        "n_runs": len(rows),
        "runs": rows,
        "metadata_schema": {
            "per_run_fields": [
                "run_id", "motion_type", "direction", "feed_rate",
                "spindle_speed", "depth_of_cut", "material", "tool_diameter",
                "air_cut", "expected_runtime_sec",
            ],
            "per_row_csv_columns": [
                "timestamp", "gcode_line_index", "gcode_line", "x", "y", "z",
                "sensor_*",
            ],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))

    csv_path = args.output.with_suffix(".csv")
    keys = ["run_id"] + [k for k in DEFAULT_LEVELS.keys()]
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    print(f"wrote DOE: {args.output} ({len(rows)} runs)")
    print(f"wrote CSV: {csv_path}")
    print(f"factor coverage:")
    for k, vs in DEFAULT_LEVELS.items():
        used = sorted({str(r.get(k)) for r in rows})
        print(f"  {k}: {used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
