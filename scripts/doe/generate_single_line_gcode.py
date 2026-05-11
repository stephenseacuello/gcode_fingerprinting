#!/usr/bin/env python3
"""Phase-7 (decoder20260511): generate single-line G-code experiments.

Reads a DOE table produced by `build_doe_table.py` and emits one short G-code
program per run row. Each program is intentionally minimal: setup → move →
exit, so that the resulting sensor recording contains exactly ONE G-code
motion of interest, surrounded by air rapids.

This is the "per_row" data the audit (AUDIT_REPORT.md, Priority 10) calls for:
sensors recorded against a known G-code line in a controlled DOE setting.

Output: one .gcode file per run plus a manifest.json mapping run_id → file.

Usage:
    python scripts/doe/generate_single_line_gcode.py \\
        --doe outputs/decoder20260511/DOE/doe_v1.json \\
        --output-dir outputs/decoder20260511/DOE/programs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GCODE_HEADER = [
    "(DOE run header)",
    "G21 (units = mm)",
    "G90 (absolute distance mode)",
    "G17 (XY plane)",
    "G94 (feed = units/min)",
]
GCODE_FOOTER = [
    "M5 (spindle stop)",
    "M30 (program end)",
]


def emit_motion(direction: str, distance_mm: float, feed_rate: float) -> list[str]:
    """Emit a single G1 straight motion."""
    delta = {
        "x_pos": ( distance_mm, 0.0, 0.0),
        "x_neg": (-distance_mm, 0.0, 0.0),
        "y_pos": (0.0,  distance_mm, 0.0),
        "y_neg": (0.0, -distance_mm, 0.0),
        "xy_diag": ( distance_mm * 0.7071,  distance_mm * 0.7071, 0.0),
    }.get(direction, (distance_mm, 0.0, 0.0))
    dx, dy, dz = delta
    return [f"G1 X{dx:.4f} Y{dy:.4f} F{feed_rate:.1f}"]


def emit_arc(motion_type: str, distance_mm: float, feed_rate: float) -> list[str]:
    """Emit a single arc move (CW = G2 or CCW = G3) with arc radius R."""
    cmd = "G2" if motion_type == "arc_cw" else "G3"
    # Use R-form for simplicity (cleaner training target than IJK)
    radius = distance_mm / 2.0
    return [f"{cmd} X{distance_mm:.4f} Y0.0000 R{radius:.4f} F{feed_rate:.1f}"]


def build_program(run: dict, distance_mm: float = 25.0) -> list[str]:
    lines = list(GCODE_HEADER)
    # Spindle on (if not air cut)
    if not run.get("air_cut", False):
        lines.append(f"M3 S{int(run['spindle_speed'])} (spindle on, RPM)")
    # Rapid to start position above stock
    lines.append("G0 X0.0 Y0.0 Z2.5 (safe Z)")
    # Plunge to depth_of_cut
    depth = -float(run.get("depth_of_cut", 0.0))
    lines.append(f"G1 Z{depth:.4f} F{float(run.get('feed_rate', 100.0)):.1f}")
    # The single G-code line we are characterising
    motion_type = run.get("motion_type", "straight")
    feed_rate = float(run.get("feed_rate", 100.0))
    if motion_type == "straight":
        lines += emit_motion(run.get("direction", "x_pos"), distance_mm, feed_rate)
    else:
        lines += emit_arc(motion_type, distance_mm, feed_rate)
    # Retract and exit
    lines.append("G0 Z2.5 (retract)")
    lines += GCODE_FOOTER
    return lines


def main() -> int:
    p = argparse.ArgumentParser(description="Emit single-line G-code programs from a DOE table")
    p.add_argument("--doe", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--distance-mm", type=float, default=25.0,
                   help="Distance of the characterising move (default 25mm)")
    args = p.parse_args()

    doe = json.loads(args.doe.read_text())
    runs = doe["runs"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"doe_source": str(args.doe), "n_runs": len(runs), "programs": {}}

    for r in runs:
        lines = build_program(r, distance_mm=args.distance_mm)
        path = args.output_dir / f"{r['run_id']}.gcode"
        path.write_text("\n".join(lines) + "\n")
        manifest["programs"][r["run_id"]] = {"path": str(path), "factors": {k: v for k, v in r.items() if k != "run_id"}}

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(runs)} programs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
