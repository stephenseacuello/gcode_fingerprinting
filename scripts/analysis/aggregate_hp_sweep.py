#!/usr/bin/env python3
"""Summarize the HP sweep results from outputs/decoder20260511/checkpoints/hp_sweep/.

Picks the winner by val_token_accuracy AND reports the impact on numeric_accuracy
(our weakest metric) and sequence_accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP = REPO / "outputs/decoder20260511/checkpoints/hp_sweep"

KEYS = ["token_accuracy", "sequence_accuracy", "type_accuracy",
        "command_accuracy", "param_type_accuracy", "numeric_accuracy"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP)
    args = p.parse_args()

    if not args.sweep_root.exists():
        print(f"no sweep results at {args.sweep_root}")
        return 1

    cells = []
    for d in sorted(args.sweep_root.iterdir()):
        if not d.is_dir(): continue
        m_path = d / "fold_1" / "results" / "metrics.json"
        if not m_path.exists(): continue
        m = json.loads(m_path.read_text())
        t = m.get("test_metrics", {})
        v = m.get("val_metrics", {})
        cells.append({
            "tag": d.name,
            "best_epoch": m.get("best_epoch"),
            "best_val_token": m.get("best_val_token_accuracy"),
            **{f"test_{k}": t.get(k) for k in KEYS},
            **{f"val_{k}": v.get(k) for k in KEYS},
        })

    if not cells:
        print("no completed cells")
        return 1

    print(f"\n{'Tag':<32} {'best_ep':<8} {'val_tok':<8} {'test_tok':<9} {'cmd':<8} {'num':<8} {'seq':<8}")
    print('-' * 90)
    for c in cells:
        print(f"  {c['tag']:<30} {c['best_epoch']:<8} "
              f"{c.get('best_val_token', 0):<8.4f} {c.get('test_token_accuracy', 0):<9.4f} "
              f"{c.get('test_command_accuracy', 0):<8.4f} {c.get('test_numeric_accuracy', 0):<8.4f} "
              f"{c.get('test_sequence_accuracy', 0):<8.4f}")

    # Pick winner
    cells_sorted = sorted(cells, key=lambda c: -(c.get('best_val_token') or 0))
    print(f"\n=== Winner by val_token_accuracy ===")
    w = cells_sorted[0]
    print(f"  {w['tag']}: val_tok={w['best_val_token']:.4f}, test_tok={w.get('test_token_accuracy', 0):.4f}, "
          f"cmd={w.get('test_command_accuracy', 0):.4f}, num={w.get('test_numeric_accuracy', 0):.4f}, seq={w.get('test_sequence_accuracy', 0):.4f}")

    # Pick winner by sequence (sometimes more useful)
    cells_by_seq = sorted(cells, key=lambda c: -(c.get('test_sequence_accuracy') or 0))
    print(f"\n=== Winner by test_sequence_accuracy ===")
    w = cells_by_seq[0]
    print(f"  {w['tag']}: seq={w['test_sequence_accuracy']:.4f}, tok={w.get('test_token_accuracy', 0):.4f}, cmd={w.get('test_command_accuracy', 0):.4f}, num={w.get('test_numeric_accuracy', 0):.4f}")

    # Save aggregate JSON with a filename keyed off the sweep dir
    out = REPO / "outputs/decoder20260511/audit" / f"{args.sweep_root.name}_summary.json"
    out.write_text(json.dumps({"cells": cells, "winner_by_val_token": cells_sorted[0], "winner_by_seq": cells_by_seq[0]}, indent=2))
    # Also write the canonical hp_sweep_summary.json for stage2 to pick up
    if args.sweep_root.name == "hp_sweep":
        canonical = REPO / "outputs/decoder20260511/audit/hp_sweep_summary.json"
        canonical.write_text(json.dumps({"cells": cells, "winner_by_val_token": cells_sorted[0], "winner_by_seq": cells_by_seq[0]}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
