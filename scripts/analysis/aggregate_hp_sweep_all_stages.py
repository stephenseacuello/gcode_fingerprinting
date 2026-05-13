#!/usr/bin/env python3
"""Aggregate HP-sweep results across Stage 1 + Stage 2 + Stage 3.

Reads metrics from:
  outputs/decoder20260511/checkpoints/hp_sweep/         (Stage 1, 8 cells)
  outputs/decoder20260511/checkpoints/hp_sweep_stage2/  (Stage 2, 12 cells)
  outputs/decoder20260511/checkpoints/hp_sweep_stage3/  (Stage 3, 14 cells)

Ranks by multiple criteria:
  - best val_token_accuracy
  - best test_token_accuracy
  - best test_command_accuracy
  - best test_sequence_accuracy
  - best test_numeric_accuracy

Picks an overall winner using a composite score and writes:
  outputs/decoder20260511/audit/hp_sweep_all_stages_summary.json
  outputs/decoder20260511/audit/hp_sweep_all_stages_summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGE_DIRS = [
    ("stage1", REPO / "outputs/decoder20260511/checkpoints/hp_sweep"),
    ("stage2", REPO / "outputs/decoder20260511/checkpoints/hp_sweep_stage2"),
    ("stage3", REPO / "outputs/decoder20260511/checkpoints/hp_sweep_stage3"),
]
KEYS = ["token_accuracy", "sequence_accuracy", "type_accuracy",
        "command_accuracy", "param_type_accuracy", "numeric_accuracy"]
OUT_JSON = REPO / "outputs/decoder20260511/audit/hp_sweep_all_stages_summary.json"
OUT_MD = REPO / "outputs/decoder20260511/audit/hp_sweep_all_stages_summary.md"


def _find_metrics_file(fold_dir: Path) -> Path | None:
    """Look for metrics.json in fold_1/results/ (no wandb) or fold_1/<wandb_id>/results/ (wandb on)."""
    direct = fold_dir / "results" / "metrics.json"
    if direct.exists():
        return direct
    # wandb sub-run case: fold_1/<run_id>/results/metrics.json
    candidates = list(fold_dir.glob("*/results/metrics.json"))
    if candidates:
        # Use the most-recently-modified one
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    return None


def load_cells(stage_dir: Path, stage_name: str) -> list[dict]:
    cells = []
    if not stage_dir.exists(): return cells
    for d in sorted(stage_dir.iterdir()):
        if not d.is_dir(): continue
        m_path = _find_metrics_file(d / "fold_1")
        if m_path is None: continue
        m = json.loads(m_path.read_text())
        t = m.get("test_metrics", {})
        v = m.get("val_metrics", {})
        cells.append({
            "stage": stage_name,
            "tag": d.name,
            "best_epoch": m.get("best_epoch"),
            "best_val_token": m.get("best_val_token_accuracy"),
            **{f"test_{k}": t.get(k) for k in KEYS},
            **{f"val_{k}": v.get(k) for k in KEYS},
        })
    return cells


def composite_score(c: dict) -> float:
    """Weighted geometric mean of test_token + test_command + test_numeric.
    Heavier weight on token (the headline metric) and command (the recoverability metric).
    """
    tok = c.get("test_token_accuracy") or 0
    cmd = c.get("test_command_accuracy") or 0
    num = c.get("test_numeric_accuracy") or 0
    return 0.5 * tok + 0.3 * cmd + 0.2 * num


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    args = p.parse_args()

    all_cells = []
    for stage_name, stage_dir in STAGE_DIRS:
        cells = load_cells(stage_dir, stage_name)
        print(f"  {stage_name}: {len(cells)} cells loaded from {stage_dir.name}")
        all_cells.extend(cells)

    if not all_cells:
        print("no cells found across any stage")
        return 1

    for c in all_cells:
        c["composite_score"] = composite_score(c)

    # Sort by each criterion
    by_val_tok = sorted(all_cells, key=lambda c: -(c.get('best_val_token') or 0))
    by_test_tok = sorted(all_cells, key=lambda c: -(c.get('test_token_accuracy') or 0))
    by_test_cmd = sorted(all_cells, key=lambda c: -(c.get('test_command_accuracy') or 0))
    by_test_seq = sorted(all_cells, key=lambda c: -(c.get('test_sequence_accuracy') or 0))
    by_test_num = sorted(all_cells, key=lambda c: -(c.get('test_numeric_accuracy') or 0))
    by_composite = sorted(all_cells, key=lambda c: -c.get('composite_score', 0))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "all_cells": all_cells,
        "winners": {
            "by_val_token": by_val_tok[0],
            "by_test_token": by_test_tok[0],
            "by_test_command": by_test_cmd[0],
            "by_test_sequence": by_test_seq[0],
            "by_test_numeric": by_test_num[0],
            "by_composite": by_composite[0],
        },
        "top10_composite": by_composite[:10],
    }, indent=2))

    # Markdown summary
    lines = ["# HP Sweep All-Stages Summary", "",
             f"**Total cells:** {len(all_cells)} ({sum(1 for c in all_cells if c['stage']=='stage1')} stage1 + "
             f"{sum(1 for c in all_cells if c['stage']=='stage2')} stage2 + "
             f"{sum(1 for c in all_cells if c['stage']=='stage3')} stage3)", "",
             "## Winners by criterion", ""]
    for name, w in [("val_token", by_val_tok[0]), ("test_token", by_test_tok[0]),
                    ("test_command", by_test_cmd[0]), ("test_sequence", by_test_seq[0]),
                    ("test_numeric", by_test_num[0]), ("composite", by_composite[0])]:
        lines.append(f"- **by {name}:** `{w['stage']}/{w['tag']}` — "
                     f"val_tok={w.get('best_val_token', 0):.4f} test_tok={w.get('test_token_accuracy', 0):.4f} "
                     f"cmd={w.get('test_command_accuracy', 0):.4f} num={w.get('test_numeric_accuracy', 0):.4f} "
                     f"seq={w.get('test_sequence_accuracy', 0):.4f} composite={w['composite_score']:.4f}")

    lines += ["", "## Top 10 by composite score", "",
              "| rank | stage | tag | val_tok | test_tok | cmd | num | seq | composite |",
              "|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(by_composite[:10], 1):
        lines.append(f"| {i} | {c['stage']} | {c['tag']} | "
                     f"{c.get('best_val_token', 0):.4f} | "
                     f"{c.get('test_token_accuracy', 0):.4f} | "
                     f"{c.get('test_command_accuracy', 0):.4f} | "
                     f"{c.get('test_numeric_accuracy', 0):.4f} | "
                     f"{c.get('test_sequence_accuracy', 0):.4f} | "
                     f"{c['composite_score']:.4f} |")

    lines += ["", "## All cells (sorted by composite)", "",
              "| stage | tag | best_ep | val_tok | test_tok | cmd | num | seq | composite |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in by_composite:
        lines.append(f"| {c['stage']} | {c['tag']} | {c.get('best_epoch', '-')} | "
                     f"{c.get('best_val_token', 0):.4f} | "
                     f"{c.get('test_token_accuracy', 0):.4f} | "
                     f"{c.get('test_command_accuracy', 0):.4f} | "
                     f"{c.get('test_numeric_accuracy', 0):.4f} | "
                     f"{c.get('test_sequence_accuracy', 0):.4f} | "
                     f"{c['composite_score']:.4f} |")

    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"\n=== COMPOSITE WINNER ===")
    w = by_composite[0]
    print(f"  {w['stage']}/{w['tag']}: composite={w['composite_score']:.4f} "
          f"tok={w.get('test_token_accuracy', 0):.4f} cmd={w.get('test_command_accuracy', 0):.4f} "
          f"num={w.get('test_numeric_accuracy', 0):.4f} seq={w.get('test_sequence_accuracy', 0):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
