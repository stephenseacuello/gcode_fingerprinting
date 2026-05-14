#!/usr/bin/env python3
"""Aggregate beam_1_metrics.json (autoregressive greedy) across folds + sweeps.

Pairs with aggregate_v8_results.py which reports the teacher-forced
(beam_0) sweep-time metrics. This script reports the deployment-true
autoregressive metrics that the paper headlines on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs" / "decoder20260511" / "checkpoints"

SWEEPS = {
    "Baseline (full_window, ss=0.5, no shortcuts)": "full_window_5fold",
    "+ positional metadata (shortcuts)": "full_window_5fold_with_shortcuts",
}


def _find_beam_metrics(fold_dir: Path, beam_width: int) -> Path | None:
    direct = fold_dir / "results" / f"beam_{beam_width}_metrics.json"
    if direct.exists():
        return direct
    cands = list(fold_dir.glob(f"*/results/beam_{beam_width}_metrics.json"))
    if cands:
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0]
    return None


def aggregate_sweep(sweep_dir: str, beam_widths: tuple[int, ...] = (0, 1)) -> dict:
    out = {}
    for bw in beam_widths:
        rows = []
        for F in range(1, 6):
            p = _find_beam_metrics(ROOT / sweep_dir / f"fold_{F}", bw)
            if p is None:
                continue
            d = json.loads(p.read_text())
            tm = d.get("test_metrics", d)
            rows.append({
                "fold": F,
                "token": tm.get("token_accuracy"),
                "sequence": tm.get("sequence_accuracy"),
                "type": tm.get("type_accuracy"),
                "command": tm.get("command_accuracy"),
                "param_type": tm.get("param_type_accuracy"),
                "sign": tm.get("sign_accuracy"),
                "numeric": tm.get("numeric_accuracy"),
            })
        if not rows:
            continue
        agg = {"n_folds": len(rows), "per_fold": rows}
        for k in ["token", "sequence", "type", "command", "param_type", "sign", "numeric"]:
            vals = [r[k] for r in rows if r.get(k) is not None]
            if not vals:
                continue
            agg[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
        out[bw] = agg
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=Path,
        default=REPO / "outputs" / "decoder20260511" / "audit" / "ar_aggregate.json",
    )
    args = p.parse_args()

    result = {}
    for label, sweep in SWEEPS.items():
        bws = aggregate_sweep(sweep)
        result[sweep] = {"label": label, "by_beam_width": bws}
        print(f"\n=== {label} ({sweep}) ===")
        for bw, agg in bws.items():
            bw_label = {0: "TF", 1: "AR-greedy", 3: "beam-3", 5: "beam-5"}.get(bw, f"beam-{bw}")
            print(f"  {bw_label} (n={agg['n_folds']} folds):")
            for metric in ["token", "command", "numeric", "type", "param_type", "sign"]:
                m = agg.get(metric)
                if m:
                    print(f"    {metric:12s} {m['mean']:.4f} ± {m['std']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
