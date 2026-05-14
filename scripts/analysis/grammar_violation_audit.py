#!/usr/bin/env python3
"""Count grammar violations in decoded predictions across folds.

Compares bigram-only AR (canonical beam_1 results) against FSM-equipped AR
(beam_1 results from *_fsm subdirs). The bigram grammar mask in the
decoder permits any PARAM letter after any move-command, including the
semantically-invalid G0/G1 -> R (R is arc-radius, valid only with
G2/G3). The inference-time FSM layer was added to forbid such
transitions.

Violations detected:
  - V1: G0 or G1 active, then R / I / J emitted.
  - V2: M30 emitted, then any PARAM letter or another command follows.

Active-command tracking is sequential: the most recently emitted
move-command is the state; a subsequent move-command updates it.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MOVE_CMDS_NO_ARC = {"G0", "G1", "G53", "G55", "G17", "G90", "G94"}
ARC_CMDS = {"G2", "G3"}
END_CMDS = {"M30"}
ALL_MOVE_CMDS = MOVE_CMDS_NO_ARC | ARC_CMDS | END_CMDS

PARAM_LETTERS = {"X", "Y", "Z", "F", "R", "S", "I", "J", "K"}
ARC_ONLY_PARAMS = {"R", "I", "J"}


def count_violations(token_stream):
    """Sequentially scan token_stream (list of strings) for grammar violations.

    Returns a dict {v1_g01_arc_param: int, v2_m30_followup: int,
    total_tokens: int, n_R_after_G01: int, n_R_after_arc: int}.
    """
    active_cmd = None
    v1 = 0  # G0/G1 active and R/I/J emitted
    v2 = 0  # M30 emitted then PARAM or command
    n_r_after_g01 = 0
    n_r_after_arc = 0

    for tok in token_stream:
        if tok in ALL_MOVE_CMDS:
            active_cmd = tok
            continue
        if active_cmd in MOVE_CMDS_NO_ARC and tok in ARC_ONLY_PARAMS:
            v1 += 1
            if tok == "R":
                n_r_after_g01 += 1
        if active_cmd in ARC_CMDS and tok == "R":
            n_r_after_arc += 1
        if active_cmd in END_CMDS and (tok in PARAM_LETTERS or tok in ALL_MOVE_CMDS):
            v2 += 1
            active_cmd = None  # Reset to avoid double-counting

    return {
        "v1_g01_arc_param": v1,
        "v2_m30_followup": v2,
        "n_R_after_G01": n_r_after_g01,
        "n_R_after_arc": n_r_after_arc,
        "total_tokens": len(token_stream),
    }


def audit_predictions_file(path: Path):
    samples = json.loads(path.read_text())
    agg = defaultdict(int)
    agg["n_samples"] = len(samples)
    n_v1_samples = 0
    n_v2_samples = 0
    for s in samples:
        toks = s.get("pred", "").split()
        v = count_violations(toks)
        for k, val in v.items():
            agg[k] += val
        if v["v1_g01_arc_param"] > 0:
            n_v1_samples += 1
        if v["v2_m30_followup"] > 0:
            n_v2_samples += 1
    agg["n_v1_samples"] = n_v1_samples
    agg["n_v2_samples"] = n_v2_samples
    agg["pct_samples_with_v1"] = 100.0 * n_v1_samples / max(len(samples), 1)
    agg["pct_samples_with_v2"] = 100.0 * n_v2_samples / max(len(samples), 1)
    return dict(agg)


def find_pred_files(sweep_root: Path, fsm: bool):
    out = {}
    for F in range(1, 6):
        if fsm:
            cands = list(sweep_root.glob(f"fold_{F}/*_fsm/results/beam_1_all_predictions.json"))
        else:
            cands = [p for p in sweep_root.glob(f"fold_{F}/*/results/beam_1_all_predictions.json") if "_fsm" not in str(p)]
        if cands:
            cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            out[F] = cands[0]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    result = {}
    for fsm, label in [(False, "bigram"), (True, "fsm")]:
        result[label] = {"per_fold": {}, "aggregate": {}}
        files = find_pred_files(args.sweep_root, fsm)
        for F, path in files.items():
            v = audit_predictions_file(path)
            v["path"] = str(path)
            result[label]["per_fold"][F] = v

        # Aggregate
        if result[label]["per_fold"]:
            total_v1 = sum(v["v1_g01_arc_param"] for v in result[label]["per_fold"].values())
            total_v2 = sum(v["v2_m30_followup"] for v in result[label]["per_fold"].values())
            total_samples = sum(v["n_samples"] for v in result[label]["per_fold"].values())
            total_v1_samples = sum(v["n_v1_samples"] for v in result[label]["per_fold"].values())
            total_v2_samples = sum(v["n_v2_samples"] for v in result[label]["per_fold"].values())
            total_tokens = sum(v["total_tokens"] for v in result[label]["per_fold"].values())
            result[label]["aggregate"] = {
                "n_folds": len(result[label]["per_fold"]),
                "n_samples": total_samples,
                "n_tokens": total_tokens,
                "v1_g01_arc_param_count": total_v1,
                "v2_m30_followup_count": total_v2,
                "n_samples_with_v1": total_v1_samples,
                "n_samples_with_v2": total_v2_samples,
                "pct_samples_with_v1": 100.0 * total_v1_samples / max(total_samples, 1),
                "pct_samples_with_v2": 100.0 * total_v2_samples / max(total_samples, 1),
                "v1_per_1000_tokens": 1000.0 * total_v1 / max(total_tokens, 1),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print(f"=== {args.sweep_root.name} ===")
    for label in ("bigram", "fsm"):
        a = result[label].get("aggregate", {})
        if not a:
            print(f"  {label}: no predictions found")
            continue
        print(f"  {label:6s}: {a['n_samples']} samples, {a['n_tokens']:,} tokens, "
              f"V1 (G0/G1->R/I/J): {a['v1_g01_arc_param_count']} occurrences "
              f"in {a['n_samples_with_v1']} samples ({a['pct_samples_with_v1']:.1f}%), "
              f"{a['v1_per_1000_tokens']:.2f} per 1k tokens. "
              f"V2 (M30 -> param/cmd): {a['v2_m30_followup_count']}.")

    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
