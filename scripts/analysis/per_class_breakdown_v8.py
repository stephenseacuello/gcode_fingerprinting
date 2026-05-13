#!/usr/bin/env python3
"""Phase-C: per-operation-class accuracy breakdown for V8.

Mirrors the V7 paper's per-class table. For each operation class
(face / pocket / adaptive / 150025 variants / damage variants), reports
the per-class token + sequence + command accuracy by joining the saved
per-sample predictions with the NPZ's operation_type_names.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def per_op_class(predictions_npz: Path, src_npz: Path) -> dict[str, Any]:
    d = np.load(predictions_npz, allow_pickle=True)
    src = np.load(src_npz, allow_pickle=True)
    op_names = np.array([str(s) for s in src["operation_type_names"]])

    pred_tokens = d["pred_tokens"]  # [N, L]
    target_tokens = d["target_tokens"]  # [N, L]
    n = pred_tokens.shape[0]
    if len(op_names) != n:
        return {"error": f"op_names len {len(op_names)} != predictions {n}"}

    PAD = 0
    EOS = 2
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"n_seq": 0, "n_seq_correct": 0,
                                                                "n_tok": 0, "n_tok_correct": 0})
    for i in range(n):
        op = op_names[i]
        # Build clean target (drop PAD) and matching pred slice.
        t = target_tokens[i].tolist()
        t_clean = [x for x in t if x != PAD]
        L = len(t_clean)
        p_clean = pred_tokens[i, :L].tolist()

        by_class[op]["n_seq"] += 1
        if t_clean == p_clean:
            by_class[op]["n_seq_correct"] += 1
        for t_id, p_id in zip(t_clean, p_clean):
            by_class[op]["n_tok"] += 1
            if t_id == p_id:
                by_class[op]["n_tok_correct"] += 1

    out = {}
    for op, v in sorted(by_class.items()):
        seq_acc = v["n_seq_correct"] / max(v["n_seq"], 1)
        tok_acc = v["n_tok_correct"] / max(v["n_tok"], 1)
        out[op] = {
            "n_test_samples": v["n_seq"],
            "sequence_accuracy": seq_acc,
            "token_accuracy": tok_acc,
            "n_total_tokens": v["n_tok"],
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", type=Path, default=REPO / "outputs" / "decoder20260511" / "checkpoints" / "per_row_5fold")
    p.add_argument("--src-root", type=Path, default=REPO / "outputs" / "decoder20260511" / "preprocessed_f98" / "per_row")
    p.add_argument("--output", type=Path, default=REPO / "outputs" / "decoder20260511" / "audit" / "per_op_class_v8.json")
    args = p.parse_args()

    per_fold = []
    aggregate: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"seq": [], "tok": [], "n": []})

    for F in [1, 2, 3, 4, 5]:
        pred_npz = args.sweep_root / f"fold_{F}" / "results" / "predictions.npz"
        src_npz = args.src_root / f"fold_{F}" / "test_sequences.npz"
        if not pred_npz.exists() or not src_npz.exists():
            continue
        breakdown = per_op_class(pred_npz, src_npz)
        per_fold.append({"fold": F, "breakdown": breakdown})
        for op, m in breakdown.items():
            if isinstance(m, dict):
                aggregate[op]["seq"].append(m["sequence_accuracy"])
                aggregate[op]["tok"].append(m["token_accuracy"])
                aggregate[op]["n"].append(m["n_test_samples"])

    agg = {}
    for op, vals in aggregate.items():
        agg[op] = {
            "sequence_accuracy_mean": float(np.mean(vals["seq"])) if vals["seq"] else float("nan"),
            "sequence_accuracy_std": float(np.std(vals["seq"])) if vals["seq"] else float("nan"),
            "token_accuracy_mean": float(np.mean(vals["tok"])) if vals["tok"] else float("nan"),
            "token_accuracy_std": float(np.std(vals["tok"])) if vals["tok"] else float("nan"),
            "n_test_samples_total": int(sum(vals["n"])),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"per_fold": per_fold, "aggregate": agg}, indent=2))

    print(f"\nPer-operation-class accuracy (mean ± std across 5 folds):")
    print(f"  {'class':<22} {'n':>6} {'seq_acc':>14} {'tok_acc':>14}")
    print("  " + "-" * 60)
    for op, v in sorted(agg.items()):
        print(f"  {op:<22} {v['n_test_samples_total']:>6} "
              f"{v['sequence_accuracy_mean']:.3f}±{v['sequence_accuracy_std']:.3f}   "
              f"{v['token_accuracy_mean']:.3f}±{v['token_accuracy_std']:.3f}")

    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
