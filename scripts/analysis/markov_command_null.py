#!/usr/bin/env python3
"""History-matched sensor-free command null for the class-prior confound control.

The decoder's 0.795 all-command-position accuracy (audit/confound_class_prior.json)
is measured under teacher forcing, i.e. with ground-truth within-window token
history. The class-modal nulls (0.572 train-derived / 0.576 test-derived) are
history-free. This script computes a sensor-free null granted the same
ground-truth history: a per-operation-class first-order Markov command model
fitted on each fold's training split and scored at every command position of the
full-window test windows (ground-truth previous command as context, matching TF
conditioning).

Output: audit/markov_command_null.json
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "outputs/decoder20260511/preprocessed_f98/full_window"
OUT = REPO / "outputs/decoder20260511/audit/markov_command_null.json"

CMD_RE = re.compile(r"\b(G0|G1|G2|G3|G53|M30)\b")


def commands(text: str) -> list[str]:
    out = []
    for line in str(text).split("\n"):
        m = CMD_RE.search(line)
        if m:
            out.append(m.group(1))
    return out


def main() -> int:
    per_fold = {}
    for fold in range(1, 6):
        tr = np.load(DATA / f"fold_{fold}/train_sequences.npz", allow_pickle=True)
        te = np.load(DATA / f"fold_{fold}/test_sequences.npz", allow_pickle=True)

        trans: dict = defaultdict(Counter)
        start: dict = defaultdict(Counter)
        for txt, op in zip(tr["gcode_texts"], tr["operation_type"]):
            cs = commands(txt)
            if not cs:
                continue
            start[op][cs[0]] += 1
            for a, b in zip(cs, cs[1:]):
                trans[(op, a)][b] += 1

        corr = tot = 0
        for txt, op in zip(te["gcode_texts"], te["operation_type"]):
            cs = commands(txt)
            prev = None
            for c in cs:
                if prev is None:
                    pred = start[op].most_common(1)[0][0] if start[op] else "G1"
                else:
                    t = trans.get((op, prev))
                    pred = t.most_common(1)[0][0] if t else "G1"
                corr += pred == c
                tot += 1
                prev = c  # ground-truth history, matching TF conditioning
        per_fold[str(fold)] = {"accuracy": corr / tot, "n_command_positions": tot}

    accs = [v["accuracy"] for v in per_fold.values()]
    out = {
        "description": (
            "Per-operation-class first-order Markov command null with ground-truth "
            "history, scored at all command positions of the full-window test "
            "windows. The history-matched sensor-free counterpart of the class-"
            "modal nulls in audit/confound_class_prior.json."
        ),
        "generator": "scripts/analysis/markov_command_null.py",
        "per_fold": per_fold,
        "mean": float(np.mean(accs)),
        "std_population": float(np.std(accs)),
        "reference": {
            "decoder_tf_command_positions": 0.795,
            "class_modal_train_derived": 0.572,
            "class_modal_test_derived": 0.576,
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"mean {out['mean']:.4f} +/- {out['std_population']:.4f} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
