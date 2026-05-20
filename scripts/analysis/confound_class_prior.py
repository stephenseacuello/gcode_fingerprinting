#!/usr/bin/env python3
"""Frozen-encoder confound probe (referee R1 M6 / R2 M1 / R3 M8).

The decoder's command head rides a frozen encoder that was trained for
nine-class *operation* classification on the *same* fold partitions. The
concern is that the headline command accuracy (full_window, ~0.888) is in
large part operation-class structure re-expressed at the command level,
rather than per-instruction sensor recovery.

This script quantifies that, using only existing artifacts:

  1. Decoder command accuracy at COMMAND-token positions (full_window,
     teacher-forced beam_0 predictions) -- reproduces the headline.
  2. Class-prior baseline: predict every command position as the modal
     command of that window's operation class (modal computed on the
     fold's TRAIN split). If this is close to (1), the command head is
     largely a re-expression of operation class.
  3. Within-window modal baseline: predict every command position as the
     modal command among that window's own command tokens (the per-window
     "stereotype" the per-row formulation is bounded by).
  4. Heterogeneity-conditioned decoder accuracy: decoder command accuracy
     restricted to windows whose command tokens are NOT all identical
     (genuine within-window command diversity) vs. homogeneous windows.

Outputs audit/confound_class_prior.json with 5-fold mean +/- std.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

CMD_RE = re.compile(r"^[GM]\d+$")  # G0 G1 G2 G3 G53 M3 M5 M30 ...

FOLD_PRED = {
    1: "fold_1/6o90io5p",
    2: "fold_2/wf2ulnx7",
    3: "fold_3/ua4ht85b",
    4: "fold_4/4rrcx1qd",
    5: "fold_5/gmtf655s",
}


def modal_command_per_class(train_npz: Path) -> dict[str, str]:
    """Modal command token per operation class from raw train G-code text."""
    d = np.load(train_npz, allow_pickle=True)
    texts = d["gcode_texts"]
    ops = [str(s) for s in d["operation_type_names"]]
    by_class: dict[str, Counter] = {}
    for txt, op in zip(texts, ops):
        c = by_class.setdefault(op, Counter())
        for line in str(txt).split("\n"):
            line = line.strip()
            if not line:
                continue
            tok = line.split()[0]
            if CMD_RE.match(tok):
                c[tok] += 1
    return {
        op: (cnt.most_common(1)[0][0] if cnt else None) for op, cnt in by_class.items()
    }


def fold_report(pred_json: Path, test_npz: Path, modal_by_class: dict[str, str]) -> dict:
    recs = json.loads(pred_json.read_text())
    src = np.load(test_npz, allow_pickle=True)
    ops = [str(s) for s in src["operation_type_names"]]
    if len(ops) != len(recs):
        return {"error": f"op len {len(ops)} != recs {len(recs)}"}

    dec_c = dec_n = 0          # decoder accuracy at command positions
    cp_c = cp_n = 0            # class-prior baseline
    wm_c = wm_n = 0            # within-window modal baseline
    het_c = het_n = 0          # decoder acc, heterogeneous windows
    hom_c = hom_n = 0          # decoder acc, homogeneous windows
    n_het_windows = n_hom_windows = 0

    for rec, op in zip(recs, ops):
        true_t = str(rec["true"]).split()
        pred_t = str(rec["pred"]).split()
        cmd_pos = [j for j, t in enumerate(true_t) if CMD_RE.match(t)]
        if not cmd_pos:
            continue
        win_cmds = [true_t[j] for j in cmd_pos]
        win_modal = Counter(win_cmds).most_common(1)[0][0]
        distinct = len(set(win_cmds))
        is_het = distinct > 1
        n_het_windows += int(is_het)
        n_hom_windows += int(not is_het)
        cp_pred = modal_by_class.get(op)

        for j in cmd_pos:
            gold = true_t[j]
            pj = pred_t[j] if j < len(pred_t) else None
            dec_ok = pj == gold
            dec_n += 1
            dec_c += int(dec_ok)
            cp_n += 1
            cp_c += int(cp_pred == gold)
            wm_n += 1
            wm_c += int(win_modal == gold)
            if is_het:
                het_n += 1
                het_c += int(dec_ok)
            else:
                hom_n += 1
                hom_c += int(dec_ok)

    def acc(c, n):
        return c / n if n else float("nan")

    return {
        "n_command_positions": dec_n,
        "decoder_command_acc": acc(dec_c, dec_n),
        "class_prior_command_acc": acc(cp_c, cp_n),
        "within_window_modal_acc": acc(wm_c, wm_n),
        "decoder_acc_heterogeneous_windows": acc(het_c, het_n),
        "decoder_acc_homogeneous_windows": acc(hom_c, hom_n),
        "n_heterogeneous_windows": n_het_windows,
        "n_homogeneous_windows": n_hom_windows,
        "frac_heterogeneous_windows": n_het_windows
        / max(n_het_windows + n_hom_windows, 1),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweep-root",
        type=Path,
        default=REPO / "outputs/decoder20260511/checkpoints/full_window_5fold",
    )
    p.add_argument(
        "--src-root",
        type=Path,
        default=REPO / "outputs/decoder20260511/preprocessed_f98/full_window",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO / "outputs/decoder20260511/audit/confound_class_prior.json",
    )
    args = p.parse_args()

    per_fold = {}
    for fold, sub in FOLD_PRED.items():
        pred_json = args.sweep_root / sub / "results" / "beam_0_all_predictions.json"
        train_npz = args.src_root / f"fold_{fold}" / "train_sequences.npz"
        test_npz = args.src_root / f"fold_{fold}" / "test_sequences.npz"
        if not (pred_json.exists() and train_npz.exists() and test_npz.exists()):
            print(f"  skip fold {fold}: missing artifact")
            continue
        modal = modal_command_per_class(train_npz)
        per_fold[fold] = fold_report(pred_json, test_npz, modal)
        per_fold[fold]["modal_command_by_class"] = modal

    keys = [
        "decoder_command_acc",
        "class_prior_command_acc",
        "within_window_modal_acc",
        "decoder_acc_heterogeneous_windows",
        "decoder_acc_homogeneous_windows",
        "frac_heterogeneous_windows",
    ]
    agg = {}
    for k in keys:
        vals = [
            per_fold[f][k]
            for f in per_fold
            if isinstance(per_fold[f].get(k), (int, float))
            and not np.isnan(per_fold[f][k])
        ]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                      "n_folds": len(vals)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"per_fold": per_fold, "aggregate": agg}, indent=2)
    )

    print("\n=== Frozen-encoder confound probe (full_window, 5-fold) ===\n")
    for k in keys:
        if k in agg:
            print(f"{k:<38} {agg[k]['mean']:.4f} +/- {agg[k]['std']:.4f}  (n={agg[k]['n_folds']})")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
