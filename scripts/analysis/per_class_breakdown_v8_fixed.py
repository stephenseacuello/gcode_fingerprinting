#!/usr/bin/env python3
"""Phase-C (FIXED): per-(head, operation-class) accuracy breakdown for V8.

ROOT-CAUSE FIX (TASK B2)
========================
The original ``per_class_breakdown_v8.py`` joined a *full-window* predictions
file (one row per WINDOW, e.g. fold-1 has N=132 windows, each a 1400-token
stream) against a *per-row* source NPZ (one row per G-code LINE, e.g. fold-1
has N=7937 lines). ``operation_type_names`` therefore had 7937 entries while
predictions had 132, and the script bailed with
``op_names len 7937 != predictions 132`` on every fold. This is the same
data-alignment bug class flagged in feedback_data_alignment.md (per-row vs
full-window indexing mismatch / mixing two preprocessing pipelines).

The fix is to align the *full-window* predictions with the *full-window*
source NPZ (``preprocessed_f98/full_window/fold_F/test_sequences.npz``), which
also has one row per window and a per-window ``operation_type_names`` that maps
1:1 onto the predictions. We verify the 1:1 join by checking that the source
``tokens`` row matches the prediction ``target_tokens`` row (modulo the trailing
EOS that predictions append).

We then compute *per-(head, op-class)* teacher-forcing accuracy directly from
the multi-head fields already stored in predictions.npz, using the SAME masking
convention as compute_per_head_per_opclass_entropy.py so the accuracies are
directly comparable to the entropy matrix:

  head        valid-position mask (from the *target* head field)
  ----------  -----------------------------------------------------------
  token       non-PAD positions (type_t != -1)
  type        non-PAD positions (type_t != -1)
  command     positions where type=COMMAND (cmd_t  != -1)
  param_type  positions where type in {PARAMETER, NUMERIC} (pt_t != -1)
  sign        positions where type=NUMERIC (type_t == 3 ; sign_t in {0,1})
  digit       positions where type=NUMERIC, pooled over the 6 digit slots
              (type_t == 3 and per-slot digit_t != 10 digit-pad)

Sentinel conventions discovered in predictions.npz:
  type_t == -1  -> PAD position
  type_t == 0   -> SPECIAL (BOS/EOS)
  type_t == 1   -> COMMAND
  type_t == 2   -> PARAMETER
  type_t == 3   -> NUMERIC
  sign_t == 2   -> "not numeric" sentinel (sign only defined where type==3)
  digit_t == 10 -> digit-pad sentinel

Accuracy is pooled over all valid positions of every window in an op-class
(position-level accuracy), then aggregated mean +/- std across folds.

Outputs audit/per_op_class_v8_fixed.json with structure:
  {
    "per_fold":  [{"fold": F, "breakdown": {op_class: {head: acc, ...,
                    "n_windows": k}}}, ...],
    "aggregate": {op_class: {head: {"mean": .., "std": .., "n_positions": ..},
                             ..., "n_windows_total": k}},
    "aggregate_head_opclass": {head: {op_class: mean_acc}}  # plotting-friendly
  }
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]

PAD_TOKEN = 0
EOS_TOKEN = 2

# type codes inside predictions.npz (type_t / type_p)
TYPE_PAD = -1
TYPE_SPECIAL = 0
TYPE_COMMAND = 1
TYPE_PARAMETER = 2
TYPE_NUMERIC = 3

SIGN_SENTINEL = 2     # sign_t value at non-numeric positions
DIGIT_PAD = 10        # digit_t value at non-numeric / unused digit slots

HEADS = ["token", "type", "command", "param_type", "sign", "digit"]


def _verify_alignment(src_tokens: np.ndarray, pred_target: np.ndarray,
                      n_check: int = 8) -> dict[str, Any]:
    """Sanity-check that source tokens and prediction targets are the same
    windows in the same order. Returns a small diagnostic dict."""
    n = min(src_tokens.shape[0], pred_target.shape[0])
    matches = 0
    checked = 0
    idxs = np.linspace(0, n - 1, min(n_check, n)).astype(int)
    for i in idxs:
        s = [x for x in src_tokens[i].tolist() if x != PAD_TOKEN]
        p = [x for x in pred_target[i].tolist() if x != PAD_TOKEN]
        # predictions append a trailing EOS that the raw `tokens` field omits;
        # strip a single trailing EOS from the prediction side before compare.
        if p and p[-1] == EOS_TOKEN and (not s or s[-1] != EOS_TOKEN):
            p = p[:-1]
        checked += 1
        if s == p:
            matches += 1
    return {"checked": checked, "matched": matches}


def per_op_class(predictions_npz: Path, src_npz: Path) -> dict[str, Any]:
    d = np.load(predictions_npz, allow_pickle=True)
    src = np.load(src_npz, allow_pickle=True)

    op_names = np.array([str(s) for s in src["operation_type_names"]])
    pred_tokens = d["pred_tokens"]      # [N, T]
    target_tokens = d["target_tokens"]  # [N, T]
    n = pred_tokens.shape[0]

    if len(op_names) != n:
        return {"error": f"op_names len {len(op_names)} != predictions {n} "
                          f"(source NPZ is the wrong granularity: use the "
                          f"full_window test_sequences.npz, not per_row)"}

    # Verify 1:1 window alignment (target tokens vs source tokens).
    align = _verify_alignment(src["tokens"], target_tokens)

    # Multi-head fields.
    type_t = d["type_t"]; type_p = d["type_p"]
    cmd_t = d["cmd_t"];   cmd_p = d["cmd_p"]
    pt_t = d["pt_t"];     pt_p = d["pt_p"]
    sign_t = d["sign_t"]; sign_p = d["sign_p"]
    digit_t = d["digit_t"]; digit_p = d["digit_p"]   # [N, T, 6]

    # Accumulators per op-class -> per head -> (correct, total).
    acc: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {h: [0, 0] for h in HEADS})
    n_windows: dict[str, int] = defaultdict(int)

    for i in range(n):
        op = op_names[i]
        n_windows[op] += 1

        tt = target_tokens[i]
        nonpad = (type_t[i] != TYPE_PAD)          # non-PAD positions

        # token head: full-vocab token id on non-PAD positions.
        c, t = _count(pred_tokens[i][nonpad], tt[nonpad])
        acc[op]["token"][0] += c; acc[op]["token"][1] += t

        # type head: non-PAD positions.
        c, t = _count(type_p[i][nonpad], type_t[i][nonpad])
        acc[op]["type"][0] += c; acc[op]["type"][1] += t

        # command head: positions where type=COMMAND.
        m = (cmd_t[i] != -1)
        c, t = _count(cmd_p[i][m], cmd_t[i][m])
        acc[op]["command"][0] += c; acc[op]["command"][1] += t

        # param_type head: positions where type in {PARAMETER, NUMERIC}.
        m = (pt_t[i] != -1)
        c, t = _count(pt_p[i][m], pt_t[i][m])
        acc[op]["param_type"][0] += c; acc[op]["param_type"][1] += t

        # sign head: positions where type=NUMERIC.
        m = (type_t[i] == TYPE_NUMERIC)
        c, t = _count(sign_p[i][m], sign_t[i][m])
        acc[op]["sign"][0] += c; acc[op]["sign"][1] += t

        # digit head: positions where type=NUMERIC, pooled over 6 slots,
        # excluding the digit-pad sentinel (matches entropy convention).
        m = (type_t[i] == TYPE_NUMERIC)
        dt = digit_t[i][m].reshape(-1)   # [k*6]
        dp = digit_p[i][m].reshape(-1)
        valid = (dt != DIGIT_PAD)
        c, t = _count(dp[valid], dt[valid])
        acc[op]["digit"][0] += c; acc[op]["digit"][1] += t

    out: dict[str, Any] = {"_alignment": align}
    for op in sorted(acc):
        rec: dict[str, Any] = {"n_windows": n_windows[op]}
        for h in HEADS:
            correct, total = acc[op][h]
            rec[h] = {
                "accuracy": (correct / total) if total else float("nan"),
                "n_positions": int(total),
            }
        out[op] = rec
    return out


def _count(pred: np.ndarray, target: np.ndarray) -> tuple[int, int]:
    total = int(target.size)
    if total == 0:
        return 0, 0
    correct = int((pred == target).sum())
    return correct, total


def _resolve_pred(sweep_root: Path, fold: int) -> Path | None:
    """Find fold_F predictions.npz, allowing an optional nested run-id dir
    (e.g. fold_1/<runid>/results/predictions.npz as used by the canonical
    full_window_5fold checkpoint)."""
    direct = sweep_root / f"fold_{fold}" / "results" / "predictions.npz"
    if direct.exists():
        return direct
    nested = sorted((sweep_root / f"fold_{fold}").glob("*/results/predictions.npz"))
    return nested[0] if nested else None


def main() -> int:
    p = argparse.ArgumentParser()
    # Canonical headline full-window checkpoint (sign~0.99, digit bathtub
    # [1.0, .89, .70, .50, .41, .74]) matching Table 1 headline_5fold.tex.
    # NB: full_window_5fold_designB_K2418 has a DEGENERATE sign/digit head
    # (sign~0.30, digit~0.08) and must NOT be used for this analysis.
    p.add_argument("--sweep-root", type=Path,
                   default=REPO / "outputs/decoder20260511/checkpoints/full_window_5fold")
    p.add_argument("--src-root", type=Path,
                   default=REPO / "outputs/decoder20260511/preprocessed_f98/full_window")
    p.add_argument("--output", type=Path,
                   default=REPO / "outputs/decoder20260511/audit/per_op_class_v8_fixed.json")
    args = p.parse_args()

    per_fold = []
    # aggregate[op][head] -> list of (correct, total) per fold for pooling
    agg_pos: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list))
    agg_fold_acc: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    agg_nwin: dict[str, int] = defaultdict(int)

    for F in [1, 2, 3, 4, 5]:
        pred_npz = _resolve_pred(args.sweep_root, F)
        src_npz = args.src_root / f"fold_{F}" / "test_sequences.npz"
        if pred_npz is None or not src_npz.exists():
            print(f"fold {F}: MISSING (pred={pred_npz}, src_exists={src_npz.exists()})")
            continue
        breakdown = per_op_class(pred_npz, src_npz)
        if "error" in breakdown:
            print(f"fold {F}: ERROR {breakdown['error']}")
            per_fold.append({"fold": F, "breakdown": breakdown})
            continue
        align = breakdown.get("_alignment", {})
        print(f"fold {F}: alignment {align['matched']}/{align['checked']} windows matched")
        per_fold.append({"fold": F, "breakdown": breakdown})
        for op, rec in breakdown.items():
            if op.startswith("_"):
                continue
            agg_nwin[op] += rec["n_windows"]
            for h in HEADS:
                a = rec[h]["accuracy"]
                npos = rec[h]["n_positions"]
                if npos > 0:
                    agg_fold_acc[op][h].append(a)
                    # recover correct count for pooled accuracy
                    agg_pos[op][h].append((int(round(a * npos)), npos))

    aggregate: dict[str, Any] = {}
    plot_friendly: dict[str, dict[str, float]] = {h: {} for h in HEADS}
    for op in sorted(agg_fold_acc):
        rec: dict[str, Any] = {"n_windows_total": agg_nwin[op]}
        for h in HEADS:
            accs = agg_fold_acc[op][h]
            if accs:
                pooled_c = sum(c for c, _ in agg_pos[op][h])
                pooled_t = sum(t for _, t in agg_pos[op][h])
                rec[h] = {
                    "mean": float(np.mean(accs)),           # mean of per-fold accs
                    "std": float(np.std(accs)),
                    "pooled": float(pooled_c / pooled_t) if pooled_t else float("nan"),
                    "n_positions": int(pooled_t),
                    "n_folds": len(accs),
                }
                # use pooled accuracy as the plotting value (matches entropy
                # which is computed on the pooled population).
                plot_friendly[h][op] = rec[h]["pooled"]
            else:
                rec[h] = {"mean": float("nan"), "std": float("nan"),
                          "pooled": float("nan"), "n_positions": 0, "n_folds": 0}
        aggregate[op] = rec

    out = {
        "per_fold": per_fold,
        "aggregate": aggregate,
        "aggregate_head_opclass": plot_friendly,
        "meta": {
            "sweep_root": str(args.sweep_root.relative_to(REPO)),
            "src_root": str(args.src_root.relative_to(REPO)),
            "heads": HEADS,
            "accuracy_type": "teacher-forcing, position-level, per (head, op-class)",
            "masking": "matches compute_per_head_per_opclass_entropy.py conventions",
            "note": ("plotting values in aggregate_head_opclass are the POOLED "
                     "across-fold accuracy (same population basis as entropy)"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    # Pretty print.
    print(f"\nPer-(head, op-class) TF accuracy (pooled across folds):")
    hdr = f"  {'op_class':<16}{'n_win':>6}  " + "".join(f"{h[:7]:>9}" for h in HEADS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for op in sorted(aggregate):
        row = f"  {op:<16}{aggregate[op]['n_windows_total']:>6}  "
        for h in HEADS:
            v = aggregate[op][h]["pooled"]
            row += f"{v:>9.3f}" if v == v else f"{'nan':>9}"
        print(row)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
