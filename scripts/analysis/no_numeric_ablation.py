#!/usr/bin/env python3
"""No-numeric-prediction ablation -- Design A (evaluation-masking).

Action item 2 from the 2026-05-21 follow-up meeting (S. Eacuello / R. Prasad):
quantify how much of the autoregressive sequence-level collapse is attributable
to the numeric (coordinate-VALUE) tokens, by re-scoring the EXISTING decoder
predictions with the numeric token positions excluded from the metric.

Design A is the cheap variant: it does NOT retrain. It takes the
already-generated 5-fold predictions and partitions every token position into

  - numeric    : a coordinate VALUE token -- a ``NUM_*`` vocabulary entry, or a
                 fused literal carrying a decimal point (e.g. ``Z0.``, ``F22.``).
  - structural : everything else -- axis letters (X/Y/Z/I/J/R/F/S), G/M command
                 codes, control tokens. This is the G-code "skeleton".

It then reports token accuracy and per-window sequence exact-match -- overall,
structural-only, and numeric-only -- for both evaluation regimes:

  - TF : teacher-forced   (beam_0_all_predictions.json)
  - AR : autoregressive   (beam_1_all_predictions.json, greedy / beam width 1)

CAVEAT -- why Design A AR is a *contaminated lower bound*, not the clean answer:
under AR the decoder consumes its own previous token. A wrong numeric token
early in a window shifts the position alignment of every later token, so
masking the numeric POSITIONS at scoring time does not undo the damage the
numeric ERRORS already did to the structural tokens around them. Design A AR
therefore under-states structural recoverability. The desync-free reference is
Design A TF (teacher forcing pins every position to ground truth). The clean
experiment is Design B -- retrain with every coordinate value collapsed to a
single placeholder token, so a numeric position can never desync the decode.

Output: outputs/decoder20260511/audit/no_numeric_ablation_designA.json
"""
from __future__ import annotations

import json
import statistics as st
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CKPT = REPO / "outputs/decoder20260511/checkpoints/full_window_5fold"
OUT = REPO / "outputs/decoder20260511/audit/no_numeric_ablation_designA.json"

# run-hash subdir per fold (the non-_fsm, unconstrained decoder)
FOLD_HASH = {1: "6o90io5p", 2: "wf2ulnx7", 3: "ua4ht85b", 4: "4rrcx1qd", 5: "gmtf655s"}
N_BINS = 10  # relative-position bins for the desync drift profile


def is_numeric(tok: str) -> bool:
    """True for a coordinate-VALUE token: a NUM_* entry or a fused decimal literal."""
    return tok.startswith("NUM_") or ("." in tok)


def score_window(true_s: str, pred_s: str) -> dict:
    """Position-aligned scoring of one decoded window against ground truth."""
    T = true_s.split()
    P = pred_s.split()
    n_struct = n_struct_ok = n_num = n_num_ok = 0
    struct_seq_ok = True
    pos_bins = [[0, 0] for _ in range(N_BINS)]  # [ok, total] for structural tokens
    L = max(len(T), 1)
    for i, t in enumerate(T):
        p = P[i] if i < len(P) else None
        ok = int(p == t)
        if is_numeric(t):
            n_num += 1
            n_num_ok += ok
        else:
            n_struct += 1
            n_struct_ok += ok
            if not ok:
                struct_seq_ok = False
            b = min(int(N_BINS * i / L), N_BINS - 1)
            pos_bins[b][0] += ok
            pos_bins[b][1] += 1
    t_sub = [t for t in T if not is_numeric(t)]
    p_sub = [p for p in P if not is_numeric(p)]
    return {
        "n_tok": len(T),
        "n_struct": n_struct,
        "n_struct_ok": n_struct_ok,
        "n_num": n_num,
        "n_num_ok": n_num_ok,
        "full_token_acc": (n_struct_ok + n_num_ok) / max(len(T), 1),
        "struct_token_acc": (n_struct_ok / n_struct) if n_struct else None,
        "num_token_acc": (n_num_ok / n_num) if n_num else None,
        "full_seq_ok": int(T == P),
        "struct_seq_ok": int(struct_seq_ok),          # all structural positions match
        "struct_subseq_ok": int(t_sub == p_sub),      # structural skeleton matches as a subsequence
        "len_true": len(T),
        "len_pred": len(P),
        "pos_bins": pos_bins,
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def agg(per_fold_vals):
    """5-fold mean +/- sample std."""
    vals = [v for v in per_fold_vals if v is not None]
    return {
        "mean": mean(vals),
        "std": st.stdev(vals) if len(vals) > 1 else 0.0,
        "per_fold": per_fold_vals,
    }


def run_regime(beam_file: str) -> dict:
    """Score one evaluation regime across all 5 folds."""
    per_fold, vocab, dot_tokens = [], Counter(), Counter()
    pos_profile = [[0, 0] for _ in range(N_BINS)]  # pooled structural [ok, total]
    for fold, h in FOLD_HASH.items():
        path = CKPT / f"fold_{fold}" / h / "results" / beam_file
        rows = json.loads(path.read_text())
        ws = [score_window(r["true"], r["pred"]) for r in rows]
        for r in rows:
            for tok in (r["true"] + " " + r["pred"]).split():
                vocab[tok] += 1
                if "." in tok and not tok.startswith("NUM_"):
                    dot_tokens[tok] += 1
        for w in ws:
            for b in range(N_BINS):
                pos_profile[b][0] += w["pos_bins"][b][0]
                pos_profile[b][1] += w["pos_bins"][b][1]
        # Token accuracies are POOLED per fold (total correct / total tokens) to
        # match the paper's per-fold token_accuracy in beam_*_metrics.json; the
        # 5-fold headline is then the mean of these per-fold pooled values.
        # (Pooled den=true reproduces the official per-fold number to ~5e-4; the
        # residual is the trailing EOS column the metrics file scores and the
        # exported true/pred strings omit -- immaterial to the structural story.)
        tot_tok = sum(w["n_tok"] for w in ws)
        tot_struct = sum(w["n_struct"] for w in ws)
        tot_num = sum(w["n_num"] for w in ws)
        per_fold.append({
            "fold": fold,
            "n_windows": len(ws),
            "n_tokens": tot_tok,
            "full_token_acc": sum(w["n_struct_ok"] + w["n_num_ok"] for w in ws) / tot_tok,
            "struct_token_acc": (sum(w["n_struct_ok"] for w in ws) / tot_struct) if tot_struct else None,
            "num_token_acc": (sum(w["n_num_ok"] for w in ws) / tot_num) if tot_num else None,
            "full_seq_exact": mean([w["full_seq_ok"] for w in ws]),
            "struct_seq_exact": mean([w["struct_seq_ok"] for w in ws]),
            "struct_subseq_exact": mean([w["struct_subseq_ok"] for w in ws]),
            "len_mismatch_rate": mean([int(w["len_true"] != w["len_pred"]) for w in ws]),
        })
    keys = ["full_token_acc", "struct_token_acc", "num_token_acc",
            "full_seq_exact", "struct_seq_exact", "struct_subseq_exact",
            "len_mismatch_rate"]
    aggregate = {k: agg([f[k] for f in per_fold]) for k in keys}
    return {
        "source": beam_file,
        "per_fold": per_fold,
        "aggregate": aggregate,
        "structural_position_profile": [
            {"bin": b, "rel_pos": f"{b/N_BINS:.1f}-{(b+1)/N_BINS:.1f}",
             "struct_token_acc": (ok / tot) if tot else None, "n": tot}
            for b, (ok, tot) in enumerate(pos_profile)
        ],
        "dot_token_examples": dot_tokens.most_common(12),
        "n_distinct_tokens": len(vocab),
    }


def main() -> None:
    result = {
        "meta": {
            "design": "A (evaluation-masking; no retrain)",
            "purpose": "Action item 2, 2026-05-21 follow-up meeting -- isolate the "
                       "numeric-token contribution to the AR sequence collapse.",
            "regimes": {"TF": "beam_0 (teacher-forced)", "AR": "beam_1 (autoregressive)"},
            "folds": list(FOLD_HASH.keys()),
            "numeric_token_rule": "startswith('NUM_') or contains '.'",
            "caveat": "AR is a contaminated lower bound: numeric errors desync the "
                      "structural decode before the metric masks them. Design B "
                      "(retrain with numeric->placeholder) is the clean test.",
        },
        "TF": run_regime("beam_0_all_predictions.json"),
        "AR": run_regime("beam_1_all_predictions.json"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    # ---- sanity gates + console summary ------------------------------------
    tf_f1 = result["TF"]["per_fold"][0]["full_token_acc"]
    ar_f1 = result["AR"]["per_fold"][0]["full_token_acc"]
    print(f"SANITY  fold-1 full token acc  TF={tf_f1:.5f} (ar_aggregate beam_0=0.74663)")
    print(f"SANITY  fold-1 full token acc  AR={ar_f1:.5f} (ar_aggregate beam_1=0.20419)")
    print()
    hdr = f"{'metric':<26}{'TF (teacher-forced)':<26}{'AR (autoregressive)':<26}"
    print(hdr)
    print("-" * len(hdr))
    for k, label in [
        ("full_token_acc",      "token acc  (all)"),
        ("struct_token_acc",    "token acc  (structural)"),
        ("num_token_acc",       "token acc  (numeric)"),
        ("full_seq_exact",      "seq exact  (all)"),
        ("struct_seq_exact",    "seq exact  (structural)"),
        ("struct_subseq_exact", "seq exact  (struct subseq)"),
        ("len_mismatch_rate",   "len(pred)!=len(true)"),
    ]:
        tf = result["TF"]["aggregate"][k]
        ar = result["AR"]["aggregate"][k]
        print(f"{label:<26}"
              f"{tf['mean']:.4f} +/- {tf['std']:.4f}      "
              f"{ar['mean']:.4f} +/- {ar['std']:.4f}")
    print()
    print("structural token acc by relative position (desync drift):")
    print(f"  {'rel-pos':<12}{'TF':<10}{'AR':<10}")
    for tfb, arb in zip(result["TF"]["structural_position_profile"],
                        result["AR"]["structural_position_profile"]):
        tv = f"{tfb['struct_token_acc']:.3f}" if tfb["struct_token_acc"] is not None else "  -  "
        av = f"{arb['struct_token_acc']:.3f}" if arb["struct_token_acc"] is not None else "  -  "
        print(f"  {tfb['rel_pos']:<12}{tv:<10}{av:<10}")
    print()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
