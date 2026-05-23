#!/usr/bin/env python3
"""Step 8: Final no-numeric ablation comparison.

Design B SS=0 (no-numbers retrain) vs the current model, like-for-like, using
the paper-grade position-aligned token-accuracy and sequence-exact-match
metrics with the +1 EOS-slot offset corrected.

The offset, root-caused in run_decoder_quick_test.py:1287-1296 +
gcode_tokenizer.py:decode(): tokenizer.decode() strips a trailing EOS;
true_ids always ends in EOS so true_toks is N-1; pred_ids ends in EOS only
when the model correctly predicts EOS at the EOS slot. The current model
predicts EOS at the end ~94% of windows under TF, so lengths usually match
natively; Design B's placeholder vocabulary never has the model predict EOS,
so every pred is N-long and naive position-aligned scoring desyncs by one.

Correction (universal): trim pred to len(true) before scoring. For the
current model this is a no-op when EOS was already stripped; for Design B it
drops the model's meaningless EOS-slot prediction.

Token accuracies are POOLED per fold (total correct / total tokens of that
class), then 5-fold mean +/- std. Sequence-exact-match is per-window 0/1,
averaged per fold.

Output: outputs/decoder20260511/audit/design_b_compare_ss0.json
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO = Path("/home/seacuello/Documents/gcode_fingerprinting")
CK = REPO / "outputs/decoder20260511/checkpoints"
OUT = REPO / "outputs/decoder20260511/audit/design_b_compare_ss0.json"

# run-hash subdir per fold for the current model (the un-wandb-suffixed runs)
CURRENT_HASH = {1: "6o90io5p", 2: "wf2ulnx7", 3: "ua4ht85b", 4: "4rrcx1qd", 5: "gmtf655s"}


def is_numeric(tok: str) -> bool:
    """Design B placeholder <NUM>, plus the current model's NUM_*/dotted literals."""
    return tok == "<NUM>" or tok.startswith("NUM_") or ("." in tok)


def score_window(true_s: str, pred_s: str) -> dict:
    """Position-aligned scoring with the +1 EOS-slot offset corrected."""
    T = true_s.split()
    P = pred_s.split()[: len(T)]                              # offset correction
    n_struct = n_struct_ok = n_num = n_num_ok = 0
    struct_seq_ok = True
    for i, t in enumerate(T):
        p = P[i] if i < len(P) else None
        ok = int(p == t)
        if is_numeric(t):
            n_num += 1; n_num_ok += ok
        else:
            n_struct += 1; n_struct_ok += ok
            if not ok:
                struct_seq_ok = False
    return dict(n_tok=len(T), n_struct=n_struct, n_struct_ok=n_struct_ok,
                n_num=n_num, n_num_ok=n_num_ok,
                full_ok=int(T == P), struct_seq_ok=int(struct_seq_ok))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def agg(vals):
    vs = [v for v in vals if v is not None]
    return {"mean": mean(vs),
            "std": st.stdev(vs) if len(vs) > 1 else 0.0,
            "per_fold": vals,
            "n_folds": len(vs)}


def score_variant(label: str, fold_paths: dict, beam_file: str) -> dict:
    per_fold = []
    for f, dir_ in fold_paths.items():
        path = dir_ / beam_file
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        ws = [score_window(r["true"], r["pred"]) for r in rows]
        tot_tok = sum(w["n_tok"] for w in ws)
        tot_struct = sum(w["n_struct"] for w in ws)
        tot_num = sum(w["n_num"] for w in ws)
        per_fold.append(dict(
            fold=f, n_windows=len(ws),
            full_token_acc=sum(w["n_struct_ok"] + w["n_num_ok"] for w in ws) / max(tot_tok, 1),
            struct_token_acc=(sum(w["n_struct_ok"] for w in ws) / tot_struct) if tot_struct else None,
            num_token_acc=(sum(w["n_num_ok"] for w in ws) / tot_num) if tot_num else None,
            full_seq_exact=mean([w["full_ok"] for w in ws]),
            struct_seq_exact=mean([w["struct_seq_ok"] for w in ws]),
        ))
    keys = ["full_token_acc", "struct_token_acc", "num_token_acc",
            "full_seq_exact", "struct_seq_exact"]
    return {"label": label,
            "folds_present": [pf["fold"] for pf in per_fold],
            "per_fold": per_fold,
            "aggregate": {k: agg([pf[k] for pf in per_fold]) for k in keys}}


def main() -> None:
    current_paths = {f: CK / f"full_window_5fold/fold_{f}/{h}/results"
                     for f, h in CURRENT_HASH.items()}
    ss0_paths = {f: CK / f"full_window_5fold_nonum_ss0/fold_{f}/results"
                 for f in range(1, 6)}

    results = {
        "meta": {
            "step": "8 (final n=5 comparison)",
            "offset_correction": "pred[:len(true)] -- drops the EOS-slot prediction",
            "scoring": "pooled token acc per fold, then 5-fold mean +/- std; "
                       "sequence exact per-window 0/1, mean per fold",
            "current_source": "checkpoints/full_window_5fold/<fold>/<hash>/results/",
            "ss0_source": "checkpoints/full_window_5fold_nonum_ss0/<fold>/results/",
        },
        "current_TF": score_variant("current model TF", current_paths,
                                    "beam_0_all_predictions.json"),
        "current_AR": score_variant("current model AR", current_paths,
                                    "beam_1_all_predictions.json"),
        "ss0_TF": score_variant("Design B SS=0 TF", ss0_paths,
                                "beam_0_all_predictions.json"),
        "ss0_AR": score_variant("Design B SS=0 AR", ss0_paths,
                                "beam_1_all_predictions.json"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    # ---- console summary ----------------------------------------------------
    print("=" * 96)
    print("Step 8: Design B SS=0 (no-numbers retrain) vs current model -- structural recoverability")
    print("=" * 96)
    print(f"folds present: current = {sorted(results['current_TF']['folds_present'])}, "
          f"Design B SS=0 = {sorted(results['ss0_TF']['folds_present'])}")
    print()

    def cell(reg_key, metric_key):
        agg_ = results[reg_key]["aggregate"][metric_key]
        m, s, n = agg_["mean"], agg_["std"], agg_["n_folds"]
        return f"{m:.4f}+/-{s:.4f}(n={n})" if m is not None else "--"

    hdr = (f"{'metric':<28}{'current TF':>21}{'current AR':>21}"
           f"{'DesignB SS=0 TF':>21}{'DesignB SS=0 AR':>21}")
    print(hdr); print("-" * len(hdr))
    for label, k in [
        ("token acc  (all)",         "full_token_acc"),
        ("token acc  (structural)",  "struct_token_acc"),
        ("token acc  (numeric/<NUM>)", "num_token_acc"),
        ("seq exact  (full)",        "full_seq_exact"),
        ("seq exact  (structural)",  "struct_seq_exact"),
    ]:
        print(f"{label:<28}"
              f"{cell('current_TF', k):>21}"
              f"{cell('current_AR', k):>21}"
              f"{cell('ss0_TF', k):>21}"
              f"{cell('ss0_AR', k):>21}")
    print()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
