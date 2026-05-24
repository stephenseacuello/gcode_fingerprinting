#!/usr/bin/env python3
"""Design B vs current model -- structural recoverability comparison (Step 5).

Action item 2, no-numeric ablation. Scores the retrained "no-numbers" decoder
(checkpoints/full_window_5fold_nonum/) with the SAME methodology
no_numeric_ablation.py used for Design A, and prints the side-by-side:

  current model  (numbers in vocab)  <- audit/no_numeric_ablation_designA.json
  Design B       (numbers -> <NUM>)  <- scored here, from its beam predictions

Design B's single numeric placeholder is <NUM>; structural = everything else
(the G-code skeleton -- command codes and axis letters). The headline question:
once the decoder no longer has to predict coordinate values, does the
structural skeleton come back under autoregressive decoding?

Token accuracies are pooled per fold (total correct / total tokens), then
averaged over the 5 folds; sequence exact-match is per-window, averaged.
Output: audit/design_b_compare.json
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO = Path("/home/seacuello/Documents/gcode_fingerprinting")
DESIGN_B = REPO / "outputs/decoder20260511/checkpoints/full_window_5fold_nonum"
DESIGN_A_JSON = REPO / "outputs/decoder20260511/audit/no_numeric_ablation_designA.json"
OUT = REPO / "outputs/decoder20260511/audit/design_b_compare.json"
FOLDS = [1, 2, 3, 4, 5]


def is_numeric(tok: str) -> bool:
    """Design B placeholder <NUM>, plus the current model's NUM_*/dotted literals."""
    return tok == "<NUM>" or tok.startswith("NUM_") or ("." in tok)


def score_window(true_s: str, pred_s: str) -> dict:
    T, P = true_s.split(), pred_s.split()
    n_struct = n_struct_ok = n_num = n_num_ok = 0
    struct_seq_ok = True
    for i, t in enumerate(T):
        ok = int((P[i] if i < len(P) else None) == t)
        if is_numeric(t):
            n_num += 1; n_num_ok += ok
        else:
            n_struct += 1; n_struct_ok += ok
            if not ok:
                struct_seq_ok = False
    return dict(n_tok=len(T), n_struct=n_struct, n_struct_ok=n_struct_ok,
                n_num=n_num, n_num_ok=n_num_ok, full_ok=int(T == P),
                struct_seq_ok=int(struct_seq_ok),
                len_mismatch=int(len(T) != len(P)))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def agg(vals):
    vs = [v for v in vals if v is not None]
    return {"mean": mean(vs), "std": st.stdev(vs) if len(vs) > 1 else 0.0,
            "per_fold": vals}


def score_regime(beam_file: str) -> dict:
    per_fold = []
    for f in FOLDS:
        rows = json.loads((DESIGN_B / f"fold_{f}/results/{beam_file}").read_text())
        ws = [score_window(r["true"], r["pred"]) for r in rows]
        tot_tok = sum(w["n_tok"] for w in ws)
        tot_struct = sum(w["n_struct"] for w in ws)
        tot_num = sum(w["n_num"] for w in ws)
        per_fold.append(dict(
            fold=f, n_windows=len(ws), n_tokens=tot_tok,
            full_token_acc=sum(w["n_struct_ok"] + w["n_num_ok"] for w in ws) / tot_tok,
            struct_token_acc=(sum(w["n_struct_ok"] for w in ws) / tot_struct) if tot_struct else None,
            num_token_acc=(sum(w["n_num_ok"] for w in ws) / tot_num) if tot_num else None,
            full_seq_exact=mean([w["full_ok"] for w in ws]),
            struct_seq_exact=mean([w["struct_seq_ok"] for w in ws]),
            len_mismatch_rate=mean([w["len_mismatch"] for w in ws]),
        ))
    keys = ["full_token_acc", "struct_token_acc", "num_token_acc",
            "full_seq_exact", "struct_seq_exact", "len_mismatch_rate"]
    return {"source": beam_file, "per_fold": per_fold,
            "aggregate": {k: agg([pf[k] for pf in per_fold]) for k in keys}}


def main() -> None:
    dB = {"TF": score_regime("beam_0_all_predictions.json"),
          "AR": score_regime("beam_1_all_predictions.json")}
    dA = json.loads(DESIGN_A_JSON.read_text())  # current model, scored in Design A

    result = {
        "meta": {"design": "B (retrained, every coordinate value -> <NUM>)",
                 "current_model_source": str(DESIGN_A_JSON.name)},
        "design_b": dB,
        "current_model": {"TF": dA["TF"]["aggregate"], "AR": dA["AR"]["aggregate"]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    def B(reg, k):  # Design B aggregate
        return dB[reg]["aggregate"][k]["mean"]

    def C(reg, k):  # current model aggregate (from Design A JSON)
        return dA[reg]["aggregate"][k]["mean"]

    print("=" * 78)
    print("Design B (no-numbers retrain) vs current model -- structural recoverability")
    print("=" * 78)
    hdr = f"{'metric':<28}{'current TF':>12}{'current AR':>12}{'DesignB TF':>13}{'DesignB AR':>13}"
    print(hdr)
    print("-" * len(hdr))
    rows = [
        ("token acc  (all)",        "full_token_acc"),
        ("token acc  (structural)", "struct_token_acc"),
        ("seq exact  (all)",        "full_seq_exact"),
        ("seq exact  (structural)", "struct_seq_exact"),
        ("len(pred)!=len(true)",    "len_mismatch_rate"),
    ]
    for label, k in rows:
        print(f"{label:<28}{C('TF',k):>12.4f}{C('AR',k):>12.4f}"
              f"{B('TF',k):>13.4f}{B('AR',k):>13.4f}")
    print()
    print(f"Design B <NUM>-placeholder token acc:  TF {B('TF','num_token_acc'):.4f}"
          f"   AR {B('AR','num_token_acc'):.4f}   (should be ~1.0 -- trivial)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
