#!/usr/bin/env python3
"""K-spectrum: how does AR structural recovery vary with numeric vocab size?

Four variants along the spectrum K = 2418 -> 335 -> 69 -> 24:
  - current : K=2418, 4-digit bucketing  (headline config: SS=0.5, dw=1.0)
  - b2      : K=335,  2-digit bucketing  (SS=0, dw=0; Design B style)
  - b1      : K=69,   1-digit bucketing  (SS=0, dw=0; Design B style)
  - nonum   : K=24,   <NUM> placeholder  (SS=0, dw=0; Design B endpoint)

Methodology caveat: the current (K=2418) endpoint uses the headline training
config (SS=0.5, dw=1.0); the three smaller-K variants share the Design-B style
(SS=0, dw=0). The three smaller-K variants are therefore mutually comparable
under a single methodology; the current endpoint is a methodologically
distinct reference. The spectrum within {24, 69, 335} reveals the
vocabulary-cardinality effect cleanly; the comparison to {2418} also carries
the SS / digit-head methodology difference.

Scoring: paper-grade position-aligned token accuracy + sequence exact-match,
applying the +1 EOS-slot offset correction (trim pred to len(true)) so the
metric is fair for both the current model (~94% TF windows have EOS-stripped
pred length already matching) and Design B-style models (model never predicts
EOS in the placeholder vocab, pred is always len+1 -> trim drops the
meaningless EOS-slot prediction).

Output: outputs/decoder20260511/audit/k_spectrum_compare.json
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO = Path("/home/seacuello/Documents/gcode_fingerprinting")
CK = REPO / "outputs/decoder20260511/checkpoints"
OUT = REPO / "outputs/decoder20260511/audit/k_spectrum_compare.json"

CURRENT_HASH = {1: "6o90io5p", 2: "wf2ulnx7", 3: "ua4ht85b", 4: "4rrcx1qd", 5: "gmtf655s"}


def is_numeric(tok: str) -> bool:
    """Numeric token: NUM_* (real or per-axis bucketing), dotted literal, or
    the Design B <NUM> placeholder."""
    return tok == "<NUM>" or tok.startswith("NUM_") or ("." in tok)


def score_window(true_s: str, pred_s: str) -> dict:
    T = true_s.split()
    P = pred_s.split()[: len(T)]                                    # offset correction
    n_struct = n_struct_ok = n_num = n_num_ok = 0
    struct_seq_ok = True
    for i, t in enumerate(T):
        p = P[i] if i < len(P) else None
        ok = int(p == t)
        if is_numeric(t):
            n_num += 1; n_num_ok += ok
        else:
            n_struct += 1; n_struct_ok += ok
            if not ok: struct_seq_ok = False
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


def score_variant(fold_paths: dict, beam_file: str) -> dict:
    per_fold = []
    for f, dir_ in fold_paths.items():
        path = dir_ / beam_file
        if not path.exists(): continue
        rows = json.loads(path.read_text())
        ws = [score_window(r["true"], r["pred"]) for r in rows]
        tot_tok = sum(w["n_tok"] for w in ws)
        tot_struct = sum(w["n_struct"] for w in ws)
        tot_num = sum(w["n_num"] for w in ws)
        per_fold.append(dict(
            fold=f,
            full_token_acc=sum(w["n_struct_ok"] + w["n_num_ok"] for w in ws) / max(tot_tok, 1),
            struct_token_acc=(sum(w["n_struct_ok"] for w in ws) / tot_struct) if tot_struct else None,
            num_token_acc=(sum(w["n_num_ok"] for w in ws) / tot_num) if tot_num else None,
            full_seq_exact=mean([w["full_ok"] for w in ws]),
            struct_seq_exact=mean([w["struct_seq_ok"] for w in ws]),
        ))
    keys = ["full_token_acc", "struct_token_acc", "num_token_acc",
            "full_seq_exact", "struct_seq_exact"]
    return {"per_fold": per_fold, "n_folds": len(per_fold),
            "aggregate": {k: agg([pf[k] for pf in per_fold]) for k in keys}}


def paired_diff(per_fold_a, per_fold_b, key):
    """Paired difference (A - B) per fold; mean, std, sign-count."""
    a = {pf["fold"]: pf[key] for pf in per_fold_a}
    b = {pf["fold"]: pf[key] for pf in per_fold_b}
    common = sorted(set(a) & set(b))
    diffs = [a[f] - b[f] for f in common if a[f] is not None and b[f] is not None]
    return dict(per_fold=diffs, mean=mean(diffs), std=st.pstdev(diffs) if diffs else 0.0,
                n_positive=sum(1 for d in diffs if d > 0), n=len(diffs))


def main() -> None:
    paths_current = {f: CK / f"full_window_5fold/fold_{f}/{h}/results"
                     for f, h in CURRENT_HASH.items()}
    paths_b2 = {f: CK / f"full_window_5fold_b2_ss0/fold_{f}/results" for f in range(1, 6)}
    paths_b1 = {f: CK / f"full_window_5fold_b1_ss0/fold_{f}/results" for f in range(1, 6)}
    paths_nonum = {f: CK / f"full_window_5fold_nonum_ss0/fold_{f}/results" for f in range(1, 6)}
    # T3.1 methodology-matched K=2418 control (SS=0, dw=0). Only present
    # if the training/eval pipeline has finished; we include it if the
    # beam_0 predictions exist on fold 1 (a cheap existence probe).
    paths_K2418_designB = {f: CK / f"full_window_5fold_designB_K2418/fold_{f}/results"
                           for f in range(1, 6)}
    have_K2418_designB = (paths_K2418_designB[1] / "beam_0_all_predictions.json").exists()

    variants = [
        ("current (4-digit, SS=0.5/dw=1.0)", 2418, paths_current),
        ("b2 (2-digit, SS=0/dw=0)",          335,  paths_b2),
        ("b1 (1-digit, SS=0/dw=0)",          69,   paths_b1),
        ("Design B (placeholder, SS=0/dw=0)", 24,  paths_nonum),
    ]
    if have_K2418_designB:
        variants.append(
            ("T3.1 control (4-digit, SS=0/dw=0)", 2418, paths_K2418_designB)
        )

    results = {"meta": {"step": "K-spectrum (4 vocab sizes)"
                                + (" + T3.1 control" if have_K2418_designB else ""),
                        "offset_correction": "pred[:len(true)]",
                        "methodology_caveat": "current is SS=0.5/dw=1.0; b2/b1/nonum are SS=0/dw=0"
                                              + ("; T3.1 control is K=2418 under SS=0/dw=0 "
                                                 "(the methodology-matched control)"
                                                 if have_K2418_designB else ""),
                        "T3.1_control_present": have_K2418_designB},
               "variants": []}
    for name, K, paths in variants:
        results["variants"].append({
            "name": name, "K": K,
            "TF": score_variant(paths, "beam_0_all_predictions.json"),
            "AR": score_variant(paths, "beam_1_all_predictions.json"),
        })

    # Paired diffs vs current (the K=2418 reference) on key AR metrics
    current_AR = results["variants"][0]["AR"]["per_fold"]
    paired = {}
    for v in results["variants"][1:]:
        paired[v["name"]] = {
            "struct_token_acc (AR)": paired_diff(v["AR"]["per_fold"], current_AR, "struct_token_acc"),
            "struct_seq_exact (AR)": paired_diff(v["AR"]["per_fold"], current_AR, "struct_seq_exact"),
        }
    results["paired_vs_current"] = paired

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    # ---- console table ------------------------------------------------------
    print("=" * 108)
    print("K-spectrum: AR structural recovery vs numeric vocabulary size")
    print("=" * 108)
    hdr = f"{'variant':<38}{'K':>6}{'TF struct tok':>16}{'AR struct tok':>16}{'TF struct seq':>16}{'AR struct seq':>16}"
    print(hdr); print("-" * len(hdr))
    for v in results["variants"]:
        st_tok_tf = v["TF"]["aggregate"]["struct_token_acc"]["mean"]
        st_tok_ar = v["AR"]["aggregate"]["struct_token_acc"]["mean"]
        st_seq_tf = v["TF"]["aggregate"]["struct_seq_exact"]["mean"]
        st_seq_ar = v["AR"]["aggregate"]["struct_seq_exact"]["mean"]
        std_tok_ar = v["AR"]["aggregate"]["struct_token_acc"]["std"]
        std_seq_ar = v["AR"]["aggregate"]["struct_seq_exact"]["std"]
        print(f"{v['name']:<38}{v['K']:>6}"
              f"{st_tok_tf:>16.4f}{st_tok_ar:>16.4f}{st_seq_tf:>16.4f}{st_seq_ar:>16.4f}")

    print("\nAR paired diffs vs current (K=2418), per-fold:")
    for name, p in paired.items():
        d = p["struct_token_acc (AR)"]
        print(f"  {name:<38}  struct_tok diff: mean={d['mean']:+.3f}+/-{d['std']:.3f}  "
              f"{d['n_positive']}/{d['n']} positive")
        d = p["struct_seq_exact (AR)"]
        print(f"  {name:<38}  struct_seq diff: mean={d['mean']:+.3f}+/-{d['std']:.3f}  "
              f"{d['n_positive']}/{d['n']} positive")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
