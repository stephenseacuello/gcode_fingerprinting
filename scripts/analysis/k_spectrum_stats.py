#!/usr/bin/env python3
"""K-spectrum inferential stats: paired t + Wilcoxon vs K=2418, plus
one-way repeated-measures ANOVA across the four K levels (fold is the
repeated subject).

Mirrors the inferential machinery used elsewhere in the paper
(audit/fsm_vs_bigram_paired_stats.json, audit/anova_results.json):
treats each test as descriptive given n=5 folds, reports both
parametric and nonparametric where applicable.

Inputs : audit/k_spectrum_compare.json (per-fold AR struct_token_acc /
         struct_seq_exact for each of the 4 K variants)
Outputs: audit/k_spectrum_stats.json + console summary table.
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

from scipy import stats as sp

REPO = Path("/home/seacuello/Documents/gcode_fingerprinting")
SRC = REPO / "outputs/decoder20260511/audit/k_spectrum_compare.json"
OUT = REPO / "outputs/decoder20260511/audit/k_spectrum_stats.json"


def cohens_dz(diffs):
    """Cohen's d_z for paired differences (mean / sd of diffs)."""
    if len(diffs) < 2:
        return None
    sd = st.stdev(diffs)
    return st.mean(diffs) / sd if sd > 0 else None


def paired_block(a, b):
    """a, b are per-fold values (same fold order). Returns paired t,
    Wilcoxon, Cohen's d_z, mean diff, sign count."""
    diffs = [ai - bi for ai, bi in zip(a, b)]
    n = len(diffs)
    n_pos = sum(1 for d in diffs if d > 0)
    t_stat, t_p = sp.ttest_rel(a, b)
    # Wilcoxon needs nonzero diffs; fall back gracefully if all-zero
    try:
        w_stat, w_p = sp.wilcoxon(a, b, zero_method="wilcox")
        w_stat = float(w_stat); w_p = float(w_p)
    except ValueError:
        w_stat, w_p = None, None
    return {
        "n": n,
        "mean_diff": st.mean(diffs),
        "sd_diff": st.stdev(diffs) if n > 1 else 0.0,
        "diffs_per_fold": diffs,
        "n_positive": n_pos,
        "paired_t": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_W": w_stat,
        "wilcoxon_p": w_p,
        "cohens_dz": cohens_dz(diffs),
    }


def rm_anova(condition_to_perfold):
    """One-way repeated-measures ANOVA. Conditions are columns, folds
    are rows (subjects). Returns F, p (with within-subject error df),
    plus eta^2_p and a Friedman nonparametric corroboration."""
    cond_names = list(condition_to_perfold.keys())
    k = len(cond_names)
    Y = [condition_to_perfold[c] for c in cond_names]            # k x n
    n = len(Y[0])
    grand = sum(sum(row) for row in Y) / (k * n)
    cond_means = [sum(row) / n for row in Y]
    subj_means = [sum(Y[c][s] for c in range(k)) / k for s in range(n)]
    SS_cond = n * sum((cm - grand) ** 2 for cm in cond_means)
    SS_subj = k * sum((sm - grand) ** 2 for sm in subj_means)
    SS_tot = sum((Y[c][s] - grand) ** 2 for c in range(k) for s in range(n))
    SS_err = SS_tot - SS_cond - SS_subj
    df_cond = k - 1
    df_err = (k - 1) * (n - 1)
    MS_cond = SS_cond / df_cond
    MS_err = SS_err / df_err if df_err > 0 else float("nan")
    F = MS_cond / MS_err if MS_err > 0 else float("nan")
    p = 1.0 - sp.f.cdf(F, df_cond, df_err) if not math.isnan(F) else float("nan")
    eta2p = SS_cond / (SS_cond + SS_err) if (SS_cond + SS_err) > 0 else None
    fried_stat, fried_p = sp.friedmanchisquare(*Y)
    return {
        "conditions": cond_names,
        "n_subjects": n,
        "F": F,
        "df_between": df_cond,
        "df_within": df_err,
        "p_value": float(p),
        "eta_squared_partial": eta2p,
        "friedman_chi2": float(fried_stat),
        "friedman_p": float(fried_p),
        "condition_means": dict(zip(cond_names, cond_means)),
    }


def main() -> None:
    d = json.loads(SRC.read_text())
    variants = {v["name"]: v for v in d["variants"]}

    # canonical short labels
    rename = {
        "current (4-digit, SS=0.5/dw=1.0)": "K2418",
        "b2 (2-digit, SS=0/dw=0)":          "K335",
        "b1 (1-digit, SS=0/dw=0)":          "K69",
        "Design B (placeholder, SS=0/dw=0)": "K24",
    }

    # per-metric, per-K, per-fold (sorted by fold index)
    per_metric = {}
    for metric in ("struct_token_acc", "struct_seq_exact"):
        per_metric[metric] = {}
        for full, short in rename.items():
            pf = variants[full]["AR"]["per_fold"]
            ordered = sorted(pf, key=lambda r: r["fold"])
            per_metric[metric][short] = [r[metric] for r in ordered]

    # ----- pairwise vs K=2418 ------------------------------------------------
    pairwise = {}
    for metric in per_metric:
        pairwise[metric] = {}
        ref = per_metric[metric]["K2418"]
        for k in ("K335", "K69", "K24"):
            pairwise[metric][f"{k}_vs_K2418"] = paired_block(per_metric[metric][k], ref)

    # ----- one-way RM-ANOVA across the 4 K levels ----------------------------
    anovas = {m: rm_anova(per_metric[m]) for m in per_metric}

    # ----- Holm correction across the 3 pairwise t-tests per metric ----------
    def holm(pvs):
        # returns adjusted p in original order, Holm-Bonferroni step-down
        order = sorted(range(len(pvs)), key=lambda i: pvs[i])
        adj = [None] * len(pvs)
        running = 0.0
        for rank, idx in enumerate(order):
            corr = (len(pvs) - rank) * pvs[idx]
            running = max(running, corr)
            adj[idx] = min(1.0, running)
        return adj

    for metric in pairwise:
        keys = list(pairwise[metric].keys())
        raw_p = [pairwise[metric][k]["paired_t_p"] for k in keys]
        adj = holm(raw_p)
        for k, ap in zip(keys, adj):
            pairwise[metric][k]["paired_t_p_holm"] = ap

    out = {
        "meta": {
            "source": str(SRC.relative_to(REPO)),
            "metric_definitions": {
                "struct_token_acc": "AR per-token accuracy on non-numeric positions",
                "struct_seq_exact": "AR per-window exact-match on the structural stream only",
            },
            "n_folds": 5,
            "note": "n=5 folds; treat all p-values descriptively (per Sec. inferential-scope)",
        },
        "per_fold": per_metric,
        "pairwise_vs_K2418": pairwise,
        "rm_anova_4K": anovas,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    # ------ console summary --------------------------------------------------
    print("=" * 94)
    print("K-spectrum inferential stats (n=5 folds, AR evaluation)")
    print("=" * 94)
    for metric in per_metric:
        print(f"\n[{metric}]  per-fold means by K:")
        for k in ("K2418", "K335", "K69", "K24"):
            vals = per_metric[metric][k]
            print(f"  {k:>6}: mean={st.mean(vals):+.4f}  std={st.stdev(vals):.4f}  "
                  f"folds={['%.3f' % v for v in vals]}")
        a = anovas[metric]
        print(f"  RM-ANOVA(4K): F({a['df_between']},{a['df_within']})={a['F']:.3f}, "
              f"p={a['p_value']:.4f}, eta2_p={a['eta_squared_partial']:.3f}; "
              f"Friedman chi2={a['friedman_chi2']:.3f}, p={a['friedman_p']:.4f}")
        print(f"  Pairwise vs K=2418 (paired t / Wilcoxon / Cohen's d_z / Holm-adj p):")
        for cmp, st_ in pairwise[metric].items():
            wp = st_["wilcoxon_p"]
            wp_s = f"{wp:.4f}" if wp is not None else "n/a"
            print(f"    {cmp:>15}: Δ={st_['mean_diff']:+.4f} "
                  f"({st_['n_positive']}/5 pos), "
                  f"t={st_['paired_t']:+.3f} p={st_['paired_t_p']:.4f} "
                  f"(Holm p={st_['paired_t_p_holm']:.4f}); "
                  f"W p={wp_s}; d_z={st_['cohens_dz']:+.3f}")
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
