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

import itertools
import json
import math
import statistics as st
from pathlib import Path

import numpy as np
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


def hedges_J(n):
    """Small-sample Hedges correction factor; J*d_z = g_z (less biased at small n)."""
    return 1.0 - 3.0 / (4 * (n - 1) - 1)


def boot_ci_mean(x, B=10000, seed=0):
    """Percentile bootstrap 95% CI for the mean of x."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    bs = rng.choice(x, size=(B, n), replace=True).mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def friedman_chi2_from_arr(Y):
    """Y: k x n matrix. Returns chi^2 statistic."""
    Y = np.asarray(Y, dtype=float)
    k, n = Y.shape
    R = np.zeros_like(Y)
    for j in range(n):
        R[:, j] = sp.rankdata(Y[:, j])
    Rsum = R.sum(axis=1)
    return float(12.0 / (n * k * (k + 1)) * (Rsum ** 2).sum() - 3 * n * (k + 1))


def exact_friedman_p(Y, max_exact_perms=10_000_000, mc_samples=200_000):
    """Friedman p-value with three regimes:
      - exact enumeration when (k!)^n <= max_exact_perms (~8M for k=4,n=5)
      - Monte-Carlo permutation otherwise (mc_samples random per-subject
        rank permutations) - returned with an "mc" flag
      For k=5, n=5, (5!)^5 = 24.8B perms is infeasible; we MC instead."""
    Y = np.asarray(Y, dtype=float)
    k, n = Y.shape
    obs = friedman_chi2_from_arr(Y)
    rankings = list(itertools.permutations(range(1, k + 1)))
    total = len(rankings) ** n
    if total <= max_exact_perms:
        count_ge = 0
        for combo in itertools.product(rankings, repeat=n):
            R = np.array(combo).T
            Rsum = R.sum(axis=1)
            chi2 = 12.0 / (n * k * (k + 1)) * (Rsum ** 2).sum() - 3 * n * (k + 1)
            if chi2 >= obs - 1e-9:
                count_ge += 1
        return obs, count_ge / total, total, "exact"
    # Monte-Carlo: sample mc_samples random rank-permutation assignments
    rng = np.random.default_rng(42)
    count_ge = 0
    for _ in range(mc_samples):
        R = np.empty((k, n), dtype=float)
        for j in range(n):
            R[:, j] = rng.permutation(k) + 1
        Rsum = R.sum(axis=1)
        chi2 = 12.0 / (n * k * (k + 1)) * (Rsum ** 2).sum() - 3 * n * (k + 1)
        if chi2 >= obs - 1e-9:
            count_ge += 1
    return obs, count_ge / mc_samples, mc_samples, "monte_carlo"


def grubbs(x):
    """Grubbs test for a single outlier. Returns G_max, G_min, idx_max, idx_min."""
    x = np.asarray(x, dtype=float)
    mu = x.mean(); sd = x.std(ddof=1)
    return ((x.max() - mu) / sd, (mu - x.min()) / sd,
            int(x.argmax()), int(x.argmin()))


def lofo_paired_t(diffs):
    """Leave-one-fold-out paired-t on a vector of per-fold paired differences."""
    out = []
    for drop in range(len(diffs)):
        sub = [d for i, d in enumerate(diffs) if i != drop]
        if len(sub) > 1 and st.stdev(sub) > 0:
            t = st.mean(sub) / (st.stdev(sub) / math.sqrt(len(sub)))
            p = 2 * (1 - sp.t.cdf(abs(t), df=len(sub) - 1))
        else:
            t, p = float("nan"), float("nan")
        out.append({"drop_fold_idx": drop + 1, "t": float(t), "p": float(p),
                    "mean_diff_remaining": st.mean(sub) if sub else None})
    return out


def paired_block(a, b):
    """a, b are per-fold values (same fold order). Returns paired t,
    Wilcoxon, Cohen's d_z, Hedges' g_z, bootstrap CI of paired diff,
    leave-one-fold-out sensitivity, mean diff, sign count."""
    diffs = [ai - bi for ai, bi in zip(a, b)]
    n = len(diffs)
    n_pos = sum(1 for d in diffs if d > 0)
    t_stat, t_p = sp.ttest_rel(a, b)
    try:
        w_stat, w_p = sp.wilcoxon(a, b, zero_method="wilcox")
        w_stat = float(w_stat); w_p = float(w_p)
    except ValueError:
        w_stat, w_p = None, None
    dz = cohens_dz(diffs)
    gz = hedges_J(n) * dz if dz is not None else None
    boot_lo, boot_hi = boot_ci_mean(diffs, seed=hash(tuple(diffs)) & 0xffff)
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
        "cohens_dz": dz,
        "hedges_gz": gz,
        "bootstrap_95ci_diff": [boot_lo, boot_hi],
        "leave_one_fold_out": lofo_paired_t(diffs),
    }


def rm_anova(condition_to_perfold, exact_friedman=True):
    """One-way repeated-measures ANOVA. Conditions are columns, folds
    are rows (subjects). Returns F, p (with within-subject error df),
    eta^2_p, asymptotic + EXACT Friedman p (the exact p is the
    primary inferential statement at n=5)."""
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
    fried_stat, fried_p_asy = sp.friedmanchisquare(*Y)
    out = {
        "conditions": cond_names,
        "n_subjects": n,
        "F": F,
        "df_between": df_cond,
        "df_within": df_err,
        "p_value_rmanova": float(p),
        "eta_squared_partial": eta2p,
        "friedman_chi2": float(fried_stat),
        "friedman_p_asymptotic": float(fried_p_asy),
        "condition_means": dict(zip(cond_names, cond_means)),
    }
    if exact_friedman:
        _, p_exact, n_perm, regime = exact_friedman_p(Y)
        out["friedman_p_exact"] = float(p_exact)
        out["friedman_n_permutations"] = n_perm
        out["friedman_regime"] = regime  # "exact" or "monte_carlo"
    return out


def main() -> None:
    d = json.loads(SRC.read_text())
    variants = {v["name"]: v for v in d["variants"]}

    # canonical short labels
    rename = {
        "current (4-digit, SS=0.5/dw=1.0)": "K2418",
        "b2 (2-digit, SS=0/dw=0)":          "K335",
        "b1 (1-digit, SS=0/dw=0)":          "K69",
        "Design B (placeholder, SS=0/dw=0)": "K24",
        "T3.1 control (4-digit, SS=0/dw=0)": "K2418_designB",
    }
    have_t3_1 = "T3.1 control (4-digit, SS=0/dw=0)" in variants

    # per-metric, per-K, per-fold (sorted by fold index)
    per_metric = {}
    short_keys = [s for full, s in rename.items() if full in variants]
    for metric in ("struct_token_acc", "struct_seq_exact"):
        per_metric[metric] = {}
        for full, short in rename.items():
            if full not in variants:
                continue
            pf = variants[full]["AR"]["per_fold"]
            ordered = sorted(pf, key=lambda r: r["fold"])
            per_metric[metric][short] = [r[metric] for r in ordered]

    # ----- pairwise vs K=2418 (headline) -------------------------------------
    pairwise = {}
    pair_targets = ["K335", "K69", "K24"]
    if have_t3_1:
        pair_targets.append("K2418_designB")
    for metric in per_metric:
        pairwise[metric] = {}
        ref = per_metric[metric]["K2418"]
        for k in pair_targets:
            pairwise[metric][f"{k}_vs_K2418"] = paired_block(per_metric[metric][k], ref)

    # ----- pairwise vs K2418_designB (the T3.1 matched-methodology ref) -----
    pairwise_vs_t3_1 = {}
    if have_t3_1:
        ref_targets = ["K335", "K69", "K24"]
        for metric in per_metric:
            pairwise_vs_t3_1[metric] = {}
            ref = per_metric[metric]["K2418_designB"]
            for k in ref_targets:
                pairwise_vs_t3_1[metric][f"{k}_vs_K2418_designB"] = \
                    paired_block(per_metric[metric][k], ref)

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

    for metric in pairwise_vs_t3_1:
        keys = list(pairwise_vs_t3_1[metric].keys())
        raw_p = [pairwise_vs_t3_1[metric][k]["paired_t_p"] for k in keys]
        adj = holm(raw_p)
        for k, ap in zip(keys, adj):
            pairwise_vs_t3_1[metric][k]["paired_t_p_holm"] = ap

    # ----- per-K bootstrap CI of mean + Grubbs outlier check ----------------
    per_K_summary = {}
    K_keys_present = [k for k in ("K2418", "K335", "K69", "K24", "K2418_designB")
                      if k in per_metric["struct_token_acc"]]
    for metric in per_metric:
        per_K_summary[metric] = {}
        for k in K_keys_present:
            vals = per_metric[metric][k]
            lo, hi = boot_ci_mean(vals, seed=hash((metric, k)) & 0xffff)
            G_max, G_min, idx_max, idx_min = grubbs(vals)
            per_K_summary[metric][k] = {
                "mean": st.mean(vals),
                "sd": st.stdev(vals),
                "bootstrap_95ci_mean": [lo, hi],
                "grubbs_G_max": float(G_max),
                "grubbs_G_min": float(G_min),
                "grubbs_idx_max_fold": idx_max + 1,
                "grubbs_idx_min_fold": idx_min + 1,
                "grubbs_critical_n5_alpha05_two_sided": 1.715,
            }

    out = {
        "meta": {
            "source": str(SRC.relative_to(REPO)),
            "metric_definitions": {
                "struct_token_acc": "AR per-token accuracy on non-numeric positions",
                "struct_seq_exact": "AR per-window exact-match on the structural stream only",
            },
            "n_folds": 5,
            "note": "n=5 folds; treat all p-values descriptively per Section sec:stats. "
                    "Primary omnibus is exact Friedman (asymptotic chi^2 unreliable at n=5).",
        },
        "per_fold": per_metric,
        "per_K_summary": per_K_summary,
        "pairwise_vs_K2418_headline": pairwise,
        "pairwise_vs_K2418_designB_matched": pairwise_vs_t3_1,
        "rm_anova": anovas,
        "T3.1_control_present": have_t3_1,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    # ------ console summary --------------------------------------------------
    print("=" * 94)
    print("K-spectrum inferential stats (n=5 folds, AR evaluation)")
    print("=" * 94)
    for metric in per_metric:
        print(f"\n[{metric}]  per-fold means by K:")
        for k in K_keys_present:
            vals = per_metric[metric][k]
            print(f"  {k:>6}: mean={st.mean(vals):+.4f}  std={st.stdev(vals):.4f}  "
                  f"folds={['%.3f' % v for v in vals]}")
        a = anovas[metric]
        print(f"  RM-ANOVA(4K): F({a['df_between']},{a['df_within']})={a['F']:.3f}, "
              f"p={a['p_value_rmanova']:.4f}, eta2_p={a['eta_squared_partial']:.3f}; "
              f"Friedman chi2={a['friedman_chi2']:.3f}, asymptotic p={a['friedman_p_asymptotic']:.4f}, "
              f"EXACT p={a.get('friedman_p_exact', float('nan')):.4f}")
        print(f"  Pairwise vs K=2418 (paired t / Wilcoxon / d_z / Hedges g_z / Holm-adj p / 95%CI):")
        for cmp, st_ in pairwise[metric].items():
            wp = st_["wilcoxon_p"]
            wp_s = f"{wp:.4f}" if wp is not None else "n/a"
            lo, hi = st_["bootstrap_95ci_diff"]
            print(f"    {cmp:>15}: Δ={st_['mean_diff']:+.4f} "
                  f"({st_['n_positive']}/5 pos), "
                  f"t={st_['paired_t']:+.3f} p={st_['paired_t_p']:.4f} "
                  f"(Holm p={st_['paired_t_p_holm']:.4f}); "
                  f"W p={wp_s}; d_z={st_['cohens_dz']:+.3f} "
                  f"(g_z={st_['hedges_gz']:+.3f}); "
                  f"95%CI=[{lo:+.4f},{hi:+.4f}]")
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
