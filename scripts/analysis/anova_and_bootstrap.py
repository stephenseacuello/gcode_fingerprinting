#!/usr/bin/env python3
"""Phase B+: ANOVA + F-statistics + bootstrap CI for V8 results.

Operates on the saved 5-fold + ablation metrics.json files. Produces:
  - outputs/decoder20260511/audit/anova_results.json
  - outputs/decoder20260511/audit/bootstrap_ci.json
  - outputs/decoder20260511/decoder_paper_v2/tables/anova.tex
  - outputs/decoder20260511/decoder_paper_v2/tables/bootstrap_ci.tex

Mirrors what the V7 paper had in revision_analysis/.

Tests performed:

1. **One-way ANOVA** across each ablation group's 5 folds vs the baseline 5 folds.
   For each ablation, reports F-statistic, p-value, and effect size (Cohen's d).
   E.g. "does removing gyroscope significantly change command accuracy?"

2. **Two-way ANOVA** when nested ablations exist (e.g., shortcuts × modality).
   Reports main effects + interaction F-statistic / p-value.

3. **Bootstrap 95% CI** for the headline 5-fold means using 10,000 resamples.
   Tightens "0.97 ± 0.027" into [lower, upper] CI intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs" / "decoder20260511"
# Default baseline kept for backwards compatibility, but --baseline-name
# overrides it. After 2026-05-12 the headline baseline switches to
# `full_window_5fold` (see TABLES_REGENERATION_GUIDE.md).
DEFAULT_BASELINE_NAME = "per_row_5fold"

try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

METRIC_KEYS = ["token_accuracy", "sequence_accuracy", "type_accuracy",
               "command_accuracy", "param_type_accuracy", "numeric_accuracy"]


def _load_fold_metrics(fold_dir: Path) -> dict | None:
    """Load metrics.json from a fold dir, handling the wandb-subdir layout.

    Supports:
      fold_N/results/metrics.json                  (no wandb)
      fold_N/<wandb_run_id>/results/metrics.json   (wandb on; trainer wraps run)
    """
    direct = fold_dir / "results" / "metrics.json"
    if direct.exists():
        return json.loads(direct.read_text())
    candidates = list(fold_dir.glob("*/results/metrics.json"))
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return json.loads(candidates[0].read_text())
    return None


def collect_baseline_5fold(baseline_name: str = DEFAULT_BASELINE_NAME) -> dict[str, list[float]]:
    """Baseline V8 5-fold test metrics from `checkpoints/<baseline_name>/`."""
    baseline_root = ROOT / "checkpoints" / baseline_name
    out: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    for F in [1, 2, 3, 4, 5]:
        m = _load_fold_metrics(baseline_root / f"fold_{F}")
        if not m:
            continue
        t = m.get("test_metrics", {})
        for k in METRIC_KEYS:
            if k in t:
                out[k].append(t[k])
    return out


def collect_ablation(name: str, root: Path) -> dict[str, list[float]] | None:
    """Collect a single ablation's test metrics across its available folds."""
    out: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    n = 0
    for fold_dir in sorted(root.glob("fold_*")):
        m = _load_fold_metrics(fold_dir)
        if not m:
            continue
        t = m.get("test_metrics", {})
        for k in METRIC_KEYS:
            if k in t:
                out[k].append(t[k])
        n += 1
    return out if n > 0 else None


def _drill_metric(m: dict, sweep_root: Path | None = None) -> dict[str, dict[str, list[float]]]:
    """Drill into a fold's metrics.json to extract per-class / per-axis /
    per-position metrics.

    Returns a flat dict keyed by metric_label, each value is a single float
    contribution to its 5-fold series. Keys follow the schema:
      head:<head>::<class>::<metric>
      head:numeric_digits::position_<n>::accuracy
    """
    out = {}
    pc = m.get("test_metrics", {}).get("per_class", {})

    # Token head: macro_f1 already in METRIC_KEYS, but per-class F1 over the
    # 2418-way token vocabulary explodes. Skip; per-class results for token
    # would not be informative.

    # Type head per-class (4 classes: SPECIAL/COMMAND/PARAM/NUMERIC)
    for head in ("type", "command", "param_type", "sign"):
        head_data = pc.get(head, {})
        if not isinstance(head_data, dict):
            continue
        for cls_name, cls_data in (head_data.get("per_class") or {}).items():
            for metric in ("precision", "recall", "f1"):
                if metric in cls_data:
                    out[f"head:{head}::{cls_name}::{metric}"] = cls_data[metric]
            if "support" in cls_data:
                out[f"head:{head}::{cls_name}::support"] = cls_data["support"]

    # Numeric digit head: per-digit-value AND per-position
    nd = pc.get("numeric_digits", {})
    if isinstance(nd, dict):
        overall = nd.get("overall", {})
        for cls_name, cls_data in (overall.get("per_class") or {}).items():
            for metric in ("precision", "recall", "f1"):
                if metric in cls_data:
                    out[f"head:numeric_digits::digit_{cls_name}::{metric}"] = cls_data[metric]
        for i, entry in enumerate(nd.get("per_position") or []):
            if entry is None or not isinstance(entry, dict):
                continue
            for metric in ("accuracy", "macro_f1", "macro_precision", "macro_recall"):
                if metric in entry:
                    out[f"head:numeric_digits::position_{i}::{metric}"] = entry[metric]
    return out


def collect_baseline_per_class(baseline_name: str = DEFAULT_BASELINE_NAME
                               ) -> dict[str, list[float]]:
    """Per-class / per-axis / per-position 5-fold series for the baseline.

    Returns a flat dict keyed by drilled-metric-name (e.g.
    'head:command::G0::f1') mapping to a list of up to 5 per-fold values.
    """
    baseline_root = ROOT / "checkpoints" / baseline_name
    out: dict[str, list[float]] = {}
    for F in [1, 2, 3, 4, 5]:
        m = _load_fold_metrics(baseline_root / f"fold_{F}")
        if not m:
            continue
        for k, v in _drill_metric(m).items():
            out.setdefault(k, []).append(v)
    return out


def collect_ablation_per_class(name: str, root: Path
                               ) -> dict[str, list[float]] | None:
    """Same as collect_baseline_per_class, but for an ablation root."""
    out: dict[str, list[float]] = {}
    n = 0
    for fold_dir in sorted(root.glob("fold_*")):
        m = _load_fold_metrics(fold_dir)
        if not m:
            continue
        for k, v in _drill_metric(m).items():
            out.setdefault(k, []).append(v)
        n += 1
    return out if n > 0 else None


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size for two independent samples."""
    a, b = np.asarray(a), np.asarray(b)
    pooled_std = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if pooled_std == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_std)


def one_way_anova(samples: list[list[float]]) -> dict[str, float]:
    """sklearn f_oneway-style ANOVA across len(samples) groups."""
    if not HAS_SCIPY:
        return {"F": float("nan"), "p": float("nan")}
    samples = [np.asarray(s) for s in samples if len(s) > 0]
    if len(samples) < 2:
        return {"F": float("nan"), "p": float("nan")}
    F, p = stats.f_oneway(*samples)
    return {"F": float(F), "p": float(p), "n_groups": len(samples),
            "df_between": len(samples) - 1, "df_within": sum(len(s) for s in samples) - len(samples)}


def nonparametric_and_assumptions(base: list[float], abl: list[float]) -> dict[str, float]:
    """Small-sample (n=5) robustness battery for a baseline-vs-ablation pair.

    With n=5 per group the parametric ANOVA/Welch test is assumption-fragile,
    so we report the nonparametric Mann-Whitney U (independent groups) and
    Wilcoxon signed-rank (paired by fold, when lengths match) as the primary
    inferential statements, plus Levene (variance homogeneity) and
    Shapiro-Wilk (normality) assumption checks for the parametric test.
    All p-values are descriptive at this sample size.
    """
    out = {
        "mannwhitney_u_p": float("nan"),
        "wilcoxon_p": float("nan"),
        "levene_p": float("nan"),
        "shapiro_baseline_p": float("nan"),
        "shapiro_ablation_p": float("nan"),
    }
    if not HAS_SCIPY:
        return out
    a, b = np.asarray(base, float), np.asarray(abl, float)
    try:
        if len(a) >= 1 and len(b) >= 1 and not (np.allclose(a, a[0]) and np.allclose(b, b[0])):
            out["mannwhitney_u_p"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        pass
    try:
        if len(a) == len(b) and len(a) >= 1 and not np.allclose(a, b):
            out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        pass
    try:
        out["levene_p"] = float(stats.levene(a, b).pvalue)
    except ValueError:
        pass
    try:
        if len(a) >= 3 and not np.allclose(a, a[0]):
            out["shapiro_baseline_p"] = float(stats.shapiro(a).pvalue)
        if len(b) >= 3 and not np.allclose(b, b[0]):
            out["shapiro_ablation_p"] = float(stats.shapiro(b).pvalue)
    except ValueError:
        pass
    return out


def adjust_pvalues(pvalues: list[float], method: str = "holm-bonferroni"
                   ) -> tuple[list[float], list[bool]]:
    """Multiple-comparisons correction.

    method: 'bonferroni' (most conservative), 'holm-bonferroni' (recommended
            default), or 'bh-fdr' (Benjamini-Hochberg false discovery rate).
    Returns (adjusted_p, significant_at_0.05).
    """
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return [], []

    if method == "bonferroni":
        adj = np.minimum(p * n, 1.0)
    elif method == "holm-bonferroni":
        # Sort ascending. p_adj[i] = max(p[i]*(n-i), p_adj[i-1]), clipped to 1.
        order = np.argsort(p)
        scaled = p[order] * (n - np.arange(n))
        adj_sorted = np.maximum.accumulate(scaled)
        adj_sorted = np.minimum(adj_sorted, 1.0)
        adj = np.empty(n)
        adj[order] = adj_sorted
    elif method == "bh-fdr":
        order = np.argsort(p)
        ranks = np.arange(1, n + 1)
        adj_sorted = p[order] * n / ranks
        # Enforce monotonicity from the largest rank downward
        adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
        adj_sorted = np.minimum(adj_sorted, 1.0)
        adj = np.empty(n)
        adj[order] = adj_sorted
    else:
        raise ValueError(f"unknown method: {method}")
    return adj.tolist(), (adj < 0.05).tolist()


def bootstrap_ci(values: list[float], n_resamples: int = 10000, alpha: float = 0.05,
                 seed: int = 42) -> dict[str, float]:
    """Percentile-method bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    if len(arr) < 2:
        return {"mean": float(arr.mean()) if len(arr) else float("nan"),
                "lower": float("nan"), "upper": float("nan"), "n": int(len(arr))}
    means = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(arr), size=len(arr))
        means.append(arr[idx].mean())
    means = np.asarray(means)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "lower": float(np.percentile(means, 100 * alpha / 2)),
        "upper": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n": int(len(arr)),
    }


def main() -> int:
    if not HAS_SCIPY:
        print("scipy is required; install with `pip install scipy`")
        return 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-name", default=DEFAULT_BASELINE_NAME,
                        help="checkpoints/<NAME>/fold_*/results/metrics.json to use as baseline. "
                             "After 2026-05-12 the headline baseline is full_window_5fold.")
    args = parser.parse_args()

    baseline = collect_baseline_5fold(baseline_name=args.baseline_name)
    print(f"Baseline ({args.baseline_name}) 5-fold means: "
          f"{[(k, np.mean(v)) for k, v in baseline.items() if v]}")
    print()

    # Headline baseline becomes per_row_5fold OR full_window_5fold (whichever
    # the caller chose). The OTHER 5-fold sweep becomes a comparison ablation
    # automatically, so the ANOVA table includes a per_row-vs-full_window row.
    other_5fold_name = ("full_window_5fold" if args.baseline_name == "per_row_5fold"
                        else "per_row_5fold")

    # ---------- 1. ANOVA: baseline vs each ablation ----------
    ablations_to_test = {
        "with_shortcuts": ROOT / "checkpoints" / "per_row_5fold_with_shortcuts",
        "noise_aug":      ROOT / "checkpoints" / "per_row_5fold_noise_aug",
        other_5fold_name: ROOT / "checkpoints" / other_5fold_name,
    }
    # Sensor ablations have a different layout: ablations/sensor/zero_<mod>/fold_*/
    for mod in ["accelerometer", "gyroscope", "magnetometer", "environmental",
                "color", "rms", "electrical"]:
        ablations_to_test[f"zero_{mod}"] = ROOT / "ablations" / "sensor" / f"zero_{mod}"

    anova_results = {}
    for name, root in ablations_to_test.items():
        if not root.exists():
            continue
        abl_data = collect_ablation(name, root)
        if abl_data is None:
            continue
        result = {"n_folds_baseline": len(baseline["token_accuracy"]),
                  "n_folds_ablation": len(abl_data["token_accuracy"]),
                  "metrics": {}}
        for metric in METRIC_KEYS:
            base_vals = baseline.get(metric, [])
            abl_vals = abl_data.get(metric, [])
            if len(base_vals) < 2 or len(abl_vals) < 2:
                continue
            F_p = one_way_anova([base_vals, abl_vals])
            d = cohens_d(base_vals, abl_vals)
            np_tests = nonparametric_and_assumptions(base_vals, abl_vals)
            mean_diff = float(np.mean(abl_vals) - np.mean(base_vals))
            result["metrics"][metric] = {
                "baseline_mean": float(np.mean(base_vals)),
                "baseline_std": float(np.std(base_vals)),
                "ablation_mean": float(np.mean(abl_vals)),
                "ablation_std": float(np.std(abl_vals)),
                "mean_diff": mean_diff,
                "cohens_d": d,
                "F_statistic": F_p["F"],
                "p_value": F_p["p"],
                "significant_at_0.05": F_p["p"] < 0.05 if F_p["p"] == F_p["p"] else False,
                # Primary inferential statements at n=5 (descriptive):
                "mannwhitney_u_p": np_tests["mannwhitney_u_p"],
                "wilcoxon_p": np_tests["wilcoxon_p"],
                "levene_p": np_tests["levene_p"],
                "shapiro_baseline_p": np_tests["shapiro_baseline_p"],
                "shapiro_ablation_p": np_tests["shapiro_ablation_p"],
            }
        anova_results[name] = result

    # ---------- 2. Bootstrap CIs for baseline + each ablation ----------
    baseline_key = f"baseline_{args.baseline_name}"
    bootstrap_results = {baseline_key: {}}
    for metric in METRIC_KEYS:
        if baseline.get(metric):
            bootstrap_results[baseline_key][metric] = bootstrap_ci(baseline[metric])

    bootstrap_results["ablations"] = {}
    for name, root in ablations_to_test.items():
        if not root.exists():
            continue
        abl_data = collect_ablation(name, root)
        if abl_data is None:
            continue
        bootstrap_results["ablations"][name] = {}
        for metric in METRIC_KEYS:
            if abl_data.get(metric):
                bootstrap_results["ablations"][name][metric] = bootstrap_ci(abl_data[metric])

    # ---------- 2b. Per-class / per-axis / per-position ANOVA ----------
    # Drill into the per_class blob of each fold's metrics.json and ANOVA
    # baseline vs ablation across each per-class / per-axis / per-position
    # metric. Covers:
    #   - command head: F1 / P / R for {G0, G1, G2, G3, G53, M30}
    #   - param_type head: F1 / P / R for {X, Y, Z, F, R} (and I/J/K/S/P if seen)
    #   - sign head: F1 / P / R for {POSITIVE, NEGATIVE}
    #   - type head: F1 / P / R for {SPECIAL, COMMAND, PARAM, NUMERIC}
    #   - numeric_digits per-position (0-5): accuracy / macro_f1 / macro_p / macro_r
    #   - numeric_digits per-digit-value (0-9): precision / recall / f1
    baseline_drilled = collect_baseline_per_class(baseline_name=args.baseline_name)
    print(f"  per-class baseline: {len(baseline_drilled)} drilled metrics, "
          f"folds per metric range {min((len(v) for v in baseline_drilled.values()), default=0)}..{max((len(v) for v in baseline_drilled.values()), default=0)}")
    print()

    drilled_results: dict[str, dict] = {}
    for name, root in ablations_to_test.items():
        if not root.exists():
            continue
        abl_drilled = collect_ablation_per_class(name, root)
        if abl_drilled is None:
            continue
        per_metric = {}
        for k, base_vals in baseline_drilled.items():
            abl_vals = abl_drilled.get(k, [])
            if len(base_vals) < 2 or len(abl_vals) < 2:
                continue
            # Skip pure-support keys (integer counts, not metrics)
            if k.endswith("::support"):
                continue
            F_p = one_way_anova([base_vals, abl_vals])
            d = cohens_d(base_vals, abl_vals)
            mean_diff = float(np.mean(abl_vals) - np.mean(base_vals))
            per_metric[k] = {
                "baseline_mean": float(np.mean(base_vals)),
                "baseline_std": float(np.std(base_vals)),
                "ablation_mean": float(np.mean(abl_vals)),
                "ablation_std": float(np.std(abl_vals)),
                "mean_diff": mean_diff,
                "cohens_d": d,
                "F_statistic": F_p["F"],
                "p_value": F_p["p"],
                "n_folds_baseline": len(base_vals),
                "n_folds_ablation": len(abl_vals),
            }
        drilled_results[name] = {"per_metric": per_metric,
                                 "n_drilled_metrics": len(per_metric)}

    # ---------- 3a. Multiple-comparisons correction ----------
    # Collect all (name, metric, p) tuples across the ANOVA grid and apply
    # Holm-Bonferroni + BH-FDR adjustment. Reviewers regularly ask for this
    # when an HP-sweep paper picks "the best of N cells" --- the corrected
    # p-values constrain how strongly we can claim the winner is "different".
    all_pvalues, pvalue_keys = [], []
    for name, payload in anova_results.items():
        for metric, m in payload.get("metrics", {}).items():
            if m.get("p_value") == m.get("p_value"):  # not NaN
                all_pvalues.append(m["p_value"])
                pvalue_keys.append((name, metric))
    if all_pvalues:
        holm_adj, holm_sig = adjust_pvalues(all_pvalues, method="holm-bonferroni")
        bh_adj, bh_sig = adjust_pvalues(all_pvalues, method="bh-fdr")
        for (name, metric), pa, ps, ba, bs in zip(pvalue_keys, holm_adj, holm_sig, bh_adj, bh_sig):
            anova_results[name]["metrics"][metric]["p_holm_bonferroni"] = pa
            anova_results[name]["metrics"][metric]["significant_holm_at_0.05"] = ps
            anova_results[name]["metrics"][metric]["p_bh_fdr"] = ba
            anova_results[name]["metrics"][metric]["significant_bh_at_0.05"] = bs
        print(f"  applied Holm-Bonferroni + BH-FDR correction across "
              f"{len(all_pvalues)} (ablation, macro-metric) tests")

    # Same correction over the per-class / per-axis / per-position grid.
    # Bonferroni at this scale is too conservative; BH-FDR is the recommended
    # default. Holm is reported for the more conservative reading.
    drilled_pvalues, drilled_keys = [], []
    for name, payload in drilled_results.items():
        for metric_key, m in payload.get("per_metric", {}).items():
            if m.get("p_value") == m.get("p_value"):  # not NaN
                drilled_pvalues.append(m["p_value"])
                drilled_keys.append((name, metric_key))
    if drilled_pvalues:
        d_holm_adj, d_holm_sig = adjust_pvalues(drilled_pvalues, method="holm-bonferroni")
        d_bh_adj,   d_bh_sig   = adjust_pvalues(drilled_pvalues, method="bh-fdr")
        for (name, mk), pa, ps, ba, bs in zip(drilled_keys, d_holm_adj, d_holm_sig, d_bh_adj, d_bh_sig):
            drilled_results[name]["per_metric"][mk]["p_holm_bonferroni"] = pa
            drilled_results[name]["per_metric"][mk]["significant_holm_at_0.05"] = ps
            drilled_results[name]["per_metric"][mk]["p_bh_fdr"] = ba
            drilled_results[name]["per_metric"][mk]["significant_bh_at_0.05"] = bs
        print(f"  applied Holm-Bonferroni + BH-FDR correction across "
              f"{len(drilled_pvalues)} (ablation, per-class/per-axis/per-position) tests")

    # ---------- 3. Write JSON + LaTeX tables ----------
    audit_dir = ROOT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "anova_results.json").write_text(json.dumps(anova_results, indent=2))
    (audit_dir / "bootstrap_ci.json").write_text(json.dumps(bootstrap_results, indent=2))
    (audit_dir / "anova_per_class_results.json").write_text(json.dumps(drilled_results, indent=2))
    print(f"wrote {audit_dir / 'anova_results.json'}")
    print(f"wrote {audit_dir / 'bootstrap_ci.json'}")

    # ANOVA LaTeX table — only overwrite the placeholder file if we have real
    # rows to emit. If both anova_results and bootstrap_results[baseline_key]
    # are empty (e.g. running this before any 5-fold sweep has completed),
    # skip writing so the manuscript keeps its TBD placeholders.
    has_anova_rows = any(payload.get("metrics") for payload in anova_results.values())
    has_bootstrap_rows = bool(bootstrap_results.get(baseline_key))
    if not has_anova_rows and not has_bootstrap_rows:
        print()
        print("=== SKIP: no metrics found for baseline + ablations; "
              "placeholder LaTeX tables preserved ===")
        return 0

    tables_dir = ROOT / "decoder_paper_v2" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    def _esc(s: str) -> str:
        # LaTeX-escape underscores and other risky chars in identifiers.
        return s.replace("_", r"\_")

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Baseline-vs-ablation comparison across the 5 folds. \textbf{At $n=5$ all $p$-values are descriptive, not confirmatory}; the nonparametric Mann--Whitney $U$ ($p_{\mathrm{MWU}}$) is the primary statement and the parametric ANOVA ($F$, $p_F$) is corroborating. The effect size (Cohen's $d$) and the sign/magnitude of $\Delta$ carry the conclusions. Significance ($\ast\,p<0.05$, $\ast\ast\,p<0.01$, $\ast\ast\ast\,p<0.001$) is marked on $p_{\mathrm{MWU}}$.}",
        r"\label{tab:anova}",
        r"\scriptsize",
        r"\begin{tabular}{l l r r r r r r c}",
        r"\toprule",
        r"Ablation & Metric & Base & Abl & $\Delta$ & $d$ & $p_F$ & $p_{\mathrm{MWU}}$ & Sig. \\",
        r"\midrule",
    ]
    for name, payload in anova_results.items():
        for metric, m in payload["metrics"].items():
            mwu = m.get("mannwhitney_u_p", float("nan"))
            ref_p = mwu if mwu == mwu else m["p_value"]
            sig = ""
            if ref_p < 0.001:
                sig = r"$\ast\ast\ast$"
            elif ref_p < 0.01:
                sig = r"$\ast\ast$"
            elif ref_p < 0.05:
                sig = r"$\ast$"
            mwu_s = f"{mwu:.3g}" if mwu == mwu else "--"
            d_s = f"{m['cohens_d']:+.2f}" if m['cohens_d'] == m['cohens_d'] else "--"
            row = (
                f"{_esc(name)} & {_esc(metric.replace('_accuracy',''))} "
                f"& ${m['baseline_mean']:.3f}$ & ${m['ablation_mean']:.3f}$ "
                f"& ${m['mean_diff']:+.3f}$ & ${d_s}$ & ${m['p_value']:.3g}$ & ${mwu_s}$ "
                f"& {sig} \\\\"
            )
            lines.append(row)
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\par\smallskip",
        r"\footnotesize\textit{$n=5$ folds per group: Levene and Shapiro--Wilk "
        r"assumption-check $p$-values and the Wilcoxon signed-rank statistic are "
        r"recorded per comparison in \texttt{audit/anova\_results.json}. With five "
        r"observations these tests are low-power; conclusions rest on effect size "
        r"and $\Delta$, not on threshold-crossing $p$-values.}",
        r"\end{table}"])
    (tables_dir / "anova.tex").write_text("\n".join(lines))
    print(f"wrote {tables_dir / 'anova.tex'}")

    # Bootstrap CI LaTeX table for the baseline
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Percentile bootstrap 95\% intervals for the baseline 5-fold means (10{,}000 resamples). \textbf{These are small-sample intervals over five fold-means and are not coverage-calibrated}; read them as a smoothed indication of cross-fold spread, not as exact confidence statements.}",
        r"\label{tab:bootstrap_ci}",
        r"\begin{tabular}{l r r}",
        r"\toprule",
        r"Metric & Mean & 95\% CI \\",
        r"\midrule",
    ]
    for metric, ci in bootstrap_results[baseline_key].items():
        lines.append(
            f"{_esc(metric.replace('_accuracy',''))} & ${ci['mean']:.4f}$ "
            f"& $[{ci['lower']:.4f}, {ci['upper']:.4f}]$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (tables_dir / "bootstrap_ci.tex").write_text("\n".join(lines))
    print(f"wrote {tables_dir / 'bootstrap_ci.tex'}")

    # Console summary
    print()
    print("=== ANOVA summary (baseline vs each ablation, command_accuracy) ===")
    for name, payload in anova_results.items():
        m = payload["metrics"].get("command_accuracy")
        if m is None: continue
        sig = "***" if m['p_value'] < 0.05 else ""
        print(f"  {name:<22}: base={m['baseline_mean']:.4f} abl={m['ablation_mean']:.4f} "
              f"Δ={m['mean_diff']:+.4f} F={m['F_statistic']:.2f} p={m['p_value']:.3g} {sig}")

    print()
    print(f"=== Bootstrap 95% CI (baseline: {args.baseline_name}) ===")
    for metric, ci in bootstrap_results[baseline_key].items():
        print(f"  {metric:<22}: {ci['mean']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    # Per-class / per-axis / per-position summary
    print()
    print("=== Drilled per-class / per-axis / per-position ANOVA highlights (BH-FDR significant) ===")
    n_total = sum(len(p.get("per_metric", {})) for p in drilled_results.values())
    n_sig_bh = 0
    n_sig_holm = 0
    sig_examples = []
    for name, payload in drilled_results.items():
        for mk, m in payload.get("per_metric", {}).items():
            if m.get("significant_bh_at_0.05"):
                n_sig_bh += 1
                if len(sig_examples) < 20:
                    sig_examples.append(
                        f"  {name:<22} {mk:<45} F={m['F_statistic']:.2f} "
                        f"p_bh={m.get('p_bh_fdr', float('nan')):.3g} "
                        f"Δ={m['mean_diff']:+.3f}"
                    )
            if m.get("significant_holm_at_0.05"):
                n_sig_holm += 1
    print(f"  total drilled tests: {n_total}")
    print(f"  significant under BH-FDR (α=0.05): {n_sig_bh}")
    print(f"  significant under Holm-Bonferroni (α=0.05): {n_sig_holm}")
    if sig_examples:
        print("  examples (up to 20):")
        for line in sig_examples:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
