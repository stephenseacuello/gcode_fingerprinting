#!/usr/bin/env python3
"""Experiment 7: Cross-Condition NLL Distribution Analysis

Compares NLL score distributions across machining conditions:
  - air-cut: adaptive, face, pocket
  - active-cut: adaptive150025, face150025, pocket150025
  - damage: damageadaptive, damageface, damagepocket

Tests whether the decoder's surprisal differs systematically between
conditions using two-sample KS tests and Cohen's d.
"""

import json
import logging
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import (
    BASE_DIR,
    FOLDS,
    load_cached_nll,
    load_cached_targets,
    compute_window_scores,
    cohens_d,
    holm_bonferroni,
    save_results,
    save_figure,
    append_to_summary,
    setup_logging,
    get_default_parser,
)

logger = logging.getLogger(__name__)

# Condition grouping
CONDITION_MAP = {
    "adaptive": "air-cut",
    "face": "air-cut",
    "pocket": "air-cut",
    "adaptive150025": "active-cut",
    "face150025": "active-cut",
    "pocket150025": "active-cut",
    "damageadaptive": "damage",
    "damageface": "damage",
    "damagepocket": "damage",
}

CONDITION_ORDER = ["air-cut", "active-cut", "damage"]


def _assign_conditions(op_type_names: list) -> np.ndarray:
    """Map each window's operation_type_name to a condition label."""
    conditions = []
    for name in op_type_names:
        # Strip trailing digits from names like "adaptive_001" -> "adaptive"
        base = name.split("_")[0] if "_" in name else name
        # Also handle names that are just the base (no underscore)
        cond = CONDITION_MAP.get(base, CONDITION_MAP.get(name, "unknown"))
        conditions.append(cond)
    return np.array(conditions)


def run_fold(fold: int, output_dir: Path) -> dict:
    """Run cross-condition analysis for one fold."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Fold {fold} ---")

    # Load cached data
    targets_data = load_cached_targets(fold)
    nll_data = load_cached_nll(fold)

    lengths = targets_data["lengths"]
    op_type_names = targets_data["operation_type_names"]
    nll_legacy = nll_data["nll_legacy"]

    # Compute window-level scores
    scores = compute_window_scores(nll_legacy, lengths)
    s_mean = scores["S_mean"].numpy()

    # Assign conditions
    conditions = _assign_conditions(op_type_names)
    unique_conditions = [c for c in CONDITION_ORDER if c in conditions]
    logger.info(f"  Conditions found: {unique_conditions}")
    for c in unique_conditions:
        n = np.sum(conditions == c)
        logger.info(f"    {c}: {n} windows, S_mean={s_mean[conditions == c].mean():.4f}")

    # Group scores by condition
    cond_scores = {}
    for cond in unique_conditions:
        mask = conditions == cond
        cond_scores[cond] = s_mean[mask]

    # Pairwise KS tests and Cohen's d
    pairs = list(combinations(unique_conditions, 2))
    ks_results = {}
    d_results = {}
    p_values_raw = []

    for c1, c2 in pairs:
        from scipy.stats import ks_2samp

        stat, p = ks_2samp(cond_scores[c1], cond_scores[c2])
        d = cohens_d(cond_scores[c1], cond_scores[c2])
        key = f"{c1}_vs_{c2}"
        ks_results[key] = {"ks_stat": float(stat), "p_value": float(p)}
        d_results[key] = float(d)
        p_values_raw.append(float(p))
        logger.info(f"  {key}: KS={stat:.4f}, p={p:.2e}, d={d:.3f}")

    # Holm-Bonferroni correction
    if p_values_raw:
        adjusted = holm_bonferroni(p_values_raw)
        for i, (c1, c2) in enumerate(pairs):
            key = f"{c1}_vs_{c2}"
            ks_results[key]["p_adjusted"] = adjusted[i]

    fold_result = {
        "fold": fold,
        "n_windows": len(s_mean),
        "condition_stats": {
            cond: {
                "n": int(np.sum(conditions == cond)),
                "mean": float(cond_scores[cond].mean()) if cond in cond_scores else None,
                "std": float(cond_scores[cond].std()) if cond in cond_scores else None,
                "median": float(np.median(cond_scores[cond])) if cond in cond_scores else None,
            }
            for cond in unique_conditions
        },
        "ks_tests": ks_results,
        "cohens_d": d_results,
    }

    # Save per-condition scores for figure generation
    torch.save(
        {cond: vals.tolist() for cond, vals in cond_scores.items()},
        fold_dir / "condition_scores.pt",
    )
    save_results(fold_result, fold_dir, "fold_results.json")
    return fold_result


def aggregate(fold_results: list, output_dir: Path) -> dict:
    """Cross-fold aggregation of KS tests and effect sizes."""
    # Pool condition scores across folds
    pooled = defaultdict(list)
    for fr in fold_results:
        for cond, stats in fr["condition_stats"].items():
            if stats["mean"] is not None:
                pooled[cond].append(stats["mean"])

    # Collect per-pair statistics across folds
    pair_ks = defaultdict(list)
    pair_d = defaultdict(list)
    pair_p = defaultdict(list)
    for fr in fold_results:
        for key, vals in fr["ks_tests"].items():
            pair_ks[key].append(vals["ks_stat"])
            pair_p[key].append(vals["p_value"])
        for key, d in fr["cohens_d"].items():
            pair_d[key].append(d)

    agg = {
        "n_folds": len(fold_results),
        "condition_means_across_folds": {
            cond: {
                "mean_of_means": float(np.mean(vals)),
                "std_of_means": float(np.std(vals)),
            }
            for cond, vals in pooled.items()
        },
        "pairwise_tests": {},
    }

    for key in pair_ks:
        agg["pairwise_tests"][key] = {
            "mean_ks": float(np.mean(pair_ks[key])),
            "std_ks": float(np.std(pair_ks[key])),
            "mean_p": float(np.mean(pair_p[key])),
            "mean_cohens_d": float(np.mean(pair_d[key])),
            "std_cohens_d": float(np.std(pair_d[key])),
            "per_fold_ks": pair_ks[key],
            "per_fold_d": pair_d[key],
        }

    save_results(agg, output_dir)
    return agg


def generate_figures(agg_results: dict, output_dir: Path, fold_results: list):
    """Generate overlapping histogram of S_mean per condition."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping figures")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Collect all per-window scores across folds for the histogram
    all_scores = defaultdict(list)
    for fr in fold_results:
        fold = fr["fold"]
        fold_dir = output_dir / f"fold_{fold}"
        scores_path = fold_dir / "condition_scores.pt"
        if scores_path.exists():
            cond_scores = torch.load(scores_path, weights_only=False)
            for cond, vals in cond_scores.items():
                all_scores[cond].extend(vals)

    if not all_scores:
        logger.warning("No condition scores found, skipping histogram")
        return

    colors = {"air-cut": "#1f77b4", "active-cut": "#ff7f0e", "damage": "#d62728"}

    # Overlapping histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in CONDITION_ORDER:
        if cond in all_scores:
            vals = np.array(all_scores[cond])
            ax.hist(
                vals, bins=80, alpha=0.45, label=f"{cond} (n={len(vals)})",
                color=colors.get(cond), density=True, edgecolor="none",
            )
    ax.set_xlabel(r"$S_{\mathrm{mean}}$ (window-level NLL)")
    ax.set_ylabel("Density")
    ax.set_title("NLL Distribution by Machining Condition")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, fig_dir, "cross_condition_histogram.pdf")
    plt.close(fig)

    # Box plot
    fig, ax = plt.subplots(figsize=(6, 5))
    data_for_box = []
    labels_for_box = []
    for cond in CONDITION_ORDER:
        if cond in all_scores:
            vals = np.array(all_scores[cond])
            data_for_box.append(vals)
            labels_for_box.append(cond)
    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        c = colors.get(labels_for_box[i], "#999999")
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_ylabel(r"$S_{\mathrm{mean}}$")
    ax.set_title("Cross-Condition NLL Comparison")
    fig.tight_layout()
    save_figure(fig, fig_dir, "cross_condition_boxplot.pdf")
    plt.close(fig)

    logger.info(f"Figures saved to {fig_dir}")


def main():
    parser = get_default_parser("Cross Condition", 7)
    args = parser.parse_args()

    setup_logging("exp07_cross_condition")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Folds: {args.folds}")

    fold_results = []
    for fold in args.folds:
        result = run_fold(fold, args.output_dir)
        fold_results.append(result)

    agg = aggregate(fold_results, args.output_dir)
    generate_figures(agg, args.output_dir, fold_results)

    # Summary
    pairs = agg.get("pairwise_tests", {})
    if pairs:
        max_d_key = max(pairs, key=lambda k: abs(pairs[k]["mean_cohens_d"]))
        max_d = pairs[max_d_key]["mean_cohens_d"]
        finding = f"Largest d={max_d:.2f} ({max_d_key})"
    else:
        finding = "No pairwise tests computed"

    append_to_summary(
        exp_id=7,
        name="Cross Condition",
        status="done",
        auroc="--",
        finding=finding,
        output_dir=BASE_DIR,
    )

    logger.info("Experiment 7 complete.")


if __name__ == "__main__":
    main()
