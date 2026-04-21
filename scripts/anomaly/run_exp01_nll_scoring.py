#!/usr/bin/env python3
"""Experiment 1: NLL-Based Anomaly Scoring (CORE)

Computes per-token NLL as continuous anomaly score. Evaluates S_mean, S_max,
S_sum, and perplexity on normal vs attacked windows for each attack type.
Reports AUROC with bootstrap CIs, Cohen's d, and detection rates at FPR thresholds.

This is the prerequisite experiment — most other experiments depend on its outputs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import (
    ATTACK_DIR,
    BASE_DIR,
    FOLDS,
    load_cached_logits,
    load_cached_nll,
    load_cached_targets,
    compute_per_token_nll,
    compute_window_scores,
    compute_auroc,
    compute_detection_at_fpr,
    bootstrap_auroc_ci,
    cohens_d,
    save_results,
    save_figure,
    append_to_summary,
    setup_logging,
    get_default_parser,
)

import logging

logger = logging.getLogger(__name__)


def run_fold(fold: int, output_dir: Path) -> dict:
    """Run Experiment 1 for one fold."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Fold {fold} ---")

    # Load cached data
    logits = load_cached_logits(fold)
    targets_data = load_cached_targets(fold)

    target_tokens = targets_data["target_tokens"]
    padding_mask = targets_data["padding_mask"]
    lengths = targets_data["lengths"]
    op_types = targets_data["operation_type"]

    legacy_logits = logits["legacy_logits"]

    # Compute NLL on normal (ground truth) sequences
    nll_normal = compute_per_token_nll(legacy_logits, target_tokens, padding_mask)
    normal_scores = compute_window_scores(nll_normal, lengths)

    # Save normal scores
    normal_results = {
        scoring: normal_scores[scoring].numpy().tolist()
        for scoring in ["S_mean", "S_max", "S_sum", "perplexity"]
    }
    save_results(normal_results, fold_dir, "normal_scores.json")

    # Load pre-generated attacks
    attack_dir = ATTACK_DIR
    attack_results = {}

    # A1: Command swap
    a1 = torch.load(attack_dir / "attack_A1_command_swap.pt", weights_only=False)
    if fold in a1["fold_attacks"] and a1["fold_attacks"][fold]["n_attacks"] > 0:
        a1_data = a1["fold_attacks"][fold]
        attack_results["A1_command_swap"] = _score_attack(
            legacy_logits, target_tokens, padding_mask, lengths,
            a1_data["attacked_tokens"], a1_data["orig_indices"],
            normal_scores,
        )

    # A2: Coordinate tampering (multiple deltas)
    a2 = torch.load(attack_dir / "attack_A2_coord_graded.pt", weights_only=False)
    for key, data in a2.get("attacks_by_delta_axis", {}).items():
        fold_data = data["fold_attacks"].get(fold, {})
        if fold_data.get("n_attacks", 0) > 0:
            attack_results[f"A2_{key}"] = _score_attack(
                legacy_logits, target_tokens, padding_mask, lengths,
                fold_data["attacked_tokens"], fold_data["orig_indices"],
                normal_scores,
            )

    # A3: Param swap
    a3_path = attack_dir / "attack_A3_param_swap.pt"
    if a3_path.exists():
        a3 = torch.load(a3_path, weights_only=False)
        if fold in a3["fold_attacks"] and a3["fold_attacks"][fold]["n_attacks"] > 0:
            a3_data = a3["fold_attacks"][fold]
            attack_results["A3_param_swap"] = _score_attack(
                legacy_logits, target_tokens, padding_mask, lengths,
                a3_data["attacked_tokens"], a3_data["orig_indices"],
                normal_scores,
            )

    # A4: Cross-operation swap
    a4_path = attack_dir / "attack_A4_cross_op.pt"
    if a4_path.exists():
        a4 = torch.load(a4_path, weights_only=False)
        if fold in a4["fold_attacks"] and a4["fold_attacks"][fold]["n_attacks"] > 0:
            a4_data = a4["fold_attacks"][fold]
            attack_results["A4_cross_op"] = _score_attack(
                legacy_logits, target_tokens, padding_mask, lengths,
                a4_data["attacked_tokens"], a4_data["orig_indices"],
                normal_scores,
            )

    save_results(attack_results, fold_dir, "attack_scores.json")

    return {
        "fold": fold,
        "n_windows": len(target_tokens),
        "normal_scores_summary": {
            scoring: {
                "mean": float(normal_scores[scoring].mean()),
                "std": float(normal_scores[scoring].std()),
            }
            for scoring in ["S_mean", "S_max", "S_sum", "perplexity"]
        },
        "attack_results": attack_results,
    }


def _score_attack(
    logits, targets, mask, lengths,
    attacked_tokens, orig_indices, normal_scores,
) -> dict:
    """Score an attack variant against normal scores."""
    results = {}
    for scoring in ["S_mean", "S_max", "S_sum", "perplexity"]:
        # Normal scores for the windows that were attacked
        normal_subset = normal_scores[scoring][orig_indices].numpy()

        # Attack scores: NLL of attacked tokens under same logit distribution
        attack_logits = logits[orig_indices]
        attack_mask = mask[orig_indices]
        attack_lengths = lengths[orig_indices]

        nll_attack = compute_per_token_nll(attack_logits, attacked_tokens, attack_mask)
        attack_window = compute_window_scores(nll_attack, attack_lengths)
        attack_subset = attack_window[scoring].numpy()

        # Compute metrics
        try:
            auroc_info = bootstrap_auroc_ci(normal_subset, attack_subset, n_bootstrap=5000)
            det_01 = compute_detection_at_fpr(normal_subset, attack_subset, 0.01)
            det_05 = compute_detection_at_fpr(normal_subset, attack_subset, 0.05)
            det_10 = compute_detection_at_fpr(normal_subset, attack_subset, 0.10)
            d = cohens_d(attack_subset, normal_subset)
        except (ValueError, ZeroDivisionError):
            auroc_info = {"auroc": 0.5, "ci_lower": 0.0, "ci_upper": 1.0, "std": 0.0}
            det_01 = det_05 = det_10 = 0.0
            d = 0.0

        results[scoring] = {
            "auroc": auroc_info["auroc"],
            "auroc_ci": [auroc_info["ci_lower"], auroc_info["ci_upper"]],
            "detection_at_fpr_01": det_01,
            "detection_at_fpr_05": det_05,
            "detection_at_fpr_10": det_10,
            "cohens_d": d,
            "n_normal": len(normal_subset),
            "n_attack": len(attack_subset),
        }

    return results


def aggregate(fold_results: list, output_dir: Path) -> dict:
    """Cross-fold aggregation."""
    # Collect AUROC per attack type per scoring method
    attack_types = set()
    for fr in fold_results:
        attack_types.update(fr["attack_results"].keys())

    agg = {}
    for attack in sorted(attack_types):
        agg[attack] = {}
        for scoring in ["S_mean", "S_max", "S_sum", "perplexity"]:
            aurocs = []
            for fr in fold_results:
                if attack in fr["attack_results"]:
                    ar = fr["attack_results"][attack]
                    if scoring in ar:
                        aurocs.append(ar[scoring]["auroc"])
            if aurocs:
                agg[attack][scoring] = {
                    "mean_auroc": float(np.mean(aurocs)),
                    "std_auroc": float(np.std(aurocs)),
                    "per_fold": aurocs,
                }

    # Find best scoring method
    best = {}
    for attack in agg:
        best_score = 0
        best_method = "S_mean"
        for scoring in agg[attack]:
            if agg[attack][scoring]["mean_auroc"] > best_score:
                best_score = agg[attack][scoring]["mean_auroc"]
                best_method = scoring
        best[attack] = {"method": best_method, "auroc": best_score}

    results = {
        "per_attack": agg,
        "best_scoring_per_attack": best,
        "n_folds": len(fold_results),
    }

    save_results(results, output_dir)
    return results


def generate_figures(agg_results: dict, output_dir: Path):
    """Generate paper-ready figures."""
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

    # Figure 1: AUROC comparison across scoring methods and attack types
    attacks = sorted(agg_results["per_attack"].keys())
    scorings = ["S_mean", "S_max", "S_sum", "perplexity"]

    # Filter to main attack types (not per-delta)
    main_attacks = [a for a in attacks if "delta" not in a]

    if main_attacks:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(main_attacks))
        width = 0.2
        for i, scoring in enumerate(scorings):
            vals = []
            errs = []
            for attack in main_attacks:
                if scoring in agg_results["per_attack"].get(attack, {}):
                    vals.append(agg_results["per_attack"][attack][scoring]["mean_auroc"])
                    errs.append(agg_results["per_attack"][attack][scoring]["std_auroc"])
                else:
                    vals.append(0)
                    errs.append(0)
            ax.bar(x + i * width, vals, width, yerr=errs, label=scoring, capsize=3)

        ax.set_xlabel("Attack Type")
        ax.set_ylabel("AUROC")
        ax.set_title("NLL-Based Anomaly Detection by Attack Type and Scoring Method")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(main_attacks, rotation=30, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        save_figure(fig, fig_dir, "nll_auroc_by_attack.pdf")
        plt.close(fig)

    logger.info(f"Figures saved to {fig_dir}")


def main():
    parser = get_default_parser("NLL Scoring", 1)
    args = parser.parse_args()

    setup_logging("exp01_nll")

    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Folds: {args.folds}")

    fold_results = []
    for fold in args.folds:
        result = run_fold(fold, args.output_dir)
        fold_results.append(result)

    agg = aggregate(fold_results, args.output_dir)
    generate_figures(agg, args.output_dir)

    # Update summary
    best = agg.get("best_scoring_per_attack", {})
    if best:
        top = max(best.values(), key=lambda x: x["auroc"])
        finding = f"Best: {top['method']} at {top['auroc']:.3f} AUROC"
    else:
        finding = "No attacks generated"

    append_to_summary(
        exp_id=1,
        name="NLL Scoring",
        status="done",
        auroc=finding.split("at ")[-1] if "at" in finding else "--",
        finding=finding,
        output_dir=BASE_DIR,
    )

    logger.info("Experiment 1 complete.")


if __name__ == "__main__":
    main()
