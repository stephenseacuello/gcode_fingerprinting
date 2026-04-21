#!/usr/bin/env python3
"""Experiment 13: NLL vs Argmax Mismatch Detection Comparison

Compares two anomaly detection strategies:
  - NLL scoring: continuous S_mean per window (from cached nll_scores.pt)
  - Argmax mismatch: binary mismatch_rate = mean(argmax != target) over non-pad positions

At each FPR level (0.01, 0.05, 0.10), compares which method detects more
attacks. Uses McNemar's test to build a 2x2 contingency table at each
threshold. Generates ROC curves for both methods.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import (
    BASE_DIR,
    ATTACK_DIR,
    FOLDS,
    load_cached_logits,
    load_cached_predictions,
    load_cached_targets,
    compute_per_token_nll,
    compute_window_scores,
    compute_auroc,
    bootstrap_auroc_ci,
    compute_detection_at_fpr,
    cohens_d,
    save_results,
    save_figure,
    append_to_summary,
    setup_logging,
    get_default_parser,
)

logger = logging.getLogger(__name__)

FPR_LEVELS = [0.01, 0.05, 0.10]


def _compute_argmax_mismatch_rate(
    logits: torch.Tensor,
    targets: torch.Tensor,
    padding_mask: torch.Tensor,
    lengths: torch.Tensor,
) -> np.ndarray:
    """Compute per-window argmax mismatch rate.

    mismatch_rate = mean(argmax(logits) != target) over non-pad positions.

    Returns:
        mismatch_rates: [N] per-window mismatch rate
    """
    preds = logits.argmax(dim=-1)  # [N, L]
    mismatches = (preds != targets).float()
    mismatches = mismatches.masked_fill(padding_mask, 0.0)

    lengths_float = lengths.float().clamp(min=1)
    mismatch_rate = mismatches.sum(dim=1) / lengths_float  # [N]

    return mismatch_rate.numpy()


def _mcnemar_test(
    nll_correct: np.ndarray,
    argmax_correct: np.ndarray,
) -> dict:
    """McNemar's test comparing two binary classifiers.

    Builds 2x2 contingency:
      b = NLL correct, argmax wrong
      c = NLL wrong, argmax correct
    """
    b = int(np.sum((nll_correct == 1) & (argmax_correct == 0)))
    c = int(np.sum((nll_correct == 0) & (argmax_correct == 1)))

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}

    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(chi2.sf(stat, df=1))

    return {"statistic": float(stat), "p_value": p_value, "b": b, "c": c}


def _score_attack(
    logits: torch.Tensor,
    targets: torch.Tensor,
    padding_mask: torch.Tensor,
    lengths: torch.Tensor,
    attacked_tokens: torch.Tensor,
    orig_indices: torch.Tensor,
) -> dict:
    """Compare NLL vs argmax detection for one attack."""
    # --- Argmax mismatch scores ---
    normal_mismatch = _compute_argmax_mismatch_rate(logits, targets, padding_mask, lengths)

    attack_logits = logits[orig_indices]
    attack_mask = padding_mask[orig_indices]
    attack_lengths = lengths[orig_indices]
    attack_mismatch = _compute_argmax_mismatch_rate(
        attack_logits, attacked_tokens, attack_mask, attack_lengths
    )
    normal_mismatch_subset = normal_mismatch[orig_indices.numpy()]

    # --- NLL scores ---
    nll_normal = compute_per_token_nll(logits, targets, padding_mask)
    normal_nll_window = compute_window_scores(nll_normal, lengths)["S_mean"].numpy()

    nll_attack = compute_per_token_nll(attack_logits, attacked_tokens, attack_mask)
    attack_nll_window = compute_window_scores(nll_attack, attack_lengths)["S_mean"].numpy()
    normal_nll_subset = normal_nll_window[orig_indices.numpy()]

    # --- Detection metrics ---
    results = {"argmax": {}, "nll": {}}

    for method, normal_scores, attack_scores in [
        ("argmax", normal_mismatch_subset, attack_mismatch),
        ("nll", normal_nll_subset, attack_nll_window),
    ]:
        try:
            auroc_info = bootstrap_auroc_ci(normal_scores, attack_scores, n_bootstrap=5000)
            det = {}
            for fpr in FPR_LEVELS:
                det[f"fpr_{fpr:.2f}"] = compute_detection_at_fpr(
                    normal_scores, attack_scores, fpr
                )
            d = cohens_d(attack_scores, normal_scores)
        except (ValueError, ZeroDivisionError):
            auroc_info = {"auroc": 0.5, "ci_lower": 0.0, "ci_upper": 1.0, "std": 0.0}
            det = {f"fpr_{fpr:.2f}": 0.0 for fpr in FPR_LEVELS}
            d = 0.0

        results[method] = {
            "auroc": auroc_info["auroc"],
            "auroc_ci": [auroc_info["ci_lower"], auroc_info["ci_upper"]],
            "detection_at_fpr": det,
            "cohens_d": d,
            "n_normal": len(normal_scores),
            "n_attack": len(attack_scores),
        }

    # --- McNemar's test at each FPR ---
    mcnemar_results = {}
    for fpr in FPR_LEVELS:
        fpr_key = f"fpr_{fpr:.2f}"
        nll_thresh = np.percentile(normal_nll_subset, 100 * (1 - fpr))
        argmax_thresh = np.percentile(normal_mismatch_subset, 100 * (1 - fpr))

        # Build combined labels and predictions
        all_nll = np.concatenate([normal_nll_subset, attack_nll_window])
        all_argmax = np.concatenate([normal_mismatch_subset, attack_mismatch])
        all_labels = np.concatenate([
            np.zeros(len(normal_nll_subset)),
            np.ones(len(attack_nll_window)),
        ])

        nll_pred_attack = (all_nll > nll_thresh).astype(int)
        argmax_pred_attack = (all_argmax > argmax_thresh).astype(int)

        nll_correct = (nll_pred_attack == all_labels).astype(int)
        argmax_correct = (argmax_pred_attack == all_labels).astype(int)

        mcnemar = _mcnemar_test(nll_correct, argmax_correct)
        mcnemar_results[fpr_key] = mcnemar

    results["mcnemar_tests"] = mcnemar_results
    return results


def run_fold(fold: int, output_dir: Path) -> dict:
    """Run Experiment 13 for one fold."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Fold {fold} ---")

    logits_data = load_cached_logits(fold)
    targets_data = load_cached_targets(fold)

    legacy_logits = logits_data["legacy_logits"]
    target_tokens = targets_data["target_tokens"]
    padding_mask = targets_data["padding_mask"]
    lengths = targets_data["lengths"]

    # Score attacks
    attack_dir = ATTACK_DIR
    attack_results = {}

    attack_files = {
        "A1_command_swap": "attack_A1_command_swap.pt",
        "A3_param_swap": "attack_A3_param_swap.pt",
        "A4_cross_op": "attack_A4_cross_op.pt",
    }

    for attack_name, attack_file in attack_files.items():
        attack_path = attack_dir / attack_file
        if not attack_path.exists():
            continue
        attack_data = torch.load(attack_path, weights_only=False)
        fold_data = attack_data.get("fold_attacks", {}).get(fold, {})
        if fold_data.get("n_attacks", 0) == 0:
            continue

        logger.info(f"  Scoring {attack_name}...")
        attack_results[attack_name] = _score_attack(
            legacy_logits, target_tokens, padding_mask, lengths,
            fold_data["attacked_tokens"], fold_data["orig_indices"],
        )

    save_results(attack_results, fold_dir, "nll_vs_argmax_results.json")

    # Per-attack summary
    summary = {}
    for attack, res in attack_results.items():
        mcnemar_05 = res.get("mcnemar_tests", {}).get("fpr_0.05", {})
        summary[attack] = {
            "nll_auroc": res["nll"]["auroc"],
            "argmax_auroc": res["argmax"]["auroc"],
            "mcnemar_p_fpr05": mcnemar_05.get("p_value", 1.0),
        }

    return {
        "fold": fold,
        "n_windows": len(target_tokens),
        "attack_results": attack_results,
        "summary": summary,
    }


def aggregate(fold_results: list, output_dir: Path) -> dict:
    """Cross-fold aggregation."""
    attack_types = set()
    for fr in fold_results:
        attack_types.update(fr["summary"].keys())

    agg = {}
    for attack in sorted(attack_types):
        nll_aurocs = []
        argmax_aurocs = []
        mcnemar_ps = []
        for fr in fold_results:
            if attack in fr["summary"]:
                s = fr["summary"][attack]
                nll_aurocs.append(s["nll_auroc"])
                argmax_aurocs.append(s["argmax_auroc"])
                mcnemar_ps.append(s["mcnemar_p_fpr05"])

        if nll_aurocs:
            agg[attack] = {
                "nll_auroc": {
                    "mean": float(np.mean(nll_aurocs)),
                    "std": float(np.std(nll_aurocs)),
                    "per_fold": nll_aurocs,
                },
                "argmax_auroc": {
                    "mean": float(np.mean(argmax_aurocs)),
                    "std": float(np.std(argmax_aurocs)),
                    "per_fold": argmax_aurocs,
                },
                "nll_better": float(np.mean(nll_aurocs)) > float(np.mean(argmax_aurocs)),
                "mcnemar_p_values": mcnemar_ps,
            }

    # Detection rate comparison
    detection_comparison = {}
    for fpr in FPR_LEVELS:
        fpr_key = f"fpr_{fpr:.2f}"
        for attack in sorted(attack_types):
            nll_dets = []
            argmax_dets = []
            for fr in fold_results:
                ar = fr.get("attack_results", {}).get(attack, {})
                nll_det = ar.get("nll", {}).get("detection_at_fpr", {}).get(fpr_key)
                argmax_det = ar.get("argmax", {}).get("detection_at_fpr", {}).get(fpr_key)
                if nll_det is not None:
                    nll_dets.append(nll_det)
                if argmax_det is not None:
                    argmax_dets.append(argmax_det)
            if nll_dets:
                if attack not in detection_comparison:
                    detection_comparison[attack] = {}
                detection_comparison[attack][fpr_key] = {
                    "nll_mean": float(np.mean(nll_dets)),
                    "argmax_mean": float(np.mean(argmax_dets)),
                }

    results = {
        "per_attack": agg,
        "detection_at_fpr": detection_comparison,
        "n_folds": len(fold_results),
    }

    save_results(results, output_dir)
    return results


def generate_figures(agg_results: dict, output_dir: Path):
    """Generate ROC and comparison figures."""
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

    per_attack = agg_results.get("per_attack", {})
    if not per_attack:
        return

    main_attacks = sorted(per_attack.keys())

    # Figure 1: AUROC comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(main_attacks))
    width = 0.35

    nll_means = [per_attack[a]["nll_auroc"]["mean"] for a in main_attacks]
    nll_stds = [per_attack[a]["nll_auroc"]["std"] for a in main_attacks]
    argmax_means = [per_attack[a]["argmax_auroc"]["mean"] for a in main_attacks]
    argmax_stds = [per_attack[a]["argmax_auroc"]["std"] for a in main_attacks]

    ax.bar(x - width / 2, nll_means, width, yerr=nll_stds,
           label="NLL (continuous)", capsize=3, color="#55A868")
    ax.bar(x + width / 2, argmax_means, width, yerr=argmax_stds,
           label="Argmax mismatch (binary)", capsize=3, color="#C44E52")

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("AUROC")
    ax.set_title("NLL vs Argmax Mismatch Detection")
    ax.set_xticks(x)
    ax.set_xticklabels(main_attacks, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, fig_dir, "nll_vs_argmax_auroc.pdf")
    plt.close(fig)

    # Figure 2: Detection rate at FPR levels
    det_comp = agg_results.get("detection_at_fpr", {})
    if det_comp and main_attacks:
        fig, axes = plt.subplots(1, len(FPR_LEVELS), figsize=(5 * len(FPR_LEVELS), 4))
        if len(FPR_LEVELS) == 1:
            axes = [axes]

        for ax, fpr in zip(axes, FPR_LEVELS):
            fpr_key = f"fpr_{fpr:.2f}"
            plot_attacks = [a for a in main_attacks if a in det_comp and fpr_key in det_comp[a]]
            if not plot_attacks:
                continue
            xx = np.arange(len(plot_attacks))
            nll_vals = [det_comp[a][fpr_key]["nll_mean"] for a in plot_attacks]
            argmax_vals = [det_comp[a][fpr_key]["argmax_mean"] for a in plot_attacks]

            ax.bar(xx - 0.175, nll_vals, 0.35, label="NLL", color="#55A868")
            ax.bar(xx + 0.175, argmax_vals, 0.35, label="Argmax", color="#C44E52")
            ax.set_title(f"Detection Rate @ FPR={fpr}")
            ax.set_xticks(xx)
            ax.set_xticklabels(plot_attacks, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("TPR")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)

        fig.suptitle("Detection Rate at Matched FPR Levels", y=1.02)
        fig.tight_layout()
        save_figure(fig, fig_dir, "detection_at_fpr.pdf")
        plt.close(fig)

    logger.info(f"Figures saved to {fig_dir}")


def main():
    parser = get_default_parser("NLL vs Argmax", 13)
    args = parser.parse_args()

    setup_logging("exp13_nll_vs_argmax")

    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Folds: {args.folds}")

    fold_results = []
    for fold in args.folds:
        result = run_fold(fold, args.output_dir)
        fold_results.append(result)

    agg = aggregate(fold_results, args.output_dir)
    generate_figures(agg, args.output_dir)

    # Update summary
    per_attack = agg.get("per_attack", {})
    if per_attack:
        best_advantage = -1
        best_attack = ""
        for attack, data in per_attack.items():
            advantage = data["nll_auroc"]["mean"] - data["argmax_auroc"]["mean"]
            if advantage > best_advantage:
                best_advantage = advantage
                best_attack = attack

        nll_auroc = per_attack[best_attack]["nll_auroc"]["mean"]
        argmax_auroc = per_attack[best_attack]["argmax_auroc"]["mean"]
        finding = (
            f"NLL AUROC={nll_auroc:.3f} vs argmax={argmax_auroc:.3f} "
            f"on {best_attack} (delta={best_advantage:+.3f})"
        )
        auroc_str = f"NLL={nll_auroc:.3f}"
    else:
        finding = "No attacks processed"
        auroc_str = "--"

    append_to_summary(
        exp_id=13,
        name="NLL vs Argmax",
        status="done",
        auroc=auroc_str,
        finding=finding,
        output_dir=BASE_DIR,
    )

    logger.info("Experiment 13 complete.")


if __name__ == "__main__":
    main()
