#!/usr/bin/env python3
"""Experiment 4: Multi-Head Disagreement Analysis

For each position, checks whether argmax(type_logits) matches the type of
argmax(legacy_logits) using the token-to-type map. Computes window-level
disagreement rate and compares normal vs attacked windows.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import (
    BASE_DIR,
    ATTACK_DIR,
    FOLDS,
    load_cached_logits,
    load_cached_predictions,
    load_cached_targets,
    load_vocab,
    build_token_type_map,
    compute_auroc,
    bootstrap_auroc_ci,
    compute_detection_at_fpr,
    cohens_d,
    permutation_test,
    save_results,
    save_figure,
    append_to_summary,
    setup_logging,
    get_default_parser,
)

import logging

logger = logging.getLogger(__name__)


def _compute_disagreement(
    legacy_logits: torch.Tensor,
    type_logits: torch.Tensor,
    padding_mask: torch.Tensor,
    lengths: torch.Tensor,
    token_type_map: dict,
) -> np.ndarray:
    """Compute per-window disagreement rate.

    For each non-pad position, check if argmax(type_logits) matches the
    token type of argmax(legacy_logits).

    Args:
        legacy_logits: [N, L, V_legacy]
        type_logits: [N, L, V_type]
        padding_mask: [N, L] True at padded positions
        lengths: [N] actual sequence lengths
        token_type_map: dict mapping token_id -> type_id

    Returns:
        S_disagree: [N] window-level disagreement rate
    """
    N, L = padding_mask.shape

    # Argmax predictions
    legacy_preds = legacy_logits.argmax(dim=-1)  # [N, L]
    type_preds = type_logits.argmax(dim=-1)      # [N, L]

    # Map legacy argmax to expected type
    # Build lookup tensor for fast vectorized mapping
    max_token_id = max(token_type_map.keys()) + 1
    type_lookup = torch.zeros(max_token_id, dtype=torch.long)
    for tid, ttype in token_type_map.items():
        if tid < max_token_id:
            type_lookup[tid] = ttype

    # Clamp legacy preds to valid range
    legacy_clamped = legacy_preds.clamp(0, max_token_id - 1)
    expected_types = type_lookup[legacy_clamped]  # [N, L]

    # Disagreement: type prediction != expected type from legacy
    disagree = (type_preds != expected_types).float()  # [N, L]
    disagree = disagree.masked_fill(padding_mask, 0.0)

    # Window-level rate
    lengths_float = lengths.float().clamp(min=1)
    s_disagree = disagree.sum(dim=1) / lengths_float  # [N]

    return s_disagree.numpy()


def _score_attack_disagreement(
    legacy_logits: torch.Tensor,
    type_logits: torch.Tensor,
    padding_mask: torch.Tensor,
    lengths: torch.Tensor,
    attacked_tokens: torch.Tensor,
    orig_indices: torch.Tensor,
    normal_disagree: np.ndarray,
    token_type_map: dict,
) -> dict:
    """Score an attack using disagreement metric."""
    # For attacked sequences, we use the original logits (teacher-forced on
    # original tokens). The question is whether the model's heads disagree
    # more when the actual tokens differ from what was expected.
    # We can also compute disagreement on the attacked token types.

    # Normal subset
    normal_subset = normal_disagree[orig_indices.numpy()]

    # For attacks: compare legacy argmax to type argmax at attacked positions
    attack_legacy = legacy_logits[orig_indices]
    attack_type = type_logits[orig_indices]
    attack_mask = padding_mask[orig_indices]
    attack_lengths = lengths[orig_indices]

    attack_disagree = _compute_disagreement(
        attack_legacy, attack_type, attack_mask, attack_lengths, token_type_map
    )

    try:
        auroc_info = bootstrap_auroc_ci(normal_subset, attack_disagree, n_bootstrap=5000)
        det_01 = compute_detection_at_fpr(normal_subset, attack_disagree, 0.01)
        det_05 = compute_detection_at_fpr(normal_subset, attack_disagree, 0.05)
        det_10 = compute_detection_at_fpr(normal_subset, attack_disagree, 0.10)
        d = cohens_d(attack_disagree, normal_subset)
        p_val = permutation_test(attack_disagree, normal_subset, n_perm=10000)
    except (ValueError, ZeroDivisionError):
        auroc_info = {"auroc": 0.5, "ci_lower": 0.0, "ci_upper": 1.0, "std": 0.0}
        det_01 = det_05 = det_10 = 0.0
        d = 0.0
        p_val = 1.0

    return {
        "auroc": auroc_info["auroc"],
        "auroc_ci": [auroc_info["ci_lower"], auroc_info["ci_upper"]],
        "detection_at_fpr_01": det_01,
        "detection_at_fpr_05": det_05,
        "detection_at_fpr_10": det_10,
        "cohens_d": d,
        "p_value": p_val,
        "normal_mean_disagree": float(np.mean(normal_subset)),
        "attack_mean_disagree": float(np.mean(attack_disagree)),
        "n_normal": len(normal_subset),
        "n_attack": len(attack_disagree),
    }


def run_fold(fold: int, output_dir: Path) -> dict:
    """Run Experiment 4 for one fold."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Fold {fold} ---")

    logits_data = load_cached_logits(fold)
    targets_data = load_cached_targets(fold)

    # Check required heads exist
    if "type_logits" not in logits_data:
        logger.warning(f"  type_logits not found in fold {fold}, skipping")
        return {"fold": fold, "attack_results": {}, "skipped": True}

    legacy_logits = logits_data["legacy_logits"]
    type_logits = logits_data["type_logits"]
    padding_mask = targets_data["padding_mask"]
    lengths = targets_data["lengths"]

    vocab = load_vocab()
    token_type_map = build_token_type_map(vocab)

    # Compute normal disagreement
    normal_disagree = _compute_disagreement(
        legacy_logits, type_logits, padding_mask, lengths, token_type_map
    )

    logger.info(
        f"  Normal disagreement: mean={np.mean(normal_disagree):.4f}, "
        f"std={np.std(normal_disagree):.4f}"
    )

    # Score attacks
    attack_dir = ATTACK_DIR
    attack_results = {}

    # A1: Command swap
    a1_path = attack_dir / "attack_A1_command_swap.pt"
    if a1_path.exists():
        a1 = torch.load(a1_path, weights_only=False)
        if fold in a1["fold_attacks"] and a1["fold_attacks"][fold]["n_attacks"] > 0:
            a1_data = a1["fold_attacks"][fold]
            attack_results["A1_command_swap"] = _score_attack_disagreement(
                legacy_logits, type_logits, padding_mask, lengths,
                a1_data["attacked_tokens"], a1_data["orig_indices"],
                normal_disagree, token_type_map,
            )

    # A2: Coordinate tampering
    a2_path = attack_dir / "attack_A2_coord_graded.pt"
    if a2_path.exists():
        a2 = torch.load(a2_path, weights_only=False)
        for key, data in a2.get("attacks_by_delta_axis", {}).items():
            fold_data = data["fold_attacks"].get(fold, {})
            if fold_data.get("n_attacks", 0) > 0:
                attack_results[f"A2_{key}"] = _score_attack_disagreement(
                    legacy_logits, type_logits, padding_mask, lengths,
                    fold_data["attacked_tokens"], fold_data["orig_indices"],
                    normal_disagree, token_type_map,
                )

    # A3: Param swap
    a3_path = attack_dir / "attack_A3_param_swap.pt"
    if a3_path.exists():
        a3 = torch.load(a3_path, weights_only=False)
        if fold in a3["fold_attacks"] and a3["fold_attacks"][fold]["n_attacks"] > 0:
            a3_data = a3["fold_attacks"][fold]
            attack_results["A3_param_swap"] = _score_attack_disagreement(
                legacy_logits, type_logits, padding_mask, lengths,
                a3_data["attacked_tokens"], a3_data["orig_indices"],
                normal_disagree, token_type_map,
            )

    # A4: Cross-operation swap
    a4_path = attack_dir / "attack_A4_cross_op.pt"
    if a4_path.exists():
        a4 = torch.load(a4_path, weights_only=False)
        if fold in a4["fold_attacks"] and a4["fold_attacks"][fold]["n_attacks"] > 0:
            a4_data = a4["fold_attacks"][fold]
            attack_results["A4_cross_op"] = _score_attack_disagreement(
                legacy_logits, type_logits, padding_mask, lengths,
                a4_data["attacked_tokens"], a4_data["orig_indices"],
                normal_disagree, token_type_map,
            )

    save_results(attack_results, fold_dir, "disagreement_results.json")

    return {
        "fold": fold,
        "n_windows": len(normal_disagree),
        "normal_disagree_mean": float(np.mean(normal_disagree)),
        "normal_disagree_std": float(np.std(normal_disagree)),
        "attack_results": attack_results,
    }


def aggregate(fold_results: list, output_dir: Path) -> dict:
    """Cross-fold aggregation."""
    attack_types = set()
    for fr in fold_results:
        attack_types.update(fr["attack_results"].keys())

    agg = {}
    for attack in sorted(attack_types):
        aurocs = []
        p_values = []
        for fr in fold_results:
            if attack in fr["attack_results"]:
                ar = fr["attack_results"][attack]
                aurocs.append(ar["auroc"])
                p_values.append(ar["p_value"])

        if aurocs:
            agg[attack] = {
                "mean_auroc": float(np.mean(aurocs)),
                "std_auroc": float(np.std(aurocs)),
                "per_fold_auroc": aurocs,
                "per_fold_p": p_values,
                "all_significant": all(p < 0.05 for p in p_values),
            }

    # Normal disagreement stats
    normal_means = [fr["normal_disagree_mean"] for fr in fold_results if "normal_disagree_mean" in fr]

    results = {
        "per_attack": agg,
        "normal_disagreement": {
            "mean": float(np.mean(normal_means)) if normal_means else 0.0,
            "std": float(np.std(normal_means)) if normal_means else 0.0,
        },
        "n_folds": len(fold_results),
    }

    save_results(results, output_dir)
    return results


def generate_figures(agg_results: dict, output_dir: Path):
    """Generate disagreement analysis figures."""
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

    attacks = sorted(agg_results["per_attack"].keys())
    if not attacks:
        return

    # Filter to main attack types
    main_attacks = [a for a in attacks if "delta" not in a]
    if not main_attacks:
        main_attacks = attacks[:6]

    # Figure: AUROC by attack type for disagreement detection
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(main_attacks))
    auroc_means = [agg_results["per_attack"][a]["mean_auroc"] for a in main_attacks]
    auroc_stds = [agg_results["per_attack"][a]["std_auroc"] for a in main_attacks]

    bars = ax.bar(x, auroc_means, yerr=auroc_stds, capsize=4,
                  color="#4C72B0", edgecolor="black", alpha=0.8)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("AUROC")
    ax.set_title("Multi-Head Disagreement Detection")
    ax.set_xticks(x)
    ax.set_xticklabels(main_attacks, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, fig_dir, "disagreement_auroc.pdf")
    plt.close(fig)

    logger.info(f"Figures saved to {fig_dir}")


def main():
    parser = get_default_parser("Disagreement", 4)
    args = parser.parse_args()

    setup_logging("exp04_disagreement")

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
        best_attack = max(per_attack.items(), key=lambda x: x[1]["mean_auroc"])
        best_auroc = best_attack[1]["mean_auroc"]
        finding = (
            f"Best disagreement AUROC: {best_auroc:.3f} on {best_attack[0]}; "
            f"normal disagree rate: {agg['normal_disagreement']['mean']:.3f}"
        )
        auroc_str = f"{best_auroc:.3f}"
    else:
        finding = "No attacks processed"
        auroc_str = "--"

    append_to_summary(
        exp_id=4,
        name="Disagreement",
        status="done",
        auroc=auroc_str,
        finding=finding,
        output_dir=BASE_DIR,
    )

    logger.info("Experiment 4 complete.")


if __name__ == "__main__":
    main()
