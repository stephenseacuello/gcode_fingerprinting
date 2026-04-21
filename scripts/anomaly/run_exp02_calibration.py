#!/usr/bin/env python3
"""Experiment 2: Calibration Analysis

Computes ECE and MCE for legacy, type, command, param_type decoder heads.
Fits temperature scaling on validation logits and compares pre/post calibration.
Generates reliability diagrams (accuracy vs confidence per bin).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import (
    BASE_DIR,
    FOLDS,
    PAD_ID,
    load_cached_logits,
    load_cached_targets,
    compute_ece,
    compute_mce,
    fit_temperature_scaling,
    save_results,
    save_figure,
    append_to_summary,
    setup_logging,
    get_default_parser,
)

import logging

logger = logging.getLogger(__name__)

N_BINS = 15
HEAD_NAMES = ["legacy", "type", "command", "param_type"]


def _extract_head_data(logits_data: dict, targets_data: dict, head: str):
    """Extract logits, targets, and padding mask for a given head.

    Returns:
        flat_confidences: [M] max softmax probability per valid position
        flat_correct: [M] 1 if argmax matches target, else 0
        logits: [N, L, V] raw logits (for temperature scaling)
        targets: [N, L] target token IDs
        mask: [N, L] True at padded positions
    """
    logits_key = f"{head}_logits"
    targets_key = f"{head}_targets" if head != "legacy" else "target_tokens"

    logits = logits_data[logits_key]
    if targets_key in targets_data:
        targets = targets_data[targets_key]
    else:
        targets = targets_data["target_tokens"]

    padding_mask = targets_data["padding_mask"]

    # Compute confidence and correctness at valid positions
    valid = ~padding_mask
    probs = F.softmax(logits, dim=-1)
    confidences, preds = probs.max(dim=-1)  # [N, L]

    flat_conf = confidences[valid].numpy()
    flat_correct = (preds[valid] == targets[valid]).float().numpy()

    return flat_conf, flat_correct, logits, targets, padding_mask


def run_fold(fold: int, output_dir: Path) -> dict:
    """Run Experiment 2 for one fold."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Fold {fold} ---")

    logits_data = load_cached_logits(fold)
    targets_data = load_cached_targets(fold)

    fold_results = {}

    for head in HEAD_NAMES:
        logits_key = f"{head}_logits"
        if logits_key not in logits_data:
            logger.info(f"  Skipping head '{head}' (not in cached logits)")
            continue

        logger.info(f"  Processing head: {head}")
        conf, correct, logits, targets, mask = _extract_head_data(
            logits_data, targets_data, head
        )

        # Pre-calibration metrics
        ece_pre = compute_ece(conf, correct, n_bins=N_BINS)
        mce_pre = compute_mce(conf, correct, n_bins=N_BINS)

        # Reliability diagram data (pre-calibration)
        bin_boundaries = np.linspace(0, 1, N_BINS + 1)
        bin_accs = []
        bin_confs = []
        bin_counts = []
        for i in range(N_BINS):
            bin_mask = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
            count = int(bin_mask.sum())
            bin_counts.append(count)
            if count > 0:
                bin_accs.append(float(correct[bin_mask].mean()))
                bin_confs.append(float(conf[bin_mask].mean()))
            else:
                bin_accs.append(0.0)
                bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)

        # Temperature scaling
        try:
            T = fit_temperature_scaling(logits, targets, mask)
            logger.info(f"    T = {T:.4f}")

            # Post-calibration metrics
            scaled_probs = F.softmax(logits / T, dim=-1)
            valid = ~mask
            scaled_conf = scaled_probs.max(dim=-1).values[valid].numpy()
            scaled_preds = scaled_probs.argmax(dim=-1)
            scaled_correct = (scaled_preds[valid] == targets[valid]).float().numpy()

            ece_post = compute_ece(scaled_conf, scaled_correct, n_bins=N_BINS)
            mce_post = compute_mce(scaled_conf, scaled_correct, n_bins=N_BINS)
        except Exception as e:
            logger.warning(f"    Temperature scaling failed: {e}")
            T = 1.0
            ece_post = ece_pre
            mce_post = mce_pre

        fold_results[head] = {
            "ece_pre": ece_pre,
            "mce_pre": mce_pre,
            "ece_post": ece_post,
            "mce_post": mce_post,
            "temperature": T,
            "n_valid_tokens": int(len(conf)),
            "mean_confidence": float(conf.mean()),
            "mean_accuracy": float(correct.mean()),
            "reliability_bins": {
                "bin_accs": bin_accs,
                "bin_confs": bin_confs,
                "bin_counts": bin_counts,
                "bin_boundaries": bin_boundaries.tolist(),
            },
        }

        logger.info(
            f"    ECE: {ece_pre:.4f} -> {ece_post:.4f} | "
            f"MCE: {mce_pre:.4f} -> {mce_post:.4f} | T={T:.3f}"
        )

    save_results(fold_results, fold_dir, "calibration_results.json")
    return {"fold": fold, "heads": fold_results}


def aggregate(fold_results: list, output_dir: Path) -> dict:
    """Cross-fold aggregation of calibration metrics."""
    agg = {}

    # Collect per-head metrics across folds
    all_heads = set()
    for fr in fold_results:
        all_heads.update(fr["heads"].keys())

    for head in sorted(all_heads):
        ece_pre_vals = []
        ece_post_vals = []
        mce_pre_vals = []
        mce_post_vals = []
        temps = []

        for fr in fold_results:
            if head in fr["heads"]:
                h = fr["heads"][head]
                ece_pre_vals.append(h["ece_pre"])
                ece_post_vals.append(h["ece_post"])
                mce_pre_vals.append(h["mce_pre"])
                mce_post_vals.append(h["mce_post"])
                temps.append(h["temperature"])

        agg[head] = {
            "ece_pre": {
                "mean": float(np.mean(ece_pre_vals)),
                "std": float(np.std(ece_pre_vals)),
                "per_fold": ece_pre_vals,
            },
            "ece_post": {
                "mean": float(np.mean(ece_post_vals)),
                "std": float(np.std(ece_post_vals)),
                "per_fold": ece_post_vals,
            },
            "mce_pre": {
                "mean": float(np.mean(mce_pre_vals)),
                "std": float(np.std(mce_pre_vals)),
                "per_fold": mce_pre_vals,
            },
            "mce_post": {
                "mean": float(np.mean(mce_post_vals)),
                "std": float(np.std(mce_post_vals)),
                "per_fold": mce_post_vals,
            },
            "temperature": {
                "mean": float(np.mean(temps)),
                "std": float(np.std(temps)),
                "per_fold": temps,
            },
        }

    results = {"per_head": agg, "n_folds": len(fold_results)}
    save_results(results, output_dir)
    return results


def generate_figures(agg_results: dict, output_dir: Path):
    """Generate reliability diagrams."""
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

    heads = sorted(agg_results["per_head"].keys())
    if not heads:
        return

    # Figure 1: ECE comparison bar chart (pre vs post calibration)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(heads))
    width = 0.35

    pre_means = [agg_results["per_head"][h]["ece_pre"]["mean"] for h in heads]
    pre_stds = [agg_results["per_head"][h]["ece_pre"]["std"] for h in heads]
    post_means = [agg_results["per_head"][h]["ece_post"]["mean"] for h in heads]
    post_stds = [agg_results["per_head"][h]["ece_post"]["std"] for h in heads]

    ax.bar(x - width / 2, pre_means, width, yerr=pre_stds,
           label="Before temp. scaling", capsize=3, color="#4C72B0")
    ax.bar(x + width / 2, post_means, width, yerr=post_stds,
           label="After temp. scaling", capsize=3, color="#55A868")

    ax.set_xlabel("Decoder Head")
    ax.set_ylabel("ECE")
    ax.set_title("Expected Calibration Error by Decoder Head")
    ax.set_xticks(x)
    ax.set_xticklabels(heads)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, fig_dir, "ece_comparison.pdf")
    plt.close(fig)

    # Figure 2: Reliability diagrams per head (using fold 1 data as representative)
    # Load fold 1 detailed results for bin data
    fold1_path = output_dir / "fold_1" / "calibration_results.json"
    if fold1_path.exists():
        with open(fold1_path) as f:
            fold1_data = json.load(f)

        n_heads = len([h for h in heads if h in fold1_data])
        if n_heads > 0:
            fig, axes = plt.subplots(1, n_heads, figsize=(5 * n_heads, 4))
            if n_heads == 1:
                axes = [axes]

            for ax, head in zip(axes, [h for h in heads if h in fold1_data]):
                bins = fold1_data[head]["reliability_bins"]
                bin_accs = np.array(bins["bin_accs"])
                bin_confs = np.array(bins["bin_confs"])
                bin_counts = np.array(bins["bin_counts"])

                # Only plot bins with samples
                has_data = np.array(bin_counts) > 0
                ax.bar(
                    bin_confs[has_data], bin_accs[has_data],
                    width=1.0 / N_BINS, alpha=0.7, edgecolor="black",
                    label="Observed",
                )
                ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
                ax.set_xlabel("Confidence")
                ax.set_ylabel("Accuracy")
                ax.set_title(f"{head} (ECE={fold1_data[head]['ece_pre']:.3f})")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.legend(fontsize=8)

            fig.suptitle("Reliability Diagrams (Fold 1)", y=1.02)
            fig.tight_layout()
            save_figure(fig, fig_dir, "reliability_diagrams.pdf")
            plt.close(fig)

    logger.info(f"Figures saved to {fig_dir}")


def main():
    parser = get_default_parser("Calibration", 2)
    args = parser.parse_args()

    setup_logging("exp02_calibration")

    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Folds: {args.folds}")

    fold_results = []
    for fold in args.folds:
        result = run_fold(fold, args.output_dir)
        fold_results.append(result)

    agg = aggregate(fold_results, args.output_dir)
    generate_figures(agg, args.output_dir)

    # Update summary
    legacy = agg["per_head"].get("legacy", {})
    if legacy:
        ece_pre = legacy["ece_pre"]["mean"]
        ece_post = legacy["ece_post"]["mean"]
        finding = (
            f"Legacy ECE: {ece_pre:.3f} -> {ece_post:.3f} after temp scaling"
        )
        auroc_str = f"ECE={ece_pre:.3f}"
    else:
        finding = "No heads processed"
        auroc_str = "--"

    append_to_summary(
        exp_id=2,
        name="Calibration",
        status="done",
        auroc=auroc_str,
        finding=finding,
        output_dir=BASE_DIR,
    )

    logger.info("Experiment 2 complete.")


if __name__ == "__main__":
    main()
