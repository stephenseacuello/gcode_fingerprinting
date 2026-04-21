#!/usr/bin/env python3
"""Experiment 10: Leave-One-Class-Out (LOCO) Analysis

For each of 9 operation-type classes, computes class-conditional NLL using
the existing decoder (without retraining). Windows from a held-out class
should have higher NLL than in-distribution windows.

For each class c:
  - Compute S_mean on windows of class c vs all other classes
  - AUROC_LOCO_c = ability to distinguish class c windows by NLL

This tests whether the decoder can identify novel/unseen operation types.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import (
    BASE_DIR,
    FOLDS,
    load_cached_nll,
    load_cached_targets,
    load_decoder,
    load_encoder_memory,
    load_preprocessed_npz,
    build_teacher_forcing_batch,
    compute_per_token_nll,
    compute_window_scores,
    compute_auroc,
    bootstrap_auroc_ci,
    cohens_d,
    save_results,
    save_figure,
    append_to_summary,
    setup_logging,
    get_default_parser,
)

logger = logging.getLogger(__name__)

# Operation type names (9 classes)
OP_TYPE_NAMES = [
    "adaptive", "face", "pocket",
    "adaptive150025", "face150025", "pocket150025",
    "damageadaptive", "damageface", "damagepocket",
]


def run_fold(fold: int, output_dir: Path, device: str) -> dict:
    """Run LOCO analysis for one fold."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Fold {fold} ---")

    # Load cached NLL and targets
    targets_data = load_cached_targets(fold)
    nll_data = load_cached_nll(fold)

    nll_legacy = nll_data["nll_legacy"]
    lengths = targets_data["lengths"]
    op_types = targets_data["operation_type"]

    # Compute window-level scores
    scores = compute_window_scores(nll_legacy, lengths)
    s_mean = scores["S_mean"].numpy()

    op_types_np = op_types.numpy() if isinstance(op_types, torch.Tensor) else np.array(op_types)
    unique_ops = sorted(set(op_types_np.tolist()))

    logger.info(f"  {len(unique_ops)} operation types found, {len(s_mean)} total windows")

    # For each class c: compute AUROC distinguishing class c from all others
    loco_results = {}
    for c in unique_ops:
        mask_c = op_types_np == c
        mask_other = ~mask_c

        scores_c = s_mean[mask_c]
        scores_other = s_mean[mask_other]

        n_c = len(scores_c)
        n_other = len(scores_other)

        if n_c < 2 or n_other < 2:
            logger.info(f"  Class {c}: insufficient data (n_c={n_c})")
            continue

        # AUROC: class c should have higher NLL if it's "out of distribution"
        try:
            auroc_info = bootstrap_auroc_ci(scores_other, scores_c, n_bootstrap=5000)
            d = cohens_d(scores_c, scores_other)
        except ValueError:
            auroc_info = {"auroc": 0.5, "ci_lower": 0.0, "ci_upper": 1.0, "std": 0.0}
            d = 0.0

        c_name = OP_TYPE_NAMES[c] if c < len(OP_TYPE_NAMES) else f"class_{c}"
        loco_results[c_name] = {
            "class_id": int(c),
            "n_class": n_c,
            "n_other": n_other,
            "class_s_mean": float(np.mean(scores_c)),
            "other_s_mean": float(np.mean(scores_other)),
            "auroc": auroc_info["auroc"],
            "auroc_ci": [auroc_info["ci_lower"], auroc_info["ci_upper"]],
            "cohens_d": d,
        }

        logger.info(
            f"  {c_name}: AUROC={auroc_info['auroc']:.3f}, "
            f"d={d:.2f}, S_mean={np.mean(scores_c):.3f} vs {np.mean(scores_other):.3f}"
        )

    save_results({"fold": fold, "loco_results": loco_results}, fold_dir, "fold_results.json")
    return {"fold": fold, "loco_results": loco_results}


def aggregate(fold_results: list, output_dir: Path) -> dict:
    """Cross-fold aggregation of LOCO results."""
    by_class = {}
    for fr in fold_results:
        for c_name, data in fr.get("loco_results", {}).items():
            if c_name not in by_class:
                by_class[c_name] = {"aurocs": [], "cohens_ds": []}
            by_class[c_name]["aurocs"].append(data["auroc"])
            by_class[c_name]["cohens_ds"].append(data["cohens_d"])

    agg = {}
    for c_name, data in by_class.items():
        agg[c_name] = {
            "mean_auroc": float(np.mean(data["aurocs"])),
            "std_auroc": float(np.std(data["aurocs"])),
            "mean_cohens_d": float(np.mean(data["cohens_ds"])),
            "per_fold_auroc": data["aurocs"],
        }

    # Rank classes by distinguishability
    ranked = sorted(agg.items(), key=lambda x: x[1]["mean_auroc"], reverse=True)

    results = {
        "per_class": agg,
        "ranking": [(name, d["mean_auroc"]) for name, d in ranked],
        "most_distinguishable": ranked[0][0] if ranked else None,
        "least_distinguishable": ranked[-1][0] if ranked else None,
        "n_folds": len(fold_results),
    }

    save_results(results, output_dir)
    return results


def generate_figures(agg_results: dict, output_dir: Path):
    """Generate LOCO analysis figures."""
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

    per_class = agg_results.get("per_class", {})
    if not per_class:
        return

    # Sort by AUROC
    classes = sorted(per_class.keys(), key=lambda c: per_class[c]["mean_auroc"], reverse=True)
    aurocs = [per_class[c]["mean_auroc"] for c in classes]
    stds = [per_class[c]["std_auroc"] for c in classes]

    # Color by condition group
    colors = []
    for c in classes:
        if "damage" in c:
            colors.append("#d62728")
        elif "150025" in c:
            colors.append("#ff7f0e")
        else:
            colors.append("#1f77b4")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(classes))
    bars = ax.bar(x, aurocs, yerr=stds, capsize=3, color=colors, edgecolor="black", alpha=0.8)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("AUROC (class vs rest)")
    ax.set_title("Leave-One-Class-Out: NLL Distinguishability per Operation Type")
    ax.set_ylim(0, 1.05)

    # Legend for condition groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Air-cut"),
        Patch(facecolor="#ff7f0e", label="Active-cut"),
        Patch(facecolor="#d62728", label="Damage"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    fig.tight_layout()
    save_figure(fig, fig_dir, "loco_auroc_by_class.pdf")
    plt.close(fig)

    logger.info(f"Figures saved to {fig_dir}")


def main():
    parser = get_default_parser("LOCO", 10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    setup_logging("exp10_loco")

    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Folds: {args.folds}")

    fold_results = []
    for fold in args.folds:
        result = run_fold(fold, args.output_dir, args.device)
        fold_results.append(result)

    agg = aggregate(fold_results, args.output_dir)
    generate_figures(agg, args.output_dir)

    # Summary
    most = agg.get("most_distinguishable", "?")
    least = agg.get("least_distinguishable", "?")
    per_class = agg.get("per_class", {})
    most_auroc = per_class.get(most, {}).get("mean_auroc", 0)
    least_auroc = per_class.get(least, {}).get("mean_auroc", 0)
    finding = f"Most distinct: {most} ({most_auroc:.3f}), least: {least} ({least_auroc:.3f})"

    append_to_summary(
        exp_id=10,
        name="LOCO",
        status="done",
        auroc="--",
        finding=finding,
        output_dir=BASE_DIR,
    )

    logger.info("Experiment 10 complete.")


if __name__ == "__main__":
    main()
