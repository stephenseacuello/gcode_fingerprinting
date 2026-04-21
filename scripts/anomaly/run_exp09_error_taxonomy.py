#!/usr/bin/env python3
"""Experiment 9: Error Taxonomy Analysis

Loads cached predictions and targets, identifies mismatches (argmax != target)
as false positives for reconstruction, and categorizes errors by:
  - Token type (command / param / numeric)
  - Operation type (9 classes)
  - Position in sequence (early / middle / late)

Loads attack results from exp01 and exp12 to identify false negatives.
Runs a chi-squared test of independence between error category and operation type.
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import (
    BASE_DIR,
    FOLDS,
    load_cached_predictions,
    load_cached_targets,
    load_vocab,
    build_token_type_map,
    build_id_to_name,
    TYPE_COMMAND,
    TYPE_PARAM,
    TYPE_NUMERIC,
    TYPE_SPECIAL,
    save_results,
    save_figure,
    append_to_summary,
    setup_logging,
    get_default_parser,
)

logger = logging.getLogger(__name__)

TYPE_NAMES = {
    TYPE_SPECIAL: "special",
    TYPE_COMMAND: "command",
    TYPE_PARAM: "param",
    TYPE_NUMERIC: "numeric",
}

POSITION_BINS = {
    "early": (0.0, 0.33),
    "middle": (0.33, 0.67),
    "late": (0.67, 1.0),
}


def _categorize_errors(
    argmax_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    padding_mask: torch.Tensor,
    lengths: torch.Tensor,
    op_types: torch.Tensor,
    token_type_map: dict,
) -> dict:
    """Categorize prediction errors by token type, operation, and position.

    Returns dict with error counts per category.
    """
    N, L = target_tokens.shape
    errors_by_type = defaultdict(int)
    errors_by_op = defaultdict(int)
    errors_by_position = defaultdict(int)
    correct_by_type = defaultdict(int)
    total_by_type = defaultdict(int)

    # Contingency: rows = token type, cols = operation type
    op_ids = sorted(set(op_types.numpy().tolist()))
    type_ids = [TYPE_COMMAND, TYPE_PARAM, TYPE_NUMERIC]
    contingency = np.zeros((len(type_ids), len(op_ids)), dtype=int)
    op_to_col = {op: i for i, op in enumerate(op_ids)}
    type_to_row = {t: i for i, t in enumerate(type_ids)}

    for i in range(N):
        seq_len = int(lengths[i].item())
        op = int(op_types[i].item())
        for j in range(min(seq_len, L)):
            if padding_mask[i, j]:
                continue
            tgt = int(target_tokens[i, j].item())
            pred = int(argmax_tokens[i, j].item())
            ttype = token_type_map.get(tgt, TYPE_SPECIAL)
            ttype_name = TYPE_NAMES.get(ttype, "special")

            total_by_type[ttype_name] += 1

            if pred != tgt:
                errors_by_type[ttype_name] += 1
                errors_by_op[op] += 1

                # Position bin
                rel_pos = j / max(seq_len - 1, 1)
                for pos_name, (lo, hi) in POSITION_BINS.items():
                    if lo <= rel_pos < hi or (pos_name == "late" and rel_pos >= hi - 0.01):
                        errors_by_position[pos_name] += 1
                        break

                # Contingency table
                if ttype in type_to_row and op in op_to_col:
                    contingency[type_to_row[ttype], op_to_col[op]] += 1
            else:
                correct_by_type[ttype_name] += 1

    return {
        "errors_by_type": dict(errors_by_type),
        "errors_by_op": dict(errors_by_op),
        "errors_by_position": dict(errors_by_position),
        "correct_by_type": dict(correct_by_type),
        "total_by_type": dict(total_by_type),
        "contingency": contingency.tolist(),
        "contingency_rows": [TYPE_NAMES[t] for t in type_ids],
        "contingency_cols": op_ids,
    }


def run_fold(fold: int, output_dir: Path) -> dict:
    """Run error taxonomy analysis for one fold."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Fold {fold} ---")

    predictions = load_cached_predictions(fold)
    targets_data = load_cached_targets(fold)
    vocab = load_vocab()
    token_type_map = build_token_type_map(vocab)

    argmax_tokens = predictions["argmax_tokens"]
    target_tokens = targets_data["target_tokens"]
    padding_mask = targets_data["padding_mask"]
    lengths = targets_data["lengths"]
    op_types = targets_data["operation_type"]

    # Categorize reconstruction errors
    error_cats = _categorize_errors(
        argmax_tokens, target_tokens, padding_mask, lengths, op_types, token_type_map
    )

    # Error rates by type
    error_rates = {}
    for ttype in error_cats["total_by_type"]:
        total = error_cats["total_by_type"][ttype]
        errs = error_cats["errors_by_type"].get(ttype, 0)
        error_rates[ttype] = float(errs / total) if total > 0 else 0.0

    logger.info(f"  Error rates by type: {error_rates}")

    # Chi-squared test
    contingency = np.array(error_cats["contingency"])
    chi2_result = {"chi2": None, "p_value": None, "dof": None, "cramers_v": None}
    if contingency.sum() > 0 and contingency.shape[0] > 1 and contingency.shape[1] > 1:
        # Remove zero rows/cols for valid chi2
        row_mask = contingency.sum(axis=1) > 0
        col_mask = contingency.sum(axis=0) > 0
        filtered = contingency[row_mask][:, col_mask]
        if filtered.shape[0] > 1 and filtered.shape[1] > 1:
            from scipy.stats import chi2_contingency
            chi2, p, dof, expected = chi2_contingency(filtered)
            n = filtered.sum()
            k = min(filtered.shape)
            cramers_v = float(np.sqrt(chi2 / (n * max(k - 1, 1)))) if n > 0 else 0.0
            chi2_result = {
                "chi2": float(chi2),
                "p_value": float(p),
                "dof": int(dof),
                "cramers_v": cramers_v,
            }
            logger.info(f"  Chi2={chi2:.2f}, p={p:.2e}, Cramer's V={cramers_v:.3f}")

    # Load false negatives from exp01 (attacks that were NOT detected)
    false_negatives = {}
    exp01_fold_dir = BASE_DIR / "exp01_nll_scoring" / f"fold_{fold}"
    exp01_path = exp01_fold_dir / "attack_scores.json"
    if exp01_path.exists():
        with open(exp01_path) as f:
            exp01_scores = json.load(f)
        for attack_name, attack_data in exp01_scores.items():
            s_mean = attack_data.get("S_mean", {})
            det_05 = s_mean.get("detection_at_fpr_05", 1.0)
            false_negatives[attack_name] = {
                "detection_rate_fpr05": det_05,
                "miss_rate_fpr05": 1.0 - det_05,
            }

    fold_result = {
        "fold": fold,
        "error_rates_by_type": error_rates,
        "errors_by_type": error_cats["errors_by_type"],
        "errors_by_op": error_cats["errors_by_op"],
        "errors_by_position": error_cats["errors_by_position"],
        "total_by_type": error_cats["total_by_type"],
        "chi2_test": chi2_result,
        "contingency": error_cats["contingency"],
        "contingency_rows": error_cats["contingency_rows"],
        "contingency_cols": error_cats["contingency_cols"],
        "false_negatives_exp01": false_negatives,
    }

    save_results(fold_result, fold_dir, "fold_results.json")
    return fold_result


def aggregate(fold_results: list, output_dir: Path) -> dict:
    """Cross-fold aggregation of error taxonomy."""
    # Pool error rates
    type_rates = defaultdict(list)
    chi2_vals = []
    cramers_vs = []

    for fr in fold_results:
        for ttype, rate in fr.get("error_rates_by_type", {}).items():
            type_rates[ttype].append(rate)
        chi2_r = fr.get("chi2_test", {})
        if chi2_r.get("chi2") is not None:
            chi2_vals.append(chi2_r["chi2"])
        if chi2_r.get("cramers_v") is not None:
            cramers_vs.append(chi2_r["cramers_v"])

    # Pool position errors
    pos_counts = defaultdict(int)
    for fr in fold_results:
        for pos, count in fr.get("errors_by_position", {}).items():
            pos_counts[pos] += count

    total_pos = sum(pos_counts.values())
    pos_fractions = {p: c / total_pos if total_pos > 0 else 0.0 for p, c in pos_counts.items()}

    results = {
        "error_rates_by_type": {
            ttype: {
                "mean": float(np.mean(rates)),
                "std": float(np.std(rates)),
                "per_fold": rates,
            }
            for ttype, rates in type_rates.items()
        },
        "error_position_distribution": pos_fractions,
        "chi2_test_aggregate": {
            "mean_chi2": float(np.mean(chi2_vals)) if chi2_vals else None,
            "mean_cramers_v": float(np.mean(cramers_vs)) if cramers_vs else None,
            "all_significant": all(
                fr.get("chi2_test", {}).get("p_value", 1.0) < 0.05
                for fr in fold_results
                if fr.get("chi2_test", {}).get("p_value") is not None
            ),
        },
        "n_folds": len(fold_results),
    }

    save_results(results, output_dir)
    return results


def generate_figures(agg_results: dict, output_dir: Path):
    """Generate error taxonomy figures."""
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

    # Figure 1: Error rate by token type
    rates = agg_results.get("error_rates_by_type", {})
    if rates:
        types = sorted(rates.keys())
        means = [rates[t]["mean"] for t in types]
        stds = [rates[t]["std"] for t in types]

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(types))
        ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0", edgecolor="black", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(types)
        ax.set_ylabel("Error Rate")
        ax.set_title("Reconstruction Error Rate by Token Type")
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=9)
        fig.tight_layout()
        save_figure(fig, fig_dir, "error_rate_by_type.pdf")
        plt.close(fig)

    # Figure 2: Error distribution by position
    pos_dist = agg_results.get("error_position_distribution", {})
    if pos_dist:
        positions = ["early", "middle", "late"]
        fracs = [pos_dist.get(p, 0.0) for p in positions]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(positions, fracs, color=["#55A868", "#4C72B0", "#C44E52"], alpha=0.8,
               edgecolor="black")
        ax.set_ylabel("Fraction of Errors")
        ax.set_title("Error Distribution by Sequence Position")
        fig.tight_layout()
        save_figure(fig, fig_dir, "error_by_position.pdf")
        plt.close(fig)

    logger.info(f"Figures saved to {fig_dir}")


def main():
    parser = get_default_parser("Error Taxonomy", 9)
    args = parser.parse_args()
    setup_logging("exp09_error_taxonomy")

    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Folds: {args.folds}")

    fold_results = []
    for fold in args.folds:
        result = run_fold(fold, args.output_dir)
        fold_results.append(result)

    agg = aggregate(fold_results, args.output_dir)
    generate_figures(agg, args.output_dir)

    # Summary
    rates = agg.get("error_rates_by_type", {})
    chi2 = agg.get("chi2_test_aggregate", {})
    parts = []
    for ttype in ["numeric", "command", "param"]:
        if ttype in rates:
            parts.append(f"{ttype}={rates[ttype]['mean']:.3f}")
    if chi2.get("mean_cramers_v") is not None:
        parts.append(f"V={chi2['mean_cramers_v']:.3f}")
    finding = "Error rates: " + ", ".join(parts) if parts else "No results"

    append_to_summary(
        exp_id=9,
        name="Error Taxonomy",
        status="done",
        auroc="--",
        finding=finding,
        output_dir=BASE_DIR,
    )

    logger.info("Experiment 9 complete.")


if __name__ == "__main__":
    main()
