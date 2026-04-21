#!/usr/bin/env python3
"""Experiment 3: Grammar Violation Scoring

Partition detection into trivially detectable (grammar-violating) vs hard
(grammar-compliant) attacks. Compute detection rate as function of
coordinate deviation magnitude for grammar-compliant attacks.
"""

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
    load_cached_targets,
    load_vocab,
    build_token_type_map,
    build_id_to_name,
    get_axis_values,
    compute_per_token_nll,
    compute_window_scores,
    compute_auroc,
    bootstrap_auroc_ci,
    compute_detection_at_fpr,
    save_results,
    save_figure,
    append_to_summary,
    setup_logging,
    get_default_parser,
    TYPE_COMMAND,
    TYPE_PARAM,
    TYPE_NUMERIC,
)

import logging

logger = logging.getLogger(__name__)


def analyze_grammar_compliance(vocab: dict) -> dict:
    """Analyze which attacks violate grammar rules vs comply."""
    analysis = {
        "grammar_compliant": {
            "A1_command_swap": "Swaps valid commands — grammar allows any command after params",
            "A2_coord_shift": "In-vocab value substitution — same structure",
            "A3_param_swap": "Swaps valid param letters — grammar allows any param after command",
            "A4_cross_op": "Valid G-code from different operation — structurally valid",
            "A5_feed_rate": "Valid F-value substitution",
        },
        "grammar_violating": {
            "A6_oov": "Out-of-vocabulary values map to UNK — trivially detectable",
            "A7_syntax": "Two consecutive commands with no params — violates FSM",
            "A8_grammar_break": "Invalid token sequences — violates transition rules",
        },
    }
    return analysis


def run_fold(fold: int, output_dir: Path) -> dict:
    """Analyze grammar-compliant vs violating for one fold."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logits = load_cached_logits(fold)
    targets_data = load_cached_targets(fold)
    vocab = load_vocab()

    target_tokens = targets_data["target_tokens"]
    padding_mask = targets_data["padding_mask"]
    lengths = targets_data["lengths"]
    legacy_logits = logits["legacy_logits"]

    # Normal NLL scores
    nll_normal = compute_per_token_nll(legacy_logits, target_tokens, padding_mask)
    normal_scores = compute_window_scores(nll_normal, lengths)["S_mean"].numpy()

    # Load A2 graded attacks and compute detection per delta
    a2_path = ATTACK_DIR / "attack_A2_coord_graded.pt"
    if not a2_path.exists():
        logger.warning("A2 attacks not found, skipping delta analysis")
        return {"fold": fold, "delta_detection": {}}

    a2 = torch.load(a2_path, weights_only=False)

    delta_results = {}
    for key, data in a2.get("attacks_by_delta_axis", {}).items():
        fold_data = data["fold_attacks"].get(fold, {})
        n_attacks = fold_data.get("n_attacks", 0)
        if n_attacks == 0:
            continue

        attacked_tokens = fold_data["attacked_tokens"]
        orig_indices = fold_data["orig_indices"]

        # Score attack
        attack_logits = legacy_logits[orig_indices]
        attack_mask = padding_mask[orig_indices]
        attack_lengths = lengths[orig_indices]

        nll_attack = compute_per_token_nll(attack_logits, attacked_tokens, attack_mask)
        attack_scores = compute_window_scores(nll_attack, attack_lengths)["S_mean"].numpy()
        normal_subset = normal_scores[orig_indices.numpy()]

        try:
            auroc = compute_auroc(normal_subset, attack_scores)
            det_05 = compute_detection_at_fpr(normal_subset, attack_scores, 0.05)
        except (ValueError, ZeroDivisionError):
            auroc = 0.5
            det_05 = 0.0

        # Parse axis and delta from key (e.g., "X_delta_0.1")
        parts = key.split("_delta_")
        axis = parts[0] if len(parts) == 2 else key
        delta = float(parts[1]) if len(parts) == 2 else 0.0

        delta_results[key] = {
            "axis": axis,
            "delta_mm": delta,
            "n_attacks": n_attacks,
            "auroc": auroc,
            "detection_at_fpr_05": det_05,
        }

    # Grammar compliance partition summary
    grammar_analysis = analyze_grammar_compliance(vocab)

    save_results(
        {"delta_detection": delta_results, "grammar_analysis": grammar_analysis},
        fold_dir,
        "results.json",
    )

    return {
        "fold": fold,
        "delta_detection": delta_results,
    }


def aggregate(fold_results: list, output_dir: Path) -> dict:
    """Aggregate delta detection results across folds."""
    # Collect by (axis, delta) across folds
    by_key = {}
    for fr in fold_results:
        for key, data in fr.get("delta_detection", {}).items():
            if key not in by_key:
                by_key[key] = []
            by_key[key].append(data)

    agg = {}
    for key, entries in by_key.items():
        aurocs = [e["auroc"] for e in entries]
        dets = [e["detection_at_fpr_05"] for e in entries]
        agg[key] = {
            "axis": entries[0]["axis"],
            "delta_mm": entries[0]["delta_mm"],
            "mean_auroc": float(np.mean(aurocs)),
            "std_auroc": float(np.std(aurocs)),
            "mean_detection_05": float(np.mean(dets)),
            "std_detection_05": float(np.std(dets)),
            "n_folds": len(entries),
        }

    # Find MDD per axis
    mdd = {}
    for axis in ["X", "Y", "Z"]:
        axis_entries = [(v["delta_mm"], v["mean_detection_05"]) for k, v in agg.items() if v["axis"] == axis]
        axis_entries.sort()
        for delta, det in axis_entries:
            if det >= 0.95:
                mdd[axis] = delta
                break
        if axis not in mdd:
            mdd[axis] = "not reached"

    results = {
        "per_delta_axis": agg,
        "minimum_detectable_delta": mdd,
        "grammar_partition": analyze_grammar_compliance(load_vocab()),
    }

    save_results(results, output_dir)
    return results


def generate_figures(agg_results: dict, output_dir: Path):
    """Detection rate vs delta curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for axis, color in [("X", "tab:blue"), ("Y", "tab:orange"), ("Z", "tab:green")]:
        deltas = []
        dets = []
        for key, data in agg_results.get("per_delta_axis", {}).items():
            if data["axis"] == axis:
                deltas.append(data["delta_mm"])
                dets.append(data["mean_detection_05"])
        if deltas:
            order = np.argsort(deltas)
            ax.plot(np.array(deltas)[order], np.array(dets)[order], "o-", label=axis, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Coordinate Deviation (mm)")
    ax.set_ylabel("Detection Rate at FPR=5%")
    ax.set_title("Grammar-Compliant A2 Attack Detection vs Severity")
    ax.axhline(0.95, ls="--", color="gray", alpha=0.5, label="95% threshold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_figure(fig, fig_dir, "detection_vs_delta.pdf")
    plt.close(fig)


def main():
    parser = get_default_parser("Grammar Violation", 3)
    args = parser.parse_args()
    setup_logging("exp03_grammar")

    fold_results = [run_fold(f, args.output_dir) for f in args.folds]
    agg = aggregate(fold_results, args.output_dir)
    generate_figures(agg, args.output_dir)

    mdd = agg.get("minimum_detectable_delta", {})
    finding = f"MDD: X={mdd.get('X','?')}mm, Y={mdd.get('Y','?')}mm, Z={mdd.get('Z','?')}mm"
    append_to_summary(3, "Grammar Partition", "done", "--", finding, BASE_DIR)
    logger.info("Experiment 3 complete.")


if __name__ == "__main__":
    main()
