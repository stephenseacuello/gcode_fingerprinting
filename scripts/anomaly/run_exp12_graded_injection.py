#!/usr/bin/env python3
"""Experiment 12: Graded Coordinate Injection (Minimum Detectable Delta)

Loads pre-generated A2 attacks from attacks/attack_A2_coord_graded.pt.
For each (axis, delta), computes detection rate using cached logits.
Scores attacked tokens against the model's predicted distribution.

Fits a logistic curve to detection_rate vs log(delta) using
scipy.optimize.curve_fit. Finds MDD (minimum detectable delta) = delta
where fitted curve crosses 95%.

Generates: detection rate vs delta curve per axis.
"""

import json
import logging
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
    load_cached_targets,
    compute_per_token_nll,
    compute_window_scores,
    compute_detection_at_fpr,
    save_results,
    save_figure,
    append_to_summary,
    setup_logging,
    get_default_parser,
)

logger = logging.getLogger(__name__)


def _fit_logistic(deltas: np.ndarray, detection_rates: np.ndarray):
    """Fit logistic curve to detection_rate vs log(delta).

    Model: f(x) = 1 / (1 + exp(-k*(log(x) - log(x_50))))

    Returns:
        k: steepness parameter
        delta_50: delta at 50% detection
        fitted_fn: callable for prediction
    """
    from scipy.optimize import curve_fit

    # Work in log-space
    log_deltas = np.log(deltas + 1e-12)

    def logistic(x, k, x_50):
        return 1.0 / (1.0 + np.exp(-k * (x - x_50)))

    try:
        p0 = [2.0, np.median(log_deltas)]
        popt, _ = curve_fit(
            logistic, log_deltas, detection_rates,
            p0=p0, maxfev=5000,
            bounds=([0.01, log_deltas.min() - 2], [50.0, log_deltas.max() + 2]),
        )
        k, log_delta_50 = popt
        delta_50 = np.exp(log_delta_50)

        def fitted_fn(x):
            return logistic(np.log(x + 1e-12), k, log_delta_50)

        return k, delta_50, fitted_fn
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Logistic fit failed: {e}")
        return None, None, None


def _find_mdd(fitted_fn, deltas: np.ndarray, target_detection: float = 0.95) -> float:
    """Find the minimum detectable delta where detection >= target."""
    if fitted_fn is None:
        return float("nan")

    fine_deltas = np.logspace(
        np.log10(max(deltas.min(), 1e-6)),
        np.log10(deltas.max()),
        num=1000,
    )
    rates = fitted_fn(fine_deltas)
    above = np.where(rates >= target_detection)[0]
    if len(above) == 0:
        return float("inf")
    return float(fine_deltas[above[0]])


def run_fold(fold: int, output_dir: Path) -> dict:
    """Run graded injection analysis for one fold."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Fold {fold} ---")

    # Load cached logits and targets
    logits_data = load_cached_logits(fold)
    targets_data = load_cached_targets(fold)

    legacy_logits = logits_data["legacy_logits"]
    target_tokens = targets_data["target_tokens"]
    padding_mask = targets_data["padding_mask"]
    lengths = targets_data["lengths"]

    # Compute normal scores for threshold setting
    nll_normal = compute_per_token_nll(legacy_logits, target_tokens, padding_mask)
    normal_scores = compute_window_scores(nll_normal, lengths)
    normal_s_mean = normal_scores["S_mean"].numpy()

    # Load pre-generated A2 graded attacks
    a2_path = ATTACK_DIR / "attack_A2_coord_graded.pt"
    if not a2_path.exists():
        logger.error(f"A2 graded attacks not found at {a2_path}")
        return {"fold": fold, "error": "A2 attacks not found"}

    a2 = torch.load(a2_path, weights_only=False)
    all_deltas = a2["deltas"]
    axes = a2["axes"]

    fold_result = {
        "fold": fold,
        "n_windows": len(target_tokens),
        "normal_s_mean_mean": float(np.mean(normal_s_mean)),
        "normal_s_mean_std": float(np.std(normal_s_mean)),
        "axes": {},
    }

    for axis in axes:
        logger.info(f"  Axis {axis}:")
        delta_values = []
        detection_01 = []
        detection_05 = []
        detection_10 = []

        for delta in all_deltas:
            key = f"{axis}_delta_{delta}"
            attack_data = a2["attacks_by_delta_axis"].get(key, {})
            fold_data = attack_data.get("fold_attacks", {}).get(fold, {})

            n_attacks = fold_data.get("n_attacks", 0)
            if n_attacks == 0:
                continue

            attacked_tokens = fold_data["attacked_tokens"]
            orig_indices = fold_data["orig_indices"]

            # Score attacked tokens against the model's predicted distribution
            attack_logits = legacy_logits[orig_indices]
            attack_mask = padding_mask[orig_indices]
            attack_lengths = lengths[orig_indices]

            nll_attack = compute_per_token_nll(attack_logits, attacked_tokens, attack_mask)
            attack_ws = compute_window_scores(nll_attack, attack_lengths)
            attack_s_mean = attack_ws["S_mean"].numpy()

            normal_subset = normal_s_mean[orig_indices.numpy()]

            det_01 = compute_detection_at_fpr(normal_subset, attack_s_mean, 0.01)
            det_05 = compute_detection_at_fpr(normal_subset, attack_s_mean, 0.05)
            det_10 = compute_detection_at_fpr(normal_subset, attack_s_mean, 0.10)

            delta_values.append(delta)
            detection_01.append(det_01)
            detection_05.append(det_05)
            detection_10.append(det_10)

            logger.info(
                f"    delta={delta:8.3f}mm: n={n_attacks:5d}, "
                f"det@1%={det_01:.3f}, det@5%={det_05:.3f}, det@10%={det_10:.3f}"
            )

        if not delta_values:
            fold_result["axes"][axis] = {"error": "no valid deltas"}
            continue

        delta_arr = np.array(delta_values)
        det_05_arr = np.array(detection_05)

        # Fit logistic curve on the FPR=5% detection rates
        k, delta_50, fitted_fn = _fit_logistic(delta_arr, det_05_arr)

        # Find MDD at each FPR level
        mdd_results = {}
        for fpr_name, det_arr in [
            ("fpr_01", detection_01),
            ("fpr_05", detection_05),
            ("fpr_10", detection_10),
        ]:
            det_np = np.array(det_arr)
            _, _, fn = _fit_logistic(delta_arr, det_np)
            mdd = _find_mdd(fn, delta_arr, target_detection=0.95)
            mdd_results[fpr_name] = float(mdd) if np.isfinite(mdd) else None

        fold_result["axes"][axis] = {
            "deltas": delta_values,
            "detection_at_fpr_01": detection_01,
            "detection_at_fpr_05": detection_05,
            "detection_at_fpr_10": detection_10,
            "logistic_k": float(k) if k is not None else None,
            "logistic_delta_50": float(delta_50) if delta_50 is not None else None,
            "mdd_95": mdd_results,
        }

    save_results(fold_result, fold_dir, "fold_results.json")
    return fold_result


def aggregate(fold_results: list, output_dir: Path) -> dict:
    """Cross-fold aggregation of MDD results."""
    axes_found = set()
    for fr in fold_results:
        axes_found.update(fr.get("axes", {}).keys())

    agg = {"n_folds": len(fold_results), "axes": {}}

    for axis in sorted(axes_found):
        axis_data = {"mdd_95": {}, "logistic_k": [], "logistic_delta_50": []}

        for fpr_key in ["fpr_01", "fpr_05", "fpr_10"]:
            mdds = []
            for fr in fold_results:
                ax = fr.get("axes", {}).get(axis, {})
                mdd_val = ax.get("mdd_95", {}).get(fpr_key)
                if mdd_val is not None:
                    mdds.append(mdd_val)
            if mdds:
                axis_data["mdd_95"][fpr_key] = {
                    "mean_mm": float(np.mean(mdds)),
                    "std_mm": float(np.std(mdds)),
                    "per_fold": mdds,
                }

        for fr in fold_results:
            ax = fr.get("axes", {}).get(axis, {})
            if ax.get("logistic_k") is not None:
                axis_data["logistic_k"].append(ax["logistic_k"])
            if ax.get("logistic_delta_50") is not None:
                axis_data["logistic_delta_50"].append(ax["logistic_delta_50"])

        axis_data["mean_k"] = (
            float(np.mean(axis_data["logistic_k"])) if axis_data["logistic_k"] else None
        )
        axis_data["mean_delta_50"] = (
            float(np.mean(axis_data["logistic_delta_50"]))
            if axis_data["logistic_delta_50"]
            else None
        )

        agg["axes"][axis] = axis_data

    # Build MDD summary table
    mdd_table = {}
    for axis in sorted(axes_found):
        mdd_table[axis] = {}
        for fpr_key in ["fpr_01", "fpr_05", "fpr_10"]:
            entry = agg["axes"].get(axis, {}).get("mdd_95", {}).get(fpr_key, {})
            mdd_table[axis][fpr_key] = entry.get("mean_mm")
    agg["mdd_table"] = mdd_table

    save_results(agg, output_dir)
    return agg


def generate_figures(agg_results: dict, output_dir: Path, fold_results: list):
    """Generate detection-rate-vs-delta figure per axis."""
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

    axes = sorted(agg_results.get("axes", {}).keys())
    axis_colors = {"X": "#1f77b4", "Y": "#ff7f0e", "Z": "#2ca02c"}

    # Figure: Detection rate vs delta for FPR=5%
    fig, ax = plt.subplots(figsize=(8, 5))

    for axis in axes:
        all_deltas = None
        all_dets = []
        for fr in fold_results:
            ax_data = fr.get("axes", {}).get(axis, {})
            if "deltas" not in ax_data:
                continue
            deltas = ax_data["deltas"]
            dets = ax_data.get("detection_at_fpr_05", [])
            if not dets:
                continue
            if all_deltas is None:
                all_deltas = np.array(deltas)
            all_dets.append(np.array(dets))

        if all_deltas is None or not all_dets:
            continue

        min_len = min(len(d) for d in all_dets)
        all_dets_trimmed = np.array([d[:min_len] for d in all_dets])
        mean_dets = np.mean(all_dets_trimmed, axis=0)
        std_dets = np.std(all_dets_trimmed, axis=0)

        color = axis_colors.get(axis, "#333333")

        # Plot data points
        ax.semilogx(
            all_deltas[:min_len], mean_dets, "o-",
            color=color, label=f"{axis}-axis", markersize=5,
        )
        ax.fill_between(
            all_deltas[:min_len],
            mean_dets - std_dets,
            np.minimum(mean_dets + std_dets, 1.0),
            alpha=0.15, color=color,
        )

        # Overlay logistic fit
        k_vals = agg_results["axes"][axis].get("logistic_k", [])
        d50_vals = agg_results["axes"][axis].get("logistic_delta_50", [])
        if k_vals and d50_vals:
            k_mean = np.mean(k_vals)
            d50_mean = np.mean(d50_vals)
            fine_d = np.logspace(
                np.log10(max(all_deltas[0], 1e-6)),
                np.log10(all_deltas[-1]),
                200,
            )

            def logistic_fn(x):
                return 1.0 / (1.0 + np.exp(-k_mean * (np.log(x + 1e-12) - np.log(d50_mean))))

            ax.semilogx(fine_d, logistic_fn(fine_d), "--", color=color, alpha=0.5)

    ax.axhline(y=0.95, color="gray", linestyle="--", linewidth=0.8, label="95% detection")
    ax.set_xlabel(r"$\delta$ (mm)")
    ax.set_ylabel("Detection Rate at FPR=5%")
    ax.set_title("Graded Coordinate Injection: Detection Rate vs Delta")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, fig_dir, "detection_vs_delta.pdf")
    plt.close(fig)

    # MDD table as text
    mdd_table = agg_results.get("mdd_table", {})
    if mdd_table:
        lines = ["Minimum Detectable Delta (MDD) at 95% Detection", ""]
        lines.append(f"{'Axis':<6} {'FPR=1%':>12} {'FPR=5%':>12} {'FPR=10%':>12}")
        lines.append("-" * 44)
        for axis in sorted(mdd_table.keys()):
            vals = mdd_table[axis]
            row = f"{axis:<6}"
            for fpr_key in ["fpr_01", "fpr_05", "fpr_10"]:
                v = vals.get(fpr_key)
                row += f" {v:>11.4f}mm" if v is not None else f" {'N/A':>12}"
            lines.append(row)

        table_path = fig_dir / "mdd_table.txt"
        table_path.write_text("\n".join(lines))
        logger.info(f"MDD table saved to {table_path}")

    logger.info(f"Figures saved to {fig_dir}")


def main():
    parser = get_default_parser("Graded Injection", 12)
    args = parser.parse_args()

    setup_logging("exp12_graded_injection")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Folds: {args.folds}")

    fold_results = []
    for fold in args.folds:
        result = run_fold(fold, args.output_dir)
        fold_results.append(result)

    agg = aggregate(fold_results, args.output_dir)
    generate_figures(agg, args.output_dir, fold_results)

    # Summary
    mdd_table = agg.get("mdd_table", {})
    findings = []
    for axis in sorted(mdd_table.keys()):
        v = mdd_table[axis].get("fpr_05")
        if v is not None:
            findings.append(f"{axis}:{v:.3f}mm")
    finding = "MDD@5%: " + ", ".join(findings) if findings else "No MDD computed"

    append_to_summary(
        exp_id=12,
        name="Graded Injection",
        status="done",
        auroc="--",
        finding=finding,
        output_dir=BASE_DIR,
    )

    logger.info("Experiment 12 complete.")


if __name__ == "__main__":
    main()
