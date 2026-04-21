#!/usr/bin/env python3
"""Experiment 8: File-Level Ensemble Aggregation

Implements four window-to-file aggregation methods for anomaly detection:
  - S_majority: fraction of windows above threshold
  - S_max: max anomaly score across windows in a file
  - S_mean: mean anomaly score across windows in a file
  - S_cusum: cumulative sum detector with slack parameter

Groups windows by operation type as a proxy for file identity, then
evaluates file-level AUROC on normal vs cross-operation attacks.
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
    load_cached_targets,
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


# ============================================================
# Aggregation Methods
# ============================================================

def agg_majority(scores: np.ndarray, threshold: float) -> float:
    """Fraction of windows above threshold."""
    return float(np.mean(scores > threshold))


def agg_max(scores: np.ndarray) -> float:
    """Maximum score across windows."""
    return float(np.max(scores)) if len(scores) > 0 else 0.0


def agg_mean(scores: np.ndarray) -> float:
    """Mean score across windows."""
    return float(np.mean(scores)) if len(scores) > 0 else 0.0


def agg_cusum(scores: np.ndarray, mu_0: float, k_slack: float = 0.5) -> float:
    """CUSUM statistic: max cumulative sum with drift allowance.

    C_w = max(0, C_{w-1} + S_w - mu_0 - k_slack)
    Returns the peak CUSUM value across the sequence.
    """
    c = 0.0
    peak = 0.0
    for s in scores:
        c = max(0.0, c + s - mu_0 - k_slack)
        peak = max(peak, c)
    return peak


def _load_exp01_normal_scores(fold: int) -> dict:
    """Load per-window normal scores from exp01 output."""
    exp01_dir = BASE_DIR / "exp01_nll_scoring" / f"fold_{fold}"
    path = exp01_dir / "normal_scores.json"
    if not path.exists():
        logger.warning(f"exp01 scores not found at {path}, falling back to cached NLL")
        return None
    with open(path) as f:
        return json.load(f)


def _group_by_operation(scores: np.ndarray, op_type_names: list) -> dict:
    """Group window scores by operation type name (proxy for file)."""
    groups = defaultdict(list)
    for i, name in enumerate(op_type_names):
        groups[name].append(scores[i])
    return {k: np.array(v) for k, v in groups.items()}


def run_fold(fold: int, output_dir: Path) -> dict:
    """Run file-level ensemble aggregation for one fold."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Fold {fold} ---")

    # Load targets for grouping info
    targets_data = load_cached_targets(fold)
    op_type_names = targets_data["operation_type_names"]
    op_types = targets_data["operation_type"]

    # Load per-window normal scores from exp01
    exp01_scores = _load_exp01_normal_scores(fold)
    if exp01_scores is None:
        # Fallback: recompute from cached NLL
        from scripts.anomaly.anomaly_scoring_utils import (
            load_cached_nll, compute_window_scores,
        )
        nll_data = load_cached_nll(fold)
        lengths = targets_data["lengths"]
        ws = compute_window_scores(nll_data["nll_legacy"], lengths)
        s_mean_arr = ws["S_mean"].numpy()
    else:
        s_mean_arr = np.array(exp01_scores["S_mean"])

    n_windows = len(s_mean_arr)
    logger.info(f"  {n_windows} windows loaded")

    # Truncate to min of scores and labels
    n_min = min(n_windows, len(op_type_names))
    s_mean_arr = s_mean_arr[:n_min]
    op_type_names = op_type_names[:n_min]

    # Group windows by operation type (proxy for file)
    grouped = _group_by_operation(s_mean_arr, op_type_names)
    file_names = sorted(grouped.keys())
    logger.info(f"  {len(file_names)} unique operation types (file proxies)")

    # Compute global normal statistics for thresholding
    mu_0 = float(np.mean(s_mean_arr))
    sigma_0 = float(np.std(s_mean_arr))
    threshold_95 = float(np.percentile(s_mean_arr, 95))

    # Determine normal vs attack files
    # Normal: air-cut and active-cut operations
    # Attack: use cross-operation as proxy (windows from different op type)
    normal_bases = {"adaptive", "face", "pocket",
                    "adaptive150025", "face150025", "pocket150025"}
    damage_bases = {"damageadaptive", "damageface", "damagepocket"}

    normal_files = []
    attack_files = []
    for name in file_names:
        base = name.split("_")[0] if "_" in name else name
        if base in normal_bases:
            normal_files.append(name)
        elif base in damage_bases:
            attack_files.append(name)

    logger.info(f"  Normal files: {len(normal_files)}, Attack files: {len(attack_files)}")

    # Compute file-level scores for each aggregation method
    methods = {}

    # S_majority
    normal_majority = np.array([agg_majority(grouped[f], threshold_95) for f in normal_files])
    attack_majority = np.array([agg_majority(grouped[f], threshold_95) for f in attack_files])
    methods["S_majority"] = {"normal": normal_majority, "attack": attack_majority}

    # S_max
    normal_max = np.array([agg_max(grouped[f]) for f in normal_files])
    attack_max = np.array([agg_max(grouped[f]) for f in attack_files])
    methods["S_max"] = {"normal": normal_max, "attack": attack_max}

    # S_mean
    normal_mean = np.array([agg_mean(grouped[f]) for f in normal_files])
    attack_mean = np.array([agg_mean(grouped[f]) for f in attack_files])
    methods["S_mean"] = {"normal": normal_mean, "attack": attack_mean}

    # S_cusum
    k_slack = 0.5 * sigma_0 if sigma_0 > 0 else 0.5
    normal_cusum = np.array([agg_cusum(grouped[f], mu_0, k_slack) for f in normal_files])
    attack_cusum = np.array([agg_cusum(grouped[f], mu_0, k_slack) for f in attack_files])
    methods["S_cusum"] = {"normal": normal_cusum, "attack": attack_cusum}

    # Evaluate each method
    fold_result = {
        "fold": fold,
        "n_windows": n_windows,
        "n_normal_files": len(normal_files),
        "n_attack_files": len(attack_files),
        "mu_0": mu_0,
        "sigma_0": sigma_0,
        "threshold_95": threshold_95,
        "k_slack": k_slack,
        "methods": {},
    }

    for method_name, data in methods.items():
        n_norm = len(data["normal"])
        n_atk = len(data["attack"])
        if n_norm < 2 or n_atk < 2:
            logger.warning(f"  {method_name}: insufficient data (normal={n_norm}, attack={n_atk})")
            fold_result["methods"][method_name] = {
                "auroc": None, "n_normal": n_norm, "n_attack": n_atk,
            }
            continue

        try:
            auroc_info = bootstrap_auroc_ci(data["normal"], data["attack"], n_bootstrap=5000)
            d = cohens_d(data["attack"], data["normal"])
        except (ValueError, ZeroDivisionError):
            auroc_info = {"auroc": 0.5, "ci_lower": 0.0, "ci_upper": 1.0, "std": 0.0}
            d = 0.0

        fold_result["methods"][method_name] = {
            "auroc": auroc_info["auroc"],
            "auroc_ci": [auroc_info["ci_lower"], auroc_info["ci_upper"]],
            "cohens_d": d,
            "n_normal": n_norm,
            "n_attack": n_atk,
            "normal_mean": float(np.mean(data["normal"])),
            "attack_mean": float(np.mean(data["attack"])),
        }
        logger.info(
            f"  {method_name}: AUROC={auroc_info['auroc']:.3f} "
            f"[{auroc_info['ci_lower']:.3f}, {auroc_info['ci_upper']:.3f}], d={d:.2f}"
        )

    # Save CUSUM traces for figure generation
    cusum_traces = {}
    example_files = (normal_files[:3] if len(normal_files) >= 3 else normal_files) + \
                    (attack_files[:3] if len(attack_files) >= 3 else attack_files)
    for fname in example_files:
        trace = []
        c = 0.0
        for s in grouped[fname]:
            c = max(0.0, c + s - mu_0 - k_slack)
            trace.append(c)
        cusum_traces[fname] = trace
    torch.save(cusum_traces, fold_dir / "cusum_traces.pt")

    save_results(fold_result, fold_dir, "fold_results.json")
    return fold_result


def aggregate(fold_results: list, output_dir: Path) -> dict:
    """Cross-fold aggregation of file-level detection results."""
    method_names = ["S_majority", "S_max", "S_mean", "S_cusum"]

    agg = {"n_folds": len(fold_results), "methods": {}}

    for method in method_names:
        aurocs = []
        ds = []
        for fr in fold_results:
            m = fr["methods"].get(method, {})
            if m.get("auroc") is not None:
                aurocs.append(m["auroc"])
            if m.get("cohens_d") is not None:
                ds.append(m["cohens_d"])

        if aurocs:
            agg["methods"][method] = {
                "mean_auroc": float(np.mean(aurocs)),
                "std_auroc": float(np.std(aurocs)),
                "per_fold_auroc": aurocs,
                "mean_cohens_d": float(np.mean(ds)) if ds else None,
            }
        else:
            agg["methods"][method] = {"mean_auroc": None}

    # Find best method
    best_method = None
    best_auroc = 0.0
    for method, data in agg["methods"].items():
        if data.get("mean_auroc") is not None and data["mean_auroc"] > best_auroc:
            best_auroc = data["mean_auroc"]
            best_method = method
    agg["best_method"] = best_method
    agg["best_auroc"] = best_auroc

    save_results(agg, output_dir)
    return agg


def generate_figures(agg_results: dict, output_dir: Path, fold_results: list):
    """Generate ensemble comparison and CUSUM chart figures."""
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

    # Figure 1: AUROC comparison bar chart
    methods_data = agg_results.get("methods", {})
    method_names = [m for m in ["S_majority", "S_max", "S_mean", "S_cusum"] if m in methods_data]
    aurocs = [methods_data[m].get("mean_auroc", 0) or 0 for m in method_names]
    stds = [methods_data[m].get("std_auroc", 0) or 0 for m in method_names]

    if method_names:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(method_names))
        bars = ax.bar(x, aurocs, yerr=stds, capsize=5, color="#4c72b0", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=15)
        ax.set_ylabel("File-Level AUROC")
        ax.set_title("Aggregation Method Comparison")
        ax.set_ylim(0, 1.05)
        for i, (bar, val) in enumerate(zip(bars, aurocs)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        save_figure(fig, fig_dir, "ensemble_auroc_comparison.pdf")
        plt.close(fig)

    # Figure 2: CUSUM chart for example sequences
    # Use fold 1 traces if available
    traces_path = output_dir / "fold_1" / "cusum_traces.pt"
    if not traces_path.exists() and fold_results:
        traces_path = output_dir / f"fold_{fold_results[0]['fold']}" / "cusum_traces.pt"

    if traces_path.exists():
        cusum_traces = torch.load(traces_path, weights_only=False)
        if cusum_traces:
            fig, ax = plt.subplots(figsize=(10, 5))

            normal_bases = {"adaptive", "face", "pocket",
                            "adaptive150025", "face150025", "pocket150025"}

            for fname, trace in cusum_traces.items():
                base = fname.split("_")[0] if "_" in fname else fname
                is_normal = base in normal_bases
                color = "#1f77b4" if is_normal else "#d62728"
                ls = "-" if is_normal else "--"
                label_prefix = "Normal" if is_normal else "Damage"
                ax.plot(trace, color=color, linestyle=ls, alpha=0.7,
                        label=f"{label_prefix}: {fname}")

            ax.set_xlabel("Window Index")
            ax.set_ylabel("CUSUM Statistic")
            ax.set_title("CUSUM Charts: Normal vs Damage Sequences")
            ax.legend(fontsize=7, loc="upper left")
            fig.tight_layout()
            save_figure(fig, fig_dir, "cusum_example_traces.pdf")
            plt.close(fig)

    logger.info(f"Figures saved to {fig_dir}")


def main():
    parser = get_default_parser("Ensemble Aggregation", 8)
    args = parser.parse_args()

    setup_logging("exp08_ensemble")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Folds: {args.folds}")

    fold_results = []
    for fold in args.folds:
        result = run_fold(fold, args.output_dir)
        fold_results.append(result)

    agg = aggregate(fold_results, args.output_dir)
    generate_figures(agg, args.output_dir, fold_results)

    # Summary
    best = agg.get("best_method", "N/A")
    best_auroc = agg.get("best_auroc", 0)
    finding = f"Best: {best} at {best_auroc:.3f} AUROC" if best else "No results"

    append_to_summary(
        exp_id=8,
        name="Ensemble Aggregation",
        status="done",
        auroc=f"{best_auroc:.3f}" if best_auroc else "--",
        finding=finding,
        output_dir=BASE_DIR,
    )

    logger.info("Experiment 8 complete.")


if __name__ == "__main__":
    main()
