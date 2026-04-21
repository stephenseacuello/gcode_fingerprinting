#!/usr/bin/env python3
"""Experiment 18: CUSUM Sequential Change-Point Detection

Loads per-window NLL scores from exp01 and groups windows by operation type
as a proxy for sequential windows in a file.

CUSUM: C_w = max(0, C_{w-1} + S_w - mu_0 - k_slack)
  mu_0 = mean S_mean on normal windows
  k_slack = 0.5
  h = threshold set at 3*std of normal scores

For each group, simulates an attack at window w* = group_size // 2.
Detection latency = number of windows after w* before CUSUM exceeds h.

Generates CUSUM charts showing normal vs attacked sequences.
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
    save_results,
    save_figure,
    append_to_summary,
    setup_logging,
    get_default_parser,
)

logger = logging.getLogger(__name__)

# CUSUM parameters as specified
K_SLACK = 0.5


def _load_exp01_scores(fold: int) -> np.ndarray:
    """Load per-window S_mean scores from exp01 results."""
    exp01_dir = BASE_DIR / "exp01_nll_scoring" / f"fold_{fold}"
    path = exp01_dir / "normal_scores.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return np.array(data["S_mean"])

    # Fallback: recompute from cached NLL
    logger.info(f"  Falling back to cached NLL for fold {fold}")
    from scripts.anomaly.anomaly_scoring_utils import (
        load_cached_nll, compute_window_scores,
    )
    targets_data = load_cached_targets(fold)
    nll_data = load_cached_nll(fold)
    ws = compute_window_scores(nll_data["nll_legacy"], targets_data["lengths"])
    return ws["S_mean"].numpy()


def _group_by_operation(scores: np.ndarray, op_type_names: list) -> dict:
    """Group window scores by operation type name (proxy for file)."""
    groups = defaultdict(list)
    for i, name in enumerate(op_type_names):
        groups[name].append(float(scores[i]))
    return {k: np.array(v) for k, v in groups.items()}


def run_cusum(scores: np.ndarray, mu_0: float, k_slack: float, h: float):
    """Run one-sided upper CUSUM.

    C_w = max(0, C_{w-1} + S_w - mu_0 - k_slack)

    Returns:
        trace: list of C_w values
        alarm_idx: index of first alarm where C_w > h (None if no alarm)
    """
    trace = []
    c = 0.0
    alarm_idx = None
    for i, s in enumerate(scores):
        c = max(0.0, c + s - mu_0 - k_slack)
        trace.append(c)
        if alarm_idx is None and c > h:
            alarm_idx = i
    return trace, alarm_idx


def _simulate_attack_at_midpoint(
    normal_scores: np.ndarray,
    mu_0: float,
    sigma_0: float,
    k_slack: float,
    h: float,
    rng: np.random.RandomState,
) -> dict:
    """Simulate attack at w* = group_size // 2.

    The attack elevates NLL from w* onward by adding 2*sigma_0 shift.
    Detection latency = number of windows after w* before CUSUM exceeds h.
    """
    n = len(normal_scores)
    if n < 4:
        return {"w_star": None, "latency": None, "detected": False}

    w_star = n // 2

    # Create attacked sequence: normal NLL + shift after w*
    attacked = normal_scores.copy()
    shift = 2.0 * sigma_0
    noise = rng.normal(0, 0.1 * shift, size=n - w_star)
    attacked[w_star:] += shift + noise

    trace, alarm_idx = run_cusum(attacked, mu_0, k_slack, h)

    if alarm_idx is not None and alarm_idx >= w_star:
        latency = alarm_idx - w_star
        detected = True
    else:
        latency = None
        detected = False

    return {
        "w_star": w_star,
        "latency": latency,
        "detected": detected,
        "trace": trace,
        "alarm_idx": alarm_idx,
    }


def run_fold(fold: int, output_dir: Path, seed: int = 42) -> dict:
    """Run CUSUM analysis for one fold."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Fold {fold} ---")
    rng = np.random.RandomState(seed + fold)

    # Load data
    targets_data = load_cached_targets(fold)
    op_type_names = targets_data["operation_type_names"]
    s_mean = _load_exp01_scores(fold)

    # Truncate to min of scores and labels
    n_min = min(len(s_mean), len(op_type_names))
    s_mean = s_mean[:n_min]
    op_type_names = op_type_names[:n_min]

    # Group by operation type
    grouped = _group_by_operation(s_mean, op_type_names)
    file_names = sorted(grouped.keys())
    logger.info(f"  {len(file_names)} file groups, {len(s_mean)} total windows")

    # Estimate normal parameters
    mu_0 = float(np.mean(s_mean))
    sigma_0 = float(np.std(s_mean))
    logger.info(f"  mu_0={mu_0:.4f}, sigma_0={sigma_0:.4f}")

    # Threshold: h = 3 * std of normal scores
    h = 3.0 * sigma_0
    logger.info(f"  k_slack={K_SLACK:.2f}, h={h:.4f}")

    # --- Normal CUSUM: no attack ---
    normal_results = {}
    for fname in file_names:
        scores = grouped[fname]
        trace, alarm_idx = run_cusum(scores, mu_0, K_SLACK, h)
        normal_results[fname] = {
            "n_windows": len(scores),
            "false_alarm": alarm_idx is not None,
            "alarm_idx": alarm_idx,
            "peak_cusum": float(max(trace)) if trace else 0.0,
        }

    n_false_alarms = sum(1 for r in normal_results.values() if r["false_alarm"])
    false_alarm_rate = n_false_alarms / max(len(normal_results), 1)
    logger.info(f"  Normal false alarm rate: {false_alarm_rate:.3f} ({n_false_alarms}/{len(normal_results)})")

    # --- Attack simulation: inject at midpoint ---
    attack_results = {}
    latencies = []
    n_detected = 0
    n_missed = 0

    for fname in file_names:
        scores = grouped[fname]
        sim = _simulate_attack_at_midpoint(scores, mu_0, sigma_0, K_SLACK, h, rng)
        attack_results[fname] = sim
        if sim["detected"]:
            n_detected += 1
            if sim["latency"] is not None:
                latencies.append(sim["latency"])
        elif sim["w_star"] is not None:
            n_missed += 1

    total_attacks = n_detected + n_missed
    detection_rate = n_detected / max(total_attacks, 1)
    mean_latency = float(np.mean(latencies)) if latencies else float("inf")
    median_latency = float(np.median(latencies)) if latencies else float("inf")

    logger.info(
        f"  Attack detection rate: {detection_rate:.3f}, "
        f"mean latency: {mean_latency:.1f} windows, "
        f"median latency: {median_latency:.1f} windows"
    )

    # Save example traces for figure generation
    cusum_examples = {"normal": {}, "attacked": {}}
    example_normal = file_names[:3]
    example_attack = file_names[:3]

    for fname in example_normal:
        trace, _ = run_cusum(grouped[fname], mu_0, K_SLACK, h)
        cusum_examples["normal"][fname] = trace

    for fname in example_attack:
        sim = attack_results.get(fname, {})
        if sim.get("trace"):
            cusum_examples["attacked"][fname] = {
                "trace": sim["trace"],
                "w_star": sim["w_star"],
                "alarm_idx": sim["alarm_idx"],
            }

    cusum_examples["h"] = h
    cusum_examples["mu_0"] = mu_0
    cusum_examples["k_slack"] = K_SLACK
    torch.save(cusum_examples, fold_dir / "cusum_examples.pt")

    fold_result = {
        "fold": fold,
        "n_windows": len(s_mean),
        "n_files": len(file_names),
        "mu_0": mu_0,
        "sigma_0": sigma_0,
        "k_slack": K_SLACK,
        "h": h,
        "normal_false_alarm_rate": false_alarm_rate,
        "attack_detection_rate": detection_rate,
        "mean_detection_latency": mean_latency,
        "median_detection_latency": median_latency,
        "n_detected": n_detected,
        "n_missed": n_missed,
        "latencies": latencies,
    }

    save_results(fold_result, fold_dir, "fold_results.json")
    return fold_result


def aggregate(fold_results: list, output_dir: Path) -> dict:
    """Cross-fold aggregation of CUSUM results."""
    fa_rates = [fr["normal_false_alarm_rate"] for fr in fold_results]
    det_rates = [fr["attack_detection_rate"] for fr in fold_results]
    mean_lats = [fr["mean_detection_latency"] for fr in fold_results if np.isfinite(fr["mean_detection_latency"])]
    median_lats = [fr["median_detection_latency"] for fr in fold_results if np.isfinite(fr["median_detection_latency"])]

    # Pool all latencies across folds
    all_latencies = []
    for fr in fold_results:
        all_latencies.extend(fr.get("latencies", []))

    agg = {
        "n_folds": len(fold_results),
        "k_slack": K_SLACK,
        "h_formula": "3 * std(normal S_mean)",
        "normal_false_alarm_rate": {
            "mean": float(np.mean(fa_rates)),
            "std": float(np.std(fa_rates)),
            "per_fold": fa_rates,
        },
        "attack_detection_rate": {
            "mean": float(np.mean(det_rates)),
            "std": float(np.std(det_rates)),
            "per_fold": det_rates,
        },
        "mean_detection_latency": {
            "mean": float(np.mean(mean_lats)) if mean_lats else None,
            "std": float(np.std(mean_lats)) if mean_lats else None,
        },
        "median_detection_latency": {
            "mean": float(np.mean(median_lats)) if median_lats else None,
        },
        "pooled_latency_stats": {
            "mean": float(np.mean(all_latencies)) if all_latencies else None,
            "median": float(np.median(all_latencies)) if all_latencies else None,
            "p25": float(np.percentile(all_latencies, 25)) if all_latencies else None,
            "p75": float(np.percentile(all_latencies, 75)) if all_latencies else None,
            "n_total": len(all_latencies),
        },
    }

    save_results(agg, output_dir)
    return agg


def generate_figures(agg_results: dict, output_dir: Path, fold_results: list):
    """Generate CUSUM chart figures."""
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

    # Figure 1: CUSUM charts (normal vs attacked)
    traces_path = output_dir / "fold_1" / "cusum_examples.pt"
    if not traces_path.exists() and fold_results:
        traces_path = output_dir / f"fold_{fold_results[0]['fold']}" / "cusum_examples.pt"

    if traces_path.exists():
        examples = torch.load(traces_path, weights_only=False)
        h_val = examples.get("h", 1.0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Normal traces
        ax = axes[0]
        for fname, trace in examples.get("normal", {}).items():
            ax.plot(trace, alpha=0.7, label=fname)
        ax.axhline(y=h_val, color="red", linestyle="--", linewidth=1.0,
                   label=f"h={h_val:.2f}")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("CUSUM Statistic $C_w$")
        ax.set_title("Normal Sequences")
        ax.legend(fontsize=7)

        # Attacked traces
        ax = axes[1]
        for fname, data in examples.get("attacked", {}).items():
            trace = data["trace"]
            w_star = data["w_star"]
            alarm = data["alarm_idx"]
            ax.plot(trace, alpha=0.7, label=fname)
            ax.axvline(x=w_star, color="orange", linestyle=":", alpha=0.6)
            if alarm is not None:
                ax.plot(alarm, trace[alarm], "rv", markersize=8)

        ax.axhline(y=h_val, color="red", linestyle="--", linewidth=1.0,
                   label=f"h={h_val:.2f}")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("CUSUM Statistic $C_w$")
        ax.set_title("Attacked Sequences (attack onset = orange)")
        ax.legend(fontsize=7)

        fig.suptitle("CUSUM Change-Point Detection", fontsize=13)
        fig.tight_layout()
        save_figure(fig, fig_dir, "cusum_charts.pdf")
        plt.close(fig)

    # Figure 2: Detection latency histogram
    all_latencies = []
    for fr in fold_results:
        all_latencies.extend(fr.get("latencies", []))

    if all_latencies:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(all_latencies, bins=30, color="#4C72B0", edgecolor="black", alpha=0.8)
        ax.axvline(x=np.mean(all_latencies), color="red", linestyle="--",
                   label=f"Mean={np.mean(all_latencies):.1f}")
        ax.axvline(x=np.median(all_latencies), color="orange", linestyle="--",
                   label=f"Median={np.median(all_latencies):.1f}")
        ax.set_xlabel("Detection Latency (windows after attack onset)")
        ax.set_ylabel("Count")
        ax.set_title("CUSUM Detection Latency Distribution")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, fig_dir, "cusum_latency_histogram.pdf")
        plt.close(fig)

    logger.info(f"Figures saved to {fig_dir}")


def main():
    parser = get_default_parser("CUSUM Detection", 18)
    args = parser.parse_args()

    setup_logging("exp18_cusum")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Folds: {args.folds}")

    fold_results = []
    for fold in args.folds:
        result = run_fold(fold, args.output_dir, seed=args.seed)
        fold_results.append(result)

    agg = aggregate(fold_results, args.output_dir)
    generate_figures(agg, args.output_dir, fold_results)

    # Summary
    det_rate = agg.get("attack_detection_rate", {}).get("mean", 0)
    lat = agg.get("pooled_latency_stats", {}).get("median")
    fa = agg.get("normal_false_alarm_rate", {}).get("mean", 0)
    finding = f"Det rate={det_rate:.3f}, median latency={lat}w, FA rate={fa:.3f}"

    append_to_summary(
        exp_id=18,
        name="CUSUM Detection",
        status="done",
        auroc="--",
        finding=finding,
        output_dir=BASE_DIR,
    )

    logger.info("Experiment 18 complete.")


if __name__ == "__main__":
    main()
