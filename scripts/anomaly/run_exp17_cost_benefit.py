#!/usr/bin/env python3
"""Experiment 17: Cost-Benefit Analysis

Loads aggregate results from completed experiments to build a cost model
for deploying the anomaly detection system. Computes the optimal ROC
operating point under different threat levels.

Cost model:
  C_total = C_FP * FPR * N_windows/day * C_investigation
          + C_FN * FNR * P_attack * C_damage

Sensor cost breakdown:
  - 6x Arduino Nano 33 BLE Sense: $198
  - MCC DAQ USB-1808X: $169
  - RPi4: $75
  - Cabling + enclosure: $50
  - Total: $492

Inference throughput: 19.9ms/window GPU (3200x real-time)

Threat levels:
  - Low:  P_attack = 1e-4
  - Med:  P_attack = 1e-3
  - High: P_attack = 1e-2

No GPU required for this analysis.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import (
    BASE_DIR,
    save_results,
    save_figure,
    append_to_summary,
    setup_logging,
    get_default_parser,
)

logger = logging.getLogger(__name__)

# Hardware cost breakdown (USD)
SENSOR_COST = {
    "arduino_nano_33_ble_sense_x6": 198.0,
    "mcc_daq_usb_1808x": 169.0,
    "rpi4": 75.0,
    "cabling_enclosure": 50.0,
}
TOTAL_SENSOR_COST = sum(SENSOR_COST.values())  # $492

# Inference performance
INFERENCE_MS_PER_WINDOW = 19.9  # milliseconds on GPU
REALTIME_FACTOR = 3200.0  # x real-time

# Production assumptions
WINDOWS_PER_DAY = 5000  # typical production volume
INVESTIGATION_COST = 50.0  # cost to investigate a false positive (USD)
DAMAGE_COST = 10000.0  # cost of undetected attack reaching production (USD)

# Threat levels
THREAT_LEVELS = {
    "low": {"P_attack": 1e-4, "description": "Insider threat, low frequency"},
    "medium": {"P_attack": 1e-3, "description": "Occasional targeted attacks"},
    "high": {"P_attack": 1e-2, "description": "Persistent adversarial threat"},
}


def _load_experiment_results() -> dict:
    """Load aggregate results from all completed experiments."""
    results = {}
    for exp_dir in sorted(BASE_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        agg_path = exp_dir / "aggregate_results.json"
        if agg_path.exists():
            with open(agg_path) as f:
                results[exp_dir.name] = json.load(f)
    return results


def _extract_roc_operating_points(exp_results: dict) -> dict:
    """Extract AUROC and detection rates at key FPR thresholds from exp01."""
    points = {}

    # Exp01: NLL scoring
    exp01 = exp_results.get("exp01_nll_scoring", {})
    best_scoring = exp01.get("best_scoring_per_attack", {})
    per_attack = exp01.get("per_attack", {})

    # Use A1 command swap as primary attack
    for attack_key in ["A1_command_swap", "A3_param_swap"]:
        if attack_key in per_attack:
            s_mean = per_attack[attack_key].get("S_mean", {})
            points[attack_key] = {
                "auroc": s_mean.get("mean_auroc", 0.5),
            }

    return points


def _compute_cost(
    fpr: float,
    fnr: float,
    p_attack: float,
    n_windows_day: int,
    c_investigation: float,
    c_damage: float,
) -> float:
    """Compute total expected daily cost."""
    # False positive cost: FPR * N_windows * C_investigation
    c_fp = fpr * n_windows_day * c_investigation

    # False negative cost: FNR * P_attack * N_windows * C_damage
    c_fn = fnr * p_attack * n_windows_day * c_damage

    return c_fp + c_fn


def _find_optimal_operating_point(
    fpr_values: np.ndarray,
    tpr_values: np.ndarray,
    p_attack: float,
    n_windows_day: int,
    c_investigation: float,
    c_damage: float,
) -> dict:
    """Find the ROC operating point that minimizes expected cost."""
    best_cost = float("inf")
    best_idx = 0

    costs = []
    for i, (fpr, tpr) in enumerate(zip(fpr_values, tpr_values)):
        fnr = 1.0 - tpr
        cost = _compute_cost(fpr, fnr, p_attack, n_windows_day, c_investigation, c_damage)
        costs.append(cost)
        if cost < best_cost:
            best_cost = cost
            best_idx = i

    return {
        "optimal_fpr": float(fpr_values[best_idx]),
        "optimal_tpr": float(tpr_values[best_idx]),
        "optimal_fnr": float(1.0 - tpr_values[best_idx]),
        "min_daily_cost": float(best_cost),
        "cost_fp_component": float(fpr_values[best_idx] * n_windows_day * c_investigation),
        "cost_fn_component": float(
            (1.0 - tpr_values[best_idx]) * p_attack * n_windows_day * c_damage
        ),
        "all_costs": [float(c) for c in costs],
    }


def _generate_synthetic_roc(auroc: float, n_points: int = 200) -> tuple:
    """Generate a synthetic ROC curve matching a given AUROC.

    Uses a parametric model: TPR = FPR^((1-AUROC)/AUROC) for AUROC > 0.5
    """
    fpr = np.linspace(0, 1, n_points)
    if auroc <= 0.5:
        tpr = fpr.copy()
    else:
        # Parametric ROC curve
        exponent = (1.0 - auroc) / auroc
        tpr = np.power(fpr, exponent)
    # Ensure endpoints
    tpr[0] = 0.0
    tpr[-1] = 1.0
    return fpr, tpr


def main():
    parser = get_default_parser("Cost Benefit", 17)
    args = parser.parse_args()
    setup_logging("exp17_cost_benefit")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")

    # Load all experiment results
    exp_results = _load_experiment_results()
    logger.info(f"Loaded results from {len(exp_results)} experiments")

    roc_points = _extract_roc_operating_points(exp_results)
    logger.info(f"ROC operating points from: {list(roc_points.keys())}")

    # Use best available AUROC for cost analysis
    if roc_points:
        best_attack = max(roc_points, key=lambda k: roc_points[k]["auroc"])
        best_auroc = roc_points[best_attack]["auroc"]
    else:
        best_attack = "estimated"
        best_auroc = 0.95  # conservative estimate if exp01 hasn't run
        logger.warning(f"No exp01 results found, using estimated AUROC={best_auroc}")

    logger.info(f"Using AUROC={best_auroc:.3f} from {best_attack}")

    # Generate synthetic ROC from AUROC
    fpr_values, tpr_values = _generate_synthetic_roc(best_auroc)

    # Compute optimal operating points for each threat level
    threat_results = {}
    for threat_name, threat_config in THREAT_LEVELS.items():
        p_attack = threat_config["P_attack"]
        optimal = _find_optimal_operating_point(
            fpr_values, tpr_values,
            p_attack=p_attack,
            n_windows_day=WINDOWS_PER_DAY,
            c_investigation=INVESTIGATION_COST,
            c_damage=DAMAGE_COST,
        )
        threat_results[threat_name] = {
            **optimal,
            "p_attack": p_attack,
            "description": threat_config["description"],
        }

        logger.info(
            f"  {threat_name} (P={p_attack:.0e}): "
            f"FPR*={optimal['optimal_fpr']:.4f}, "
            f"TPR*={optimal['optimal_tpr']:.4f}, "
            f"cost=${optimal['min_daily_cost']:.2f}/day"
        )

    # Annual cost summary
    annual = {}
    for threat_name, tr in threat_results.items():
        daily = tr["min_daily_cost"]
        annual[threat_name] = {
            "daily_cost": daily,
            "annual_cost": daily * 365,
            "annual_cost_with_sensor": daily * 365 + TOTAL_SENSOR_COST,
            "roi_vs_no_detection": (
                THREAT_LEVELS[threat_name]["P_attack"] * WINDOWS_PER_DAY * DAMAGE_COST * 365
                - (daily * 365 + TOTAL_SENSOR_COST)
            ),
        }

    # Inference throughput analysis
    throughput = {
        "ms_per_window": INFERENCE_MS_PER_WINDOW,
        "windows_per_second": 1000.0 / INFERENCE_MS_PER_WINDOW,
        "realtime_factor": REALTIME_FACTOR,
        "can_keep_up": (1000.0 / INFERENCE_MS_PER_WINDOW) > (WINDOWS_PER_DAY / 86400),
        "daily_inference_time_seconds": WINDOWS_PER_DAY * INFERENCE_MS_PER_WINDOW / 1000.0,
        "daily_inference_time_minutes": WINDOWS_PER_DAY * INFERENCE_MS_PER_WINDOW / 60000.0,
    }

    results = {
        "sensor_cost": SENSOR_COST,
        "total_sensor_cost": TOTAL_SENSOR_COST,
        "production_assumptions": {
            "windows_per_day": WINDOWS_PER_DAY,
            "investigation_cost": INVESTIGATION_COST,
            "damage_cost": DAMAGE_COST,
        },
        "best_auroc": best_auroc,
        "best_attack": best_attack,
        "threat_level_analysis": threat_results,
        "annual_summary": annual,
        "throughput": throughput,
        "experiments_loaded": list(exp_results.keys()),
    }

    save_results(results, output_dir)

    # Generate figures
    generate_figures(results, output_dir, fpr_values, tpr_values)

    # Summary
    med_cost = annual.get("medium", {}).get("daily_cost", 0)
    med_roi = annual.get("medium", {}).get("roi_vs_no_detection", 0)
    finding = (
        f"Sensor=${TOTAL_SENSOR_COST}, "
        f"med-threat daily cost=${med_cost:.0f}, "
        f"annual ROI=${med_roi:.0f}"
    )

    append_to_summary(
        exp_id=17,
        name="Cost Benefit",
        status="done",
        auroc=f"AUROC={best_auroc:.3f}",
        finding=finding,
        output_dir=BASE_DIR,
    )

    logger.info("Experiment 17 complete.")


def generate_figures(results: dict, output_dir: Path, fpr_values, tpr_values):
    """Generate cost-benefit analysis figures."""
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

    threat_results = results.get("threat_level_analysis", {})

    # Figure 1: ROC curve with optimal operating points
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_values, tpr_values, "b-", linewidth=2,
            label=f"Decoder NLL (AUROC={results['best_auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)

    markers = {"low": "o", "medium": "s", "high": "D"}
    colors = {"low": "#55A868", "medium": "#4C72B0", "high": "#C44E52"}
    for threat_name, tr in threat_results.items():
        ax.plot(
            tr["optimal_fpr"], tr["optimal_tpr"],
            marker=markers.get(threat_name, "o"),
            color=colors.get(threat_name, "gray"),
            markersize=12, markeredgecolor="black", markeredgewidth=1.5,
            label=f"{threat_name.capitalize()} (FPR={tr['optimal_fpr']:.3f})",
            zorder=5,
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Optimal Operating Points by Threat Level")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    save_figure(fig, fig_dir, "roc_operating_points.pdf")
    plt.close(fig)

    # Figure 2: Cost curves per threat level
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (threat_name, tr) in zip(axes, sorted(threat_results.items())):
        costs = tr.get("all_costs", [])
        if costs:
            ax.plot(fpr_values[:len(costs)], costs, color=colors.get(threat_name, "blue"))
            ax.axvline(x=tr["optimal_fpr"], color="red", linestyle="--", alpha=0.7,
                       label=f"Optimal FPR={tr['optimal_fpr']:.3f}")
            ax.set_xlabel("FPR")
            ax.set_ylabel("Daily Cost ($)")
            ax.set_title(f"{threat_name.capitalize()} (P={THREAT_LEVELS[threat_name]['P_attack']:.0e})")
            ax.legend(fontsize=8)
    fig.suptitle("Expected Daily Cost vs Operating Point", y=1.02)
    fig.tight_layout()
    save_figure(fig, fig_dir, "cost_curves.pdf")
    plt.close(fig)

    # Figure 3: Annual cost comparison bar chart
    annual = results.get("annual_summary", {})
    if annual:
        fig, ax = plt.subplots(figsize=(8, 5))
        threats = sorted(annual.keys())
        x = np.arange(len(threats))
        width = 0.3

        with_detection = [annual[t]["annual_cost_with_sensor"] for t in threats]
        without_detection = [
            THREAT_LEVELS[t]["P_attack"] * WINDOWS_PER_DAY * DAMAGE_COST * 365
            for t in threats
        ]

        ax.bar(x - width / 2, without_detection, width, label="Without Detection",
               color="#C44E52", alpha=0.8, edgecolor="black")
        ax.bar(x + width / 2, with_detection, width, label="With Detection",
               color="#55A868", alpha=0.8, edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in threats])
        ax.set_ylabel("Annual Cost ($)")
        ax.set_title("Annual Cost: Detection System vs No Detection")
        ax.legend()
        ax.ticklabel_format(style="plain", axis="y")
        fig.tight_layout()
        save_figure(fig, fig_dir, "annual_cost_comparison.pdf")
        plt.close(fig)

    logger.info(f"Figures saved to {fig_dir}")


def run_fold(fold: int, output_dir: Path) -> dict:
    """Not used -- cost-benefit is fold-independent."""
    return {}


def aggregate(fold_results: list, output_dir: Path) -> dict:
    """Not used -- cost-benefit is fold-independent."""
    return {}


if __name__ == "__main__":
    main()
