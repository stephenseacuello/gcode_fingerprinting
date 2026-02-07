#!/usr/bin/env python3
"""Find Minimal Sensor Set for Target Accuracy.

This script analyzes ablation results to identify the minimum set of sensors
that achieves a target accuracy threshold (default: 95%).

Uses a greedy forward selection approach:
1. Start with empty set
2. Add sensor that provides largest accuracy gain
3. Repeat until target accuracy achieved or all sensors added

Usage:
    python scripts/analysis/find_minimal_sensor_set.py \
        --results-dir outputs/ablation_study_2026_02_06/analysis \
        --output outputs/ablation_study_2026_02_06/summary/recommended_config.md \
        --target-accuracy 0.95
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Set

import pandas as pd
import numpy as np


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load results CSV."""
    csv_path = results_dir / 'all_results.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def get_sensor_loi_performance(df: pd.DataFrame) -> dict:
    """Get LOI (single sensor) performance for all sensors."""
    loi_df = df[(df['study'] == 'sensor_individual') & (df['ablation_type'] == 'loi')]

    if loi_df.empty:
        return {}

    performance = {}
    for sensor in loi_df['included'].unique():
        sensor_df = loi_df[loi_df['included'] == sensor]
        performance[sensor] = {
            'mean': sensor_df['test_accuracy'].mean(),
            'std': sensor_df['test_accuracy'].std(),
            'n': len(sensor_df),
        }

    return performance


def get_sensor_loo_importance(df: pd.DataFrame) -> dict:
    """Get LOO (leave-one-out) importance for all sensors."""
    loo_df = df[(df['study'] == 'sensor_individual') & (df['ablation_type'] == 'loo')]
    baseline_df = df[(df['study'] == 'sensor_individual') & (df['ablation_type'] == 'baseline')]

    if loo_df.empty:
        return {}

    baseline_acc = baseline_df['test_accuracy'].mean() if not baseline_df.empty else 1.0

    importance = {}
    for sensor in loo_df['removed'].unique():
        sensor_df = loo_df[loo_df['removed'] == sensor]
        acc_with_removed = sensor_df['test_accuracy'].mean()
        importance[sensor] = {
            'accuracy_drop': baseline_acc - acc_with_removed,
            'accuracy_when_removed': acc_with_removed,
            'n': len(sensor_df),
        }

    return importance


def get_modality_importance(df: pd.DataFrame) -> dict:
    """Get modality importance from LOO experiments."""
    loo_df = df[(df['study'] == 'modality_grouped') & (df['ablation_type'] == 'loo')]
    baseline_df = df[(df['study'] == 'modality_grouped') & (df['ablation_type'] == 'baseline')]

    if loo_df.empty:
        return {}

    baseline_acc = baseline_df['test_accuracy'].mean() if not baseline_df.empty else 1.0

    importance = {}
    for modality in loo_df['removed'].unique():
        mod_df = loo_df[loo_df['removed'] == modality]
        acc_with_removed = mod_df['test_accuracy'].mean()
        importance[modality] = {
            'accuracy_drop': baseline_acc - acc_with_removed,
            'accuracy_when_removed': acc_with_removed,
        }

    return importance


def greedy_sensor_selection(
    sensor_loi: dict,
    sensor_loo: dict,
    target_accuracy: float,
    all_sensors: List[str]
) -> Tuple[List[str], List[float]]:
    """
    Greedy forward selection of sensors.

    Since we don't have multi-sensor subset experiments, we use a heuristic:
    - Start with empty set
    - Add sensor with highest standalone accuracy
    - Estimate combined accuracy (with diminishing returns)
    - Continue until target reached

    Returns list of sensors and estimated accuracies at each step.
    """
    selected = []
    accuracies = []
    remaining = set(all_sensors)

    # Sort sensors by LOI performance
    sorted_sensors = sorted(
        sensor_loi.items(),
        key=lambda x: x[1]['mean'],
        reverse=True
    )

    current_acc = 0.0

    for sensor, perf in sorted_sensors:
        if sensor not in remaining:
            continue

        # Estimate accuracy gain (with diminishing returns)
        single_acc = perf['mean']

        if len(selected) == 0:
            # First sensor: use its standalone accuracy
            estimated_acc = single_acc
        else:
            # Subsequent sensors: diminishing returns model
            # Each additional sensor adds a fraction of the gap to 100%
            gap = 1.0 - current_acc
            gain = single_acc * gap * 0.5  # 50% of theoretical max gain
            estimated_acc = min(1.0, current_acc + gain)

        selected.append(sensor)
        remaining.remove(sensor)
        current_acc = estimated_acc
        accuracies.append(current_acc)

        if current_acc >= target_accuracy:
            break

    return selected, accuracies


def estimate_minimal_modalities(modality_importance: dict, target_accuracy: float) -> List[str]:
    """Estimate minimal modality set based on LOO importance."""
    # Sort by accuracy drop (most important first)
    sorted_mods = sorted(
        modality_importance.items(),
        key=lambda x: x[1]['accuracy_drop'],
        reverse=True
    )

    # If any single modality removal drops below target, it's essential
    essential = []
    optional = []

    for mod, info in sorted_mods:
        if info['accuracy_when_removed'] < target_accuracy:
            essential.append(mod)
        else:
            optional.append(mod)

    return essential, optional


def generate_report(
    sensor_selection: Tuple[List[str], List[float]],
    sensor_loi: dict,
    sensor_loo: dict,
    modality_importance: dict,
    target_accuracy: float,
    output_path: Path
):
    """Generate markdown report with recommendations."""
    selected_sensors, accuracies = sensor_selection

    lines = [
        "# Recommended Sensor Configuration",
        "",
        f"**Target Accuracy**: {target_accuracy*100:.0f}%",
        "",
        "---",
        "",
        "## Minimum Viable Sensor Set",
        "",
    ]

    if selected_sensors:
        lines.append(f"To achieve ≥{target_accuracy*100:.0f}% accuracy, use the following sensors:")
        lines.append("")

        for i, (sensor, acc) in enumerate(zip(selected_sensors, accuracies), 1):
            lines.append(f"{i}. **{sensor}** (estimated: {acc*100:.1f}%)")

        lines.append("")
        lines.append(f"**Total sensors needed**: {len(selected_sensors)} of 16")
        lines.append("")
    else:
        lines.append("No sensor selection data available.")
        lines.append("")

    # Sensor ranking
    lines.extend([
        "## Full Sensor Ranking (Standalone Performance)",
        "",
        "| Rank | Sensor | Standalone Acc | Location |",
        "|:-----|:-------|:---------------|:---------|",
    ])

    sensor_locations = {
        'frame_b1': 'Frame (back)', 'frame_b2': 'Frame (back)',
        'frame_l1': 'Frame (left)', 'frame_l2': 'Frame (left)', 'frame_l3': 'Frame (left)',
        'frame_r1': 'Frame (right)', 'frame_r2': 'Frame (right)',
        'spindle1': 'Spindle', 'spindle2': 'Spindle',
        'xa_motor': 'X-axis motor',
        'y_bed__1': 'Y-bed', 'y_bed__2': 'Y-bed', 'y_bed__3': 'Y-bed', 'y_bed__4': 'Y-bed',
        'z_gant_1': 'Z-gantry', 'z_gant_2': 'Z-gantry',
    }

    sorted_loi = sorted(sensor_loi.items(), key=lambda x: x[1]['mean'], reverse=True)
    for rank, (sensor, perf) in enumerate(sorted_loi, 1):
        location = sensor_locations.get(sensor, 'Unknown')
        lines.append(f"| {rank} | {sensor} | {perf['mean']*100:.2f}% | {location} |")

    lines.append("")

    # Sensor importance (LOO)
    lines.extend([
        "## Sensor Importance (Accuracy Drop When Removed)",
        "",
        "| Sensor | Acc Drop | Critical? |",
        "|:-------|:---------|:----------|",
    ])

    sorted_loo = sorted(sensor_loo.items(), key=lambda x: x[1]['accuracy_drop'], reverse=True)
    for sensor, info in sorted_loo:
        drop = info['accuracy_drop'] * 100
        critical = "Yes" if drop > 5 else "No"
        lines.append(f"| {sensor} | {drop:.2f}% | {critical} |")

    lines.append("")

    # Modality recommendations
    lines.extend([
        "## Modality Recommendations",
        "",
    ])

    if modality_importance:
        essential, optional = estimate_minimal_modalities(modality_importance, target_accuracy)

        if essential:
            lines.append("**Essential modalities** (removing causes drop below target):")
            for mod in essential:
                drop = modality_importance[mod]['accuracy_drop'] * 100
                lines.append(f"- {mod} (drop: {drop:.1f}%)")
            lines.append("")

        if optional:
            lines.append("**Optional modalities** (can be removed and still meet target):")
            for mod in optional:
                drop = modality_importance[mod]['accuracy_drop'] * 100
                lines.append(f"- {mod} (drop: {drop:.1f}%)")
            lines.append("")
    else:
        lines.append("No modality importance data available.")
        lines.append("")

    # Deployment recommendations
    lines.extend([
        "## Deployment Recommendations",
        "",
        "### Cost-Optimized Configuration",
        "",
    ])

    if selected_sensors:
        cost_per_sensor = 30  # Approximate cost of Arduino Nano 33 BLE Sense
        min_cost = len(selected_sensors) * cost_per_sensor
        full_cost = 16 * cost_per_sensor

        lines.extend([
            f"- **Minimum sensors**: {len(selected_sensors)}",
            f"- **Estimated cost**: ${min_cost} (vs ${full_cost} for full array)",
            f"- **Cost savings**: {(1 - len(selected_sensors)/16)*100:.0f}%",
            "",
        ])

    lines.extend([
        "### Fault-Tolerant Configuration",
        "",
        "For production deployments, consider:",
        "- Adding redundant sensors at critical locations (spindle, motor)",
        "- Monitoring sensor health via self-diagnostics",
        "- Implementing graceful degradation when sensors fail",
        "",
        "### Physical Placement Notes",
        "",
        "- Frame sensors act as bandpass filters, attenuating high-frequency noise",
        "- Spindle sensors provide direct cutting interface measurement",
        "- Bed sensors capture workpiece-side vibrations",
        "",
    ])

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Find minimal sensor set')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to analysis results directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for markdown report')
    parser.add_argument('--target-accuracy', type=float, default=0.95,
                       help='Target accuracy threshold (default: 0.95)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    df = load_results(results_dir)
    if df.empty:
        print(f"ERROR: No results found at {results_dir / 'all_results.csv'}")
        return

    print(f"Loaded {len(df)} experiment results")
    print(f"Target accuracy: {args.target_accuracy*100:.0f}%")

    # Get sensor performance metrics
    sensor_loi = get_sensor_loi_performance(df)
    sensor_loo = get_sensor_loo_importance(df)
    modality_importance = get_modality_importance(df)

    print(f"Found {len(sensor_loi)} sensors with LOI data")
    print(f"Found {len(sensor_loo)} sensors with LOO data")
    print(f"Found {len(modality_importance)} modalities with importance data")

    # Run greedy selection
    all_sensors = list(sensor_loi.keys())
    sensor_selection = greedy_sensor_selection(
        sensor_loi, sensor_loo, args.target_accuracy, all_sensors
    )

    selected, accuracies = sensor_selection
    print(f"\nMinimal sensor set ({len(selected)} sensors):")
    for s, a in zip(selected, accuracies):
        print(f"  + {s}: {a*100:.1f}%")

    # Generate report
    generate_report(
        sensor_selection, sensor_loi, sensor_loo,
        modality_importance, args.target_accuracy, output_path
    )


if __name__ == '__main__':
    main()
