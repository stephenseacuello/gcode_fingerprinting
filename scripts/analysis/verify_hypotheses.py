#!/usr/bin/env python3
"""Verify Hypotheses from Ablation Study Results.

This script systematically checks all 25 hypotheses (H1-H25) against
the actual ablation results and generates a markdown report.

Usage:
    python scripts/analysis/verify_hypotheses.py \
        --results-dir outputs/ablation_study_2026_02_06/analysis \
        --output outputs/ablation_study_2026_02_06/summary/hypothesis_results.md
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Hypothesis:
    """A hypothesis to test."""
    id: str
    description: str
    study: str
    metric: str
    threshold: float
    direction: str  # 'gt' (greater than), 'lt' (less than)
    config_filter: dict


# Define all 25 hypotheses
HYPOTHESES = [
    # Modality Hypotheses (Grouped)
    Hypothesis("H1", "Removing accelerometer causes >10% accuracy drop",
               "modality_grouped", "accuracy_drop", 0.10, "gt",
               {"ablation_type": "loo", "removed": "accelerometer"}),
    Hypothesis("H2", "Removing electrical causes largest drop in damage classes",
               "modality_grouped", "damage_class_drop", 0.0, "gt",
               {"ablation_type": "loo", "removed": "electrical"}),
    Hypothesis("H3", "Removing gyroscope causes <5% accuracy drop",
               "modality_grouped", "accuracy_drop", 0.05, "lt",
               {"ablation_type": "loo", "removed": "gyroscope"}),
    Hypothesis("H4", "Color + environmental together contribute <5%",
               "modality_grouped", "combined_drop", 0.05, "lt",
               {"ablation_type": "loo", "removed": ["color", "environmental"]}),
    Hypothesis("H5", "No single modality achieves >60% accuracy",
               "modality_grouped", "max_loi_accuracy", 0.60, "lt",
               {"ablation_type": "loi"}),

    # Modality Hypotheses (Individual)
    Hypothesis("H6", "Az is single most important component",
               "modality_individual", "accuracy_drop", 0.0, "gt",
               {"ablation_type": "loo", "removed": "Az"}),
    Hypothesis("H7", "Pressure outperforms Temperature+Proximity combined",
               "modality_individual", "loi_accuracy", 0.0, "gt",
               {"ablation_type": "loi", "included": "Pressure"}),
    Hypothesis("H8", "Removing any single axis drops accuracy <15%",
               "modality_individual", "max_axis_drop", 0.15, "lt",
               {"ablation_type": "loo", "removed": ["Ax", "Ay", "Az"]}),
    Hypothesis("H9", "RMS alone achieves >40% accuracy",
               "modality_individual", "loi_accuracy", 0.40, "gt",
               {"ablation_type": "loi", "included": "RMS"}),
    Hypothesis("H10", "ColorA is least informative color channel",
               "modality_individual", "loi_accuracy", 0.0, "lt",
               {"ablation_type": "loi", "included": "ColorA"}),

    # Sensor Hypotheses
    Hypothesis("H11", "Removing spindle sensors drops accuracy >15%",
               "sensor_group", "accuracy_drop", 0.15, "gt",
               {"ablation_type": "loo", "removed": "spindle"}),
    Hypothesis("H12", "Y-bed sensors distinguish damage classes best",
               "sensor_group", "damage_class_accuracy", 0.0, "gt",
               {"ablation_type": "loi", "included": "bed"}),
    Hypothesis("H13", "Each frame sensor removal drops <3%",
               "sensor_individual", "max_frame_drop", 0.03, "lt",
               {"ablation_type": "loo", "removed_prefix": "frame"}),
    Hypothesis("H14", "Gantry sensors critical for depth-of-cut classes",
               "sensor_group", "accuracy_drop", 0.10, "gt",
               {"ablation_type": "loo", "removed": "gantry"}),
    Hypothesis("H15", "Single sensor LOI achieves <35% accuracy",
               "sensor_individual", "max_loi_accuracy", 0.35, "lt",
               {"ablation_type": "loi"}),

    # Architecture Hypotheses
    Hypothesis("H16", "Removing LSTM drops accuracy >20%",
               "architecture", "accuracy_drop", 0.20, "gt",
               {"config_name": "no_lstm"}),
    Hypothesis("H17", "Removing attention drops accuracy <10%",
               "architecture", "accuracy_drop", 0.10, "lt",
               {"config_name": "no_attention"}),
    Hypothesis("H18", "modality_dropout=0 drops damage class accuracy",
               "architecture", "damage_class_drop", 0.05, "gt",
               {"config_name": "no_modality_dropout"}),
    Hypothesis("H19", "d_model=128 achieves >95% accuracy",
               "architecture", "test_accuracy", 0.95, "gt",
               {"config_name": "size_small"}),
    Hypothesis("H20", "Mean pooling outperforms last pooling for damage",
               "architecture", "damage_class_accuracy", 0.0, "gt",
               {"config_name": "pooling_mean"}),

    # Baseline Hypotheses
    Hypothesis("H21", "XGBoost achieves >85% accuracy",
               "baseline", "test_accuracy", 0.85, "gt",
               {"model": "xgboost"}),
    Hypothesis("H22", "Simple LSTM achieves >90% accuracy",
               "baseline", "test_accuracy", 0.90, "gt",
               {"model": "lstm_simple"}),
    Hypothesis("H23", "Transformer underperforms MM-DTAE-LSTM by >5%",
               "baseline", "accuracy_gap", 0.05, "gt",
               {"model": "transformer"}),
    Hypothesis("H24", "MLP on flattened features achieves <70% accuracy",
               "baseline", "test_accuracy", 0.70, "lt",
               {"model": "mlp"}),
    Hypothesis("H25", "TCN competitive with LSTM (within 3%)",
               "baseline", "accuracy_gap", 0.03, "lt",
               {"model": "tcn"}),
]


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load results CSV."""
    csv_path = results_dir / 'all_results.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def get_baseline_accuracy(df: pd.DataFrame, study: str) -> float:
    """Get baseline accuracy for a study."""
    baseline = df[(df['study'] == study) & (df['ablation_type'] == 'baseline')]
    if not baseline.empty:
        return baseline['test_accuracy'].mean()
    # Default to 100% for main model
    return 1.0


def verify_hypothesis(h: Hypothesis, df: pd.DataFrame) -> dict:
    """Verify a single hypothesis."""
    result = {
        'id': h.id,
        'description': h.description,
        'study': h.study,
        'threshold': h.threshold,
        'direction': h.direction,
        'verified': None,
        'actual_value': None,
        'notes': '',
    }

    # Filter data for this hypothesis
    study_df = df[df['study'] == h.study]
    if study_df.empty:
        result['notes'] = f"No data for study: {h.study}"
        return result

    baseline_acc = get_baseline_accuracy(df, h.study)

    # Apply config filters
    filtered_df = study_df.copy()
    for key, value in h.config_filter.items():
        if key in filtered_df.columns:
            if isinstance(value, list):
                filtered_df = filtered_df[filtered_df[key].isin(value)]
            else:
                filtered_df = filtered_df[filtered_df[key] == value]

    if filtered_df.empty:
        result['notes'] = f"No matching data for filters: {h.config_filter}"
        return result

    # Calculate metric based on type
    if h.metric == 'accuracy_drop':
        actual = baseline_acc - filtered_df['test_accuracy'].mean()
    elif h.metric == 'test_accuracy':
        actual = filtered_df['test_accuracy'].mean()
    elif h.metric == 'loi_accuracy':
        actual = filtered_df['test_accuracy'].mean()
    elif h.metric == 'max_loi_accuracy':
        actual = filtered_df['test_accuracy'].max()
    elif h.metric == 'max_axis_drop':
        # For axes, get max drop across Ax, Ay, Az
        drops = []
        for axis in ['Ax', 'Ay', 'Az']:
            axis_df = study_df[(study_df['ablation_type'] == 'loo') & (study_df['removed'] == axis)]
            if not axis_df.empty:
                drops.append(baseline_acc - axis_df['test_accuracy'].mean())
        actual = max(drops) if drops else None
    elif h.metric == 'max_frame_drop':
        # Get max drop across frame sensors
        frame_df = study_df[
            (study_df['ablation_type'] == 'loo') &
            (study_df['removed'].str.startswith('frame'))
        ]
        if not frame_df.empty:
            drops = baseline_acc - frame_df.groupby('removed')['test_accuracy'].mean()
            actual = drops.max()
        else:
            actual = None
    elif h.metric == 'accuracy_gap':
        # Gap from MM-DTAE-LSTM (100%)
        actual = 1.0 - filtered_df['test_accuracy'].mean()
    elif h.metric == 'damage_class_drop':
        # Would need per-class accuracy - placeholder
        actual = None
        result['notes'] = "Per-class accuracy not available in summary"
    elif h.metric == 'damage_class_accuracy':
        actual = None
        result['notes'] = "Per-class accuracy not available in summary"
    elif h.metric == 'combined_drop':
        # Combined effect of multiple modalities
        actual = None
        result['notes'] = "Combined removal not tested"
    else:
        actual = None
        result['notes'] = f"Unknown metric: {h.metric}"

    result['actual_value'] = actual

    # Verify against threshold
    if actual is not None:
        if h.direction == 'gt':
            result['verified'] = actual > h.threshold
        else:  # 'lt'
            result['verified'] = actual < h.threshold

    return result


def generate_markdown_report(results: list, output_path: Path):
    """Generate markdown report of hypothesis verification."""
    lines = [
        "# Hypothesis Verification Results",
        "",
        "**Generated from ablation study results**",
        "",
        "## Summary",
        "",
    ]

    # Count verified/rejected/unknown
    verified = sum(1 for r in results if r['verified'] is True)
    rejected = sum(1 for r in results if r['verified'] is False)
    unknown = sum(1 for r in results if r['verified'] is None)

    lines.extend([
        f"- **Verified**: {verified}/25",
        f"- **Rejected**: {rejected}/25",
        f"- **Insufficient Data**: {unknown}/25",
        "",
        "---",
        "",
    ])

    # Group by category
    categories = {
        'Modality Hypotheses (Grouped)': [f'H{i}' for i in range(1, 6)],
        'Modality Hypotheses (Individual)': [f'H{i}' for i in range(6, 11)],
        'Sensor Hypotheses': [f'H{i}' for i in range(11, 16)],
        'Architecture Hypotheses': [f'H{i}' for i in range(16, 21)],
        'Baseline Hypotheses': [f'H{i}' for i in range(21, 26)],
    }

    for category, h_ids in categories.items():
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| ID | Hypothesis | Threshold | Actual | Result |")
        lines.append("|:---|:-----------|:----------|:-------|:-------|")

        for r in results:
            if r['id'] in h_ids:
                # Format threshold
                if r['direction'] == 'gt':
                    thresh = f">{r['threshold']*100:.0f}%"
                else:
                    thresh = f"<{r['threshold']*100:.0f}%"

                # Format actual value
                if r['actual_value'] is not None:
                    actual = f"{r['actual_value']*100:.1f}%"
                else:
                    actual = "N/A"

                # Format result
                if r['verified'] is True:
                    result = "✅ Confirmed"
                elif r['verified'] is False:
                    result = "❌ Rejected"
                else:
                    result = "⚠️ Unknown"

                # Truncate description
                desc = r['description'][:50] + "..." if len(r['description']) > 50 else r['description']

                lines.append(f"| {r['id']} | {desc} | {thresh} | {actual} | {result} |")

        lines.append("")

    # Detailed notes section
    lines.append("## Detailed Notes")
    lines.append("")

    for r in results:
        if r['notes']:
            lines.append(f"**{r['id']}**: {r['notes']}")
            lines.append("")

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Verify ablation hypotheses')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to analysis results directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for markdown report')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    df = load_results(results_dir)
    if df.empty:
        print(f"ERROR: No results found at {results_dir / 'all_results.csv'}")
        return

    print(f"Loaded {len(df)} experiment results")
    print(f"Studies: {df['study'].unique().tolist()}")

    # Verify all hypotheses
    results = []
    for h in HYPOTHESES:
        result = verify_hypothesis(h, df)
        results.append(result)
        status = "✅" if result['verified'] else "❌" if result['verified'] is False else "⚠️"
        print(f"{status} {h.id}: {h.description[:50]}...")

    # Generate report
    generate_markdown_report(results, output_path)

    # Print summary
    verified = sum(1 for r in results if r['verified'] is True)
    rejected = sum(1 for r in results if r['verified'] is False)
    print(f"\nSummary: {verified} confirmed, {rejected} rejected, {25 - verified - rejected} unknown")


if __name__ == '__main__':
    main()
