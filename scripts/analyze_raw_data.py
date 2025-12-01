#!/usr/bin/env python3
"""
Comprehensive Raw Data Analysis Script

This script performs complete statistical analysis and visualization of raw G-code
sensor data BEFORE preprocessing. It generates:
- Statistical summaries and data quality reports
- Distribution and correlation analysis
- Temporal pattern detection
- G-code token analysis
- Publication-quality visualizations
- Comprehensive markdown report

Usage:
    python analyze_raw_data.py --data-dir data/ --output-dir analysis_results
"""
import argparse
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from miracle.dataset.data_exploration import RawDataExplorer
from miracle.dataset.gcode_analysis import GCodeAnalyzer
from miracle.dataset.visualization_utils import (
    setup_publication_style,
    plot_sensor_distributions,
    plot_correlation_heatmap,
    plot_time_series,
    plot_pca_2d,
    plot_tsne_2d,
    plot_outliers,
    plot_token_frequencies,
    plot_cooccurrence_matrix,
    plot_missing_values,
    plot_autocorrelation,
    save_figure,
    detect_operation_type,
    get_operation_color_map,
)


def generate_markdown_report(
    data_explorer: RawDataExplorer,
    gcode_analyzer: GCodeAnalyzer,
    output_dir: Path,
    figure_paths: dict,
    operation_types: List[str] = None,
    color_map: Dict[str, str] = None
) -> str:
    """
    Generate comprehensive markdown report.

    Args:
        data_explorer: RawDataExplorer with computed statistics
        gcode_analyzer: GCodeAnalyzer with computed statistics
        output_dir: Output directory
        figure_paths: Dictionary mapping figure names to file paths
        operation_types: List of operation types detected from filenames
        color_map: Color mapping for operation types

    Returns:
        Path to generated markdown file
    """
    report_lines = []

    # Header
    report_lines.append("# Raw Data Analysis Report")
    report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\n**Data Directory:** {data_explorer.csv_files[0].parent}")
    report_lines.append(f"\n**Number of Files:** {len(data_explorer.csv_files)}")
    report_lines.append("\n---\n")

    # Executive Summary
    report_lines.append("## Executive Summary\n")
    report_lines.append(f"- **Total Samples:** {len(data_explorer.combined_df):,}")
    report_lines.append(f"- **Total Files:** {len(data_explorer.data_frames)}")

    feature_cols = data_explorer.identify_feature_columns()
    report_lines.append(f"- **Continuous Features:** {len(feature_cols['continuous'])}")
    report_lines.append(f"- **Categorical Features:** {len(feature_cols['categorical'])}")

    if 'vocabulary_stats' in gcode_analyzer.statistics:
        vocab_stats = gcode_analyzer.statistics['vocabulary_stats']
        report_lines.append(f"- **G-code Vocabulary Size:** {vocab_stats['vocab_size']}")
        report_lines.append(f"- **Total G-code Commands:** {vocab_stats['total_tokens']:,}")

    # Operation types summary
    if operation_types:
        unique_ops = sorted(set(operation_types))
        report_lines.append(f"- **Operation Types Detected:** {len(unique_ops)} ({', '.join([op.capitalize() for op in unique_ops])})")

    report_lines.append("\n")

    # Operation Types Section
    if operation_types and color_map:
        report_lines.append("## Operation Types\n")
        report_lines.append("Detected operation types from filenames:\n")
        report_lines.append("\n| Operation Type | Files | Color |\n")
        report_lines.append("|----------------|-------|-------|\n")

        for op_type in sorted(set(operation_types)):
            count = operation_types.count(op_type)
            color_hex = color_map.get(op_type, '#7f7f7f')
            # Create a colored circle using HTML/markdown
            report_lines.append(f"| {op_type.capitalize()} | {count} | <span style='color:{color_hex}'>●</span> `{color_hex}` |\n")

        report_lines.append("\n")

    # Data Quality Check
    quality_report = data_explorer.check_data_quality()
    report_lines.append("## Data Quality Assessment\n")

    if quality_report['warnings']:
        report_lines.append("### ⚠️ Warnings\n")
        for warning in quality_report['warnings']:
            report_lines.append(f"- {warning}")
        report_lines.append("\n")

    if quality_report['recommendations']:
        report_lines.append("### 💡 Recommendations\n")
        for rec in quality_report['recommendations']:
            report_lines.append(f"- {rec}")
        report_lines.append("\n")

    # Sensor Statistics
    if 'sensor_stats' in data_explorer.statistics:
        report_lines.append("## Sensor Statistics\n")

        sensor_stats = data_explorer.statistics['sensor_stats']

        # Summary table
        report_lines.append("### Statistical Summary\n")
        report_lines.append("| Sensor | Mean | Std | Min | Max | Skewness | Kurtosis | Missing % |")
        report_lines.append("|--------|------|-----|-----|-----|----------|----------|-----------|")

        for _, row in sensor_stats.iterrows():
            report_lines.append(
                f"| {row['sensor']} | {row['mean']:.3f} | {row['std']:.3f} | "
                f"{row['min']:.3f} | {row['max']:.3f} | {row['skewness']:.2f} | "
                f"{row['kurtosis']:.2f} | {row['missing_pct']:.2f}% |"
            )

        report_lines.append("\n")

        # Highlight issues
        high_skew = sensor_stats[abs(sensor_stats['skewness']) > 2.0]
        if not high_skew.empty:
            report_lines.append("### Highly Skewed Sensors\n")
            report_lines.append("These sensors may benefit from log transformation or robust scaling:\n")
            for _, row in high_skew.iterrows():
                report_lines.append(f"- **{row['sensor']}**: skewness = {row['skewness']:.2f}")
            report_lines.append("\n")

    # Distribution Analysis
    if 'distributions' in data_explorer.statistics:
        report_lines.append("## Distribution Analysis\n")

        dist_df = data_explorer.statistics['distributions']

        normal_sensors = dist_df[dist_df['is_normal_shapiro'] == True]
        non_normal_sensors = dist_df[dist_df['is_normal_shapiro'] == False]

        report_lines.append(f"- **Normal Distributions (Shapiro-Wilk p > 0.05):** {len(normal_sensors)}/{len(dist_df)}")
        report_lines.append(f"- **Non-Normal Distributions:** {len(non_normal_sensors)}/{len(dist_df)}")

        if not non_normal_sensors.empty:
            report_lines.append("\n**Non-normal sensors** (consider robust preprocessing):")
            for _, row in non_normal_sensors.head(10).iterrows():
                report_lines.append(f"- {row['sensor']} (p={row['shapiro_p']:.4f})")

        report_lines.append("\n")

    # Correlation Analysis
    if 'high_correlations' in data_explorer.statistics:
        report_lines.append("## Correlation Analysis\n")

        high_corr = data_explorer.statistics['high_correlations']

        if high_corr:
            report_lines.append(f"**Highly Correlated Pairs (|r| > 0.95):** {len(high_corr)}\n")
            report_lines.append("| Sensor 1 | Sensor 2 | Correlation |")
            report_lines.append("|----------|----------|-------------|")

            for sensor1, sensor2, corr in high_corr[:15]:  # Show top 15
                report_lines.append(f"| {sensor1} | {sensor2} | {corr:.3f} |")

            report_lines.append("\n💡 Consider removing one sensor from each highly correlated pair.\n")
        else:
            report_lines.append("✅ No highly correlated sensor pairs found (|r| > 0.95)\n")

    # Outlier Analysis
    if 'outlier_counts' in data_explorer.statistics:
        report_lines.append("## Outlier Analysis\n")

        outlier_counts = data_explorer.statistics['outlier_counts']
        total_samples = len(data_explorer.combined_df)

        high_outlier_sensors = {k: v for k, v in outlier_counts.items()
                               if v > total_samples * 0.05}

        if high_outlier_sensors:
            report_lines.append(f"**Sensors with >5% Outliers:** {len(high_outlier_sensors)}\n")
            report_lines.append("| Sensor | Outliers | Percentage |")
            report_lines.append("|--------|----------|------------|")

            for sensor, count in sorted(high_outlier_sensors.items(),
                                       key=lambda x: x[1], reverse=True):
                pct = 100 * count / total_samples
                report_lines.append(f"| {sensor} | {count:,} | {pct:.2f}% |")

            report_lines.append("\n")
        else:
            report_lines.append("✅ No sensors with excessive outliers (>5%)\n")

    # G-code Analysis
    report_lines.append("## G-code Token Analysis\n")

    if 'vocabulary_stats' in gcode_analyzer.statistics:
        vocab_stats = gcode_analyzer.statistics['vocabulary_stats']

        report_lines.append("### Vocabulary Statistics\n")
        report_lines.append(f"- **Vocabulary Size:** {vocab_stats['vocab_size']}")
        report_lines.append(f"- **Total Tokens:** {vocab_stats['total_tokens']:,}")
        report_lines.append(f"- **Type-Token Ratio:** {vocab_stats['type_token_ratio']:.4f}")
        report_lines.append(f"- **Top 10 Coverage:** {100*vocab_stats['coverage_top_10']:.1f}%")
        report_lines.append(f"- **Singleton Tokens:** {vocab_stats['singleton_count']} ({vocab_stats['singleton_pct']:.1f}%)")
        report_lines.append("\n")

        report_lines.append("### Most Common Tokens\n")
        report_lines.append("| Rank | Token | Frequency |")
        report_lines.append("|------|-------|-----------|")

        for idx, (token, count) in enumerate(vocab_stats['most_common_10'], 1):
            pct = 100 * count / vocab_stats['total_tokens']
            report_lines.append(f"| {idx} | `{token}` | {count:,} ({pct:.1f}%) |")

        report_lines.append("\n")

    # Rare Tokens
    if 'rare_tokens' in gcode_analyzer.statistics:
        rare_stats = gcode_analyzer.statistics['rare_tokens']

        report_lines.append("### Rare Tokens Analysis\n")
        report_lines.append(f"- **Rare Tokens (< {rare_stats['threshold']} occurrences):** {rare_stats['num_rare_tokens']} ({rare_stats['pct_rare_tokens']:.1f}%)")
        report_lines.append(f"- **Total Rare Occurrences:** {rare_stats['total_rare_occurrences']:,} ({rare_stats['pct_rare_occurrences']:.2f}%)")

        if rare_stats['rare_tokens']:
            report_lines.append("\n**Rarest Tokens:**")
            for token, count in rare_stats['rare_tokens'][:10]:
                report_lines.append(f"- `{token}`: {count} occurrences")

        report_lines.append("\n")

    # Sequence Analysis
    if 'sequence_analysis' in gcode_analyzer.statistics:
        seq_stats = gcode_analyzer.statistics['sequence_analysis']

        report_lines.append("### Sequence Pattern Analysis\n")
        report_lines.append(f"- **Window Size:** {seq_stats['window_size']}")
        report_lines.append(f"- **Unique Sequences:** {seq_stats['unique_sequences']:,} / {seq_stats['total_sequences']:,}")
        report_lines.append(f"- **Sequence Diversity:** {seq_stats['sequence_diversity']:.4f}")
        report_lines.append(f"- **Repeated Sequences:** {seq_stats['repeated_sequences']:,}")
        report_lines.append("\n")

    # Face vs Pocket Detection
    if 'face_vs_pocket' in gcode_analyzer.statistics:
        fvp_stats = gcode_analyzer.statistics['face_vs_pocket']

        report_lines.append("### Face vs Pocket Pattern Detection\n")
        report_lines.append(f"- **Z Motion Ratio:** {100*fvp_stats['z_motion_ratio']:.1f}%")
        report_lines.append(f"- **Arc Command Ratio:** {100*fvp_stats['arc_command_ratio']:.1f}%")
        report_lines.append(f"- **Rapid-to-Linear Ratio:** {fvp_stats['rapid_to_linear_ratio']:.2f}")
        report_lines.append(f"\n**Interpretation:** {fvp_stats['interpretation']}\n")

    # Per-File Statistics
    if 'per_file_stats' in data_explorer.statistics:
        report_lines.append("## Per-File Statistics\n")

        file_stats = data_explorer.statistics['per_file_stats']

        report_lines.append("| File | Rows | Sensors | Missing % |")
        report_lines.append("|------|------|---------|-----------|")

        for _, row in file_stats.iterrows():
            report_lines.append(
                f"| {row['file']} | {row['num_rows']:,} | {row['num_sensors']} | "
                f"{row.get('missing_pct', 0):.2f}% |"
            )

        report_lines.append("\n")

    # Visualizations
    report_lines.append("## Visualizations\n")

    for fig_name, fig_path in figure_paths.items():
        rel_path = fig_path.relative_to(output_dir)
        # Convert spaces to more readable format
        title = fig_name.replace('_', ' ').title()
        report_lines.append(f"### {title}\n")
        report_lines.append(f"![{title}]({rel_path})\n")

    # Recommendations
    report_lines.append("## Preprocessing Recommendations\n")

    recommendations = []

    # Based on distributions
    if 'distributions' in data_explorer.statistics:
        dist_df = data_explorer.statistics['distributions']
        non_normal = len(dist_df[dist_df['is_normal_shapiro'] == False])

        if non_normal > len(dist_df) * 0.5:
            recommendations.append(
                "✅ **Use RobustScaler or QuantileTransformer** - Over 50% of sensors are non-normally distributed"
            )
        else:
            recommendations.append(
                "✅ **StandardScaler is appropriate** - Most sensors are normally distributed"
            )

    # Based on outliers
    if 'outlier_counts' in data_explorer.statistics:
        outlier_counts = data_explorer.statistics['outlier_counts']
        total_samples = len(data_explorer.combined_df)
        high_outlier = sum(1 for count in outlier_counts.values()
                          if count > total_samples * 0.05)

        if high_outlier > 0:
            recommendations.append(
                f"⚠️ **Investigate {high_outlier} sensors with >5% outliers** - May indicate sensor issues or interesting events"
            )

    # Based on correlations
    if 'high_correlations' in data_explorer.statistics:
        high_corr = data_explorer.statistics['high_correlations']

        if high_corr:
            recommendations.append(
                f"✅ **Remove redundant features** - {len(high_corr)} highly correlated sensor pairs found"
            )

    # Based on missing values
    if 'sensor_stats' in data_explorer.statistics:
        sensor_stats = data_explorer.statistics['sensor_stats']
        high_missing = sensor_stats[sensor_stats['missing_pct'] > 10]

        if not high_missing.empty:
            recommendations.append(
                f"⚠️ **Handle {len(high_missing)} sensors with >10% missing values** - Consider imputation or removal"
            )

    # Based on vocabulary
    if 'vocabulary_stats' in gcode_analyzer.statistics:
        vocab_stats = gcode_analyzer.statistics['vocabulary_stats']

        if vocab_stats['singleton_pct'] > 20:
            recommendations.append(
                f"✅ **Consider minimum frequency threshold for tokens** - {vocab_stats['singleton_pct']:.1f}% are singletons"
            )

    for rec in recommendations:
        report_lines.append(f"{rec}\n")

    report_lines.append("\n---\n")
    report_lines.append(f"\n*Report generated by analyze_raw_data.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Write report
    report_path = output_dir / "RAW_DATA_ANALYSIS_REPORT.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\n✅ Report saved to: {report_path}")

    return report_path


def main():
    parser = argparse.ArgumentParser(description="Analyze raw G-code sensor data before preprocessing")
    parser.add_argument('--data-dir', type=Path, required=True,
                       help="Directory containing aligned CSV files")
    parser.add_argument('--output-dir', type=Path, default=Path('analysis_results'),
                       help="Output directory for results")
    parser.add_argument('--sample-rate', type=float, default=None,
                       help="Sample rate for large datasets (e.g., 0.1 for 10%%)")
    parser.add_argument('--max-sensors', type=int, default=12,
                       help="Maximum sensors to plot in detailed visualizations")
    parser.add_argument('--skip-pca-tsne', action='store_true',
                       help="Skip PCA/t-SNE visualizations (faster)")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Setup plotting style
    setup_publication_style()

    # Find CSV files
    csv_files = sorted(args.data_dir.glob("*_aligned.csv"))
    if not csv_files:
        print(f"❌ No *_aligned.csv files found in {args.data_dir}")
        return

    print(f"Found {len(csv_files)} CSV files")
    print(f"Output directory: {args.output_dir}\n")

    # ========================================
    # PART 1: Sensor Data Exploration
    # ========================================
    print("="*60)
    print("PART 1: SENSOR DATA EXPLORATION")
    print("="*60)

    explorer = RawDataExplorer(csv_files)
    explorer.load_all_data(sample_rate=args.sample_rate)

    # Detect operation types from filenames
    operation_types = [detect_operation_type(f.name) for f in csv_files]
    color_map = get_operation_color_map(operation_types)

    print(f"\nDetected Operation Types:")
    for op_type in sorted(set(operation_types)):
        count = operation_types.count(op_type)
        print(f"  - {op_type.capitalize()}: {count} files (color: {color_map[op_type]})")

    feature_cols = explorer.identify_feature_columns()
    continuous_cols = feature_cols['continuous']

    print(f"\nIdentified Features:")
    print(f"  - Continuous: {len(continuous_cols)}")
    print(f"  - Categorical: {len(feature_cols['categorical'])}")
    print(f"  - G-code column: {feature_cols['gcode']}")

    # Compute statistics
    print("\n1. Computing sensor statistics...")
    sensor_stats = explorer.compute_sensor_statistics()

    print("2. Detecting outliers...")
    outlier_masks = explorer.detect_outliers(method='iqr', threshold=1.5)

    print("3. Analyzing correlations...")
    corr_matrix, high_corr_pairs = explorer.analyze_correlations(threshold=0.95)

    print("4. Analyzing distributions...")
    dist_df = explorer.analyze_distributions()

    print("5. Detecting temporal patterns...")
    autocorr_results = explorer.detect_temporal_patterns(max_lag=50)

    print("6. Checking data quality...")
    quality_report = explorer.check_data_quality()

    print("7. Computing per-file statistics...")
    file_stats = explorer.analyze_per_file_statistics()

    # ========================================
    # PART 2: G-code Token Analysis
    # ========================================
    print("\n" + "="*60)
    print("PART 2: G-CODE TOKEN ANALYSIS")
    print("="*60)

    gcode_column = feature_cols['gcode'] if feature_cols['gcode'] else 'gcode_text'
    gcode_analyzer = GCodeAnalyzer(csv_files, gcode_column=gcode_column)
    gcode_analyzer.load_all_gcodes()

    print("\n1. Computing token frequencies...")
    token_counts = gcode_analyzer.compute_token_frequencies()

    print("2. Computing vocabulary statistics...")
    vocab_stats = gcode_analyzer.compute_vocabulary_stats()

    print("3. Analyzing token sequences...")
    seq_stats = gcode_analyzer.analyze_token_sequences(window_size=10)

    print("4. Identifying rare tokens...")
    rare_stats = gcode_analyzer.identify_rare_tokens(threshold=10)

    print("5. Computing co-occurrence matrix...")
    cooccur_matrix, cooccur_labels = gcode_analyzer.compute_cooccurrence_matrix(top_n=20)

    print("6. Analyzing command complexity...")
    complexity_df = gcode_analyzer.analyze_command_complexity()

    print("7. Analyzing per-file tokens...")
    per_file_tokens = gcode_analyzer.analyze_per_file_tokens()

    print("8. Detecting face vs pocket patterns...")
    fvp_stats = gcode_analyzer.detect_face_vs_pocket_patterns()

    # ========================================
    # PART 3: Visualizations
    # ========================================
    print("\n" + "="*60)
    print("PART 3: GENERATING VISUALIZATIONS")
    print("="*60)

    figure_paths = {}

    # 1. Sensor distributions
    print("\n1. Plotting sensor distributions...")
    sensors_to_plot = continuous_cols[:args.max_sensors]
    fig = plot_sensor_distributions(explorer.combined_df, sensors_to_plot, plot_type='hist')
    paths = save_figure(fig, figures_dir / '01_sensor_distributions')
    figure_paths['sensor_distributions'] = paths[0]
    plt.close(fig)

    # 2. Correlation heatmap
    print("2. Plotting correlation heatmap...")
    fig = plot_correlation_heatmap(corr_matrix)
    paths = save_figure(fig, figures_dir / '02_correlation_heatmap')
    figure_paths['correlation_heatmap'] = paths[0]
    plt.close(fig)

    # 3. Missing values
    print("3. Plotting missing values...")
    fig = plot_missing_values(explorer.combined_df, continuous_cols)
    paths = save_figure(fig, figures_dir / '03_missing_values')
    figure_paths['missing_values'] = paths[0]
    plt.close(fig)

    # 4. Time series
    print("4. Plotting time series...")
    sensors_for_ts = continuous_cols[:6]  # Plot 6 sensors
    fig = plot_time_series(explorer.combined_df, sensors_for_ts, max_points=5000)
    paths = save_figure(fig, figures_dir / '04_time_series')
    figure_paths['time_series'] = paths[0]
    plt.close(fig)

    # 5. Autocorrelation
    print("5. Plotting autocorrelation...")
    fig = plot_autocorrelation(autocorr_results, sensors=sensors_to_plot)
    paths = save_figure(fig, figures_dir / '05_autocorrelation')
    figure_paths['autocorrelation'] = paths[0]
    plt.close(fig)

    # 6. Outlier example
    print("6. Plotting outlier analysis (example)...")
    if outlier_masks:
        example_sensor = list(outlier_masks.keys())[0]
        fig = plot_outliers(explorer.combined_df, example_sensor, outlier_masks[example_sensor])
        paths = save_figure(fig, figures_dir / '06_outlier_example')
        figure_paths['outlier_analysis'] = paths[0]
        plt.close(fig)

    # 7. Token frequencies
    print("7. Plotting token frequencies...")
    fig = plot_token_frequencies(token_counts, top_n=25)
    paths = save_figure(fig, figures_dir / '07_token_frequencies')
    figure_paths['token_frequencies'] = paths[0]
    plt.close(fig)

    # 8. Co-occurrence matrix
    print("8. Plotting co-occurrence matrix...")
    fig = plot_cooccurrence_matrix(cooccur_matrix, cooccur_labels)
    paths = save_figure(fig, figures_dir / '08_cooccurrence_matrix')
    figure_paths['cooccurrence_matrix'] = paths[0]
    plt.close(fig)

    # 9. PCA (optional)
    if not args.skip_pca_tsne:
        print("9. Plotting PCA projection (color-coded by operation type)...")
        # Sample data for PCA with labels
        sample_size = min(5000, len(explorer.combined_df))

        # Create labels for each sample based on file origin
        # Build a mapping from operation type to label index
        unique_op_types = sorted(set(operation_types))
        op_type_to_label = {op_type: idx for idx, op_type in enumerate(unique_op_types)}
        sample_label_names = {idx: op_type for op_type, idx in op_type_to_label.items()}

        # Add a temporary column to track operation type for each row
        # This is more robust than trying to match lengths
        combined_with_labels = explorer.combined_df.copy()

        # Create label array that matches combined_df length
        # We'll use the file source information if available in explorer
        file_labels = []
        for file_idx, df in enumerate(explorer.data_frames):
            op_type = operation_types[file_idx]
            file_label = op_type_to_label[op_type]
            file_labels.extend([file_label] * len(df))

        # Verify length matches (handle potential mismatch)
        if len(file_labels) != len(combined_with_labels):
            print(f"  Warning: Label length mismatch ({len(file_labels)} vs {len(combined_with_labels)})")
            print(f"  Using simplified labeling approach...")
            # Fall back to unlabeled visualization
            sample_df = explorer.combined_df[continuous_cols].sample(n=sample_size, random_state=42).fillna(0)

            fig, _ = plot_pca_2d(
                sample_df.values,
                labels=None,
                label_names=None,
                color_map=None,
                title='PCA: Raw Sensor Features'
            )
        else:
            # Labels match, proceed normally
            combined_with_labels['_operation_label'] = file_labels

            # Sample with labels
            sample_df = combined_with_labels.sample(n=sample_size, random_state=42)
            sampled_labels = sample_df['_operation_label'].values
            sample_features = sample_df[continuous_cols].fillna(0)

            fig, _ = plot_pca_2d(
                sample_features.values,
                labels=sampled_labels,
                label_names=sample_label_names,
                color_map=color_map,
                title='PCA: Raw Sensor Features by Operation Type'
            )

        paths = save_figure(fig, figures_dir / '09_pca_projection')
        figure_paths['pca_projection'] = paths[0]
        plt.close(fig)

        # 10. t-SNE (optional)
        print("10. Plotting t-SNE projection (color-coded by operation type)...")
        tsne_sample_size = min(2000, len(explorer.combined_df))

        if len(file_labels) != len(combined_with_labels):
            # Fall back to unlabeled
            tsne_df = explorer.combined_df[continuous_cols].sample(n=tsne_sample_size, random_state=42).fillna(0)

            fig, _ = plot_tsne_2d(
                tsne_df.values,
                labels=None,
                label_names=None,
                color_map=None,
                perplexity=30,
                title='t-SNE: Raw Sensor Features'
            )
        else:
            # Use labels
            tsne_sample = combined_with_labels.sample(n=tsne_sample_size, random_state=42)
            tsne_labels = tsne_sample['_operation_label'].values
            tsne_features = tsne_sample[continuous_cols].fillna(0)

            fig, _ = plot_tsne_2d(
                tsne_features.values,
                labels=tsne_labels,
                label_names=sample_label_names,
                color_map=color_map,
                perplexity=30,
                title='t-SNE: Raw Sensor Features by Operation Type'
            )

        paths = save_figure(fig, figures_dir / '10_tsne_projection')
        figure_paths['tsne_projection'] = paths[0]
        plt.close(fig)

    # ========================================
    # PART 4: Generate Report
    # ========================================
    print("\n" + "="*60)
    print("PART 4: GENERATING REPORT")
    print("="*60)

    report_path = generate_markdown_report(
        explorer,
        gcode_analyzer,
        args.output_dir,
        figure_paths,
        operation_types=operation_types,
        color_map=color_map
    )

    # Save statistics to CSV
    print("\nSaving detailed statistics to CSV...")
    sensor_stats.to_csv(args.output_dir / 'sensor_statistics.csv', index=False)
    dist_df.to_csv(args.output_dir / 'distribution_analysis.csv', index=False)
    file_stats.to_csv(args.output_dir / 'per_file_statistics.csv', index=False)
    complexity_df.to_csv(args.output_dir / 'gcode_complexity.csv', index=False)
    per_file_tokens.to_csv(args.output_dir / 'per_file_tokens.csv', index=False)

    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - Report: {report_path.name}")
    print(f"  - Figures: {figures_dir.name}/ ({len(figure_paths)} visualizations)")
    print(f"  - CSV files: 5 detailed statistics files")


if __name__ == "__main__":
    main()
