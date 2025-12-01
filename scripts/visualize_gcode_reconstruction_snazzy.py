#!/usr/bin/env python3
"""
Snazzy G-Code Reconstruction Visualizations for Paper
Generates publication-quality figures showing model reconstruction performance.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from difflib import SequenceMatcher
import torch
from collections import Counter, defaultdict

# Set publication-quality matplotlib defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42,  # TrueType fonts
    'ps.fonttype': 42,
})

# Color palette
COLORS = {
    'correct': '#2ecc71',      # Emerald green
    'incorrect': '#e74c3c',    # Red
    'partial': '#f39c12',      # Orange
    'missing': '#95a5a6',      # Gray
    'command': '#3498db',      # Blue
    'param': '#9b59b6',        # Purple
    'value': '#16a085',        # Teal
    'background': '#ecf0f1',   # Light gray
}


def parse_demo_outputs(json_path: Path) -> List[Dict]:
    """Parse the demo outputs JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def tokenize_gcode(gcode_str: str) -> List[str]:
    """Tokenize a G-code string into components."""
    if gcode_str == "[empty]" or not gcode_str:
        return []

    # Simple tokenization - split by space and preserve components
    tokens = gcode_str.split()
    result = []
    for tok in tokens:
        if tok.startswith('G') or tok.startswith('M'):
            result.append(tok)
        elif tok[0] in 'XYZIJKFRS':
            result.append(tok[0])  # Parameter name
            result.append(tok[1:])  # Parameter value
        else:
            result.append(tok)
    return result


def compute_token_alignment(pred_tokens: str, actual_gcode: str) -> Tuple[List, List, List]:
    """
    Compute alignment between predicted tokens and actual G-code.
    Returns: (pred_list, actual_list, match_types)
    """
    pred_list = pred_tokens.split()
    actual_list = tokenize_gcode(actual_gcode)

    # Use SequenceMatcher to find alignment
    matcher = SequenceMatcher(None, pred_list, actual_list)
    matches = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            matches.extend(['correct'] * (i2 - i1))
        elif tag == 'replace':
            matches.extend(['partial'] * (i2 - i1))
        elif tag == 'delete':
            matches.extend(['incorrect'] * (i2 - i1))
        elif tag == 'insert':
            pass  # Handled by padding

    # Pad to max length
    max_len = max(len(pred_list), len(actual_list))
    while len(pred_list) < max_len:
        pred_list.append('')
        matches.append('missing')
    while len(actual_list) < max_len:
        actual_list.append('')

    return pred_list, actual_list, matches


def plot_token_alignment_gallery(samples: List[Dict], output_path: Path):
    """
    Figure 1: Token Alignment Gallery (DNA-style sequence alignment)
    Shows 6 diverse examples with color-coded tokens.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Select 6 diverse samples
    indices = [0, 2, 11, 14, 16, 19]  # Mix of different error patterns

    for idx, sample_idx in enumerate(indices):
        sample = samples[sample_idx]
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        pred_tokens = sample['predicted_tokens']
        actual_gcode = sample['actual_gcode']

        pred_list, actual_list, match_types = compute_token_alignment(pred_tokens, actual_gcode)

        # Create horizontal alignment visualization
        y_positions = [1, 0]  # Predicted at top, actual at bottom
        max_tokens = len(pred_list)

        # Plot predicted tokens
        for i, (token, match) in enumerate(zip(pred_list, match_types)):
            if token:
                color = COLORS[match]
                rect = mpatches.Rectangle(
                    (i, y_positions[0] - 0.3), 0.9, 0.6,
                    facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8
                )
                ax.add_patch(rect)
                ax.text(i + 0.45, y_positions[0], token,
                       ha='center', va='center', fontsize=8, weight='bold')

        # Plot actual tokens
        for i, token in enumerate(actual_list):
            if token:
                rect = mpatches.Rectangle(
                    (i, y_positions[1] - 0.3), 0.9, 0.6,
                    facecolor=COLORS['background'], edgecolor='black',
                    linewidth=1.5, alpha=0.6
                )
                ax.add_patch(rect)
                ax.text(i + 0.45, y_positions[1], token,
                       ha='center', va='center', fontsize=8)

        # Draw connection lines
        for i in range(min(len(pred_list), len(actual_list))):
            if pred_list[i] and actual_list[i]:
                linestyle = '-' if match_types[i] == 'correct' else '--'
                linewidth = 2 if match_types[i] == 'correct' else 1
                ax.plot([i + 0.45, i + 0.45],
                       [y_positions[0] - 0.3, y_positions[1] + 0.3],
                       linestyle=linestyle, linewidth=linewidth,
                       color='gray', alpha=0.5)

        # Formatting
        ax.set_xlim(-0.5, max_tokens)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(['Predicted', 'Actual'])
        ax.set_xticks([])
        ax.set_title(f'Sample {sample_idx}', weight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['correct'], label='Correct'),
        mpatches.Patch(facecolor=COLORS['partial'], label='Partial Match'),
        mpatches.Patch(facecolor=COLORS['incorrect'], label='Incorrect'),
        mpatches.Patch(facecolor=COLORS['missing'], label='Missing'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False)

    plt.suptitle('Token-Level Alignment: Predicted vs Actual G-Code',
                fontsize=16, weight='bold', y=0.995)

    # Save in multiple formats
    for ext in ['png', 'pdf', 'svg']:
        output_file = output_path / f'figure1_token_alignment.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Figure 1 saved: {output_path / 'figure1_token_alignment.*'}")


def plot_gcode_diff_viewer(samples: List[Dict], output_path: Path):
    """
    Figure 2: G-Code Diff Viewer with syntax highlighting
    Shows character-level differences with clear visual indicators.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle('G-Code Reconstruction: Predicted vs Actual',
                fontsize=16, weight='bold')

    # Select 4 representative samples
    sample_indices = [0, 11, 14, 16]

    for ax_idx, sample_idx in enumerate(sample_indices):
        ax = axes[ax_idx]
        sample = samples[sample_idx]

        reconstructed = sample['reconstructed_gcode']
        actual = sample['actual_gcode']

        # Calculate edit distance
        matcher = SequenceMatcher(None, reconstructed, actual)
        similarity = matcher.ratio() * 100

        # Display strings with highlighting
        y_pos = [0.7, 0.3]

        # Predicted
        ax.text(0.02, y_pos[0], 'Predicted:', weight='bold', fontsize=11,
               transform=ax.transAxes)
        ax.text(0.18, y_pos[0], reconstructed if reconstructed != "[empty]" else "∅",
               fontsize=12, family='monospace',
               bbox=dict(boxstyle='round', facecolor=COLORS['incorrect'], alpha=0.3),
               transform=ax.transAxes)

        # Actual
        ax.text(0.02, y_pos[1], 'Actual:', weight='bold', fontsize=11,
               transform=ax.transAxes)
        ax.text(0.18, y_pos[1], actual, fontsize=12, family='monospace',
               bbox=dict(boxstyle='round', facecolor=COLORS['correct'], alpha=0.3),
               transform=ax.transAxes)

        # Similarity score
        color = COLORS['correct'] if similarity > 80 else \
                COLORS['partial'] if similarity > 50 else COLORS['incorrect']
        ax.text(0.88, 0.5, f'{similarity:.1f}% match',
               fontsize=11, weight='bold', color=color,
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'Sample {sample_idx}', loc='left', fontsize=10, style='italic')

    plt.tight_layout()

    # Save in multiple formats
    for ext in ['png', 'pdf', 'svg']:
        output_file = output_path / f'figure2_diff_viewer.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Figure 2 saved: {output_path / 'figure2_diff_viewer.*'}")


def plot_reconstruction_pipeline(samples: List[Dict], output_path: Path):
    """
    Figure 3: Reconstruction Success Pipeline
    Sankey-style diagram showing success/failure at each stage.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate statistics
    total_samples = len(samples)
    non_empty_recon = sum(1 for s in samples if s['reconstructed_gcode'] != "[empty]")
    matches = sum(1 for s in samples if s['match'])

    # Define stages
    stages = ['Token\nPrediction', 'Grammar\nValidation', 'Exact\nMatch']
    stage_counts = [total_samples, non_empty_recon, matches]
    stage_x = [0.2, 0.5, 0.8]

    # Draw flow diagram
    for i in range(len(stages)):
        # Draw stage box
        count = stage_counts[i]
        percentage = (count / total_samples) * 100

        color = COLORS['correct'] if percentage > 80 else \
                COLORS['partial'] if percentage > 50 else COLORS['incorrect']

        rect = mpatches.FancyBboxPatch(
            (stage_x[i] - 0.08, 0.4), 0.16, 0.2,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor='black', linewidth=2, alpha=0.7
        )
        ax.add_patch(rect)

        # Add text
        ax.text(stage_x[i], 0.5, stages[i],
               ha='center', va='center', fontsize=13, weight='bold')
        ax.text(stage_x[i], 0.35, f'{count}/{total_samples}',
               ha='center', va='top', fontsize=11)
        ax.text(stage_x[i], 0.32, f'({percentage:.1f}%)',
               ha='center', va='top', fontsize=9, style='italic')

        # Draw arrow to next stage
        if i < len(stages) - 1:
            arrow = mpatches.FancyArrowPatch(
                (stage_x[i] + 0.08, 0.5), (stage_x[i + 1] - 0.08, 0.5),
                arrowstyle='->', mutation_scale=30, linewidth=3,
                color='gray', alpha=0.6
            )
            ax.add_patch(arrow)

            # Show dropout
            dropout = stage_counts[i] - stage_counts[i + 1]
            if dropout > 0:
                ax.text((stage_x[i] + stage_x[i + 1]) / 2, 0.25,
                       f'↓ {dropout} failed', ha='center',
                       fontsize=9, color=COLORS['incorrect'], style='italic')

    # Add failure annotations
    grammar_failures = total_samples - non_empty_recon
    ax.text(0.5, 0.15,
           f'Grammar Failures: {grammar_failures} ({grammar_failures/total_samples*100:.1f}%)',
           ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor=COLORS['incorrect'], alpha=0.2))

    ax.text(0.8, 0.15,
           f'Near Misses: {non_empty_recon - matches}',
           ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor=COLORS['partial'], alpha=0.2))

    # Formatting
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.7)
    ax.axis('off')
    ax.set_title('G-Code Reconstruction Pipeline: Success Rate by Stage',
                fontsize=16, weight='bold', pad=20)

    # Save in multiple formats
    for ext in ['png', 'pdf', 'svg']:
        output_file = output_path / f'figure3_reconstruction_pipeline.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Figure 3 saved: {output_path / 'figure3_reconstruction_pipeline.*'}")


def plot_error_heatmap(samples: List[Dict], output_path: Path):
    """
    Figure 4: Error Pattern Heatmap
    Shows which token positions have errors across samples.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Build error matrix
    max_tokens = 0
    error_data = []

    for sample in samples:
        pred_list, actual_list, match_types = compute_token_alignment(
            sample['predicted_tokens'], sample['actual_gcode']
        )
        max_tokens = max(max_tokens, len(match_types))

        # Convert to numeric: 1 = correct, 0.5 = partial, 0 = incorrect/missing
        error_row = []
        for match in match_types:
            if match == 'correct':
                error_row.append(1.0)
            elif match == 'partial':
                error_row.append(0.5)
            else:
                error_row.append(0.0)
        error_data.append(error_row)

    # Pad rows to same length
    for row in error_data:
        while len(row) < max_tokens:
            row.append(np.nan)

    error_matrix = np.array(error_data)

    # Plot heatmap
    sns.heatmap(error_matrix, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Match Quality'},
                xticklabels=range(max_tokens),
                yticklabels=[f'S{i}' for i in range(len(samples))])

    ax.set_xlabel('Token Position', fontsize=12, weight='bold')
    ax.set_ylabel('Sample ID', fontsize=12, weight='bold')
    ax.set_title('Error Pattern Heatmap: Token-Level Match Quality Across Samples',
                fontsize=14, weight='bold', pad=15)

    # Add annotations for systematic errors
    # Calculate per-position accuracy
    position_acc = np.nanmean(error_matrix, axis=0)
    problematic_positions = np.where(position_acc < 0.3)[0]

    if len(problematic_positions) > 0:
        ax.text(0.5, -0.08,
               f'⚠️ Problematic positions: {list(problematic_positions)}',
               ha='center', transform=ax.transAxes,
               fontsize=10, color=COLORS['incorrect'],
               bbox=dict(boxstyle='round', facecolor=COLORS['incorrect'], alpha=0.2))

    plt.tight_layout()

    # Save in multiple formats
    for ext in ['png', 'pdf', 'svg']:
        output_file = output_path / f'figure4_error_heatmap.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Figure 4 saved: {output_path / 'figure4_error_heatmap.*'}")


def plot_grammar_violations(samples: List[Dict], output_path: Path):
    """
    Figure 5: Grammar Violation Breakdown
    Shows types of grammar validation failures.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Categorize failures
    categories = {
        'Empty Reconstruction': 0,
        'Missing Command': 0,
        'Value Mismatch': 0,
        'Successful': 0,
    }

    examples = defaultdict(list)

    for sample in samples:
        recon = sample['reconstructed_gcode']
        actual = sample['actual_gcode']

        if recon == "[empty]":
            categories['Empty Reconstruction'] += 1
            examples['Empty Reconstruction'].append(sample['sample_id'])
        elif sample['match']:
            categories['Successful'] += 1
            examples['Successful'].append(sample['sample_id'])
        elif 'G1' in actual and 'G1' not in recon:
            categories['Missing Command'] += 1
            examples['Missing Command'].append(sample['sample_id'])
        else:
            categories['Value Mismatch'] += 1
            examples['Value Mismatch'].append(sample['sample_id'])

    # Plot 1: Donut chart
    colors_list = [COLORS['incorrect'], COLORS['partial'],
                  COLORS['value'], COLORS['correct']]
    wedges, texts, autotexts = ax1.pie(
        categories.values(), labels=categories.keys(), autopct='%1.1f%%',
        colors=colors_list, startangle=90,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
        autotext.set_fontsize(11)

    ax1.set_title('Reconstruction Outcome Distribution',
                 fontsize=13, weight='bold', pad=15)

    # Plot 2: Bar chart with examples
    ax2.barh(list(categories.keys()), list(categories.values()),
            color=colors_list, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Number of Samples', fontsize=12, weight='bold')
    ax2.set_title('Failure Mode Frequency', fontsize=13, weight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # Add count labels
    for i, (cat, count) in enumerate(categories.items()):
        ax2.text(count + 0.3, i, str(count), va='center', fontsize=10, weight='bold')

    plt.suptitle('Grammar Validation Analysis', fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()

    # Save in multiple formats
    for ext in ['png', 'pdf', 'svg']:
        output_file = output_path / f'figure5_grammar_violations.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Figure 5 saved: {output_path / 'figure5_grammar_violations.*'}")


def plot_reconstruction_quality_metrics(samples: List[Dict], output_path: Path):
    """
    Figure 6: Reconstruction Quality Metrics
    Shows various quality measures: similarity scores, token counts, etc.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Calculate metrics
    similarities = []
    pred_token_counts = []
    actual_token_counts = []
    reconstruction_success = []

    for sample in samples:
        # Similarity
        matcher = SequenceMatcher(
            None,
            sample['reconstructed_gcode'],
            sample['actual_gcode']
        )
        similarities.append(matcher.ratio() * 100)

        # Token counts
        pred_token_counts.append(len(sample['predicted_tokens'].split()))
        actual_token_counts.append(len(tokenize_gcode(sample['actual_gcode'])))

        # Success
        reconstruction_success.append(1 if sample['reconstructed_gcode'] != "[empty]" else 0)

    # Plot 1: Similarity distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(similarities, bins=20, color=COLORS['command'],
            edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(similarities), color=COLORS['incorrect'],
               linestyle='--', linewidth=2, label=f'Mean: {np.mean(similarities):.1f}%')
    ax1.set_xlabel('Similarity Score (%)', fontsize=11, weight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, weight='bold')
    ax1.set_title('String Similarity Distribution', fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: Token count comparison
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(samples))
    width = 0.35
    ax2.bar(x - width/2, pred_token_counts, width, label='Predicted',
           color=COLORS['param'], edgecolor='black', alpha=0.7)
    ax2.bar(x + width/2, actual_token_counts, width, label='Actual',
           color=COLORS['value'], edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Sample ID', fontsize=11, weight='bold')
    ax2.set_ylabel('Token Count', fontsize=11, weight='bold')
    ax2.set_title('Predicted vs Actual Token Counts', fontsize=12, weight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 3: Reconstruction success rate
    ax3 = fig.add_subplot(gs[1, 0])
    success_rate = (sum(reconstruction_success) / len(reconstruction_success)) * 100
    failure_rate = 100 - success_rate

    bars = ax3.bar(['Success', 'Failure'], [success_rate, failure_rate],
                  color=[COLORS['correct'], COLORS['incorrect']],
                  edgecolor='black', linewidth=2, alpha=0.7)
    ax3.set_ylabel('Percentage (%)', fontsize=11, weight='bold')
    ax3.set_title('Grammar Validation Success Rate', fontsize=12, weight='bold')
    ax3.set_ylim(0, 100)

    # Add percentage labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom',
                fontsize=12, weight='bold')

    # Plot 4: Quality score scatter
    ax4 = fig.add_subplot(gs[1, 1])
    quality_scores = [s * r for s, r in zip(similarities, reconstruction_success)]
    colors_scatter = [COLORS['correct'] if r == 1 else COLORS['incorrect']
                     for r in reconstruction_success]

    ax4.scatter(range(len(samples)), quality_scores,
               c=colors_scatter, s=100, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('Sample ID', fontsize=11, weight='bold')
    ax4.set_ylabel('Quality Score (Similarity × Success)', fontsize=11, weight='bold')
    ax4.set_title('Overall Reconstruction Quality', fontsize=12, weight='bold')
    ax4.grid(alpha=0.3, linestyle='--')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['correct'],
              markersize=10, label='Valid Reconstruction'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['incorrect'],
              markersize=10, label='Invalid Reconstruction')
    ]
    ax4.legend(handles=legend_elements, loc='best')

    plt.suptitle('G-Code Reconstruction: Quality Metrics Analysis',
                fontsize=16, weight='bold')

    # Save in multiple formats
    for ext in ['png', 'pdf', 'svg']:
        output_file = output_path / f'figure6_quality_metrics.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Figure 6 saved: {output_path / 'figure6_quality_metrics.*'}")


def create_summary_report(samples: List[Dict], output_path: Path):
    """Create a text summary report of the visualizations."""
    report = []
    report.append("=" * 80)
    report.append("G-CODE RECONSTRUCTION VISUALIZATION SUMMARY")
    report.append("=" * 80)
    report.append("")

    # Overall statistics
    total = len(samples)
    matches = sum(1 for s in samples if s['match'])
    non_empty = sum(1 for s in samples if s['reconstructed_gcode'] != "[empty]")

    report.append(f"Total Samples: {total}")
    report.append(f"Exact Matches: {matches} ({matches/total*100:.1f}%)")
    report.append(f"Valid Reconstructions: {non_empty} ({non_empty/total*100:.1f}%)")
    report.append(f"Grammar Failures: {total - non_empty} ({(total-non_empty)/total*100:.1f}%)")
    report.append("")

    # Similarity statistics
    similarities = []
    for sample in samples:
        matcher = SequenceMatcher(None, sample['reconstructed_gcode'], sample['actual_gcode'])
        similarities.append(matcher.ratio() * 100)

    report.append("Similarity Statistics:")
    report.append(f"  Mean: {np.mean(similarities):.2f}%")
    report.append(f"  Median: {np.median(similarities):.2f}%")
    report.append(f"  Std Dev: {np.std(similarities):.2f}%")
    report.append(f"  Min: {np.min(similarities):.2f}%")
    report.append(f"  Max: {np.max(similarities):.2f}%")
    report.append("")

    # Figures generated
    report.append("Figures Generated:")
    report.append("  1. Token Alignment Gallery (DNA-style visualization)")
    report.append("  2. G-Code Diff Viewer (syntax-highlighted comparison)")
    report.append("  3. Reconstruction Pipeline (success rate funnel)")
    report.append("  4. Error Pattern Heatmap (systematic error identification)")
    report.append("  5. Grammar Violations Breakdown (failure mode analysis)")
    report.append("  6. Quality Metrics (comprehensive statistics)")
    report.append("")

    report.append("All figures saved in PNG (300 DPI), PDF, and SVG formats.")
    report.append("=" * 80)

    # Save report
    report_path = output_path / 'visualization_summary.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"\n✓ Summary report saved: {report_path}")
    print('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(
        description='Generate snazzy G-code reconstruction visualizations for paper'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='reports/paper_gcode_reconstruction/demo_outputs.json',
        help='Path to demo outputs JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='figures/gcode_reconstruction',
        help='Output directory for figures'
    )

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("SNAZZY G-CODE RECONSTRUCTION VISUALIZATIONS")
    print(f"{'='*80}\n")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}\n")

    # Load data
    print("Loading data...")
    samples = parse_demo_outputs(input_path)
    print(f"✓ Loaded {len(samples)} samples\n")

    # Generate all figures
    print("Generating visualizations...\n")

    plot_token_alignment_gallery(samples, output_path)
    plot_gcode_diff_viewer(samples, output_path)
    plot_reconstruction_pipeline(samples, output_path)
    plot_error_heatmap(samples, output_path)
    plot_grammar_violations(samples, output_path)
    plot_reconstruction_quality_metrics(samples, output_path)

    # Create summary
    print("")
    create_summary_report(samples, output_path)

    print(f"\n{'='*80}")
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print(f"{'='*80}\n")
    print(f"Check {output_path}/ for all figures in PNG, PDF, and SVG formats.")
    print("")


if __name__ == '__main__':
    main()
