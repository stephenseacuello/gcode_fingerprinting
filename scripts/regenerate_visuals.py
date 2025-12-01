#!/usr/bin/env python3
"""
Regenerate high-quality visualizations from existing evaluation results.

This script reads saved evaluation data and regenerates all plots with
professional styling suitable for publications, presentations, or posters.

Usage:
    # Publication quality (300 DPI, serif fonts, vector graphics)
    python scripts/regenerate_visuals.py \
        --input reports/test_comprehensive \
        --output reports/publication \
        --preset publication

    # Presentation quality (large fonts, bright colors)
    python scripts/regenerate_visuals.py \
        --input reports/test_comprehensive \
        --output reports/presentation \
        --preset presentation \
        --dpi 200

    # Custom settings
    python scripts/regenerate_visuals.py \
        --input reports/test_comprehensive \
        --output reports/custom \
        --dpi 300 \
        --font-scale 1.5 \
        --formats png svg pdf
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

# Quality presets
PRESETS = {
    'publication': {
        'dpi': 300,
        'font_scale': 1.3,
        'figure_scale': 1.2,
        'style': 'seaborn-v0_8-paper',
        'palette': 'colorblind',
        'formats': ['png', 'svg', 'pdf'],
        'font_family': 'serif',
        'use_tex': False,  # Set True if LaTeX installed
    },
    'presentation': {
        'dpi': 200,
        'font_scale': 1.8,
        'figure_scale': 1.4,
        'style': 'seaborn-v0_8-talk',
        'palette': 'bright',
        'formats': ['png'],
        'font_family': 'sans-serif',
        'use_tex': False,
    },
    'poster': {
        'dpi': 300,
        'font_scale': 2.2,
        'figure_scale': 1.6,
        'style': 'seaborn-v0_8-poster',
        'palette': 'bright',
        'formats': ['png', 'pdf'],
        'font_family': 'sans-serif',
        'use_tex': False,
    },
    'web': {
        'dpi': 150,
        'font_scale': 1.2,
        'figure_scale': 1.0,
        'style': 'seaborn-v0_8-whitegrid',
        'palette': 'Set2',
        'formats': ['png'],
        'font_family': 'sans-serif',
        'use_tex': False,
    },
}


def setup_style(preset_name: str = 'publication', dpi: Optional[int] = None,
                font_scale: Optional[float] = None):
    """Configure matplotlib for high-quality output."""
    preset = PRESETS.get(preset_name, PRESETS['publication'])

    # Override with custom values if provided
    if dpi is not None:
        preset['dpi'] = dpi
    if font_scale is not None:
        preset['font_scale'] = font_scale

    # Set style
    try:
        plt.style.use(preset['style'])
    except:
        plt.style.use('seaborn-v0_8-darkgrid')

    # Set seaborn context
    sns.set_context("paper", font_scale=preset['font_scale'])
    sns.set_palette(preset['palette'])

    # Configure matplotlib
    mpl.rcParams['figure.dpi'] = preset['dpi']
    mpl.rcParams['savefig.dpi'] = preset['dpi']
    mpl.rcParams['font.family'] = preset['font_family']
    mpl.rcParams['font.size'] = 12 * preset['font_scale']
    mpl.rcParams['axes.titlesize'] = 16 * preset['font_scale']
    mpl.rcParams['axes.labelsize'] = 14 * preset['font_scale']
    mpl.rcParams['xtick.labelsize'] = 11 * preset['font_scale']
    mpl.rcParams['ytick.labelsize'] = 11 * preset['font_scale']
    mpl.rcParams['legend.fontsize'] = 11 * preset['font_scale']
    mpl.rcParams['figure.titlesize'] = 18 * preset['font_scale']
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.1
    mpl.rcParams['savefig.facecolor'] = 'white'
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.3

    if preset['use_tex']:
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    return preset


def save_figure(fig, output_path: Path, formats: List[str], dpi: int):
    """Save figure in multiple formats."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        if fmt == 'png':
            fig.savefig(output_path.with_suffix('.png'), dpi=dpi,
                       bbox_inches='tight', facecolor='white', edgecolor='none')
        elif fmt == 'svg':
            fig.savefig(output_path.with_suffix('.svg'), format='svg',
                       bbox_inches='tight')
        elif fmt == 'pdf':
            fig.savefig(output_path.with_suffix('.pdf'), format='pdf',
                       bbox_inches='tight')

    plt.close(fig)


def regenerate_confusion_matrix(data: Dict, output_path: Path, title: str,
                                class_names: List[str], formats: List[str],
                                dpi: int, figure_scale: float):
    """Regenerate confusion matrix plot."""
    cm = np.array(data['confusion_matrix'])

    # Create figure
    fig_size = (10 * figure_scale, 8 * figure_scale)
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)

    # Enhance text
    for text in disp.text_.ravel():
        if text.get_text() != '0':
            text.set_fontweight('bold')

    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    save_figure(fig, output_path, formats, dpi)
    print(f"✓ Generated: {output_path.name}")


def regenerate_accuracy_comparison(metrics: Dict, output_path: Path,
                                   formats: List[str], dpi: int,
                                   figure_scale: float):
    """Regenerate accuracy comparison bar chart."""
    # Extract metrics
    metric_names = []
    metric_values = []

    key_metrics = ['type', 'command', 'param_type', 'param_value', 'overall']
    for key in key_metrics:
        if key in metrics:
            metric_names.append(key.replace('_', ' ').title())
            metric_values.append(metrics[key] * 100)  # Convert to percentage

    # Create figure
    fig_size = (12 * figure_scale, 7 * figure_scale)
    fig, ax = plt.subplots(figsize=fig_size)

    # Create bars
    colors = sns.color_palette('Set2', len(metric_names))
    bars = ax.bar(metric_names, metric_values, color=colors,
                 edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom',
               fontsize=12*figure_scale, fontweight='bold')

    # Styling
    ax.set_title('Model Performance Across Metrics',
                fontweight='bold', pad=20)
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Rotate x labels if needed
    if len(metric_names) > 5:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    save_figure(fig, output_path, formats, dpi)
    print(f"✓ Generated: {output_path.name}")


def regenerate_per_class_f1(metrics_data: Dict, output_path: Path,
                            title: str, formats: List[str], dpi: int,
                            figure_scale: float, max_classes: int = 15):
    """Regenerate per-class F1 score chart."""
    # Parse metrics data
    classes = []
    f1_scores = []

    for class_name, class_metrics in metrics_data.items():
        if isinstance(class_metrics, dict) and 'f1-score' in class_metrics:
            classes.append(class_name)
            f1_scores.append(class_metrics['f1-score'])

    if not classes:
        print(f"⚠️  No F1 scores found for {title}")
        return

    # Sort by F1 score (descending) and take top N
    sorted_indices = np.argsort(f1_scores)[::-1][:max_classes]
    classes = [classes[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]

    # Create figure
    fig_size = (14 * figure_scale, 8 * figure_scale)
    fig, ax = plt.subplots(figsize=fig_size)

    # Create horizontal bars
    y_pos = np.arange(len(classes))
    colors = sns.color_palette('RdYlGn', len(classes))

    bars = ax.barh(y_pos, f1_scores, color=colors,
                  edgecolor='black', linewidth=1.2, alpha=0.8)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax.text(score + 0.02, bar.get_y() + bar.get_height()/2,
               f'{score:.3f}',
               ha='left', va='center', fontweight='bold')

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.set_xlabel('F1 Score', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_figure(fig, output_path, formats, dpi)
    print(f"✓ Generated: {output_path.name}")


def regenerate_operation_performance(operation_data: Dict, output_path: Path,
                                    formats: List[str], dpi: int,
                                    figure_scale: float):
    """Regenerate operation type performance chart."""
    if not operation_data:
        print("⚠️  No operation performance data found")
        return

    operations = list(operation_data.keys())
    accuracies = [operation_data[op]['accuracy'] * 100 for op in operations]

    # Create figure
    fig_size = (14 * figure_scale, 8 * figure_scale)
    fig, ax = plt.subplots(figsize=fig_size)

    # Create bars
    colors = sns.color_palette('viridis', len(operations))
    bars = ax.bar(operations, accuracies, color=colors,
                 edgecolor='black', linewidth=1.5, alpha=0.85)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontweight='bold')

    # Styling
    ax.set_title('Accuracy by Operation Type', fontweight='bold', pad=20)
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_xlabel('Operation Type', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    save_figure(fig, output_path, formats, dpi)
    print(f"✓ Generated: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate high-quality visualizations from evaluation results')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with evaluation results')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for high-quality visuals')
    parser.add_argument('--preset', type=str, default='publication',
                       choices=list(PRESETS.keys()),
                       help='Quality preset (publication/presentation/poster/web)')
    parser.add_argument('--dpi', type=int, default=None,
                       help='Override DPI setting')
    parser.add_argument('--font-scale', type=float, default=None,
                       help='Override font scale')
    parser.add_argument('--formats', nargs='+',
                       default=None,
                       choices=['png', 'svg', 'pdf'],
                       help='Output formats (default: from preset)')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 1

    print(f"\n{'='*80}")
    print(f"REGENERATING HIGH-QUALITY VISUALIZATIONS")
    print(f"{'='*80}\n")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Preset: {args.preset}")

    # Setup style
    preset_config = setup_style(args.preset, args.dpi, args.font_scale)
    dpi = preset_config['dpi']
    figure_scale = preset_config['figure_scale']
    formats = args.formats if args.formats else preset_config['formats']

    print(f"DPI:    {dpi}")
    print(f"Scale:  {figure_scale}x")
    print(f"Formats: {', '.join(formats)}")
    print()

    # Create output directories
    cm_dir = output_dir / 'confusion_matrices'
    charts_dir = output_dir / 'bar_charts'

    # Load accuracy breakdown
    accuracy_file = input_dir / 'accuracy_breakdown.json'
    if accuracy_file.exists():
        print("📊 Loading accuracy breakdown...")
        with open(accuracy_file) as f:
            accuracy_data = json.load(f)

        # Regenerate accuracy comparison
        print("\n🎨 Generating accuracy comparison...")
        regenerate_accuracy_comparison(
            accuracy_data,
            charts_dir / 'accuracy_comparison',
            formats, dpi, figure_scale
        )

    # Load and regenerate per-class metrics
    metrics_dir = input_dir / 'per_class_metrics'
    if metrics_dir.exists():
        print("\n🎨 Generating per-class F1 charts...")

        # Command metrics
        command_file = metrics_dir / 'command_metrics.json'
        if command_file.exists():
            with open(command_file) as f:
                command_metrics = json.load(f)
            regenerate_per_class_f1(
                command_metrics,
                charts_dir / 'command_f1_scores',
                'F1 Scores by G-Code Command',
                formats, dpi, figure_scale
            )

        # Param type metrics
        param_type_file = metrics_dir / 'param_type_metrics.json'
        if param_type_file.exists():
            with open(param_type_file) as f:
                param_metrics = json.load(f)
            regenerate_per_class_f1(
                param_metrics,
                charts_dir / 'param_type_f1_scores',
                'F1 Scores by Parameter Type',
                formats, dpi, figure_scale
            )

    # Check for operation-specific data
    operation_file = input_dir / 'operation_analysis' / 'operation_specific_errors.json'
    if operation_file.exists():
        print("\n🎨 Generating operation performance chart...")
        with open(operation_file) as f:
            operation_data = json.load(f)

        # Extract accuracy per operation
        if 'operation_performance' in operation_data:
            regenerate_operation_performance(
                operation_data['operation_performance'],
                charts_dir / 'operation_performance',
                formats, dpi, figure_scale
            )

    print(f"\n{'='*80}")
    print(f"✅ REGENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nOutput location: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - Accuracy comparison charts")
    print(f"  - Per-class F1 score charts")
    print(f"  - Operation performance charts")
    print(f"\nFormats: {', '.join(formats)}")
    print(f"DPI: {dpi}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
