#!/usr/bin/env python3
"""
Generate all figures for the paper from evaluation results and training history.

Generates:
- Training/validation curves (loss and accuracy)
- Confusion matrices for each prediction head
- Ablation comparison bar charts (sensor, modality, architecture)
- t-SNE visualizations of latent space
- Per-head accuracy breakdown

Usage:
    python scripts/visualization/generate_paper_figures.py \
        --results-dir outputs/jan26/evaluation \
        --output-dir outputs/jan26/figures \
        --ablation-dir outputs/jan26
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")


def setup_style():
    """Set up consistent plotting style for paper figures."""
    if not HAS_PLOTTING:
        return

    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (6, 4),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    sns.set_palette("colorblind")


def plot_training_curves(history_path: str, output_dir: Path, name: str = "training"):
    """Plot training and validation curves from history file."""
    if not HAS_PLOTTING:
        return

    if not os.path.exists(history_path):
        print(f"  Skipping training curves: {history_path} not found")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.set_xlim(1, len(epochs))

    # Accuracy curves
    ax = axes[1]
    ax.plot(epochs, history['train_acc'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_acc'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Token Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    ax.set_xlim(1, len(epochs))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_curves.pdf')
    plt.savefig(output_dir / f'{name}_curves.png')
    plt.close()
    print(f"  Saved: {name}_curves.pdf/png")


def plot_confusion_matrix(cm: List[List[int]], labels: List[str], title: str,
                          output_path: Path, normalize: bool = True):
    """Plot a confusion matrix."""
    if not HAS_PLOTTING:
        return

    cm = np.array(cm)
    if normalize and cm.sum() > 0:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), max(5, len(labels) * 0.7)))

    sns.heatmap(cm_norm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=labels, yticklabels=labels,
                cmap='Blues', ax=ax, cbar=True,
                annot_kws={'size': 8})

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.savefig(output_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {output_path.stem}.pdf/png")


def plot_confusion_matrices(results_path: str, output_dir: Path):
    """Plot all confusion matrices from evaluation results."""
    if not HAS_PLOTTING:
        return

    if not os.path.exists(results_path):
        print(f"  Skipping confusion matrices: {results_path} not found")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Get test confusion matrices
    test_metrics = results.get('test_metrics', {})
    cms = test_metrics.get('confusion_matrices', {})

    # Type confusion matrix
    if 'type' in cms and cms['type']:
        type_labels = ['CMD', 'PARAM', 'NUM', 'EOS']
        plot_confusion_matrix(
            cms['type'], type_labels, 'Type Classification Confusion Matrix',
            output_dir / 'type_confusion_matrix'
        )

    # Command confusion matrix
    if 'command' in cms and cms['command']:
        cmd_labels = ['G0', 'G1', 'G2', 'G3', 'M3', 'M5']
        n_classes = len(cms['command'])
        plot_confusion_matrix(
            cms['command'], cmd_labels[:n_classes],
            'Command Classification Confusion Matrix',
            output_dir / 'command_confusion_matrix'
        )

    # Parameter type confusion matrix
    if 'param' in cms and cms['param']:
        param_labels = ['X', 'Y', 'Z', 'F', 'S', 'I', 'J', 'K', 'R', 'P']
        n_classes = len(cms['param'])
        plot_confusion_matrix(
            cms['param'], param_labels[:n_classes],
            'Parameter Type Confusion Matrix',
            output_dir / 'parameter_confusion_matrix'
        )


def plot_sensor_ablation(ablation_dir: Path, output_dir: Path):
    """Plot sensor ablation results comparison."""
    if not HAS_PLOTTING:
        return

    sensor_dir = ablation_dir / 'sensor_ablations'
    if not sensor_dir.exists():
        print(f"  Skipping sensor ablation: {sensor_dir} not found")
        return

    # Collect results from each sensor
    results = {}
    for sensor_path in sorted(sensor_dir.glob('only_*')):
        sensor_name = sensor_path.name.replace('only_', '')
        results_file = sensor_path / 'results.json'

        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            test_metrics = data.get('test_metrics', {})
            results[sensor_name] = {
                'token_acc': test_metrics.get('token_acc', 0),
                'sequence_acc': test_metrics.get('sequence_acc', 0),
            }

    if not results:
        print("  No sensor ablation results found")
        return

    # Sort by token accuracy
    sorted_sensors = sorted(results.items(), key=lambda x: x[1]['token_acc'], reverse=True)
    sensors = [s[0] for s in sorted_sensors]
    token_accs = [s[1]['token_acc'] * 100 for s in sorted_sensors]
    seq_accs = [s[1]['sequence_acc'] * 100 for s in sorted_sensors]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(sensors))
    width = 0.35

    bars1 = ax.bar(x - width/2, token_accs, width, label='Token Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, seq_accs, width, label='Sequence Accuracy', color='darkorange')

    ax.set_xlabel('Sensor')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Single Sensor Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(sensors, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / 'sensor_ablation_comparison.pdf')
    plt.savefig(output_dir / 'sensor_ablation_comparison.png')
    plt.close()
    print("  Saved: sensor_ablation_comparison.pdf/png")

    # Save data for paper table
    table_data = {sensor: results[sensor] for sensor in sensors}
    with open(output_dir / 'sensor_ablation_data.json', 'w') as f:
        json.dump(table_data, f, indent=2)
    print("  Saved: sensor_ablation_data.json")


def plot_modality_ablation(ablation_dir: Path, output_dir: Path):
    """Plot modality ablation results comparison."""
    if not HAS_PLOTTING:
        return

    modality_dir = ablation_dir / 'modality_ablations'
    if not modality_dir.exists():
        print(f"  Skipping modality ablation: {modality_dir} not found")
        return

    # Collect results
    results = {}
    for modality_path in sorted(modality_dir.glob('only_*')):
        modality_name = modality_path.name.replace('only_', '')
        results_file = modality_path / 'results.json'

        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            test_metrics = data.get('test_metrics', {})
            results[modality_name] = {
                'token_acc': test_metrics.get('token_acc', 0),
                'sequence_acc': test_metrics.get('sequence_acc', 0),
            }

    if not results:
        print("  No modality ablation results found")
        return

    # Sort by token accuracy
    sorted_modalities = sorted(results.items(), key=lambda x: x[1]['token_acc'], reverse=True)
    modalities = [m[0] for m in sorted_modalities]
    token_accs = [m[1]['token_acc'] * 100 for m in sorted_modalities]
    seq_accs = [m[1]['sequence_acc'] * 100 for m in sorted_modalities]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(modalities))
    width = 0.35

    bars1 = ax.bar(x - width/2, token_accs, width, label='Token Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, seq_accs, width, label='Sequence Accuracy', color='darkorange')

    ax.set_xlabel('Modality')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Single Modality Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(modalities, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'modality_ablation_comparison.pdf')
    plt.savefig(output_dir / 'modality_ablation_comparison.png')
    plt.close()
    print("  Saved: modality_ablation_comparison.pdf/png")

    # Save data for paper table
    with open(output_dir / 'modality_ablation_data.json', 'w') as f:
        json.dump(dict(sorted_modalities), f, indent=2)
    print("  Saved: modality_ablation_data.json")


def plot_architecture_ablation(ablation_dir: Path, output_dir: Path):
    """Plot architecture ablation results (cross-attention, etc.)."""
    if not HAS_PLOTTING:
        return

    arch_dir = ablation_dir / 'architecture_ablations'
    if not arch_dir.exists():
        print(f"  Skipping architecture ablation: {arch_dir} not found")
        return

    # Collect results from cross_attention subdirectory
    results = {}
    cross_attn_dir = arch_dir / 'cross_attention'
    if cross_attn_dir.exists():
        for config_path in cross_attn_dir.glob('cross_attn_*'):
            config_name = config_path.name
            results_file = config_path / 'results.json'

            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                test_metrics = data.get('test_metrics', {})
                results[config_name] = {
                    'token_acc': test_metrics.get('token_acc', 0),
                    'sequence_acc': test_metrics.get('sequence_acc', 0),
                }

    if not results:
        print("  No architecture ablation results found")
        return

    # Create comparison figure
    configs = list(results.keys())
    token_accs = [results[c]['token_acc'] * 100 for c in configs]
    seq_accs = [results[c]['sequence_acc'] * 100 for c in configs]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, token_accs, width, label='Token Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, seq_accs, width, label='Sequence Accuracy', color='darkorange')

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Cross-Attention Ablation')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_seed42', '').replace('_', ' ') for c in configs])
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'architecture_ablation_comparison.pdf')
    plt.savefig(output_dir / 'architecture_ablation_comparison.png')
    plt.close()
    print("  Saved: architecture_ablation_comparison.pdf/png")

    # Save data
    with open(output_dir / 'architecture_ablation_data.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  Saved: architecture_ablation_data.json")


def plot_metrics_summary(results_dir: Path, output_dir: Path):
    """Plot summary of per-head metrics across experiments."""
    if not HAS_PLOTTING:
        return

    # Find all full_metrics.json files
    metrics_files = list(results_dir.glob('**/full_metrics.json'))
    if not metrics_files:
        print("  No full_metrics.json files found")
        return

    # For each file, extract per-head accuracies
    all_data = []
    for mf in metrics_files:
        with open(mf) as f:
            data = json.load(f)
        test_metrics = data.get('test_metrics', {})
        exp_name = mf.parent.name

        all_data.append({
            'experiment': exp_name,
            'type_acc': test_metrics.get('type_acc', 0) * 100,
            'command_acc': test_metrics.get('command_acc', 0) * 100,
            'param_acc': test_metrics.get('param_acc', 0) * 100,
            'digit_avg_acc': test_metrics.get('digit_avg_acc', 0) * 100,
            'token_acc': test_metrics.get('token_acc', 0) * 100,
        })

    if not all_data:
        return

    # Create grouped bar chart
    experiments = [d['experiment'] for d in all_data]
    metrics = ['type_acc', 'command_acc', 'param_acc', 'digit_avg_acc', 'token_acc']
    metric_labels = ['Type', 'Command', 'Param', 'Digit Avg', 'Token']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(experiments))
    width = 0.15
    colors = sns.color_palette("colorblind", len(metrics))

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [d[metric] for d in all_data]
        ax.bar(x + i * width, values, width, label=label, color=colors[i])

    ax.set_xlabel('Experiment')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Head Accuracy Breakdown')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_summary.pdf')
    plt.savefig(output_dir / 'metrics_summary.png')
    plt.close()
    print("  Saved: metrics_summary.pdf/png")


def generate_latex_tables(ablation_dir: Path, output_dir: Path):
    """Generate LaTeX table snippets for the paper."""

    # Sensor ablation table
    sensor_data_file = output_dir / 'sensor_ablation_data.json'
    if sensor_data_file.exists():
        with open(sensor_data_file) as f:
            data = json.load(f)

        latex = "% Single Sensor Performance (Table 2)\n"
        latex += "\\begin{tabular}{lcc}\n\\toprule\n"
        latex += "\\textbf{Sensor} & \\textbf{Token Acc.} & \\textbf{Seq. Acc.} \\\\\n\\midrule\n"

        for sensor, metrics in data.items():
            token = metrics['token_acc'] * 100
            seq = metrics['sequence_acc'] * 100
            latex += f"{sensor.replace('_', '\\_')} & {token:.2f}\\% & {seq:.2f}\\% \\\\\n"

        latex += "\\bottomrule\n\\end{tabular}\n"

        with open(output_dir / 'table_sensor_ablation.tex', 'w') as f:
            f.write(latex)
        print("  Saved: table_sensor_ablation.tex")

    # Modality ablation table
    modality_data_file = output_dir / 'modality_ablation_data.json'
    if modality_data_file.exists():
        with open(modality_data_file) as f:
            data = json.load(f)

        latex = "% Single Modality Performance (Table 4)\n"
        latex += "\\begin{tabular}{lcc}\n\\toprule\n"
        latex += "\\textbf{Modality} & \\textbf{Token Acc.} & \\textbf{Seq. Acc.} \\\\\n\\midrule\n"

        for modality, metrics in data.items():
            token = metrics['token_acc'] * 100
            seq = metrics['sequence_acc'] * 100
            latex += f"{modality.replace('_', '\\_')} & {token:.2f}\\% & {seq:.2f}\\% \\\\\n"

        latex += "\\bottomrule\n\\end{tabular}\n"

        with open(output_dir / 'table_modality_ablation.tex', 'w') as f:
            f.write(latex)
        print("  Saved: table_modality_ablation.tex")

    # Architecture ablation table
    arch_data_file = output_dir / 'architecture_ablation_data.json'
    if arch_data_file.exists():
        with open(arch_data_file) as f:
            data = json.load(f)

        latex = "% Architecture Ablation (Table 7)\n"
        latex += "\\begin{tabular}{lcc}\n\\toprule\n"
        latex += "\\textbf{Configuration} & \\textbf{Token Acc.} & \\textbf{Seq. Acc.} \\\\\n\\midrule\n"

        for config, metrics in data.items():
            name = config.replace('_seed42', '').replace('_', ' ')
            token = metrics['token_acc'] * 100
            seq = metrics['sequence_acc'] * 100
            latex += f"{name} & {token:.2f}\\% & {seq:.2f}\\% \\\\\n"

        latex += "\\bottomrule\n\\end{tabular}\n"

        with open(output_dir / 'table_architecture_ablation.tex', 'w') as f:
            f.write(latex)
        print("  Saved: table_architecture_ablation.tex")


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures from evaluation results')
    parser.add_argument('--results-dir', type=str, default='outputs/jan26/evaluation',
                        help='Directory containing full_metrics.json files')
    parser.add_argument('--ablation-dir', type=str, default='outputs/jan26',
                        help='Directory containing ablation experiment results')
    parser.add_argument('--output-dir', type=str, default='outputs/jan26/figures',
                        help='Output directory for figures')
    parser.add_argument('--history-path', type=str, help='Path to training history JSON')
    args = parser.parse_args()

    if not HAS_PLOTTING:
        print("ERROR: matplotlib and seaborn are required. Install with:")
        print("  pip install matplotlib seaborn")
        sys.exit(1)

    setup_style()

    results_dir = Path(args.results_dir)
    ablation_dir = Path(args.ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("GENERATING PAPER FIGURES")
    print("="*60)

    # Training curves (if history available)
    if args.history_path:
        print("\n1. Training Curves")
        plot_training_curves(args.history_path, output_dir)

    # Confusion matrices
    print("\n2. Confusion Matrices")
    for metrics_file in results_dir.glob('**/full_metrics.json'):
        exp_name = metrics_file.parent.name
        print(f"  Processing: {exp_name}")
        exp_output = output_dir / exp_name
        exp_output.mkdir(exist_ok=True)
        plot_confusion_matrices(str(metrics_file), exp_output)

    # Ablation comparisons
    print("\n3. Sensor Ablation")
    plot_sensor_ablation(ablation_dir, output_dir)

    print("\n4. Modality Ablation")
    plot_modality_ablation(ablation_dir, output_dir)

    print("\n5. Architecture Ablation")
    plot_architecture_ablation(ablation_dir, output_dir)

    # Metrics summary
    print("\n6. Metrics Summary")
    plot_metrics_summary(results_dir, output_dir)

    # LaTeX tables
    print("\n7. LaTeX Tables")
    generate_latex_tables(ablation_dir, output_dir)

    print("\n" + "="*60)
    print(f"Figures saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
