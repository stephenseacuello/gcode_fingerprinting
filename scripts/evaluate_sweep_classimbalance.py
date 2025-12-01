#!/usr/bin/env python3
"""
Evaluate sweep results with focus on class imbalance metrics.

This script analyzes the fast validation sweep to validate success criteria:
1. G0 predictions < 50% (down from 100%)
2. Command diversity entropy > 1.5 (up from 0)
3. Non-G0 accuracy > 20%
4. Operation type accuracy > 60%

Usage:
    python scripts/evaluate_sweep_classimbalance.py \
        --sweep-results outputs/sweep_analysis/sweep_results.csv \
        --data-dir outputs/processed_with_ops \
        --vocab-path data/vocabulary.json \
        --output outputs/sweep_fast_validation_analysis
"""

import argparse
import pandas as pd
import numpy as np
import torch
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import confusion_matrix

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.multihead_lm import MultiHeadGCodeLM
from miracle.dataset.target_utils import TokenDecomposer
from miracle.utilities.device import get_device


def load_checkpoint(checkpoint_path: Path, decomposer: TokenDecomposer, device):
    """Load model from checkpoint."""
    print(f"  Loading: {checkpoint_path.name}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
    else:
        config_dict = {
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
        }

    # Infer dimensions
    backbone_state = checkpoint['backbone_state_dict']
    multihead_state = checkpoint['multihead_state_dict']

    n_continuous = backbone_state['encoders.0.proj.0.weight'].shape[1] if 'encoders.0.proj.0.weight' in backbone_state else 135
    n_categorical = 4

    # Check vocab size
    checkpoint_vocab_size = multihead_state['embed.weight'].shape[0] if 'embed.weight' in multihead_state else 170

    if checkpoint_vocab_size != decomposer.vocab_size:
        print(f"    ⚠️  Vocab size mismatch: checkpoint={checkpoint_vocab_size}, current={decomposer.vocab_size}")
        return None, None, None

    vocab_size = checkpoint_vocab_size

    # Create models
    backbone_config = ModelConfig(
        sensor_dims=[n_continuous, n_categorical],
        d_model=config_dict.get('hidden_dim', 128),
        lstm_layers=config_dict.get('num_layers', 2),
        gcode_vocab=vocab_size,
        n_heads=config_dict.get('num_heads', 4),
    )

    backbone = MM_DTAE_LSTM(backbone_config).to(device)
    multihead_lm = MultiHeadGCodeLM(
        d_model=config_dict.get('hidden_dim', 128),
        n_commands=decomposer.n_commands,
        n_param_types=decomposer.n_param_types,
        n_param_values=decomposer.n_param_values,
        nhead=config_dict.get('num_heads', 4),
        num_layers=config_dict.get('num_layers', 2),
        dropout=config_dict.get('dropout', 0.1),
        vocab_size=vocab_size,
        n_operation_types=6,  # 6 operation types
    ).to(device)

    # Load weights
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    multihead_lm.load_state_dict(checkpoint['multihead_state_dict'])

    backbone.eval()
    multihead_lm.eval()

    return backbone, multihead_lm, config_dict


def load_validation_data(data_dir: Path):
    """Load validation sequences."""
    val_path = data_dir / 'val_sequences.npz'

    print(f"  Loading validation data: {val_path}")
    data = np.load(val_path, allow_pickle=True)

    # Handle different data formats
    if 'tokens' in data:
        sequences = data['tokens']
        sensor_data = [
            {
                'continuous': data['continuous'][i],
                'categorical': data['categorical'][i]
            }
            for i in range(len(sequences))
        ]
        operation_types = data['operation_type'] if 'operation_type' in data else None
    else:
        raise ValueError(f"Unknown data format. Keys: {list(data.keys())}")

    print(f"    ✓ Loaded {len(sequences)} validation samples")
    return sequences, sensor_data, operation_types


def evaluate_with_class_metrics(backbone, multihead_lm, decomposer,
                                 val_sequences, val_sensor, val_operation_types,
                                 device):
    """
    Evaluate model with class imbalance metrics.

    Returns:
        metrics dict with:
        - command_acc, overall_acc (standard)
        - g0_prediction_ratio
        - command_entropy
        - per_command_accuracy
        - operation_type_acc
        - operation_confusion_matrix
    """
    total_samples = len(val_sequences)

    # Accumulators
    command_correct = 0
    overall_correct = 0
    operation_correct = 0

    # Class imbalance metrics
    all_command_predictions = []
    all_command_targets = []
    all_operation_predictions = []
    all_operation_targets = []

    # Per-command accuracy tracking
    per_command_correct = Counter()
    per_command_total = Counter()

    with torch.no_grad():
        for idx, (seq, sensor) in enumerate(zip(val_sequences, val_sensor)):
            # Prepare sensor input
            sensor_tensor = {
                'continuous': torch.FloatTensor(sensor['continuous']).unsqueeze(0).to(device),
                'categorical': torch.LongTensor(sensor['categorical']).unsqueeze(0).to(device),
            }

            # Decompose target
            decomposed = decomposer.decompose_sequence(seq)

            # Get embeddings
            embeddings = backbone.encode(sensor_tensor)

            # Prepare inputs
            type_gate = torch.LongTensor(decomposed['type_gate']).unsqueeze(0).to(device)
            commands = torch.LongTensor(decomposed['commands']).unsqueeze(0).to(device)
            param_types = torch.LongTensor(decomposed['param_types']).unsqueeze(0).to(device)
            param_values = torch.LongTensor(decomposed['param_values']).unsqueeze(0).to(device)

            # Forward pass
            outputs = multihead_lm(
                embeddings,
                type_gate[:, :-1],
                commands[:, :-1],
                param_types[:, :-1],
                param_values[:, :-1]
            )

            # Get predictions
            cmd_pred = outputs['command_logits'].argmax(dim=-1)[0]
            cmd_target = commands[0, 1:]

            # Collect predictions and targets
            all_command_predictions.extend(cmd_pred.cpu().numpy().tolist())
            all_command_targets.extend(cmd_target.cpu().numpy().tolist())

            # Per-command accuracy
            for pred, target in zip(cmd_pred.cpu().numpy(), cmd_target.cpu().numpy()):
                per_command_total[target] += 1
                if pred == target:
                    per_command_correct[target] += 1

            # Overall accuracy
            command_correct += (cmd_pred == cmd_target).float().mean().item()

            # Operation type prediction (if available)
            if 'operation_logits' in outputs and val_operation_types is not None:
                op_pred = outputs['operation_logits'].argmax(dim=-1).item()
                op_target = val_operation_types[idx]

                all_operation_predictions.append(op_pred)
                all_operation_targets.append(op_target)

                if op_pred == op_target:
                    operation_correct += 1

    # Calculate standard metrics
    metrics = {
        'command_acc': command_correct / total_samples,
        'num_samples': total_samples,
    }

    # Calculate G0 prediction ratio
    pred_counts = Counter(all_command_predictions)
    total_predictions = sum(pred_counts.values())
    g0_predictions = pred_counts.get(0, 0)  # G0 is typically command ID 0
    metrics['g0_prediction_ratio'] = g0_predictions / total_predictions if total_predictions > 0 else 0

    # Calculate command prediction entropy
    pred_probs = np.array([pred_counts.get(i, 0) for i in range(decomposer.n_commands)]) / total_predictions
    metrics['command_entropy'] = scipy_entropy(pred_probs[pred_probs > 0], base=2)

    # Per-command accuracy
    per_command_acc = {}
    for cmd_id in range(decomposer.n_commands):
        if per_command_total[cmd_id] > 0:
            acc = per_command_correct[cmd_id] / per_command_total[cmd_id]
            per_command_acc[cmd_id] = acc
        else:
            per_command_acc[cmd_id] = 0.0

    metrics['per_command_accuracy'] = per_command_acc

    # Non-G0 accuracy
    non_g0_correct = sum(per_command_correct[i] for i in range(1, decomposer.n_commands))
    non_g0_total = sum(per_command_total[i] for i in range(1, decomposer.n_commands))
    metrics['non_g0_accuracy'] = non_g0_correct / non_g0_total if non_g0_total > 0 else 0.0

    # Operation type metrics
    if val_operation_types is not None:
        metrics['operation_type_acc'] = operation_correct / total_samples
        metrics['operation_confusion_matrix'] = confusion_matrix(
            all_operation_targets,
            all_operation_predictions,
            labels=list(range(6))
        ).tolist()
    else:
        metrics['operation_type_acc'] = 0.0
        metrics['operation_confusion_matrix'] = None

    # Command prediction distribution
    metrics['command_prediction_distribution'] = dict(pred_counts)

    return metrics


def validate_success_criteria(metrics: Dict) -> Dict:
    """Check if run meets success criteria."""
    criteria = {
        'g0_ratio_below_50': metrics['g0_prediction_ratio'] < 0.5,
        'entropy_above_1.5': metrics['command_entropy'] > 1.5,
        'non_g0_acc_above_20': metrics['non_g0_accuracy'] > 0.20,
        'operation_acc_above_60': metrics['operation_type_acc'] > 0.60
    }

    all_pass = all(criteria.values())
    num_pass = sum(criteria.values())

    return {
        'all_criteria_met': all_pass,
        'num_criteria_met': num_pass,
        'total_criteria': len(criteria),
        'individual_criteria': criteria,
        'summary': f"{'✓' if all_pass else '✗'} {num_pass}/{len(criteria)} criteria met"
    }


def analyze_sweep_runs(sweep_csv: Path, data_dir: Path, vocab_path: Path,
                       device, max_runs: int = None):
    """Analyze all runs in the sweep."""
    print("=" * 80)
    print("ANALYZING SWEEP RUNS WITH CLASS IMBALANCE METRICS")
    print("=" * 80)
    print()

    # Load sweep results
    df = pd.read_csv(sweep_csv)
    print(f"Found {len(df)} runs in sweep")

    if max_runs:
        df = df.head(max_runs)
        print(f"Analyzing first {max_runs} runs")

    print()

    # Load vocabulary and data
    decomposer = TokenDecomposer(str(vocab_path))
    val_sequences, val_sensor, val_operation_types = load_validation_data(data_dir)

    # Analyze each run
    all_results = []

    for idx, row in df.iterrows():
        run_id = row['run_id']
        run_name = row['name']

        print(f"\n[{idx+1}/{len(df)}] Analyzing {run_name} ({run_id})")
        print("-" * 80)

        # Find checkpoint
        output_dir = Path(row['output-dir']) if 'output-dir' in row else Path("outputs/sweep_fast_validation")
        checkpoint_path = output_dir / 'checkpoint_best.pt'

        if not checkpoint_path.exists():
            print(f"  ⚠️  Checkpoint not found: {checkpoint_path}")
            continue

        # Load model
        backbone, multihead_lm, config = load_checkpoint(checkpoint_path, decomposer, device)

        if backbone is None:
            print(f"  ⚠️  Failed to load checkpoint")
            continue

        # Evaluate with class metrics
        print(f"  Evaluating on {len(val_sequences)} validation samples...")
        metrics = evaluate_with_class_metrics(
            backbone, multihead_lm, decomposer,
            val_sequences, val_sensor, val_operation_types,
            device
        )

        # Validate criteria
        validation = validate_success_criteria(metrics)

        # Print summary
        print(f"\n  Results:")
        print(f"    Command Accuracy: {metrics['command_acc']:.4f} ({metrics['command_acc']*100:.2f}%)")
        print(f"    G0 Prediction Ratio: {metrics['g0_prediction_ratio']:.4f} ({metrics['g0_prediction_ratio']*100:.2f}%)")
        print(f"    Command Entropy: {metrics['command_entropy']:.4f}")
        print(f"    Non-G0 Accuracy: {metrics['non_g0_accuracy']:.4f} ({metrics['non_g0_accuracy']*100:.2f}%)")
        print(f"    Operation Type Accuracy: {metrics['operation_type_acc']:.4f} ({metrics['operation_type_acc']*100:.2f}%)")
        print(f"\n  Success Criteria: {validation['summary']}")

        # Store results
        result = {
            'run_id': run_id,
            'run_name': run_name,
            'config': {
                'command_weight': row.get('command_weight', None),
                'label_smoothing': row.get('label_smoothing', None),
                'hidden_dim': row.get('hidden_dim', None),
                'learning_rate': row.get('learning_rate', None),
            },
            'metrics': metrics,
            'validation': validation
        }

        all_results.append(result)

    print("\n" + "=" * 80)
    print(f"Completed analysis of {len(all_results)} runs")
    print("=" * 80)

    return all_results


def create_visualizations(results: List[Dict], output_dir: Path, decomposer: TokenDecomposer):
    """Generate publication-quality comprehensive visualizations."""
    print("\nGenerating enhanced visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set publication-quality style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Extract data
    df = pd.DataFrame([
        {
            'run_name': r['run_name'],
            'command_weight': r['config']['command_weight'],
            'label_smoothing': r['config']['label_smoothing'],
            'hidden_dim': r['config']['hidden_dim'],
            'learning_rate': r['config']['learning_rate'],
            'command_acc': r['metrics']['command_acc'],
            'g0_ratio': r['metrics']['g0_prediction_ratio'],
            'entropy': r['metrics']['command_entropy'],
            'non_g0_acc': r['metrics']['non_g0_accuracy'],
            'operation_acc': r['metrics']['operation_type_acc'],
            'criteria_met': r['validation']['num_criteria_met'],
        }
        for r in results
    ])

    # 1. SUCCESS CRITERIA RADAR CHART - BEST vs WORST
    print("  Creating enhanced success criteria radar chart...")
    best_idx = df['criteria_met'].idxmax()
    worst_idx = df['criteria_met'].idxmin()

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

    categories = ['Command\nAccuracy', 'Prediction\nDiversity\n(1-G0)',
                  'Entropy\n(normalized)', 'Non-G0\nAccuracy', 'Operation\nAccuracy']

    # Best run
    best_values = [
        df.loc[best_idx, 'command_acc'],
        1 - df.loc[best_idx, 'g0_ratio'],
        df.loc[best_idx, 'entropy'] / 3.0,
        df.loc[best_idx, 'non_g0_acc'],
        df.loc[best_idx, 'operation_acc']
    ]

    # Worst run
    worst_values = [
        df.loc[worst_idx, 'command_acc'],
        1 - df.loc[worst_idx, 'g0_ratio'],
        df.loc[worst_idx, 'entropy'] / 3.0,
        df.loc[worst_idx, 'non_g0_acc'],
        df.loc[worst_idx, 'operation_acc']
    ]

    # Target values
    target_values = [0.8, 0.5, 0.5, 0.2, 0.6]  # Reasonable targets

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

    # Close the plots
    best_values += best_values[:1]
    worst_values += worst_values[:1]
    target_values += target_values[:1]
    angles += angles[:1]

    # Plot
    ax.plot(angles, best_values, 'o-', linewidth=3, label=f'Best (criteria: {df.loc[best_idx, "criteria_met"]}/4)',
            color='#2E86AB', markersize=8)
    ax.fill(angles, best_values, alpha=0.25, color='#2E86AB')

    ax.plot(angles, worst_values, 's-', linewidth=2, label=f'Worst (criteria: {df.loc[worst_idx, "criteria_met"]}/4)',
            color='#E63946', markersize=6, alpha=0.7)
    ax.fill(angles, worst_values, alpha=0.15, color='#E63946')

    ax.plot(angles, target_values, 'D--', linewidth=2, label='Target Thresholds',
            color='#2A9D8F', markersize=5, alpha=0.8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, weight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Class Imbalance Solution: Best vs Worst Configuration\nPerformance Comparison',
                 size=16, pad=30, weight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / 'success_criteria_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. COMPREHENSIVE HYPERPARAMETER ANALYSIS - 3x2 grid
    print("  Creating comprehensive hyperparameter analysis...")
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    params = ['command_weight', 'label_smoothing', 'hidden_dim', 'learning_rate']
    param_labels = ['Command Weight', 'Label Smoothing', 'Hidden Dimension', 'Learning Rate']
    metrics = [('command_acc', 'Command Accuracy'), ('entropy', 'Prediction Entropy')]

    for idx, (param, param_label) in enumerate(zip(params, param_labels)):
        # Plot 1: param vs command accuracy
        ax1 = fig.add_subplot(gs[idx, 0])
        scatter = ax1.scatter(df[param], df['command_acc'],
                             c=df['criteria_met'], cmap='RdYlGn',
                             s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                             vmin=0, vmax=4)

        # Add text labels
        for _, row in df.iterrows():
            ax1.annotate(f"{row['criteria_met']}",
                        (row[param], row['command_acc']),
                        ha='center', va='center', fontsize=9, weight='bold')

        ax1.set_xlabel(param_label, fontsize=12, weight='bold')
        ax1.set_ylabel('Command Accuracy', fontsize=12, weight='bold')
        ax1.set_title(f'{param_label} Impact on Command Accuracy', fontsize=13, weight='bold')
        ax1.grid(True, alpha=0.4, linestyle='--')

        # Plot 2: param vs entropy
        ax2 = fig.add_subplot(gs[idx, 1])
        scatter2 = ax2.scatter(df[param], df['entropy'],
                              c=df['g0_ratio'], cmap='RdYlGn_r',
                              s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                              vmin=0, vmax=1)

        # Add reference line at entropy = 1.5
        ax2.axhline(1.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target (1.5)')

        ax2.set_xlabel(param_label, fontsize=12, weight='bold')
        ax2.set_ylabel('Prediction Entropy (bits)', fontsize=12, weight='bold')
        ax2.set_title(f'{param_label} Impact on Prediction Diversity', fontsize=13, weight='bold')
        ax2.grid(True, alpha=0.4, linestyle='--')
        ax2.legend(fontsize=10)

        # Add colorbar for second plot
        if idx == 0:
            cbar = plt.colorbar(scatter, ax=ax1, pad=0.02)
            cbar.set_label('Success Criteria Met', fontsize=10, weight='bold')
            cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.02)
            cbar2.set_label('G0 Prediction Ratio', fontsize=10, weight='bold')

    # Add correlation heatmap in remaining space
    ax_corr = fig.add_subplot(gs[2, :])
    corr_data = df[['command_weight', 'label_smoothing', 'hidden_dim', 'learning_rate',
                     'command_acc', 'entropy', 'g0_ratio', 'non_g0_acc']].corr()
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                ax=ax_corr, cbar_kws={'label': 'Correlation Coefficient'},
                linewidths=1, linecolor='white')
    ax_corr.set_title('Hyperparameter Correlation Matrix', fontsize=14, weight='bold', pad=15)

    plt.suptitle('Comprehensive Hyperparameter Analysis', fontsize=18, weight='bold', y=0.995)
    plt.savefig(output_dir / 'hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. CLASS IMBALANCE METRICS - Multi-panel comparison
    print("  Creating class imbalance metrics comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Sort by criteria met
    df_sorted = df.sort_values('criteria_met', ascending=False)

    # Panel 1: G0 Ratio
    ax = axes[0, 0]
    colors = ['#27AE60' if r < 0.5 else '#E74C3C' for r in df_sorted['g0_ratio']]
    bars = ax.barh(range(len(df_sorted)), df_sorted['g0_ratio'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=3, label='Target < 50%', alpha=0.8)
    ax.axvline(0.9, color='red', linestyle=':', linewidth=2, label='Baseline (90%)', alpha=0.6)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels([f"{int(row['criteria_met'])}/4" for _, row in df_sorted.iterrows()], fontsize=10)
    ax.set_xlabel('G0 Prediction Ratio', fontsize=12, weight='bold')
    ax.set_ylabel('Success Criteria Met', fontsize=12, weight='bold')
    ax.set_title('G0 Over-Prediction Problem\n(Lower is Better)', fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, axis='x')

    # Add value labels
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(row['g0_ratio'] + 0.02, i, f"{row['g0_ratio']:.1%}",
               va='center', fontsize=9, weight='bold')

    # Panel 2: Entropy
    ax = axes[0, 1]
    colors = ['#27AE60' if e > 1.5 else '#E74C3C' for e in df_sorted['entropy']]
    bars = ax.barh(range(len(df_sorted)), df_sorted['entropy'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axvline(1.5, color='black', linestyle='--', linewidth=3, label='Target > 1.5', alpha=0.8)
    ax.axvline(0.0, color='red', linestyle=':', linewidth=2, label='Baseline (0)', alpha=0.6)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels([f"{int(row['criteria_met'])}/4" for _, row in df_sorted.iterrows()], fontsize=10)
    ax.set_xlabel('Prediction Entropy (bits)', fontsize=12, weight='bold')
    ax.set_ylabel('Success Criteria Met', fontsize=12, weight='bold')
    ax.set_title('Prediction Diversity\n(Higher is Better)', fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, axis='x')

    # Add value labels
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(row['entropy'] + 0.05, i, f"{row['entropy']:.2f}",
               va='center', fontsize=9, weight='bold')

    # Panel 3: Non-G0 Accuracy
    ax = axes[1, 0]
    colors = ['#27AE60' if a > 0.2 else '#E74C3C' for a in df_sorted['non_g0_acc']]
    bars = ax.barh(range(len(df_sorted)), df_sorted['non_g0_acc'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axvline(0.2, color='black', linestyle='--', linewidth=3, label='Target > 20%', alpha=0.8)
    ax.axvline(0.0, color='red', linestyle=':', linewidth=2, label='Baseline (0%)', alpha=0.6)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels([f"{int(row['criteria_met'])}/4" for _, row in df_sorted.iterrows()], fontsize=10)
    ax.set_xlabel('Non-G0 Command Accuracy', fontsize=12, weight='bold')
    ax.set_ylabel('Success Criteria Met', fontsize=12, weight='bold')
    ax.set_title('Minority Class Performance\n(Higher is Better)', fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, axis='x')

    # Add value labels
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(row['non_g0_acc'] + 0.01, i, f"{row['non_g0_acc']:.1%}",
               va='center', fontsize=9, weight='bold')

    # Panel 4: Operation Type Accuracy
    ax = axes[1, 1]
    colors = ['#27AE60' if a > 0.6 else '#E74C3C' for a in df_sorted['operation_acc']]
    bars = ax.barh(range(len(df_sorted)), df_sorted['operation_acc'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axvline(0.6, color='black', linestyle='--', linewidth=3, label='Target > 60%', alpha=0.8)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels([f"{int(row['criteria_met'])}/4" for _, row in df_sorted.iterrows()], fontsize=10)
    ax.set_xlabel('Operation Type Accuracy', fontsize=12, weight='bold')
    ax.set_ylabel('Success Criteria Met', fontsize=12, weight='bold')
    ax.set_title('Sequence Classification\n(Higher is Better)', fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, axis='x')

    # Add value labels
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(row['operation_acc'] + 0.01, i, f"{row['operation_acc']:.1%}",
               va='center', fontsize=9, weight='bold')

    plt.suptitle('Class Imbalance Solution: Success Criteria Analysis', fontsize=18, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'class_imbalance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. ENHANCED OPERATION TYPE CONFUSION MATRIX (Best Run)
    print("  Creating enhanced operation confusion matrix...")
    best_result = results[df['criteria_met'].idxmax()]

    if best_result['metrics']['operation_confusion_matrix'] is not None:
        cm = np.array(best_result['metrics']['operation_confusion_matrix'])

        # Calculate percentages and metrics
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Calculate precision, recall, F1 for each class
        precision = np.diag(cm) / cm.sum(axis=0)
        recall = np.diag(cm) / cm.sum(axis=1)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)

        # Main confusion matrix with dual annotations
        ax_main = fig.add_subplot(gs[0])
        operation_names = ['adaptive', 'adpt150025', 'damage', 'face', 'pocket', 'unknown']

        # Create custom annotations with counts and percentages
        annotations = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = cm_pct[i, j]
                if count > 0:
                    annotations[i, j] = f'{count}\n({pct:.1f}%)'
                else:
                    annotations[i, j] = '0'

        sns.heatmap(cm, annot=annotations, fmt='', cmap='YlOrRd', ax=ax_main,
                   xticklabels=operation_names, yticklabels=operation_names,
                   cbar_kws={'label': 'Count'}, linewidths=2, linecolor='white',
                   vmin=0, square=True)

        ax_main.set_xlabel('Predicted Operation Type', fontsize=14, fontweight='bold')
        ax_main.set_ylabel('True Operation Type', fontsize=14, fontweight='bold')
        ax_main.set_title(f'Operation Type Confusion Matrix - Best Run\n{best_result["run_name"]} (Criteria: {best_result["validation"]["num_criteria_met"]}/4)',
                         fontsize=16, fontweight='bold', pad=20)
        ax_main.tick_params(labelsize=11)

        # Metrics panel
        ax_metrics = fig.add_subplot(gs[1])
        ax_metrics.axis('off')

        # Create metrics table
        metrics_data = []
        for i, op_name in enumerate(operation_names):
            metrics_data.append([
                op_name,
                f'{precision[i]:.2f}' if not np.isnan(precision[i]) else 'N/A',
                f'{recall[i]:.2f}' if not np.isnan(recall[i]) else 'N/A',
                f'{f1[i]:.2f}' if not np.isnan(f1[i]) else 'N/A'
            ])

        # Add overall accuracy
        overall_acc = np.trace(cm) / cm.sum()

        table = ax_metrics.table(
            cellText=metrics_data,
            colLabels=['Operation', 'Precision', 'Recall', 'F1'],
            cellLoc='center',
            loc='upper center',
            bbox=[0, 0.3, 1, 0.6]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color cells by performance
        for i in range(1, len(operation_names) + 1):
            for j in range(1, 4):
                cell = table[(i, j)]
                try:
                    val = float(cell.get_text().get_text())
                    if val >= 0.8:
                        cell.set_facecolor('#90EE90')
                    elif val >= 0.6:
                        cell.set_facecolor('#FFFFE0')
                    else:
                        cell.set_facecolor('#FFB6C6')
                except:
                    cell.set_facecolor('#D3D3D3')

        # Add overall accuracy text
        ax_metrics.text(0.5, 0.15, f'Overall Accuracy\n{overall_acc:.1%}',
                       ha='center', va='center', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

        plt.savefig(output_dir / 'operation_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 6. ENHANCED CRITERIA MET SUMMARY - Multi-panel Analysis
    print("  Creating enhanced criteria summary...")
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Panel 1: Criteria distribution bar chart with percentages
    ax1 = fig.add_subplot(gs[0, 0])
    criteria_counts = df['criteria_met'].value_counts().sort_index()
    total_runs = len(df)

    colors = ['#E74C3C' if c < 2 else '#F39C12' if c < 4 else '#27AE60' for c in criteria_counts.index]
    bars = ax1.bar(criteria_counts.index, criteria_counts.values, color=colors,
                   alpha=0.8, edgecolor='black', linewidth=2, width=0.6)

    ax1.set_xlabel('Number of Success Criteria Met (out of 4)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Number of Runs', fontsize=13, fontweight='bold')
    ax1.set_title('Criteria Success Distribution', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks([0, 1, 2, 3, 4])
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim(0, criteria_counts.max() * 1.2)

    # Add value labels with percentages on bars
    for bar, (idx, count) in zip(bars, criteria_counts.items()):
        height = bar.get_height()
        pct = count / total_runs * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)} runs\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Panel 2: Criteria breakdown by hyperparameter - Command Weight
    ax2 = fig.add_subplot(gs[0, 1])
    for cw in df['command_weight'].unique():
        subset = df[df['command_weight'] == cw]
        criteria_dist = subset['criteria_met'].value_counts().reindex([0, 1, 2, 3, 4], fill_value=0)
        ax2.plot([0, 1, 2, 3, 4], criteria_dist.values, marker='o', markersize=10,
                linewidth=2.5, label=f'cmd_weight={cw}', alpha=0.8)

    ax2.set_xlabel('Criteria Met', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Number of Runs', fontsize=13, fontweight='bold')
    ax2.set_title('Criteria vs Command Weight', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks([0, 1, 2, 3, 4])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='best')

    # Panel 3: Criteria breakdown by hyperparameter - Hidden Dim
    ax3 = fig.add_subplot(gs[1, 0])
    for hd in sorted(df['hidden_dim'].unique()):
        subset = df[df['hidden_dim'] == hd]
        criteria_dist = subset['criteria_met'].value_counts().reindex([0, 1, 2, 3, 4], fill_value=0)
        ax3.plot([0, 1, 2, 3, 4], criteria_dist.values, marker='s', markersize=10,
                linewidth=2.5, label=f'hidden_dim={hd}', alpha=0.8)

    ax3.set_xlabel('Criteria Met', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Number of Runs', fontsize=13, fontweight='bold')
    ax3.set_title('Criteria vs Hidden Dimension', fontsize=15, fontweight='bold', pad=15)
    ax3.set_xticks([0, 1, 2, 3, 4])
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=10, loc='best')

    # Panel 4: Success rate summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Calculate success statistics
    individual_criteria = [
        ('G0 Ratio < 50%', (df['g0_ratio'] < 0.5).sum()),
        ('Entropy > 1.5', (df['entropy'] > 1.5).sum()),
        ('Non-G0 Acc > 20%', (df['non_g0_acc'] > 0.2).sum()),
        ('Op Acc > 60%', (df['operation_type_acc'] > 0.6).sum()),
    ]

    table_data = []
    for criterion, count in individual_criteria:
        pct = count / total_runs * 100
        status = '✓ PASS' if pct >= 50 else '✗ FAIL'
        table_data.append([criterion, f'{count}/{total_runs}', f'{pct:.1f}%', status])

    # Add overall row
    all_pass = (df['criteria_met'] == 4).sum()
    all_pct = all_pass / total_runs * 100
    table_data.append(['ALL CRITERIA', f'{all_pass}/{total_runs}', f'{all_pct:.1f}%',
                      '✓ PASS' if all_pct > 0 else '✗ FAIL'])

    table = ax4.table(
        cellText=table_data,
        colLabels=['Criterion', 'Runs Passing', 'Success Rate', 'Status'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0.2, 1, 0.7]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white', size=12)

    # Color code cells
    for i in range(1, len(table_data) + 1):
        # Color by success rate
        pct_text = table[(i, 2)].get_text().get_text()
        pct_val = float(pct_text.replace('%', ''))

        if i == len(table_data):  # Last row (ALL CRITERIA)
            color = '#D5F4E6' if pct_val > 0 else '#FADBD8'
            for j in range(4):
                table[(i, j)].set_facecolor(color)
                table[(i, j)].set_text_props(weight='bold', size=12)
        else:
            color = '#D5F4E6' if pct_val >= 50 else '#FADBD8'
            for j in range(4):
                table[(i, j)].set_facecolor(color)

    ax4.text(0.5, 0.05, f'Total Configurations Tested: {total_runs}',
            ha='center', va='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))

    plt.savefig(output_dir / 'criteria_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. PARALLEL COORDINATES PLOT - Hyperparameter Space Exploration
    print("  Creating parallel coordinates visualization...")
    try:
        from pandas.plotting import parallel_coordinates

        fig, ax = plt.subplots(figsize=(18, 8))

        # Normalize continuous variables for visualization
        plot_df = df.copy()

        # Normalize each continuous variable to [0, 1]
        for col in ['command_weight', 'label_smoothing', 'hidden_dim', 'learning_rate',
                    'command_acc', 'entropy', 'g0_ratio', 'non_g0_acc', 'operation_type_acc']:
            if col in plot_df.columns:
                min_val = plot_df[col].min()
                max_val = plot_df[col].max()
                if max_val > min_val:
                    plot_df[f'{col}_norm'] = (plot_df[col] - min_val) / (max_val - min_val)
                else:
                    plot_df[f'{col}_norm'] = 0.5

        # Create categorical variable for color coding
        plot_df['performance'] = plot_df['criteria_met'].apply(
            lambda x: 'Poor (0-1)' if x < 2 else 'Fair (2-3)' if x < 4 else 'Excellent (4)'
        )

        # Select columns for parallel coordinates
        pc_cols = ['command_weight_norm', 'label_smoothing_norm', 'hidden_dim_norm',
                  'learning_rate_norm', 'command_acc_norm', 'entropy_norm',
                  'g0_ratio_norm', 'non_g0_acc_norm', 'operation_type_acc_norm', 'performance']

        parallel_coordinates(plot_df[pc_cols], 'performance',
                           color=['#E74C3C', '#F39C12', '#27AE60'],
                           alpha=0.6, linewidth=2.5, ax=ax)

        # Customize axis labels
        axis_labels = [
            'Command\nWeight',
            'Label\nSmoothing',
            'Hidden\nDim',
            'Learning\nRate',
            'Command\nAccuracy',
            'Entropy',
            'G0\nRatio',
            'Non-G0\nAccuracy',
            'Operation\nAccuracy'
        ]
        ax.set_xticks(range(len(axis_labels)))
        ax.set_xticklabels(axis_labels, fontsize=11, fontweight='bold')
        ax.set_ylabel('Normalized Value [0-1]', fontsize=13, fontweight='bold')
        ax.set_title('Parallel Coordinates: Hyperparameter-Performance Relationships',
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.legend(title='Criteria Met', fontsize=11, title_fontsize=12, loc='upper right')

        # Add vertical lines to separate hyperparameters from metrics
        ax.axvline(x=3.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax.text(1.5, 0.95, 'Hyperparameters', ha='center', va='bottom',
               fontsize=12, fontweight='bold', style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
        ax.text(6.5, 0.95, 'Performance Metrics', ha='center', va='bottom',
               fontsize=12, fontweight='bold', style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_dir / 'parallel_coordinates.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved 7 enhanced visualization plots to {output_dir}")

    except Exception as e:
        print(f"  ! Could not create parallel coordinates plot: {e}")
        print(f"  ✓ Saved 6 enhanced visualization plots to {output_dir}")


def generate_report(results: List[Dict], output_dir: Path, decomposer: TokenDecomposer):
    """Generate comprehensive markdown report."""
    print("\nGenerating final report...")

    report_path = output_dir / 'class_imbalance_report.md'

    # Find best run
    best_idx = max(range(len(results)), key=lambda i: results[i]['validation']['num_criteria_met'])
    best_run = results[best_idx]

    # Calculate summary statistics
    num_all_pass = sum(1 for r in results if r['validation']['all_criteria_met'])
    avg_g0_ratio = np.mean([r['metrics']['g0_prediction_ratio'] for r in results])
    avg_entropy = np.mean([r['metrics']['command_entropy'] for r in results])
    avg_non_g0_acc = np.mean([r['metrics']['non_g0_accuracy'] for r in results])
    avg_operation_acc = np.mean([r['metrics']['operation_type_acc'] for r in results])

    with open(report_path, 'w') as f:
        f.write("# Fast Validation Sweep - Class Imbalance Analysis Report\n\n")
        f.write(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Runs**: {len(results)}\n")
        f.write(f"- **Runs Meeting All Criteria**: {num_all_pass}/{len(results)} ({num_all_pass/len(results)*100:.1f}%)\n")
        f.write(f"- **Average G0 Ratio**: {avg_g0_ratio:.3f} (Target: < 0.50)\n")
        f.write(f"- **Average Entropy**: {avg_entropy:.3f} (Target: > 1.50)\n")
        f.write(f"- **Average Non-G0 Accuracy**: {avg_non_g0_acc:.3f} (Target: > 0.20)\n")
        f.write(f"- **Average Operation Accuracy**: {avg_operation_acc:.3f} (Target: > 0.60)\n\n")

        # Success Criteria
        f.write("## Success Criteria Validation\n\n")
        f.write("The fast validation sweep tested class weighting approach with these goals:\n\n")
        f.write("| Criterion | Target | Status |\n")
        f.write("|-----------|--------|--------|\n")
        f.write(f"| G0 predictions | < 50% (was 100%) | {'✓ PASS' if avg_g0_ratio < 0.5 else '✗ FAIL'} ({avg_g0_ratio*100:.1f}%) |\n")
        f.write(f"| Command diversity entropy | > 1.5 (was 0) | {'✓ PASS' if avg_entropy > 1.5 else '✗ FAIL'} ({avg_entropy:.2f}) |\n")
        f.write(f"| Non-G0 accuracy | > 20% (was 0%) | {'✓ PASS' if avg_non_g0_acc > 0.2 else '✗ FAIL'} ({avg_non_g0_acc*100:.1f}%) |\n")
        f.write(f"| Operation type accuracy | > 60% | {'✓ PASS' if avg_operation_acc > 0.6 else '✗ FAIL'} ({avg_operation_acc*100:.1f}%) |\n\n")

        # Best Configuration
        f.write("## Best Configuration\n\n")
        f.write(f"**Run**: {best_run['run_name']} ({best_run['run_id']})\n\n")
        f.write(f"**Criteria Met**: {best_run['validation']['summary']}\n\n")

        f.write("### Hyperparameters:\n\n")
        config = best_run['config']
        f.write(f"- **command_weight**: {config['command_weight']}\n")
        f.write(f"- **label_smoothing**: {config['label_smoothing']}\n")
        f.write(f"- **hidden_dim**: {config['hidden_dim']}\n")
        f.write(f"- **learning_rate**: {config['learning_rate']}\n\n")

        f.write("### Performance:\n\n")
        metrics = best_run['metrics']
        f.write(f"- **Command Accuracy**: {metrics['command_acc']:.4f} ({metrics['command_acc']*100:.2f}%)\n")
        f.write(f"- **G0 Prediction Ratio**: {metrics['g0_prediction_ratio']:.4f} ({metrics['g0_prediction_ratio']*100:.2f}%)\n")
        f.write(f"- **Command Entropy**: {metrics['command_entropy']:.4f} bits\n")
        f.write(f"- **Non-G0 Accuracy**: {metrics['non_g0_accuracy']:.4f} ({metrics['non_g0_accuracy']*100:.2f}%)\n")
        f.write(f"- **Operation Type Accuracy**: {metrics['operation_type_acc']:.4f} ({metrics['operation_type_acc']*100:.2f}%)\n\n")

        # Per-command accuracy
        f.write("### Per-Command Accuracy:\n\n")
        f.write("| Command | Accuracy |\n")
        f.write("|---------|----------|\n")
        command_names = decomposer.command_tokens
        for cmd_id, acc in sorted(metrics['per_command_accuracy'].items()):
            cmd_name = command_names[cmd_id] if cmd_id < len(command_names) else f"CMD_{cmd_id}"
            f.write(f"| {cmd_name} | {acc:.4f} ({acc*100:.2f}%) |\n")
        f.write("\n")

        # Visualizations
        f.write("## Visualizations\n\n")
        f.write("- [Success Criteria Radar](success_criteria_radar.png)\n")
        f.write("- [Hyperparameter Analysis](hyperparameter_analysis.png)\n")
        f.write("- [G0 Ratio Comparison](g0_ratio_comparison.png)\n")
        f.write("- [Entropy Comparison](entropy_comparison.png)\n")
        f.write("- [Operation Confusion Matrix](operation_confusion_matrix.png)\n")
        f.write("- [Criteria Summary](criteria_summary.png)\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        if num_all_pass > 0:
            f.write("### ✓ SUCCESS: Class Imbalance Problem Solved!\n\n")
            f.write("The class weighting approach successfully addressed the G0 prediction bias.\n\n")
            f.write("**Next Steps:**\n\n")
            f.write("1. **Deploy Best Configuration**: Use the best hyperparameters for production training\n")
            f.write("2. **Extended Training**: Train for 50-100 epochs with best config\n")
            f.write("3. **Test Set Evaluation**: Validate performance on held-out test set\n")
            f.write("4. **Monitor Deployment**: Track prediction diversity in production\n\n")
        else:
            f.write("### ⚠️ PARTIAL SUCCESS: Further Tuning Needed\n\n")
            f.write("Class weighting improved prediction diversity, but some criteria not fully met.\n\n")
            f.write("**Recommended Adjustments:**\n\n")

            if avg_g0_ratio >= 0.5:
                f.write("- **Increase command_weight**: Try 20.0-25.0 (current: 10-15)\n")

            if avg_entropy < 1.5:
                f.write("- **Add regularization**: Increase dropout or add L2 penalty\n")

            if avg_non_g0_acc < 0.2:
                f.write("- **Class-specific training**: Consider focal loss or per-class weighting\n")

            if avg_operation_acc < 0.6:
                f.write("- **Increase operation_weight**: Try 5.0-10.0 (current: 2.0)\n")
                f.write("- **Add operation-specific features**: Consider task-specific layers\n\n")

        # Detailed Results Table
        f.write("## Detailed Results\n\n")
        f.write("| Run | command_weight | label_smoothing | Command Acc | G0 Ratio | Entropy | Non-G0 Acc | Operation Acc | Criteria |\n")
        f.write("|-----|----------------|-----------------|-------------|----------|---------|------------|---------------|----------|\n")

        # Sort by criteria met
        sorted_results = sorted(results, key=lambda r: r['validation']['num_criteria_met'], reverse=True)

        for r in sorted_results:
            config = r['config']
            m = r['metrics']
            v = r['validation']
            f.write(f"| {r['run_name'][:20]} | ")
            f.write(f"{config['command_weight']} | ")
            f.write(f"{config['label_smoothing']} | ")
            f.write(f"{m['command_acc']:.3f} | ")
            f.write(f"{m['g0_prediction_ratio']:.3f} | ")
            f.write(f"{m['command_entropy']:.3f} | ")
            f.write(f"{m['non_g0_accuracy']:.3f} | ")
            f.write(f"{m['operation_type_acc']:.3f} | ")
            f.write(f"{v['num_criteria_met']}/4 |\n")

        f.write("\n")

    print(f"  ✓ Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate sweep with class imbalance metrics')
    parser.add_argument('--sweep-results', type=str,
                       default='outputs/sweep_analysis/sweep_results.csv',
                       help='Path to sweep results CSV')
    parser.add_argument('--data-dir', type=str,
                       default='outputs/processed_with_ops',
                       help='Directory with validation data')
    parser.add_argument('--vocab-path', type=str,
                       default='data/vocabulary.json',
                       help='Path to vocabulary JSON')
    parser.add_argument('--output', type=str,
                       default='outputs/sweep_fast_validation_analysis',
                       help='Output directory for analysis')
    parser.add_argument('--max-runs', type=int, default=None,
                       help='Maximum number of runs to analyze (for testing)')

    args = parser.parse_args()

    # Setup
    sweep_csv = Path(args.sweep_results)
    data_dir = Path(args.data_dir)
    vocab_path = Path(args.vocab_path)
    output_dir = Path(args.output)

    # Validate inputs
    if not sweep_csv.exists():
        print(f"❌ Sweep results not found: {sweep_csv}")
        return 1

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return 1

    if not vocab_path.exists():
        print(f"❌ Vocabulary file not found: {vocab_path}")
        return 1

    # Get device
    device = get_device()
    print(f"Using device: {device}\n")

    # Load decomposer for report generation
    decomposer = TokenDecomposer(str(vocab_path))

    # Analyze all runs
    results = analyze_sweep_runs(sweep_csv, data_dir, vocab_path, device, max_runs=args.max_runs)

    if len(results) == 0:
        print("❌ No runs successfully analyzed")
        return 1

    # Save results to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    results_json = output_dir / 'detailed_results.json'
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Saved detailed results to {results_json}")

    # Create visualizations
    create_visualizations(results, output_dir, decomposer)

    # Generate report
    generate_report(results, output_dir, decomposer)

    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - detailed_results.json")
    print(f"  - class_imbalance_report.md")
    print(f"  - 6 visualization PNGs")
    print(f"\nNext: Review {output_dir}/class_imbalance_report.md for full analysis")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
