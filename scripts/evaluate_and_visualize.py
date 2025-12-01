#!/usr/bin/env python3
"""
Comprehensive Test Evaluation and Visualization Script
Generates all figures needed for the final presentation.
"""
import os
import sys
import json
import platform
from pathlib import Path

if platform.system() == 'Darwin':
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.manifold import TSNE
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.multihead_lm import MultiHeadGCodeLM
from miracle.dataset.dataset import GCodeDataset, collate_fn
from miracle.dataset.target_utils import TokenDecomposer
from torch.utils.data import DataLoader

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.facecolor'] = 'white'


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config_dict = checkpoint.get('config', {})

    return checkpoint, config_dict


def create_models(config_dict, decomposer, device):
    """Create backbone and multi-head models."""
    # Build ModelConfig with correct parameter names
    model_config = ModelConfig(
        sensor_dims=[219, 4],  # continuous and categorical dims
        d_model=config_dict.get('hidden_dim', 256),
        n_heads=config_dict.get('num_heads', 8),
        lstm_layers=config_dict.get('num_layers', 5),
        dropout=config_dict.get('dropout', 0.2),
        gcode_vocab=decomposer.vocab_size,
    )

    backbone = MM_DTAE_LSTM(model_config).to(device)

    # Multi-head LM
    model = MultiHeadGCodeLM(
        d_model=config_dict.get('hidden_dim', 256),
        vocab_size=decomposer.vocab_size,
        n_commands=decomposer.n_commands,
        n_param_types=decomposer.n_param_types,
        n_param_values=decomposer.n_param_values,
        n_operation_types=9,  # 9-class operation type
        n_heads=config_dict.get('num_heads', 8),
        n_layers=2,
        dropout=config_dict.get('dropout', 0.2),
        param_value_regression=True,
    ).to(device)

    return backbone, model


def evaluate_model(backbone, model, dataloader, decomposer, device):
    """Run evaluation and collect predictions."""
    backbone.eval()
    model.eval()

    all_operation_preds = []
    all_operation_targets = []
    all_command_preds = []
    all_command_targets = []
    all_param_type_preds = []
    all_param_type_targets = []
    all_embeddings = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            continuous = batch['continuous'].to(device)
            categorical = batch['categorical'].float().to(device)
            tokens = batch['tokens'].to(device)
            lengths = batch['lengths'].to(device)
            operation_types = batch.get('operation_type', None)

            tgt_in = tokens[:, :-1]
            tgt_out = tokens[:, 1:]

            # Forward
            mods = [continuous, categorical]
            backbone_out = backbone(mods=mods, lengths=lengths, gcode_in=tgt_in)
            memory = backbone_out['memory']

            logits = model(memory, tgt_in)

            # Get predictions
            operation_pred = torch.argmax(logits['operation_logits'], dim=-1)
            command_pred = torch.argmax(logits['command_logits'], dim=-1)
            param_type_pred = torch.argmax(logits['param_type_logits'], dim=-1)

            # Decompose targets
            tgt_decomposed = decomposer.decompose_batch(tgt_out)

            # Collect predictions (using sequence-level operation type)
            if operation_types is not None:
                all_operation_preds.extend(operation_pred[:, 0].cpu().numpy())
                all_operation_targets.extend(operation_types.cpu().numpy())

                # Store embeddings for t-SNE
                seq_embedding = memory.mean(dim=1)
                all_embeddings.append(seq_embedding.cpu().numpy())
                all_labels.extend(operation_types.cpu().numpy())

            # Token-level metrics
            mask = tgt_decomposed['type'] != 0
            all_command_preds.extend(command_pred[mask].cpu().numpy())
            all_command_targets.extend(tgt_decomposed['command'][mask].cpu().numpy())
            all_param_type_preds.extend(param_type_pred[mask].cpu().numpy())
            all_param_type_targets.extend(tgt_decomposed['param_type'][mask].cpu().numpy())

            total_samples += continuous.size(0)

    results = {
        'operation_preds': np.array(all_operation_preds),
        'operation_targets': np.array(all_operation_targets),
        'command_preds': np.array(all_command_preds),
        'command_targets': np.array(all_command_targets),
        'param_type_preds': np.array(all_param_type_preds),
        'param_type_targets': np.array(all_param_type_targets),
        'embeddings': np.vstack(all_embeddings) if all_embeddings else None,
        'labels': np.array(all_labels) if all_labels else None,
    }

    return results


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_class_metrics(y_true, y_pred, class_names, save_path):
    """Plot per-class precision, recall, F1."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2196F3')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#4CAF50')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#FF9800')

    ax.set_xlabel('Operation Type', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Classification Metrics (9-Class Operation Type)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

    return {'precision': precision, 'recall': recall, 'f1': f1, 'support': support}


def plot_tsne_embeddings(embeddings, labels, class_names, save_path):
    """Plot t-SNE visualization of embeddings."""
    if embeddings is None or len(embeddings) < 10:
        print("Insufficient embeddings for t-SNE")
        return

    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Colors for 9 classes
    colors = [
        '#4CAF50', '#2E7D32', '#1B5E20',  # Face: greens
        '#2196F3', '#1565C0', '#0D47A1',  # Pocket: blues
        '#FF9800', '#EF6C00', '#E65100'   # Adaptive: oranges
    ]

    fig, ax = plt.subplots(figsize=(12, 10))

    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=color, label=cls_name, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE Embedding Space: 9-Class Operation Type Classification', fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_summary(results, class_names, save_path):
    """Create a summary dashboard."""
    fig = plt.figure(figsize=(16, 12))

    # Overall accuracy
    op_acc = (results['operation_preds'] == results['operation_targets']).mean()
    cmd_acc = (results['command_preds'] == results['command_targets']).mean()
    pt_acc = (results['param_type_preds'] == results['param_type_targets']).mean()

    # 1. Overall metrics bar chart
    ax1 = fig.add_subplot(2, 2, 1)
    metrics = ['Operation\n(9-class)', 'Command', 'Param Type']
    values = [op_acc, cmd_acc, pt_acc]
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    bars = ax1.bar(metrics, values, color=colors)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Multi-Head Model Accuracy', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, values):
        ax1.annotate(f'{val:.1%}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 2. Operation type distribution
    ax2 = fig.add_subplot(2, 2, 2)
    unique, counts = np.unique(results['operation_targets'], return_counts=True)
    colors_pie = ['#4CAF50', '#2E7D32', '#1B5E20', '#2196F3', '#1565C0', '#0D47A1', '#FF9800', '#EF6C00', '#E65100']
    ax2.pie(counts, labels=[class_names[i] for i in unique], autopct='%1.0f%%',
            colors=[colors_pie[i] for i in unique], startangle=90)
    ax2.set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')

    # 3. Confusion matrix (small version)
    ax3 = fig.add_subplot(2, 2, 3)
    cm = confusion_matrix(results['operation_targets'], results['operation_preds'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax3,
                xticklabels=[c.replace('\n', ' ') for c in class_names],
                yticklabels=[c.replace('\n', ' ') for c in class_names])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    ax3.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=8)

    # 4. Per-class recall
    ax4 = fig.add_subplot(2, 2, 4)
    precision, recall, f1, _ = precision_recall_fscore_support(
        results['operation_targets'], results['operation_preds'],
        labels=range(len(class_names)), zero_division=0
    )
    x = np.arange(len(class_names))
    ax4.bar(x, recall, color=colors_pie)
    ax4.set_xticks(x)
    ax4.set_xticklabels([c.replace('\n', ' ') for c in class_names], rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Recall')
    ax4.set_title('Per-Class Recall', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 1.1)
    ax4.grid(axis='y', alpha=0.3)

    plt.suptitle('G-Code Fingerprinting: Test Set Evaluation Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

    return {'operation_acc': op_acc, 'command_acc': cmd_acc, 'param_type_acc': pt_acc}


def main():
    # Paths
    base_path = Path(__file__).parent.parent
    checkpoint_path = base_path / "outputs/operation_acc_model_v1/checkpoint_best.pt"
    data_dir = base_path / "outputs/processed_hybrid"
    vocab_path = base_path / "data/vocabulary_1digit_hybrid.json"
    output_dir = base_path / "outputs/figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load decomposer
    print("Loading token decomposer...")
    decomposer = TokenDecomposer(str(vocab_path))

    # Load checkpoint
    checkpoint, config_dict = load_checkpoint(checkpoint_path, device)

    # Create models
    print("Creating models...")
    backbone, model = create_models(config_dict, decomposer, device)

    # Load weights
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded model weights")

    # Load validation dataset (using as test set)
    print("Loading dataset...")
    val_dataset = GCodeDataset(str(data_dir / "val"))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)

    # 9-class operation type names
    class_names = [
        'Face\nAir', 'Face\nEngaged', 'Face\nDamaged',
        'Pocket\nAir', 'Pocket\nEngaged', 'Pocket\nDamaged',
        'Adaptive\nAir', 'Adaptive\nEngaged', 'Adaptive\nDamaged'
    ]

    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate_model(backbone, model, val_loader, decomposer, device)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Confusion matrix
    plot_confusion_matrix(
        results['operation_targets'], results['operation_preds'],
        class_names, '9-Class Operation Type Confusion Matrix',
        output_dir / 'confusion_matrix_9class.png'
    )

    # 2. Per-class metrics
    metrics = plot_per_class_metrics(
        results['operation_targets'], results['operation_preds'],
        class_names, output_dir / 'per_class_metrics.png'
    )

    # 3. t-SNE
    if results['embeddings'] is not None:
        plot_tsne_embeddings(
            results['embeddings'], results['labels'],
            class_names, output_dir / 'tsne_real_embeddings.png'
        )

    # 4. Summary dashboard
    summary = plot_model_summary(results, class_names, output_dir / 'evaluation_summary.png')

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Operation Type Accuracy (9-class): {summary['operation_acc']:.2%}")
    print(f"Command Accuracy: {summary['command_acc']:.2%}")
    print(f"Param Type Accuracy: {summary['param_type_acc']:.2%}")
    print("\nPer-class metrics:")
    for i, name in enumerate(class_names):
        print(f"  {name.replace(chr(10), ' ')}: P={metrics['precision'][i]:.2f}, R={metrics['recall'][i]:.2f}, F1={metrics['f1'][i]:.2f}")

    # Save metrics to JSON
    metrics_out = {
        'operation_acc': float(summary['operation_acc']),
        'command_acc': float(summary['command_acc']),
        'param_type_acc': float(summary['param_type_acc']),
        'per_class': {
            class_names[i].replace('\n', ' '): {
                'precision': float(metrics['precision'][i]),
                'recall': float(metrics['recall'][i]),
                'f1': float(metrics['f1'][i]),
                'support': int(metrics['support'][i])
            }
            for i in range(len(class_names))
        }
    }

    with open(output_dir / 'test_evaluation_metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nSaved metrics to: {output_dir / 'test_evaluation_metrics.json'}")

    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
