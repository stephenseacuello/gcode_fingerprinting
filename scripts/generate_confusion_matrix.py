#!/usr/bin/env python3
"""
Generate confusion matrix visualization for 9-class operation type classification.

Usage:
    python scripts/generate_confusion_matrix.py \
        --model-path outputs/operation_acc_model_v1/best_model.pt \
        --data-dir outputs/processed_hybrid \
        --output outputs/figures/confusion_matrix.png
"""

import argparse
import sys
from pathlib import Path
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.dataset.dataset import GCodeDataset
from miracle.model.backbone import SensorTransformerEncoder
from miracle.model.multihead_lm import MultiHeadGCodeLM


# Operation type labels (9 classes)
OPERATION_LABELS = [
    'face_air',
    'face_engaged',
    'face_damaged',
    'pocket_air',
    'pocket_engaged',
    'pocket_damaged',
    'adaptive_air',
    'adaptive_engaged',
    'adaptive_damaged'
]

# Short labels for confusion matrix
SHORT_LABELS = [
    'F-Air', 'F-Eng', 'F-Dmg',
    'P-Air', 'P-Eng', 'P-Dmg',
    'A-Air', 'A-Eng', 'A-Dmg'
]


def load_model(model_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)

    # Get config from checkpoint
    config = checkpoint.get('config', {})

    # Create backbone
    backbone = SensorTransformerEncoder(
        continuous_dim=config.get('continuous_dim', 219),
        categorical_dims=config.get('categorical_dims', [4]),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 5),
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.2)
    )

    # Create multi-head LM
    model = MultiHeadGCodeLM(
        backbone=backbone,
        hidden_dim=config.get('hidden_dim', 256),
        vocab_size=config.get('vocab_size', 69),
        num_commands=config.get('num_commands', 11),
        num_param_types=config.get('num_param_types', 7),
        num_param_values=config.get('num_param_values', 10),
        num_operation_types=9
    )

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'multihead_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['multihead_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def evaluate_model(model, dataloader, device):
    """Run model on dataset and collect predictions."""
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            continuous = batch['continuous'].to(device)
            categorical = batch['categorical'].to(device)

            # Get model outputs
            outputs = model(continuous, categorical)

            # Get operation predictions
            if 'operation' in outputs:
                op_logits = outputs['operation']
                op_preds = op_logits.argmax(dim=-1)

                # Get targets
                op_targets = batch['operation_type'].to(device)

                all_preds.extend(op_preds.cpu().numpy().tolist())
                all_targets.extend(op_targets.cpu().numpy().tolist())

    return np.array(all_preds), np.array(all_targets)


def plot_confusion_matrix(y_true, y_pred, labels, output_path: str):
    """Create and save confusion matrix heatmap."""
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # Normalized (percentages)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
        vmin=0,
        vmax=1
    )
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")

    return cm, cm_normalized


def plot_per_class_metrics(y_true, y_pred, labels, output_path: str):
    """Create per-class precision/recall/F1 bar chart."""
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

    # Extract metrics
    precisions = [report[label]['precision'] for label in labels]
    recalls = [report[label]['recall'] for label in labels]
    f1s = [report[label]['f1-score'] for label in labels]
    supports = [report[label]['support'] for label in labels]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(labels))
    width = 0.25

    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#2196F3')
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#4CAF50')
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#FF9800')

    ax.set_xlabel('Operation Type', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

    # Add support counts as text
    for i, (label, support) in enumerate(zip(labels, supports)):
        ax.annotate(f'n={support}', xy=(i, 0.02), ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Per-class metrics saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrix for operation type classification')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='outputs/processed_hybrid', help='Data directory')
    parser.add_argument('--output', type=str, default='outputs/figures/confusion_matrix.png', help='Output path')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Dataset split')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data_dir = Path(args.data_dir)
    dataset = GCodeDataset(data_dir / f'{args.split}_sequences.npz')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    print(f"Loaded {len(dataset)} samples from {args.split} split")

    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, device)

    # Evaluate
    y_pred, y_true = evaluate_model(model, dataloader, device)
    print(f"Predictions shape: {y_pred.shape}, Targets shape: {y_true.shape}")

    # Generate confusion matrix
    cm, cm_norm = plot_confusion_matrix(
        y_true, y_pred,
        SHORT_LABELS,
        str(output_path)
    )

    # Generate per-class metrics
    metrics_path = output_path.parent / 'per_class_metrics.png'
    report = plot_per_class_metrics(
        y_true, y_pred,
        SHORT_LABELS,
        str(metrics_path)
    )

    # Print classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=SHORT_LABELS))

    # Calculate overall accuracy
    accuracy = (y_pred == y_true).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Save metrics to JSON
    metrics_json_path = output_path.parent / 'classification_metrics.json'
    metrics_data = {
        'accuracy': float(accuracy),
        'per_class': {
            label: {
                'precision': report[label]['precision'],
                'recall': report[label]['recall'],
                'f1-score': report[label]['f1-score'],
                'support': report[label]['support']
            }
            for label in SHORT_LABELS
        },
        'macro_avg': report['macro avg'],
        'weighted_avg': report['weighted avg']
    }
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\nMetrics saved to: {metrics_json_path}")


if __name__ == '__main__':
    main()
