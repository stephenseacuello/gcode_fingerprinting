#!/usr/bin/env python3
"""
Extract metrics from W&B runs for presentation slides.

Usage:
    python scripts/extract_presentation_metrics.py \
        --wandb-dir wandb/run-20251129_210625-4ju36hob \
        --output outputs/figures
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_wandb_summary(wandb_dir: Path) -> dict:
    """Load wandb-summary.json from a run directory."""
    summary_path = wandb_dir / 'files' / 'wandb-summary.json'
    if not summary_path.exists():
        # Try direct path
        summary_path = wandb_dir / 'wandb-summary.json'

    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find wandb-summary.json in {wandb_dir}")

    with open(summary_path, 'r') as f:
        return json.load(f)


def load_wandb_history(wandb_dir: Path) -> list:
    """Load training history from wandb-history.jsonl."""
    history_path = wandb_dir / 'files' / 'wandb-history.jsonl'
    if not history_path.exists():
        history_path = wandb_dir / 'wandb-history.jsonl'

    if not history_path.exists():
        print(f"Warning: Could not find wandb-history.jsonl in {wandb_dir}")
        return []

    history = []
    with open(history_path, 'r') as f:
        for line in f:
            if line.strip():
                history.append(json.loads(line))
    return history


def extract_key_metrics(summary: dict) -> dict:
    """Extract key metrics for presentation."""
    metrics = {
        'operation_acc': summary.get('val/operation_acc', 'N/A'),
        'train_operation_acc': summary.get('train/operation_acc', 'N/A'),
        'val_loss': summary.get('val/loss', 'N/A'),
        'train_loss': summary.get('train/loss', 'N/A'),
        'command_acc': summary.get('val/command_acc', 'N/A'),
        'param_type_acc': summary.get('val/param_type_acc', 'N/A'),
        'param_value_mae': summary.get('val/param_value_mae', 'N/A'),
        'composite_acc': summary.get('val/composite_acc', 'N/A'),
        'epoch': summary.get('epoch', 'N/A'),
        'learning_rate': summary.get('learning_rate', 'N/A'),
    }

    # Extract per-class recall
    for cls in ['G0', 'G1', 'G2', 'G3', 'F']:
        key = f'val/recall_{cls}'
        metrics[f'recall_{cls}'] = summary.get(key, 'N/A')

    return metrics


def plot_loss_curves(history: list, output_path: Path):
    """Plot training and validation loss curves."""
    if not history:
        print("No history data available for loss curves")
        return

    epochs = []
    train_losses = []
    val_losses = []

    for entry in history:
        if 'epoch' in entry and 'train/loss' in entry:
            epochs.append(entry['epoch'])
            train_losses.append(entry.get('train/loss', np.nan))
            val_losses.append(entry.get('val/loss', np.nan))

    if not epochs:
        print("No epoch data found in history")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_losses, label='Train Loss', color='#2196F3', linewidth=2)
    ax.plot(epochs, val_losses, label='Val Loss', color='#FF5722', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Loss curves saved to: {output_path / 'loss_curves.png'}")


def plot_accuracy_curves(history: list, output_path: Path):
    """Plot operation accuracy over epochs."""
    if not history:
        print("No history data available for accuracy curves")
        return

    epochs = []
    train_op_acc = []
    val_op_acc = []

    for entry in history:
        if 'epoch' in entry:
            epochs.append(entry['epoch'])
            train_op_acc.append(entry.get('train/operation_acc', np.nan))
            val_op_acc.append(entry.get('val/operation_acc', np.nan))

    if not epochs:
        print("No epoch data found in history")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_op_acc, label='Train Operation Acc', color='#4CAF50', linewidth=2)
    ax.plot(epochs, val_op_acc, label='Val Operation Acc', color='#9C27B0', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Operation Type Classification Accuracy', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Accuracy curves saved to: {output_path / 'accuracy_curves.png'}")


def create_metrics_table(metrics: dict, output_path: Path):
    """Create a formatted metrics table for presentation."""
    table_content = """
# Model Performance Metrics

## Overall Results
| Metric | Value |
|--------|-------|
| Operation Accuracy (9-class) | {operation_acc:.4f} |
| Validation Loss | {val_loss:.4f} |
| Training Loss | {train_loss:.4f} |
| Epochs Trained | {epoch} |

## Head-wise Performance
| Head | Accuracy |
|------|----------|
| Command | {command_acc:.4f} |
| Param Type | {param_type_acc:.4f} |
| Param Value MAE | {param_value_mae:.4f} |
| Composite | {composite_acc:.4f} |

## Per-Class Recall
| Class | Recall |
|-------|--------|
| G0 | {recall_G0} |
| G1 | {recall_G1} |
| G2 | {recall_G2} |
| G3 | {recall_G3} |
| F | {recall_F} |
""".format(**{k: v if v != 'N/A' else 'N/A' for k, v in metrics.items()})

    with open(output_path / 'metrics_summary.md', 'w') as f:
        f.write(table_content)

    print(f"Metrics summary saved to: {output_path / 'metrics_summary.md'}")

    # Also save as JSON
    with open(output_path / 'metrics_summary.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics JSON saved to: {output_path / 'metrics_summary.json'}")


def print_metrics_summary(metrics: dict):
    """Print formatted metrics summary to console."""
    print("\n" + "="*60)
    print("PRESENTATION METRICS SUMMARY")
    print("="*60)

    print("\n📊 MAIN RESULTS:")
    print(f"   Operation Accuracy (9-class): {metrics['operation_acc']:.4f}" if isinstance(metrics['operation_acc'], float) else f"   Operation Accuracy: {metrics['operation_acc']}")
    print(f"   Validation Loss: {metrics['val_loss']:.4f}" if isinstance(metrics['val_loss'], float) else f"   Validation Loss: {metrics['val_loss']}")
    print(f"   Epochs Trained: {metrics['epoch']}")

    print("\n📈 PER-CLASS RECALL:")
    for cls in ['G0', 'G1', 'G2', 'G3', 'F']:
        val = metrics.get(f'recall_{cls}', 'N/A')
        if isinstance(val, float):
            print(f"   {cls}: {val:.4f}")
        else:
            print(f"   {cls}: {val}")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Extract metrics from W&B runs')
    parser.add_argument('--wandb-dir', type=str, help='Path to wandb run directory')
    parser.add_argument('--output', type=str, default='outputs/figures', help='Output directory')
    parser.add_argument('--find-latest', action='store_true', help='Find latest wandb run automatically')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find wandb directory
    if args.find_latest or not args.wandb_dir:
        # Find most recent wandb run
        wandb_base = Path('wandb')
        if wandb_base.exists():
            runs = sorted(wandb_base.glob('run-*'), key=lambda x: x.stat().st_mtime, reverse=True)
            if runs:
                wandb_dir = runs[0]
                print(f"Using latest run: {wandb_dir}")
            else:
                print("No wandb runs found")
                return
        else:
            print("No wandb directory found")
            return
    else:
        wandb_dir = Path(args.wandb_dir)

    # Load data
    print(f"Loading from: {wandb_dir}")

    try:
        summary = load_wandb_summary(wandb_dir)
        print(f"Loaded summary with {len(summary)} keys")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    history = load_wandb_history(wandb_dir)
    print(f"Loaded history with {len(history)} entries")

    # Extract metrics
    metrics = extract_key_metrics(summary)

    # Print summary
    print_metrics_summary(metrics)

    # Create outputs
    create_metrics_table(metrics, output_path)

    if history:
        plot_loss_curves(history, output_path)
        plot_accuracy_curves(history, output_path)

    print(f"\n✅ All outputs saved to: {output_path}")


if __name__ == '__main__':
    main()
