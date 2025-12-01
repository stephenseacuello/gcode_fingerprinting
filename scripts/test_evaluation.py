#!/usr/bin/env python3
"""
Comprehensive Test Set Evaluation

Evaluates a trained model on the held-out test set and generates detailed metrics.
"""
import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.dataset.dataset import GCodeDataset, collate_fn

def load_model_and_config(checkpoint_path: Path, device: str = 'auto'):
    """Load model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint or config file
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
    else:
        # Try to load from config.json in same directory
        config_path = checkpoint_path.parent / 'config.json'
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Check if checkpoint has attention pooling (Phase 1 improvement)
    has_attention_pooling = 'fp_head.query' in state_dict

    # Create model config, adjusting for Phase 1 compatibility
    if 'use_attention_pooling' not in config_dict:
        config_dict['use_attention_pooling'] = has_attention_pooling

    config = ModelConfig(**config_dict)
    model = MM_DTAE_LSTM(config)

    # Load state dict
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully")
    print(f"   Config: d_model={config.d_model}, lstm_layers={config.lstm_layers}")
    print(f"   Attention Pooling: {'Yes' if has_attention_pooling else 'No (pre-Phase 1)'}")

    return model, config, device

def evaluate_on_test_set(
    model: MM_DTAE_LSTM,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    vocab_size: int,
):
    """Evaluate model on test set."""
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)

    model.eval()
    all_predictions = []
    all_targets = []
    all_losses = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Evaluation"):
            # Move to device
            continuous = batch['continuous'].to(device)
            categorical = batch['categorical'].float().to(device)
            tokens = batch['tokens'].to(device)
            lengths = batch['lengths'].to(device)

            B, T, D_cont = continuous.shape
            D_cat = categorical.size(-1)

            # Create modalities matching model's sensor_dims
            # Model expects sensor_dims=[135, 4] = [continuous, categorical]
            mods = [continuous]
            if D_cat > 0:
                mods.append(categorical.float())  # Convert to float for model

            # Forward pass
            outputs = model(mods, lengths, gcode_in=tokens[:, :-1])

            # Get predictions
            logits = outputs['gcode_logits']  # [B, T-1, V]
            targets = tokens[:, 1:]  # [B, T-1]

            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                ignore_index=0,
                reduction='mean'
            )
            all_losses.append(loss.item())

            # Get predictions
            preds = logits.argmax(dim=-1)  # [B, T-1]

            # Store
            all_predictions.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all
    all_predictions = torch.cat(all_predictions, dim=0)  # [N, T]
    all_targets = torch.cat(all_targets, dim=0)  # [N, T]
    avg_loss = np.mean(all_losses)

    return all_predictions, all_targets, avg_loss

def compute_detailed_metrics(predictions, targets, vocab_path: Path):
    """Compute detailed metrics by token type."""
    print("\n" + "=" * 80)
    print("COMPUTING DETAILED METRICS")
    print("=" * 80)

    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']
    id_to_token = {v: k for k, v in vocab.items()}

    # Define token categories
    g_command_ids = [vocab.get(f'G{i}', -1) for i in range(100)]
    g_command_ids = [x for x in g_command_ids if x != -1]

    m_command_ids = [vocab.get(f'M{i}', -1) for i in range(100)]
    m_command_ids = [x for x in m_command_ids if x != -1]

    # Flatten predictions and targets
    preds_flat = predictions.reshape(-1).numpy()
    targets_flat = targets.reshape(-1).numpy()

    # Create masks
    valid_mask = targets_flat != 0  # Not padding
    g_mask = np.isin(targets_flat, g_command_ids) & valid_mask
    m_mask = np.isin(targets_flat, m_command_ids) & valid_mask
    num_mask = valid_mask & ~g_mask & ~m_mask

    # Compute accuracies
    metrics = {}
    metrics['overall_acc'] = (preds_flat[valid_mask] == targets_flat[valid_mask]).mean()
    metrics['g_command_acc'] = (preds_flat[g_mask] == targets_flat[g_mask]).mean() if g_mask.any() else 0
    metrics['m_command_acc'] = (preds_flat[m_mask] == targets_flat[m_mask]).mean() if m_mask.any() else 0
    metrics['numeric_acc'] = (preds_flat[num_mask] == targets_flat[num_mask]).mean() if num_mask.any() else 0

    # Compute per-token-type counts
    metrics['total_tokens'] = valid_mask.sum()
    metrics['g_command_count'] = g_mask.sum()
    metrics['m_command_count'] = m_mask.sum()
    metrics['numeric_count'] = num_mask.sum()

    # Sequence-level exact match
    seq_matches = (predictions == targets).all(dim=1)
    metrics['sequence_exact_match'] = seq_matches.float().mean().item()

    return metrics, id_to_token

def plot_confusion_matrix(predictions, targets, id_to_token, output_dir: Path, top_n=30):
    """Plot confusion matrix for top N tokens."""
    print(f"\nGenerating confusion matrix (top {top_n} tokens)...")

    # Flatten
    preds_flat = predictions.reshape(-1).numpy()
    targets_flat = targets.reshape(-1).numpy()

    # Remove padding
    valid_mask = targets_flat != 0
    preds_flat = preds_flat[valid_mask]
    targets_flat = targets_flat[valid_mask]

    # Get top N most frequent tokens
    unique, counts = np.unique(targets_flat, return_counts=True)
    top_indices = np.argsort(counts)[-top_n:]
    top_tokens = unique[top_indices]

    # Filter to only top tokens
    mask = np.isin(targets_flat, top_tokens)
    preds_top = preds_flat[mask]
    targets_top = targets_flat[mask]

    # Compute confusion matrix
    cm = confusion_matrix(targets_top, preds_top, labels=top_tokens)

    # Create labels
    labels = [id_to_token.get(int(i), f'ID_{i}') for i in top_tokens]

    # Plot
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Confusion Matrix - Top {top_n} Tokens (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'test_confusion_matrix_top{top_n}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_path}")

def create_test_report(
    metrics: Dict,
    avg_loss: float,
    checkpoint_path: Path,
    output_dir: Path,
):
    """Create comprehensive test results report."""
    print("\nCreating test results report...")

    report = f"""# Test Set Evaluation Results

**Model:** `{checkpoint_path.parent.name}`
**Checkpoint:** `{checkpoint_path.name}`
**Evaluation Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Overall Performance

| Metric | Value |
|--------|-------|
| **Test Loss** | {avg_loss:.4f} |
| **Overall Accuracy** | {metrics['overall_acc']*100:.2f}% |
| **Sequence Exact Match** | {metrics['sequence_exact_match']*100:.2f}% |

---

## Per-Token-Type Performance

| Token Type | Accuracy | Count | Percentage |
|-----------|----------|-------|------------|
| **G-Commands** | {metrics['g_command_acc']*100:.2f}% | {metrics['g_command_count']:,} | {metrics['g_command_count']/metrics['total_tokens']*100:.2f}% |
| **M-Commands** | {metrics['m_command_acc']*100:.2f}% | {metrics['m_command_count']:,} | {metrics['m_command_count']/metrics['total_tokens']*100:.2f}% |
| **Numeric/Other** | {metrics['numeric_acc']*100:.2f}% | {metrics['numeric_count']:,} | {metrics['numeric_count']/metrics['total_tokens']*100:.2f}% |
| **Total** | {metrics['overall_acc']*100:.2f}% | {metrics['total_tokens']:,} | 100.00% |

---

## Key Findings

### ✅ Strengths
- G-Command prediction accuracy: **{metrics['g_command_acc']*100:.1f}%**
- Overall token prediction: **{metrics['overall_acc']*100:.1f}%**
- Sequence exact match: **{metrics['sequence_exact_match']*100:.1f}%**

### Analysis

**Class Imbalance Handling:**
- G-commands represent only {metrics['g_command_count']/metrics['total_tokens']*100:.2f}% of tokens
- Achieved {metrics['g_command_acc']*100:.1f}% accuracy despite rarity
- Class weighting strategy **{'successful' if metrics['g_command_acc'] > 0.95 else 'needs improvement'}**

**Generalization:**
- Test loss: {avg_loss:.4f}
- Compare to validation loss to assess overfitting
- Exact sequence match: {metrics['sequence_exact_match']*100:.1f}%

---

## Comparison: Validation vs Test

_Note: Add validation metrics from sweep results for comparison_

| Metric | Validation | Test | Delta |
|--------|-----------|------|-------|
| G-Command Acc | TODO% | {metrics['g_command_acc']*100:.2f}% | TODO |
| Overall Acc | TODO% | {metrics['overall_acc']*100:.2f}% | TODO |
| Loss | TODO | {avg_loss:.4f} | TODO |

---

## Visualizations

- Confusion Matrix: `test_confusion_matrix_top30.png`
- Per-Token-Type Breakdown: See metrics table above

---

## Recommendations

### If Test Performance ≈ Validation Performance
✅ Model generalizes well - ready for deployment consideration

### If Test Performance << Validation Performance
⚠️ Overfitting detected - consider:
- Adding dropout/regularization
- Reducing model capacity
- Data augmentation
- More training data

### Next Steps
1. Compare with baseline model (pre-Phase 1)
2. Perform error analysis on failure cases
3. Test on additional real-world data
4. Optimize for inference speed

---

**Report Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    output_path = output_dir / 'TEST_RESULTS.md'
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"   Saved: {output_path}")
    return output_path

def main():
    """Main evaluation function."""
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=Path, default=Path('outputs/processed'),
                        help='Path to processed data')
    parser.add_argument('--output-dir', type=Path, default=Path('reports'),
                        help='Output directory for reports')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)

    # Load model
    model, config, device = load_model_and_config(args.checkpoint, args.device)

    # Load test dataset
    print(f"\nLoading test dataset from: {args.data_dir}")
    test_dataset = GCodeDataset(args.data_dir / 'test_sequences.npz')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    print(f"✅ Loaded {len(test_dataset)} test sequences")

    # Evaluate
    predictions, targets, avg_loss = evaluate_on_test_set(
        model, test_loader, device, config.gcode_vocab
    )

    # Compute metrics
    # Vocabulary is in data/ directory, not in processed data directory
    vocab_path = Path('data/vocabulary.json')
    metrics, id_to_token = compute_detailed_metrics(predictions, targets, vocab_path)

    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"\nAccuracy Breakdown:")
    print(f"  Overall:     {metrics['overall_acc']*100:.2f}%")
    print(f"  G-Commands:  {metrics['g_command_acc']*100:.2f}% ({metrics['g_command_count']} tokens)")
    print(f"  M-Commands:  {metrics['m_command_acc']*100:.2f}% ({metrics['m_command_count']} tokens)")
    print(f"  Numeric:     {metrics['numeric_acc']*100:.2f}% ({metrics['numeric_count']} tokens)")
    print(f"\nSequence Exact Match: {metrics['sequence_exact_match']*100:.2f}%")

    # Generate visualizations
    plot_confusion_matrix(predictions, targets, id_to_token, args.output_dir)

    # Create report
    report_path = create_test_report(metrics, avg_loss, args.checkpoint, args.output_dir)

    print("\n" + "=" * 80)
    print("✅ TEST EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\nReport saved to: {report_path}")

if __name__ == '__main__':
    main()
