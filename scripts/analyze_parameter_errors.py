#!/usr/bin/env python3
"""
Parameter Error Analysis Script

Analyzes parameter prediction errors (param_type and param_value) from the best sweep checkpoint.
Generates confusion matrices, error patterns, and recommendations for parameter weight adjustments.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from miracle.model.multihead_lm import MultiHeadGCodeLM
from miracle.dataset.dataset import GCodeDataset


def load_vocabulary(vocab_path: Path) -> Dict:
    """Load vocabulary mapping."""
    with open(vocab_path) as f:
        return json.load(f)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[MultiHeadGCodeLM, Dict]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model architecture info from checkpoint
    model_state = checkpoint['model_state_dict']

    # Infer dimensions from saved weights
    embedding_weight = model_state['embedding.weight']
    vocab_size, hidden_dim = embedding_weight.shape

    # Get other dimensions from transformer layers
    num_heads = 4  # From sweep config
    num_layers = 2  # From sweep config

    # Get number of classes from output heads
    type_head_weight = model_state['type_head.weight']
    command_head_weight = model_state['command_head.weight']
    param_type_head_weight = model_state['param_type_head.weight']
    param_value_head_weight = model_state['param_value_head.weight']
    operation_head_weight = model_state['operation_head.weight']

    num_types = type_head_weight.shape[0]
    num_commands = command_head_weight.shape[0]
    num_param_types = param_type_head_weight.shape[0]
    num_param_values = param_value_head_weight.shape[0]
    num_operations = operation_head_weight.shape[0]

    # Create model
    model = MultiHeadGCodeLM(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_types=num_types,
        num_commands=num_commands,
        num_param_types=num_param_types,
        num_param_values=num_param_values,
        num_operations=num_operations,
        dropout=0.1
    )

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return model, checkpoint


def analyze_predictions(
    model: MultiHeadGCodeLM,
    dataloader: DataLoader,
    device: torch.device,
    vocab: Dict
) -> Dict:
    """Run inference and collect predictions vs ground truth."""

    results = {
        'type_pred': [],
        'type_true': [],
        'command_pred': [],
        'command_true': [],
        'param_type_pred': [],
        'param_type_true': [],
        'param_value_pred': [],
        'param_value_true': [],
        'operation_pred': [],
        'operation_true': [],
        'token_ids': [],
        'sequence_ids': []
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            type_labels = batch['type_labels'].to(device)
            command_labels = batch['command_labels'].to(device)
            param_type_labels = batch['param_type_labels'].to(device)
            param_value_labels = batch['param_value_labels'].to(device)
            operation_labels = batch['operation_labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)

            # Get predictions
            type_pred = outputs['type_logits'].argmax(dim=-1)
            command_pred = outputs['command_logits'].argmax(dim=-1)
            param_type_pred = outputs['param_type_logits'].argmax(dim=-1)
            param_value_pred = outputs['param_value_logits'].argmax(dim=-1)
            operation_pred = outputs['operation_logits'].argmax(dim=-1)

            # Collect results (only where attention_mask is 1)
            mask = attention_mask.bool().cpu()

            for i in range(input_ids.size(0)):
                seq_mask = mask[i]
                results['type_pred'].extend(type_pred[i][seq_mask].cpu().numpy())
                results['type_true'].extend(type_labels[i][seq_mask].cpu().numpy())
                results['command_pred'].extend(command_pred[i][seq_mask].cpu().numpy())
                results['command_true'].extend(command_labels[i][seq_mask].cpu().numpy())
                results['param_type_pred'].extend(param_type_pred[i][seq_mask].cpu().numpy())
                results['param_type_true'].extend(param_type_labels[i][seq_mask].cpu().numpy())
                results['param_value_pred'].extend(param_value_pred[i][seq_mask].cpu().numpy())
                results['param_value_true'].extend(param_value_labels[i][seq_mask].cpu().numpy())
                results['token_ids'].extend(input_ids[i][seq_mask].cpu().numpy())
                results['sequence_ids'].extend([batch_idx * dataloader.batch_size + i] * seq_mask.sum().item())

            # Operation predictions (one per sequence)
            results['operation_pred'].extend(operation_pred.cpu().numpy())
            results['operation_true'].extend(operation_labels.cpu().numpy())

    # Convert to numpy arrays
    for key in results:
        if key not in ['sequence_ids']:
            results[key] = np.array(results[key])

    return results


def create_confusion_matrix(y_true, y_pred, labels, title, output_path, figsize=(10, 8)):
    """Create and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

    fig, ax = plt.subplots(figsize=figsize)

    # Use percentage format
    sns.heatmap(
        cm_normalized * 100,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Percentage (%)'},
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return cm, cm_normalized


def analyze_error_patterns(results: Dict, vocab: Dict, output_dir: Path):
    """Analyze and report detailed error patterns."""

    # Create reverse mappings
    token_to_info = {}
    for token, info in vocab['tokens'].items():
        token_id = info['id']
        token_to_info[token_id] = {
            'token': token,
            'type': info.get('type', 'unknown'),
            'command': info.get('command', 'unknown'),
            'param_type': info.get('param_type', 'unknown'),
            'param_value': info.get('param_value', 'unknown')
        }

    # Get unique labels
    param_type_labels = sorted(set(results['param_type_true']))
    param_value_labels = sorted(set(results['param_value_true']))

    # Create label to name mappings
    id_to_param_type = vocab.get('param_types', {})
    id_to_param_value = vocab.get('param_values', {})

    # Reverse mappings
    param_type_names = {idx: name for name, idx in id_to_param_type.items()}
    param_value_names = {idx: str(val) for val, idx in id_to_param_value.items()}

    print("\n" + "="*80)
    print("PARAMETER ERROR ANALYSIS")
    print("="*80)

    # 1. Overall accuracy
    param_type_acc = (results['param_type_pred'] == results['param_type_true']).mean() * 100
    param_value_acc = (results['param_value_pred'] == results['param_value_true']).mean() * 100

    print(f"\n📊 Overall Accuracy:")
    print(f"  Param Type:  {param_type_acc:.2f}%")
    print(f"  Param Value: {param_value_acc:.2f}%")

    # 2. Param Type Confusion Matrix
    print(f"\n🔍 Param Type Confusion Analysis:")
    cm_param_type, cm_param_type_norm = create_confusion_matrix(
        results['param_type_true'],
        results['param_type_pred'],
        [param_type_names.get(i, f"Type{i}") for i in param_type_labels],
        "Parameter Type Confusion Matrix",
        output_dir / "param_type_confusion.png",
        figsize=(8, 6)
    )

    # Find most confused param types
    errors_param_type = defaultdict(int)
    for true_idx, pred_idx in zip(results['param_type_true'], results['param_type_pred']):
        if true_idx != pred_idx:
            true_name = param_type_names.get(true_idx, f"Type{true_idx}")
            pred_name = param_type_names.get(pred_idx, f"Type{pred_idx}")
            errors_param_type[(true_name, pred_name)] += 1

    print(f"\n  Top 5 Param Type Confusions:")
    for (true_name, pred_name), count in sorted(errors_param_type.items(), key=lambda x: -x[1])[:5]:
        percentage = count / len(results['param_type_true']) * 100
        print(f"    {true_name} → {pred_name}: {count} ({percentage:.2f}%)")

    # 3. Param Value Confusion Matrix (top confusions only for readability)
    print(f"\n🔍 Param Value Confusion Analysis:")

    # Find most confused param values
    errors_param_value = defaultdict(int)
    for true_idx, pred_idx in zip(results['param_value_true'], results['param_value_pred']):
        if true_idx != pred_idx:
            true_name = param_value_names.get(true_idx, f"Value{true_idx}")
            pred_name = param_value_names.get(pred_idx, f"Value{pred_idx}")
            errors_param_value[(true_name, pred_name)] += 1

    print(f"\n  Top 10 Param Value Confusions:")
    for (true_name, pred_name), count in sorted(errors_param_value.items(), key=lambda x: -x[1])[:10]:
        percentage = count / len(results['param_value_true']) * 100
        print(f"    {true_name} → {pred_name}: {count} ({percentage:.2f}%)")

    # Create simplified confusion matrix for top confused values
    top_confused_values = set()
    for (true_name, pred_name), count in sorted(errors_param_value.items(), key=lambda x: -x[1])[:20]:
        # Get indices
        true_idx = [k for k, v in param_value_names.items() if v == true_name][0]
        pred_idx = [k for k, v in param_value_names.items() if v == pred_name][0]
        top_confused_values.add(true_idx)
        top_confused_values.add(pred_idx)

    if len(top_confused_values) > 0:
        top_confused_values = sorted(list(top_confused_values))

        # Filter results to only include top confused values
        mask = np.isin(results['param_value_true'], top_confused_values) | np.isin(results['param_value_pred'], top_confused_values)
        filtered_true = results['param_value_true'][mask]
        filtered_pred = results['param_value_pred'][mask]

        create_confusion_matrix(
            filtered_true,
            filtered_pred,
            [param_value_names.get(i, f"Value{i}") for i in top_confused_values],
            "Parameter Value Confusion Matrix (Top Confused Values)",
            output_dir / "param_value_confusion_top.png",
            figsize=(12, 10)
        )

    # 4. Error correlation with command type
    print(f"\n🔗 Error Correlation with Command Type:")

    command_names = vocab.get('commands', {})
    command_names = {idx: name for name, idx in command_names.items()}

    param_type_errors_by_command = defaultdict(int)
    param_type_total_by_command = defaultdict(int)
    param_value_errors_by_command = defaultdict(int)
    param_value_total_by_command = defaultdict(int)

    for i, cmd_true in enumerate(results['command_true']):
        cmd_name = command_names.get(cmd_true, f"Command{cmd_true}")

        param_type_total_by_command[cmd_name] += 1
        param_value_total_by_command[cmd_name] += 1

        if results['param_type_pred'][i] != results['param_type_true'][i]:
            param_type_errors_by_command[cmd_name] += 1

        if results['param_value_pred'][i] != results['param_value_true'][i]:
            param_value_errors_by_command[cmd_name] += 1

    print(f"\n  Param Type Error Rate by Command:")
    for cmd_name in sorted(param_type_total_by_command.keys()):
        errors = param_type_errors_by_command[cmd_name]
        total = param_type_total_by_command[cmd_name]
        error_rate = errors / total * 100 if total > 0 else 0
        print(f"    {cmd_name}: {error_rate:.2f}% ({errors}/{total})")

    print(f"\n  Param Value Error Rate by Command:")
    for cmd_name in sorted(param_value_total_by_command.keys()):
        errors = param_value_errors_by_command[cmd_name]
        total = param_value_total_by_command[cmd_name]
        error_rate = errors / total * 100 if total > 0 else 0
        print(f"    {cmd_name}: {error_rate:.2f}% ({errors}/{total})")

    # 5. Analyze numeric proximity (for param values)
    print(f"\n📏 Numeric Proximity Analysis (Param Values):")

    # Try to parse param values as floats
    numeric_errors = []
    for true_idx, pred_idx in zip(results['param_value_true'], results['param_value_pred']):
        if true_idx != pred_idx:
            true_name = param_value_names.get(true_idx, None)
            pred_name = param_value_names.get(pred_idx, None)
            if true_name and pred_name:
                try:
                    true_val = float(true_name)
                    pred_val = float(pred_name)
                    diff = abs(true_val - pred_val)
                    numeric_errors.append(diff)
                except ValueError:
                    pass

    if len(numeric_errors) > 0:
        numeric_errors = np.array(numeric_errors)
        print(f"  Total numeric errors: {len(numeric_errors)}")
        print(f"  Mean absolute difference: {numeric_errors.mean():.4f}")
        print(f"  Median absolute difference: {np.median(numeric_errors):.4f}")
        print(f"  Max absolute difference: {numeric_errors.max():.4f}")

        # Distribution of errors
        close_errors = (numeric_errors < 1.0).sum()
        medium_errors = ((numeric_errors >= 1.0) & (numeric_errors < 10.0)).sum()
        far_errors = (numeric_errors >= 10.0).sum()

        print(f"  Error distribution:")
        print(f"    Close (< 1.0): {close_errors} ({close_errors/len(numeric_errors)*100:.1f}%)")
        print(f"    Medium (1.0-10.0): {medium_errors} ({medium_errors/len(numeric_errors)*100:.1f}%)")
        print(f"    Far (> 10.0): {far_errors} ({far_errors/len(numeric_errors)*100:.1f}%)")

    # 6. Generate recommendations
    print(f"\n💡 Recommendations:")

    recommendations = []

    # Based on param type accuracy
    if param_type_acc < 95.0:
        recommendations.append(
            f"1. Increase param_type_weight: Current accuracy {param_type_acc:.2f}% is below 95% target. "
            f"Try sweeping param_type_weight in [5.0, 10.0, 15.0]"
        )

    # Based on param value accuracy
    if param_value_acc < 98.0:
        recommendations.append(
            f"2. Increase param_value_weight: Current accuracy {param_value_acc:.2f}% is below 98% target. "
            f"Try sweeping param_value_weight in [5.0, 10.0, 15.0]"
        )

    # Based on numeric proximity
    if len(numeric_errors) > 0 and far_errors > 0:
        recommendations.append(
            f"3. Consider Focal Loss for param values: {far_errors} errors have large absolute differences (>10.0). "
            f"Focal loss can help focus on hard examples."
        )

    # Based on command-specific errors
    high_error_commands = []
    for cmd_name in param_value_total_by_command.keys():
        errors = param_value_errors_by_command[cmd_name]
        total = param_value_total_by_command[cmd_name]
        error_rate = errors / total * 100 if total > 0 else 0
        if error_rate > 5.0:
            high_error_commands.append((cmd_name, error_rate))

    if len(high_error_commands) > 0:
        cmd_list = ", ".join([f"{cmd} ({rate:.1f}%)" for cmd, rate in high_error_commands])
        recommendations.append(
            f"4. Command-specific analysis needed: High param value error rates for: {cmd_list}. "
            f"Consider command-conditioned parameter prediction."
        )

    if len(recommendations) == 0:
        recommendations.append("No major issues detected! Current configuration is performing well.")

    for rec in recommendations:
        print(f"  {rec}")

    # Save recommendations to file
    with open(output_dir / "recommendations.txt", "w") as f:
        f.write("PARAMETER OPTIMIZATION RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Current Performance:\n")
        f.write(f"  Param Type Accuracy:  {param_type_acc:.2f}%\n")
        f.write(f"  Param Value Accuracy: {param_value_acc:.2f}%\n\n")
        f.write("Recommendations:\n\n")
        for rec in recommendations:
            f.write(f"{rec}\n\n")

    print(f"\n✅ Analysis complete! Visualizations and recommendations saved to {output_dir}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze parameter prediction errors")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/sweep_fast_validation/checkpoint_best.pt",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="outputs/processed_with_ops",
        help="Path to processed data directory"
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="data/vocabulary.json",
        help="Path to vocabulary file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/parameter_analysis",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )

    args = parser.parse_args()

    # Convert to Path objects
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    vocab_path = Path(args.vocab_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load vocabulary
    print(f"\nLoading vocabulary from {vocab_path}...")
    vocab = load_vocabulary(vocab_path)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load validation dataset
    print(f"\nLoading validation data from {data_dir}...")
    val_data = np.load(data_dir / "val_sequences.npz")

    val_dataset = GCodeDataset(
        input_ids=val_data['input_ids'],
        type_labels=val_data['type_labels'],
        command_labels=val_data['command_labels'],
        param_type_labels=val_data['param_type_labels'],
        param_value_labels=val_data['param_value_labels'],
        operation_labels=val_data['operation_labels'],
        attention_mask=val_data['attention_mask']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Validation set: {len(val_dataset)} sequences")

    # Analyze predictions
    print(f"\nRunning inference on validation set...")
    results = analyze_predictions(model, val_loader, device, vocab)

    # Analyze error patterns
    analyze_error_patterns(results, vocab, output_dir)


if __name__ == "__main__":
    main()
