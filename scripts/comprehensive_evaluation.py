#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script

Generates ALL confusion matrices, per-class metrics, and detailed reports.
Automatically detects vocabulary size from checkpoint.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/comprehensive_evaluation.py \
        --checkpoint outputs/sweep_27v7pl9i_best/checkpoint_best.pt \
        --test-data outputs/processed_hybrid/test_sequences.npz \
        --output reports/ugkjmojf_comprehensive

Output:
    - confusion_matrices/ (type, command, operation, param_type, param_value, gcode_token)
    - per_class_metrics/ (precision, recall, F1 for each class)
    - summary_report.txt
    - accuracy_breakdown.json
"""

import argparse
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy import stats
import base64
from io import BytesIO
from collections import defaultdict, Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.multihead_lm import MultiHeadGCodeLM
from miracle.dataset.target_utils import TokenDecomposer
from miracle.utilities.device import get_device
from miracle.dataset.dataset import GCodeDataset, collate_fn
from miracle.training.metrics import levenshtein_distance
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_number(num: float, max_decimals: int = 4) -> float:
    """Format number to max decimal places, removing trailing zeros."""
    if isinstance(num, (int, np.integer)):
        return int(num)
    return round(float(num), max_decimals)


def embed_image_as_base64(image_path: Path) -> str:
    """Convert image file to base64 string for embedding in HTML."""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')


def get_color_for_accuracy(acc: float) -> str:
    """Return color based on accuracy level."""
    if acc >= 90:
        return '#2ecc71'  # Green
    elif acc >= 70:
        return '#f39c12'  # Orange
    elif acc >= 50:
        return '#e67e22'  # Dark orange
    else:
        return '#e74c3c'  # Red


# ============================================================================
# MAIN EVALUATION FUNCTIONS
# ============================================================================

def auto_detect_vocab(checkpoint_path: Path) -> Tuple[int, str]:
    """
    Auto-detect vocabulary size and path from checkpoint.

    Returns:
        (vocab_size, suggested_vocab_path)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Check multihead state dict
    if 'multihead_state_dict' in checkpoint:
        if 'embed.weight' in checkpoint['multihead_state_dict']:
            vocab_size = checkpoint['multihead_state_dict']['embed.weight'].shape[0]
        else:
            raise ValueError("Could not find embed.weight in checkpoint")
    else:
        raise ValueError("No multihead_state_dict in checkpoint")

    # Suggest vocabulary path based on size
    vocab_path_map = {
        69: "data/vocabulary_1digit_hybrid.json",
        170: "data/vocabulary.json",
        # Add more as needed
    }

    suggested_path = vocab_path_map.get(vocab_size, f"data/vocabulary_{vocab_size}tokens.json")

    return vocab_size, suggested_path


def load_checkpoint(checkpoint_path: Path, vocab_path: Path, device):
    """Load checkpoint and create models with correct vocab size."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    # Extract actual vocab dict (handle nested structure)
    if isinstance(vocab_data, dict) and 'vocab' in vocab_data:
        vocab = vocab_data['vocab']
    else:
        vocab = vocab_data

    # Infer vocab size from checkpoint
    if 'multihead_state_dict' in checkpoint and 'embed.weight' in checkpoint['multihead_state_dict']:
        checkpoint_vocab_size = checkpoint['multihead_state_dict']['embed.weight'].shape[0]
        print(f"  Checkpoint vocab size: {checkpoint_vocab_size}")
        print(f"  Vocabulary file size: {len(vocab)}")

        if checkpoint_vocab_size != len(vocab):
            raise ValueError(
                f"Vocabulary mismatch: checkpoint has {checkpoint_vocab_size} tokens, "
                f"vocab file has {len(vocab)} tokens"
            )
    else:
        checkpoint_vocab_size = len(vocab)

    vocab_size = checkpoint_vocab_size

    # Create decomposer (pass vocab path, not the loaded dict)
    decomposer = TokenDecomposer(str(vocab_path))

    # Get config from checkpoint
    config = checkpoint.get('config', {})

    # Infer sensor dimensions from checkpoint
    backbone_state = checkpoint['backbone_state_dict']
    if 'encoders.0.proj.0.weight' in backbone_state:
        n_continuous = backbone_state['encoders.0.proj.0.weight'].shape[1]
    else:
        n_continuous = 219  # Default for hybrid

    n_categorical = 4  # Standard

    # Get architecture params
    hidden_dim = config.get('hidden_dim', 256)
    num_heads = config.get('num_heads', 8)
    num_layers = config.get('num_layers', 5)
    dropout = config.get('dropout', 0.2)

    print(f"\nModel Configuration:")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_layers: {num_layers}")
    print(f"  dropout: {dropout}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  n_continuous: {n_continuous}")
    print(f"  n_categorical: {n_categorical}")

    # Create backbone
    model_config = ModelConfig(
        sensor_dims=[n_continuous, n_categorical],
        d_model=hidden_dim,
        lstm_layers=num_layers,
        gcode_vocab=vocab_size,
        n_heads=num_heads,
        dropout=dropout,
    )

    backbone = MM_DTAE_LSTM(model_config).to(device)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    backbone.eval()

    # Create multihead LM
    # Note: n_types (4) is built into the model, not a parameter
    multihead_lm = MultiHeadGCodeLM(
        d_model=hidden_dim,
        vocab_size=vocab_size,
        n_commands=decomposer.n_commands,
        n_param_types=decomposer.n_param_types,
        n_param_values=decomposer.n_param_values,
        nhead=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    multihead_lm.load_state_dict(checkpoint['multihead_state_dict'])
    multihead_lm.eval()

    return {
        'backbone': backbone,
        'multihead_lm': multihead_lm,
        'decomposer': decomposer,
        'device': device,
        'config': config,
    }


def evaluate_comprehensive(model_dict: dict, test_data_path: Path, batch_size: int = 8) -> dict:
    """Run comprehensive evaluation and collect all predictions."""
    print(f"\nEvaluating on: {test_data_path}")

    backbone = model_dict['backbone']
    multihead_lm = model_dict['multihead_lm']
    decomposer = model_dict['decomposer']
    device = model_dict['device']

    # Load test data
    test_dataset = GCodeDataset(str(test_data_path))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Storage for ALL predictions and targets
    all_preds = {
        'type': [],
        'command': [],
        'param_type': [],
        'param_value': [],
        'gcode_token': [],  # Reconstructed full token
    }

    all_targets = {
        'type': [],
        'command': [],
        'param_type': [],
        'param_value': [],
        'gcode_token': [],
    }

    # Check if model has operation head
    has_operation = hasattr(multihead_lm, 'operation_head')
    if has_operation:
        all_preds['operation'] = []
        all_targets['operation'] = []

    # Per-sample accuracies
    per_sample_acc = {
        'type': [],
        'command': [],
        'param_type': [],
        'param_value': [],
        'overall': [],
    }

    # Additional data collection for visualizations
    batch_losses = []  # For loss curves
    all_embeddings = []  # For t-SNE
    all_embedding_labels = []  # Labels for t-SNE coloring (token types)
    all_embedding_operation_labels = []  # Labels for t-SNE coloring (operation types)
    sample_predictions = []  # Store sample predictions with metadata

    # Per-operation type metrics
    operation_accuracies = {
        'type': {},
        'command': {},
        'param_type': {},
        'param_value': {},
        'overall': {}
    }

    # ============================================================================
    # ADVANCED ANALYSIS: New data structures
    # ============================================================================

    # Category 1: Error Analysis
    error_cases = []  # Detailed error information
    positional_accuracy_data = []  # Accuracy by token position

    # Category 2: Attention Analysis
    attention_samples = []  # Attention weights for subset of samples (limit 100)

    # Category 4: Sequence Analysis
    sequence_metrics = []  # Length, edit distance, partial accuracy

    # Category 6: Operation-Specific Analysis
    operation_predictions = {i: defaultdict(list) for i in range(9)}  # Per-operation predictions
    operation_targets = {i: defaultdict(list) for i in range(9)}  # Per-operation targets

    # Loss criterion
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    # Evaluation loop
    print("\nRunning evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            continuous = batch['continuous'].to(device)
            categorical = batch['categorical'].to(device).float()
            tokens = batch['tokens'].to(device)

            # Get operation type for per-operation metrics
            operation_ids = batch.get('operation_type', torch.zeros(tokens.size(0), dtype=torch.long)).cpu().numpy()

            # Clip tokens to valid vocabulary range
            vocab_size = multihead_lm.embed.num_embeddings
            tokens = torch.clamp(tokens, min=0, max=vocab_size-1)

            # Sequence lengths
            batch_size_current = continuous.size(0)
            seq_len = continuous.size(1)
            lengths = torch.full((batch_size_current,), seq_len, dtype=torch.long, device=device)

            # Prepare targets (teacher forcing) - FIXED to match training code
            tgt_in = tokens[:, :-1]
            tgt_out = tokens[:, 1:]

            # Forward backbone - FIXED to include gcode_in parameter
            mods = [continuous, categorical]
            backbone_out = backbone(mods=mods, lengths=lengths, gcode_in=tgt_in)
            memory = backbone_out['memory']

            # Decompose targets
            tgt_decomposed = decomposer.decompose_batch(tgt_out)

            # Forward multihead LM
            logits = multihead_lm(memory, tgt_in)

            # Get predictions for each head - FIXED to remove incorrect [:, :-1] slicing
            type_preds = torch.argmax(logits['type_logits'], dim=-1)
            command_preds = torch.argmax(logits['command_logits'], dim=-1)
            param_type_preds = torch.argmax(logits['param_type_logits'], dim=-1)

            # Handle regression vs classification for param_value
            if 'param_value_regression' in logits:
                param_value_preds = torch.round(logits['param_value_regression'].squeeze(-1)).long()
                param_value_preds = torch.clamp(param_value_preds, 0, decomposer.n_param_values - 1)
            else:
                param_value_preds = torch.argmax(logits['param_value_logits'], dim=-1)

            # Reconstruct full gcode tokens from predictions
            gcode_preds = decomposer.compose_batch({
                'type': type_preds,
                'command_id': command_preds,
                'param_type_id': param_type_preds,
                'param_value_id': param_value_preds
            })

            # Store predictions
            all_preds['type'].append(type_preds.cpu())
            all_preds['command'].append(command_preds.cpu())
            all_preds['param_type'].append(param_type_preds.cpu())
            all_preds['param_value'].append(param_value_preds.cpu())
            all_preds['gcode_token'].append(gcode_preds.cpu())

            # Store targets
            all_targets['type'].append(tgt_decomposed['type'].cpu())
            all_targets['command'].append(tgt_decomposed['command_id'].cpu())
            all_targets['param_type'].append(tgt_decomposed['param_type_id'].cpu())
            all_targets['param_value'].append(tgt_decomposed['param_value_id'].cpu())
            all_targets['gcode_token'].append(tgt_out.cpu())

            # Compute losses for loss curves
            type_loss = ce_loss(
                logits['type_logits'].reshape(-1, logits['type_logits'].size(-1)),
                tgt_decomposed['type'].reshape(-1)
            ).mean().item()

            command_loss = ce_loss(
                logits['command_logits'].reshape(-1, logits['command_logits'].size(-1)),
                tgt_decomposed['command_id'].reshape(-1)
            ).mean().item()

            param_type_loss = ce_loss(
                logits['param_type_logits'].reshape(-1, logits['param_type_logits'].size(-1)),
                tgt_decomposed['param_type_id'].reshape(-1)
            ).mean().item()

            if 'param_value_regression' in logits:
                param_value_loss = torch.nn.functional.mse_loss(
                    logits['param_value_regression'].squeeze(-1),
                    tgt_decomposed['param_value_id'].float()
                ).item()
            else:
                param_value_loss = ce_loss(
                    logits['param_value_logits'].reshape(-1, logits['param_value_logits'].size(-1)),
                    tgt_decomposed['param_value_id'].reshape(-1)
                ).mean().item()

            total_loss = type_loss + command_loss + param_type_loss + param_value_loss

            batch_losses.append({
                'type': type_loss,
                'command': command_loss,
                'param_type': param_type_loss,
                'param_value': param_value_loss,
                'total': total_loss
            })

            # Store embeddings for t-SNE (subsample to avoid memory issues)
            if len(all_embeddings) < 5000:  # Limit to 5000 samples
                all_embeddings.append(memory.mean(dim=1).cpu())  # [B, hidden_dim]
                all_embedding_labels.extend(tgt_decomposed['type'][:, 0].cpu().tolist())
                all_embedding_operation_labels.extend(operation_ids.tolist())

            # Compute per-sample accuracies and collect sample predictions
            B = type_preds.size(0)
            for b in range(B):
                type_acc = (type_preds[b] == tgt_decomposed['type'][b]).float().mean().item()
                command_acc = (command_preds[b] == tgt_decomposed['command_id'][b]).float().mean().item()
                param_type_acc = (param_type_preds[b] == tgt_decomposed['param_type_id'][b]).float().mean().item()
                param_value_acc = (param_value_preds[b] == tgt_decomposed['param_value_id'][b]).float().mean().item()

                overall_correct = (
                    (type_preds[b] == tgt_decomposed['type'][b]) &
                    (command_preds[b] == tgt_decomposed['command_id'][b]) &
                    (param_type_preds[b] == tgt_decomposed['param_type_id'][b]) &
                    (param_value_preds[b] == tgt_decomposed['param_value_id'][b])
                ).float().mean().item()

                per_sample_acc['type'].append(type_acc)
                per_sample_acc['command'].append(command_acc)
                per_sample_acc['param_type'].append(param_type_acc)
                per_sample_acc['param_value'].append(param_value_acc)
                per_sample_acc['overall'].append(overall_correct)

                # Track accuracies by operation type
                op_id = int(operation_ids[b])
                if op_id not in operation_accuracies['type']:
                    operation_accuracies['type'][op_id] = []
                    operation_accuracies['command'][op_id] = []
                    operation_accuracies['param_type'][op_id] = []
                    operation_accuracies['param_value'][op_id] = []
                    operation_accuracies['overall'][op_id] = []

                operation_accuracies['type'][op_id].append(type_acc)
                operation_accuracies['command'][op_id].append(command_acc)
                operation_accuracies['param_type'][op_id].append(param_type_acc)
                operation_accuracies['param_value'][op_id].append(param_value_acc)
                operation_accuracies['overall'][op_id].append(overall_correct)

                # Store sample predictions (limit to 100 correct and 100 incorrect)
                if len(sample_predictions) < 200:
                    is_correct = overall_correct == 1.0
                    sample_predictions.append({
                        'is_correct': is_correct,
                        'target_tokens': tgt_out[b].cpu().tolist(),
                        'pred_tokens': gcode_preds[b].cpu().tolist(),
                        'type_acc': type_acc,
                        'command_acc': command_acc,
                        'param_type_acc': param_type_acc,
                        'param_value_acc': param_value_acc,
                    })

                # ============================================================================
                # ADVANCED ANALYSIS: Data collection
                # ============================================================================

                # Category 1: Error Analysis - Track error cases with full context
                if overall_correct != 1.0:
                    # Get actual sequence length (non-padding tokens)
                    target_seq = tgt_out[b].cpu().tolist()
                    pred_seq = gcode_preds[b].cpu().tolist()

                    # Compute edit distance
                    edit_dist = levenshtein_distance(pred_seq, target_seq)

                    # Positional correctness (token-by-token)
                    positional_correct = (gcode_preds[b] == tgt_out[b]).cpu().numpy()

                    error_cases.append({
                        'operation_id': op_id,
                        'target_seq': target_seq,
                        'pred_seq': pred_seq,
                        'edit_distance': edit_dist,
                        'positional_correct': positional_correct,
                        'embedding': memory[b].mean(dim=0).cpu().numpy(),
                        'type_correct': type_preds[b].eq(tgt_decomposed['type'][b]).cpu().numpy(),
                        'command_correct': command_preds[b].eq(tgt_decomposed['command_id'][b]).cpu().numpy(),
                        'param_type_correct': param_type_preds[b].eq(tgt_decomposed['param_type_id'][b]).cpu().numpy(),
                    })

                # Category 2: Attention Analysis - Extract attention for limited samples
                if len(attention_samples) < 100:
                    try:
                        attn_data = multihead_lm.extract_attention_weights(
                            memory[b:b+1],
                            tgt_out[b:b+1],
                            average_heads=True
                        )
                        attention_samples.append({
                            'attention': attn_data['attention'],
                            'layer_attentions': attn_data['layer_attentions'],
                            'target_tokens': target_seq,
                            'operation_id': op_id,
                        })
                    except Exception as e:
                        # Skip if attention extraction fails
                        pass

                # Category 4: Sequence Analysis - Track length and edit distance
                target_seq = tgt_out[b].cpu().tolist()
                pred_seq = gcode_preds[b].cpu().tolist()

                # Get actual length (excluding padding if present)
                # Assuming PAD token is 0
                actual_length = len([t for t in target_seq if t != 0])

                # Compute edit distance
                edit_dist = levenshtein_distance(pred_seq, target_seq)

                # Partial correctness (percentage of tokens correct)
                partial_correct = (gcode_preds[b] == tgt_out[b]).float().mean().item()

                sequence_metrics.append({
                    'length': actual_length,
                    'operation_id': op_id,
                    'overall_acc': overall_correct,
                    'partial_correct': partial_correct,
                    'edit_distance': edit_dist,
                    'type_acc': type_acc,
                    'command_acc': command_acc,
                    'param_type_acc': param_type_acc,
                    'param_value_acc': param_value_acc,
                })

                # Track positional accuracy
                positional_correct = (gcode_preds[b] == tgt_out[b]).cpu().numpy()
                for pos, is_correct in enumerate(positional_correct):
                    if pos < actual_length:  # Only count non-padding positions
                        positional_accuracy_data.append({
                            'position': pos,
                            'correct': int(is_correct),
                            'operation_id': op_id,
                        })

                # Category 6: Operation-Specific Analysis - Store per-operation predictions
                operation_predictions[op_id]['type'].append(type_preds[b].cpu())
                operation_predictions[op_id]['command'].append(command_preds[b].cpu())
                operation_predictions[op_id]['param_type'].append(param_type_preds[b].cpu())
                operation_predictions[op_id]['param_value'].append(param_value_preds[b].cpu())

                operation_targets[op_id]['type'].append(tgt_decomposed['type'][b].cpu())
                operation_targets[op_id]['command'].append(tgt_decomposed['command_id'][b].cpu())
                operation_targets[op_id]['param_type'].append(tgt_decomposed['param_type_id'][b].cpu())
                operation_targets[op_id]['param_value'].append(tgt_decomposed['param_value_id'][b].cpu())

    # Concatenate all predictions and targets (skip empty lists)
    for key in all_preds:
        if len(all_preds[key]) > 0:
            all_preds[key] = torch.cat(all_preds[key], dim=0).flatten().numpy()
            all_targets[key] = torch.cat(all_targets[key], dim=0).flatten().numpy()
        else:
            # If empty, remove from dicts (e.g., operation head not used)
            pass

    # Convert to percentages
    accuracies = {
        'type': np.mean(per_sample_acc['type']) * 100,
        'command': np.mean(per_sample_acc['command']) * 100,
        'param_type': np.mean(per_sample_acc['param_type']) * 100,
        'param_value': np.mean(per_sample_acc['param_value']) * 100,
        'overall': np.mean(per_sample_acc['overall']) * 100,
    }

    # Concatenate embeddings for t-SNE
    if len(all_embeddings) > 0:
        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    else:
        all_embeddings = np.array([])

    # Aggregate per-operation accuracies
    operation_acc_summary = {}
    for op_id in operation_accuracies['type'].keys():
        operation_acc_summary[op_id] = {
            'type': np.mean(operation_accuracies['type'][op_id]) * 100,
            'command': np.mean(operation_accuracies['command'][op_id]) * 100,
            'param_type': np.mean(operation_accuracies['param_type'][op_id]) * 100,
            'param_value': np.mean(operation_accuracies['param_value'][op_id]) * 100,
            'overall': np.mean(operation_accuracies['overall'][op_id]) * 100,
            'n_samples': len(operation_accuracies['overall'][op_id])
        }

    print(f"\n✓ Evaluation Complete!")
    print(f"  Type:        {accuracies['type']:.2f}%")
    print(f"  Command:     {accuracies['command']:.2f}%")
    print(f"  Param Type:  {accuracies['param_type']:.2f}%")
    print(f"  Param Value: {accuracies['param_value']:.2f}%")
    print(f"  Overall:     {accuracies['overall']:.2f}%")
    print(f"  Collected {len(batch_losses)} batches for loss curves")
    print(f"  Collected {len(all_embeddings)} samples for t-SNE")
    print(f"  Collected {len(sample_predictions)} sample predictions")
    print(f"  Tracked {len(operation_acc_summary)} operation types")
    print(f"  Collected {len(error_cases)} error cases for analysis")
    print(f"  Collected {len(attention_samples)} attention samples")
    print(f"  Collected {len(sequence_metrics)} sequence-level metrics")
    print(f"  Tracked {len(positional_accuracy_data)} positional accuracy datapoints")

    return {
        'predictions': all_preds,
        'targets': all_targets,
        'accuracies': accuracies,
        'decomposer': decomposer,
        'batch_losses': batch_losses,
        'embeddings': all_embeddings,
        'embedding_labels': all_embedding_labels,
        'embedding_operation_labels': all_embedding_operation_labels,
        'operation_accuracies': operation_acc_summary,
        'sample_predictions': sample_predictions,
        # Advanced analysis data
        'error_cases': error_cases,
        'attention_samples': attention_samples,
        'sequence_metrics': sequence_metrics,
        'positional_accuracy_data': positional_accuracy_data,
        'operation_predictions': operation_predictions,
        'operation_targets': operation_targets,
    }


def generate_confusion_matrix(y_true, y_pred, num_classes, class_names, output_path: Path, title: str):
    """Generate and save confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, num_classes * 0.5), max(8, num_classes * 0.4)))

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True if num_classes <= 20 else False,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names if num_classes <= 20 else False,
        yticklabels=class_names if num_classes <= 20 else False,
        ax=ax,
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")

    return cm


def compute_per_class_metrics(y_true, y_pred, num_classes, class_names) -> dict:
    """Compute precision, recall, F1 for each class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    metrics = {}
    for i, class_name in enumerate(class_names):
        metrics[class_name] = {
            'precision': format_number(precision[i]),
            'recall': format_number(recall[i]),
            'f1': format_number(f1[i]),
            'support': int(support[i]),
        }

    return metrics


def generate_normalized_confusion_matrix(y_true, y_pred, num_classes, class_names, output_path: Path, title: str):
    """Generate and save row-normalized confusion matrix (shows recall per class)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Row normalization (each row sums to 1, shows recall)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=(max(10, num_classes * 0.5), max(8, num_classes * 0.4)))

    sns.heatmap(
        cm_normalized,
        annot=True if num_classes <= 20 else False,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        xticklabels=class_names if num_classes <= 20 else False,
        yticklabels=class_names if num_classes <= 20 else False,
        ax=ax,
        cbar_kws={'label': 'Recall (Row-Normalized)'}
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{title} (Normalized)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def generate_accuracy_bar_chart(accuracies: dict, output_path: Path):
    """Generate bar chart comparing all accuracy metrics."""
    metrics = list(accuracies.keys())
    values = [accuracies[m] for m in metrics]
    colors = [get_color_for_accuracy(v) for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Test Set Performance by Metric', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def generate_loss_curves(batch_losses: List[dict], output_path: Path):
    """Generate loss curves from evaluation batch losses."""
    if not batch_losses:
        print("  No batch losses to plot")
        return

    # Extract loss components
    type_losses = [b['type'] for b in batch_losses]
    command_losses = [b['command'] for b in batch_losses]
    param_type_losses = [b['param_type'] for b in batch_losses]
    param_value_losses = [b['param_value'] for b in batch_losses]
    total_losses = [b['total'] for b in batch_losses]

    batches = list(range(len(batch_losses)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Test Set Loss Curves', fontsize=16, fontweight='bold')

    # Type loss
    axes[0, 0].plot(batches, type_losses, label='Type Loss', color='#3498db', linewidth=2)
    axes[0, 0].set_title('Type Classification Loss')
    axes[0, 0].set_xlabel('Batch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(alpha=0.3)

    # Command loss
    axes[0, 1].plot(batches, command_losses, label='Command Loss', color='#e74c3c', linewidth=2)
    axes[0, 1].set_title('Command Classification Loss')
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(alpha=0.3)

    # Param type loss
    axes[0, 2].plot(batches, param_type_losses, label='Param Type Loss', color='#2ecc71', linewidth=2)
    axes[0, 2].set_title('Parameter Type Loss')
    axes[0, 2].set_xlabel('Batch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(alpha=0.3)

    # Param value loss
    axes[1, 0].plot(batches, param_value_losses, label='Param Value Loss', color='#f39c12', linewidth=2)
    axes[1, 0].set_title('Parameter Value Loss')
    axes[1, 0].set_xlabel('Batch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(alpha=0.3)

    # Total loss
    axes[1, 1].plot(batches, total_losses, label='Total Loss', color='#9b59b6', linewidth=2)
    axes[1, 1].set_title('Total Loss')
    axes[1, 1].set_xlabel('Batch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(alpha=0.3)

    # All losses together
    axes[1, 2].plot(batches, type_losses, label='Type', color='#3498db', alpha=0.7)
    axes[1, 2].plot(batches, command_losses, label='Command', color='#e74c3c', alpha=0.7)
    axes[1, 2].plot(batches, param_type_losses, label='Param Type', color='#2ecc71', alpha=0.7)
    axes[1, 2].plot(batches, param_value_losses, label='Param Value', color='#f39c12', alpha=0.7)
    axes[1, 2].set_title('All Loss Components')
    axes[1, 2].set_xlabel('Batch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def generate_tsne_plot(embeddings: np.ndarray, labels: List[int], output_path: Path):
    """Generate t-SNE visualization of model embeddings."""
    if len(embeddings) == 0 or len(embeddings) < 10:
        print("  Not enough embeddings for t-SNE")
        return

    print(f"  Computing t-SNE for {len(embeddings)} samples...")

    # Subsample if too many points
    if len(embeddings) > 5000:
        indices = np.random.choice(len(embeddings), 5000, replace=False)
        embeddings = embeddings[indices]
        labels = [labels[i] for i in indices]

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    type_names = ['SPECIAL', 'COMMAND', 'PARAMETER', 'NUMERIC']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for type_id, (name, color) in enumerate(zip(type_names, colors)):
        mask = np.array(labels) == type_id
        if mask.sum() > 0:
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=color,
                label=name,
                alpha=0.6,
                s=20,
                edgecolors='black',
                linewidth=0.5
            )

    ax.set_title('t-SNE Visualization of Model Embeddings\n(Colored by Token Type)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def generate_tsne_plot_by_operation(embeddings: np.ndarray, labels: List[int], output_path: Path, operation_names: dict):
    """Generate t-SNE visualization of model embeddings colored by operation type."""
    if len(embeddings) == 0 or len(embeddings) < 10:
        print("  Not enough embeddings for operation t-SNE")
        return

    print(f"  Computing operation t-SNE for {len(embeddings)} samples...")

    # Subsample if too many points
    if len(embeddings) > 5000:
        indices = np.random.choice(len(embeddings), 5000, replace=False)
        embeddings = embeddings[indices]
        labels = [labels[i] for i in indices]

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define colors for different operation types
    unique_ops = sorted(set(labels))
    colors_palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

    for i, op_id in enumerate(unique_ops):
        mask = np.array(labels) == op_id
        if mask.sum() > 0:
            op_name = operation_names.get(op_id, f'Operation {op_id}')
            color = colors_palette[i % len(colors_palette)]
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=color,
                label=op_name,
                alpha=0.6,
                s=20,
                edgecolors='black',
                linewidth=0.5
            )

    ax.set_title('t-SNE Visualization of Model Embeddings\n(Colored by Operation Type)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def generate_operation_accuracy_chart(operation_accuracies: dict, operation_names: dict, output_path: Path):
    """Generate bar chart showing accuracy by operation type."""
    if not operation_accuracies:
        print("  No operation accuracy data")
        return

    # Sort by operation ID
    op_ids = sorted(operation_accuracies.keys())
    op_labels = [operation_names.get(op_id, f'Op {op_id}') for op_id in op_ids]

    # Extract metrics
    type_accs = [operation_accuracies[op_id]['type'] for op_id in op_ids]
    command_accs = [operation_accuracies[op_id]['command'] for op_id in op_ids]
    param_type_accs = [operation_accuracies[op_id]['param_type'] for op_id in op_ids]
    param_value_accs = [operation_accuracies[op_id]['param_value'] for op_id in op_ids]
    overall_accs = [operation_accuracies[op_id]['overall'] for op_id in op_ids]

    # Create grouped bar chart
    x = np.arange(len(op_labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 8))

    bars1 = ax.bar(x - 2*width, type_accs, width, label='Type', color='#3498db')
    bars2 = ax.bar(x - width, command_accs, width, label='Command', color='#e74c3c')
    bars3 = ax.bar(x, param_type_accs, width, label='Param Type', color='#2ecc71')
    bars4 = ax.bar(x + width, param_value_accs, width, label='Param Value', color='#f39c12')
    bars5 = ax.bar(x + 2*width, overall_accs, width, label='Overall', color='#9b59b6')

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Operation Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(op_labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def generate_per_class_f1_charts(metrics_dict: dict, output_dir: Path, chart_name: str):
    """Generate bar chart of per-class F1 scores."""
    # Sort by F1 score
    sorted_items = sorted(metrics_dict.items(), key=lambda x: x[1]['f1'], reverse=True)

    # Filter out classes with no support
    sorted_items = [(name, m) for name, m in sorted_items if m['support'] > 0]

    if not sorted_items:
        print(f"  No data for {chart_name}")
        return

    # Limit to top 20 classes if too many
    if len(sorted_items) > 20:
        sorted_items = sorted_items[:20]

    class_names = [name for name, _ in sorted_items]
    f1_scores = [m['f1'] * 100 for _, m in sorted_items]
    colors = [get_color_for_accuracy(f1) for f1 in f1_scores]

    fig, ax = plt.subplots(figsize=(12, max(6, len(class_names) * 0.3)))
    bars = ax.barh(class_names, f1_scores, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, val in zip(bars, f1_scores):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{chart_name} F1 Scores (Top {len(class_names)})',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    plt.tight_layout()
    output_path = output_dir / f'{chart_name.lower().replace(" ", "_")}_f1_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def generate_sample_predictions_table(sample_predictions: List[dict], vocab: dict,
                                      output_dir: Path, decomposer):
    """Generate HTML table of sample predictions."""
    if not sample_predictions:
        print("  No sample predictions to display")
        return

    # Separate correct and incorrect
    correct_samples = [s for s in sample_predictions if s['is_correct']]
    incorrect_samples = [s for s in sample_predictions if not s['is_correct']]

    # Limit to 10 each
    correct_samples = correct_samples[:10]
    incorrect_samples = incorrect_samples[:10]

    # Create reverse vocab
    id_to_token = {v: k for k, v in vocab.items()}

    def decode_tokens(token_ids):
        return ' '.join([id_to_token.get(tid, f'UNK({tid})') for tid in token_ids if tid not in [0, 2]])  # Skip PAD and EOS

    html_content = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }
        h2 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        th { background-color: #34495e; color: white; padding: 12px; text-align: left; font-weight: bold; }
        td { padding: 10px; border-bottom: 1px solid #ecf0f1; }
        tr:hover { background-color: #f8f9fa; }
        .correct { background-color: #d4edda; }
        .incorrect { background-color: #f8d7da; }
        .tokens { font-family: 'Courier New', monospace; font-size: 0.9em; }
        .metrics { font-size: 0.85em; color: #7f8c8d; }
    </style>
</head>
<body>
    <h1>Sample Predictions</h1>
"""

    # Correct predictions
    if correct_samples:
        html_content += "<h2>✓ Correct Predictions (Sample)</h2>\n<table>\n"
        html_content += "<tr><th>#</th><th>Target Sequence</th><th>Predicted Sequence</th><th>Accuracies</th></tr>\n"

        for i, sample in enumerate(correct_samples, 1):
            target_seq = decode_tokens(sample['target_tokens'])
            pred_seq = decode_tokens(sample['pred_tokens'])
            metrics = f"Type: {sample['type_acc']*100:.1f}% | Cmd: {sample['command_acc']*100:.1f}% | PT: {sample['param_type_acc']*100:.1f}% | PV: {sample['param_value_acc']*100:.1f}%"

            html_content += f"""
<tr class="correct">
    <td><b>{i}</b></td>
    <td class="tokens">{target_seq}</td>
    <td class="tokens">{pred_seq}</td>
    <td class="metrics">{metrics}</td>
</tr>
"""
        html_content += "</table>\n"

    # Incorrect predictions
    if incorrect_samples:
        html_content += "<h2>✗ Incorrect Predictions (Sample)</h2>\n<table>\n"
        html_content += "<tr><th>#</th><th>Target Sequence</th><th>Predicted Sequence</th><th>Accuracies</th></tr>\n"

        for i, sample in enumerate(incorrect_samples, 1):
            target_seq = decode_tokens(sample['target_tokens'])
            pred_seq = decode_tokens(sample['pred_tokens'])
            metrics = f"Type: {sample['type_acc']*100:.1f}% | Cmd: {sample['command_acc']*100:.1f}% | PT: {sample['param_type_acc']*100:.1f}% | PV: {sample['param_value_acc']*100:.1f}%"

            html_content += f"""
<tr class="incorrect">
    <td><b>{i}</b></td>
    <td class="tokens">{target_seq}</td>
    <td class="tokens">{pred_seq}</td>
    <td class="metrics">{metrics}</td>
</tr>
"""
        html_content += "</table>\n"

    html_content += """
</body>
</html>
"""

    output_path = output_dir / 'sample_predictions.html'
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"  Saved: {output_path.name}")


# ============================================================================
# CATEGORY 1: ERROR ANALYSIS FUNCTIONS
# ============================================================================

def generate_error_pattern_matrix(error_cases: List[dict], decomposer, output_dir: Path, head_name: str):
    """Generate heatmap showing top error patterns (misclassification pairs)."""
    if not error_cases or len(error_cases) == 0:
        print(f"  Skipped {head_name} error patterns: No error cases")
        return

    # Count misclassification pairs based on head type
    error_pairs = Counter()

    for error in error_cases:
        if head_name == 'type':
            true_labels = error['type_correct']
            # This is already boolean array, we need to find mismatches
            # Skip for now as we need actual predictions vs targets
            pass
        # For simplicity, we'll show edit distance distribution instead
        # since we don't have easy access to per-token true/pred pairs here

    # Since we don't have the right data structure, show edit distance distribution instead
    edit_distances = [e['edit_distance'] for e in error_cases]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(edit_distances, bins=20, color='#e74c3c', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Edit Distance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Error Edit Distance Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f'error_edit_distance_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def analyze_positional_accuracy(positional_data: List[dict], output_path: Path):
    """Analyze accuracy by token position in sequence."""
    if not positional_data or len(positional_data) == 0:
        print("  Skipped positional accuracy: No data")
        return

    # Group by position
    position_stats = defaultdict(list)
    for item in positional_data:
        position_stats[item['position']].append(item['correct'])

    # Compute accuracy per position
    positions = sorted(position_stats.keys())
    accuracies = [np.mean(position_stats[pos]) * 100 for pos in positions]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(positions, accuracies, marker='o', linewidth=2, markersize=4, color='#3498db')
    ax.axhline(y=np.mean(accuracies), color='#e74c3c', linestyle='--', label=f'Mean: {np.mean(accuracies):.1f}%')

    ax.set_xlabel('Token Position in Sequence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy by Token Position', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def generate_edit_distance_analysis(sequence_metrics: List[dict], output_path: Path):
    """Generate edit distance distribution histogram."""
    if not sequence_metrics or len(sequence_metrics) == 0:
        print("  Skipped edit distance analysis: No data")
        return

    edit_distances = [m['edit_distance'] for m in sequence_metrics]

    # Categorize failures
    categories = {
        'Perfect (0)': sum(1 for d in edit_distances if d == 0),
        '1-2 errors': sum(1 for d in edit_distances if 1 <= d <= 2),
        '3-5 errors': sum(1 for d in edit_distances if 3 <= d <= 5),
        '6+ errors': sum(1 for d in edit_distances if d >= 6),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    ax1.hist(edit_distances, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    ax1.axvline(x=np.mean(edit_distances), color='#e74c3c', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(edit_distances):.2f}')
    ax1.axvline(x=np.median(edit_distances), color='#2ecc71', linestyle='--',
                linewidth=2, label=f'Median: {np.median(edit_distances):.0f}')
    ax1.set_xlabel('Edit Distance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Edit Distance Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Pie chart of categories
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    ax2.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax2.set_title('Error Severity Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def cluster_error_embeddings(error_cases: List[dict], output_path: Path, n_clusters: int = 5):
    """Cluster error embeddings to find systematic error patterns."""
    if not error_cases or len(error_cases) < 10:
        print("  Skipped error clustering: Not enough error cases")
        return

    # Extract embeddings
    embeddings = np.array([e['embedding'] for e in error_cases])
    operation_ids = [e['operation_id'] for e in error_cases]

    # K-means clustering
    kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    for cluster_id in range(kmeans.n_clusters):
        mask = cluster_labels == cluster_id
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=colors[cluster_id % len(colors)], label=f'Cluster {cluster_id}',
                  alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

    ax.set_title('Error Case Clustering (t-SNE)', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


# ============================================================================
# CATEGORY 2: ATTENTION ANALYSIS FUNCTIONS
# ============================================================================

def extract_and_save_attention_weights(attention_samples: List[dict], output_path: Path):
    """Save attention weights to NPZ file for future analysis."""
    if not attention_samples or len(attention_samples) == 0:
        print("  Skipped attention weights: No samples")
        return

    # Save as compressed NPZ
    np.savez_compressed(
        output_path,
        attention_samples=attention_samples,
    )

    print(f"  Saved: {output_path.name} ({len(attention_samples)} samples)")


def generate_attention_heatmaps(attention_samples: List[dict], output_dir: Path, n_samples: int = 10):
    """Generate attention heatmaps for sample sequences."""
    if not attention_samples or len(attention_samples) == 0:
        print("  Skipped attention heatmaps: No samples")
        return

    # Generate heatmaps for first n_samples
    for idx, sample in enumerate(attention_samples[:n_samples]):
        attention = sample['attention'][0]  # [Tg, Tm] - averaged across layers and heads

        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(attention, cmap='viridis', cbar_kws={'label': 'Attention Weight'}, ax=ax)

        ax.set_xlabel('Encoder Position (Source)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Decoder Position (Target)', fontsize=12, fontweight='bold')
        ax.set_title(f'Attention Heatmap - Sample {idx+1} (Op {sample["operation_id"]})',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_path = output_dir / f'attention_heatmap_sample_{idx}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  Saved: {n_samples} attention heatmaps")


def analyze_head_specialization(attention_samples: List[dict], output_path: Path):
    """Analyze what each attention head attends to (placeholder - needs per-head data)."""
    if not attention_samples or len(attention_samples) == 0:
        print("  Skipped head specialization: No samples")
        return

    # This would require per-head attention weights
    # For now, show attention entropy distribution

    entropies = []
    for sample in attention_samples:
        attention = sample['attention'][0]  # [Tg, Tm]
        # Compute entropy per target position
        entropy = -np.sum(attention * np.log(attention + 1e-10), axis=-1)
        entropies.extend(entropy.tolist())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(entropies, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Attention Entropy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Attention Entropy Distribution\n(Higher = More Dispersed Attention)',
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def generate_layer_progression_plot(attention_samples: List[dict], output_path: Path):
    """Show how attention evolves across layers."""
    if not attention_samples or len(attention_samples) == 0:
        print("  Skipped layer progression: No samples")
        return

    # Average attention across all samples, per layer
    num_layers = len(attention_samples[0]['layer_attentions'])

    layer_entropies = []
    for layer_idx in range(num_layers):
        layer_attn_list = []
        for sample in attention_samples:
            layer_attn = sample['layer_attentions'][layer_idx][0]  # [Tg, Tm]
            # Compute mean entropy for this layer
            entropy = -np.sum(layer_attn * np.log(layer_attn + 1e-10), axis=-1).mean()
            layer_attn_list.append(entropy)
        layer_entropies.append(np.mean(layer_attn_list))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(num_layers), layer_entropies, marker='o', linewidth=2, markersize=8, color='#3498db')
    ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Attention Entropy', fontsize=12, fontweight='bold')
    ax.set_title('Attention Entropy Across Layers\n(Lower = More Focused)',
                fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xticks(range(num_layers))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


# ============================================================================
# CATEGORY 4: SEQUENCE ANALYSIS FUNCTIONS
# ============================================================================

def analyze_accuracy_vs_length(sequence_metrics: List[dict], output_path: Path):
    """Analyze how accuracy varies with sequence length."""
    if not sequence_metrics or len(sequence_metrics) == 0:
        print("  Skipped accuracy vs length: No data")
        return

    # Bin by length
    length_bins = [(0, 10), (11, 20), (21, 30), (31, 40), (41, 100)]
    bin_labels = ['0-10', '11-20', '21-30', '31-40', '41+']

    bin_accuracies = []
    for bin_range, label in zip(length_bins, bin_labels):
        matching = [m for m in sequence_metrics if bin_range[0] <= m['length'] <= bin_range[1]]
        if matching:
            acc = np.mean([m['overall_acc'] for m in matching]) * 100
            bin_accuracies.append(acc)
        else:
            bin_accuracies.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [get_color_for_accuracy(acc) for acc in bin_accuracies]
    bars = ax.bar(bin_labels, bin_accuracies, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, val in zip(bars, bin_accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Sequence Length', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def generate_edit_distance_by_operation(sequence_metrics: List[dict], operation_names: dict, output_path: Path):
    """Box plot of edit distance by operation type."""
    if not sequence_metrics or len(sequence_metrics) == 0:
        print("  Skipped edit distance by operation: No data")
        return

    # Group by operation
    op_edit_distances = defaultdict(list)
    for m in sequence_metrics:
        op_edit_distances[m['operation_id']].append(m['edit_distance'])

    # Prepare data for box plot
    op_ids = sorted(op_edit_distances.keys())
    data = [op_edit_distances[op_id] for op_id in op_ids]
    labels = [operation_names.get(op_id, f'Op {op_id}') for op_id in op_ids]

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

    # Color boxes
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
              '#1abc9c', '#e67e22', '#34495e', '#95a5a6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Operation Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Edit Distance', fontsize=12, fontweight='bold')
    ax.set_title('Edit Distance Distribution by Operation Type', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def analyze_partial_accuracy(sequence_metrics: List[dict], operation_names: dict, output_path: Path):
    """Analyze partial success rates (sequences with high but not perfect accuracy)."""
    if not sequence_metrics or len(sequence_metrics) == 0:
        print("  Skipped partial accuracy: No data")
        return

    # Categorize by partial correctness
    def categorize(partial_correct):
        if partial_correct == 1.0:
            return 'Perfect (100%)'
        elif partial_correct >= 0.9:
            return 'Near Perfect (90-99%)'
        elif partial_correct >= 0.5:
            return 'Partial (50-89%)'
        else:
            return 'Failed (<50%)'

    # Group by operation
    op_categories = defaultdict(lambda: defaultdict(int))
    for m in sequence_metrics:
        category = categorize(m['partial_correct'])
        op_categories[m['operation_id']][category] += 1

    # Prepare stacked bar chart data
    op_ids = sorted(op_categories.keys())
    categories = ['Perfect (100%)', 'Near Perfect (90-99%)', 'Partial (50-89%)', 'Failed (<50%)']
    colors_map = {'Perfect (100%)': '#2ecc71', 'Near Perfect (90-99%)': '#f39c12',
                  'Partial (50-89%)': '#e67e22', 'Failed (<50%)': '#e74c3c'}

    fig, ax = plt.subplots(figsize=(14, 6))

    bottom = np.zeros(len(op_ids))
    for category in categories:
        values = [op_categories[op_id][category] for op_id in op_ids]
        ax.bar([operation_names.get(op_id, f'Op {op_id}') for op_id in op_ids],
               values, bottom=bottom, label=category, color=colors_map[category],
               edgecolor='black', linewidth=0.5)
        bottom += values

    ax.set_xlabel('Operation Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Sequences', fontsize=12, fontweight='bold')
    ax.set_title('Partial Accuracy Breakdown by Operation', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


# ============================================================================
# CATEGORY 6: OPERATION-SPECIFIC ANALYSIS FUNCTIONS
# ============================================================================

def generate_per_operation_confusion_matrices(operation_predictions: dict, operation_targets: dict,
                                              decomposer, operation_names: dict, output_path: Path):
    """Generate grid of confusion matrices, one per operation."""
    if not operation_predictions or len(operation_predictions) == 0:
        print("  Skipped per-operation confusion matrices: No data")
        return

    # Use param_type head as it's most interesting
    ops_with_data = [op_id for op_id in operation_predictions.keys()
                     if len(operation_predictions[op_id]['param_type']) > 0]

    if len(ops_with_data) == 0:
        print("  Skipped per-operation confusion matrices: No predictions")
        return

    # Create grid (3x3 for 9 operations)
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axes = axes.flatten()

    param_type_names = decomposer.param_tokens

    for idx, op_id in enumerate(sorted(ops_with_data)):
        if idx >= 9:
            break

        # Concatenate predictions and targets
        preds = torch.cat(operation_predictions[op_id]['param_type']).numpy()
        targets = torch.cat(operation_targets[op_id]['param_type']).numpy()

        # Compute confusion matrix
        cm = confusion_matrix(targets, preds, labels=list(range(decomposer.n_param_types)))

        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=param_type_names, yticklabels=param_type_names,
                   cbar=False)
        axes[idx].set_title(f'{operation_names.get(op_id, f"Op {op_id}")}', fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')

    # Hide unused subplots
    for idx in range(len(ops_with_data), 9):
        axes[idx].axis('off')

    plt.suptitle('Parameter Type Confusion Matrices by Operation', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def analyze_cross_operation_transfer(operation_predictions: dict, operation_accuracies: dict,
                                     operation_names: dict, output_path: Path):
    """Analyze similarity/transfer between operations."""
    if not operation_accuracies or len(operation_accuracies) == 0:
        print("  Skipped cross-operation transfer: No data")
        return

    # Create similarity matrix based on accuracy patterns
    op_ids = sorted(operation_accuracies.keys())
    n_ops = len(op_ids)

    # Similarity based on accuracy vectors
    similarity_matrix = np.zeros((n_ops, n_ops))

    for i, op_i in enumerate(op_ids):
        vec_i = np.array([operation_accuracies[op_i]['type'],
                          operation_accuracies[op_i]['command'],
                          operation_accuracies[op_i]['param_type'],
                          operation_accuracies[op_i]['param_value']])
        for j, op_j in enumerate(op_ids):
            vec_j = np.array([operation_accuracies[op_j]['type'],
                              operation_accuracies[op_j]['command'],
                              operation_accuracies[op_j]['param_type'],
                              operation_accuracies[op_j]['param_value']])
            # Cosine similarity
            similarity = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j) + 1e-10)
            similarity_matrix[i, j] = similarity

    fig, ax = plt.subplots(figsize=(10, 8))
    labels = [operation_names.get(op_id, f'Op {op_id}') for op_id in op_ids]

    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
               xticklabels=labels, yticklabels=labels, ax=ax,
               vmin=0, vmax=1, cbar_kws={'label': 'Similarity'})

    ax.set_title('Cross-Operation Performance Similarity\n(Based on Accuracy Patterns)',
                fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def compare_damaged_vs_clean_operations(operation_accuracies: dict, operation_names: dict, output_path: Path):
    """Compare performance on clean vs damaged operations."""
    if not operation_accuracies or len(operation_accuracies) == 0:
        print("  Skipped damaged vs clean comparison: No data")
        return

    # Define pairs: (clean_op_id, damaged_op_id)
    pairs = [
        (0, 6, 'face'),      # face vs damageface
        (1, 7, 'pocket'),    # pocket vs damagepocket
        (2, 8, 'adaptive'),  # adaptive vs damageadaptive
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(pairs))
    width = 0.35

    clean_accs = []
    damaged_accs = []

    for clean_id, damaged_id, _ in pairs:
        if clean_id in operation_accuracies and damaged_id in operation_accuracies:
            clean_accs.append(operation_accuracies[clean_id]['overall'])
            damaged_accs.append(operation_accuracies[damaged_id]['overall'])
        else:
            clean_accs.append(0)
            damaged_accs.append(0)

    bars1 = ax.bar(x - width/2, clean_accs, width, label='Clean', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, damaged_accs, width, label='Damaged', color='#e74c3c', edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Statistical significance (placeholder - would need proper test)
    for i, (clean_acc, damaged_acc) in enumerate(zip(clean_accs, damaged_accs)):
        diff = abs(clean_acc - damaged_acc)
        if diff > 10:  # Arbitrary threshold
            ax.text(i, max(clean_acc, damaged_acc) + 5, '**', ha='center', fontsize=16)

    ax.set_ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance: Clean vs Damaged Operations', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([pair[2].capitalize() for pair in pairs])
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def generate_operation_error_modes(operation_predictions: dict, operation_targets: dict,
                                   decomposer, operation_names: dict, output_path: Path):
    """Identify errors unique to each operation type."""
    if not operation_predictions or len(operation_predictions) == 0:
        print("  Skipped operation error modes: No data")
        return

    error_modes = {}

    for op_id in operation_predictions.keys():
        if len(operation_predictions[op_id]['param_type']) == 0:
            continue

        # Get predictions and targets for param_type
        preds = torch.cat(operation_predictions[op_id]['param_type']).numpy()
        targets = torch.cat(operation_targets[op_id]['param_type']).numpy()

        # Find most common errors (misclassifications)
        errors = []
        for i in range(len(preds)):
            if preds[i] != targets[i]:
                errors.append((targets[i], preds[i]))

        if errors:
            error_counts = Counter(errors)
            top_errors = error_counts.most_common(5)

            param_type_names = decomposer.param_tokens

            error_modes[operation_names.get(op_id, f'Op {op_id}')] = [
                {
                    'true_class': param_type_names[true_idx] if true_idx < len(param_type_names) else f'Unknown({true_idx})',
                    'pred_class': param_type_names[pred_idx] if pred_idx < len(param_type_names) else f'Unknown({pred_idx})',
                    'count': count
                }
                for (true_idx, pred_idx), count in top_errors
            ]

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(error_modes, f, indent=2)

    print(f"  Saved: {output_path.name}")


def generate_all_outputs(eval_results: dict, output_dir: Path, vocab: dict):
    """Generate all confusion matrices and reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_dir = output_dir / 'confusion_matrices'
    cm_dir.mkdir(exist_ok=True)

    metrics_dir = output_dir / 'per_class_metrics'
    metrics_dir.mkdir(exist_ok=True)

    predictions = eval_results['predictions']
    targets = eval_results['targets']
    decomposer = eval_results['decomposer']
    accuracies = eval_results['accuracies']

    print("\n" + "="*80)
    print("GENERATING CONFUSION MATRICES")
    print("="*80)

    # 1. Type confusion matrix
    print("\n1. Token Type Confusion Matrix")
    # Type IDs: SPECIAL=0, COMMAND=1, PARAMETER=2, NUMERIC=3
    type_names = ['SPECIAL', 'COMMAND', 'PARAMETER', 'NUMERIC']
    n_types = 4  # Fixed: always 4 types
    type_cm = generate_confusion_matrix(
        targets['type'],
        predictions['type'],
        num_classes=n_types,
        class_names=type_names,
        output_path=cm_dir / 'type_confusion_matrix.png',
        title='Token Type Classification'
    )

    type_metrics = compute_per_class_metrics(
        targets['type'], predictions['type'],
        n_types, type_names
    )

    # 2. Command confusion matrix
    print("\n2. Command Confusion Matrix")
    # Use command_tokens list for names (ordered by ID)
    command_names = [decomposer.command_tokens[i] if i < len(decomposer.command_tokens) else f'CMD_{i}' for i in range(decomposer.n_commands)]
    command_cm = generate_confusion_matrix(
        targets['command'],
        predictions['command'],
        num_classes=decomposer.n_commands,
        class_names=command_names,
        output_path=cm_dir / 'command_confusion_matrix.png',
        title='G-Code Command Classification'
    )

    command_metrics = compute_per_class_metrics(
        targets['command'], predictions['command'],
        decomposer.n_commands, command_names
    )

    # 3. Parameter Type confusion matrix
    print("\n3. Parameter Type Confusion Matrix")
    # Use param_tokens list for names (ordered by ID)
    param_type_names = [decomposer.param_tokens[i] if i < len(decomposer.param_tokens) else f'PARAM_{i}' for i in range(decomposer.n_param_types)]
    param_type_cm = generate_confusion_matrix(
        targets['param_type'],
        predictions['param_type'],
        num_classes=decomposer.n_param_types,
        class_names=param_type_names,
        output_path=cm_dir / 'param_type_confusion_matrix.png',
        title='Parameter Type Classification'
    )

    param_type_metrics = compute_per_class_metrics(
        targets['param_type'], predictions['param_type'],
        decomposer.n_param_types, param_type_names
    )

    # 4. Parameter Value confusion matrix
    print("\n4. Parameter Value Confusion Matrix")
    param_value_names = [f'BUCKET_{i}' for i in range(decomposer.n_param_values)]
    param_value_cm = generate_confusion_matrix(
        targets['param_value'],
        predictions['param_value'],
        num_classes=decomposer.n_param_values,
        class_names=param_value_names,
        output_path=cm_dir / 'param_value_confusion_matrix.png',
        title='Parameter Value Bucket Classification'
    )

    param_value_metrics = compute_per_class_metrics(
        targets['param_value'], predictions['param_value'],
        decomposer.n_param_values, param_value_names
    )

    # 5. Full G-code Token confusion matrix
    print("\n5. Full G-Code Token Confusion Matrix")
    # Create reverse vocab mapping
    id_to_token = {v: k for k, v in vocab.items()}
    gcode_token_names = [id_to_token.get(i, f'TOKEN_{i}') for i in range(len(vocab))]

    gcode_cm = generate_confusion_matrix(
        targets['gcode_token'],
        predictions['gcode_token'],
        num_classes=len(vocab),
        class_names=gcode_token_names,
        output_path=cm_dir / 'gcode_token_confusion_matrix.png',
        title='Full G-Code Token Classification'
    )

    gcode_metrics = compute_per_class_metrics(
        targets['gcode_token'], predictions['gcode_token'],
        len(vocab), gcode_token_names
    )

    # Generate normalized confusion matrices
    print("\n" + "="*80)
    print("GENERATING NORMALIZED CONFUSION MATRICES")
    print("="*80)

    print("\n1. Type (Normalized)")
    generate_normalized_confusion_matrix(
        targets['type'], predictions['type'],
        n_types, type_names,
        cm_dir / 'type_confusion_matrix_normalized.png',
        'Token Type Classification'
    )

    print("\n2. Command (Normalized)")
    generate_normalized_confusion_matrix(
        targets['command'], predictions['command'],
        decomposer.n_commands, command_names,
        cm_dir / 'command_confusion_matrix_normalized.png',
        'G-Code Command Classification'
    )

    print("\n3. Parameter Type (Normalized)")
    generate_normalized_confusion_matrix(
        targets['param_type'], predictions['param_type'],
        decomposer.n_param_types, param_type_names,
        cm_dir / 'param_type_confusion_matrix_normalized.png',
        'Parameter Type Classification'
    )

    print("\n4. Parameter Value (Normalized)")
    generate_normalized_confusion_matrix(
        targets['param_value'], predictions['param_value'],
        decomposer.n_param_values, param_value_names,
        cm_dir / 'param_value_confusion_matrix_normalized.png',
        'Parameter Value Bucket Classification'
    )

    # Generate additional visualizations
    print("\n" + "="*80)
    print("GENERATING ADDITIONAL VISUALIZATIONS")
    print("="*80)

    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    bar_charts_dir = output_dir / 'bar_charts'
    bar_charts_dir.mkdir(exist_ok=True)

    # Accuracy bar chart
    print("\n1. Accuracy Comparison Bar Chart")
    generate_accuracy_bar_chart(accuracies, bar_charts_dir / 'accuracy_comparison.png')

    # Loss curves
    print("\n2. Loss Curves")
    if 'batch_losses' in eval_results and eval_results['batch_losses']:
        generate_loss_curves(eval_results['batch_losses'], viz_dir / 'loss_curves.png')
    else:
        print("  Skipped: No batch loss data available")

    # t-SNE plot (by token type)
    print("\n3. t-SNE Embedding Visualization (by Token Type)")
    if 'embeddings' in eval_results and len(eval_results['embeddings']) > 0:
        generate_tsne_plot(
            eval_results['embeddings'],
            eval_results['embedding_labels'],
            viz_dir / 'tsne_embeddings_by_token_type.png'
        )
    else:
        print("  Skipped: No embedding data available")

    # t-SNE plot (by operation type)
    print("\n4. t-SNE Embedding Visualization (by Operation Type)")
    if ('embeddings' in eval_results and len(eval_results['embeddings']) > 0 and
        'embedding_operation_labels' in eval_results and len(eval_results['embedding_operation_labels']) > 0):
        # Define operation names
        operation_names = {
            0: 'face',
            1: 'pocket',
            2: 'adaptive',
            3: 'face150025',
            4: 'pocket150025',
            5: 'adaptive150025',
            6: 'damageface',
            7: 'damagepocket',
            8: 'damageadaptive'
        }
        generate_tsne_plot_by_operation(
            eval_results['embeddings'],
            eval_results['embedding_operation_labels'],
            viz_dir / 'tsne_embeddings_by_operation.png',
            operation_names
        )
    else:
        print("  Skipped: No operation embedding data available")

    # Operation-specific accuracy breakdown
    print("\n5. Operation Type Performance Chart")
    if 'operation_accuracies' in eval_results and eval_results['operation_accuracies']:
        generate_operation_accuracy_chart(
            eval_results['operation_accuracies'],
            operation_names,
            bar_charts_dir / 'operation_type_performance.png'
        )
    else:
        print("  Skipped: No operation accuracy data available")

    # Per-class F1 charts
    print("\n6. Per-Class F1 Score Charts")
    generate_per_class_f1_charts(type_metrics, bar_charts_dir, 'Token Type')
    generate_per_class_f1_charts(command_metrics, bar_charts_dir, 'G-Code Commands')
    generate_per_class_f1_charts(param_type_metrics, bar_charts_dir, 'Parameter Types')

    # Sample predictions table
    print("\n7. Sample Predictions Table")
    if 'sample_predictions' in eval_results and eval_results['sample_predictions']:
        samples_dir = output_dir / 'sample_predictions'
        samples_dir.mkdir(exist_ok=True)
        generate_sample_predictions_table(
            eval_results['sample_predictions'],
            vocab,
            samples_dir,
            decomposer
        )
    else:
        print("  Skipped: No sample predictions available")

    # ============================================================================
    # ADVANCED ANALYSIS VISUALIZATIONS
    # ============================================================================

    print("\n" + "="*80)
    print("GENERATING ADVANCED ANALYSIS VISUALIZATIONS")
    print("="*80)

    # Create directories for advanced analysis
    error_analysis_dir = output_dir / 'error_analysis'
    error_analysis_dir.mkdir(exist_ok=True)

    attention_analysis_dir = output_dir / 'attention_analysis'
    attention_analysis_dir.mkdir(exist_ok=True)

    sequence_analysis_dir = output_dir / 'sequence_analysis'
    sequence_analysis_dir.mkdir(exist_ok=True)

    operation_analysis_dir = output_dir / 'operation_analysis'
    operation_analysis_dir.mkdir(exist_ok=True)

    # Category 1: Error Analysis
    print("\nCategory 1: Error Analysis")

    print("\n1.1 Error Pattern Analysis")
    if 'error_cases' in eval_results and eval_results['error_cases']:
        generate_error_pattern_matrix(
            eval_results['error_cases'],
            decomposer,
            error_analysis_dir,
            'type'
        )
    else:
        print("  Skipped: No error cases")

    print("\n1.2 Positional Accuracy Analysis")
    if 'positional_accuracy_data' in eval_results and eval_results['positional_accuracy_data']:
        analyze_positional_accuracy(
            eval_results['positional_accuracy_data'],
            error_analysis_dir / 'positional_accuracy.png'
        )
    else:
        print("  Skipped: No positional accuracy data")

    print("\n1.3 Edit Distance Analysis")
    if 'sequence_metrics' in eval_results and eval_results['sequence_metrics']:
        generate_edit_distance_analysis(
            eval_results['sequence_metrics'],
            error_analysis_dir / 'edit_distance_distribution.png'
        )
    else:
        print("  Skipped: No sequence metrics")

    print("\n1.4 Error Embedding Clustering")
    if 'error_cases' in eval_results and eval_results['error_cases']:
        cluster_error_embeddings(
            eval_results['error_cases'],
            error_analysis_dir / 'error_clusters_tsne.png'
        )
    else:
        print("  Skipped: No error cases")

    # Category 2: Attention Analysis
    print("\nCategory 2: Attention Analysis")

    print("\n2.1 Save Attention Weights")
    if 'attention_samples' in eval_results and eval_results['attention_samples']:
        extract_and_save_attention_weights(
            eval_results['attention_samples'],
            attention_analysis_dir / 'attention_weights.npz'
        )
    else:
        print("  Skipped: No attention samples")

    print("\n2.2 Generate Attention Heatmaps")
    if 'attention_samples' in eval_results and eval_results['attention_samples']:
        generate_attention_heatmaps(
            eval_results['attention_samples'],
            attention_analysis_dir,
            n_samples=10
        )
    else:
        print("  Skipped: No attention samples")

    print("\n2.3 Attention Head Specialization")
    if 'attention_samples' in eval_results and eval_results['attention_samples']:
        analyze_head_specialization(
            eval_results['attention_samples'],
            attention_analysis_dir / 'attention_entropy_distribution.png'
        )
    else:
        print("  Skipped: No attention samples")

    print("\n2.4 Layer Progression Analysis")
    if 'attention_samples' in eval_results and eval_results['attention_samples']:
        generate_layer_progression_plot(
            eval_results['attention_samples'],
            attention_analysis_dir / 'layer_progression.png'
        )
    else:
        print("  Skipped: No attention samples")

    # Category 4: Sequence Analysis
    print("\nCategory 4: Sequence Analysis")

    print("\n4.1 Accuracy vs Length")
    if 'sequence_metrics' in eval_results and eval_results['sequence_metrics']:
        analyze_accuracy_vs_length(
            eval_results['sequence_metrics'],
            sequence_analysis_dir / 'accuracy_vs_length.png'
        )
    else:
        print("  Skipped: No sequence metrics")

    print("\n4.2 Edit Distance by Operation")
    if 'sequence_metrics' in eval_results and eval_results['sequence_metrics']:
        generate_edit_distance_by_operation(
            eval_results['sequence_metrics'],
            operation_names,
            sequence_analysis_dir / 'edit_distance_by_operation.png'
        )
    else:
        print("  Skipped: No sequence metrics")

    print("\n4.3 Partial Accuracy Breakdown")
    if 'sequence_metrics' in eval_results and eval_results['sequence_metrics']:
        analyze_partial_accuracy(
            eval_results['sequence_metrics'],
            operation_names,
            sequence_analysis_dir / 'partial_accuracy_breakdown.png'
        )
    else:
        print("  Skipped: No sequence metrics")

    # Category 6: Operation-Specific Analysis
    print("\nCategory 6: Operation-Specific Analysis")

    print("\n6.1 Per-Operation Confusion Matrices")
    if 'operation_predictions' in eval_results and 'operation_targets' in eval_results:
        generate_per_operation_confusion_matrices(
            eval_results['operation_predictions'],
            eval_results['operation_targets'],
            decomposer,
            operation_names,
            operation_analysis_dir / 'confusion_matrices_grid.png'
        )
    else:
        print("  Skipped: No operation predictions/targets")

    print("\n6.2 Cross-Operation Transfer Analysis")
    if 'operation_accuracies' in eval_results and eval_results['operation_accuracies']:
        analyze_cross_operation_transfer(
            eval_results['operation_predictions'],
            eval_results['operation_accuracies'],
            operation_names,
            operation_analysis_dir / 'cross_operation_similarity.png'
        )
    else:
        print("  Skipped: No operation accuracies")

    print("\n6.3 Damaged vs Clean Comparison")
    if 'operation_accuracies' in eval_results and eval_results['operation_accuracies']:
        compare_damaged_vs_clean_operations(
            eval_results['operation_accuracies'],
            operation_names,
            operation_analysis_dir / 'damage_impact_comparison.png'
        )
    else:
        print("  Skipped: No operation accuracies")

    print("\n6.4 Operation-Specific Error Modes")
    if 'operation_predictions' in eval_results and 'operation_targets' in eval_results:
        generate_operation_error_modes(
            eval_results['operation_predictions'],
            eval_results['operation_targets'],
            decomposer,
            operation_names,
            operation_analysis_dir / 'operation_specific_errors.json'
        )
    else:
        print("  Skipped: No operation predictions/targets")

    # Save per-class metrics
    print("\n" + "="*80)
    print("SAVING PER-CLASS METRICS")
    print("="*80)

    with open(metrics_dir / 'type_metrics.json', 'w') as f:
        json.dump(type_metrics, f, indent=2)
    print(f"  Saved: type_metrics.json")

    with open(metrics_dir / 'command_metrics.json', 'w') as f:
        json.dump(command_metrics, f, indent=2)
    print(f"  Saved: command_metrics.json")

    with open(metrics_dir / 'param_type_metrics.json', 'w') as f:
        json.dump(param_type_metrics, f, indent=2)
    print(f"  Saved: param_type_metrics.json")

    with open(metrics_dir / 'param_value_metrics.json', 'w') as f:
        json.dump(param_value_metrics, f, indent=2)
    print(f"  Saved: param_value_metrics.json")

    with open(metrics_dir / 'gcode_token_metrics.json', 'w') as f:
        json.dump(gcode_metrics, f, indent=2)
    print(f"  Saved: gcode_token_metrics.json")

    # Generate summary report
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("COMPREHENSIVE EVALUATION REPORT")
    report_lines.append("="*80)
    report_lines.append("")

    report_lines.append("OVERALL ACCURACIES")
    report_lines.append("-" * 40)
    for metric_name, acc_value in accuracies.items():
        report_lines.append(f"  {metric_name:15s}: {acc_value:6.2f}%")
    report_lines.append("")

    report_lines.append("PER-CLASS PERFORMANCE SUMMARY")
    report_lines.append("-" * 40)

    # Type metrics summary
    report_lines.append("\nToken Type:")
    for class_name, metrics in type_metrics.items():
        report_lines.append(f"  {class_name:10s} - P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f} (n={metrics['support']})")

    # Top/Bottom commands
    report_lines.append("\nTop 5 Commands (by F1):")
    sorted_commands = sorted(command_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
    for class_name, metrics in sorted_commands[:5]:
        report_lines.append(f"  {class_name:15s} - P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f} (n={metrics['support']})")

    report_lines.append("\nBottom 5 Commands (by F1):")
    for class_name, metrics in sorted_commands[-5:]:
        if metrics['support'] > 0:  # Only show classes with samples
            report_lines.append(f"  {class_name:15s} - P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f} (n={metrics['support']})")

    # Parameter types
    report_lines.append("\nParameter Types:")
    sorted_params = sorted(param_type_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
    for class_name, metrics in sorted_params:
        if metrics['support'] > 0:
            report_lines.append(f"  {class_name:10s} - P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f} (n={metrics['support']})")

    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("OUTPUT FILES")
    report_lines.append("="*80)
    report_lines.append("\nConfusion Matrices (Raw Counts):")
    report_lines.append("  - confusion_matrices/type_confusion_matrix.png")
    report_lines.append("  - confusion_matrices/command_confusion_matrix.png")
    report_lines.append("  - confusion_matrices/param_type_confusion_matrix.png")
    report_lines.append("  - confusion_matrices/param_value_confusion_matrix.png")
    report_lines.append("  - confusion_matrices/gcode_token_confusion_matrix.png")
    report_lines.append("\nConfusion Matrices (Normalized):")
    report_lines.append("  - confusion_matrices/type_confusion_matrix_normalized.png")
    report_lines.append("  - confusion_matrices/command_confusion_matrix_normalized.png")
    report_lines.append("  - confusion_matrices/param_type_confusion_matrix_normalized.png")
    report_lines.append("  - confusion_matrices/param_value_confusion_matrix_normalized.png")
    report_lines.append("\nBar Charts:")
    report_lines.append("  - bar_charts/accuracy_comparison.png")
    report_lines.append("  - bar_charts/operation_type_performance.png")
    report_lines.append("  - bar_charts/token_type_f1_chart.png")
    report_lines.append("  - bar_charts/g-code_commands_f1_chart.png")
    report_lines.append("  - bar_charts/parameter_types_f1_chart.png")
    report_lines.append("\nVisualizations:")
    report_lines.append("  - visualizations/loss_curves.png")
    report_lines.append("  - visualizations/tsne_embeddings_by_token_type.png")
    report_lines.append("  - visualizations/tsne_embeddings_by_operation.png")
    report_lines.append("\nSample Predictions:")
    report_lines.append("  - sample_predictions/sample_predictions.html")
    report_lines.append("\nError Analysis:")
    report_lines.append("  - error_analysis/error_edit_distance_distribution.png")
    report_lines.append("  - error_analysis/positional_accuracy.png")
    report_lines.append("  - error_analysis/edit_distance_distribution.png")
    report_lines.append("  - error_analysis/error_clusters_tsne.png")
    report_lines.append("\nAttention Analysis:")
    report_lines.append("  - attention_analysis/attention_weights.npz")
    report_lines.append("  - attention_analysis/attention_heatmap_sample_{0-9}.png (10 files)")
    report_lines.append("  - attention_analysis/attention_entropy_distribution.png")
    report_lines.append("  - attention_analysis/layer_progression.png")
    report_lines.append("\nSequence Analysis:")
    report_lines.append("  - sequence_analysis/accuracy_vs_length.png")
    report_lines.append("  - sequence_analysis/edit_distance_by_operation.png")
    report_lines.append("  - sequence_analysis/partial_accuracy_breakdown.png")
    report_lines.append("\nOperation Analysis:")
    report_lines.append("  - operation_analysis/confusion_matrices_grid.png")
    report_lines.append("  - operation_analysis/cross_operation_similarity.png")
    report_lines.append("  - operation_analysis/damage_impact_comparison.png")
    report_lines.append("  - operation_analysis/operation_specific_errors.json")
    report_lines.append("\nPer-Class Metrics:")
    report_lines.append("  - per_class_metrics/type_metrics.json")
    report_lines.append("  - per_class_metrics/command_metrics.json")
    report_lines.append("  - per_class_metrics/param_type_metrics.json")
    report_lines.append("  - per_class_metrics/param_value_metrics.json")
    report_lines.append("  - per_class_metrics/gcode_token_metrics.json")
    report_lines.append("")
    report_lines.append("="*80)

    report_text = "\n".join(report_lines)

    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write(report_text)

    print(report_text)

    # Save accuracy breakdown
    with open(output_dir / 'accuracy_breakdown.json', 'w') as f:
        json.dump(accuracies, f, indent=2)

    print(f"\n✓ All outputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Model Evaluation - Generate ALL confusion matrices and metrics'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (.pt)'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Path to test data (.npz file)'
    )
    parser.add_argument(
        '--vocab-path',
        type=str,
        default=None,
        help='Path to vocabulary JSON (auto-detected if not provided)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for reports and figures'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for evaluation (default: 8)'
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    test_data_path = Path(args.test_data)
    output_dir = Path(args.output)

    # Auto-detect vocabulary if not provided
    if args.vocab_path is None:
        print("Auto-detecting vocabulary from checkpoint...")
        vocab_size, suggested_vocab_path = auto_detect_vocab(checkpoint_path)
        vocab_path = Path(suggested_vocab_path)
        print(f"  Detected vocab_size: {vocab_size}")
        print(f"  Using vocabulary: {vocab_path}")

        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Auto-detected vocabulary file not found: {vocab_path}\n"
                f"Please specify --vocab-path manually"
            )
    else:
        vocab_path = Path(args.vocab_path)

    # Load vocabulary for final output generation
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    # Extract actual vocab dict (handle nested structure)
    if isinstance(vocab_data, dict) and 'vocab' in vocab_data:
        vocab = vocab_data['vocab']
    else:
        vocab = vocab_data

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load checkpoint
    model_dict = load_checkpoint(checkpoint_path, vocab_path, device)

    # Run evaluation
    eval_results = evaluate_comprehensive(model_dict, test_data_path, args.batch_size)

    # Generate all outputs
    generate_all_outputs(eval_results, output_dir, vocab)

    print("\n" + "="*80)
    print("✓ COMPREHENSIVE EVALUATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
