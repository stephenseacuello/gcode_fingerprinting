#!/usr/bin/env python3
"""
Fixed training script with enhanced features for sweeps
- Converts boolean flags to proper boolean arguments
- Exposes hidden features (focal loss, Huber loss, etc.)
- Adds advanced training options
- Standardizes argument naming
"""
import os
import platform
import sys
from pathlib import Path
import json
import argparse

# Enable MPS fallback for Mac
if platform.system() == 'Darwin':
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  Warning: wandb not installed. Training without logging.")

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.multihead_lm import MultiHeadGCodeLM
from miracle.dataset.dataset import GCodeDataset, collate_fn
from miracle.dataset.data_augmentation import AugmentedGCodeDataset, get_rare_token_ids
from miracle.dataset.target_utils import TokenDecomposer
from miracle.training.losses import MultiHeadGCodeLoss, FocalLoss
from miracle.training.grammar_constraints import GCodeGrammarConstraints
from miracle.utilities.device import get_device, print_device_info
from collections import defaultdict
import psutil  # For memory monitoring


def compute_per_class_stats(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                           num_classes: int, class_names: list = None) -> dict:
    """
    Compute per-class prediction distribution and recall.

    Args:
        pred: [B, T] predicted class IDs
        target: [B, T] target class IDs
        mask: [B, T] boolean mask for valid positions
        num_classes: Total number of classes
        class_names: List of class names for logging

    Returns:
        Dictionary with per-class stats
    """
    stats = {
        'pred_counts': defaultdict(int),
        'target_counts': defaultdict(int),
        'correct_counts': defaultdict(int),
    }

    if not mask.any():
        return stats

    pred_masked = pred[mask]
    target_masked = target[mask]

    # Count predictions and targets per class
    for c in range(num_classes):
        stats['pred_counts'][c] = (pred_masked == c).sum().item()
        stats['target_counts'][c] = (target_masked == c).sum().item()
        stats['correct_counts'][c] = ((pred_masked == c) & (target_masked == c)).sum().item()

    return stats


def generate_example_prediction(model, backbone, decomposer, val_loader, device, vocab, num_examples=2):
    """
    Generate and print example predictions vs ground truth for debugging.

    Args:
        model: MultiHeadGCodeLM model
        backbone: MM_DTAE_LSTM backbone
        decomposer: TokenDecomposer for token composition/decomposition
        val_loader: Validation dataloader
        device: Torch device
        vocab: Vocabulary dictionary (id -> token)
        num_examples: Number of examples to show
    """
    model.eval()
    backbone.eval()

    # Build reverse mappings if not already present
    id2command = {v: k for k, v in decomposer.command2id.items()}
    id2param = {v: k for k, v in decomposer.param2id.items()}

    # Precision per parameter for 4-digit vocabulary reconstruction
    param_precision = {
        'X': 0.001, 'Y': 0.001, 'Z': 0.001,
        'A': 0.001, 'B': 0.001, 'C': 0.001,
        'I': 0.0001, 'J': 0.0001, 'K': 0.0001,
        'F': 1.0, 'S': 10.0, 'R': 0.0001,
        'P': 0.001, 'Q': 0.001, 'E': 0.0001
    }

    # Get a batch from validation set
    batch = next(iter(val_loader))

    with torch.no_grad():
        # Move to device
        continuous = batch['continuous'].to(device)
        categorical = batch['categorical'].float().to(device)
        tokens = batch['tokens'].to(device)
        lengths = batch['lengths'].to(device)

        # Get target (ground truth) - exclude BOS token
        tgt_in = tokens[:, :-1]
        tgt_out = tokens[:, 1:]

        # Forward through backbone
        mods = [continuous, categorical]
        backbone_out = backbone(mods=mods, lengths=lengths, gcode_in=tgt_in)
        memory = backbone_out['memory']

        # Get predictions using teacher forcing (same as training)
        logits = model(memory, tgt_in)

        # Get predicted tokens for each head
        type_pred = torch.argmax(logits['type_logits'], dim=-1)
        command_pred = torch.argmax(logits['command_logits'], dim=-1)
        param_type_pred = torch.argmax(logits['param_type_logits'], dim=-1)

        # Handle param_value based on model type
        if 'param_value_regression' in logits:
            param_value_regression = logits['param_value_regression'].squeeze(-1)
            param_value_pred = torch.round(param_value_regression).long()
            param_value_pred = torch.clamp(param_value_pred, 0, decomposer.n_param_values - 1)
        else:
            param_value_pred = torch.argmax(logits['param_value_logits'], dim=-1)

        # Decompose ground truth targets
        tgt_decomposed = decomposer.decompose_batch(tgt_out)

        print(f"\n{'='*80}")
        print(f"EXAMPLE PREDICTIONS (Teacher Forcing)")
        print(f"{'='*80}")

        # Show a few examples
        for b in range(min(num_examples, tokens.size(0))):
            # Get valid mask for this sequence
            valid_mask = tgt_decomposed['type'][b] != 0  # Non-PAD tokens
            valid_len = valid_mask.sum().item()

            if valid_len == 0:
                continue

            # Build predicted G-code string with proper reconstruction
            pred_parts = []
            current_param = None  # Track current parameter letter
            for t in range(min(valid_len, 20)):  # Limit to first 20 tokens
                tp = type_pred[b, t].item()
                if tp == 0:  # SPECIAL (PAD/BOS/EOS)
                    break
                elif tp == 1:  # COMMAND
                    cmd_id = command_pred[b, t].item()
                    if cmd_id in id2command:
                        pred_parts.append(id2command[cmd_id])
                    else:
                        pred_parts.append(f"CMD{cmd_id}")
                    current_param = None
                elif tp == 2:  # PARAMETER
                    param_id = param_type_pred[b, t].item()
                    if param_id in id2param:
                        current_param = id2param[param_id]
                    else:
                        current_param = f"P{param_id}"
                    # Don't append yet - wait for numeric value
                elif tp == 3:  # NUMERIC - combine with previous param
                    if 'param_value_regression' in logits:
                        val = param_value_regression[b, t].item()
                        if current_param:
                            pred_parts.append(f"{current_param}{val:.3f}")
                        else:
                            pred_parts.append(f"{val:.3f}")
                    else:
                        bucket_idx = param_value_pred[b, t].item()
                        precision = param_precision.get(current_param, 0.001)
                        value = bucket_idx * precision
                        if current_param:
                            # Format based on precision
                            if precision >= 1.0:
                                pred_parts.append(f"{current_param}{value:.0f}")
                            else:
                                pred_parts.append(f"{current_param}{value:.3f}")
                        else:
                            pred_parts.append(f"{value:.3f}")
                    current_param = None

            # Build ground truth G-code string with proper reconstruction
            gt_parts = []
            gt_current_param = None  # Track current parameter letter
            for t in range(min(valid_len, 20)):
                tp = tgt_decomposed['type'][b, t].item()
                if tp == 0:  # SPECIAL
                    break
                elif tp == 1:  # COMMAND
                    cmd_id = tgt_decomposed['command_id'][b, t].item()
                    if cmd_id in id2command:
                        gt_parts.append(id2command[cmd_id])
                    else:
                        gt_parts.append(f"CMD{cmd_id}")
                    gt_current_param = None
                elif tp == 2:  # PARAMETER
                    param_id = tgt_decomposed['param_type_id'][b, t].item()
                    if param_id in id2param:
                        gt_current_param = id2param[param_id]
                    else:
                        gt_current_param = f"P{param_id}"
                    # Don't append yet - wait for numeric value
                elif tp == 3:  # NUMERIC - combine with previous param
                    if 'param_value_raw' in batch:
                        raw_val = batch['param_value_raw'][b, t + 1].item()  # +1 because of BOS offset
                        if gt_current_param:
                            precision = param_precision.get(gt_current_param, 0.001)
                            if precision >= 1.0:
                                gt_parts.append(f"{gt_current_param}{raw_val:.0f}")
                            else:
                                gt_parts.append(f"{gt_current_param}{raw_val:.3f}")
                        else:
                            gt_parts.append(f"{raw_val:.3f}")
                    else:
                        bucket_idx = tgt_decomposed['param_value_id'][b, t].item()
                        precision = param_precision.get(gt_current_param, 0.001)
                        value = bucket_idx * precision
                        if gt_current_param:
                            if precision >= 1.0:
                                gt_parts.append(f"{gt_current_param}{value:.0f}")
                            else:
                                gt_parts.append(f"{gt_current_param}{value:.3f}")
                        else:
                            gt_parts.append(f"{value:.3f}")
                    gt_current_param = None

            # Format and print
            pred_str = ' '.join(pred_parts)
            gt_str = ' '.join(gt_parts)

            print(f"\n  Sample {b + 1}:")
            print(f"    GT:   {gt_str[:100]}{'...' if len(gt_str) > 100 else ''}")
            print(f"    Pred: {pred_str[:100]}{'...' if len(pred_str) > 100 else ''}")

            # Calculate token-by-token accuracy for this sample
            correct = 0
            total = 0
            for t in range(min(valid_len, 20)):
                gt_type = tgt_decomposed['type'][b, t].item()
                pred_type = type_pred[b, t].item()
                if gt_type == pred_type:
                    correct += 1
                total += 1

            if total > 0:
                print(f"    Type Accuracy: {correct}/{total} = {100*correct/total:.1f}%")


def print_enhanced_epoch_stats(train_metrics: dict, val_metrics: dict,
                               decomposer, epoch: int, max_epochs: int):
    """
    Print enhanced per-class statistics at the end of each epoch.

    Args:
        train_metrics: Training metrics including per-class stats
        val_metrics: Validation metrics including per-class stats
        decomposer: TokenDecomposer for class name lookup
        epoch: Current epoch number
        max_epochs: Maximum epochs
    """
    print(f"\n{'='*80}")
    print(f"ENHANCED EPOCH STATS - Epoch {epoch + 1}/{max_epochs}")
    print(f"{'='*80}")

    # Command distribution (G0, G1, G2, G3, etc.)
    if 'command_stats' in val_metrics:
        cmd_stats = val_metrics['command_stats']
        total_pred = sum(cmd_stats['pred_counts'].values())
        total_target = sum(cmd_stats['target_counts'].values())

        print(f"\n📊 Command Distribution (Validation):")
        print(f"   {'Command':<10} {'Predicted':>10} {'Actual':>10} {'Recall':>10}")
        print(f"   {'-'*45}")

        # Show key commands: G0, G1, G2, G3
        key_commands = ['G0', 'G1', 'G2', 'G3']
        for cmd_name in key_commands:
            if cmd_name in decomposer.command2id:
                cmd_id = decomposer.command2id[cmd_name]
                pred_count = cmd_stats['pred_counts'].get(cmd_id, 0)
                target_count = cmd_stats['target_counts'].get(cmd_id, 0)
                correct = cmd_stats['correct_counts'].get(cmd_id, 0)

                pred_pct = (pred_count / total_pred * 100) if total_pred > 0 else 0
                target_pct = (target_count / total_target * 100) if total_target > 0 else 0
                recall = (correct / target_count * 100) if target_count > 0 else 0

                print(f"   {cmd_name:<10} {pred_pct:>9.1f}% {target_pct:>9.1f}% {recall:>9.1f}%")

    # Param type distribution (X, Y, Z, F, etc.)
    if 'param_type_stats' in val_metrics:
        pt_stats = val_metrics['param_type_stats']
        total_pred = sum(pt_stats['pred_counts'].values())
        total_target = sum(pt_stats['target_counts'].values())

        print(f"\n🎯 Parameter Type Distribution (Validation):")
        print(f"   {'ParamType':<10} {'Predicted':>10} {'Actual':>10} {'Recall':>10}")
        print(f"   {'-'*45}")

        # Show key param types: X, Y, Z, F
        key_params = ['X', 'Y', 'Z', 'F']
        for param_name in key_params:
            if param_name in decomposer.param2id:
                param_id = decomposer.param2id[param_name]
                pred_count = pt_stats['pred_counts'].get(param_id, 0)
                target_count = pt_stats['target_counts'].get(param_id, 0)
                correct = pt_stats['correct_counts'].get(param_id, 0)

                pred_pct = (pred_count / total_pred * 100) if total_pred > 0 else 0
                target_pct = (target_count / total_target * 100) if total_target > 0 else 0
                recall = (correct / target_count * 100) if target_count > 0 else 0

                # Highlight F parameter with warning if recall is low
                if param_name == 'F' and recall < 50:
                    print(f"   {param_name:<10} {pred_pct:>9.1f}% {target_pct:>9.1f}% {recall:>9.1f}% ⚠️")
                else:
                    print(f"   {param_name:<10} {pred_pct:>9.1f}% {target_pct:>9.1f}% {recall:>9.1f}%")

    # Operation type distribution (face, adaptive, pocket, etc.)
    if 'operation_stats' in val_metrics:
        op_stats = val_metrics['operation_stats']
        total_pred = sum(op_stats['pred_counts'].values())
        total_target = sum(op_stats['target_counts'].values())

        # Operation type names (matches order in data files)
        operation_names = ['face', 'damageface', 'face150025', 'pocket', 'damagepocket', 'pocket150025', 'adaptive', 'damageadaptive', 'adaptive150025']

        print(f"\n🏭 Operation Type Distribution (Validation):")
        print(f"   {'Operation':<15} {'Predicted':>10} {'Actual':>10} {'Recall':>10}")
        print(f"   {'-'*50}")

        for op_id, op_name in enumerate(operation_names):
            pred_count = op_stats['pred_counts'].get(op_id, 0)
            target_count = op_stats['target_counts'].get(op_id, 0)
            correct = op_stats['correct_counts'].get(op_id, 0)

            pred_pct = (pred_count / total_pred * 100) if total_pred > 0 else 0
            target_pct = (target_count / total_target * 100) if total_target > 0 else 0
            recall = (correct / target_count * 100) if target_count > 0 else 0

            # Highlight low recall operations
            if recall < 50 and target_count > 0:
                print(f"   {op_name:<15} {pred_pct:>9.1f}% {target_pct:>9.1f}% {recall:>9.1f}% ⚠️")
            else:
                print(f"   {op_name:<15} {pred_pct:>9.1f}% {target_pct:>9.1f}% {recall:>9.1f}%")

    # Rare class recall summary
    print(f"\n🔍 Rare Class Recall Summary (Validation):")
    rare_classes = []

    if 'command_stats' in val_metrics:
        cmd_stats = val_metrics['command_stats']
        for cmd_name in ['G1', 'G2', 'G3']:
            if cmd_name in decomposer.command2id:
                cmd_id = decomposer.command2id[cmd_name]
                target_count = cmd_stats['target_counts'].get(cmd_id, 0)
                correct = cmd_stats['correct_counts'].get(cmd_id, 0)
                recall = (correct / target_count * 100) if target_count > 0 else 0
                rare_classes.append(f"{cmd_name}: {recall:.1f}%")

    if 'param_type_stats' in val_metrics:
        pt_stats = val_metrics['param_type_stats']
        if 'F' in decomposer.param2id:
            param_id = decomposer.param2id['F']
            target_count = pt_stats['target_counts'].get(param_id, 0)
            correct = pt_stats['correct_counts'].get(param_id, 0)
            recall = (correct / target_count * 100) if target_count > 0 else 0
            rare_classes.append(f"F: {recall:.1f}%")

    print(f"   {' | '.join(rare_classes)}")

    # Sequence-level accuracy
    if 'sequence_acc' in val_metrics:
        seq_correct = val_metrics.get('sequence_correct', 0)
        seq_total = val_metrics.get('sequence_total', 0)
        seq_acc = val_metrics['sequence_acc'] * 100
        print(f"\n📈 Sequence Accuracy: {seq_acc:.2f}% ({seq_correct}/{seq_total} complete sequences)")


def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_config(config_path: Path):
    """Load training configuration from JSON."""
    with open(config_path, 'r') as f:
        full_config = json.load(f)

    # Extract and flatten the nested config
    model_config = full_config.get('model_config', {})
    training_args = full_config.get('training_args', {})

    # Create flattened config with expected keys
    config = {
        'hidden_dim': model_config.get('d_model', 128),
        'num_layers': model_config.get('lstm_layers', 2),
        'num_heads': model_config.get('n_heads', 4),
        'batch_size': training_args.get('batch_size', 8),
        'learning_rate': training_args.get('lr', 0.001),
        'weight_decay': training_args.get('weight_decay', 0.01),
        'grad_clip': training_args.get('grad_clip', 1.0),
        'optimizer': training_args.get('optimizer', 'adamw'),
        'label_smoothing': 0.0,
    }

    return config


def train_epoch_multihead(model, backbone, decomposer, dataloader, optimizer, loss_fn, device, grad_clip=1.0,
                          grammar_constraints=None, grammar_weight=0.0, accumulation_steps=1, gradient_penalty=0.0):
    """Train for one epoch with multi-head model."""
    model.train()
    backbone.train()

    total_loss = 0
    all_losses = {
        'type': 0,
        'command': 0,
        'param_type': 0,
        'param_value': 0,
        'operation': 0,
        'grammar': 0,
    }

    # Accuracy tracking
    correct_type = 0
    correct_command = 0
    correct_param_type = 0
    correct_param_value = 0
    correct_operation = 0
    total_tokens = 0
    total_command_tokens = 0
    total_param_tokens = 0
    total_numeric_tokens = 0
    total_sequences = 0

    # Regression-specific metrics
    total_regression_mae = 0
    total_regression_within_tolerance = 0
    total_regression_count = 0

    # Token reconstruction accuracy
    correct_reconstructed = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        continuous = batch['continuous'].to(device)
        categorical = batch['categorical'].float().to(device)
        tokens = batch['tokens'].to(device)
        residuals = batch.get('residuals', torch.zeros_like(tokens).float()).to(device)
        param_value_raw = batch.get('param_value_raw', torch.zeros_like(tokens).float()).to(device)
        lengths = batch['lengths'].to(device)
        operation_type = batch.get('operation_type', torch.zeros(tokens.size(0), dtype=torch.long)).to(device)

        # Teacher forcing: input = tokens[:, :-1], target = tokens[:, 1:]
        tgt_in = tokens[:, :-1]

        # Forward through backbone
        mods = [continuous, categorical]
        backbone_out = backbone(mods=mods, lengths=lengths, gcode_in=tgt_in)
        memory = backbone_out['memory']
        tgt_out = tokens[:, 1:]

        # Forward through multi-head LM
        logits = model(memory, tgt_in)

        # Decompose targets
        if 'param_value_coarse_logits' in logits:
            tgt_decomposed = decomposer.decompose_batch_hybrid(tgt_out)
        else:
            tgt_decomposed = decomposer.decompose_batch(tgt_out)

        # Add operation type to targets
        tgt_decomposed['operation_type'] = operation_type
        tgt_decomposed['param_value_residual'] = residuals[:, 1:]
        tgt_decomposed['param_value_raw'] = param_value_raw[:, 1:]

        # Compute loss
        loss, loss_dict = loss_fn(logits, tgt_decomposed)

        # Compute grammar constraint loss if enabled
        grammar_loss = torch.tensor(0.0, device=device)
        if grammar_constraints is not None and grammar_weight > 0:
            grammar_loss_dict = grammar_constraints.compute_constraint_losses(
                predictions=logits,
                targets=tgt_decomposed,
                current_tokens=tgt_out
            )
            grammar_loss = grammar_weight * grammar_loss_dict['total_constraint']
            loss = loss + grammar_loss

        # Add gradient penalty if enabled
        if gradient_penalty > 0:
            grad_norm = 0
            for param in list(model.parameters()) + list(backbone.parameters()):
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            grad_penalty = gradient_penalty * grad_norm
            loss = loss + grad_penalty

        # Check for NaN/Inf in loss before backward pass
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️  WARNING: NaN/Inf loss detected at batch {batch_idx}!")
            print(f"   Loss components: type={loss_dict.get('type', 'N/A')}, cmd={loss_dict.get('command', 'N/A')}, "
                  f"op={loss_dict.get('operation', 'N/A')}")
            # Skip this batch to prevent corrupting gradients
            optimizer.zero_grad()
            continue

        # Backward with gradient accumulation
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        # Only step optimizer after accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            # Log gradient norm every 50 steps for debugging
            if batch_idx % 50 == 0:
                total_grad_norm = 0.0
                for p in list(model.parameters()) + list(backbone.parameters()):
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                # Store for W&B logging if enabled
                if hasattr(train_epoch_multihead, 'last_grad_norm'):
                    train_epoch_multihead.last_grad_norm = total_grad_norm

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(backbone.parameters()), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        # Track losses
        total_loss += loss.item()
        for k in all_losses:
            if k == 'grammar':
                all_losses[k] += grammar_loss.item()
            else:
                all_losses[k] += loss_dict.get(k, 0)

        # Compute accuracies
        B, T = tgt_out.shape

        # Predictions
        with torch.no_grad():
            type_pred = torch.argmax(logits['type_logits'], dim=-1)
            command_pred = torch.argmax(logits['command_logits'], dim=-1)
            param_type_pred = torch.argmax(logits['param_type_logits'], dim=-1)

            # Operation prediction
            if 'operation_logits' in logits:
                operation_pred = torch.argmax(logits['operation_logits'], dim=-1)
            else:
                operation_pred = None

            # Parameter value prediction
            is_regression_mode = 'param_value_regression' in logits

            if is_regression_mode:
                param_value_regression_pred = logits['param_value_regression'].squeeze(-1)
                param_value_regression_target = tgt_decomposed.get('param_value_raw', torch.zeros_like(param_value_regression_pred))
                param_value_pred = torch.round(param_value_regression_pred).long()
                param_value_pred = torch.clamp(param_value_pred, 0, decomposer.n_param_values - 1)
                param_value_target_key = 'param_value_id'
            elif 'param_value_coarse_logits' in logits:
                param_value_pred = torch.argmax(logits['param_value_coarse_logits'], dim=-1)
                param_value_target_key = 'param_value_coarse_id'
            else:
                param_value_pred = torch.argmax(logits['param_value_logits'], dim=-1)
                param_value_target_key = 'param_value_id'

            # Mask (ignore PAD tokens, type=0)
            valid_mask = tgt_decomposed['type'] != 0

            # Type accuracy
            correct_type += (type_pred == tgt_decomposed['type'])[valid_mask].sum().item()

            # Command accuracy
            command_mask = (tgt_decomposed['type'] == 1) & valid_mask
            if command_mask.any():
                correct_command += (command_pred == tgt_decomposed['command_id'])[command_mask].sum().item()
                total_command_tokens += command_mask.sum().item()

            # Parameter type accuracy
            param_mask = ((tgt_decomposed['type'] == 2) | (tgt_decomposed['type'] == 3)) & valid_mask
            if param_mask.any():
                correct_param_type += (param_type_pred == tgt_decomposed['param_type_id'])[param_mask].sum().item()
                total_param_tokens += param_mask.sum().item()

            # Parameter value accuracy/metrics
            numeric_mask = (tgt_decomposed['type'] == 3) & valid_mask
            if numeric_mask.any():
                if is_regression_mode:
                    regression_pred = param_value_regression_pred[numeric_mask]
                    regression_target = param_value_regression_target[numeric_mask]
                    mae = torch.abs(regression_pred - regression_target).sum().item()
                    total_regression_mae += mae
                    within_tolerance = (torch.abs(regression_pred - regression_target) < 1.0).sum().item()
                    total_regression_within_tolerance += within_tolerance
                    total_regression_count += numeric_mask.sum().item()
                else:
                    correct_param_value += (param_value_pred == tgt_decomposed[param_value_target_key])[numeric_mask].sum().item()

                total_numeric_tokens += numeric_mask.sum().item()

            total_tokens += valid_mask.sum().item()
            total_sequences += B

            # Operation accuracy
            if operation_pred is not None and 'operation_type' in tgt_decomposed:
                correct_operation += (operation_pred == tgt_decomposed['operation_type']).sum().item()

            # Reconstruct tokens and check accuracy
            if 'param_value_coarse_logits' not in logits and not is_regression_mode:
                predicted_targets = {
                    'type': type_pred,
                    'command_id': command_pred,
                    'param_type_id': param_type_pred,
                    'param_value_id': param_value_pred,
                }
                reconstructed_tokens = decomposer.compose_batch(predicted_targets)
                correct_reconstructed += (reconstructed_tokens == tgt_out)[valid_mask].sum().item()

        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'grammar': f"{grammar_loss.item():.4f}"})

    # Average losses and accuracies
    n_batches = len(dataloader)
    type_acc = correct_type / max(total_tokens, 1)
    command_acc = correct_command / max(total_command_tokens, 1)
    param_type_acc = correct_param_type / max(total_param_tokens, 1)
    operation_acc = correct_operation / max(total_sequences, 1)
    overall_acc = correct_reconstructed / max(total_tokens, 1)

    metrics = {
        'loss': total_loss / n_batches,
        **{f'{k}_loss': v / n_batches for k, v in all_losses.items()},
        'type_acc': type_acc,
        'command_acc': command_acc,
        'param_type_acc': param_type_acc,
        'operation_acc': operation_acc,
        'overall_acc': overall_acc,
    }

    # Add regression or bucketing metrics
    if total_regression_count > 0:
        param_value_mae = total_regression_mae / max(total_regression_count, 1)
        param_value_tolerance_acc = total_regression_within_tolerance / max(total_regression_count, 1)
        metrics['param_value_mae'] = param_value_mae
        metrics['param_value_tolerance_acc'] = param_value_tolerance_acc
        metrics['composite_acc'] = command_acc * param_type_acc * param_value_tolerance_acc
    else:
        param_value_acc = correct_param_value / max(total_numeric_tokens, 1)
        metrics['param_value_acc'] = param_value_acc
        metrics['composite_acc'] = command_acc * param_type_acc * param_value_acc

    return metrics


def validate_multihead(model, backbone, decomposer, dataloader, loss_fn, device, vocab_size):
    """Validate multi-head model with enhanced per-class statistics."""
    model.eval()
    backbone.eval()

    total_loss = 0
    all_losses = {
        'type': 0,
        'command': 0,
        'param_type': 0,
        'param_value': 0,
        'operation': 0,
    }

    # Accuracy tracking
    correct_type = 0
    correct_command = 0
    correct_param_type = 0
    correct_param_value = 0
    correct_operation = 0
    total_tokens = 0
    total_command_tokens = 0
    total_param_tokens = 0
    total_numeric_tokens = 0
    total_sequences = 0

    # Regression-specific metrics
    total_regression_mae = 0
    total_regression_within_tolerance = 0
    total_regression_count = 0

    # Token reconstruction accuracy
    correct_reconstructed = 0

    # Enhanced per-class tracking
    command_pred_counts = defaultdict(int)
    command_target_counts = defaultdict(int)
    command_correct_counts = defaultdict(int)

    param_type_pred_counts = defaultdict(int)
    param_type_target_counts = defaultdict(int)
    param_type_correct_counts = defaultdict(int)

    # Per-operation type tracking (9 types: face, damageface, face150025, pocket, damagepocket, pocket150025, adaptive, damageadaptive, adaptive150025)
    operation_pred_counts = defaultdict(int)
    operation_target_counts = defaultdict(int)
    operation_correct_counts = defaultdict(int)

    # Sequence-level accuracy
    sequence_correct = 0
    sequence_total = 0

    pbar = tqdm(dataloader, desc="Validation")
    with torch.no_grad():
        for batch in pbar:
            # Move to device
            continuous = batch['continuous'].to(device)
            categorical = batch['categorical'].float().to(device)
            tokens = batch['tokens'].to(device)
            residuals = batch.get('residuals', torch.zeros_like(tokens).float()).to(device)
            param_value_raw = batch.get('param_value_raw', torch.zeros_like(tokens).float()).to(device)
            lengths = batch['lengths'].to(device)
            operation_type = batch.get('operation_type', torch.zeros(tokens.size(0), dtype=torch.long)).to(device)

            # Teacher forcing
            tgt_in = tokens[:, :-1]

            # Forward through backbone
            mods = [continuous, categorical]
            backbone_out = backbone(mods=mods, lengths=lengths, gcode_in=tgt_in)
            memory = backbone_out['memory']
            tgt_out = tokens[:, 1:]

            # Forward
            logits = model(memory, tgt_in)

            # Decompose targets
            if 'param_value_coarse_logits' in logits:
                tgt_decomposed = decomposer.decompose_batch_hybrid(tgt_out)
            else:
                tgt_decomposed = decomposer.decompose_batch(tgt_out)

            # Add operation type to targets
            tgt_decomposed['operation_type'] = operation_type
            tgt_decomposed['param_value_residual'] = residuals[:, 1:]
            tgt_decomposed['param_value_raw'] = param_value_raw[:, 1:]

            # Compute loss
            loss, loss_dict = loss_fn(logits, tgt_decomposed)
            total_loss += loss.item()
            for k in all_losses:
                all_losses[k] += loss_dict.get(k, 0)

            # Compute accuracies
            B, T = tgt_out.shape

            # Predictions
            type_pred = torch.argmax(logits['type_logits'], dim=-1)
            command_pred = torch.argmax(logits['command_logits'], dim=-1)
            param_type_pred = torch.argmax(logits['param_type_logits'], dim=-1)

            # Operation prediction
            if 'operation_logits' in logits:
                operation_pred = torch.argmax(logits['operation_logits'], dim=-1)
            else:
                operation_pred = None

            # Parameter value prediction
            is_regression_mode = 'param_value_regression' in logits

            if is_regression_mode:
                param_value_regression_pred = logits['param_value_regression'].squeeze(-1)
                param_value_regression_target = tgt_decomposed.get('param_value_raw', torch.zeros_like(param_value_regression_pred))
                param_value_pred = torch.round(param_value_regression_pred).long()
                param_value_pred = torch.clamp(param_value_pred, 0, decomposer.n_param_values - 1)
                param_value_target_key = 'param_value_id'
            elif 'param_value_coarse_logits' in logits:
                param_value_pred = torch.argmax(logits['param_value_coarse_logits'], dim=-1)
                param_value_target_key = 'param_value_coarse_id'
            else:
                param_value_pred = torch.argmax(logits['param_value_logits'], dim=-1)
                param_value_target_key = 'param_value_id'

            # Mask (ignore PAD tokens, type=0)
            valid_mask = tgt_decomposed['type'] != 0

            # Type accuracy
            correct_type += (type_pred == tgt_decomposed['type'])[valid_mask].sum().item()

            # Command accuracy with per-class tracking
            command_mask = (tgt_decomposed['type'] == 1) & valid_mask
            if command_mask.any():
                correct_command += (command_pred == tgt_decomposed['command_id'])[command_mask].sum().item()
                total_command_tokens += command_mask.sum().item()

                # Per-class stats for commands
                cmd_pred_masked = command_pred[command_mask]
                cmd_target_masked = tgt_decomposed['command_id'][command_mask]
                for c in range(decomposer.n_commands):
                    command_pred_counts[c] += (cmd_pred_masked == c).sum().item()
                    command_target_counts[c] += (cmd_target_masked == c).sum().item()
                    command_correct_counts[c] += ((cmd_pred_masked == c) & (cmd_target_masked == c)).sum().item()

            # Parameter type accuracy with per-class tracking
            param_mask = ((tgt_decomposed['type'] == 2) | (tgt_decomposed['type'] == 3)) & valid_mask
            if param_mask.any():
                correct_param_type += (param_type_pred == tgt_decomposed['param_type_id'])[param_mask].sum().item()
                total_param_tokens += param_mask.sum().item()

                # Per-class stats for param types
                pt_pred_masked = param_type_pred[param_mask]
                pt_target_masked = tgt_decomposed['param_type_id'][param_mask]
                for c in range(decomposer.n_param_types):
                    param_type_pred_counts[c] += (pt_pred_masked == c).sum().item()
                    param_type_target_counts[c] += (pt_target_masked == c).sum().item()
                    param_type_correct_counts[c] += ((pt_pred_masked == c) & (pt_target_masked == c)).sum().item()

            # Parameter value accuracy/metrics
            numeric_mask = (tgt_decomposed['type'] == 3) & valid_mask
            if numeric_mask.any():
                if is_regression_mode:
                    regression_pred = param_value_regression_pred[numeric_mask]
                    regression_target = param_value_regression_target[numeric_mask]
                    mae = torch.abs(regression_pred - regression_target).sum().item()
                    total_regression_mae += mae
                    within_tolerance = (torch.abs(regression_pred - regression_target) < 1.0).sum().item()
                    total_regression_within_tolerance += within_tolerance
                    total_regression_count += numeric_mask.sum().item()
                else:
                    correct_param_value += (param_value_pred == tgt_decomposed[param_value_target_key])[numeric_mask].sum().item()

                total_numeric_tokens += numeric_mask.sum().item()

            total_tokens += valid_mask.sum().item()
            total_sequences += B

            # Operation accuracy with per-class tracking
            if operation_pred is not None and 'operation_type' in tgt_decomposed:
                correct_operation += (operation_pred == tgt_decomposed['operation_type']).sum().item()

                # Per-class stats for operation types (9 types: face, damageface, face150025, pocket, damagepocket, pocket150025, adaptive, damageadaptive, adaptive150025)
                op_pred = operation_pred.cpu()
                op_target = tgt_decomposed['operation_type'].cpu()
                for c in range(9):  # 9 operation types
                    operation_pred_counts[c] += (op_pred == c).sum().item()
                    operation_target_counts[c] += (op_target == c).sum().item()
                    operation_correct_counts[c] += ((op_pred == c) & (op_target == c)).sum().item()

            # Reconstruct tokens and check accuracy + sequence-level tracking
            if 'param_value_coarse_logits' not in logits and not is_regression_mode:
                predicted_targets = {
                    'type': type_pred,
                    'command_id': command_pred,
                    'param_type_id': param_type_pred,
                    'param_value_id': param_value_pred,
                }
                reconstructed_tokens = decomposer.compose_batch(predicted_targets)
                correct_reconstructed += (reconstructed_tokens == tgt_out)[valid_mask].sum().item()

                # Sequence-level accuracy: check if entire sequence is correct
                for b in range(B):
                    seq_valid_mask = valid_mask[b]
                    if seq_valid_mask.any():
                        seq_correct = (reconstructed_tokens[b] == tgt_out[b])[seq_valid_mask].all().item()
                        if seq_correct:
                            sequence_correct += 1
                        sequence_total += 1

    # Average metrics
    n_batches = len(dataloader)
    type_acc = correct_type / max(total_tokens, 1)
    command_acc = correct_command / max(total_command_tokens, 1)
    param_type_acc = correct_param_type / max(total_param_tokens, 1)
    operation_acc = correct_operation / max(total_sequences, 1)
    overall_acc = correct_reconstructed / max(total_tokens, 1)

    metrics = {
        'loss': total_loss / n_batches,
        **{f'{k}_loss': v / n_batches for k, v in all_losses.items()},
        'type_acc': type_acc,
        'command_acc': command_acc,
        'param_type_acc': param_type_acc,
        'operation_acc': operation_acc,
        'overall_acc': overall_acc,
    }

    # Add regression or bucketing metrics
    if total_regression_count > 0:
        param_value_mae = total_regression_mae / max(total_regression_count, 1)
        param_value_tolerance_acc = total_regression_within_tolerance / max(total_regression_count, 1)
        metrics['param_value_mae'] = param_value_mae
        metrics['param_value_tolerance_acc'] = param_value_tolerance_acc
        metrics['composite_acc'] = command_acc * param_type_acc * param_value_tolerance_acc
    else:
        param_value_acc = correct_param_value / max(total_numeric_tokens, 1)
        metrics['param_value_acc'] = param_value_acc
        metrics['composite_acc'] = command_acc * param_type_acc * param_value_acc

    # Add enhanced per-class stats
    metrics['command_stats'] = {
        'pred_counts': dict(command_pred_counts),
        'target_counts': dict(command_target_counts),
        'correct_counts': dict(command_correct_counts),
    }
    metrics['param_type_stats'] = {
        'pred_counts': dict(param_type_pred_counts),
        'target_counts': dict(param_type_target_counts),
        'correct_counts': dict(param_type_correct_counts),
    }
    metrics['operation_stats'] = {
        'pred_counts': dict(operation_pred_counts),
        'target_counts': dict(operation_target_counts),
        'correct_counts': dict(operation_correct_counts),
    }

    # Add sequence-level accuracy
    metrics['sequence_acc'] = sequence_correct / max(sequence_total, 1)
    metrics['sequence_correct'] = sequence_correct
    metrics['sequence_total'] = sequence_total

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train multi-head model with enhanced features')

    # Basic configuration
    parser.add_argument('--config', type=str, default='configs/phase1_best.json',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='outputs/processed_v2',
                        help='Path to preprocessed data directory')
    parser.add_argument('--vocab-path', type=str, default='data/gcode_vocab_v2.json',
                        help='Path to vocabulary file')
    parser.add_argument('--output-dir', type=str, default='outputs/multihead_v2',
                        help='Output directory for checkpoints')
    parser.add_argument('--class-weights-path', type=str, default=None,
                        help='Path to class weights JSON file')

    # Training control
    parser.add_argument('--max-epochs', type=int, default=50,
                        help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--track-metric', type=str, default='operation_acc',
                        choices=['operation_acc', 'composite_acc', 'type_acc', 'param_type_acc'],
                        help='Metric to track for early stopping (default: operation_acc for 9-class classification)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint path')

    # Model architecture (standardized naming with hyphens)
    parser.add_argument('--hidden-dim', type=int, default=None, dest='hidden_dim',
                        help='Hidden dimension (overrides config)')
    parser.add_argument('--num-layers', type=int, default=None, dest='num_layers',
                        help='Number of LSTM layers (overrides config)')
    parser.add_argument('--num-heads', type=int, default=None, dest='num_heads',
                        help='Number of attention heads (overrides config)')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout probability (default: 0.1)')

    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=None, dest='learning_rate',
                        help='Learning rate (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None, dest='batch_size',
                        help='Batch size (overrides config)')
    parser.add_argument('--weight-decay', type=float, default=None, dest='weight_decay',
                        help='Weight decay (overrides config)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--grad_clip', type=float, default=None, dest='grad_clip_override',
                        help='Gradient clipping value (overrides config)')

    # Loss weights
    parser.add_argument('--command-weight', type=float, default=None, dest='command_weight',
                        help='Loss weight for command prediction (default: 3.0)')
    parser.add_argument('--operation-weight', type=float, default=None, dest='operation_weight',
                        help='Loss weight for operation type prediction (default: 2.0)')
    parser.add_argument('--param-type-weight', type=float, default=None, dest='param_type_weight',
                        help='Loss weight for parameter type prediction (default: 1.0)')
    parser.add_argument('--param-value-weight', type=float, default=None, dest='param_value_weight',
                        help='Loss weight for parameter value prediction (default: 1.0)')
    parser.add_argument('--grammar-weight', type=float, default=None, dest='grammar_weight',
                        help='Loss weight for grammar constraint violations (default: 0.1)')
    parser.add_argument('--label-smoothing', type=float, default=None, dest='label_smoothing',
                        help='Label smoothing for regularization (default: 0.0)')

    # Data augmentation (converted to boolean argument)
    parser.add_argument('--augmentation', type=str2bool, nargs='?', const=True, default=False,
                        help='Use data augmentation (true/false)')
    parser.add_argument('--oversample-factor', type=int, default=3,
                        help='Oversampling factor (if using augmentation)')

    # Advanced augmentation
    parser.add_argument('--noise-std', type=float, default=0.0,
                        help='Noise injection standard deviation (default: 0.0)')
    parser.add_argument('--mixup-alpha', type=float, default=0.0,
                        help='Mixup alpha parameter (default: 0.0, disabled)')
    parser.add_argument('--cutmix-prob', type=float, default=0.0,
                        help='CutMix probability (default: 0.0, disabled)')

    # Learning rate scheduler
    parser.add_argument('--lr-scheduler', type=str, default='none',
                        choices=['none', 'cosine', 'plateau', 'step', 'cyclic', 'onecycle'],
                        help='Learning rate scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='Number of warmup epochs')
    parser.add_argument('--cosine-t-max', type=int, default=None,
                        help='Cosine annealing T_max (default: max_epochs)')
    parser.add_argument('--plateau-patience', type=int, default=5,
                        help='ReduceLROnPlateau patience')
    parser.add_argument('--plateau-factor', type=float, default=0.5,
                        help='ReduceLROnPlateau factor')

    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd', 'rmsprop'],
                        help='Optimizer type')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam/AdamW beta1 parameter')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam/AdamW beta2 parameter')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (only for SGD optimizer)')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps')

    # Advanced regularization
    parser.add_argument('--gradient-penalty', type=float, default=0.0,
                        help='Gradient penalty weight (default: 0.0)')
    parser.add_argument('--use-swa', type=str2bool, nargs='?', const=True, default=False,
                        help='Use Stochastic Weight Averaging')
    parser.add_argument('--swa-start-epoch', type=int, default=75,
                        help='SWA start epoch')

    # Advanced loss functions
    parser.add_argument('--use-focal-loss', type=str2bool, nargs='?', const=True, default=False,
                        help='Use focal loss for imbalanced classes')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='Focal loss alpha parameter')
    parser.add_argument('--use-huber-loss', type=str2bool, nargs='?', const=True, default=False,
                        help='Use Huber loss for regression')
    parser.add_argument('--huber-delta', type=float, default=1.0,
                        help='Huber loss delta parameter')

    # Mode collapse prevention
    parser.add_argument('--detect-mode-collapse', type=str2bool, nargs='?', const=True, default=False,
                        help='Enable mode collapse detection')
    parser.add_argument('--mode-collapse-threshold', type=float, default=0.8,
                        help='Mode collapse detection threshold')
    parser.add_argument('--diversity-penalty', type=float, default=0.0,
                        help='Diversity penalty weight')

    # Model initialization
    parser.add_argument('--init-strategy', type=str, default='xavier_uniform',
                        choices=['xavier_uniform', 'xavier_normal', 'kaiming_uniform',
                                'kaiming_normal', 'orthogonal'],
                        help='Weight initialization strategy')
    parser.add_argument('--init-gain', type=float, default=1.0,
                        help='Initialization gain parameter')

    # W&B logging
    parser.add_argument('--use-wandb', type=str2bool, nargs='?', const=True, default=False,
                        help='Enable W&B logging')
    parser.add_argument('--wandb-project', type=str, default='gcode-fingerprinting',
                        help='W&B project name')
    parser.add_argument('--run-name', type=str, default='multihead-enhanced',
                        help='W&B run name')

    # Legacy underscore arguments for backward compatibility
    parser.add_argument('--hidden_dim', type=int, default=None, dest='hidden_dim_legacy',
                        help='Hidden dimension (legacy, use --hidden-dim)')
    parser.add_argument('--num_layers', type=int, default=None, dest='num_layers_legacy',
                        help='Number of layers (legacy, use --num-layers)')
    parser.add_argument('--num_heads', type=int, default=None, dest='num_heads_legacy',
                        help='Number of heads (legacy, use --num-heads)')
    parser.add_argument('--learning_rate', type=float, default=None, dest='learning_rate_legacy',
                        help='Learning rate (legacy, use --learning-rate)')
    parser.add_argument('--batch_size', type=int, default=None, dest='batch_size_legacy',
                        help='Batch size (legacy, use --batch-size)')
    parser.add_argument('--weight_decay', type=float, default=None, dest='weight_decay_legacy',
                        help='Weight decay (legacy, use --weight-decay)')
    parser.add_argument('--command_weight', type=float, default=None, dest='command_weight_legacy',
                        help='Command weight (legacy, use --command-weight)')
    parser.add_argument('--operation_weight', type=float, default=None, dest='operation_weight_legacy',
                        help='Operation weight (legacy, use --operation-weight)')
    parser.add_argument('--param_type_weight', type=float, default=None, dest='param_type_weight_legacy',
                        help='Param type weight (legacy, use --param-type-weight)')
    parser.add_argument('--param_value_weight', type=float, default=None, dest='param_value_weight_legacy',
                        help='Param value weight (legacy, use --param-value-weight)')
    parser.add_argument('--grammar_weight', type=float, default=None, dest='grammar_weight_legacy',
                        help='Grammar weight (legacy, use --grammar-weight)')
    parser.add_argument('--label_smoothing', type=float, default=None, dest='label_smoothing_legacy',
                        help='Label smoothing (legacy, use --label-smoothing)')
    parser.add_argument('--use-augmentation', action='store_true', dest='use_augmentation_legacy',
                        help='Use augmentation (legacy, use --augmentation)')

    # Additional legacy underscore arguments for wandb sweep compatibility
    parser.add_argument('--accumulation_steps', type=int, default=None, dest='accumulation_steps_legacy',
                        help='Accumulation steps (legacy, use --accumulation-steps)')
    parser.add_argument('--focal_gamma', type=float, default=None, dest='focal_gamma_legacy',
                        help='Focal gamma (legacy, use --focal-gamma)')
    parser.add_argument('--lr_scheduler', type=str, default=None, dest='lr_scheduler_legacy',
                        help='LR scheduler (legacy, use --lr-scheduler)')
    parser.add_argument('--max_epochs', type=int, default=None, dest='max_epochs_legacy',
                        help='Max epochs (legacy, use --max-epochs)')
    parser.add_argument('--oversample_factor', type=int, default=None, dest='oversample_factor_legacy',
                        help='Oversample factor (legacy, use --oversample-factor)')
    parser.add_argument('--plateau_factor', type=float, default=None, dest='plateau_factor_legacy',
                        help='Plateau factor (legacy, use --plateau-factor)')
    parser.add_argument('--plateau_patience', type=int, default=None, dest='plateau_patience_legacy',
                        help='Plateau patience (legacy, use --plateau-patience)')
    parser.add_argument('--use_focal_loss', type=str, default=None, dest='use_focal_loss_legacy',
                        help='Use focal loss (legacy, use --use-focal-loss)')
    parser.add_argument('--warmup_epochs', type=int, default=None, dest='warmup_epochs_legacy',
                        help='Warmup epochs (legacy, use --warmup-epochs)')

    args = parser.parse_args()

    # Handle legacy arguments (prefer new hyphenated versions)
    if args.hidden_dim is None and args.hidden_dim_legacy is not None:
        args.hidden_dim = args.hidden_dim_legacy
    if args.num_layers is None and args.num_layers_legacy is not None:
        args.num_layers = args.num_layers_legacy
    if args.num_heads is None and args.num_heads_legacy is not None:
        args.num_heads = args.num_heads_legacy
    if args.learning_rate is None and args.learning_rate_legacy is not None:
        args.learning_rate = args.learning_rate_legacy
    if args.batch_size is None and args.batch_size_legacy is not None:
        args.batch_size = args.batch_size_legacy
    if args.weight_decay is None and args.weight_decay_legacy is not None:
        args.weight_decay = args.weight_decay_legacy
    if args.command_weight is None and args.command_weight_legacy is not None:
        args.command_weight = args.command_weight_legacy
    if args.operation_weight is None and args.operation_weight_legacy is not None:
        args.operation_weight = args.operation_weight_legacy
    if args.param_type_weight is None and args.param_type_weight_legacy is not None:
        args.param_type_weight = args.param_type_weight_legacy
    if args.param_value_weight is None and args.param_value_weight_legacy is not None:
        args.param_value_weight = args.param_value_weight_legacy
    if args.grammar_weight is None and args.grammar_weight_legacy is not None:
        args.grammar_weight = args.grammar_weight_legacy
    if args.label_smoothing is None and args.label_smoothing_legacy is not None:
        args.label_smoothing = args.label_smoothing_legacy
    if not args.augmentation and args.use_augmentation_legacy:
        args.augmentation = True

    # Handle additional legacy arguments for wandb sweep compatibility
    if args.accumulation_steps is None and args.accumulation_steps_legacy is not None:
        args.accumulation_steps = args.accumulation_steps_legacy
    if args.focal_gamma is None and args.focal_gamma_legacy is not None:
        args.focal_gamma = args.focal_gamma_legacy
    if args.lr_scheduler is None and args.lr_scheduler_legacy is not None:
        args.lr_scheduler = args.lr_scheduler_legacy
    if args.max_epochs is None and args.max_epochs_legacy is not None:
        args.max_epochs = args.max_epochs_legacy
    if args.oversample_factor is None and args.oversample_factor_legacy is not None:
        args.oversample_factor = args.oversample_factor_legacy
    if args.plateau_factor is None and args.plateau_factor_legacy is not None:
        args.plateau_factor = args.plateau_factor_legacy
    if args.plateau_patience is None and args.plateau_patience_legacy is not None:
        args.plateau_patience = args.plateau_patience_legacy
    if args.use_focal_loss_legacy is not None:
        # Handle string "True"/"False" from wandb sweep
        args.use_focal_loss = str(args.use_focal_loss_legacy).lower() in ('true', '1', 'yes')
    if args.warmup_epochs is None and args.warmup_epochs_legacy is not None:
        args.warmup_epochs = args.warmup_epochs_legacy

    # Validate arguments
    def validate_args(args):
        """Validate training arguments before starting."""
        errors = []

        # Check required paths exist
        config_path = Path(args.config)
        if not config_path.exists():
            errors.append(f"Config file not found: {args.config}")

        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            errors.append(f"Data directory not found: {args.data_dir}")

        vocab_path = Path(args.vocab_path)
        if not vocab_path.exists():
            errors.append(f"Vocabulary file not found: {args.vocab_path}")

        # Validate hyperparameter ranges
        if args.learning_rate is not None and args.learning_rate <= 0:
            errors.append(f"Learning rate must be positive, got: {args.learning_rate}")

        if args.batch_size is not None and args.batch_size <= 0:
            errors.append(f"Batch size must be positive, got: {args.batch_size}")

        if args.hidden_dim is not None and args.hidden_dim <= 0:
            errors.append(f"Hidden dimension must be positive, got: {args.hidden_dim}")

        if args.num_layers is not None and args.num_layers <= 0:
            errors.append(f"Number of layers must be positive, got: {args.num_layers}")

        if args.num_heads is not None and args.num_heads <= 0:
            errors.append(f"Number of heads must be positive, got: {args.num_heads}")

        # Validate hidden_dim / num_heads divisibility
        if args.hidden_dim is not None and args.num_heads is not None:
            if args.hidden_dim % args.num_heads != 0:
                errors.append(f"hidden_dim ({args.hidden_dim}) must be divisible by num_heads ({args.num_heads})")

        # Print errors and exit if any found
        if errors:
            print("❌ Validation errors found:")
            for error in errors:
                print(f"   - {error}")
            sys.exit(1)

    validate_args(args)

    # Create output directory
    output_dir = Path(args.output_dir)
    if args.use_wandb:
        import wandb
        if wandb.run is not None and wandb.run.id:
            if wandb.run.sweep_id is not None:
                output_dir = output_dir / wandb.run.id
                print(f"Sweep detected - checkpoints will be saved to: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ENHANCED TRAINING WITH MULTI-HEAD ARCHITECTURE")
    print("=" * 80)

    # Load config
    print(f"\nLoading config from: {args.config}")
    config = load_config(Path(args.config))

    # Override config with command-line arguments
    if args.hidden_dim is not None:
        config['hidden_dim'] = args.hidden_dim
    if args.num_layers is not None:
        config['num_layers'] = args.num_layers
    if args.num_heads is not None:
        config['num_heads'] = args.num_heads
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.weight_decay is not None:
        config['weight_decay'] = args.weight_decay
    if args.optimizer != 'adamw':
        config['optimizer'] = args.optimizer

    # Validate hidden_dim / num_heads divisibility
    if config['hidden_dim'] % config['num_heads'] != 0:
        raise ValueError(
            f"❌ Configuration Error: hidden_dim ({config['hidden_dim']}) must be divisible by "
            f"num_heads ({config['num_heads']})"
        )

    # Store weights and other parameters
    command_weight = args.command_weight if args.command_weight is not None else 3.0
    grad_clip = args.grad_clip_override if args.grad_clip_override is not None else config.get('grad_clip', args.grad_clip)
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else config.get('label_smoothing', 0.0)
    operation_weight = args.operation_weight if args.operation_weight is not None else 2.0
    param_type_weight = args.param_type_weight if args.param_type_weight is not None else 1.0
    param_value_weight = args.param_value_weight if args.param_value_weight is not None else 1.0
    grammar_weight = args.grammar_weight if args.grammar_weight is not None else 0.1
    dropout = args.dropout if args.dropout is not None else 0.1

    # Load class weights if provided
    command_class_weights = None
    param_type_class_weights = None
    param_value_class_weights = None
    operation_type_class_weights = None

    if args.class_weights_path:
        print(f"\n📊 Loading class weights from: {args.class_weights_path}")
        with open(args.class_weights_path, 'r') as f:
            class_weights_data = json.load(f)

        command_class_weights = torch.tensor(class_weights_data['command_weights'], dtype=torch.float32)
        param_type_class_weights = torch.tensor(class_weights_data['param_type_weights'], dtype=torch.float32)
        param_value_class_weights = torch.tensor(class_weights_data['param_value_weights'], dtype=torch.float32)

        # Load operation_type weights if available
        if 'operation_type_weights' in class_weights_data:
            operation_type_class_weights = torch.tensor(class_weights_data['operation_type_weights'], dtype=torch.float32)
            print(f"   ✅ operation_type class weights loaded ({len(class_weights_data['operation_type_weights'])} classes)")

    print(f"\n✅ Config loaded:")
    print(f"   Model: hidden_dim={config['hidden_dim']}, layers={config['num_layers']}, heads={config['num_heads']}")
    print(f"   Training: batch_size={config['batch_size']}, lr={config['learning_rate']}, optimizer={config['optimizer']}")
    print(f"   Augmentation: {args.augmentation}")
    print(f"   Loss weights: cmd={command_weight}, op={operation_weight}, grammar={grammar_weight}")
    print(f"   Advanced: focal_loss={args.use_focal_loss}, huber_loss={args.use_huber_loss}")
    print(f"   Regularization: dropout={dropout}, grad_clip={grad_clip}, grad_penalty={args.gradient_penalty}")

    # Setup device
    device = get_device()
    print()
    print_device_info(device)

    # Move class weights to device
    if command_class_weights is not None:
        command_class_weights = command_class_weights.to(device)
        param_type_class_weights = param_type_class_weights.to(device)
        param_value_class_weights = param_value_class_weights.to(device)
        if operation_type_class_weights is not None:
            operation_type_class_weights = operation_type_class_weights.to(device)
        print(f"✅ Class weights moved to {device}")

    # Load token decomposer
    print(f"\nLoading token decomposer from: {args.vocab_path}")
    decomposer = TokenDecomposer(args.vocab_path)
    print(f"✅ Token decomposer loaded")
    print(f"   Commands: {decomposer.n_commands}")
    print(f"   Param types: {decomposer.n_param_types}")
    print(f"   Param values: {decomposer.n_param_values}")
    print(f"   Vocab size: {len(decomposer.vocab)}")

    # Initialize grammar constraints
    print(f"\nInitializing grammar constraints...")
    grammar_constraints = GCodeGrammarConstraints(decomposer.vocab, device=device)
    print(f"✅ Grammar constraints initialized on {device}")

    # Initialize W&B
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                **config,
                'multi_head': True,
                'augmentation': args.augmentation,
                'oversample_factor': args.oversample_factor if args.augmentation else 1,
                'vocab_size': len(decomposer.vocab),
                'command_weight': command_weight,
                'operation_weight': operation_weight,
                'grammar_weight': grammar_weight,
                'label_smoothing': label_smoothing,
                'grad_clip': grad_clip,
                'use_class_weights': args.class_weights_path is not None,
                'lr_scheduler': args.lr_scheduler,
                'warmup_epochs': args.warmup_epochs,
                'beta1': args.beta1,
                'beta2': args.beta2,
                'accumulation_steps': args.accumulation_steps,
                'effective_batch_size': config['batch_size'] * args.accumulation_steps,
                'use_focal_loss': args.use_focal_loss,
                'use_huber_loss': args.use_huber_loss,
                'gradient_penalty': args.gradient_penalty,
                'noise_std': args.noise_std,
                'mixup_alpha': args.mixup_alpha,
            }
        )
        print("✅ W&B initialized")

    # Load datasets
    data_dir = Path(args.data_dir)
    print(f"\nLoading datasets from: {data_dir}")

    train_base = GCodeDataset(data_dir / 'train_sequences.npz')
    val_base = GCodeDataset(data_dir / 'val_sequences.npz')

    if args.augmentation:
        rare_token_ids = get_rare_token_ids(args.vocab_path)
        print(f"✅ Found {len(rare_token_ids)} rare token IDs")

        train_dataset = AugmentedGCodeDataset(
            base_dataset=train_base,
            oversample_rare=True,
            oversample_factor=args.oversample_factor,
            rare_token_ids=rare_token_ids,
            augment=True,
        )
    else:
        train_dataset = train_base

    val_dataset = val_base

    # Compute dataset dimensions
    n_continuous = train_base.continuous.size(-1)
    n_categorical = train_base.categorical.size(-1)
    vocab_size = len(decomposer.vocab)

    print(f"✅ Datasets loaded:")
    print(f"   Train: {len(train_dataset)} sequences")
    print(f"   Val: {len(val_dataset)} sequences")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Continuous features: {n_continuous}")
    print(f"   Categorical features: {n_categorical}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Create backbone model
    print("\nCreating backbone model...")
    backbone_config = ModelConfig(
        sensor_dims=[n_continuous, n_categorical],
        d_model=config['hidden_dim'],
        lstm_layers=config['num_layers'],
        gcode_vocab=vocab_size,
        n_heads=config['num_heads'],
        dropout=dropout,
    )

    backbone = MM_DTAE_LSTM(backbone_config).to(device)

    # Create multi-head LM
    print("Creating multi-head language model...")
    multihead_lm = MultiHeadGCodeLM(
        d_model=config['hidden_dim'],
        n_commands=decomposer.n_commands,
        n_param_types=decomposer.n_param_types,
        n_param_values=decomposer.n_param_values,
        nhead=config['num_heads'],
        num_layers=config['num_layers'],
        vocab_size=vocab_size,
    ).to(device)

    # Apply custom initialization if specified
    if args.init_strategy != 'xavier_uniform':
        print(f"Applying {args.init_strategy} initialization with gain={args.init_gain}")
        for model in [backbone, multihead_lm]:
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    if args.init_strategy == 'xavier_normal':
                        nn.init.xavier_normal_(param, gain=args.init_gain)
                    elif args.init_strategy == 'kaiming_uniform':
                        nn.init.kaiming_uniform_(param, nonlinearity='relu')
                    elif args.init_strategy == 'kaiming_normal':
                        nn.init.kaiming_normal_(param, nonlinearity='relu')
                    elif args.init_strategy == 'orthogonal':
                        nn.init.orthogonal_(param, gain=args.init_gain)

    print(f"✅ Models created:")
    print(f"   Backbone parameters: {sum(p.numel() for p in backbone.parameters()):,}")
    print(f"   Multi-head LM parameters: {sum(p.numel() for p in multihead_lm.parameters()):,}")
    print(f"   Total parameters: {sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in multihead_lm.parameters()):,}")

    # Create loss function (with focal loss for class imbalance)
    use_focal = getattr(args, 'use_focal_loss', False)
    focal_gamma = getattr(args, 'focal_gamma', 2.0)

    if use_focal:
        print(f"✅ Using FOCAL LOSS with gamma={focal_gamma}")

    loss_fn = MultiHeadGCodeLoss(
        pad_token_id=0,
        type_weight=1.0,
        command_weight=command_weight,
        param_type_weight=param_type_weight,
        param_value_weight=param_value_weight,
        operation_weight=operation_weight,
        label_smoothing=label_smoothing,
        command_class_weights=command_class_weights,
        param_type_class_weights=param_type_class_weights,
        param_value_class_weights=param_value_class_weights,
        use_focal_loss=use_focal,
        focal_gamma=focal_gamma,
    )

    # Create optimizer
    all_params = list(backbone.parameters()) + list(multihead_lm.parameters())

    if config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            all_params,
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01),
            betas=(args.beta1, args.beta2),
        )
    elif config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            all_params,
            lr=config['learning_rate'],
            betas=(args.beta1, args.beta2),
        )
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            all_params,
            lr=config['learning_rate'],
            momentum=args.momentum,
            weight_decay=config.get('weight_decay', 0.01),
        )
    elif config['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            all_params,
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01),
        )

    print(f"✅ Optimizer: {config['optimizer']}")
    print(f"✅ Learning rate: {config['learning_rate']}")

    # Create learning rate scheduler
    scheduler = None
    if args.lr_scheduler == 'cosine':
        t_max = args.cosine_t_max if args.cosine_t_max else (args.max_epochs - args.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=config['learning_rate'] * 0.01,
        )
        print(f"✅ LR Scheduler: CosineAnnealingLR (T_max={t_max})")
    elif args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=args.plateau_factor,
            patience=args.plateau_patience,
        )
        print(f"✅ LR Scheduler: ReduceLROnPlateau (patience={args.plateau_patience}, factor={args.plateau_factor})")
    elif args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, args.max_epochs // 4),
            gamma=0.5,
        )
        print(f"✅ LR Scheduler: StepLR")
    elif args.lr_scheduler == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config['learning_rate'] * 0.1,
            max_lr=config['learning_rate'],
            step_size_up=len(train_loader) * 2,
            mode='triangular2',
        )
        print(f"✅ LR Scheduler: CyclicLR")
    elif args.lr_scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'] * 10,
            epochs=args.max_epochs,
            steps_per_epoch=len(train_loader),
        )
        print(f"✅ LR Scheduler: OneCycleLR")

    if args.warmup_epochs > 0:
        print(f"✅ Warmup: {args.warmup_epochs} epochs")

    if args.accumulation_steps > 1:
        print(f"✅ Gradient Accumulation: {args.accumulation_steps} steps")

    # SWA setup
    swa_model = None
    swa_scheduler = None
    if args.use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        print(f"✅ SWA enabled, starting at epoch {args.swa_start_epoch}")
        # SWA will be initialized when reaching swa_start_epoch

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING ENHANCED TRAINING")
    print("=" * 80)
    print()

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = -float('inf')
    best_cmd_acc = 0.0
    patience_counter = 0
    checkpoint_saved = False

    if args.resume:
        if Path(args.resume).exists():
            print(f"🔄 Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)

            backbone.load_state_dict(checkpoint['backbone_state_dict'])
            multihead_lm.load_state_dict(checkpoint['multihead_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            start_epoch = checkpoint.get('epoch', 0)
            best_val_acc = checkpoint.get('val_acc', -float('inf'))
            checkpoint_saved = True

            print(f"✓ Resumed from epoch {start_epoch}")
            print(f"✓ Best validation metric so far: {best_val_acc:.2%}")
            print()

    for epoch in range(start_epoch, args.max_epochs):
        print("=" * 80)
        print(f"Epoch {epoch + 1}/{args.max_epochs}")
        print("=" * 80)

        # Apply warmup schedule
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            warmup_factor = (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['learning_rate'] * warmup_factor
            print(f"🔥 Warmup: LR = {optimizer.param_groups[0]['lr']:.2e}")

        # Initialize SWA at specified epoch
        if args.use_swa and epoch == args.swa_start_epoch and swa_model is None:
            from torch.optim.swa_utils import AveragedModel, SWALR
            swa_model = AveragedModel(multihead_lm)
            swa_scheduler = SWALR(optimizer, swa_lr=config['learning_rate'] * 0.05)
            print("🔄 SWA initialized")

        # Train
        train_metrics = train_epoch_multihead(
            multihead_lm, backbone, decomposer, train_loader,
            optimizer, loss_fn, device, grad_clip,
            grammar_constraints=grammar_constraints,
            grammar_weight=grammar_weight,
            accumulation_steps=args.accumulation_steps,
            gradient_penalty=args.gradient_penalty
        )

        # Update SWA model
        if swa_model is not None and epoch >= args.swa_start_epoch:
            swa_model.update_parameters(multihead_lm)
            swa_scheduler.step()

        # Validate
        val_metrics = validate_multihead(
            multihead_lm, backbone, decomposer, val_loader,
            loss_fn, device, vocab_size
        )

        # Print comprehensive metrics
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch + 1}/{args.max_epochs}")
        print(f"{'='*80}")

        # Training metrics
        train_comp = train_metrics.get('composite_acc', 0)
        print(f"TRAIN: Loss={train_metrics['loss']:.4f} | "
              f"Cmd={train_metrics['command_acc']:.2%} | "
              f"ParamType={train_metrics['param_type_acc']:.2%} | "
              f"Composite={train_comp:.4f}")

        # Validation metrics
        val_comp = val_metrics.get('composite_acc', 0)
        print(f"VAL:   Loss={val_metrics['loss']:.4f} | "
              f"Cmd={val_metrics['command_acc']:.2%} | "
              f"ParamType={val_metrics['param_type_acc']:.2%} | "
              f"Composite={val_comp:.4f}")

        # Show regression metrics if available
        if 'param_value_mae' in val_metrics:
            print(f"       ParamMAE={val_metrics['param_value_mae']:.4f} | "
                  f"ParamTolAcc={val_metrics.get('param_value_tolerance_acc', 0):.2%}")

        # Print enhanced per-class statistics
        print_enhanced_epoch_stats(train_metrics, val_metrics, decomposer, epoch, args.max_epochs)

        # Memory monitoring every 10 epochs (detect potential leaks in long runs)
        if epoch % 10 == 0:
            mem_info = psutil.Process().memory_info()
            mem_gb = mem_info.rss / (1024 ** 3)
            print(f"\n💾 Memory usage: {mem_gb:.2f} GB")
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({'system/memory_gb': mem_gb, 'epoch': epoch + 1})

        # Track metric for early stopping
        # Use --track-metric argument to select (default: operation_acc for 9-class classification)
        metric_to_track = args.track_metric
        if metric_to_track in val_metrics:
            current_metric = val_metrics[metric_to_track]
        else:
            # Fallback chain: operation_acc > composite_acc > param_type_acc
            if 'operation_acc' in val_metrics:
                metric_to_track = 'operation_acc'
                current_metric = val_metrics[metric_to_track]
            elif 'composite_acc' in val_metrics:
                metric_to_track = 'composite_acc'
                current_metric = val_metrics[metric_to_track]
            else:
                metric_to_track = 'param_type_acc'
                current_metric = val_metrics[metric_to_track]
            print(f"⚠️  Requested metric '{args.track_metric}' not found, using '{metric_to_track}'")

        # Step scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if epoch >= args.warmup_epochs and scheduler is not None:
            if args.lr_scheduler == 'plateau':
                scheduler.step(current_metric)
            elif args.lr_scheduler in ['cyclic', 'onecycle']:
                pass  # These step per batch, not per epoch
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

        # Log to W&B
        if args.use_wandb and WANDB_AVAILABLE:
            # Filter out non-loggable items (dicts) from metrics
            train_loggable = {k: v for k, v in train_metrics.items()
                              if not isinstance(v, dict)}
            val_loggable = {k: v for k, v in val_metrics.items()
                            if not isinstance(v, dict)}

            # Build log dict with standard metrics
            log_dict = {
                'epoch': epoch + 1,
                **{f'train/{k}': v for k, v in train_loggable.items()},
                **{f'val/{k}': v for k, v in val_loggable.items()},
                'learning_rate': current_lr,
            }

            # Add per-class recall metrics for key commands
            if 'command_stats' in val_metrics:
                cmd_stats = val_metrics['command_stats']
                for cmd_name in ['G0', 'G1', 'G2', 'G3']:
                    if cmd_name in decomposer.command2id:
                        cmd_id = decomposer.command2id[cmd_name]
                        target_count = cmd_stats['target_counts'].get(cmd_id, 0)
                        correct = cmd_stats['correct_counts'].get(cmd_id, 0)
                        recall = correct / max(target_count, 1)
                        log_dict[f'val/recall_{cmd_name}'] = recall

            # Add per-param type recall for F parameter
            if 'param_type_stats' in val_metrics:
                pt_stats = val_metrics['param_type_stats']
                if 'F' in decomposer.param2id:
                    param_id = decomposer.param2id['F']
                    target_count = pt_stats['target_counts'].get(param_id, 0)
                    correct = pt_stats['correct_counts'].get(param_id, 0)
                    recall = correct / max(target_count, 1)
                    log_dict['val/recall_F'] = recall

            wandb.log(log_dict)

        # Early stopping
        if current_metric > best_val_acc:
            best_val_acc = current_metric
            patience_counter = 0

            # Save checkpoint
            checkpoint_path = output_dir / 'checkpoint_best.pt'
            torch.save({
                'epoch': epoch + 1,
                'backbone_state_dict': backbone.state_dict(),
                'multihead_state_dict': multihead_lm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config,
                'metric_tracked': metric_to_track,
            }, checkpoint_path)

            # Validate checkpoint was saved correctly
            if checkpoint_path.exists():
                ckpt_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                if ckpt_size_mb < 0.1:
                    print(f"\n⚠️  WARNING: Checkpoint file suspiciously small ({ckpt_size_mb:.2f} MB)")
                else:
                    print(f"\n✅ Checkpoint saved: {ckpt_size_mb:.1f} MB")
            else:
                print(f"\n❌ ERROR: Checkpoint file was not created!")

            checkpoint_saved = True
            is_best = True
            print(f"✅ NEW BEST {metric_to_track.upper()}: {best_val_acc:.4f} "
                  f"(improved from {best_val_acc:.4f}) | LR={current_lr:.2e}")
        else:
            patience_counter += 1
            is_best = False
            print(f"\n📊 Tracking {metric_to_track}: {current_metric:.4f} "
                  f"(best: {best_val_acc:.4f}) | "
                  f"Patience: {patience_counter}/{args.patience} | "
                  f"LR={current_lr:.2e}")

        # Print example predictions every 5 epochs or when best metric improves (reduces overhead)
        if is_best or epoch % 5 == 0 or epoch == 0:
            generate_example_prediction(
                multihead_lm, backbone, decomposer, val_loader, device,
                vocab=decomposer.vocab, num_examples=2
            )

        if patience_counter >= args.patience:
            print(f"\n⏹️  EARLY STOPPING: No improvement in {metric_to_track} for {args.patience} epochs")
            print(f"    Best {metric_to_track}: {best_val_acc:.4f}")
            break

    # Training complete
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    # Finalize SWA if used
    if args.use_swa and swa_model is not None:
        print("\n🔄 Finalizing SWA model...")
        # Update batch normalization statistics for the SWA model
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

        # Save the SWA model checkpoint
        swa_checkpoint_path = output_dir / 'checkpoint_swa.pt'
        torch.save({
            'epoch': args.max_epochs,
            'backbone_state_dict': backbone.state_dict(),
            'multihead_state_dict': swa_model.module.state_dict(),  # Get the averaged model
            'config': config,
            'swa_model': True,
        }, swa_checkpoint_path)
        print(f"✅ SWA model saved to: {swa_checkpoint_path}")

    if checkpoint_saved:
        print(f"Best {metric_to_track}: {best_val_acc:.4f}")
        print(f"Checkpoint saved to: {output_dir / 'checkpoint_best.pt'}")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    print("\n✅ Enhanced training completed successfully!")


if __name__ == '__main__':
    main()