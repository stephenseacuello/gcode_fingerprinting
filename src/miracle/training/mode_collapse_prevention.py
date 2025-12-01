#!/usr/bin/env python3
"""
Mode Collapse Prevention Utilities

This module provides tools to detect and prevent mode collapse in training:
- Class frequency analysis
- Prediction diversity monitoring (entropy, per-class accuracy)
- Class weighting computation
- Focal loss implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
from pathlib import Path
import json


def compute_class_frequencies(dataloader, decomposer) -> Dict[str, torch.Tensor]:
    """
    Analyze class frequencies in the dataset.

    Returns:
        Dict with 'command', 'param_type', 'param_value' frequencies as tensors
    """
    command_counts = Counter()
    param_type_counts = Counter()
    param_value_counts = Counter()

    print("\n" + "=" * 80)
    print("ANALYZING CLASS FREQUENCIES")
    print("=" * 80)

    total_tokens = 0

    for batch in dataloader:
        # Handle dictionary batch format from collate_fn
        tokens = batch['tokens']  # [B, T]
        B, T = tokens.shape

        # Decompose all tokens
        decomposed = decomposer.decompose_batch(tokens)

        # Count occurrences
        command_ids = decomposed['command_id'].cpu().numpy().flatten()
        param_type_ids = decomposed['param_type_id'].cpu().numpy().flatten()
        param_value_ids = decomposed['param_value_id'].cpu().numpy().flatten()

        for cmd_id in command_ids:
            if cmd_id >= 0:  # Ignore padding
                command_counts[int(cmd_id)] += 1
                total_tokens += 1

        for pt_id in param_type_ids:
            if pt_id >= 0:
                param_type_counts[int(pt_id)] += 1

        for pv_id in param_value_ids:
            if pv_id >= 0:
                param_value_counts[int(pv_id)] += 1

    print(f"\nTotal tokens analyzed: {total_tokens:,}")

    # Convert to tensors
    n_commands = decomposer.n_commands
    n_param_types = decomposer.n_param_types
    n_param_values = decomposer.n_param_values

    command_freq = torch.zeros(n_commands)
    param_type_freq = torch.zeros(n_param_types)
    param_value_freq = torch.zeros(n_param_values)

    for cmd_id, count in command_counts.items():
        if cmd_id < n_commands:
            command_freq[cmd_id] = count

    for pt_id, count in param_type_counts.items():
        if pt_id < n_param_types:
            param_type_freq[pt_id] = count

    for pv_id, count in param_value_counts.items():
        if pv_id < n_param_values:
            param_value_freq[pv_id] = count

    # Normalize to probabilities
    command_freq = command_freq / (command_freq.sum() + 1e-8)
    param_type_freq = param_type_freq / (param_type_freq.sum() + 1e-8)
    param_value_freq = param_value_freq / (param_value_freq.sum() + 1e-8)

    # Print top classes
    print("\n📊 Class Distribution Analysis:")
    print("\nTop 10 Commands:")
    top_commands = torch.argsort(command_freq, descending=True)[:10]
    for rank, cmd_id in enumerate(top_commands, 1):
        freq = command_freq[cmd_id].item()
        if freq > 0:
            cmd_name = decomposer.command_tokens[cmd_id] if cmd_id < len(decomposer.command_tokens) else f"ID_{cmd_id}"
            print(f"  {rank}. {cmd_name}: {freq:.2%}")

    print("\nTop 5 Parameter Types:")
    top_param_types = torch.argsort(param_type_freq, descending=True)[:5]
    for rank, pt_id in enumerate(top_param_types, 1):
        freq = param_type_freq[pt_id].item()
        if freq > 0:
            pt_name = decomposer.param_tokens[pt_id] if pt_id < len(decomposer.param_tokens) else f"ID_{pt_id}"
            print(f"  {rank}. {pt_name}: {freq:.2%}")

    frequencies = {
        'command': command_freq,
        'param_type': param_type_freq,
        'param_value': param_value_freq,
    }

    print("=" * 80 + "\n")

    return frequencies


def compute_class_weights(frequencies: Dict[str, torch.Tensor],
                         method: str = 'inverse',
                         smooth: float = 0.1) -> Dict[str, torch.Tensor]:
    """
    Compute class weights to address imbalance.

    Args:
        frequencies: Dict of class frequencies (probabilities)
        method: 'inverse' or 'sqrt_inverse'
        smooth: Smoothing factor to prevent extreme weights

    Returns:
        Dict of class weights as tensors
    """
    weights = {}

    for key, freq in frequencies.items():
        if method == 'inverse':
            # Inverse frequency: w_i = 1 / freq_i
            weight = 1.0 / (freq + smooth)
        elif method == 'sqrt_inverse':
            # Square root inverse frequency (less aggressive)
            weight = 1.0 / torch.sqrt(freq + smooth)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize weights to sum to num_classes
        weight = weight / weight.mean()

        weights[key] = weight

    print(f"\n📊 Class Weights ({method}):")
    for key, weight in weights.items():
        top_5 = torch.argsort(weight, descending=True)[:5]
        print(f"\n  {key} - Top 5 weights:")
        for idx in top_5:
            print(f"    Class {idx}: {weight[idx]:.3f}")

    return weights


def save_class_weights(weights: Dict[str, torch.Tensor], output_path: Path):
    """Save class weights to JSON file."""
    # Map keys to match training script expectations
    key_mapping = {
        'command': 'command_weights',
        'param_type': 'param_type_weights',
        'param_value': 'param_value_weights'
    }

    weights_dict = {
        key_mapping.get(key, key): weight.tolist()
        for key, weight in weights.items()
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(weights_dict, f, indent=2)

    print(f"\n✓ Saved class weights to: {output_path}")


def load_class_weights(weights_path: Path, device) -> Dict[str, torch.Tensor]:
    """Load class weights from JSON file."""
    with open(weights_path, 'r') as f:
        weights_dict = json.load(f)

    weights = {
        key: torch.tensor(weight, device=device, dtype=torch.float32)
        for key, weight in weights_dict.items()
    }

    print(f"✓ Loaded class weights from: {weights_path}")
    return weights


def compute_prediction_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute entropy of prediction distribution.
    High entropy = diverse predictions
    Low entropy = mode collapse

    Args:
        logits: [B, T, C] logits
        dim: Dimension to compute entropy over

    Returns:
        Entropy values [B, T]
    """
    probs = F.softmax(logits, dim=dim)
    log_probs = F.log_softmax(logits, dim=dim)
    entropy = -(probs * log_probs).sum(dim=dim)
    return entropy


def compute_per_class_accuracy(predictions: torch.Tensor,
                               targets: torch.Tensor,
                               num_classes: int) -> Dict[int, float]:
    """
    Compute accuracy for each class separately.

    Args:
        predictions: [N] predicted class IDs
        targets: [N] target class IDs
        num_classes: Total number of classes

    Returns:
        Dict mapping class_id -> accuracy
    """
    per_class_acc = {}

    for class_id in range(num_classes):
        mask = (targets == class_id)
        if mask.sum() > 0:
            correct = (predictions[mask] == targets[mask]).float().sum()
            total = mask.sum()
            per_class_acc[class_id] = (correct / total).item()
        else:
            per_class_acc[class_id] = 0.0

    return per_class_acc


def compute_diversity_metrics(command_logits: torch.Tensor,
                              command_targets: torch.Tensor,
                              decomposer,
                              top_k: int = 10) -> Dict[str, float]:
    """
    Compute metrics to detect mode collapse.

    Returns:
        Dict with:
        - pred_entropy: Average prediction entropy
        - pred_diversity: Number of unique predicted classes
        - top_k_acc: Per-class accuracy for top K most frequent commands
    """
    B, T, C = command_logits.shape

    # Compute entropy
    entropy = compute_prediction_entropy(command_logits, dim=-1)
    avg_entropy = entropy.mean().item()
    max_entropy = np.log(C)  # Maximum possible entropy

    # Compute diversity (unique predictions)
    predictions = command_logits.argmax(dim=-1).flatten()
    unique_preds = torch.unique(predictions).numel()

    # Per-class accuracy for top K commands
    per_class_acc = compute_per_class_accuracy(
        predictions.flatten(),
        command_targets.flatten(),
        C
    )

    # Get top K most frequent commands
    top_k_classes = list(range(min(top_k, C)))
    top_k_acc = {
        f"cmd_{i}_acc": per_class_acc.get(i, 0.0)
        for i in top_k_classes
    }

    metrics = {
        'pred_entropy': avg_entropy,
        'pred_entropy_ratio': avg_entropy / max_entropy,  # 0-1, closer to 1 is better
        'pred_diversity': unique_preds,
        'pred_diversity_ratio': unique_preds / C,  # 0-1, closer to 1 is better
        **top_k_acc,
    }

    return metrics


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Where:
    - p_t is the model's estimated probability for the true class
    - α_t is the class weight
    - γ (gamma) is the focusing parameter (higher = more focus on hard examples)

    References:
        https://arxiv.org/abs/1708.02002
    """

    def __init__(self,
                 num_classes: int,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        """
        Args:
            num_classes: Number of classes
            alpha: Class weights [C]. If None, all classes have equal weight
            gamma: Focusing parameter (default: 2.0)
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Label smoothing factor (0-1)
        """
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.register_buffer('alpha', torch.ones(num_classes))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, C] logits
            targets: [N] target class IDs (long)

        Returns:
            Focal loss value
        """
        # Get probabilities
        p = F.softmax(inputs, dim=-1)  # [N, C]

        # Get class probabilities
        ce = F.cross_entropy(
            inputs, targets,
            weight=None,  # We'll apply alpha manually
            label_smoothing=self.label_smoothing,
            reduction='none'
        )  # [N]

        # Get true class probabilities
        p_t = p.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)  # [N]

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha (class weights)
        if self.alpha is not None:
            alpha_t = self.alpha.gather(dim=0, index=targets)  # [N]
            focal_weight = alpha_t * focal_weight

        # Focal loss
        loss = focal_weight * ce  # [N]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def print_mode_collapse_warning(metrics: Dict[str, float], threshold: float = 0.3):
    """
    Print warning if mode collapse is detected.

    Args:
        metrics: Diversity metrics from compute_diversity_metrics
        threshold: Threshold for diversity ratio (below this = warning)
    """
    diversity_ratio = metrics.get('pred_diversity_ratio', 1.0)
    entropy_ratio = metrics.get('pred_entropy_ratio', 1.0)

    if diversity_ratio < threshold or entropy_ratio < 0.5:
        print("\n" + "=" * 80)
        print("⚠️  WARNING: POSSIBLE MODE COLLAPSE DETECTED")
        print("=" * 80)
        print(f"  Prediction Diversity: {diversity_ratio:.1%} (threshold: {threshold:.0%})")
        print(f"  Entropy Ratio: {entropy_ratio:.1%} (healthy: > 50%)")

        if diversity_ratio < 0.1:
            print("\n  🚨 CRITICAL: Model is predicting < 10% of possible classes!")
            print("  Recommendation: Stop training and adjust hyperparameters")
        elif diversity_ratio < threshold:
            print("\n  ⚠️  Model predictions are not diverse enough")
            print("  Recommendation: Consider class balancing or focal loss")

        # Check per-class accuracies
        cmd_0_acc = metrics.get('cmd_0_acc', 0.0)
        if cmd_0_acc > 0.9:
            print(f"\n  🚨 Class 0 (most common) accuracy: {cmd_0_acc:.1%}")
            print("  Model may be collapsing to most frequent class")

        print("=" * 80 + "\n")
