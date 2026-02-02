#!/usr/bin/env python3
"""
Train Sensor Multi-Head Decoder for G-code Generation.

This script trains the SensorMultiHeadDecoder which:
1. Uses frozen MM-DTAE-LSTM encoder (100% operation accuracy)
2. Operation-conditioned transformer decoder
3. Multi-head outputs (type, command, param_type, digits)

Features:
- Comprehensive command-line arguments
- Multiple optimizer choices (AdamW, Adam, SGD, RMSprop)
- Multiple LR schedulers (cosine, plateau, step, cyclic, onecycle)
- Curriculum learning (structure → coarse digits → full precision)
- Scheduled sampling (teacher forcing decay)
- Focal loss for class imbalance
- SWA (Stochastic Weight Averaging)
- Class-balanced sampling
- Gradient accumulation
- WandB logging

Author: Claude Code
Date: December 2025
"""

import os
import platform
import sys
import json
import math
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# Enable MPS fallback for Mac
if platform.system() == 'Darwin':
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, ReduceLROnPlateau, StepLR, CyclicLR, OneCycleLR
)
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Training without logging.")

from miracle.dataset.decoder_dataset import (
    DecoderDatasetFromSplits,
    decoder_collate_fn,
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
)
from miracle.dataset.data_augmentation import (
    DataAugmenter,
    AugmentedGCodeDataset,
    get_rare_token_ids,
)
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.model.digit_value_head import DigitByDigitLoss
from miracle.model.model import (
    EnhancedEncoder,
    MultiHeadAttentionPooling,
    MultiScaleTemporalEncoder,
    AuxiliarySupervisionHeads,
    compute_auxiliary_loss,
)
from miracle.training.losses import FocalLoss, PositionWeightedCrossEntropy, PositionWeightedFocalLoss
from miracle.training.grammar_constraints import GCodeGrammarConstraints
from miracle.training.metrics import compute_comprehensive_metrics


# ============================================================================
# MM-DTAE-LSTM Encoder (copied for loading)
# ============================================================================

class MM_DTAE_LSTM(nn.Module):
    """MM-DTAE-LSTM encoder for sensor feature extraction."""

    def __init__(
        self,
        input_dim: int = 155,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        n_classes: int = 9,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        noise_factor: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.noise_factor = noise_factor
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.encoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        encoder_output_dim = hidden_dim * self.num_directions
        self.bottleneck = nn.Sequential(
            nn.Linear(encoder_output_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.decoder_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, n_classes)
        )

        self.temporal_attention = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Softmax(dim=1)
        )

    def encode(self, x: torch.Tensor) -> tuple:
        """Encode input sequence to latent representation."""
        x = self.input_proj(x)
        encoded, (h_n, c_n) = self.encoder_lstm(x)
        latent = self.bottleneck(encoded)
        return latent, (h_n, c_n)

    def classify(self, latent: torch.Tensor) -> tuple:
        """Classify operation type from latent."""
        attn_weights = self.temporal_attention(latent)
        pooled = (latent * attn_weights).sum(dim=1)
        logits = self.classification_head(pooled)
        return logits, attn_weights


# ============================================================================
# Training Infrastructure
# ============================================================================

class CurriculumScheduler:
    """Curriculum learning: structure → coarse digits → full precision."""

    def __init__(
        self,
        n_phases: int = 3,
        epochs_per_phase: int = 30,
        structure_weight: float = 1.0,
        digit_weight_p2: float = 0.5,
        digit_weight_p3: float = 1.0,
        value_weight_p3: float = 0.5
    ):
        self.n_phases = n_phases
        self.epochs_per_phase = epochs_per_phase
        self.phase_names = ["Structure Only", "Coarse Digits", "Full Precision"]
        # Configurable weights
        self.structure_weight = structure_weight
        self.digit_weight_p2 = digit_weight_p2
        self.digit_weight_p3 = digit_weight_p3
        self.value_weight_p3 = value_weight_p3

    def get_phase(self, epoch: int) -> int:
        return min(epoch // self.epochs_per_phase, self.n_phases - 1)

    def get_loss_weights(self, epoch: int) -> dict:
        phase = self.get_phase(epoch)
        if phase == 0:
            return {'structure': self.structure_weight, 'digit': 0.0, 'value': 0.0}
        elif phase == 1:
            return {'structure': self.structure_weight, 'digit': self.digit_weight_p2, 'value': 0.0}
        else:
            return {'structure': self.structure_weight, 'digit': self.digit_weight_p3, 'value': self.value_weight_p3}

    def get_phase_info(self, epoch: int) -> str:
        phase = self.get_phase(epoch)
        return f"Phase {phase + 1}/{self.n_phases}: {self.phase_names[phase]}"


class ScheduledSampling:
    """Scheduled sampling for teacher forcing decay."""

    def __init__(
        self,
        start_ratio: float = 1.0,
        end_ratio: float = 0.5,
        total_epochs: int = 100,
        decay_type: str = 'cosine'
    ):
        self.start = start_ratio
        self.end = end_ratio
        self.total = total_epochs
        self.decay_type = decay_type

    def get_ratio(self, epoch: int) -> float:
        progress = min(epoch / max(self.total - 1, 1), 1.0)
        if self.decay_type == 'linear':
            return self.start - progress * (self.start - self.end)
        elif self.decay_type == 'exponential':
            decay = math.log(self.end / self.start) / max(self.total - 1, 1)
            return self.start * math.exp(decay * epoch)
        elif self.decay_type == 'cosine':
            return self.end + (self.start - self.end) * (1 + math.cos(math.pi * progress)) / 2
        return self.start


def create_optimizer(model, args, encoder=None):
    """Create optimizer based on args.

    Supports differential learning rates for encoder and decoder when
    encoder is unfrozen (args.unfreeze_encoder_layers != 0).

    Args:
        model: The decoder model
        args: Training arguments
        encoder: Optional encoder model (for unfreezing)

    Returns:
        Optimizer with appropriate parameter groups
    """
    param_groups = []

    # Decoder params (full learning rate)
    decoder_params = [p for p in model.parameters() if p.requires_grad]
    if decoder_params:
        param_groups.append({
            'params': decoder_params,
            'lr': args.learning_rate,
            'name': 'decoder'
        })

    # Encoder params (scaled learning rate) if unfreezing
    if encoder is not None and args.unfreeze_encoder_layers != 0:
        if hasattr(encoder, 'get_unfrozen_params'):
            # EnhancedEncoder with get_unfrozen_params method
            encoder_params = encoder.get_unfrozen_params(args.unfreeze_encoder_layers)
        else:
            # Standard encoder - unfreeze all or last N layers
            if args.unfreeze_encoder_layers == -1:
                encoder_params = list(encoder.parameters())
            else:
                # Unfreeze last N modules
                encoder_params = []
                modules_list = list(encoder.modules())
                # Get last N modules that have parameters
                param_modules = [m for m in modules_list
                                if any(p.requires_grad for p in m.parameters(recurse=False))]
                for module in param_modules[-args.unfreeze_encoder_layers:]:
                    encoder_params.extend([p for p in module.parameters() if p.requires_grad])

        if encoder_params:
            # Enable gradients for encoder params
            for p in encoder_params:
                p.requires_grad = True

            param_groups.append({
                'params': encoder_params,
                'lr': args.learning_rate * args.encoder_lr_scale,
                'name': 'encoder'
            })
            print(f"  Encoder unfreezing: {len(encoder_params)} params at LR={args.learning_rate * args.encoder_lr_scale:.2e}")

    if args.optimizer == 'adamw':
        return torch.optim.AdamW(
            param_groups, weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    elif args.optimizer == 'adam':
        return torch.optim.Adam(
            param_groups, betas=(args.beta1, args.beta2)
        )
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(
            param_groups, momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(
            param_groups, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def create_scheduler(optimizer, args, total_steps=None):
    """Create LR scheduler based on args."""
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    if args.lr_scheduler == 'none':
        return None
    elif args.lr_scheduler == 'cosine':
        t_max = args.cosine_t_max or max(args.max_epochs - args.warmup_epochs, 1)
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=getattr(args, 'min_lr', 1e-6))
    elif args.lr_scheduler == 'cosine_restarts':
        # Cosine annealing with warm restarts
        t_0 = getattr(args, 'restart_period', 30)
        t_mult = getattr(args, 'restart_mult', 1.0)
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=t_0, T_mult=int(t_mult),
            eta_min=getattr(args, 'min_lr', 1e-6)
        )
    elif args.lr_scheduler == 'plateau':
        return ReduceLROnPlateau(
            optimizer, mode='max', factor=args.plateau_factor,
            patience=args.plateau_patience
        )
    elif args.lr_scheduler == 'step':
        return StepLR(
            optimizer,
            step_size=args.step_lr_step_size,
            gamma=args.step_lr_gamma
        )
    elif args.lr_scheduler == 'cyclic':
        return CyclicLR(
            optimizer,
            base_lr=args.learning_rate / args.cyclic_base_lr_div,
            max_lr=args.learning_rate,
            cycle_momentum=False
        )
    elif args.lr_scheduler == 'onecycle':
        return OneCycleLR(
            optimizer, max_lr=args.learning_rate, total_steps=total_steps
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.lr_scheduler}")


def get_warmup_lr(epoch, warmup_epochs, base_lr, warmup_type='linear'):
    """Get learning rate during warmup phase."""
    if epoch >= warmup_epochs:
        return base_lr
    progress = epoch / max(warmup_epochs, 1)
    if warmup_type == 'cosine':
        # Cosine warmup: smoother transition
        return base_lr * (1 - math.cos(math.pi * progress)) / 2
    else:
        # Linear warmup
        return base_lr * progress


def get_progressive_augment_prob(epoch, max_epochs, start_prob, end_prob, ramp_epochs=None):
    """Get augmentation probability for progressive augmentation."""
    if ramp_epochs is None:
        ramp_epochs = int(0.7 * max_epochs)
    if epoch >= ramp_epochs:
        return end_prob
    progress = epoch / max(ramp_epochs, 1)
    return start_prob + progress * (end_prob - start_prob)


# ============================================================================
# G-code Decoding and Sample Prediction Display
# ============================================================================

OPERATION_NAMES = [
    "adaptive",        # 0
    "adaptive150025",  # 1
    "face",            # 2
    "face150025",      # 3
    "pocket",          # 4
    "pocket150025",    # 5
    "damageadaptive",  # 6
    "damageface",      # 7
    "damagepocket",    # 8
]


def tokens_to_gcode(token_ids, id2token):
    """Convert token IDs to human-readable G-code string."""
    parts = []
    for tid in token_ids:
        if tid in (PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID):
            continue
        token = id2token.get(tid, f"UNK{tid}")

        # Parse token format
        if token.startswith("NUM_"):
            # Format: NUM_X_1650 -> X1.650
            try:
                _, param, val_str = token.split("_", 2)
                # Handle negative values (e.g., NUM_Z_-043)
                if val_str.startswith("-"):
                    val = -int(val_str[1:]) / 1000.0
                else:
                    val = int(val_str) / 1000.0
                parts.append(f"{param}{val:.3f}")
            except (ValueError, IndexError):
                parts.append(token)
        elif token in ('X', 'Y', 'Z', 'R', 'F', 'I', 'J', 'K', 'A', 'B', 'C'):
            # Standalone param letter (shouldn't happen with hybrid vocab)
            parts.append(token)
        elif token.startswith("G") or token.startswith("M"):
            # Command
            parts.append(token)
        else:
            parts.append(token)

    return " ".join(parts)


def show_sample_predictions(
    encoder, decoder, val_loader, id2token, device,
    num_samples=3, max_tokens=10
):
    """Display sample G-code predictions vs ground truth."""
    encoder.eval()
    decoder.eval()

    # Get one batch
    batch = next(iter(val_loader))

    with torch.no_grad():
        # Use the same field names as validate()
        sensor_features = batch['sensor_features'].to(device)
        input_tokens = batch['input_tokens'].to(device)
        target_tokens = batch['target_tokens'].to(device)
        operation_type = batch.get('operation_type', torch.zeros(sensor_features.size(0), dtype=torch.long)).to(device)

        # Encoder forward
        # Note: EnhancedEncoder.encode() returns (pooled_features, memory_sequence)
        #       MM_DTAE_LSTM.encode() returns (latent_sequence, hidden_state)
        enc_out1, enc_out2 = encoder.encode(sensor_features)
        # EnhancedEncoder: out1 = pooled (2D), out2 = memory (3D) -> use out2
        # MM_DTAE_LSTM: out1 = sequence (3D), out2 = hidden -> use out1
        sensor_emb = enc_out2 if enc_out1.dim() == 2 else enc_out1
        classify_input = enc_out1 if enc_out1.dim() == 2 else enc_out1
        op_logits, _ = encoder.classify(classify_input)
        op_pred = op_logits.argmax(-1)

        # Decoder forward (teacher forcing)
        outputs = decoder(
            tokens=input_tokens,
            sensor_embeddings=sensor_emb,
            operation_type=op_pred,
        )

        # Get predicted tokens from legacy logits
        # Use raw logits for sample display (same as evaluation)
        if 'raw_legacy_logits' in outputs:
            pred_tokens = outputs['raw_legacy_logits'].argmax(-1)
        else:
            pred_tokens = outputs['legacy_logits'].argmax(-1)

        print(f"\n{'='*70}")
        print("SAMPLE PREDICTIONS (Teacher Forcing)")
        print(f"{'='*70}")

        for b in range(min(num_samples, input_tokens.size(0))):
            op_gt = operation_type[b].item()
            op_p = op_pred[b].item()
            op_name = OPERATION_NAMES[op_gt] if op_gt < len(OPERATION_NAMES) else f"Op{op_gt}"

            # Get ground truth and prediction tokens
            gt_toks = target_tokens[b, :max_tokens].cpu().tolist()
            pr_toks = pred_tokens[b, :max_tokens].cpu().tolist()

            # Convert to G-code strings
            gt_gcode = tokens_to_gcode(gt_toks, id2token)
            pr_gcode = tokens_to_gcode(pr_toks, id2token)

            # Token-level accuracy for this sample
            valid_mask = (target_tokens[b, :max_tokens] != PAD_TOKEN_ID)
            n_valid = valid_mask.sum().item()
            n_correct = ((pred_tokens[b, :max_tokens] == target_tokens[b, :max_tokens]) & valid_mask).sum().item()
            acc = n_correct / max(n_valid, 1) * 100

            print(f"\nSample {b+1} [{op_name}] (Op pred: {op_p}, gt: {op_gt})")
            print(f"  GT:   {gt_gcode}")
            print(f"  Pred: {pr_gcode}")
            print(f"  Token Acc: {n_correct}/{n_valid} = {acc:.1f}%")

            # Show token-by-token comparison
            comparison = []
            for i in range(min(max_tokens, n_valid)):
                match = "✓" if gt_toks[i] == pr_toks[i] else "✗"
                comparison.append(f"{match}")
            print(f"  Match: {' '.join(comparison)}")


def show_per_operation_accuracy(val_metrics, operation_counts):
    """Display per-operation accuracy breakdown."""
    print(f"\n  Per-Operation Accuracy:")
    for op_id in sorted(operation_counts.keys()):
        op_name = OPERATION_NAMES[op_id] if op_id < len(OPERATION_NAMES) else f"Op{op_id}"
        count = operation_counts[op_id]
        acc_key = f'op_{op_id}_acc'
        if acc_key in val_metrics:
            acc = val_metrics[acc_key] * 100
            print(f"    {op_name:12s}: {acc:5.1f}% ({count:4d} samples)")


# ============================================================================
# Training Functions
# ============================================================================

def create_class_balanced_sampler(dataset, power=0.5):
    """Create weighted sampler for operation type balancing."""
    labels = []
    for i in range(len(dataset)):
        op_type = dataset[i]['operation_type']
        if hasattr(op_type, 'item'):
            op_type = op_type.item()
        labels.append(op_type)

    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=9)
    weights = 1.0 / ((class_counts + 1e-6) ** power)
    sample_weights = weights[labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )


def create_error_focused_sampler(dataset, vocab: dict, error_boost: float = 2.0):
    """Create sampler that overweights sequences containing frequently-confused tokens.

    Based on error analysis, the most confused tokens are:
    - G1 (often mispredicted as X, Y)
    - X, Y (confused with each other and G1)
    - NUM_X and NUM_Y (mode collapse to common values)

    Args:
        dataset: Training dataset
        vocab: Vocabulary dictionary (token -> idx)
        error_boost: Weight multiplier for error-prone sequences (default: 2.0)

    Returns:
        WeightedRandomSampler with higher weights for error-prone sequences
    """
    # Identify frequently-confused tokens from error analysis
    confused_tokens = {
        'G1', 'G0', 'G2', 'G3',  # Commands often confused with param types
        'X', 'Y', 'Z',           # Param types confused with commands
    }

    # Also boost sequences with NUM_X and NUM_Y (where mode collapse happens)
    num_xy_prefixes = ('NUM_X_', 'NUM_Y_')

    # Build reverse vocab for token string lookup
    idx_to_token = {v: k for k, v in vocab.items()}

    weights = []
    for i in range(len(dataset)):
        sample = dataset[i]
        # Dataset uses 'target_tokens' key for token sequence
        tokens = sample.get('target_tokens', sample.get('tokens', None))
        if tokens is None:
            weights.append(1.0)
            continue

        # Convert token indices to strings
        weight = 1.0
        for tok_idx in tokens:
            if hasattr(tok_idx, 'item'):
                tok_idx = tok_idx.item()
            tok_str = idx_to_token.get(tok_idx, '')

            # Boost for confused command/param tokens
            if tok_str in confused_tokens:
                weight += error_boost * 0.5

            # Boost for NUM_X/NUM_Y tokens (mode collapse issue)
            if tok_str.startswith(num_xy_prefixes):
                weight += error_boost * 0.3

        # Cap weight to prevent extreme oversampling
        weights.append(min(weight, 5.0))

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True
    )


def apply_initialization(model, strategy, gain=1.0):
    """Apply weight initialization strategy."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if strategy == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            elif strategy == 'xavier_normal':
                nn.init.xavier_normal_(module.weight, gain=gain)
            elif strategy == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif strategy == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif strategy == 'orthogonal':
                nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


# ============================================================================
# Multi-Head Loss Function
# ============================================================================

def create_xy_boosted_class_weights(vocab_path: str, xy_boost: float = 2.0, device='cpu'):
    """Create class weights that boost NUM_X and NUM_Y tokens.

    This addresses the finding from error analysis that NUM_X (43.8% acc) and
    NUM_Y (50.2% acc) are the main bottlenecks while NUM_Z achieves 98% accuracy.

    Args:
        vocab_path: Path to vocabulary JSON file
        xy_boost: Weight multiplier for NUM_X and NUM_Y tokens (default: 2.0)
        device: Device to place weights tensor on

    Returns:
        Class weights tensor of shape (vocab_size,)
    """
    import json
    with open(vocab_path) as f:
        vocab_data = json.load(f)

    vocab = vocab_data.get('vocab', vocab_data)
    vocab_size = len(vocab)

    # Initialize all weights to 1.0
    weights = torch.ones(vocab_size, device=device)

    # Boost weights for NUM_X and NUM_Y tokens
    for token, idx in vocab.items():
        if token.startswith('NUM_X_') or token.startswith('NUM_Y_'):
            weights[idx] = xy_boost

    return weights


class SensorMultiHeadLoss(nn.Module):
    """Combined loss for all prediction heads."""

    def __init__(self, args, class_weights=None):
        super().__init__()
        self.args = args

        # Type loss (with focal option)
        if args.use_focal_loss:
            self.type_loss = FocalLoss(
                gamma=args.focal_gamma,
                alpha=args.focal_alpha,
                ignore_index=-1
            )
        else:
            self.type_loss = nn.CrossEntropyLoss(
                ignore_index=-1, label_smoothing=args.label_smoothing
            )

        # Command loss (extra focal for rare commands)
        if args.use_focal_loss:
            self.command_loss = FocalLoss(
                gamma=args.focal_gamma + args.command_focal_gamma_boost,
                alpha=args.focal_alpha,
                ignore_index=-1
            )
        else:
            self.command_loss = nn.CrossEntropyLoss(
                ignore_index=-1, label_smoothing=args.label_smoothing
            )

        # Param type loss
        self.param_type_loss = nn.CrossEntropyLoss(
            ignore_index=-1, label_smoothing=args.label_smoothing
        )

        # Digit loss
        self.digit_loss = DigitByDigitLoss(
            n_digit_positions=6,
            aux_loss_weight=args.aux_regression_weight,
            label_smoothing=args.label_smoothing
        )

        # Legacy token loss (for comparison)
        # Optionally use class weights to boost NUM_X and NUM_Y tokens
        # Optionally use position weighting to focus on early positions
        use_position_weights = getattr(args, 'use_position_weights', True) and not getattr(args, 'no_position_weights', False)

        if use_position_weights:
            # Build position weights: [scale, ..., 1.0] decaying from position 0
            scale = getattr(args, 'position_weight_scale', 3.0)
            max_len = 32
            # Decay: [3.0, 2.5, 2.0, 1.5, 1.2, 1.0, 1.0, ...]
            position_weights = []
            for i in range(max_len):
                if i == 0:
                    position_weights.append(scale)
                elif i < 5:
                    # Linear decay from scale to 1.0 over positions 0-5
                    position_weights.append(scale - (scale - 1.0) * (i / 5))
                else:
                    position_weights.append(1.0)

            if args.use_focal_loss:
                self.legacy_loss = PositionWeightedFocalLoss(
                    gamma=args.focal_gamma,
                    alpha=args.focal_alpha,
                    position_weights=position_weights,
                    max_len=max_len,
                    ignore_index=PAD_TOKEN_ID,
                )
            else:
                self.legacy_loss = PositionWeightedCrossEntropy(
                    position_weights=position_weights,
                    max_len=max_len,
                    ignore_index=PAD_TOKEN_ID,
                    label_smoothing=args.label_smoothing
                )
            self.use_position_weights = True
        else:
            self.legacy_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=PAD_TOKEN_ID,
                label_smoothing=args.label_smoothing
            )
            self.use_position_weights = False

    def forward(self, outputs, targets, curriculum_weights=None):
        """Compute multi-head loss."""
        if curriculum_weights is None:
            curriculum_weights = {'structure': 1.0, 'digit': 1.0, 'value': 1.0}

        losses = {}

        # Flatten for loss computation
        B, L = targets['type'].shape if 'type' in targets else targets['target_tokens'].shape[:2]

        # Structure losses
        if 'type' in targets:
            type_logits = outputs['type_logits'].view(-1, outputs['type_logits'].size(-1))
            type_targets = targets['type'].view(-1)
            losses['type'] = self.type_loss(type_logits, type_targets)

        if 'command' in targets:
            cmd_logits = outputs['command_logits'].view(-1, outputs['command_logits'].size(-1))
            cmd_targets = targets['command'].view(-1)
            losses['command'] = self.command_loss(cmd_logits, cmd_targets)

        if 'param_type' in targets:
            pt_logits = outputs['param_type_logits'].view(-1, outputs['param_type_logits'].size(-1))
            pt_targets = targets['param_type'].view(-1)
            losses['param_type'] = self.param_type_loss(pt_logits, pt_targets)

        # Digit losses (if targets available)
        if 'sign' in targets and 'digits' in targets:
            digit_loss, digit_metrics = self.digit_loss(
                outputs,
                targets['sign'],
                targets['digits'],
                targets.get('values', torch.zeros_like(targets['sign'], dtype=torch.float)),
                targets.get('numeric_mask', torch.ones_like(targets['sign'], dtype=torch.bool))
            )
            losses['digit'] = digit_loss
        else:
            losses['digit'] = torch.tensor(0.0, device=outputs['type_logits'].device)

        # Legacy loss (for comparison/ablation)
        # IMPORTANT: Use raw_legacy_logits for training loss to avoid corrupted gradients
        # from type constraint masking. Type constraint should only affect inference.
        if 'target_tokens' in targets:
            # Prefer raw logits for training (unconstrained by type predictions)
            # This allows proper gradient flow even when type predictions are wrong
            if 'raw_legacy_logits' in outputs:
                training_logits = outputs['raw_legacy_logits']
            else:
                training_logits = outputs['legacy_logits']

            if self.use_position_weights:
                # Position-weighted loss expects [B, L, C] format
                legacy_targets = targets['target_tokens']  # [B, L]
                losses['legacy'] = self.legacy_loss(training_logits, legacy_targets)
            else:
                # Standard loss uses flattened format
                legacy_logits = training_logits.view(-1, training_logits.size(-1))
                legacy_targets = targets['target_tokens'].view(-1)
                losses['legacy'] = self.legacy_loss(legacy_logits, legacy_targets)

        # Weighted combination
        total = 0.0
        if 'type' in losses:
            total += self.args.type_weight * losses['type'] * curriculum_weights['structure']
        if 'command' in losses:
            total += self.args.command_weight * losses['command'] * curriculum_weights['structure']
        if 'param_type' in losses:
            total += self.args.param_type_weight * losses['param_type'] * curriculum_weights['structure']
        if 'digit' in losses:
            total += self.args.digit_weight * losses['digit'] * curriculum_weights['digit']
        if 'legacy' in losses:
            total += self.args.legacy_weight * losses['legacy']

        losses['total'] = total
        return total, losses


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(encoder, decoder, train_loader, optimizer, loss_fn, curriculum,
                scheduled_sampling, epoch, args, device, grammar_constraints=None,
                scaler=None, use_auxiliary=False):
    """Train one epoch.

    Args:
        encoder: Encoder model (frozen or partially unfrozen)
        decoder: Decoder model
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        curriculum: Curriculum scheduler
        scheduled_sampling: Scheduled sampling scheduler
        epoch: Current epoch
        args: Training arguments
        device: Device
        grammar_constraints: Optional grammar constraints
        scaler: Optional AMP scaler
        use_auxiliary: Whether to use auxiliary supervision heads
    """
    decoder.train()

    # Set encoder mode based on unfreezing
    encoder_unfrozen = args.unfreeze_encoder_layers != 0
    if encoder_unfrozen:
        encoder.train()  # Enable training mode for unfrozen encoder
    else:
        encoder.eval()

    # Get curriculum weights
    curriculum_weights = curriculum.get_loss_weights(epoch) if curriculum else None
    tf_ratio = scheduled_sampling.get_ratio(epoch) if scheduled_sampling else 1.0

    # Determine if using AMP (only on CUDA)
    use_amp = scaler is not None and device.type == 'cuda'

    total_loss = 0
    metrics = defaultdict(float)
    n_batches = 0
    op_correct = 0
    op_total = 0
    token_correct = 0
    token_total = 0
    seq_correct = 0
    seq_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        sensor_features = batch['sensor_features'].to(device)
        input_tokens = batch['input_tokens'].to(device)
        target_tokens = batch['target_tokens'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        gt_operation_type = batch['operation_type'].to(device)

        # Encoder forward - with or without gradients
        # Note: EnhancedEncoder.encode() returns (pooled_features, memory_sequence)
        #       MM_DTAE_LSTM.encode() returns (latent_sequence, hidden_state)
        # For decoder, we need the sequence representation (not pooled)
        if encoder_unfrozen:
            # Encoder is being fine-tuned - need gradients
            enc_out1, enc_out2 = encoder.encode(sensor_features)
            # EnhancedEncoder: out1 = pooled (2D), out2 = memory (3D) -> use out2
            # MM_DTAE_LSTM: out1 = sequence (3D), out2 = hidden -> use out1
            sensor_emb = enc_out2 if enc_out1.dim() == 2 else enc_out1
            classify_input = enc_out1 if enc_out1.dim() == 2 else enc_out1
            op_logits, aux_outputs = encoder.classify(classify_input)
            operation_type = op_logits.argmax(-1)
        else:
            # Frozen encoder - no gradients
            with torch.no_grad():
                enc_out1, enc_out2 = encoder.encode(sensor_features)
                # EnhancedEncoder: out1 = pooled (2D), out2 = memory (3D) -> use out2
                # MM_DTAE_LSTM: out1 = sequence (3D), out2 = hidden -> use out1
                sensor_emb = enc_out2 if enc_out1.dim() == 2 else enc_out1
                classify_input = enc_out1 if enc_out1.dim() == 2 else enc_out1
                op_logits, aux_outputs = encoder.classify(classify_input)
                operation_type = op_logits.argmax(-1)

        # Track encoder operation classification accuracy
        op_correct += (operation_type == gt_operation_type).sum().item()
        op_total += gt_operation_type.size(0)

        # Decoder forward (with scheduled sampling support)
        # Use AMP autocast context if enabled
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=sensor_emb,
                operation_type=operation_type,
                tgt_key_padding_mask=padding_mask,
                teacher_forcing_ratio=tf_ratio,
            )

            # Build targets
            targets = {
                'target_tokens': target_tokens,
            }

            # Loss computation
            loss, loss_dict = loss_fn(outputs, targets, curriculum_weights)

            # Add auxiliary loss if using enhanced encoder with auxiliary heads
            if use_auxiliary and aux_outputs is not None:
                # Build auxiliary targets from batch metadata if available
                aux_targets = {}

                # Extract auxiliary targets from batch if available
                if 'operation_type' in batch:
                    aux_targets['motion_type'] = batch['operation_type'].to(device)
                if 'seq_length' in batch:
                    aux_targets['seq_length'] = batch['seq_length'].to(device)

                # Compute auxiliary loss
                aux_loss, aux_loss_dict = compute_auxiliary_loss(
                    aux_outputs, aux_targets,
                    weight=args.auxiliary_loss_weight,
                    weights={
                        'motion': args.auxiliary_motion_weight,
                        'length': args.auxiliary_length_weight,
                        'magnitude': args.auxiliary_magnitude_weight,
                        'param_presence': args.auxiliary_param_presence_weight,
                    }
                )

                loss = loss + aux_loss
                loss_dict['auxiliary'] = aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss
                for k, v in aux_loss_dict.items():
                    loss_dict[f'aux_{k}'] = v.item() if torch.is_tensor(v) else v

            # Add encoder operation classification loss
            if encoder_unfrozen and args.op_loss_weight > 0:
                op_cls_loss = nn.functional.cross_entropy(op_logits, gt_operation_type)
                loss = loss + args.op_loss_weight * op_cls_loss
                loss_dict['op_cls'] = op_cls_loss.item()

            # Add grammar constraint loss if enabled
            if grammar_constraints is not None and args.grammar_weight > 0:
                constraint_losses = grammar_constraints.compute_constraint_losses(
                    predictions=outputs,
                    targets=targets,
                    current_tokens=input_tokens,
                )
                grammar_loss = constraint_losses['total_constraint']
                loss = loss + args.grammar_weight * grammar_loss
                loss_dict['grammar_loss'] = grammar_loss.item()

        # Gradient accumulation
        loss = loss / args.accumulation_steps

        if use_amp:
            # Mixed precision backward
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % args.accumulation_steps == 0:
            if use_amp:
                # Unscale before grad clip
                scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    # Clip gradients for both decoder and unfrozen encoder params
                    all_params = list(decoder.parameters())
                    if encoder_unfrozen:
                        all_params.extend([p for p in encoder.parameters() if p.requires_grad])
                    torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip > 0:
                    all_params = list(decoder.parameters())
                    if encoder_unfrozen:
                        all_params.extend([p for p in encoder.parameters() if p.requires_grad])
                    torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
                optimizer.step()
            optimizer.zero_grad()

        # Track metrics
        total_loss += loss.item() * args.accumulation_steps
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                metrics[k] += v.item()
            else:
                metrics[k] += v
        n_batches += 1

        # Track token and sequence accuracy
        with torch.no_grad():
            logits = outputs['legacy_logits']
            preds = logits.argmax(dim=-1)
            mask = (target_tokens != 0)  # ignore PAD
            token_correct += ((preds == target_tokens) & mask).sum().item()
            token_total += mask.sum().item()
            B = target_tokens.size(0)
            for b in range(B):
                m = mask[b]
                if m.sum() > 0:
                    if (preds[b][m] == target_tokens[b][m]).all():
                        seq_correct += 1
                    seq_total += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item() * args.accumulation_steps:.4f}",
            'tf': f"{tf_ratio:.2f}"
        })

    return {
        'loss': total_loss / n_batches,
        'token': token_correct / token_total if token_total > 0 else 0,
        'sequence': seq_correct / seq_total if seq_total > 0 else 0,
        'encoder_op_acc': op_correct / op_total if op_total > 0 else 0,
        **{k: v / n_batches for k, v in metrics.items()}
    }


def validate(encoder, decoder, val_loader, loss_fn, args, device, comprehensive=False):
    """Validate model. If comprehensive=True, also compute precision/recall/F1/BLEU/ED."""
    decoder.eval()
    encoder.eval()

    total_loss = 0
    correct = defaultdict(int)
    total = defaultdict(int)
    seq_correct = 0
    seq_total = 0
    n_batches = 0
    op_correct = 0
    op_total = 0
    all_preds_list = []
    all_targets_list = []
    all_pred_seqs = []
    all_target_seqs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            sensor_features = batch['sensor_features'].to(device)
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            gt_operation_type = batch['operation_type'].to(device)

            enc_out1, enc_out2 = encoder.encode(sensor_features)
            sensor_emb = enc_out2 if enc_out1.dim() == 2 else enc_out1
            classify_input = enc_out1 if enc_out1.dim() == 2 else enc_out1
            op_logits, _ = encoder.classify(classify_input)
            operation_type = op_logits.argmax(-1)

            # Track encoder operation classification accuracy
            op_correct += (operation_type == gt_operation_type).sum().item()
            op_total += gt_operation_type.size(0)

            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=sensor_emb,
                operation_type=operation_type,
                tgt_key_padding_mask=padding_mask,
            )

            targets = {'target_tokens': target_tokens}
            loss, _ = loss_fn(outputs, targets)
            total_loss += loss.item()

            if 'raw_legacy_logits' in outputs:
                legacy_pred = outputs['raw_legacy_logits'].argmax(-1)
            else:
                legacy_pred = outputs['legacy_logits'].argmax(-1)
            B, T = target_tokens.shape

            if comprehensive:
                all_preds_list.append(legacy_pred)
                all_targets_list.append(target_tokens)

            for b in range(B):
                eos_positions = (target_tokens[b] == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    eos_pos = eos_positions[0].item() + 1
                else:
                    eos_pos = T

                valid_positions = torch.zeros(T, dtype=torch.bool, device=target_tokens.device)
                valid_positions[:eos_pos] = True
                valid_mask_b = valid_positions & (target_tokens[b] != PAD_TOKEN_ID)

                correct['token'] += (legacy_pred[b][valid_mask_b] == target_tokens[b][valid_mask_b]).sum().item()
                total['token'] += valid_mask_b.sum().item()

                if valid_mask_b.sum() > 0:
                    pred_seq = legacy_pred[b][valid_mask_b]
                    target_seq = target_tokens[b][valid_mask_b]
                    if comprehensive:
                        all_pred_seqs.append(pred_seq.cpu().tolist())
                        all_target_seqs.append(target_seq.cpu().tolist())
                    if (pred_seq == target_seq).all():
                        seq_correct += 1
                    seq_total += 1

            n_batches += 1

    accuracies = {k: correct[k] / max(total[k], 1) for k in correct}
    seq_accuracy = seq_correct / max(seq_total, 1)

    result = {
        'loss': total_loss / n_batches,
        'sequence': seq_accuracy,
        'encoder_op_acc': op_correct / op_total if op_total > 0 else 0,
        **accuracies
    }

    if comprehensive and all_preds_list:
        all_preds_t = torch.cat(all_preds_list, dim=0)
        all_targets_t = torch.cat(all_targets_list, dim=0)
        comp_metrics = compute_comprehensive_metrics(
            all_preds_t, all_targets_t,
            all_pred_seqs, all_target_seqs,
            pad_token=PAD_TOKEN_ID,
        )
        result['comprehensive'] = comp_metrics

    return result


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train Sensor Multi-Head Decoder')

    # ==================== MODEL ARCHITECTURE ====================
    parser.add_argument('--d-model', type=int, default=192,
                        help='Model dimension (default: 192)')
    parser.add_argument('--n-heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--n-layers', type=int, default=4,
                        help='Number of transformer layers (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--embed-dropout', type=float, default=0.1,
                        help='Embedding dropout rate (default: 0.1)')
    parser.add_argument('--sensor-dim', type=int, default=128,
                        help='Sensor embedding dimension (default: 128)')
    parser.add_argument('--n-operations', type=int, default=9,
                        help='Number of operation types (default: 9)')
    parser.add_argument('--n-types', type=int, default=4,
                        help='Number of token types (default: 4)')
    parser.add_argument('--n-commands', type=int, default=6,
                        help='Number of G/M commands (default: 6)')
    parser.add_argument('--n-param-types', type=int, default=10,
                        help='Number of parameter types (default: 10)')
    parser.add_argument('--max-seq-len', type=int, default=32,
                        help='Maximum sequence length (default: 32)')
    # Extended architecture params
    parser.add_argument('--d-ff', type=int, default=None,
                        help='Feedforward dim (default: 4*d_model)')
    parser.add_argument('--d-ff-multiplier', type=float, default=4.0,
                        help='FF dim as multiplier of d_model (default: 4.0)')
    parser.add_argument('--operation-embed-dim', type=int, default=32,
                        help='Operation embedding dimension (default: 32)')
    parser.add_argument('--max-int-digits', type=int, default=2,
                        help='Max integer digits for value head (default: 2)')
    parser.add_argument('--n-decimal-digits', type=int, default=4,
                        help='Decimal digits for value head (default: 4)')
    parser.add_argument('--pos-encoding-max-len', type=int, default=512,
                        help='Max length for positional encoding (default: 512)')

    # ==================== COMPONENT ABLATION FLAGS ====================
    parser.add_argument('--no-operation-conditioning', action='store_true',
                        help='Ablation: disable operation embedding conditioning')
    parser.add_argument('--no-cross-attention', action='store_true',
                        help='Ablation: use self-attention only (no sensor memory)')
    parser.add_argument('--no-positional-encoding', action='store_true',
                        help='Ablation: disable positional encoding')

    # ==================== TYPE-CONSTRAINED DECODING (BREAKTHROUGH Phase 1) ====================
    parser.add_argument('--use-type-constraint', action='store_true', default=True,
                        help='Use type predictions to constrain token prediction (default: True)')
    parser.add_argument('--no-type-constraint', action='store_true',
                        help='Ablation: disable type-constrained decoding')

    # ==================== SENSOR VALUE PRIOR (BREAKTHROUGH Phase 2) ====================
    parser.add_argument('--use-sensor-prior', action='store_true',
                        help='Enable sensor value prior for direct sensor-to-digit prediction')
    parser.add_argument('--sensor-prior-weight', type=float, default=0.5,
                        help='Weight for sensor prior bias (0-1, default: 0.5)')

    parser.add_argument('--use-self-conditioning', action='store_true',
                        help='Enable self-conditioning on previous predictions')

    # ==================== ENCODER IMPROVEMENTS (PHASE 1.6) ====================
    parser.add_argument('--use-enhanced-encoder', action='store_true',
                        help='Use enhanced encoder with multi-scale temporal processing')
    parser.add_argument('--use-multiscale-encoder', action='store_true',
                        help='Use multi-scale temporal encoder (alias for --use-enhanced-encoder)')
    parser.add_argument('--encoder-n-scales', type=int, default=4,
                        help='Number of conv scales in multi-scale encoder (default: 4)')
    parser.add_argument('--encoder-kernel-sizes', type=int, nargs='+',
                        default=[3, 5, 7, 11],
                        help='Kernel sizes for multi-scale conv (default: [3, 5, 7, 11])')
    parser.add_argument('--encoder-dilations', type=int, nargs='+',
                        default=[1, 2, 4, 8],
                        help='Dilation rates for multi-scale conv (default: [1, 2, 4, 8])')
    parser.add_argument('--encoder-lstm-layers', type=int, default=2,
                        help='Number of LSTM layers in encoder (default: 2)')
    parser.add_argument('--use-multihead-pooling', action='store_true',
                        help='Use multi-head attention pooling instead of simple attention')
    parser.add_argument('--pooling-n-heads', type=int, default=4,
                        help='Number of attention heads for pooling (default: 4)')
    parser.add_argument('--pooling-n-queries', type=int, default=8,
                        help='Number of learned query vectors for pooling (default: 8)')
    parser.add_argument('--pooling-dropout', type=float, default=0.1,
                        help='Dropout for attention pooling (default: 0.1)')
    parser.add_argument('--unfreeze-encoder-layers', type=int, default=0,
                        help='Number of encoder layers to unfreeze (0=frozen, -1=all, default: 0)')
    parser.add_argument('--encoder-lr-scale', type=float, default=0.1,
                        help='LR multiplier for unfrozen encoder layers (default: 0.1)')
    parser.add_argument('--use-auxiliary-heads', action='store_true',
                        help='Add auxiliary supervision heads to encoder')
    parser.add_argument('--op-loss-weight', type=float, default=1.0,
                        help='Weight for encoder operation classification loss (default: 1.0)')
    parser.add_argument('--auxiliary-loss-weight', type=float, default=0.3,
                        help='Weight for auxiliary losses (default: 0.3)')
    parser.add_argument('--auxiliary-motion-weight', type=float, default=1.0,
                        help='Weight for motion type prediction (default: 1.0)')
    parser.add_argument('--auxiliary-length-weight', type=float, default=0.5,
                        help='Weight for length prediction (default: 0.5)')
    parser.add_argument('--auxiliary-magnitude-weight', type=float, default=0.8,
                        help='Weight for magnitude prediction (default: 0.8)')
    parser.add_argument('--auxiliary-param-presence-weight', type=float, default=0.7,
                        help='Weight for param presence prediction (default: 0.7)')
    parser.add_argument('--encoder-hidden-dim', type=int, default=256,
                        help='Hidden dimension for encoder (default: 256)')
    parser.add_argument('--encoder-dropout', type=float, default=0.3,
                        help='Dropout rate for encoder (default: 0.3)')

    # ==================== REGULARIZATION ====================
    parser.add_argument('--drop-path-rate', type=float, default=0.0,
                        help='Stochastic depth / drop path rate (0 = disabled, default: 0.0)')
    parser.add_argument('--use-gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing for memory efficiency')
    parser.add_argument('--use-amp', action='store_true',
                        help='Enable mixed precision (AMP) training for CUDA (no-op on MPS)')

    # ==================== DATA & PATHS ====================
    parser.add_argument('--split-dir', type=str, required=True)
    parser.add_argument('--vocab-path', type=str, required=True)
    parser.add_argument('--encoder-path', type=str, default=None,
                        help='Path to pretrained encoder (omit to train from scratch)')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)

    # ==================== DATA AUGMENTATION ====================
    parser.add_argument('--augment', action='store_true',
                        help='Enable sensor data augmentation')
    parser.add_argument('--augment-prob', type=float, default=0.3,
                        help='Probability of each augmentation (default: 0.3)')
    parser.add_argument('--noise-level', type=float, default=0.01,
                        help='Gaussian noise std as fraction of signal (default: 0.01)')
    parser.add_argument('--shift-range', type=int, default=1,
                        help='Max temporal shift ±N timesteps (default: 1)')
    parser.add_argument('--scale-range-min', type=float, default=0.98,
                        help='Min magnitude scale (default: 0.98)')
    parser.add_argument('--scale-range-max', type=float, default=1.02,
                        help='Max magnitude scale (default: 1.02)')
    parser.add_argument('--time-warp-sigma', type=float, default=0.1,
                        help='Time warp distortion strength (default: 0.1)')
    parser.add_argument('--feature-dropout-prob', type=float, default=0.05,
                        help='Probability of dropping feature dims (default: 0.05)')
    parser.add_argument('--cutout-length', type=int, default=2,
                        help='Max temporal cutout window (default: 2)')
    parser.add_argument('--jitter-sigma', type=float, default=0.005,
                        help='Time-dependent jitter noise std (default: 0.005)')
    parser.add_argument('--oversample-rare', action='store_true',
                        help='Oversample rare G/M command sequences')
    parser.add_argument('--oversample-factor', type=int, default=3,
                        help='Repetition factor for rare sequences (default: 3)')

    # ==================== PROGRESSIVE AUGMENTATION ====================
    parser.add_argument('--progressive-augmentation', action='store_true',
                        help='Enable progressive augmentation (start conservative, increase over training)')
    parser.add_argument('--augment-start', type=float, default=0.1,
                        help='Initial augmentation probability (default: 0.1)')
    parser.add_argument('--augment-end', type=float, default=0.5,
                        help='Final augmentation probability (default: 0.5)')
    parser.add_argument('--augment-ramp-epochs', type=int, default=None,
                        help='Epochs to ramp augmentation (default: 70 percent of max_epochs)')

    # ==================== MIXUP AUGMENTATION ====================
    parser.add_argument('--use-mixup', action='store_true',
                        help='Enable token mixup augmentation')
    parser.add_argument('--mixup-prob', type=float, default=0.3,
                        help='Probability of applying mixup (default: 0.3)')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='Beta distribution alpha for mixup (default: 0.2)')

    # ==================== SEGMENT PERMUTATION ====================
    parser.add_argument('--permute-segments-prob', type=float, default=0.0,
                        help='Probability of permuting segments (default: 0.0 = disabled)')

    # ==================== TRAINING BASICS ====================
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--track-metric', type=str, default='token',
                        choices=['loss', 'token', 'sequence', 'composite'],
                        help='Metric to track for early stopping (added sequence)')

    # ==================== OPTIMIZER ====================
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd', 'rmsprop'])
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--momentum', type=float, default=0.9)

    # ==================== LR SCHEDULER ====================
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=['none', 'cosine', 'plateau', 'step', 'cyclic', 'onecycle', 'cosine_restarts'])
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--warmup-type', type=str, default='linear',
                        choices=['linear', 'cosine'],
                        help='Warmup type: linear or cosine (default: linear)')
    parser.add_argument('--cosine-t-max', type=int, default=None)
    parser.add_argument('--plateau-patience', type=int, default=5)
    parser.add_argument('--plateau-factor', type=float, default=0.5)
    parser.add_argument('--restart-period', type=int, default=30,
                        help='Period for cosine restarts scheduler (default: 30)')
    parser.add_argument('--restart-mult', type=float, default=1.0,
                        help='Multiplier for restart period growth (default: 1.0)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum LR for schedulers (default: 1e-6)')

    # ==================== REGULARIZATION ====================
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--accumulation-steps', type=int, default=1)
    parser.add_argument('--label-smoothing', type=float, default=0.1)

    # ==================== LOSS FUNCTIONS ====================
    parser.add_argument('--use-focal-loss', action='store_true')
    parser.add_argument('--focal-gamma', type=float, default=3.0)

    # ==================== ADVANCED LOSS IMPROVEMENTS ====================
    parser.add_argument('--use-eos-calibration', action='store_true',
                        help='Enable EOS calibration loss for sequence length accuracy')
    parser.add_argument('--eos-calibration-weight', type=float, default=0.2,
                        help='Weight for EOS calibration loss (default: 0.2)')
    parser.add_argument('--use-dual-head-value', action='store_true',
                        help='Enable dual-head value loss (bucket + residual)')
    parser.add_argument('--dual-head-weight', type=float, default=0.3,
                        help='Weight for dual-head value loss (default: 0.3)')
    parser.add_argument('--use-consistency-loss', action='store_true',
                        help='Enable consistency loss between digit and regression outputs')
    parser.add_argument('--consistency-weight', type=float, default=0.1,
                        help='Weight for consistency loss (default: 0.1)')

    # ==================== LOSS WEIGHTS ====================
    parser.add_argument('--type-weight', type=float, default=1.0)
    parser.add_argument('--command-weight', type=float, default=2.5)
    parser.add_argument('--param-type-weight', type=float, default=1.5)
    parser.add_argument('--digit-weight', type=float, default=1.0)
    parser.add_argument('--legacy-weight', type=float, default=1.0)
    parser.add_argument('--aux-regression-weight', type=float, default=0.1)

    # ==================== CLASS BALANCING ====================
    parser.add_argument('--use-class-weights', action='store_true')
    parser.add_argument('--sampler-power', type=float, default=0.5)
    parser.add_argument('--xy-boost', type=float, default=1.0,
                        help='Boost weight for NUM_X and NUM_Y tokens (default: 1.0 = no boost). '
                             'Error analysis showed NUM_X (43.8%%) and NUM_Y (50.2%%) are bottlenecks.')

    # ==================== ERROR-FOCUSED SAMPLING (BREAKTHROUGH Phase 4) ====================
    parser.add_argument('--use-error-sampling', action='store_true',
                        help='Enable error-focused sampling to oversample sequences with confused tokens')
    parser.add_argument('--error-boost', type=float, default=2.0,
                        help='Weight multiplier for error-prone sequences (default: 2.0)')

    # ==================== SWA ====================
    parser.add_argument('--use-swa', action='store_true',
                        help='Enable Stochastic Weight Averaging')
    parser.add_argument('--swa-start-epoch', type=int, default=75,
                        help='Epoch to start SWA (default: 75)')
    parser.add_argument('--swa-lr', type=float, default=5e-5,
                        help='Learning rate for SWA phase (default: 5e-5)')
    parser.add_argument('--swa-update-freq', type=int, default=1,
                        help='Update SWA model every N epochs (default: 1)')
    parser.add_argument('--swa-anneal-epochs', type=int, default=10,
                        help='Epochs to anneal LR to SWA LR (default: 10)')
    parser.add_argument('--swa-anneal-strategy', type=str, default='cos',
                        choices=['cos', 'linear'],
                        help='SWA LR annealing strategy (default: cos)')
    parser.add_argument('--swa-bn-update-batches', type=int, default=None,
                        help='Number of batches for SWA BN update (default: all)')

    # ==================== CURRICULUM LEARNING ====================
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--curriculum-phases', type=int, default=3)
    parser.add_argument('--curriculum-epochs-per-phase', type=int, default=20,
                        help='Faster curriculum phases (was 30)')

    # ==================== SCHEDULED SAMPLING ====================
    parser.add_argument('--scheduled-sampling', action='store_true')
    parser.add_argument('--teacher-forcing-start', type=float, default=1.0)
    parser.add_argument('--teacher-forcing-end', type=float, default=0.2,
                        help='More aggressive TF decay endpoint (was 0.5)')
    parser.add_argument('--teacher-forcing-decay', type=str, default='cosine',
                        choices=['linear', 'exponential', 'cosine'])
    parser.add_argument('--teacher-forcing-decay-epochs', type=int, default=None,
                        help='Epochs over which to decay TF (default: max_epochs)')

    # ==================== GRAMMAR CONSTRAINTS ====================
    parser.add_argument('--use-grammar-constraints', action='store_true',
                        help='Enable grammar constraint losses during training')
    parser.add_argument('--grammar-weight', type=float, default=0.1,
                        help='Weight for grammar constraint loss (default: 0.1)')
    # Grammar constraint feature flags
    parser.add_argument('--allow-modal-commands', action='store_true',
                        help='Allow G-code modal behavior (default: False)')
    parser.add_argument('--grammar-constraint-arc', action='store_true', default=True,
                        help='Enable arc radius constraint (default: True)')
    parser.add_argument('--no-grammar-constraint-arc', action='store_false', dest='grammar_constraint_arc',
                        help='Disable arc radius constraint')
    parser.add_argument('--grammar-constraint-feed', action='store_true', default=True,
                        help='Enable feed rate constraint (default: True)')
    parser.add_argument('--no-grammar-constraint-feed', action='store_false', dest='grammar_constraint_feed',
                        help='Disable feed rate constraint')
    parser.add_argument('--grammar-constraint-modal', action='store_true', default=True,
                        help='Enable modal state constraint (default: True)')
    parser.add_argument('--no-grammar-constraint-modal', action='store_false', dest='grammar_constraint_modal',
                        help='Disable modal state constraint')
    # Individual constraint weights
    parser.add_argument('--constraint-arc-radius-weight', type=float, default=1.0,
                        help='Arc radius constraint weight (default: 1.0)')
    parser.add_argument('--constraint-rapid-feed-weight', type=float, default=0.5,
                        help='Rapid feed constraint weight (default: 0.5)')
    parser.add_argument('--constraint-modal-state-weight', type=float, default=0.3,
                        help='Modal state constraint weight (default: 0.3)')
    parser.add_argument('--constraint-alternating-arc-weight', type=float, default=0.4,
                        help='Alternating arc constraint weight (default: 0.4)')
    parser.add_argument('--constraint-linear-feed-weight', type=float, default=0.3,
                        help='Linear cutting feed constraint weight (default: 0.3)')
    parser.add_argument('--constraint-z-retract-weight', type=float, default=0.2,
                        help='Z-retract pattern constraint weight (default: 0.2)')
    parser.add_argument('--constraint-modal-group-weight', type=float, default=0.6,
                        help='RS-274D modal group rule weight (default: 0.6)')
    parser.add_argument('--constraint-cmd-param-weight', type=float, default=0.4,
                        help='Command-param association weight (default: 0.4)')
    parser.add_argument('--constraint-single-letter-weight', type=float, default=0.3,
                        help='Single letter rule constraint weight (default: 0.3)')
    parser.add_argument('--constraint-context-window', type=int, default=5,
                        help='Context window for modal constraints (default: 5)')
    parser.add_argument('--constraint-line-window', type=int, default=8,
                        help='Context window for single letter rule (default: 8)')

    # ==================== SCST / REINFORCEMENT LEARNING ====================
    parser.add_argument('--use-scst', action='store_true',
                        help='Enable SCST (Self-Critical Sequence Training) fine-tuning')
    parser.add_argument('--scst-start-epoch', type=int, default=100,
                        help='Epoch to start SCST training (default: 100)')
    parser.add_argument('--scst-weight', type=float, default=0.5,
                        help='SCST loss weight vs cross-entropy (default: 0.5)')
    parser.add_argument('--scst-lr', type=float, default=1e-5,
                        help='Learning rate for SCST phase (default: 1e-5)')
    parser.add_argument('--sample-temperature', type=float, default=0.8,
                        help='Temperature for SCST sampling (default: 0.8)')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                        help='Reward scaling factor (default: 1.0)')
    parser.add_argument('--scst-max-length', type=int, default=32,
                        help='Max generation length for SCST (default: 32)')
    parser.add_argument('--use-mixed-reward', action='store_true', default=True,
                        help='Combine sequence + token rewards (default: True)')

    # ==================== CURRICULUM LEARNING (EXTENDED) ====================
    parser.add_argument('--curriculum-structure-weight', type=float, default=1.0,
                        help='Structure loss weight in curriculum (default: 1.0)')
    parser.add_argument('--curriculum-digit-weight-p2', type=float, default=0.5,
                        help='Digit weight in phase 2 (default: 0.5)')
    parser.add_argument('--curriculum-digit-weight-p3', type=float, default=1.0,
                        help='Digit weight in phase 3 (default: 1.0)')
    parser.add_argument('--curriculum-value-weight-p3', type=float, default=0.5,
                        help='Value regression weight in phase 3 (default: 0.5)')

    # ==================== FOCAL LOSS (EXTENDED) ====================
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='Alpha for focal loss (default: 0.25)')
    parser.add_argument('--digit-focal-gamma', type=float, default=5.0,
                        help='Gamma for per-digit focal loss (default: 5.0)')
    parser.add_argument('--command-focal-gamma-boost', type=float, default=1.0,
                        help='Extra gamma boost for command head (default: 1.0)')

    # ==================== POSITION WEIGHTING ====================
    parser.add_argument('--use-position-weights', action='store_true', default=True,
                        help='Use position-dependent loss weights (default: True)')
    parser.add_argument('--no-position-weights', action='store_true',
                        help='Ablation: disable position-weighted loss')
    parser.add_argument('--position-weight-scale', type=float, default=3.0,
                        help='Weight for position 0 (decays towards 1.0, default: 3.0)')

    # ==================== SAMPLING / DECODING ====================
    parser.add_argument('--beam-width', type=int, default=5,
                        help='Beam width for beam search (default: 5)')
    parser.add_argument('--beam-length-penalty', type=float, default=0.6,
                        help='Length penalty for beam search (default: 0.6)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='K for top-k sampling (default: 50)')
    parser.add_argument('--nucleus-p', type=float, default=0.9,
                        help='P for nucleus sampling (default: 0.9)')
    parser.add_argument('--repetition-penalty', type=float, default=1.2,
                        help='Penalty factor for repeated tokens (default: 1.2)')
    parser.add_argument('--type-temperature', type=float, default=1.0,
                        help='Temperature for type head (default: 1.0)')
    parser.add_argument('--command-temperature', type=float, default=1.0,
                        help='Temperature for command head (default: 1.0)')
    parser.add_argument('--param-type-temperature', type=float, default=1.0,
                        help='Temperature for param type head (default: 1.0)')
    parser.add_argument('--param-value-temperature', type=float, default=1.0,
                        help='Temperature for param value head (default: 1.0)')

    # ==================== INFERENCE CONSTRAINTS ====================
    parser.add_argument('--inference-type-boost', type=float, default=10.0,
                        help='Logit boost for valid type transitions (default: 10.0)')
    parser.add_argument('--inference-param-boost', type=float, default=10.0,
                        help='Logit boost for required params (default: 10.0)')
    parser.add_argument('--inference-numeric-boost', type=float, default=50.0,
                        help='Logit boost for numeric after param (default: 50.0)')

    # ==================== CONTRASTIVE LOSS ====================
    parser.add_argument('--contrastive-temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss (default: 0.07)')
    parser.add_argument('--use-contrastive-loss', action='store_true',
                        help='Enable contrastive fingerprint loss')
    parser.add_argument('--contrastive-weight', type=float, default=0.3,
                        help='Weight for contrastive loss (default: 0.3)')

    # ==================== MULTI-TASK LOSS ====================
    parser.add_argument('--recon-weight', type=float, default=0.5,
                        help='Reconstruction loss weight (default: 0.5)')
    parser.add_argument('--contrast-weight', type=float, default=0.3,
                        help='Contrastive loss weight (default: 0.3)')
    parser.add_argument('--cls-weight', type=float, default=0.5,
                        help='Classification loss weight (default: 0.5)')
    parser.add_argument('--adaptive-weighting', action='store_true', default=True,
                        help='Use uncertainty-based adaptive weighting (default: True)')

    # ==================== VALIDATION / EVALUATION ====================
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Validate every N epochs (default: 1)')
    parser.add_argument('--val-beam-search', action='store_true',
                        help='Use beam search during validation')
    parser.add_argument('--val-beam-width', type=int, default=3,
                        help='Beam width for validation (default: 3)')
    parser.add_argument('--eval-sequence-acc', action='store_true',
                        help='Compute sequence accuracy (slower)')

    # ==================== LR SCHEDULER (EXTENDED) ====================
    parser.add_argument('--step-lr-step-size', type=int, default=30,
                        help='Step size for StepLR (default: 30)')
    parser.add_argument('--step-lr-gamma', type=float, default=0.1,
                        help='Gamma for StepLR (default: 0.1)')
    parser.add_argument('--cyclic-base-lr-div', type=float, default=10.0,
                        help='Base LR = learning_rate / this (default: 10.0)')

    # ==================== MULTI-HEAD MODEL PARAMS ====================
    parser.add_argument('--n-param-values', type=int, default=100,
                        help='Number of param value classes (default: 100)')
    parser.add_argument('--token-drop-prob', type=float, default=0.0,
                        help='Token drop probability for input noise (default: 0.0)')
    parser.add_argument('--sensor-weight', type=float, default=0.7,
                        help='Weight for sensor vs token classification (default: 0.7)')
    parser.add_argument('--sensor-input-dim', type=int, default=155,
                        help='Dimension of continuous sensor features (default: 155)')
    parser.add_argument('--head-hidden-ratio', type=float, default=0.5,
                        help='Hidden dim ratio for heads (default: 0.5)')

    # ==================== DIGIT VALUE HEAD ====================
    parser.add_argument('--digit-dropout', type=float, default=0.1,
                        help='Dropout for digit value head (default: 0.1)')
    parser.add_argument('--digit-hidden-dim', type=int, default=None,
                        help='Hidden dim for digit MLP (default: d_model)')
    parser.add_argument('--digit-embed-ratio', type=float, default=0.25,
                        help='Digit embedding dim as ratio of d_model (default: 0.25)')
    parser.add_argument('--weight-init-std', type=float, default=0.02,
                        help='Std for weight initialization (default: 0.02)')
    parser.add_argument('--n-digit-positions', type=int, default=6,
                        help='Number of digit positions (default: 6)')

    # ==================== TRAINING LOOP ====================
    parser.add_argument('--show-samples-num', type=int, default=3,
                        help='Number of samples to show in predictions (default: 3)')
    parser.add_argument('--show-samples-max-tokens', type=int, default=10,
                        help='Max tokens to show per sample (default: 10)')

    # ==================== INITIALIZATION ====================
    parser.add_argument('--init-strategy', type=str, default='xavier_uniform',
                        choices=['xavier_uniform', 'xavier_normal', 'kaiming_uniform',
                                'kaiming_normal', 'orthogonal', 'default'])
    parser.add_argument('--init-gain', type=float, default=1.0)

    # ==================== LOGGING ====================
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='gcode-sensor-multihead')
    parser.add_argument('--run-name', type=str, default=None)

    # ==================== DISPLAY OPTIONS ====================
    parser.add_argument('--print-every', type=int, default=10,
                        help='Print sample predictions every N epochs')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of samples to show in predictions')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For fully deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    device = get_device()

    print("=" * 60)
    print("SENSOR MULTI-HEAD DECODER TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")

    # Load config if provided
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
            for k, v in config.items():
                if not hasattr(args, k.replace('-', '_')):
                    setattr(args, k.replace('-', '_'), v)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # WandB
    if args.use_wandb and WANDB_AVAILABLE:
        run_name = args.run_name or f"sensor-multihead-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # ============ LOAD VOCABULARY ============
    print("\nLoading vocabulary...")
    with open(args.vocab_path) as f:
        vocab_data = json.load(f)
    # Handle both formats: direct vocab dict or nested under 'vocab' key
    vocab = vocab_data.get('vocab', vocab_data)
    vocab_size = len(vocab)
    print(f"  Vocabulary size: {vocab_size}")

    # Build id2token mapping for sample predictions
    id2token = {v: k for k, v in vocab.items()}

    # ============ LOAD DATASETS ============
    print("\nLoading datasets...")
    train_dataset_base = DecoderDatasetFromSplits(
        args.split_dir, 'train', max_token_len=args.max_seq_len
    )
    val_dataset = DecoderDatasetFromSplits(
        args.split_dir, 'val', max_token_len=args.max_seq_len
    )
    print(f"  Train samples (base): {len(train_dataset_base)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Apply data augmentation if enabled
    if args.augment:
        print(f"\n  Data augmentation enabled:")
        print(f"    Augment prob: {args.augment_prob}")
        print(f"    Noise level: {args.noise_level}")
        print(f"    Shift range: ±{args.shift_range}")
        print(f"    Scale range: ({args.scale_range_min}, {args.scale_range_max})")
        print(f"    Time warp sigma: {args.time_warp_sigma}")
        print(f"    Feature dropout: {args.feature_dropout_prob}")
        print(f"    Cutout length: {args.cutout_length}")
        print(f"    Jitter sigma: {args.jitter_sigma}")

        augmenter = DataAugmenter(
            noise_level=args.noise_level,
            shift_range=args.shift_range,
            scale_range=(args.scale_range_min, args.scale_range_max),
            augment_prob=args.augment_prob,
            time_warp_sigma=args.time_warp_sigma,
            feature_dropout_prob=args.feature_dropout_prob,
            cutout_length=args.cutout_length,
            jitter_sigma=args.jitter_sigma,
        )

        # Get rare token IDs if oversampling
        rare_token_ids = None
        if args.oversample_rare:
            try:
                rare_token_ids = get_rare_token_ids(args.vocab_path)
                print(f"    Oversampling rare tokens: {args.oversample_factor}x")
            except Exception as e:
                print(f"    Warning: Could not get rare tokens: {e}")
                rare_token_ids = None

        train_dataset = AugmentedGCodeDataset(
            base_dataset=train_dataset_base,
            oversample_rare=args.oversample_rare,
            oversample_factor=args.oversample_factor,
            rare_token_ids=rare_token_ids,
            augmenter=augmenter,
            augment=True,
        )
        print(f"  Train samples (augmented): {len(train_dataset)}")
    else:
        train_dataset = train_dataset_base

    # Create data loaders
    if args.use_error_sampling:
        # Error-focused sampling (Breakthrough Phase 4)
        print("  Using ERROR-FOCUSED sampling (Phase 4 Breakthrough)")
        base_ds = train_dataset_base if args.augment else train_dataset
        sampler = create_error_focused_sampler(base_ds, vocab, error_boost=args.error_boost)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=sampler, collate_fn=decoder_collate_fn
        )
    elif args.use_class_weights:
        # Use base dataset for sampling weights computation
        base_ds = train_dataset_base if args.augment else train_dataset
        sampler = create_class_balanced_sampler(base_ds, power=args.sampler_power)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=sampler, collate_fn=decoder_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, collate_fn=decoder_collate_fn
        )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=decoder_collate_fn
    )

    # ============ LOAD ENCODER ============
    use_enhanced = args.use_enhanced_encoder or args.use_multiscale_encoder

    if use_enhanced:
        print("\nCreating ENHANCED encoder (Phase 1.6 improvements)...")
        print(f"  Multi-scale: {args.encoder_n_scales} scales")
        print(f"  Kernel sizes: {args.encoder_kernel_sizes}")
        print(f"  Dilations: {args.encoder_dilations}")
        print(f"  Multi-head pooling: {args.use_multihead_pooling}")
        if args.use_multihead_pooling:
            print(f"    Heads: {args.pooling_n_heads}, Queries: {args.pooling_n_queries}")
        print(f"  Auxiliary heads: {args.use_auxiliary_heads}")
        if args.use_auxiliary_heads:
            print(f"    Loss weight: {args.auxiliary_loss_weight}")
        print(f"  Unfreeze layers: {args.unfreeze_encoder_layers}")
        if args.unfreeze_encoder_layers != 0:
            print(f"    LR scale: {args.encoder_lr_scale}")

        encoder = EnhancedEncoder(
            input_dim=train_dataset_base.get_sensor_dim(),
            hidden_dim=args.encoder_hidden_dim,
            latent_dim=args.sensor_dim,
            n_operations=args.n_operations,
            # Multi-scale encoder config
            use_multiscale=True,
            n_scales=args.encoder_n_scales,
            kernel_sizes=args.encoder_kernel_sizes,
            dilations=args.encoder_dilations,
            lstm_layers=args.encoder_lstm_layers,
            # Pooling config
            use_multihead_pooling=args.use_multihead_pooling,
            pooling_n_heads=args.pooling_n_heads,
            pooling_n_queries=args.pooling_n_queries,
            # Auxiliary heads
            use_auxiliary_heads=args.use_auxiliary_heads,
            # General
            dropout=args.encoder_dropout,
        )

        # Try to load pretrained weights if available
        if args.encoder_path and Path(args.encoder_path).exists():
            print(f"  Loading pretrained weights from: {args.encoder_path}")
            checkpoint = torch.load(args.encoder_path, map_location=device, weights_only=False)
            # Handle checkpoint format
            if isinstance(checkpoint, dict) and 'encoder_state_dict' in checkpoint:
                state_dict = checkpoint['encoder_state_dict']
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Try to load matching keys (partial load for enhanced encoder)
            missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Note: {len(missing)} keys not found in checkpoint (new enhanced layers)")
            if unexpected:
                print(f"  Note: {len(unexpected)} unexpected keys in checkpoint (ignored)")
        else:
            print("  Initializing enhanced encoder from scratch (no pretrained weights)")

        encoder.to(device)

        # Handle freezing
        if args.unfreeze_encoder_layers == 0:
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            print(f"  Enhanced encoder loaded (fully frozen)")
        else:
            # Partial unfreeze - set encoder to train mode for unfrozen layers
            encoder.train()
            # First freeze all
            for p in encoder.parameters():
                p.requires_grad = False
            # Then unfreeze specified layers via get_unfrozen_params
            unfrozen_params = encoder.get_unfrozen_params(args.unfreeze_encoder_layers)
            for p in unfrozen_params:
                p.requires_grad = True
            print(f"  Enhanced encoder loaded ({len(unfrozen_params)} params unfrozen)")

        n_enc_params = sum(p.numel() for p in encoder.parameters())
        n_enc_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"  Encoder params: {n_enc_params:,} total, {n_enc_trainable:,} trainable")

    else:
        # Standard MM-DTAE-LSTM encoder
        print("\nLoading standard MM-DTAE-LSTM encoder...")
        encoder = MM_DTAE_LSTM(
            input_dim=train_dataset_base.get_sensor_dim(),
            hidden_dim=256,
            latent_dim=args.sensor_dim,
            n_classes=args.n_operations,
        )
        checkpoint = torch.load(args.encoder_path, map_location=device, weights_only=False)
        # Handle checkpoint format (may have model_state_dict or be direct state dict)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            encoder.load_state_dict(checkpoint)
        encoder.to(device)

        # Handle freezing/unfreezing
        if args.unfreeze_encoder_layers == 0:
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            print(f"  Encoder loaded (frozen)")
        else:
            # Partial unfreeze for standard encoder
            encoder.train()
            # First freeze all
            for p in encoder.parameters():
                p.requires_grad = False

            # Unfreeze layers based on arg
            if args.unfreeze_encoder_layers == -1:
                # Unfreeze all
                for p in encoder.parameters():
                    p.requires_grad = True
                print(f"  Encoder loaded (all layers unfrozen)")
            else:
                # Unfreeze last N layers (classification head, bottleneck, etc.)
                modules_to_unfreeze = []
                if args.unfreeze_encoder_layers >= 1:
                    modules_to_unfreeze.append(encoder.classification_head)
                if args.unfreeze_encoder_layers >= 2:
                    modules_to_unfreeze.append(encoder.temporal_attention)
                    modules_to_unfreeze.append(encoder.bottleneck)
                if args.unfreeze_encoder_layers >= 3:
                    modules_to_unfreeze.append(encoder.encoder_lstm)

                for module in modules_to_unfreeze:
                    for p in module.parameters():
                        p.requires_grad = True

                n_unfrozen = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
                print(f"  Encoder loaded ({n_unfrozen:,} params unfrozen)")

    # Store auxiliary head info for training loop
    use_auxiliary = use_enhanced and args.use_auxiliary_heads

    # ============ CREATE DECODER ============
    print("\nCreating decoder...")
    # Log ablation flags if any are active
    ablation_flags = []
    if args.no_operation_conditioning:
        ablation_flags.append("no_op_cond")
    if args.no_cross_attention:
        ablation_flags.append("no_cross_attn")
    if args.no_positional_encoding:
        ablation_flags.append("no_pos_enc")
    if ablation_flags:
        print(f"  ABLATION MODE: {', '.join(ablation_flags)}")

    decoder = SensorMultiHeadDecoder(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        sensor_dim=args.sensor_dim,
        n_operations=args.n_operations,
        n_types=args.n_types,
        n_commands=args.n_commands,
        n_param_types=args.n_param_types,
        dropout=args.dropout,
        embed_dropout=args.embed_dropout,
        max_seq_len=args.max_seq_len,
        # Ablation flags
        no_operation_conditioning=args.no_operation_conditioning,
        no_cross_attention=args.no_cross_attention,
        no_positional_encoding=args.no_positional_encoding,
        # Regularization
        drop_path_rate=args.drop_path_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        # Sensor Value Prior (Phase 2)
        use_sensor_prior=args.use_sensor_prior,
        sensor_prior_weight=args.sensor_prior_weight,
    ).to(device)

    # Apply initialization
    if args.init_strategy != 'default':
        apply_initialization(decoder, args.init_strategy, args.init_gain)

    # ============ TYPE-CONSTRAINED DECODING (Phase 1 Breakthrough) ============
    # Set vocabulary on decoder to enable type-constrained decoding
    # This uses the 99.8% accurate type_head to constrain legacy_logits
    use_type_constraint = args.use_type_constraint and not args.no_type_constraint
    if use_type_constraint:
        print("  Enabling TYPE-CONSTRAINED DECODING...")
        decoder.set_vocab(vocab)
    else:
        print("  Type constraint disabled (ablation mode)")
        decoder.use_type_constraint = False

    # ============ SENSOR VALUE PRIOR (Phase 2 Breakthrough) ============
    if args.use_sensor_prior:
        print(f"  Enabling SENSOR VALUE PRIOR (weight={args.sensor_prior_weight})")

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Decoder parameters: {n_params:,}")

    # ============ TRAINING COMPONENTS ============
    # Pass encoder for potential unfreezing
    optimizer = create_optimizer(decoder, args, encoder=encoder if args.unfreeze_encoder_layers != 0 else None)
    total_steps = len(train_loader) * args.max_epochs
    scheduler = create_scheduler(optimizer, args, total_steps=total_steps)

    # Create class weights for X/Y boost if enabled
    class_weights = None
    if args.xy_boost > 1.0:
        print(f"  Creating class weights with X/Y boost = {args.xy_boost}x")
        class_weights = create_xy_boosted_class_weights(
            args.vocab_path, xy_boost=args.xy_boost, device=device
        )
        n_boosted = (class_weights > 1.0).sum().item()
        print(f"  Boosted {n_boosted} NUM_X/NUM_Y tokens")

    loss_fn = SensorMultiHeadLoss(args, class_weights=class_weights)

    # Print position weight status
    if loss_fn.use_position_weights:
        scale = getattr(args, 'position_weight_scale', 3.0)
        print(f"  POSITION-WEIGHTED LOSS enabled (scale={scale:.1f}x at position 0)")
    else:
        print("  Position weights disabled (ablation mode)")

    # Curriculum and scheduled sampling
    curriculum = CurriculumScheduler(
        n_phases=args.curriculum_phases,
        epochs_per_phase=args.curriculum_epochs_per_phase,
        structure_weight=args.curriculum_structure_weight,
        digit_weight_p2=args.curriculum_digit_weight_p2,
        digit_weight_p3=args.curriculum_digit_weight_p3,
        value_weight_p3=args.curriculum_value_weight_p3
    ) if args.curriculum else None

    scheduled_sampling = ScheduledSampling(
        args.teacher_forcing_start, args.teacher_forcing_end,
        args.max_epochs, args.teacher_forcing_decay
    ) if args.scheduled_sampling else None

    # Grammar constraints
    grammar_constraints = None
    if args.use_grammar_constraints:
        grammar_constraints = GCodeGrammarConstraints(vocab=vocab, device=device)
        print(f"Grammar constraints enabled with weight {args.grammar_weight}")

    # SWA - Full Integration
    swa_model = None
    swa_scheduler = None
    swa_n_updates = 0
    if args.use_swa:
        swa_model = AveragedModel(decoder)
        swa_scheduler = SWALR(
            optimizer,
            swa_lr=args.swa_lr,
            anneal_epochs=args.swa_anneal_epochs,
            anneal_strategy=args.swa_anneal_strategy,
        )
        print(f"\nSWA enabled:")
        print(f"  Start epoch: {args.swa_start_epoch}")
        print(f"  SWA LR: {args.swa_lr}")
        print(f"  Update freq: {args.swa_update_freq}")
        print(f"  Anneal epochs: {args.swa_anneal_epochs}")
        print(f"  Anneal strategy: {args.swa_anneal_strategy}")

    # ============ MIXED PRECISION (AMP) ============
    scaler = None
    if args.use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("\nMixed precision (AMP) enabled for CUDA")
    elif args.use_amp:
        print(f"\nNote: --use-amp requested but device is {device.type}, AMP disabled (requires CUDA)")

    # ============ TRAINING LOOP ============
    best_metric = 0 if args.track_metric != 'loss' else float('inf')
    patience_counter = 0

    # Training history for visualization
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_seq_acc': [],
        'lr': [],
    }

    print("\nStarting training...")
    for epoch in range(args.max_epochs):
        # Log curriculum phase
        if curriculum:
            print(f"\n{curriculum.get_phase_info(epoch)}")

        if scheduled_sampling:
            print(f"Teacher forcing: {scheduled_sampling.get_ratio(epoch):.2%}")

        # Train
        train_metrics = train_epoch(
            encoder, decoder, train_loader, optimizer, loss_fn,
            curriculum, scheduled_sampling, epoch, args, device,
            grammar_constraints=grammar_constraints,
            scaler=scaler,
            use_auxiliary=use_auxiliary
        )

        # Validate
        val_metrics = validate(encoder, decoder, val_loader, loss_fn, args, device)

        # LR scheduler
        if scheduler is not None:
            if args.lr_scheduler == 'plateau':
                if args.track_metric == 'loss':
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step(val_metrics.get(args.track_metric, 0))
            else:
                scheduler.step()

        # SWA - Full Integration
        in_swa_phase = args.use_swa and epoch >= args.swa_start_epoch
        if in_swa_phase:
            # Update SWA model at specified frequency
            if (epoch - args.swa_start_epoch) % args.swa_update_freq == 0:
                swa_model.update_parameters(decoder)
                swa_n_updates += 1
                print(f"  SWA update #{swa_n_updates} applied")

            # Use SWA scheduler instead of normal scheduler
            swa_scheduler.step()

            # Validate SWA model periodically (every 5 SWA updates)
            if swa_n_updates > 0 and swa_n_updates % 5 == 0:
                print(f"\n  Validating SWA model (after {swa_n_updates} updates)...")
                # Update BN stats with a subset of training data
                bn_update_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=decoder_collate_fn
                )
                torch.optim.swa_utils.update_bn(
                    bn_update_loader,
                    swa_model,
                    device=device,
                )
                # Validate SWA model
                swa_val_metrics = validate(encoder, swa_model.module, val_loader, loss_fn, args, device)
                print(f"  SWA Val Token Acc: {swa_val_metrics.get('token', 0):.2%}")
                print(f"  SWA Val Sequence Acc: {swa_val_metrics.get('sequence', 0):.2%}")

                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'swa/val_token_acc': swa_val_metrics.get('token', 0),
                        'swa/val_sequence_acc': swa_val_metrics.get('sequence', 0),
                        'swa/n_updates': swa_n_updates,
                    })

        # Logging
        train_tok = train_metrics.get('token', train_metrics.get('acc', 0))
        val_tok = val_metrics.get('token', 0)
        val_seq = val_metrics.get('sequence', 0)
        train_op = train_metrics.get('encoder_op_acc', 0)
        val_op = val_metrics.get('encoder_op_acc', 0)
        print(f"\nEpoch {epoch+1}/{args.max_epochs}")
        print(f"  Loss:           train={train_metrics['loss']:.4f}  val={val_metrics['loss']:.4f}")
        print(f"  Token Acc:      train={train_tok:.2%}  val={val_tok:.2%}")
        train_seq = train_metrics.get('sequence', 0)
        print(f"  Sequence Acc:   train={train_seq:.2%}  val={val_seq:.2%}")
        print(f"  Encoder Op Acc: train={train_op:.2%}  val={val_op:.2%}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Update training history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_metrics.get('token', train_metrics.get('acc', 0)))
        history['val_acc'].append(val_metrics.get('token', 0))
        history['val_seq_acc'].append(val_metrics.get('sequence', 0))
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Show sample predictions every N epochs
        if (epoch + 1) % args.print_every == 0 or epoch == 0:
            show_sample_predictions(
                encoder, decoder, val_loader, id2token, device,
                num_samples=args.num_samples, max_tokens=8
            )

        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'val/loss': val_metrics['loss'],
                'val/token_acc': val_metrics.get('token', 0),
                'train/encoder_op_acc': train_metrics.get('encoder_op_acc', 0),
                'val/encoder_op_acc': val_metrics.get('encoder_op_acc', 0),
                'lr': optimizer.param_groups[0]['lr'],
            })

        # Checkpointing
        if args.track_metric == 'loss':
            current_metric = val_metrics['loss']
            is_better = current_metric < best_metric
        else:
            current_metric = val_metrics.get(args.track_metric, 0)
            is_better = current_metric > best_metric

        if is_better:
            best_metric = current_metric
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': decoder.state_dict(),
                'encoder_state_dict': encoder.state_dict(),  # Save encoder for ensemble compatibility
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'args': vars(args),
            }, output_dir / 'best_model.pt')
            print(f"  NEW BEST: {args.track_metric} = {best_metric:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # ============ SWA FINALIZATION ============
    if args.use_swa and swa_n_updates > 0:
        print("\n" + "=" * 60)
        print("SWA FINALIZATION")
        print("=" * 60)
        print(f"Total SWA updates: {swa_n_updates}")

        # Update batch norm statistics with full training data
        print("Updating SWA batch norm with full training data...")
        bn_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=decoder_collate_fn,
        )

        # Limit BN update batches if specified
        if args.swa_bn_update_batches is not None:
            from itertools import islice
            bn_loader_limited = list(islice(bn_loader, args.swa_bn_update_batches))
            print(f"  Using {len(bn_loader_limited)} batches for BN update")

            # Manual BN update for limited batches
            swa_model.train()
            with torch.no_grad():
                for batch in tqdm(bn_loader_limited, desc="BN Update"):
                    sensor_features = batch['sensor_features'].to(device)
                    input_tokens = batch['input_tokens'].to(device)
                    padding_mask = batch['padding_mask'].to(device)

                    # Get encoder output
                    sensor_emb, _ = encoder.encode(sensor_features)
                    op_logits, _ = encoder.classify(sensor_emb)
                    operation_type = op_logits.argmax(-1)

                    # Forward pass through SWA model to update BN stats
                    _ = swa_model.module(
                        tokens=input_tokens,
                        sensor_embeddings=sensor_emb,
                        operation_type=operation_type,
                        tgt_key_padding_mask=padding_mask,
                    )
            swa_model.eval()
        else:
            torch.optim.swa_utils.update_bn(bn_loader, swa_model, device=device)

        # Validate SWA model
        print("\nValidating final SWA model...")
        swa_val_metrics = validate(encoder, swa_model.module, val_loader, loss_fn, args, device)
        print(f"  SWA Val Loss: {swa_val_metrics['loss']:.4f}")
        print(f"  SWA Val Token Acc: {swa_val_metrics.get('token', 0):.2%}")
        print(f"  SWA Val Sequence Acc: {swa_val_metrics.get('sequence', 0):.2%}")

        # Save SWA model
        swa_save_path = output_dir / 'swa_model.pt'
        torch.save({
            'model_state_dict': swa_model.module.state_dict(),
            'encoder_state_dict': encoder.state_dict(),  # Save encoder for ensemble compatibility
            'n_updates': swa_n_updates,
            'val_metrics': swa_val_metrics,
            'args': vars(args),
        }, swa_save_path)
        print(f"\nSWA model saved to: {swa_save_path}")

        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'swa/final_val_loss': swa_val_metrics['loss'],
                'swa/final_val_token_acc': swa_val_metrics.get('token', 0),
                'swa/final_val_sequence_acc': swa_val_metrics.get('sequence', 0),
                'swa/total_updates': swa_n_updates,
            })

    # ============ FINAL EVALUATION ============
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    test_dataset = DecoderDatasetFromSplits(
        args.split_dir, 'test', max_token_len=args.max_seq_len
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=decoder_collate_fn
    )

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device, weights_only=False)
    decoder.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = validate(encoder, decoder, test_loader, loss_fn, args, device, comprehensive=True)

    print(f"\nBest Model Test Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Token Acc: {test_metrics.get('token', 0):.2%}")
    print(f"  Sequence Acc: {test_metrics.get('sequence', 0):.2%}")
    print(f"  Encoder Op Acc: {test_metrics.get('encoder_op_acc', 0):.2%}")
    if 'comprehensive' in test_metrics:
        cm = test_metrics['comprehensive']
        print(f"  Precision: {cm.get('precision_macro', 0):.4f}")
        print(f"  Recall: {cm.get('recall_macro', 0):.4f}")
        print(f"  F1: {cm.get('f1_macro', 0):.4f}")
        print(f"  BLEU-1: {cm.get('bleu_1', 0):.4f}")
        print(f"  BLEU-4: {cm.get('bleu_4', 0):.4f}")
        print(f"  Edit Distance: {cm.get('edit_distance', 0):.4f}")
        print(f"  Exact Match: {cm.get('exact_match', 0):.2%}")

    # Also evaluate SWA model on test set if available
    swa_test_metrics = None
    if args.use_swa and swa_n_updates > 0:
        print(f"\nSWA Model Test Results:")
        swa_test_metrics = validate(encoder, swa_model.module, test_loader, loss_fn, args, device, comprehensive=True)
        print(f"  Loss: {swa_test_metrics['loss']:.4f}")
        print(f"  Token Acc: {swa_test_metrics.get('token', 0):.2%}")
        print(f"  Sequence Acc: {swa_test_metrics.get('sequence', 0):.2%}")

        # Compare and recommend
        best_token_acc = test_metrics.get('token', 0)
        swa_token_acc = swa_test_metrics.get('token', 0)
        if swa_token_acc > best_token_acc:
            print(f"\n  * SWA model is better by {(swa_token_acc - best_token_acc)*100:.2f}% token accuracy")
            print(f"  * Recommend using swa_model.pt for deployment")
        else:
            print(f"\n  * Best checkpoint is better by {(best_token_acc - swa_token_acc)*100:.2f}% token accuracy")
            print(f"  * Recommend using best_model.pt for deployment")

    # Save results
    results = {
        'best_val_metric': best_metric,
        'test_metrics': {
            'loss': test_metrics['loss'],
            'token_acc': test_metrics.get('token', 0),
            'sequence_acc': test_metrics.get('sequence', 0),
            'encoder_op_acc': test_metrics.get('encoder_op_acc', 0),
            **(test_metrics.get('comprehensive', {})),
        },
        'args': vars(args),
    }

    if swa_test_metrics is not None:
        results['swa_test_metrics'] = {
            'loss': swa_test_metrics['loss'],
            'token_acc': swa_test_metrics.get('token', 0),
            'sequence_acc': swa_test_metrics.get('sequence', 0),
            'n_updates': swa_n_updates,
            **(swa_test_metrics.get('comprehensive', {})),
        }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save training history for visualization
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {output_dir / 'history.json'}")

    if args.use_wandb and WANDB_AVAILABLE:
        test_log = {
            'test/loss': test_metrics['loss'],
            'test/token_acc': test_metrics.get('token', 0),
            'test/sequence_acc': test_metrics.get('sequence', 0),
        }
        for k, v in test_metrics.get('comprehensive', {}).items():
            test_log[f'test/{k}'] = v
        wandb.log(test_log)
        if swa_test_metrics is not None:
            swa_log = {
                'swa/test_loss': swa_test_metrics['loss'],
                'swa/test_token_acc': swa_test_metrics.get('token', 0),
                'swa/test_sequence_acc': swa_test_metrics.get('sequence', 0),
            }
            for k, v in swa_test_metrics.get('comprehensive', {}).items():
                swa_log[f'swa/test_{k}'] = v
            wandb.log(swa_log)
        wandb.finish()

    print(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
