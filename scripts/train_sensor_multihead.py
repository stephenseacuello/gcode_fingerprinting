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
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.model.digit_value_head import DigitByDigitLoss
from miracle.training.losses import FocalLoss


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

    def __init__(self, n_phases: int = 3, epochs_per_phase: int = 30):
        self.n_phases = n_phases
        self.epochs_per_phase = epochs_per_phase
        self.phase_names = ["Structure Only", "Coarse Digits", "Full Precision"]

    def get_phase(self, epoch: int) -> int:
        return min(epoch // self.epochs_per_phase, self.n_phases - 1)

    def get_loss_weights(self, epoch: int) -> dict:
        phase = self.get_phase(epoch)
        if phase == 0:
            return {'structure': 1.0, 'digit': 0.0, 'value': 0.0}
        elif phase == 1:
            return {'structure': 1.0, 'digit': 0.5, 'value': 0.0}
        else:
            return {'structure': 1.0, 'digit': 1.0, 'value': 0.5}

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


def create_optimizer(model, args):
    """Create optimizer based on args."""
    params = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer == 'adamw':
        return torch.optim.AdamW(
            params, lr=args.learning_rate, weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    elif args.optimizer == 'adam':
        return torch.optim.Adam(
            params, lr=args.learning_rate, betas=(args.beta1, args.beta2)
        )
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(
            params, lr=args.learning_rate, momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(
            params, lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def create_scheduler(optimizer, args, total_steps=None):
    """Create LR scheduler based on args."""
    if args.lr_scheduler == 'none':
        return None
    elif args.lr_scheduler == 'cosine':
        t_max = args.cosine_t_max or (args.max_epochs - args.warmup_epochs)
        return CosineAnnealingLR(optimizer, T_max=t_max)
    elif args.lr_scheduler == 'plateau':
        return ReduceLROnPlateau(
            optimizer, mode='max', factor=args.plateau_factor,
            patience=args.plateau_patience
        )
    elif args.lr_scheduler == 'step':
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.lr_scheduler == 'cyclic':
        return CyclicLR(
            optimizer, base_lr=args.learning_rate/10,
            max_lr=args.learning_rate, cycle_momentum=False
        )
    elif args.lr_scheduler == 'onecycle':
        return OneCycleLR(
            optimizer, max_lr=args.learning_rate, total_steps=total_steps
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.lr_scheduler}")


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
        sensor_emb, _ = encoder.encode(sensor_features)
        op_logits, _ = encoder.classify(sensor_emb)
        op_pred = op_logits.argmax(-1)

        # Decoder forward (teacher forcing)
        outputs = decoder(
            tokens=input_tokens,
            sensor_embeddings=sensor_emb,
            operation_type=op_pred,
        )

        # Get predicted tokens from legacy logits
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

class SensorMultiHeadLoss(nn.Module):
    """Combined loss for all prediction heads."""

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Type loss (with focal option)
        if args.use_focal_loss:
            self.type_loss = FocalLoss(gamma=args.focal_gamma, ignore_index=-1)
        else:
            self.type_loss = nn.CrossEntropyLoss(
                ignore_index=-1, label_smoothing=args.label_smoothing
            )

        # Command loss (extra focal for rare commands)
        if args.use_focal_loss:
            self.command_loss = FocalLoss(gamma=args.focal_gamma + 1, ignore_index=-1)
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
        self.legacy_loss = nn.CrossEntropyLoss(
            ignore_index=PAD_TOKEN_ID, label_smoothing=args.label_smoothing
        )

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
        if 'target_tokens' in targets:
            legacy_logits = outputs['legacy_logits'].view(-1, outputs['legacy_logits'].size(-1))
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
                scheduled_sampling, epoch, args, device):
    """Train one epoch."""
    decoder.train()
    encoder.eval()

    # Get curriculum weights
    curriculum_weights = curriculum.get_loss_weights(epoch) if curriculum else None
    tf_ratio = scheduled_sampling.get_ratio(epoch) if scheduled_sampling else 1.0

    total_loss = 0
    metrics = defaultdict(float)
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        sensor_features = batch['sensor_features'].to(device)
        input_tokens = batch['input_tokens'].to(device)
        target_tokens = batch['target_tokens'].to(device)
        padding_mask = batch['padding_mask'].to(device)

        # Frozen encoder
        with torch.no_grad():
            sensor_emb, _ = encoder.encode(sensor_features)
            op_logits, _ = encoder.classify(sensor_emb)
            operation_type = op_logits.argmax(-1)

        # Decoder forward
        outputs = decoder(
            tokens=input_tokens,
            sensor_embeddings=sensor_emb,
            operation_type=operation_type,
            tgt_key_padding_mask=padding_mask,
        )

        # Build targets
        targets = {
            'target_tokens': target_tokens,
        }

        # Loss computation
        loss, loss_dict = loss_fn(outputs, targets, curriculum_weights)

        # Gradient accumulation
        loss = loss / args.accumulation_steps
        loss.backward()

        if (batch_idx + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
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

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item() * args.accumulation_steps:.4f}",
            'tf': f"{tf_ratio:.2f}"
        })

    return {
        'loss': total_loss / n_batches,
        **{k: v / n_batches for k, v in metrics.items()}
    }


def validate(encoder, decoder, val_loader, loss_fn, args, device):
    """Validate model."""
    decoder.eval()
    encoder.eval()

    total_loss = 0
    correct = defaultdict(int)
    total = defaultdict(int)
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            sensor_features = batch['sensor_features'].to(device)
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            padding_mask = batch['padding_mask'].to(device)

            # Frozen encoder
            sensor_emb, _ = encoder.encode(sensor_features)
            op_logits, _ = encoder.classify(sensor_emb)
            operation_type = op_logits.argmax(-1)

            # Decoder forward
            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=sensor_emb,
                operation_type=operation_type,
                tgt_key_padding_mask=padding_mask,
            )

            # Targets
            targets = {'target_tokens': target_tokens}

            # Loss
            loss, _ = loss_fn(outputs, targets)
            total_loss += loss.item()

            # Legacy token accuracy (main metric for now)
            legacy_pred = outputs['legacy_logits'].argmax(-1)
            valid_mask = target_tokens != PAD_TOKEN_ID
            correct['token'] += (legacy_pred[valid_mask] == target_tokens[valid_mask]).sum().item()
            total['token'] += valid_mask.sum().item()

            n_batches += 1

    accuracies = {k: correct[k] / max(total[k], 1) for k in correct}

    return {
        'loss': total_loss / n_batches,
        **accuracies
    }


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train Sensor Multi-Head Decoder')

    # ==================== MODEL ARCHITECTURE ====================
    parser.add_argument('--d-model', type=int, default=192)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--embed-dropout', type=float, default=0.1)
    parser.add_argument('--sensor-dim', type=int, default=128)
    parser.add_argument('--n-operations', type=int, default=9)
    parser.add_argument('--n-types', type=int, default=4)
    parser.add_argument('--n-commands', type=int, default=6)
    parser.add_argument('--n-param-types', type=int, default=10)
    parser.add_argument('--max-seq-len', type=int, default=32)

    # ==================== COMPONENT ABLATION FLAGS ====================
    parser.add_argument('--no-operation-conditioning', action='store_true',
                        help='Ablation: disable operation embedding conditioning')
    parser.add_argument('--no-cross-attention', action='store_true',
                        help='Ablation: use self-attention only (no sensor memory)')
    parser.add_argument('--no-positional-encoding', action='store_true',
                        help='Ablation: disable positional encoding')

    # ==================== DATA & PATHS ====================
    parser.add_argument('--split-dir', type=str, required=True)
    parser.add_argument('--vocab-path', type=str, required=True)
    parser.add_argument('--encoder-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)

    # ==================== TRAINING BASICS ====================
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--track-metric', type=str, default='token',
                        choices=['loss', 'token', 'composite'])

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
                        choices=['none', 'cosine', 'plateau', 'step', 'cyclic', 'onecycle'])
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--cosine-t-max', type=int, default=None)
    parser.add_argument('--plateau-patience', type=int, default=5)
    parser.add_argument('--plateau-factor', type=float, default=0.5)

    # ==================== REGULARIZATION ====================
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--accumulation-steps', type=int, default=1)
    parser.add_argument('--label-smoothing', type=float, default=0.1)

    # ==================== LOSS FUNCTIONS ====================
    parser.add_argument('--use-focal-loss', action='store_true')
    parser.add_argument('--focal-gamma', type=float, default=3.0)

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

    # ==================== SWA ====================
    parser.add_argument('--use-swa', action='store_true')
    parser.add_argument('--swa-start-epoch', type=int, default=75)
    parser.add_argument('--swa-lr', type=float, default=5e-5)

    # ==================== CURRICULUM LEARNING ====================
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--curriculum-phases', type=int, default=3)
    parser.add_argument('--curriculum-epochs-per-phase', type=int, default=30)

    # ==================== SCHEDULED SAMPLING ====================
    parser.add_argument('--scheduled-sampling', action='store_true')
    parser.add_argument('--teacher-forcing-start', type=float, default=1.0)
    parser.add_argument('--teacher-forcing-end', type=float, default=0.5)
    parser.add_argument('--teacher-forcing-decay', type=str, default='cosine',
                        choices=['linear', 'exponential', 'cosine'])

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


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    args = parse_args()
    device = get_device()

    print("=" * 60)
    print("SENSOR MULTI-HEAD DECODER TRAINING")
    print("=" * 60)
    print(f"Device: {device}")

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
    train_dataset = DecoderDatasetFromSplits(
        args.split_dir, 'train', max_token_len=args.max_seq_len
    )
    val_dataset = DecoderDatasetFromSplits(
        args.split_dir, 'val', max_token_len=args.max_seq_len
    )
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Create data loaders
    if args.use_class_weights:
        sampler = create_class_balanced_sampler(train_dataset, power=args.sampler_power)
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
    print("\nLoading frozen encoder...")
    encoder = MM_DTAE_LSTM(
        input_dim=train_dataset.get_sensor_dim(),
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
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"  Encoder loaded (frozen)")

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
    ).to(device)

    # Apply initialization
    if args.init_strategy != 'default':
        apply_initialization(decoder, args.init_strategy, args.init_gain)

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Decoder parameters: {n_params:,}")

    # ============ TRAINING COMPONENTS ============
    optimizer = create_optimizer(decoder, args)
    total_steps = len(train_loader) * args.max_epochs
    scheduler = create_scheduler(optimizer, args, total_steps=total_steps)
    loss_fn = SensorMultiHeadLoss(args)

    # Curriculum and scheduled sampling
    curriculum = CurriculumScheduler(
        args.curriculum_phases, args.curriculum_epochs_per_phase
    ) if args.curriculum else None

    scheduled_sampling = ScheduledSampling(
        args.teacher_forcing_start, args.teacher_forcing_end,
        args.max_epochs, args.teacher_forcing_decay
    ) if args.scheduled_sampling else None

    # SWA
    if args.use_swa:
        swa_model = AveragedModel(decoder)
        swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)

    # ============ TRAINING LOOP ============
    best_metric = 0 if args.track_metric != 'loss' else float('inf')
    patience_counter = 0

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
            curriculum, scheduled_sampling, epoch, args, device
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

        # SWA
        if args.use_swa and epoch >= args.swa_start_epoch:
            swa_model.update_parameters(decoder)
            swa_scheduler.step()

        # Logging
        print(f"\nEpoch {epoch+1}/{args.max_epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Token Acc: {val_metrics.get('token', 0):.2%}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

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
    if args.use_swa:
        print("\nUpdating SWA batch norm...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device)
        torch.save(swa_model.state_dict(), output_dir / 'swa_model.pt')

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

    test_metrics = validate(encoder, decoder, test_loader, loss_fn, args, device)

    print(f"Test Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Token Acc: {test_metrics.get('token', 0):.2%}")

    # Save results
    results = {
        'best_val_metric': best_metric,
        'test_metrics': test_metrics,
        'args': vars(args),
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({'test/loss': test_metrics['loss'], 'test/token_acc': test_metrics.get('token', 0)})
        wandb.finish()

    print(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
