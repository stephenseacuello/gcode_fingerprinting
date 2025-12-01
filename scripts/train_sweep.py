#!/usr/bin/env python3
"""
Training script for wandb sweeps with class weight support.

This script is designed to work with wandb sweeps and incorporates
the class weight fix for handling severe class imbalance.
"""
import os
import platform

# Enable MPS fallback for Mac
if platform.system() == 'Darwin':
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed")

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.dataset.dataset import GCodeDataset, collate_fn
from miracle.training.losses import MultiTaskLoss, GCodeLoss
from miracle.training.metrics import MetricsTracker
from miracle.utilities.device import get_device, print_device_info


def compute_token_type_accuracy(predictions, targets, vocab_path='data/vocabulary.json'):
    """Compute accuracy by token type for proper monitoring."""
    with open(vocab_path) as f:
        vocab_data = json.load(f)

    vocab = vocab_data.get('vocab', vocab_data.get('token2id', {}))

    # Categorize token IDs
    g_ids = set(idx for token, idx in vocab.items()
                if isinstance(token, str) and token.startswith('G'))
    m_ids = set(idx for token, idx in vocab.items()
                if isinstance(token, str) and token.startswith('M'))
    num_ids = set(idx for token, idx in vocab.items()
                  if isinstance(token, str) and token.startswith('NUM_'))

    # Create masks
    pad_mask = targets == 0
    g_mask = torch.tensor([t.item() in g_ids for t in targets]) & ~pad_mask
    m_mask = torch.tensor([t.item() in m_ids for t in targets]) & ~pad_mask
    num_mask = torch.tensor([t.item() in num_ids for t in targets]) & ~pad_mask
    valid_mask = ~pad_mask

    return {
        'overall_acc': (predictions[valid_mask] == targets[valid_mask]).float().mean().item() if valid_mask.any() else 0,
        'g_command_acc': (predictions[g_mask] == targets[g_mask]).float().mean().item() if g_mask.any() else 0,
        'm_command_acc': (predictions[m_mask] == targets[m_mask]).float().mean().item() if m_mask.any() else 0,
        'numeric_acc': (predictions[num_mask] == targets[num_mask]).float().mean().item() if num_mask.any() else 0,
    }


def log_prediction_samples(predictions, targets, vocab_path='data/vocabulary.json', n_samples=3):
    """Log sample predictions for debugging."""
    with open(vocab_path) as f:
        vocab_data = json.load(f)

    vocab = vocab_data.get('vocab', vocab_data.get('token2id', {}))
    id_to_token = {v: k for k, v in vocab.items()}

    print("\n  Sample Predictions:")
    for i in range(min(n_samples, len(predictions))):
        # Get first 10 non-padding tokens
        pred_seq = predictions[i][:10]
        target_seq = targets[i][:10]

        pred_tokens = [id_to_token.get(int(p), f'UNK_{p}') for p in pred_seq]
        target_tokens = [id_to_token.get(int(t), f'UNK_{t}') for t in target_seq]

        matches = ['✓' if p == t else '✗' for p, t in zip(pred_seq, target_seq)]

        print(f"    Sample {i+1}:")
        print(f"      Target: {' '.join(target_tokens[:10])}")
        print(f"      Pred:   {' '.join(pred_tokens[:10])}")
        print(f"      Match:  {' '.join(matches)}")

    # Count unique tokens
    unique_preds = torch.unique(predictions[predictions != 0])
    unique_targets = torch.unique(targets[targets != 0])

    print(f"\n  Unique tokens predicted: {len(unique_preds)}/{len(vocab)} ({len(unique_preds)/len(vocab)*100:.1f}%)")
    print(f"  Unique tokens in targets: {len(unique_targets)}")

    # WARNING: Check for model collapse
    if len(unique_preds) < 20:
        print(f"  ⚠️  WARNING: Model predicting very few unique tokens! Possible collapse.")
        print(f"     Predicted token IDs: {sorted(unique_preds.tolist())[:20]}")

    return len(unique_preds)


def train_epoch(model, loader, criterion, optimizer, device, scheduler=None, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, desc="Training"):
        continuous = batch['continuous'].to(device)
        categorical = batch['categorical'].float().to(device)
        tokens = batch['tokens'].to(device)
        lengths = batch['lengths'].to(device)

        # Forward with teacher forcing
        # CRITICAL: Model predicts NEXT token at each position
        # Input: tokens[:, :-1] (all but last)
        # Target: tokens[:, 1:] (all but first)
        mods = [continuous, categorical]
        outputs = model(mods=mods, lengths=lengths, gcode_in=tokens[:, :-1], modality_dropout_p=0.1)

        # Loss (G-code prediction only)
        targets = {'gcode_tokens': tokens}
        loss, loss_dict = criterion(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # Step OneCycleLR per batch (CRITICAL for proper warmup/annealing)
        # OneCycleLR must be stepped after EVERY batch, not per epoch
        # Other schedulers (CosineAnnealingLR, ReduceLROnPlateau) are stepped per epoch
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        # Check for NaN/Inf before accumulating
        if torch.isfinite(loss):
            total_loss += loss.item()
        else:
            print(f"⚠️  Non-finite loss detected in training: {loss.item()}")
            continue

        # Collect predictions (FIXED: Align predictions with targets)
        # Model output: [B, T-1, V] predicting next token
        # Targets: tokens[:, 1:] (shift by 1)
        if 'gcode_logits' in outputs:
            preds = torch.argmax(outputs['gcode_logits'], dim=-1)  # [B, T-1]
            target_tokens = tokens[:, 1:]  # [B, T-1] - next tokens
            all_preds.append(preds.cpu().flatten())
            all_targets.append(target_tokens.cpu().flatten())

    # Compute metrics
    metrics = {'loss': total_loss / len(loader)}
    if all_preds:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        token_metrics = compute_token_type_accuracy(all_preds, all_targets)
        metrics.update(token_metrics)

    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, desc="Validation"):
        continuous = batch['continuous'].to(device)
        categorical = batch['categorical'].float().to(device)
        tokens = batch['tokens'].to(device)
        lengths = batch['lengths'].to(device)

        # CRITICAL: Model predicts NEXT token at each position
        # Input: tokens[:, :-1] (all but last)
        # Target: tokens[:, 1:] (all but first)
        mods = [continuous, categorical]
        outputs = model(mods=mods, lengths=lengths, gcode_in=tokens[:, :-1], modality_dropout_p=0.0)

        targets = {'gcode_tokens': tokens}
        loss, loss_dict = criterion(outputs, targets)

        # Check for NaN/Inf before accumulating
        if torch.isfinite(loss):
            total_loss += loss.item()
        else:
            print(f"⚠️  Non-finite loss detected in validation: {loss.item()}")

        # Collect predictions (FIXED: Align predictions with targets)
        # Model output: [B, T-1, V] predicting next token
        # Targets: tokens[:, 1:] (shift by 1)
        if 'gcode_logits' in outputs:
            preds = torch.argmax(outputs['gcode_logits'], dim=-1)  # [B, T-1]
            target_tokens = tokens[:, 1:]  # [B, T-1] - next tokens
            all_preds.append(preds.cpu().flatten())
            all_targets.append(target_tokens.cpu().flatten())

    metrics = {'loss': total_loss / len(loader)}
    if all_preds:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        token_metrics = compute_token_type_accuracy(all_preds, all_targets)
        metrics.update(token_metrics)

    return metrics


def train():
    """Main training function for wandb sweep."""
    # Initialize wandb
    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb is required for sweeps. Install with: pip install wandb")

    wandb.init()
    config = wandb.config

    # Set default values if not running in a sweep
    default_config = {
        'batch_size': 32,
        'hidden_dim': 384,
        'num_layers': 3,
        'num_heads': 6,
        'learning_rate': 0.0003,
        'weight_decay': 0.01,
        'use_class_weights': True,
        'class_weight_alpha': 2.0,
        'use_focal_loss': False,
        'focal_gamma': 2.0,
        'label_smoothing': 0.05,
        'max_epochs': 30,
        'patience': 10,
        'grad_clip': 1.0,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
    }

    # Fill in missing config values with defaults
    for key, value in default_config.items():
        if not hasattr(config, key):
            setattr(config, key, value)

    # Setup device
    device = get_device()
    print_device_info(device)

    # Load datasets
    data_dir = Path('data')
    print("Loading datasets...")
    train_dataset = GCodeDataset(data_dir / 'train_sequences.npz')
    val_dataset = GCodeDataset(data_dir / 'val_sequences.npz')

    # Load metadata
    with open(data_dir / 'train_sequences_metadata.json') as f:
        metadata = json.load(f)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Vocab size: {metadata['vocab_size']}")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Model
    model_config = ModelConfig(
        sensor_dims=[metadata['n_continuous_features'], metadata['n_categorical_features']],
        d_model=config.hidden_dim,
        lstm_layers=config.num_layers,
        gcode_vocab=metadata['vocab_size'],
        n_heads=config.num_heads,
        fp_dim=128,
    )
    model = MM_DTAE_LSTM(model_config).to(device)
    print(f"Model parameters: {model.count_params(model):,}")

    # Load class weights if enabled
    class_weights = None
    if config.use_class_weights:
        class_weights_path = data_dir / 'class_weights.pt'
        if class_weights_path.exists():
            class_weights = torch.load(class_weights_path, map_location='cpu', weights_only=True)
            # Scale weights by alpha
            if hasattr(config, 'class_weight_alpha'):
                class_weights = class_weights * config.class_weight_alpha / 2.0  # Normalize to original 2.0
            class_weights = class_weights.to(device)
            print(f"✅ Loaded class weights (min={class_weights.min():.2f}, max={class_weights.max():.2f})")
        else:
            print("⚠️  Class weights requested but not found! Run train_with_class_weights.py first")

    # Loss with class weights
    criterion = MultiTaskLoss(
        vocab_size=metadata['vocab_size'],
        pad_token_id=0,
        adaptive=False,
    )

    # Override gcode_loss with weighted version
    use_focal = getattr(config, 'use_focal_loss', False)
    criterion.gcode_loss = GCodeLoss(
        vocab_size=metadata['vocab_size'],
        pad_token_id=0,
        label_smoothing=getattr(config, 'label_smoothing', 0.05),
        class_weights=class_weights,
        use_focal_loss=use_focal,
        focal_gamma=getattr(config, 'focal_gamma', 2.0),
    )

    # Optimizer
    opt_name = getattr(config, 'optimizer', 'adamw')
    if opt_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

    # Scheduler
    # Supports 3 scheduler types:
    # - 'cosine': CosineAnnealingLR for smooth learning rate decay
    # - 'plateau': ReduceLROnPlateau for adaptive LR reduction when validation plateaus
    # - 'onecycle': OneCycleLR for super-convergence (modern best practice)
    #
    # OneCycleLR Usage:
    #   - Provides automatic warmup + cosine decay in a single scheduler
    #   - Requires steps_per_epoch to calculate total steps
    #   - Must be stepped AFTER EACH BATCH (not per epoch) - see line 98
    #   - Proven to achieve better performance with less tuning (Smith, 2019)
    #   - Recommended for new experiments
    scheduler = None
    sched_name = getattr(config, 'scheduler', 'cosine')
    max_epochs = getattr(config, 'max_epochs', 30)

    if sched_name == 'cosine':
        # Cosine annealing: smooth decay from initial LR to eta_min
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=config.learning_rate * 0.01
        )
    elif sched_name == 'plateau':
        # Adaptive: reduces LR when validation loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
    elif sched_name == 'onecycle':
        # OneCycleLR: combines warmup + cosine annealing with momentum cycling
        # NOTE: This scheduler must be stepped per-batch (see training loop below)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,  # Peak learning rate
            epochs=max_epochs,
            steps_per_epoch=len(train_loader),  # Required for per-batch stepping
        )

    # Training loop
    best_g_acc = 0.0
    patience = getattr(config, 'patience', 10)
    patience_counter = 0

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=scheduler,
            grad_clip=getattr(config, 'grad_clip', 1.0)
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Step scheduler (except OneCycleLR)
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'train/loss': train_metrics['loss'],
            'train/overall_acc': train_metrics['overall_acc'],
            'train/g_command_acc': train_metrics['g_command_acc'],
            'train/m_command_acc': train_metrics['m_command_acc'],
            'train/numeric_acc': train_metrics['numeric_acc'],
            'val/loss': val_metrics['loss'],
            'val/overall_acc': val_metrics['overall_acc'],
            'val/g_command_acc': val_metrics['g_command_acc'],  # KEY METRIC!
            'val/m_command_acc': val_metrics['m_command_acc'],
            'val/numeric_acc': val_metrics['numeric_acc'],
        })

        # Print
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, G-cmd Acc: {train_metrics['g_command_acc']*100:.1f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, G-cmd Acc: {val_metrics['g_command_acc']*100:.1f}%")

        # Log prediction samples every 5 epochs for debugging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("\n  Logging prediction samples...")
            model.eval()
            with torch.no_grad():
                # Get a small batch for logging
                batch = next(iter(val_loader))
                continuous = batch['continuous'].to(device)
                categorical = batch['categorical'].float().to(device)
                tokens = batch['tokens'].to(device)
                lengths = batch['lengths'].to(device)

                mods = [continuous, categorical]
                outputs = model(mods=mods, lengths=lengths, gcode_in=tokens[:, :-1], modality_dropout_p=0.0)

                preds = torch.argmax(outputs['gcode_logits'], dim=-1)  # [B, T-1]
                target_tokens = tokens[:, 1:]  # [B, T-1]

                # Log samples
                n_unique = log_prediction_samples(preds, target_tokens)

                # Log unique token count to wandb
                wandb.log({'unique_tokens_predicted': n_unique})

                # Early stopping if model has collapsed
                if n_unique < 20:
                    print(f"\n  ⚠️  EARLY STOP: Model collapsed (only {n_unique} unique tokens)")
                    print(f"  This indicates a critical training failure. Stopping early.")
                    break

        # Early stopping based on G-command accuracy
        if val_metrics['g_command_acc'] > best_g_acc:
            best_g_acc = val_metrics['g_command_acc']
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'g_command_acc': best_g_acc,
                'config': model_config.__dict__,
            }, f'outputs/sweep_best_{wandb.run.id}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break

    wandb.log({'best_g_command_acc': best_g_acc})
    print(f"\n✅ Training complete! Best G-command accuracy: {best_g_acc*100:.1f}%")


if __name__ == '__main__':
    train()
