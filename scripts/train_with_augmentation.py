#!/usr/bin/env python3
"""
Training script with data augmentation - Phase 2

This script uses:
- Vocabulary v2 (170 tokens with 2-digit bucketing)
- AugmentedGCodeDataset with 3x oversampling for G/M commands
- Sensor noise, temporal shift, magnitude scaling

Run with: python train_with_augmentation.py --use-wandb
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

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.dataset.dataset import GCodeDataset, collate_fn
from miracle.dataset.data_augmentation import AugmentedGCodeDataset, get_rare_token_ids
from miracle.training.losses import MultiTaskLoss, GCodeLoss
from miracle.utilities.device import get_device, print_device_info

# Import helper functions from train_sweep
from train_sweep import train_epoch, validate


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
        'scheduler': training_args.get('scheduler', None),
        'label_smoothing': 0.0,  # Can add to config later
    }

    return config


def main():
    parser = argparse.ArgumentParser(description='Train with data augmentation')
    parser.add_argument('--config', type=str, default='configs/phase1_best.json',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='outputs/processed_v2',
                        help='Path to preprocessed data directory')
    parser.add_argument('--vocab-path', type=str, default='data/gcode_vocab_v2.json',
                        help='Path to vocabulary file')
    parser.add_argument('--output-dir', type=str, default='outputs/augmented_v2',
                        help='Output directory for checkpoints')
    parser.add_argument('--oversample-factor', type=int, default=3,
                        help='Oversampling factor for rare G/M commands')
    parser.add_argument('--max-epochs', type=int, default=50,
                        help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable W&B logging')
    parser.add_argument('--wandb-project', type=str, default='gcode-fingerprinting',
                        help='W&B project name')
    parser.add_argument('--run-name', type=str, default='augmented-vocab-v2',
                        help='W&B run name')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRAINING WITH DATA AUGMENTATION - PHASE 2")
    print("=" * 80)

    # Load config
    print(f"\nLoading config from: {args.config}")
    config = load_config(Path(args.config))
    print(f"✅ Config loaded:")
    print(f"   hidden_dim: {config['hidden_dim']}")
    print(f"   num_layers: {config['num_layers']}")
    print(f"   num_heads: {config['num_heads']}")
    print(f"   batch_size: {config['batch_size']}")
    print(f"   learning_rate: {config['learning_rate']}")
    print(f"   optimizer: {config['optimizer']}")
    print(f"   scheduler: {config['scheduler']}")
    print(f"   oversample_factor: {args.oversample_factor}x")

    # Setup device
    device = get_device()
    print()
    print_device_info(device)

    # Initialize W&B
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                **config,
                'oversample_factor': args.oversample_factor,
                'vocab_size': 170,
                'data_augmentation': True,
            }
        )
        print("✅ W&B initialized")
    else:
        print("⚠️  W&B disabled")

    # Load base datasets
    data_dir = Path(args.data_dir)
    print(f"\nLoading datasets from: {data_dir}")

    train_base = GCodeDataset(data_dir / 'train_sequences.npz')
    val_base = GCodeDataset(data_dir / 'val_sequences.npz')

    # Get rare token IDs (G/M commands)
    rare_token_ids = get_rare_token_ids(args.vocab_path)
    print(f"✅ Found {len(rare_token_ids)} rare token IDs (G/M commands)")

    # Wrap with augmentation
    train_dataset = AugmentedGCodeDataset(
        base_dataset=train_base,
        oversample_rare=True,
        oversample_factor=args.oversample_factor,
        rare_token_ids=rare_token_ids,
        augment=True,
    )

    # Validation: no augmentation, no oversampling
    val_dataset = val_base

    # Compute dataset dimensions
    n_continuous = train_base.continuous.size(-1)
    n_categorical = train_base.categorical.size(-1)
    vocab_size = 170  # From vocab v2

    print(f"✅ Datasets loaded:")
    print(f"   Train: {len(train_base)} base → {len(train_dataset)} augmented")
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

    # Create model
    print("\nCreating model...")
    model_config = ModelConfig(
        sensor_dims=[n_continuous, n_categorical],
        d_model=config['hidden_dim'],
        lstm_layers=config['num_layers'],
        gcode_vocab=vocab_size,
        n_heads=config['num_heads'],
    )

    model = MM_DTAE_LSTM(model_config).to(device)

    print(f"✅ Model created:")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   d_model: {config['hidden_dim']}")
    print(f"   LSTM layers: {config['num_layers']}")
    print(f"   Attention heads: {config['num_heads']}")
    print(f"   Flash Attention: Enabled")
    print(f"   Attention Pooling: Enabled")

    # Create loss function
    loss_fn = MultiTaskLoss(
        vocab_size=vocab_size,
        pad_token_id=0,
        adaptive=False,
    )

    # Create optimizer
    if config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
        )

    print(f"✅ Optimizer: {config['optimizer']}")

    # Create scheduler
    if config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_epochs,
        )
    elif config['scheduler'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'],
            epochs=args.max_epochs,
            steps_per_epoch=len(train_loader),
        )
    else:
        scheduler = None

    print(f"✅ Scheduler: {config['scheduler']}")
    print(f"✅ Learning rate: {config['learning_rate']}")

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print()

    best_val_acc = 0.0
    best_g_cmd_acc = 0.0
    patience_counter = 0

    for epoch in range(args.max_epochs):
        print("=" * 80)
        print(f"Epoch {epoch + 1}/{args.max_epochs}")
        print("=" * 80)

        # Train
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, device,
            scheduler=scheduler if config['scheduler'] == 'onecycle' else None,
            grad_clip=args.grad_clip,
        )

        # Validate
        val_metrics = validate(
            model, val_loader, loss_fn, device,
        )

        # Step scheduler (if not OneCycleLR)
        if scheduler is not None and config['scheduler'] != 'onecycle':
            scheduler.step()

        # Print metrics
        print(f"\nTrain Metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Overall Acc: {train_metrics['overall_acc']:.2%}")
        print(f"  G-command Acc: {train_metrics['g_command_acc']:.2%}")
        print(f"  M-command Acc: {train_metrics['m_command_acc']:.2%}")
        print(f"  Numeric Acc: {train_metrics['numeric_acc']:.2%}")

        print(f"\nValidation Metrics:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Overall Acc: {val_metrics['overall_acc']:.2%}")
        print(f"  G-command Acc: {val_metrics['g_command_acc']:.2%}")
        print(f"  M-command Acc: {val_metrics['m_command_acc']:.2%}")
        print(f"  Numeric Acc: {val_metrics['numeric_acc']:.2%}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Log to W&B
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/overall_acc': train_metrics['overall_acc'],
                'train/g_command_acc': train_metrics['g_command_acc'],
                'train/m_command_acc': train_metrics['m_command_acc'],
                'train/numeric_acc': train_metrics['numeric_acc'],
                'val/loss': val_metrics['loss'],
                'val/overall_acc': val_metrics['overall_acc'],
                'val/g_command_acc': val_metrics['g_command_acc'],
                'val/m_command_acc': val_metrics['m_command_acc'],
                'val/numeric_acc': val_metrics['numeric_acc'],
                'learning_rate': optimizer.param_groups[0]['lr'],
                'unique_tokens_predicted': val_metrics.get('unique_tokens', 0),
                'best_overall_acc': best_val_acc,
                'best_g_command_acc': best_g_cmd_acc,
            })

        # Early stopping
        if val_metrics['overall_acc'] > best_val_acc:
            best_val_acc = val_metrics['overall_acc']
            patience_counter = 0

            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config,
            }, output_dir / 'checkpoint_best.pt')

            print(f"\n✅ New best validation accuracy: {best_val_acc:.2%}")

        else:
            patience_counter += 1

        if val_metrics['g_command_acc'] > best_g_cmd_acc:
            best_g_cmd_acc = val_metrics['g_command_acc']

        if patience_counter >= args.patience:
            print(f"\n⚠️  Early stopping: No improvement for {args.patience} epochs")
            break

        print()

    # Training complete
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best G-command Accuracy: {best_g_cmd_acc:.2%}")
    print(f"Best Overall Accuracy: {best_val_acc:.2%}")
    print(f"Checkpoint saved to: {output_dir / 'checkpoint_best.pt'}")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    print("\n✅ Training script completed successfully!")
    print("\nNext steps:")
    print(f"1. Evaluate on test set: python test_evaluation.py")
    print(f"2. Load checkpoint: {output_dir / 'checkpoint_best.pt'}")


if __name__ == '__main__':
    main()
