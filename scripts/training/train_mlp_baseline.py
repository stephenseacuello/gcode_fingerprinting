#!/usr/bin/env python3
"""
MLP Baseline for G-code Fingerprinting.

This script trains a simple MLP baseline that:
1. Flattens temporal sensor data
2. Uses fully-connected layers (no temporal modeling)
3. Outputs token predictions directly

This establishes a lower bound showing the importance of temporal modeling.

Usage:
    python scripts/train_mlp_baseline.py --output-dir outputs/baselines/mlp
    python scripts/train_mlp_baseline.py --hidden-dim 512 --n-layers 3

Author: Claude Code
Date: December 2025
"""

import os
import platform
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

# Enable MPS fallback for Mac
if platform.system() == 'Darwin':
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLPBaseline(nn.Module):
    """
    Simple MLP baseline that flattens temporal data.

    Takes sensor data of shape (batch, seq_len, features) and predicts
    token sequences directly without temporal modeling.
    """

    def __init__(
        self,
        input_dim: int = 155,
        temporal_len: int = 64,
        hidden_dim: int = 512,
        n_layers: int = 3,
        vocab_size: int = 668,
        max_output_len: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.temporal_len = temporal_len
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_output_len = max_output_len

        # Flatten input
        flat_input_dim = input_dim * temporal_len

        # Build MLP layers
        layers = []
        in_dim = flat_input_dim

        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output heads for each position
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size)
            for _ in range(max_output_len)
        ])

        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor, target_len: int = None):
        """
        Forward pass.

        Args:
            x: Sensor data (batch, seq_len, features)
            target_len: Number of output positions to predict

        Returns:
            logits: (batch, target_len, vocab_size)
        """
        batch_size = x.size(0)

        # Flatten temporal dimension
        x_flat = x.view(batch_size, -1)

        # Pad or truncate to expected size
        expected_size = self.input_dim * self.temporal_len
        if x_flat.size(1) < expected_size:
            padding = torch.zeros(batch_size, expected_size - x_flat.size(1), device=x.device)
            x_flat = torch.cat([x_flat, padding], dim=1)
        elif x_flat.size(1) > expected_size:
            x_flat = x_flat[:, :expected_size]

        # Encode
        hidden = self.encoder(x_flat)

        # Generate outputs for each position
        if target_len is None:
            target_len = self.max_output_len

        target_len = min(target_len, self.max_output_len)

        outputs = []
        for i in range(target_len):
            logits = self.output_heads[i](hidden)
            outputs.append(logits)

        # Stack: (batch, target_len, vocab_size)
        return torch.stack(outputs, dim=1)


class MLPDataset(Dataset):
    """Dataset for MLP baseline that loads from split files."""

    def __init__(self, split_dir: str, split: str = 'train'):
        self.split_dir = Path(split_dir)

        # Load data
        data_path = self.split_dir / f'{split}_sequences.npz'
        data = np.load(data_path, allow_pickle=True)

        self.sensor_data = data['continuous']  # (N, 64, 155)
        self.token_ids = data['tokens']  # (N, 7)
        self.operations = data.get('operation_type', None)

        print(f"Loaded {len(self.sensor_data)} {split} samples")

    def __len__(self):
        return len(self.sensor_data)

    def __getitem__(self, idx):
        sensors = torch.tensor(self.sensor_data[idx], dtype=torch.float32)
        tokens = torch.tensor(self.token_ids[idx], dtype=torch.long)

        return {
            'sensors': sensors,
            'tokens': tokens,
        }


def collate_fn(batch):
    """Collate function for MLP dataset."""
    # Get max lengths
    max_sensor_len = max(b['sensors'].size(0) for b in batch)
    max_token_len = max(b['tokens'].size(0) for b in batch)

    # Get feature dim
    feature_dim = batch[0]['sensors'].size(1)

    # Pad sensors
    sensors = torch.zeros(len(batch), max_sensor_len, feature_dim)
    for i, b in enumerate(batch):
        sensors[i, :b['sensors'].size(0)] = b['sensors']

    # Pad tokens (use 0 as padding)
    tokens = torch.zeros(len(batch), max_token_len, dtype=torch.long)
    token_mask = torch.zeros(len(batch), max_token_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        tokens[i, :b['tokens'].size(0)] = b['tokens']
        token_mask[i, :b['tokens'].size(0)] = True

    return {
        'sensors': sensors,
        'tokens': tokens,
        'token_mask': token_mask,
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        sensors = batch['sensors'].to(device)
        tokens = batch['tokens'].to(device)
        mask = batch['token_mask'].to(device)

        optimizer.zero_grad()

        # Forward
        target_len = tokens.size(1)
        logits = model(sensors, target_len)

        # Compute loss (only on valid tokens)
        logits_flat = logits.view(-1, logits.size(-1))
        tokens_flat = tokens.view(-1)
        mask_flat = mask.view(-1)

        # Masked loss
        loss = criterion(logits_flat[mask_flat], tokens_flat[mask_flat])

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item() * mask_flat.sum().item()

        preds = logits_flat.argmax(dim=-1)
        total_correct += ((preds == tokens_flat) & mask_flat).sum().item()
        total_tokens += mask_flat.sum().item()

        pbar.set_postfix({
            'loss': loss.item(),
            'acc': total_correct / max(total_tokens, 1) * 100,
        })

    return {
        'loss': total_loss / max(total_tokens, 1),
        'accuracy': total_correct / max(total_tokens, 1),
    }


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    total_seq_correct = 0
    total_sequences = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            sensors = batch['sensors'].to(device)
            tokens = batch['tokens'].to(device)
            mask = batch['token_mask'].to(device)

            # Forward
            target_len = tokens.size(1)
            logits = model(sensors, target_len)

            # Compute loss
            logits_flat = logits.view(-1, logits.size(-1))
            tokens_flat = tokens.view(-1)
            mask_flat = mask.view(-1)

            loss = criterion(logits_flat[mask_flat], tokens_flat[mask_flat])
            total_loss += loss.item() * mask_flat.sum().item()

            # Token accuracy
            preds = logits_flat.argmax(dim=-1)
            total_correct += ((preds == tokens_flat) & mask_flat).sum().item()
            total_tokens += mask_flat.sum().item()

            # Sequence accuracy
            preds_seq = logits.argmax(dim=-1)  # (batch, seq_len)
            for i in range(tokens.size(0)):
                seq_len = mask[i].sum().item()
                if (preds_seq[i, :seq_len] == tokens[i, :seq_len]).all():
                    total_seq_correct += 1
                total_sequences += 1

    return {
        'loss': total_loss / max(total_tokens, 1),
        'token_accuracy': total_correct / max(total_tokens, 1),
        'sequence_accuracy': total_seq_correct / max(total_sequences, 1),
    }


def main():
    parser = argparse.ArgumentParser(description='Train MLP baseline')
    parser.add_argument('--split-dir', type=str, default='outputs/processed_v3',
                        help='Directory with train/val/test splits')
    parser.add_argument('--vocab-path', type=str, default='data/vocabulary_4digit_full.json',
                        help='Path to vocabulary file')
    parser.add_argument('--output-dir', type=str, default='outputs/baselines/mlp',
                        help='Output directory')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=3,
                        help='Number of MLP layers')
    parser.add_argument('--temporal-len', type=int, default=64,
                        help='Expected temporal length')
    parser.add_argument('--max-output-len', type=int, default=32,
                        help='Maximum output sequence length')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Log to Weights & Biases')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load vocabulary
    with open(args.vocab_path) as f:
        vocab_data = json.load(f)
    # Handle nested vocab structure
    if 'vocab' in vocab_data:
        vocab = vocab_data['vocab']
    elif 'token_to_id' in vocab_data:
        vocab = vocab_data['token_to_id']
    else:
        vocab = vocab_data
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Create datasets
    train_dataset = MLPDataset(args.split_dir, 'train')
    val_dataset = MLPDataset(args.split_dir, 'val')

    # Infer input dimensions from data
    sample = train_dataset[0]
    input_dim = sample['sensors'].size(1)
    print(f"Input dimension: {input_dim}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Create model
    model = MLPBaseline(
        input_dim=input_dim,
        temporal_len=args.temporal_len,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        vocab_size=vocab_size,
        max_output_len=args.max_output_len,
        dropout=args.dropout,
    ).to(device)

    print(f"Model parameters: {model.n_params:,}")

    # Optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    # Initialize wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project='gcode-baselines',
            name=f'mlp_h{args.hidden_dim}_l{args.n_layers}_seed{args.seed}',
            config=vars(args),
        )

    # Training loop
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(args.max_epochs):
        print(f"\nEpoch {epoch + 1}/{args.max_epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']*100:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Token Acc: {val_metrics['token_accuracy']*100:.2f}%, Seq Acc: {val_metrics['sequence_accuracy']*100:.2f}%")

        # Log to wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/token_accuracy': val_metrics['token_accuracy'],
                'val/sequence_accuracy': val_metrics['sequence_accuracy'],
                'lr': optimizer.param_groups[0]['lr'],
            })

        # Check for improvement
        if val_metrics['token_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['token_accuracy']
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_token_accuracy': val_metrics['token_accuracy'],
                'val_sequence_accuracy': val_metrics['sequence_accuracy'],
                'config': vars(args),
            }, output_dir / 'best_model.pt')

            print(f"  NEW BEST: {best_val_acc*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Final evaluation on test set
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)

    test_dataset = MLPDataset(args.split_dir, 'test')
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"Test Token Accuracy: {test_metrics['token_accuracy']*100:.2f}%")
    print(f"Test Sequence Accuracy: {test_metrics['sequence_accuracy']*100:.2f}%")

    # Save results
    results = {
        'model': 'MLP Baseline',
        'config': vars(args),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'total_params': model.n_params,
        'best_epoch': checkpoint['epoch'],
        'val_token_accuracy': checkpoint['val_token_accuracy'],
        'val_sequence_accuracy': checkpoint['val_sequence_accuracy'],
        'test_token_accuracy': test_metrics['token_accuracy'],
        'test_sequence_accuracy': test_metrics['sequence_accuracy'],
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            'test/token_accuracy': test_metrics['token_accuracy'],
            'test/sequence_accuracy': test_metrics['sequence_accuracy'],
        })
        wandb.finish()


if __name__ == '__main__':
    main()
