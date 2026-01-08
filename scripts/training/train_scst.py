#!/usr/bin/env python3
"""
SCST Fine-tuning for Sensor Multi-Head Decoder.

This script performs Self-Critical Sequence Training (SCST) to optimize
sequence-level metrics. It should be run AFTER initial cross-entropy training.

SCST directly optimizes sequence accuracy using REINFORCE with a greedy baseline:
- Sample: Random sequence from model
- Baseline: Greedy decoded sequence
- Reward: Sequence accuracy (full match)
- Loss: -log_prob * (sample_reward - baseline_reward)

Usage:
    python scripts/train_scst.py \
        --checkpoint outputs/sensor_multihead_v3/best_model.pt \
        --split-dir outputs/stratified_splits_v2 \
        --vocab-path data/vocabulary_4digit_hybrid.json \
        --encoder-path outputs/mm_dtae_lstm_v2/best_model.pt \
        --output-dir outputs/sensor_multihead_scst

Author: Claude Code
Date: December 2025
"""

import os
import platform
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Enable MPS fallback for Mac
if platform.system() == 'Darwin':
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from miracle.dataset.decoder_dataset import (
    DecoderDatasetFromSplits,
    decoder_collate_fn,
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
)
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.training.reinforce import SCSTTrainer


# ============================================================================
# MM-DTAE-LSTM Encoder (copied from train_sensor_multihead.py)
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


def load_encoder(encoder_path, device):
    """Load frozen encoder."""
    checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder = MM_DTAE_LSTM(
        input_dim=155,
        hidden_dim=256,
        latent_dim=128,
        n_classes=9,
        num_lstm_layers=2,
        dropout=0.3,
        bidirectional=True,
    )
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder = encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def parse_args():
    parser = argparse.ArgumentParser(description='SCST Fine-tuning')

    # Required paths
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pre-trained decoder checkpoint')
    parser.add_argument('--split-dir', type=str, required=True)
    parser.add_argument('--vocab-path', type=str, required=True)
    parser.add_argument('--encoder-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)

    # SCST parameters
    parser.add_argument('--scst-epochs', type=int, default=20,
                        help='Number of SCST fine-tuning epochs')
    parser.add_argument('--sample-temperature', type=float, default=0.8,
                        help='Temperature for sampling during SCST')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                        help='Scale factor for rewards')
    parser.add_argument('--mixed-training', action='store_true',
                        help='Use mixed CE + SCST loss')
    parser.add_argument('--ce-weight', type=float, default=0.3,
                        help='Weight for CE loss in mixed training')
    parser.add_argument('--scst-weight', type=float, default=0.7,
                        help='Weight for SCST loss in mixed training')

    # Training
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (smaller for SCST due to memory)')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--grad-clip', type=float, default=0.5,
                        help='Gradient clipping (tighter for SCST)')
    parser.add_argument('--max-seq-len', type=int, default=32)

    # Logging
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='gcode-scst-finetune')
    parser.add_argument('--run-name', type=str, default=None)

    return parser.parse_args()


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def evaluate(encoder, decoder, val_loader, device, scst_trainer):
    """Evaluate model on validation set."""
    decoder.eval()
    encoder.eval()

    total_samples = 0
    seq_correct = 0
    token_correct = 0
    token_total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            sensor_features = batch['sensor_features'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            sensor_emb, _ = encoder.encode(sensor_features)
            op_logits, _ = encoder.classify(sensor_emb)
            operation_type = op_logits.argmax(-1)

            # Generate sequences
            generated, _ = decoder.generate(
                sensor_embeddings=sensor_emb,
                operation_type=operation_type,
                bos_token_id=BOS_TOKEN_ID,
                eos_token_id=EOS_TOKEN_ID,
                max_length=32,
                top_k=1,  # Greedy
            )

            # Compute sequence accuracy
            B = target_tokens.size(0)
            for b in range(B):
                target_mask = target_tokens[b] != PAD_TOKEN_ID
                target_seq = target_tokens[b][target_mask]

                pred_seq = generated[b]
                if pred_seq[0] == BOS_TOKEN_ID:
                    pred_seq = pred_seq[1:]

                # Token accuracy
                compare_len = min(len(pred_seq), len(target_seq))
                if compare_len > 0:
                    correct = (pred_seq[:compare_len] == target_seq[:compare_len]).sum().item()
                    token_correct += correct
                    token_total += len(target_seq)

                # Sequence accuracy
                if len(pred_seq) >= len(target_seq):
                    if (pred_seq[:len(target_seq)] == target_seq).all():
                        seq_correct += 1
                total_samples += 1

    return {
        'sequence_accuracy': seq_correct / max(total_samples, 1),
        'token_accuracy': token_correct / max(token_total, 1),
    }


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    with open(args.vocab_path) as f:
        vocab_data = json.load(f)
    # Handle both flat vocab and nested vocab with 'vocab' key
    vocab = vocab_data.get('vocab', vocab_data) if isinstance(vocab_data, dict) and 'vocab' in vocab_data else vocab_data
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Load encoder
    encoder = load_encoder(args.encoder_path, device)
    print("Loaded frozen encoder")

    # Load pre-trained decoder
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    decoder_args = checkpoint.get('args', {})

    decoder = SensorMultiHeadDecoder(
        vocab_size=vocab_size,
        d_model=decoder_args.get('d_model', 192),
        n_heads=decoder_args.get('n_heads', 8),
        n_layers=decoder_args.get('n_layers', 4),
        sensor_dim=decoder_args.get('sensor_dim', 128),
        n_operations=decoder_args.get('n_operations', 9),
        n_types=decoder_args.get('n_types', 4),
        n_commands=decoder_args.get('n_commands', 6),
        n_param_types=decoder_args.get('n_param_types', 10),
        dropout=decoder_args.get('dropout', 0.3),
        embed_dropout=decoder_args.get('embed_dropout', 0.1),
        max_seq_len=args.max_seq_len,
        # Include features used during training
        use_sensor_prior=decoder_args.get('use_sensor_prior', False),
        sensor_prior_weight=decoder_args.get('sensor_prior_weight', 0.5),
        max_int_digits=decoder_args.get('max_int_digits', 2),
        n_decimal_digits=decoder_args.get('n_decimal_digits', 4),
    )
    # Load with strict=False to handle type_token_mask buffer
    missing, unexpected = decoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if unexpected:
        print(f"Note: {len(unexpected)} unexpected keys ignored (e.g., type_token_mask buffer)")
    decoder = decoder.to(device)
    print(f"Loaded pre-trained decoder from {args.checkpoint}")

    # Load datasets
    train_dataset = DecoderDatasetFromSplits(
        split_dir=args.split_dir,
        split='train',
        max_token_len=args.max_seq_len
    )
    val_dataset = DecoderDatasetFromSplits(
        split_dir=args.split_dir,
        split='val',
        max_token_len=args.max_seq_len
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=decoder_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=decoder_collate_fn
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Initialize SCST trainer
    scst_trainer = SCSTTrainer(
        pad_token_id=PAD_TOKEN_ID,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        max_length=args.max_seq_len,
        sample_temperature=args.sample_temperature,
        reward_scale=args.reward_scale,
    )

    # Optimizer (lower LR for fine-tuning)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # Initialize wandb
    if args.use_wandb and WANDB_AVAILABLE:
        run_name = args.run_name or f"scst_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Initial evaluation
    print("\nInitial evaluation...")
    initial_metrics = evaluate(encoder, decoder, val_loader, device, scst_trainer)
    print(f"Initial - Seq Acc: {initial_metrics['sequence_accuracy']:.4f}, "
          f"Token Acc: {initial_metrics['token_accuracy']:.4f}")

    # SCST Fine-tuning
    best_seq_acc = initial_metrics['sequence_accuracy']

    print(f"\nStarting SCST fine-tuning for {args.scst_epochs} epochs...")
    for epoch in range(args.scst_epochs):
        decoder.train()
        total_loss = 0
        metrics = defaultdict(float)
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.scst_epochs}")
        for batch in pbar:
            sensor_features = batch['sensor_features'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            optimizer.zero_grad()

            # Compute SCST loss
            loss, batch_metrics = scst_trainer.compute_scst_loss(
                encoder=encoder,
                decoder=decoder,
                sensor_features=sensor_features,
                target_tokens=target_tokens,
                device=device,
            )

            loss.backward()

            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)

            optimizer.step()

            total_loss += loss.item()
            for k, v in batch_metrics.items():
                metrics[k] += v
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'adv': f"{batch_metrics['advantage']:.3f}",
            })

        # Average metrics
        avg_metrics = {k: v / n_batches for k, v in metrics.items()}
        avg_metrics['loss'] = total_loss / n_batches

        # Validation
        val_metrics = evaluate(encoder, decoder, val_loader, device, scst_trainer)

        # Logging
        print(f"\nEpoch {epoch+1}: Loss={avg_metrics['loss']:.4f}, "
              f"Val Seq Acc={val_metrics['sequence_accuracy']:.4f}, "
              f"Val Token Acc={val_metrics['token_accuracy']:.4f}")

        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': avg_metrics['loss'],
                'train/sample_reward': avg_metrics.get('sample_reward', 0),
                'train/baseline_reward': avg_metrics.get('baseline_reward', 0),
                'train/advantage': avg_metrics.get('advantage', 0),
                'val/sequence_accuracy': val_metrics['sequence_accuracy'],
                'val/token_accuracy': val_metrics['token_accuracy'],
            })

        # Save best model
        if val_metrics['sequence_accuracy'] > best_seq_acc:
            best_seq_acc = val_metrics['sequence_accuracy']
            save_path = output_dir / 'best_model.pt'
            torch.save({
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_seq_acc': best_seq_acc,
                'val_metrics': val_metrics,
                'args': vars(args),
            }, save_path)
            print(f"Saved best model with seq_acc={best_seq_acc:.4f}")

    # Final save
    final_path = output_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.scst_epochs,
        'best_seq_acc': best_seq_acc,
        'args': vars(args),
    }, final_path)

    print(f"\nSCST fine-tuning complete!")
    print(f"Best sequence accuracy: {best_seq_acc:.4f}")
    print(f"Improvement: {best_seq_acc - initial_metrics['sequence_accuracy']:.4f}")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    main()
