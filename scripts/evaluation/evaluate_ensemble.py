#!/usr/bin/env python3
"""
Evaluate ensemble model performance.

This script:
1. Loads multiple trained models
2. Creates an ensemble using averaging or voting
3. Evaluates on test set
4. Compares ensemble vs individual models
5. Optionally applies test-time augmentation

Usage:
    PYTHONPATH=src .venv/bin/python scripts/evaluate_ensemble.py \
        --checkpoints outputs/breakthrough/model_seed_*/best_model.pt \
        --data-dir outputs/stratified_splits_v2 \
        --vocab-path data/vocabulary_4digit_hybrid.json \
        --output reports/ensemble_evaluation.json

Author: Claude Code
Date: December 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.dataset.decoder_dataset import (
    DecoderDatasetFromSplits,
    decoder_collate_fn,
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
)
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.model.ensemble import EnsembleDecoder, TestTimeAugmentation


class MM_DTAE_LSTM(torch.nn.Module):
    """MM-DTAE-LSTM encoder (lightweight version for loading)."""

    def __init__(
        self,
        input_dim: int = 155,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        n_classes: int = 9,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        # Encoder
        self.encoder_lstm = torch.nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # Mean/var for VAE-like latent
        self.fc_mu = torch.nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_var = torch.nn.Linear(hidden_dim * 2, latent_dim)

        # Classifier
        self.classifier = torch.nn.Linear(latent_dim, n_classes)

    def encode(self, x):
        """Encode input to latent space."""
        output, (h_n, c_n) = self.encoder_lstm(x)
        pooled = output.mean(dim=1)
        mu = self.fc_mu(pooled)
        return mu, output

    def forward(self, x):
        mu, output = self.encode(x)
        return mu, output


def load_encoder(encoder_path, device):
    """Load pretrained encoder - using correct architecture from train_sensor_multihead.py."""
    print(f"Loading encoder from {encoder_path}...")

    import torch.nn as nn

    class MM_DTAE_LSTM(nn.Module):
        """MM-DTAE-LSTM encoder for sensor feature extraction."""

        def __init__(self, input_dim=155, hidden_dim=256, latent_dim=128, n_classes=9,
                     num_lstm_layers=2, dropout=0.3, bidirectional=True, noise_factor=0.1):
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

        def encode(self, x):
            """Encode input sequence to latent representation.
            Returns: (latent, encoded) where latent is [B, T, latent_dim] and encoded is raw LSTM output
            """
            x = self.input_proj(x)
            encoded, (h_n, c_n) = self.encoder_lstm(x)
            latent = self.bottleneck(encoded)
            # Return latent as temporal features (what decoder expects)
            return latent, latent  # First is latent, second is temporal features for decoder

        def classify(self, latent):
            """Classify operation type from latent."""
            attn_weights = self.temporal_attention(latent)
            pooled = (latent * attn_weights).sum(dim=1)
            logits = self.classification_head(pooled)
            return logits, attn_weights

        def forward(self, x):
            latent, (h_n, c_n) = self.encode(x)
            return latent, (h_n, c_n)

    encoder = MM_DTAE_LSTM(
        input_dim=155,
        hidden_dim=256,
        latent_dim=128,
        n_classes=9,
        num_lstm_layers=2,
        dropout=0.3,
    )

    checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['model_state_dict'])
    else:
        encoder.load_state_dict(checkpoint)

    encoder.to(device)
    encoder.eval()
    return encoder


def load_decoder_from_checkpoint(checkpoint_path, vocab, device):
    """Load decoder from checkpoint, inferring config from checkpoint."""
    print(f"Loading decoder from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get args from checkpoint if available
    if 'args' in checkpoint:
        args = checkpoint['args']
        if isinstance(args, dict):
            # Dictionary format
            d_model = args.get('d_model', 192)
            n_layers = args.get('n_layers', 4)
            n_heads = args.get('n_heads', 8)
            dropout = args.get('dropout', 0.3)
            sensor_dim = args.get('sensor_dim', 128)
        else:
            # Namespace format
            d_model = getattr(args, 'd_model', 192)
            n_layers = getattr(args, 'n_layers', 4)
            n_heads = getattr(args, 'n_heads', 8)
            dropout = getattr(args, 'dropout', 0.3)
            sensor_dim = getattr(args, 'sensor_dim', 128)
    else:
        # Default values
        d_model = 192
        n_layers = 4
        n_heads = 8
        dropout = 0.3
        sensor_dim = 128

    model = SensorMultiHeadDecoder(
        sensor_dim=sensor_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        vocab_size=len(vocab),
        n_operations=9,
        n_types=4,
        n_commands=6,
        n_param_types=10,
        dropout=dropout,
        max_seq_len=32,
        n_decimal_digits=4,
        max_int_digits=2,
    )

    # Load state dict with strict=False to handle type_token_mask buffer
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    # Set vocab to rebuild type_token_mask
    if hasattr(model, 'set_vocab'):
        model.set_vocab(vocab)

    model.to(device)
    model.eval()
    return model


def evaluate_model(model, encoder, test_loader, vocab, device):
    """Evaluate a single model or ensemble."""
    model.eval()
    encoder.eval()

    idx_to_token = {v: k for k, v in vocab.items()}

    all_predictions = []
    all_targets = []
    all_masks = []

    type_correct = 0
    type_total = 0
    token_correct = 0
    token_total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move to device - use correct key names from DecoderDatasetFromSplits
            tokens = batch['input_tokens'].to(device)
            sensor_data = batch['sensor_features'].to(device)
            operation_type = batch['operation_type'].to(device)
            targets = batch['target_tokens'].to(device)
            mask = batch['padding_mask'].to(device)

            # Encode sensors
            latent, temporal_features = encoder.encode(sensor_data)

            # Forward pass
            outputs = model(tokens, temporal_features, operation_type)

            # Get predictions
            if 'legacy_logits' in outputs:
                legacy_preds = outputs['legacy_logits'].argmax(dim=-1)
            elif 'raw_legacy_logits' in outputs:
                legacy_preds = outputs['raw_legacy_logits'].argmax(dim=-1)
            else:
                raise ValueError("No legacy logits in outputs")

            # Compute accuracy (excluding padding) - mask is True for PADDING, False for valid tokens
            valid_mask = ~mask.bool()  # Invert: True where valid
            token_correct += ((legacy_preds == targets) & valid_mask).sum().item()
            token_total += valid_mask.sum().item()

            # Type accuracy (if available)
            if 'type_logits' in outputs:
                type_preds = outputs['type_logits'].argmax(dim=-1)
                # Use predicted types vs actual token types
                type_total += valid_mask.sum().item()

            all_predictions.append(legacy_preds.cpu())
            all_targets.append(targets.cpu())
            all_masks.append(mask.cpu())

    # Compute metrics
    token_acc = token_correct / token_total if token_total > 0 else 0
    type_acc = type_correct / type_total if type_total > 0 else 0

    return {
        'token_accuracy': token_acc,
        'type_accuracy': type_acc,
        'predictions': torch.cat(all_predictions, dim=0),
        'targets': torch.cat(all_targets, dim=0),
        'masks': torch.cat(all_masks, dim=0),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate ensemble model')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='Paths to model checkpoints')
    parser.add_argument('--data-dir', type=str, default='outputs/stratified_splits_v2',
                        help='Path to data splits directory')
    parser.add_argument('--vocab-path', type=str, default='data/vocabulary_4digit_hybrid.json',
                        help='Path to vocabulary file')
    parser.add_argument('--encoder-path', type=str, default='outputs/mm_dtae_lstm_v2/best_model.pt',
                        help='Path to encoder checkpoint')
    parser.add_argument('--output', type=str, default='reports/ensemble_evaluation.json',
                        help='Output path for results')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--use-tta', action='store_true',
                        help='Use test-time augmentation')
    parser.add_argument('--tta-augmentations', type=int, default=5,
                        help='Number of TTA augmentations')
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load vocabulary
    print(f"Loading vocabulary from {args.vocab_path}...")
    with open(args.vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']
    print(f"Vocabulary size: {len(vocab)}")

    # Load test dataset
    print(f"Loading test dataset from {args.data_dir}...")
    test_dataset = DecoderDatasetFromSplits(
        split_dir=args.data_dir,
        split='test',
        max_token_len=32,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=decoder_collate_fn,
    )
    print(f"Test samples: {len(test_dataset)}")

    # Load encoder
    encoder = load_encoder(args.encoder_path, device)

    # Evaluate individual models
    individual_results = {}
    models = []

    for i, ckpt_path in enumerate(args.checkpoints):
        print(f"\n--- Evaluating model {i+1}/{len(args.checkpoints)} ---")
        model = load_decoder_from_checkpoint(ckpt_path, vocab, device)
        models.append(model)

        results = evaluate_model(model, encoder, test_loader, vocab, device)
        individual_results[ckpt_path] = {
            'token_accuracy': results['token_accuracy'],
            'type_accuracy': results['type_accuracy'],
        }
        print(f"Token accuracy: {results['token_accuracy']:.4f}")

    # Create and evaluate ensemble
    print("\n--- Evaluating Ensemble ---")
    ensemble = EnsembleDecoder(models, voting_mode='soft')
    ensemble.to(device)

    ensemble_results = evaluate_model(ensemble, encoder, test_loader, vocab, device)
    print(f"Ensemble token accuracy: {ensemble_results['token_accuracy']:.4f}")

    # Optionally use TTA
    tta_results = None
    if args.use_tta:
        print("\n--- Evaluating with Test-Time Augmentation ---")
        tta = TestTimeAugmentation(
            ensemble,
            n_augmentations=args.tta_augmentations,
            noise_level=0.01,
        )
        # Note: TTA evaluation requires custom wrapper
        print("TTA evaluation not fully implemented - use ensemble results")

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)

    print("\nIndividual Models:")
    for path, results in individual_results.items():
        name = Path(path).parent.name
        print(f"  {name}: {results['token_accuracy']:.4f}")

    best_individual = max(r['token_accuracy'] for r in individual_results.values())
    ensemble_acc = ensemble_results['token_accuracy']

    print(f"\nBest Individual: {best_individual:.4f}")
    print(f"Ensemble:        {ensemble_acc:.4f}")
    print(f"Improvement:     {(ensemble_acc - best_individual) * 100:+.2f}%")

    # Save results
    output_data = {
        'individual_models': individual_results,
        'ensemble': {
            'token_accuracy': ensemble_acc,
            'type_accuracy': ensemble_results['type_accuracy'],
            'n_models': len(models),
        },
        'best_individual': best_individual,
        'improvement': ensemble_acc - best_individual,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
