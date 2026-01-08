#!/usr/bin/env python3
"""
Simple ensemble evaluation that matches training evaluation logic.

IMPORTANT: Uses EnhancedEncoder (not basic MM_DTAE_LSTM) to match training.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.dataset.decoder_dataset import (
    DecoderDatasetFromSplits,
    decoder_collate_fn,
)
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.model.model import EnhancedEncoder


def load_model(checkpoint_path, vocab, device):
    """Load a decoder model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt['args']

    model = SensorMultiHeadDecoder(
        sensor_dim=args.get('sensor_dim', 128),
        d_model=args.get('d_model', 320),
        n_heads=args.get('n_heads', 8),
        n_layers=args.get('n_layers', 5),
        vocab_size=len(vocab),
        n_operations=args.get('n_operations', 9),
        n_types=args.get('n_types', 4),
        n_commands=args.get('n_commands', 6),
        n_param_types=args.get('n_param_types', 10),
        dropout=args.get('dropout', 0.32),
        max_seq_len=args.get('max_seq_len', 32),
        n_decimal_digits=args.get('n_decimal_digits', 4),
        max_int_digits=args.get('max_int_digits', 2),
        embed_dropout=args.get('embed_dropout', 0.17),
        drop_path_rate=args.get('drop_path_rate', 0.11),
        use_sensor_prior=args.get('use_sensor_prior', True),
        sensor_prior_weight=args.get('sensor_prior_weight', 0.51),
    )

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if hasattr(model, 'set_vocab'):
        model.set_vocab(vocab)
    model.to(device)
    model.eval()
    return model


PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 2


def create_encoder_for_seed(seed, train_args, encoder_path, device):
    """Create and initialize encoder with specific seed."""
    # Set seed for reproducible initialization
    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = EnhancedEncoder(
        input_dim=train_args.get('sensor_input_dim', 155),
        hidden_dim=train_args.get('encoder_hidden_dim', 256),
        latent_dim=train_args.get('sensor_dim', 128),
        n_operations=train_args.get('n_operations', 9),
        use_multiscale=True,
        n_scales=train_args.get('encoder_n_scales', 4),
        kernel_sizes=train_args.get('encoder_kernel_sizes', [3, 5, 7, 11]),
        dilations=train_args.get('encoder_dilations', [1, 2, 4, 8]),
        lstm_layers=train_args.get('encoder_lstm_layers', 2),
        use_multihead_pooling=train_args.get('use_multihead_pooling', True),
        pooling_n_heads=train_args.get('pooling_n_heads', 4),
        pooling_n_queries=train_args.get('pooling_n_queries', 8),
        use_auxiliary_heads=train_args.get('use_auxiliary_heads', False),
        dropout=train_args.get('encoder_dropout', 0.3),
    )

    # Load encoder weights
    ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder_state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.to(device)
    encoder.eval()
    return encoder


def evaluate_single(model, encoder, dataloader, device):
    """Evaluate a single model."""
    model.eval()
    encoder.eval()

    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            tokens = batch['input_tokens'].to(device)
            sensor_data = batch['sensor_features'].to(device)
            targets = batch['target_tokens'].to(device)
            mask = batch['padding_mask'].to(device)

            # Encode - matches training logic for EnhancedEncoder vs MM_DTAE_LSTM
            # EnhancedEncoder.encode() returns (pooled_features[2D], memory_sequence[3D])
            # MM_DTAE_LSTM.encode() returns (latent_sequence[3D], hidden_state)
            enc_out1, enc_out2 = encoder.encode(sensor_data)
            # EnhancedEncoder: out1 = pooled (2D), out2 = memory (3D) -> use out2 for decoder
            # MM_DTAE_LSTM: out1 = sequence (3D), out2 = hidden -> use out1 for decoder
            sensor_emb = enc_out2 if enc_out1.dim() == 2 else enc_out1
            classify_input = enc_out1 if enc_out1.dim() == 2 else enc_out1
            op_logits, _ = encoder.classify(classify_input)
            operation_type = op_logits.argmax(-1)

            # Forward
            outputs = model(tokens, sensor_emb, operation_type, tgt_key_padding_mask=mask)

            # Use raw_legacy_logits (without type constraint masking)
            if 'raw_legacy_logits' in outputs:
                logits = outputs['raw_legacy_logits']
            else:
                logits = outputs['legacy_logits']

            preds = logits.argmax(dim=-1)
            B, T = targets.shape

            # Per-sample accuracy (up to first EOS)
            for b in range(B):
                eos_positions = (targets[b] == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    eos_pos = eos_positions[0].item() + 1
                else:
                    eos_pos = T

                valid_positions = torch.zeros(T, dtype=torch.bool, device=device)
                valid_positions[:eos_pos] = True
                valid_mask = valid_positions & (targets[b] != PAD_TOKEN_ID)

                total_correct += (preds[b][valid_mask] == targets[b][valid_mask]).sum().item()
                total_tokens += valid_mask.sum().item()

    return total_correct / total_tokens if total_tokens > 0 else 0


def evaluate_ensemble(models, encoder, dataloader, device):
    """Evaluate ensemble by averaging logits (single shared encoder)."""
    for m in models:
        m.eval()
    encoder.eval()

    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Ensemble eval", leave=False):
            tokens = batch['input_tokens'].to(device)
            sensor_data = batch['sensor_features'].to(device)
            targets = batch['target_tokens'].to(device)
            mask = batch['padding_mask'].to(device)

            # Encode - matches training logic for EnhancedEncoder vs MM_DTAE_LSTM
            enc_out1, enc_out2 = encoder.encode(sensor_data)
            sensor_emb = enc_out2 if enc_out1.dim() == 2 else enc_out1
            classify_input = enc_out1 if enc_out1.dim() == 2 else enc_out1
            op_logits, _ = encoder.classify(classify_input)
            operation_type = op_logits.argmax(-1)

            # Collect logits from all models
            all_logits = []
            for model in models:
                outputs = model(tokens, sensor_emb, operation_type, tgt_key_padding_mask=mask)
                if 'raw_legacy_logits' in outputs:
                    logits = outputs['raw_legacy_logits']
                else:
                    logits = outputs['legacy_logits']
                all_logits.append(logits)

            # Average logits (soft voting)
            ensemble_logits = torch.stack(all_logits).mean(dim=0)
            preds = ensemble_logits.argmax(dim=-1)
            B, T = targets.shape

            # Per-sample accuracy (up to first EOS)
            for b in range(B):
                eos_positions = (targets[b] == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    eos_pos = eos_positions[0].item() + 1
                else:
                    eos_pos = T

                valid_positions = torch.zeros(T, dtype=torch.bool, device=device)
                valid_positions[:eos_pos] = True
                valid_mask = valid_positions & (targets[b] != PAD_TOKEN_ID)

                total_correct += (preds[b][valid_mask] == targets[b][valid_mask]).sum().item()
                total_tokens += valid_mask.sum().item()

    return total_correct / total_tokens if total_tokens > 0 else 0


def evaluate_ensemble_multi_encoder(models, encoders, dataloader, device):
    """Evaluate ensemble with multiple encoders (one per model).

    Each model uses its own encoder for inference, then logits are averaged.
    This is needed when models were trained with different seeds.
    """
    for m in models:
        m.eval()
    for e in encoders:
        e.eval()

    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Ensemble eval (multi-encoder)", leave=False):
            tokens = batch['input_tokens'].to(device)
            sensor_data = batch['sensor_features'].to(device)
            targets = batch['target_tokens'].to(device)
            mask = batch['padding_mask'].to(device)

            # Collect logits from all models, each with its own encoder
            all_logits = []
            for model, encoder in zip(models, encoders):
                # Encode with this model's encoder
                enc_out1, enc_out2 = encoder.encode(sensor_data)
                sensor_emb = enc_out2 if enc_out1.dim() == 2 else enc_out1
                classify_input = enc_out1 if enc_out1.dim() == 2 else enc_out1
                op_logits, _ = encoder.classify(classify_input)
                operation_type = op_logits.argmax(-1)

                # Forward through decoder
                outputs = model(tokens, sensor_emb, operation_type, tgt_key_padding_mask=mask)
                if 'raw_legacy_logits' in outputs:
                    logits = outputs['raw_legacy_logits']
                else:
                    logits = outputs['legacy_logits']
                all_logits.append(logits)

            # Average logits (soft voting)
            ensemble_logits = torch.stack(all_logits).mean(dim=0)
            preds = ensemble_logits.argmax(dim=-1)
            B, T = targets.shape

            # Per-sample accuracy (up to first EOS)
            for b in range(B):
                eos_positions = (targets[b] == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    eos_pos = eos_positions[0].item() + 1
                else:
                    eos_pos = T

                valid_positions = torch.zeros(T, dtype=torch.bool, device=device)
                valid_positions[:eos_pos] = True
                valid_mask = valid_positions & (targets[b] != PAD_TOKEN_ID)

                total_correct += (preds[b][valid_mask] == targets[b][valid_mask]).sum().item()
                total_tokens += valid_mask.sum().item()

    return total_correct / total_tokens if total_tokens > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--split-dir', default='outputs/stratified_splits_full_vocab')
    parser.add_argument('--vocab-path', default='data/vocabulary_4digit_full.json')
    parser.add_argument('--encoder-path', default='outputs/mm_dtae_lstm_v2/best_model.pt')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output', default='reports/ensemble_results.json')
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load vocab
    with open(args.vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']
    print(f"Vocab size: {len(vocab)}")

    # Load test data
    test_dataset = DecoderDatasetFromSplits(
        split_dir=args.split_dir,
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

    # Evaluate individual models - each with its own encoder
    models = []
    encoders = []
    individual_results = {}
    has_saved_encoders = []

    for ckpt_path in args.checkpoints:
        name = Path(ckpt_path).parent.name
        print(f"\nEvaluating {name}...")

        # Load model and get its training seed
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_args = ckpt.get('args', {})
        model_seed = model_args.get('seed', 123)

        # Check if encoder state is saved in checkpoint (new format)
        if 'encoder_state_dict' in ckpt:
            print(f"  Loading encoder from checkpoint (saved state)...")
            model_encoder = create_encoder_for_seed(model_seed, model_args, args.encoder_path, device)
            # Load the exact encoder state that was used during training
            model_encoder.load_state_dict(ckpt['encoder_state_dict'], strict=False)
            has_saved_encoders.append(True)
        else:
            # Old format - recreate encoder with seed (may not match exactly)
            print(f"  Creating encoder with seed {model_seed} (no saved encoder in checkpoint)...")
            model_encoder = create_encoder_for_seed(model_seed, model_args, args.encoder_path, device)
            has_saved_encoders.append(False)

        # Load decoder
        model = load_model(ckpt_path, vocab, device)
        models.append(model)
        encoders.append(model_encoder)

        acc = evaluate_single(model, model_encoder, test_loader, device)
        individual_results[name] = acc
        print(f"  Token accuracy: {acc*100:.2f}%")

    # Check if ensemble is valid
    # Valid if: all have saved encoders OR all have same seed
    all_have_saved_encoders = all(has_saved_encoders)
    all_seeds = [torch.load(p, map_location=device, weights_only=False).get('args', {}).get('seed', 123)
                 for p in args.checkpoints]
    all_same_seed = len(set(all_seeds)) == 1

    if all_have_saved_encoders and all_same_seed:
        # Same seed with saved encoders - can use single encoder
        print("\nEvaluating ensemble (same seed, saved encoder states)...")
        ensemble_acc = evaluate_ensemble(models, encoders[0], test_loader, device)
        print(f"  Ensemble accuracy: {ensemble_acc*100:.2f}%")
    elif all_have_saved_encoders and not all_same_seed:
        # Different seeds but all have saved encoders - use multi-encoder ensemble
        print("\nEvaluating ensemble (different seeds, each model uses its saved encoder)...")
        ensemble_acc = evaluate_ensemble_multi_encoder(models, encoders, test_loader, device)
        print(f"  Ensemble accuracy: {ensemble_acc*100:.2f}%")
    elif all_same_seed:
        # All same seed without saved encoders - can still do ensemble
        print("\nEvaluating ensemble (all models share same seed)...")
        ensemble_acc = evaluate_ensemble(models, encoders[0], test_loader, device)
        print(f"  Ensemble accuracy: {ensemble_acc*100:.2f}%")
    else:
        # Different seeds and no saved encoders - ensemble not meaningful
        print("\nWARNING: Models trained with different seeds have different encoders.")
        print("Ensemble of these models is NOT valid because each decoder expects")
        print("sensor embeddings from its own encoder initialization.")
        print("\nTo create a valid ensemble, either:")
        print("  1. Train all models with the same seed (same encoder init)")
        print("  2. Use models with saved encoder checkpoints (new training script)")
        print("  3. Use a shared frozen encoder for all seeds")
        ensemble_acc = None

    # Summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    for name, acc in individual_results.items():
        print(f"  {name}: {acc*100:.2f}%")
    best_individual = max(individual_results.values())
    print(f"\nBest individual: {best_individual*100:.2f}%")
    if ensemble_acc is not None:
        print(f"Ensemble:        {ensemble_acc*100:.2f}%")
        print(f"Improvement:     {(ensemble_acc - best_individual)*100:+.2f}%")
    else:
        print("Ensemble:        N/A (different encoder seeds)")

    # Save
    results = {
        'individual': {k: v for k, v in individual_results.items()},
        'ensemble': ensemble_acc,
        'best_individual': best_individual,
        'improvement': (ensemble_acc - best_individual) if ensemble_acc else None,
        'note': 'Ensemble not valid - models trained with different encoder seeds' if ensemble_acc is None else None,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
