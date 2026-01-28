#!/usr/bin/env python3
"""
Comprehensive evaluation script that computes full metrics for paper reporting.

Computes:
- Train/val/test loss and accuracy
- Precision, Recall, F1 per token type (type, command, param, digit)
- Per-head accuracy breakdown
- Confusion matrices
- Saves detailed results for figure generation

Usage:
    python scripts/evaluation/evaluate_full_metrics.py \
        --checkpoint outputs/jan26/sensor_ablations/only_xa_motor/best_model.pt \
        --split-dir outputs/jan23_followup/no_leakage \
        --vocab-path data/vocabulary_4digit_full.json \
        --encoder-path outputs/production/encoder/best_model.pt \
        --output-dir outputs/jan26/evaluation/only_xa_motor
"""

import argparse
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Windows and macOS have issues with multiprocessing DataLoader workers
NUM_WORKERS = 0 if platform.system() in ('Windows', 'Darwin') else 4

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import EnhancedEncoder
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn


def load_models(
    checkpoint_path: str,
    encoder_path: str,
    vocab_path: str,
    device: str,
    config: Optional[Dict] = None
) -> Tuple[EnhancedEncoder, SensorMultiHeadDecoder, Dict]:
    """Load encoder and decoder models from checkpoints."""

    # Load vocabulary
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    # Handle different vocab file formats
    if 'vocab' in vocab_data:
        vocab = vocab_data['vocab']
        vocab_size = len(vocab.get('token_to_id', vocab))
    elif 'token_to_id' in vocab_data:
        vocab = vocab_data
        vocab_size = len(vocab['token_to_id'])
    else:
        vocab = vocab_data
        vocab_size = len(vocab)

    # Default config if not provided
    if config is None:
        config = {
            'd_model': 192,
            'n_heads': 8,
            'n_layers': 4,
            'dropout': 0.3,
            'embed_dropout': 0.1,
        }

    # Load checkpoint to get config
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'args' in ckpt:
        saved_args = ckpt['args']
        config.update({
            'd_model': saved_args.get('d_model', config['d_model']),
            'n_heads': saved_args.get('n_heads', config['n_heads']),
            'n_layers': saved_args.get('n_layers', config['n_layers']),
            'dropout': saved_args.get('dropout', config['dropout']),
        })

    # Infer input dimension from checkpoint
    state_dict = ckpt.get('model_state_dict', ckpt)
    if 'sensor_proj.weight' in state_dict:
        input_dim = state_dict['sensor_proj.weight'].shape[1]
    else:
        input_dim = 296  # Default full features

    # Create encoder
    encoder = EnhancedEncoder(
        input_dim=input_dim,
        hidden_dim=256,
        latent_dim=128,
        n_operations=9,
        use_multiscale=True,
        n_scales=4,
    )

    # Load encoder weights
    if os.path.exists(encoder_path):
        enc_ckpt = torch.load(encoder_path, map_location='cpu', weights_only=False)
        encoder.load_state_dict(enc_ckpt.get('model_state_dict', enc_ckpt), strict=False)
    encoder.to(device)
    encoder.eval()

    # Create decoder
    d_model = config['d_model']
    n_heads = config['n_heads']
    if d_model % n_heads != 0:
        n_heads = 4 if d_model % 4 == 0 else 1

    decoder = SensorMultiHeadDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        sensor_dim=128,
        n_operations=9,
        n_types=4,
        n_commands=6,
        n_param_types=10,
        max_seq_len=32,
    )

    # Load decoder weights
    decoder.load_state_dict(state_dict, strict=False)
    decoder.to(device)
    decoder.eval()

    return encoder, decoder, config


def compute_metrics_for_split(
    encoder: EnhancedEncoder,
    decoder: SensorMultiHeadDecoder,
    dataloader: DataLoader,
    device: str,
    compute_confusion: bool = False
) -> Dict:
    """Compute comprehensive metrics for a data split."""

    encoder.eval()
    decoder.eval()

    # Accumulators
    total_loss = 0.0
    total_samples = 0

    # Per-head predictions and targets
    all_type_preds = []
    all_type_targets = []
    all_cmd_preds = []
    all_cmd_targets = []
    all_param_preds = []
    all_param_targets = []
    all_digit_preds = [[] for _ in range(6)]  # 6 digit positions
    all_digit_targets = [[] for _ in range(6)]

    # Token-level
    total_tokens = 0
    correct_tokens = 0
    total_sequences = 0
    correct_sequences = 0

    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')

    with torch.no_grad():
        for batch in dataloader:
            sensors = batch['sensors'].to(device)
            operations = batch['operations'].to(device)
            targets = batch['targets'].to(device)
            target_types = batch['target_types'].to(device)
            target_commands = batch['target_commands'].to(device)
            target_param_types = batch['target_param_types'].to(device)
            target_digits = batch['target_digits'].to(device)
            lengths = batch['lengths']

            batch_size = sensors.size(0)
            total_samples += batch_size

            # Get encoder output
            encoder_out = encoder(sensors)
            latents = encoder_out['latent']

            # Get decoder predictions
            decoder_out = decoder(
                targets[:, :-1],
                latents,
                operations
            )

            # Compute loss
            logits = decoder_out['logits']
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = targets[:, 1:].reshape(-1)
            loss = criterion(flat_logits, flat_targets)
            total_loss += loss.item()

            # Get predictions
            type_logits = decoder_out['type_logits']
            cmd_logits = decoder_out['command_logits']
            param_logits = decoder_out['param_type_logits']
            digit_logits = decoder_out['digit_logits']

            # Compute per-head accuracy
            type_preds = type_logits.argmax(dim=-1)
            cmd_preds = cmd_logits.argmax(dim=-1)
            param_preds = param_logits.argmax(dim=-1)

            # Collect for metrics (flatten and filter valid)
            for i in range(batch_size):
                seq_len = min(lengths[i] - 1, type_preds.size(1))

                # Type predictions
                all_type_preds.extend(type_preds[i, :seq_len].cpu().numpy())
                all_type_targets.extend(target_types[i, 1:seq_len+1].cpu().numpy())

                # Command predictions
                all_cmd_preds.extend(cmd_preds[i, :seq_len].cpu().numpy())
                all_cmd_targets.extend(target_commands[i, 1:seq_len+1].cpu().numpy())

                # Param predictions
                all_param_preds.extend(param_preds[i, :seq_len].cpu().numpy())
                all_param_targets.extend(target_param_types[i, 1:seq_len+1].cpu().numpy())

                # Digit predictions
                for d in range(min(6, digit_logits.size(2))):
                    digit_pred = digit_logits[i, :seq_len, d].argmax(dim=-1)
                    all_digit_preds[d].extend(digit_pred.cpu().numpy())
                    all_digit_targets[d].extend(target_digits[i, 1:seq_len+1, d].cpu().numpy())

            # Token accuracy (using composed tokens)
            token_preds = decoder_out['logits'].argmax(dim=-1)
            for i in range(batch_size):
                seq_len = min(lengths[i] - 1, token_preds.size(1))
                pred_seq = token_preds[i, :seq_len]
                target_seq = targets[i, 1:seq_len+1]

                # Count correct tokens
                mask = target_seq != -100
                correct = (pred_seq == target_seq) & mask
                correct_tokens += correct.sum().item()
                total_tokens += mask.sum().item()

                # Sequence accuracy
                total_sequences += 1
                if correct.sum() == mask.sum():
                    correct_sequences += 1

    # Compute metrics
    metrics = {
        'loss': total_loss / max(total_tokens, 1),
        'token_acc': correct_tokens / max(total_tokens, 1),
        'sequence_acc': correct_sequences / max(total_sequences, 1),
        'n_samples': total_samples,
        'n_tokens': total_tokens,
    }

    # Per-head accuracy
    type_correct = sum(p == t for p, t in zip(all_type_preds, all_type_targets) if t != -100)
    type_total = sum(1 for t in all_type_targets if t != -100)
    metrics['type_acc'] = type_correct / max(type_total, 1)

    cmd_correct = sum(p == t for p, t in zip(all_cmd_preds, all_cmd_targets) if t != -100)
    cmd_total = sum(1 for t in all_cmd_targets if t != -100)
    metrics['command_acc'] = cmd_correct / max(cmd_total, 1)

    param_correct = sum(p == t for p, t in zip(all_param_preds, all_param_targets) if t != -100)
    param_total = sum(1 for t in all_param_targets if t != -100)
    metrics['param_acc'] = param_correct / max(param_total, 1)

    # Digit accuracy per position
    digit_accs = []
    for d in range(6):
        d_correct = sum(p == t for p, t in zip(all_digit_preds[d], all_digit_targets[d]) if t != -100)
        d_total = sum(1 for t in all_digit_targets[d] if t != -100)
        acc = d_correct / max(d_total, 1)
        digit_accs.append(acc)
        metrics[f'digit_{d}_acc'] = acc
    metrics['digit_avg_acc'] = sum(digit_accs) / len(digit_accs)

    # Precision, Recall, F1 for each head
    def compute_prf(preds, targets, name):
        # Filter out ignore index
        valid_mask = [t != -100 for t in targets]
        valid_preds = [p for p, v in zip(preds, valid_mask) if v]
        valid_targets = [t for t, v in zip(targets, valid_mask) if v]

        if len(valid_targets) == 0:
            return {}

        # Get unique classes
        classes = sorted(set(valid_targets))

        p, r, f1, support = precision_recall_fscore_support(
            valid_targets, valid_preds, labels=classes, average=None, zero_division=0
        )

        # Macro averages
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            valid_targets, valid_preds, average='macro', zero_division=0
        )

        # Weighted averages
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            valid_targets, valid_preds, average='weighted', zero_division=0
        )

        result = {
            f'{name}_precision_macro': float(p_macro),
            f'{name}_recall_macro': float(r_macro),
            f'{name}_f1_macro': float(f1_macro),
            f'{name}_precision_weighted': float(p_weighted),
            f'{name}_recall_weighted': float(r_weighted),
            f'{name}_f1_weighted': float(f1_weighted),
        }

        # Per-class metrics
        for i, cls in enumerate(classes):
            result[f'{name}_class_{cls}_precision'] = float(p[i])
            result[f'{name}_class_{cls}_recall'] = float(r[i])
            result[f'{name}_class_{cls}_f1'] = float(f1[i])
            result[f'{name}_class_{cls}_support'] = int(support[i])

        return result

    metrics.update(compute_prf(all_type_preds, all_type_targets, 'type'))
    metrics.update(compute_prf(all_cmd_preds, all_cmd_targets, 'command'))
    metrics.update(compute_prf(all_param_preds, all_param_targets, 'param'))

    # Confusion matrices (if requested)
    if compute_confusion:
        def safe_confusion(preds, targets, name):
            valid_mask = [t != -100 for t in targets]
            valid_preds = [p for p, v in zip(preds, valid_mask) if v]
            valid_targets = [t for t, v in zip(targets, valid_mask) if v]
            if len(valid_targets) == 0:
                return None
            cm = confusion_matrix(valid_targets, valid_preds)
            return cm.tolist()

        metrics['confusion_matrices'] = {
            'type': safe_confusion(all_type_preds, all_type_targets, 'type'),
            'command': safe_confusion(all_cmd_preds, all_cmd_targets, 'command'),
            'param': safe_confusion(all_param_preds, all_param_targets, 'param'),
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to decoder checkpoint')
    parser.add_argument('--split-dir', type=str, required=True, help='Path to data splits')
    parser.add_argument('--vocab-path', type=str, required=True, help='Path to vocabulary JSON')
    parser.add_argument('--encoder-path', type=str, required=True, help='Path to encoder checkpoint')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--config', type=str, help='Path to config JSON (optional)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default=None, help='Device (auto-detected if not specified)')
    parser.add_argument('--include-train', action='store_true', help='Also evaluate on training set')
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f"Using device: {args.device}")

    # Load config
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading models...")
    encoder, decoder, config = load_models(
        args.checkpoint, args.encoder_path, args.vocab_path, args.device, config
    )

    # Create datasets
    print("Loading datasets...")

    # Infer columns from checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    columns = None
    if 'columns' in ckpt:
        columns = ckpt['columns']

    val_dataset = DecoderDatasetFromSplits(
        args.split_dir, 'val', args.vocab_path, columns=columns
    )
    test_dataset = DecoderDatasetFromSplits(
        args.split_dir, 'test', args.vocab_path, columns=columns
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=NUM_WORKERS, pin_memory=True
    )

    results = {
        'checkpoint': args.checkpoint,
        'split_dir': args.split_dir,
        'config': config,
    }

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = compute_metrics_for_split(
        encoder, decoder, val_loader, args.device, compute_confusion=True
    )
    results['val_metrics'] = val_metrics
    print(f"  Val Loss: {val_metrics['loss']:.4f}")
    print(f"  Val Token Acc: {val_metrics['token_acc']:.2%}")
    print(f"  Val Sequence Acc: {val_metrics['sequence_acc']:.2%}")
    print(f"  Val Type Acc: {val_metrics['type_acc']:.2%}")
    print(f"  Val Command Acc: {val_metrics['command_acc']:.2%}")
    print(f"  Val Param Acc: {val_metrics['param_acc']:.2%}")
    print(f"  Val Digit Avg Acc: {val_metrics['digit_avg_acc']:.2%}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = compute_metrics_for_split(
        encoder, decoder, test_loader, args.device, compute_confusion=True
    )
    results['test_metrics'] = test_metrics
    print(f"  Test Loss: {test_metrics['loss']:.4f}")
    print(f"  Test Token Acc: {test_metrics['token_acc']:.2%}")
    print(f"  Test Sequence Acc: {test_metrics['sequence_acc']:.2%}")
    print(f"  Test Type Acc: {test_metrics['type_acc']:.2%}")
    print(f"  Test Command Acc: {test_metrics['command_acc']:.2%}")
    print(f"  Test Param Acc: {test_metrics['param_acc']:.2%}")
    print(f"  Test Digit Avg Acc: {test_metrics['digit_avg_acc']:.2%}")

    # Optionally evaluate on training set
    if args.include_train:
        print("\nEvaluating on training set...")
        train_dataset = DecoderDatasetFromSplits(
            args.split_dir, 'train', args.vocab_path, columns=columns
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=decoder_collate_fn, num_workers=NUM_WORKERS, pin_memory=True
        )
        train_metrics = compute_metrics_for_split(
            encoder, decoder, train_loader, args.device, compute_confusion=False
        )
        results['train_metrics'] = train_metrics
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Train Token Acc: {train_metrics['token_acc']:.2%}")

    # Print summary table
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    print(f"{'Metric':<25} {'Val':>12} {'Test':>12}")
    print("-"*60)
    print(f"{'Token Accuracy':<25} {val_metrics['token_acc']:>11.2%} {test_metrics['token_acc']:>11.2%}")
    print(f"{'Sequence Accuracy':<25} {val_metrics['sequence_acc']:>11.2%} {test_metrics['sequence_acc']:>11.2%}")
    print(f"{'Type Accuracy':<25} {val_metrics['type_acc']:>11.2%} {test_metrics['type_acc']:>11.2%}")
    print(f"{'Command Accuracy':<25} {val_metrics['command_acc']:>11.2%} {test_metrics['command_acc']:>11.2%}")
    print(f"{'Param Type Accuracy':<25} {val_metrics['param_acc']:>11.2%} {test_metrics['param_acc']:>11.2%}")
    print(f"{'Digit Avg Accuracy':<25} {val_metrics['digit_avg_acc']:>11.2%} {test_metrics['digit_avg_acc']:>11.2%}")
    print("-"*60)
    print(f"{'Type F1 (macro)':<25} {val_metrics.get('type_f1_macro', 0):>11.2%} {test_metrics.get('type_f1_macro', 0):>11.2%}")
    print(f"{'Command F1 (macro)':<25} {val_metrics.get('command_f1_macro', 0):>11.2%} {test_metrics.get('command_f1_macro', 0):>11.2%}")
    print(f"{'Param F1 (macro)':<25} {val_metrics.get('param_f1_macro', 0):>11.2%} {test_metrics.get('param_f1_macro', 0):>11.2%}")
    print("="*60)

    # Save results
    results_path = output_dir / 'full_metrics.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save confusion matrices separately for easier plotting
    cm_path = output_dir / 'confusion_matrices.json'
    cm_data = {
        'val': val_metrics.get('confusion_matrices', {}),
        'test': test_metrics.get('confusion_matrices', {}),
    }
    with open(cm_path, 'w') as f:
        json.dump(cm_data, f, indent=2)
    print(f"Confusion matrices saved to: {cm_path}")


if __name__ == '__main__':
    main()
