#!/usr/bin/env python3
"""
Generate REAL Confusion Matrices from Model Inference.

This script:
1. Loads the best 2K vocab model (frame_r2 single sensor, 93.50% accuracy)
2. Runs inference on the test set using teacher forcing
3. Generates actual confusion matrices from model predictions
4. Saves publication-quality figures

Author: Claude Code
Date: January 2026
"""

import os
import sys
import json
import platform
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict, Counter

# Set up paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'src'))

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import confusion_matrix

# Imports from the project
from miracle.model.model import EnhancedEncoder
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn

# Windows compatibility
NUM_WORKERS = 0

# Paths
DATA_DIR = BASE_DIR / 'outputs' / 'stratified_splits_2k_vocab'
VOCAB_PATH = BASE_DIR / 'data' / 'vocabulary_4digit_full.json'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'real_confusion_matrices'

# Use baseline model which has both encoder and decoder saved
# (sensor ablation checkpoints don't include encoder_state_dict)
MODEL_PATH = BASE_DIR / 'outputs' / 'full_pipeline_2k_vocab' / 'baselines' / 'no_augment_seed42' / 'best_model.pt'


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_vocab():
    """Load vocabulary."""
    with open(VOCAB_PATH) as f:
        return json.load(f)


def get_sensor_mask(continuous_columns, sensor_id='frame_r2'):
    """Create mask to keep only the specified sensor's features."""
    mask = np.zeros(len(continuous_columns), dtype=np.float32)
    for i, col in enumerate(continuous_columns):
        if sensor_id in col:
            mask[i] = 1.0
    return mask


def load_models(device):
    """Load encoder and decoder models from checkpoint."""
    print("Loading models...")

    # Load metadata
    with open(DATA_DIR / 'metadata.json') as f:
        metadata = json.load(f)
    input_dim = metadata['n_continuous_features']  # 296

    # Load vocab
    vocab_data = load_vocab()
    vocab_size = len(vocab_data['vocab'])

    # Load checkpoint with both encoder and decoder
    print(f"Loading from: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

    # Get args from checkpoint to match architecture
    args = ckpt.get('args', {})
    d_model = args.get('d_model', 192)
    n_heads = args.get('n_heads', 8)
    n_layers = args.get('n_layers', 4)
    d_ff = int(d_model * args.get('d_ff_multiplier', 4.0))
    max_seq_len = args.get('max_seq_len', 32)

    print(f"  Architecture: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")

    # Create encoder
    encoder = EnhancedEncoder(
        input_dim=input_dim,
        hidden_dim=args.get('encoder_hidden_dim', 256),
        latent_dim=args.get('sensor_dim', 128),
        n_operations=args.get('n_operations', 9),
        use_multiscale=args.get('use_multiscale_encoder', True),
        n_scales=args.get('encoder_n_scales', 4),
        pooling_n_heads=args.get('pooling_n_heads', 4),
        pooling_n_queries=args.get('pooling_n_queries', 8),
    )

    # Load encoder state
    encoder.load_state_dict(ckpt['encoder_state_dict'], strict=False)
    print("  Encoder loaded successfully")

    # Create decoder
    decoder = SensorMultiHeadDecoder(
        sensor_dim=args.get('sensor_dim', 128),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        vocab_size=vocab_size,
        n_operations=args.get('n_operations', 9),
        n_types=args.get('n_types', 4),
        n_commands=args.get('n_commands', 6),
        n_param_types=args.get('n_param_types', 10),
        d_ff=d_ff,
        dropout=args.get('dropout', 0.3),
        max_seq_len=max_seq_len,
        n_decimal_digits=args.get('n_decimal_digits', 4),
        max_int_digits=args.get('max_int_digits', 2),
        embed_dropout=args.get('embed_dropout', 0.1),
        drop_path_rate=args.get('drop_path_rate', 0.0),
        use_sensor_prior=args.get('use_sensor_prior', False),
        sensor_prior_weight=args.get('sensor_prior_weight', 0.5),
    )

    # Load decoder state
    decoder.load_state_dict(ckpt['model_state_dict'], strict=False)
    print("  Decoder loaded successfully")

    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    print(f"  Input dim: {input_dim}")
    print(f"  Vocab size: {vocab_size}")

    return encoder, decoder, metadata


def run_inference(encoder, decoder, test_loader, device, sensor_mask):
    """Run inference and collect predictions."""
    print("\nRunning inference on test set...")

    all_preds = []
    all_targets = []
    sensor_mask_tensor = torch.from_numpy(sensor_mask).float().to(device)

    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            # Apply sensor mask
            sensor_data = sensor_data * sensor_mask_tensor

            # Encode
            encoder_out = encoder(sensor_data)
            sensor_memory = encoder_out['memory']

            # Decode with teacher forcing
            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=sensor_memory,
                operation_type=operations,
            )

            # Get legacy token predictions
            legacy_logits = outputs.get('legacy_logits')
            if legacy_logits is not None:
                preds = legacy_logits.argmax(dim=-1)
                mask = target_tokens != 0  # Non-padding

                for i in range(preds.size(0)):
                    seq_mask = mask[i]
                    pred_seq = preds[i][seq_mask].cpu().numpy()
                    target_seq = target_tokens[i][seq_mask].cpu().numpy()

                    all_preds.extend(pred_seq)
                    all_targets.extend(target_seq)

                    total_tokens += len(target_seq)
                    correct_tokens += (pred_seq == target_seq).sum()

    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    print(f"\nOverall token accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total tokens: {total_tokens}, Correct: {correct_tokens}")

    return np.array(all_preds), np.array(all_targets)


def categorize_tokens(preds, targets, vocab):
    """Categorize tokens into types for analysis."""
    inv_vocab = {v: k for k, v in vocab.items()}

    def get_token_type(token_id):
        """Determine token category."""
        token = inv_vocab.get(token_id, '')
        if token in ['<pad>', '<sos>', '<eos>', '<unk>', 'EOS', 'SOS', 'BOS']:
            return 'special'
        elif token.startswith(('G', 'M')) and len(token) <= 4:
            return 'command'
        # Parameter tokens: single letter (X, Y, Z) or letter+value (X0., Y0.)
        elif token and token[0] in 'XYZIJKFRABCS' and not token.startswith('NUM_'):
            return 'param'
        else:
            return 'numeric'

    pred_types = [get_token_type(p) for p in preds]
    target_types = [get_token_type(t) for t in targets]

    return pred_types, target_types, inv_vocab


def plot_type_confusion(pred_types, target_types, output_path):
    """Plot token type confusion matrix."""
    print("\nGenerating token type confusion matrix...")

    type_labels = ['command', 'param', 'numeric', 'special']
    cm = confusion_matrix(target_types, pred_types, labels=type_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    np.nan_to_num(cm_norm, copy=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    display_labels = ['Command', 'Parameter', 'Numeric', 'Special']
    ax.set_xticks(np.arange(len(type_labels)))
    ax.set_yticks(np.arange(len(type_labels)))
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_yticklabels(display_labels, fontsize=11)

    ax.set_xlabel('Predicted Type', fontsize=12)
    ax.set_ylabel('True Type', fontsize=12)
    ax.set_title('Token Type Confusion Matrix', fontsize=14)

    plt.colorbar(im, ax=ax, label='Normalized Frequency')

    # Add annotations
    for i in range(len(type_labels)):
        for j in range(len(type_labels)):
            text = ax.text(j, i, f'{cm_norm[i, j]:.2f}\n({cm[i, j]})',
                         ha='center', va='center', fontsize=9,
                         color='white' if cm_norm[i, j] > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    # Print per-type accuracy
    print("\n  Per-type accuracy:")
    for i, t in enumerate(type_labels):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum()
            print(f"    {display_labels[i]}: {acc:.2%} ({cm[i, i]}/{cm[i].sum()})")


def plot_command_confusion(preds, targets, inv_vocab, output_path):
    """Plot G/M command confusion matrix."""
    print("\nGenerating command confusion matrix...")

    # Filter to command tokens only
    cmd_preds = []
    cmd_targets = []

    for p, t in zip(preds, targets):
        t_token = inv_vocab.get(t, '')
        if t_token.startswith(('G', 'M')) and len(t_token) <= 4:
            cmd_targets.append(t_token)
            p_token = inv_vocab.get(p, '')
            # Map non-command predictions to 'Other'
            if p_token.startswith(('G', 'M')) and len(p_token) <= 4:
                cmd_preds.append(p_token)
            else:
                cmd_preds.append('Other')

    if not cmd_targets:
        print("  No command tokens found!")
        return

    # Include 'Other' in labels if it appears in predictions
    unique_cmds = sorted(set(cmd_targets))
    if 'Other' in set(cmd_preds):
        unique_cmds_ext = unique_cmds + ['Other']
    else:
        unique_cmds_ext = unique_cmds

    cm = confusion_matrix(cmd_targets, cmd_preds, labels=unique_cmds_ext)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    np.nan_to_num(cm_norm, copy=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(unique_cmds_ext)))
    ax.set_yticks(np.arange(len(unique_cmds)))  # Y-axis only shows true commands
    ax.set_xticklabels(unique_cmds_ext, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(unique_cmds, fontsize=9)

    ax.set_xlabel('Predicted Command', fontsize=12)
    ax.set_ylabel('True Command', fontsize=12)
    ax.set_title('G-code Command Confusion Matrix', fontsize=14)

    plt.colorbar(im, ax=ax, label='Normalized Frequency')

    # Add annotations with both percentage and count
    for i in range(len(unique_cmds)):
        for j in range(len(unique_cmds_ext)):
            if cm[i, j] > 0:  # Show any non-zero cell
                ax.text(j, i, f'{cm_norm[i, j]:.2f}\n({cm[i, j]})',
                       ha='center', va='center', fontsize=9,
                       color='white' if cm_norm[i, j] > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_parameter_confusion(preds, targets, inv_vocab, output_path):
    """Plot parameter type confusion matrix."""
    print("\nGenerating parameter confusion matrix...")

    param_letters = 'XYZIJKFRABCS'
    param_preds = []
    param_targets = []

    for p, t in zip(preds, targets):
        t_token = inv_vocab.get(t, '')
        # Include both single-char (X, Y, Z) and multi-char (X0., Y0.) parameter tokens
        # Exclude NUM_ tokens which are numeric values
        if t_token and t_token[0] in param_letters and not t_token.startswith('NUM_'):
            param_targets.append(t_token[0])
            p_token = inv_vocab.get(p, '')
            if p_token and p_token[0] in param_letters and not p_token.startswith('NUM_'):
                param_preds.append(p_token[0])
            else:
                param_preds.append('?')

    if not param_targets:
        print("  No parameter tokens found!")
        return

    unique_params = sorted(set(param_targets))
    if '?' in set(param_preds):
        unique_params_ext = unique_params + ['?']
    else:
        unique_params_ext = unique_params

    cm = confusion_matrix(param_targets, param_preds, labels=unique_params)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    np.nan_to_num(cm_norm, copy=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(unique_params)))
    ax.set_yticks(np.arange(len(unique_params)))
    ax.set_xticklabels(unique_params, fontsize=11)
    ax.set_yticklabels(unique_params, fontsize=11)

    ax.set_xlabel('Predicted Parameter', fontsize=12)
    ax.set_ylabel('True Parameter', fontsize=12)
    ax.set_title('Parameter Type Confusion Matrix', fontsize=14)

    plt.colorbar(im, ax=ax, label='Normalized Frequency')

    # Add annotations
    for i in range(len(unique_params)):
        for j in range(len(unique_params)):
            if cm[i].sum() > 0:
                ax.text(j, i, f'{cm_norm[i, j]:.2f}\n({cm[i, j]})',
                       ha='center', va='center', fontsize=8,
                       color='white' if cm_norm[i, j] > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_numeric_confusion(preds, targets, inv_vocab, output_path):
    """Plot numeric value accuracy per parameter type."""
    print("\nGenerating numeric value accuracy chart...")

    def get_param_type(token):
        """Extract parameter type from NUM_X_1234 format."""
        if token and token.startswith('NUM_'):
            parts = token.split('_')
            return parts[1] if len(parts) >= 2 else None
        return None

    # Track exact match accuracy per parameter type
    param_correct = defaultdict(int)
    param_total = defaultdict(int)

    for p, t in zip(preds, targets):
        t_token = inv_vocab.get(t, '')
        p_token = inv_vocab.get(p, '')

        # Check if target is a NUM_ token
        param_type = get_param_type(t_token)
        if param_type:
            param_total[param_type] += 1
            if p == t:  # Exact token match
                param_correct[param_type] += 1

    if not param_total:
        print("  No numeric tokens found!")
        return

    # Calculate per-parameter accuracy
    param_types = sorted(param_total.keys())
    accuracies = []
    counts = []
    for pt in param_types:
        acc = param_correct[pt] / param_total[pt] * 100 if param_total[pt] > 0 else 0
        accuracies.append(acc)
        counts.append(param_total[pt])

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c', '#9b59b6', '#3498db', '#2ecc71', '#f39c12']
    x = np.arange(len(param_types))
    bars = ax.bar(x, accuracies, color=colors[:len(param_types)], edgecolor='black', linewidth=1)

    ax.set_xlabel('Parameter Type', fontsize=12)
    ax.set_ylabel('Exact Value Accuracy (%)', fontsize=12)
    ax.set_title('Numeric Value Prediction Accuracy by Parameter Type', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'NUM_{pt}' for pt in param_types], fontsize=11)
    ax.set_ylim(0, 105)

    # Add value labels
    for bar, acc, count in zip(bars, accuracies, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=10)

    # Add overall accuracy line
    total_correct = sum(param_correct.values())
    total_count = sum(param_total.values())
    overall_acc = total_correct / total_count * 100 if total_count > 0 else 0
    ax.axhline(y=overall_acc, color='red', linestyle='--', linewidth=2,
               label=f'Overall: {overall_acc:.1f}%')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    print(f"  Overall numeric accuracy: {overall_acc:.2f}% ({total_correct}/{total_count})")


def plot_overall_summary(pred_types, target_types, output_path):
    """Plot overall accuracy bar chart."""
    print("\nGenerating overall summary chart...")

    type_labels = ['command', 'param', 'numeric', 'special']
    display_labels = ['Command', 'Parameter', 'Numeric', 'Special']

    # Calculate per-type accuracy
    type_correct = defaultdict(int)
    type_total = defaultdict(int)

    for p, t in zip(pred_types, target_types):
        type_total[t] += 1
        if p == t:
            type_correct[t] += 1

    accuracies = []
    counts = []
    for t in type_labels:
        if type_total[t] > 0:
            acc = type_correct[t] / type_total[t] * 100
        else:
            acc = 0
        accuracies.append(acc)
        counts.append(type_total[t])

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#95a5a6']
    x = np.arange(len(type_labels))
    bars = ax.bar(x, accuracies, color=colors, edgecolor='black', linewidth=1)

    ax.set_xlabel('Token Category', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Category Token Accuracy\n(All Sensors, Baseline Model)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_ylim(0, 105)

    # Add value labels
    for bar, acc, count in zip(bars, accuracies, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=10)

    # Add horizontal line for overall accuracy
    overall_acc = sum(type_correct.values()) / sum(type_total.values()) * 100
    ax.axhline(y=overall_acc, color='red', linestyle='--', linewidth=2,
               label=f'Overall: {overall_acc:.1f}%')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 60)
    print("Generating REAL Confusion Matrices from Model Inference")
    print("=" * 60)

    device = get_device()
    print(f"Using device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    encoder, decoder, metadata = load_models(device)

    # Load vocabulary
    vocab_data = load_vocab()
    vocab = vocab_data['vocab']

    # Use all sensors (no masking) since baseline model was trained on all sensors
    sensor_mask = np.ones(len(metadata['continuous_columns']), dtype=np.float32)
    print(f"Sensor mask: {int(sensor_mask.sum())}/{len(sensor_mask)} features active (all sensors)")

    # Load test dataset
    test_dataset = DecoderDatasetFromSplits(
        split_dir=str(DATA_DIR),
        split='test',
        max_token_len=32,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=decoder_collate_fn,
        num_workers=NUM_WORKERS,
    )

    print(f"Test set: {len(test_dataset)} samples")

    # Run inference
    preds, targets = run_inference(encoder, decoder, test_loader, device, sensor_mask)

    # Categorize tokens
    pred_types, target_types, inv_vocab = categorize_tokens(preds, targets, vocab)

    # Generate all confusion matrix figures
    plot_type_confusion(pred_types, target_types, OUTPUT_DIR / 'confusion_type.png')
    plot_command_confusion(preds, targets, inv_vocab, OUTPUT_DIR / 'confusion_command.png')
    plot_parameter_confusion(preds, targets, inv_vocab, OUTPUT_DIR / 'confusion_parameter.png')
    plot_numeric_confusion(preds, targets, inv_vocab, OUTPUT_DIR / 'confusion_numeric.png')
    plot_overall_summary(pred_types, target_types, OUTPUT_DIR / 'confusion_overall.png')

    print(f"\n{'='*60}")
    print("CONFUSION MATRIX GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("\nFiles generated:")
    for f in OUTPUT_DIR.glob('*.png'):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
