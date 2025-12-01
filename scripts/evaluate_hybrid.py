#!/usr/bin/env python3
"""
Evaluate hybrid multihead G-code model on test set.
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.model.multihead_lm import MultiHeadGCodeLM
from miracle.model.encoder import GCodeEncoder
from miracle.dataset.target_utils import TokenDecomposer
from miracle.dataset.dataloader import GCodeDataset


def evaluate_model(checkpoint_path, data_dir, vocab_path, split='test'):
    """Evaluate model on specified split."""

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    vocab = vocab_data.get('vocab', vocab_data.get('token2id', {}))
    vocab_size = len(vocab)

    # Initialize token decomposer
    decomposer = TokenDecomposer(vocab_path)

    # Get model config from checkpoint
    model_config = checkpoint.get('config', {})

    # Initialize encoder
    print("Initializing encoder...")
    encoder = GCodeEncoder(
        continuous_dim=model_config.get('continuous_dim', 7),
        categorical_dims=model_config.get('categorical_dims', [4]),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_layers=model_config.get('num_layers', 5),
        num_heads=model_config.get('num_heads', 8),
        dropout=model_config.get('dropout', 0.1),
    )

    # Initialize multihead model
    print("Initializing multihead model...")
    model = MultiHeadGCodeLM(
        vocab_size=vocab_size,
        n_commands=decomposer.n_commands,
        n_param_types=decomposer.n_param_types,
        n_param_values=decomposer.n_param_values,
        n_operation_types=model_config.get('n_operation_types', 5),
        d_model=model_config.get('hidden_dim', 256),
        nhead=model_config.get('num_heads', 8),
        num_decoder_layers=model_config.get('num_decoder_layers', 4),
        dim_feedforward=model_config.get('dim_feedforward', 1024),
        dropout=model_config.get('dropout', 0.1),
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load data
    print(f"Loading {split} data from {data_dir}...")
    dataset = GCodeDataset(data_dir, split=split)
    print(f"Loaded {len(dataset)} samples")

    # Move to device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    encoder.to(device)
    model.to(device)

    # Evaluation metrics
    total_samples = 0
    correct_type = 0
    correct_command = 0
    correct_param_type = 0
    correct_param_value = 0
    correct_operation = 0

    print(f"\nEvaluating on {split} set...")

    with torch.no_grad():
        for i in range(len(dataset)):
            # Get sample
            sample = dataset[i]

            # Prepare batch (add batch dimension)
            continuous = torch.from_numpy(sample['continuous']).unsqueeze(0).float().to(device)
            categorical = torch.from_numpy(sample['categorical']).unsqueeze(0).long().to(device)
            tokens = torch.from_numpy(sample['tokens']).unsqueeze(0).long().to(device)
            operation_type = torch.tensor([sample['operation_type']]).long().to(device)

            # Encoder
            memory = encoder(continuous, categorical)

            # Prepare decoder input (shift right with BOS)
            bos_id = vocab.get('<BOS>', 1)
            tgt_in = torch.cat([
                torch.full((1, 1), bos_id, dtype=torch.long, device=device),
                tokens[:, :-1]
            ], dim=1)

            tgt_out = tokens  # Ground truth

            # Forward through multihead model
            logits = model(memory, tgt_in)

            # Decompose targets (hybrid mode detection)
            if 'param_value_coarse_logits' in logits:
                tgt_decomposed = decomposer.decompose_batch_hybrid(tgt_out)
            else:
                tgt_decomposed = decomposer.decompose_batch(tgt_out)

            # Predictions
            type_pred = torch.argmax(logits['type_logits'], dim=-1)
            command_pred = torch.argmax(logits['command_logits'], dim=-1)
            param_type_pred = torch.argmax(logits['param_type_logits'], dim=-1)
            operation_pred = torch.argmax(logits['operation_logits'], dim=-1)

            # Parameter value prediction (hybrid vs standard mode)
            if 'param_value_coarse_logits' in logits:
                param_value_pred = torch.argmax(logits['param_value_coarse_logits'], dim=-1)
                param_value_target_key = 'param_value_coarse_id'
            else:
                param_value_pred = torch.argmax(logits['param_value_logits'], dim=-1)
                param_value_target_key = 'param_value_id'

            # Create mask for valid positions (ignore padding)
            pad_id = vocab.get('<PAD>', 0)
            valid_mask = (tgt_out != pad_id)

            # Count correct predictions
            seq_len = valid_mask.sum().item()
            total_samples += seq_len

            correct_type += ((type_pred == tgt_decomposed['type']) & valid_mask).sum().item()
            correct_command += ((command_pred == tgt_decomposed['command_id']) & valid_mask).sum().item()
            correct_param_type += ((param_type_pred == tgt_decomposed['param_type_id']) & valid_mask).sum().item()
            correct_param_value += ((param_value_pred == tgt_decomposed[param_value_target_key]) & valid_mask).sum().item()
            correct_operation += (operation_pred == operation_type).sum().item()

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(dataset)} samples...")

    # Compute accuracies
    type_acc = 100.0 * correct_type / total_samples if total_samples > 0 else 0
    command_acc = 100.0 * correct_command / total_samples if total_samples > 0 else 0
    param_type_acc = 100.0 * correct_param_type / total_samples if total_samples > 0 else 0
    param_value_acc = 100.0 * correct_param_value / total_samples if total_samples > 0 else 0
    operation_acc = 100.0 * correct_operation / len(dataset) if len(dataset) > 0 else 0

    print(f"\n{'='*60}")
    print(f"Evaluation Results on {split.upper()} set:")
    print(f"{'='*60}")
    print(f"Total tokens evaluated: {total_samples}")
    print(f"Type Accuracy:          {type_acc:.2f}%")
    print(f"Command Accuracy:       {command_acc:.2f}%")
    print(f"Param Type Accuracy:    {param_type_acc:.2f}%")
    print(f"Param Value Accuracy:   {param_value_acc:.2f}%")
    print(f"Operation Accuracy:     {operation_acc:.2f}%")
    print(f"{'='*60}")

    return {
        'type_acc': type_acc,
        'command_acc': command_acc,
        'param_type_acc': param_type_acc,
        'param_value_acc': param_value_acc,
        'operation_acc': operation_acc,
        'total_tokens': total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate hybrid multihead G-code model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with preprocessed data')
    parser.add_argument('--vocab-path', type=str, required=True, help='Path to vocabulary JSON')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Data split to evaluate')

    args = parser.parse_args()

    results = evaluate_model(args.checkpoint, args.data_dir, args.vocab_path, args.split)

    return results


if __name__ == '__main__':
    main()
