#!/usr/bin/env python3
"""
Test inference on local checkpoint file.

Usage:
    python scripts/test_local_checkpoint.py --checkpoint outputs/final_model/checkpoint_best.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.dataset.dataset import GCodeDataset, collate_fn
from miracle.dataset.target_utils import TokenDecomposer
from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.multihead_lm import MultiHeadGCodeLM


def load_model_from_checkpoint(checkpoint_path: str, vocab_path: str, device='cpu'):
    """
    Load model from local checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        vocab_path: Path to vocabulary file
        device: Device to load model on

    Returns:
        backbone, multihead_lm, config, decomposer
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val accuracy: {checkpoint.get('val_acc', 0):.2%}")
    print(f"  Config: {config}")

    # Create decomposer
    decomposer = TokenDecomposer(vocab_path)

    # Get model dimensions
    n_continuous = 8  # From preprocessing
    n_categorical = 18  # From preprocessing
    vocab_size = 170  # From vocab v2

    # Create backbone model
    backbone_config = ModelConfig(
        sensor_dims=[n_continuous, n_categorical],
        d_model=config.get('hidden_dim', 128),
        lstm_layers=config.get('num_layers', 2),
        gcode_vocab=vocab_size,
        n_heads=config.get('num_heads', 4),
    )
    backbone = MM_DTAE_LSTM(backbone_config).to(device)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    backbone.eval()

    # Create multi-head LM
    multihead_lm = MultiHeadGCodeLM(
        d_model=config.get('hidden_dim', 128),
        n_commands=decomposer.n_commands,
        n_param_types=decomposer.n_param_types,
        n_param_values=decomposer.n_param_values,
        nhead=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 2),
        vocab_size=vocab_size,
    ).to(device)
    multihead_lm.load_state_dict(checkpoint['multihead_state_dict'])
    multihead_lm.eval()

    print(f"✓ Model loaded successfully")

    return backbone, multihead_lm, config, decomposer


def evaluate_model(backbone, multihead_lm, decomposer, test_loader, device='cpu'):
    """
    Evaluate model on test set.

    Args:
        backbone: MM_DTAE_LSTM model
        multihead_lm: MultiHeadGCodeLM model
        decomposer: TokenDecomposer
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Dictionary of metrics
    """
    backbone.eval()
    multihead_lm.eval()

    all_preds = {
        'type': [],
        'command': [],
        'param_type': [],
        'param_value': [],
    }

    all_targets = {
        'type': [],
        'command': [],
        'param_type': [],
        'param_value': [],
    }

    print("\nRunning inference on test set...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            continuous = batch['continuous'].to(device)
            categorical = batch['categorical'].to(device)
            tokens = batch['tokens'].to(device)

            # Forward through backbone to get memory
            memory = backbone(continuous, categorical)

            # Prepare decoder input (shift tokens right, add BOS)
            bos_token = 1  # Assuming 1 is BOS
            tgt_in = torch.cat([
                torch.full((tokens.size(0), 1), bos_token, dtype=torch.long, device=device),
                tokens[:, :-1]
            ], dim=1)

            # Target output (for accuracy computation)
            tgt_out = tokens[:, 1:]

            # Decompose targets
            tgt_decomposed = decomposer.decompose_batch(tgt_out)

            # Forward through multi-head LM
            logits = multihead_lm(memory, tgt_in)

            # Argmax predictions
            type_preds = torch.argmax(logits['type_logits'], dim=-1)
            command_preds = torch.argmax(logits['command_logits'], dim=-1)
            param_type_preds = torch.argmax(logits['param_type_logits'], dim=-1)
            param_value_preds = torch.argmax(logits['param_value_logits'], dim=-1)

            # Store
            all_preds['type'].append(type_preds.cpu())
            all_preds['command'].append(command_preds.cpu())
            all_preds['param_type'].append(param_type_preds.cpu())
            all_preds['param_value'].append(param_value_preds.cpu())

            all_targets['type'].append(tgt_decomposed['type'].cpu())
            all_targets['command'].append(tgt_decomposed['command_id'].cpu())
            all_targets['param_type'].append(tgt_decomposed['param_type_id'].cpu())
            all_targets['param_value'].append(tgt_decomposed['param_value_id'].cpu())

    # Concatenate all batches
    for key in all_preds:
        all_preds[key] = torch.cat(all_preds[key]).flatten()
        all_targets[key] = torch.cat(all_targets[key]).flatten()

    # Compute accuracies
    metrics = {}
    for key in all_preds:
        correct = (all_preds[key] == all_targets[key]).sum().item()
        total = len(all_preds[key])
        accuracy = 100.0 * correct / total
        metrics[f'{key}_accuracy'] = accuracy

    # Overall accuracy (all heads correct)
    all_correct = (
        (all_preds['type'] == all_targets['type']) &
        (all_preds['command'] == all_targets['command']) &
        (all_preds['param_type'] == all_targets['param_type']) &
        (all_preds['param_value'] == all_targets['param_value'])
    ).sum().item()

    metrics['overall_accuracy'] = 100.0 * all_correct / len(all_preds['type'])

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Test local checkpoint on test set")

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--vocab-file',
        type=str,
        default='gcode_vocab_v2.json',
        help='Vocabulary filename'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run on'
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    vocab_path = str(data_dir / args.vocab_file)

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = GCodeDataset(data_dir / 'test_sequences.npz')

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"✓ Loaded {len(test_dataset)} test samples")

    # Load model
    backbone, multihead_lm, config, decomposer = load_model_from_checkpoint(
        args.checkpoint, vocab_path, args.device
    )

    # Evaluate
    metrics = evaluate_model(
        backbone, multihead_lm, decomposer, test_loader, args.device
    )

    # Print results
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy:       {metrics['overall_accuracy']:.2f}%")
    print(f"Type Accuracy:          {metrics['type_accuracy']:.2f}%")
    print(f"Command Accuracy:       {metrics['command_accuracy']:.2f}%")
    print(f"Param Type Accuracy:    {metrics['param_type_accuracy']:.2f}%")
    print(f"Param Value Accuracy:   {metrics['param_value_accuracy']:.2f}%")
    print("=" * 60)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Config: {config}")


if __name__ == "__main__":
    main()
