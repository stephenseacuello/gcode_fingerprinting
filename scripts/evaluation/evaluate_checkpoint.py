#!/usr/bin/env python3
"""
Evaluate a checkpoint on the test set and generate comprehensive metrics.

Usage:
    python scripts/evaluate_checkpoint.py \
        --checkpoint outputs/best_from_sweep/checkpoint_best.pt \
        --test-data outputs/processed_quick/test_sequences.npz \
        --vocab-path data/vocabulary.json \
        --output outputs/evaluation_results
"""

import argparse
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.multihead_lm import MultiHeadGCodeLM
from miracle.dataset.target_utils import TokenDecomposer
from miracle.utilities.device import get_device


def load_checkpoint(checkpoint_path: Path, decomposer: TokenDecomposer, device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config from checkpoint
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
    else:
        # Default config
        config_dict = {
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
        }

    # Infer dimensions from state dict to ensure compatibility
    backbone_state = checkpoint['backbone_state_dict']
    multihead_state = checkpoint['multihead_state_dict']

    # Infer sensor dimensions
    if 'encoders.0.proj.0.weight' in backbone_state:
        n_continuous = backbone_state['encoders.0.proj.0.weight'].shape[1]
    else:
        n_continuous = 135  # fallback

    # Infer n_categorical (need to figure this out from the model structure)
    n_categorical = 4  # Default - models typically use 4 categorical features

    # Infer vocab_size from checkpoint
    if 'embed.weight' in multihead_state:
        checkpoint_vocab_size = multihead_state['embed.weight'].shape[0]
    else:
        checkpoint_vocab_size = 170  # fallback

    # Warn if vocab sizes don't match
    if checkpoint_vocab_size != decomposer.vocab_size:
        print(f"\n⚠️  WARNING: Vocabulary size mismatch!")
        print(f"   Checkpoint was trained with vocab_size={checkpoint_vocab_size}")
        print(f"   Current vocabulary has {decomposer.vocab_size} tokens")
        print(f"   This checkpoint is incompatible with the current vocabulary.")
        print(f"\n   To use this checkpoint, you need to:")
        print(f"   1. Use a vocabulary file that matches the checkpoint (170 tokens)")
        print(f"   OR")
        print(f"   2. Train a new model with the current vocabulary\n")
        raise ValueError(f"Vocabulary mismatch: checkpoint has {checkpoint_vocab_size} tokens, current vocab has {decomposer.vocab_size} tokens")

    vocab_size = checkpoint_vocab_size

    # Infer digit-by-digit head usage and operation count from checkpoint
    use_digit_value_head = any(k.startswith('digit_value_head') for k in multihead_state.keys())
    # Default operations; prefer checkpoint shape if available
    n_operation_types = 6
    if 'operation_head.4.weight' in multihead_state:
        n_operation_types = multihead_state['operation_head.4.weight'].shape[0]
    # Digit head settings (fallback to defaults if not found)
    max_int_digits = config_dict.get('max_int_digits', 2)
    n_decimal_digits = config_dict.get('n_decimal_digits', 4)
    if use_digit_value_head:
        # Try to infer number of digit positions from checkpoint
        digit_positions = []
        for k in multihead_state.keys():
            if k.startswith('digit_value_head.digit_heads'):
                try:
                    digit_positions.append(int(k.split('.')[2]))
                except Exception:
                    continue
        if digit_positions:
            n_positions = max(digit_positions) + 1
            # Assume first two are integer digits if not specified in config
            if 'max_int_digits' not in config_dict:
                max_int_digits = 2
            if 'n_decimal_digits' not in config_dict:
                n_decimal_digits = max(0, n_positions - max_int_digits)

    # Create backbone config (for MM-DTAE-LSTM)
    backbone_config = ModelConfig(
        sensor_dims=[n_continuous, n_categorical],
        d_model=config_dict.get('hidden_dim', 128),
        lstm_layers=config_dict.get('num_layers', 2),
        gcode_vocab=vocab_size,
        n_heads=config_dict.get('num_heads', 4),
    )

    # Create backbone
    backbone = MM_DTAE_LSTM(backbone_config).to(device)

    # Create multi-head LM (digit-aware + correct op count)
    multihead_lm = MultiHeadGCodeLM(
        d_model=config_dict.get('hidden_dim', 128),
        n_commands=decomposer.n_commands,
        n_param_types=decomposer.n_param_types,
        n_param_values=decomposer.n_param_values,
        nhead=config_dict.get('num_heads', 4),
        num_layers=config_dict.get('num_layers', 2),
        dropout=config_dict.get('dropout', 0.1),
        vocab_size=vocab_size,
        n_operation_types=n_operation_types,
        use_digit_value_head=use_digit_value_head,
        max_int_digits=max_int_digits,
        n_decimal_digits=n_decimal_digits,
    ).to(device)

    # Load weights
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    multihead_lm.load_state_dict(checkpoint['multihead_state_dict'])

    backbone.eval()
    multihead_lm.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    val_acc = checkpoint.get('val_acc', 'unknown')

    print(f"✓ Loaded checkpoint from epoch {epoch}")
    print(f"  Validation accuracy: {val_acc}")
    print(f"  Model: {config_dict.get('hidden_dim')}d, {config_dict.get('num_layers')} layers, {config_dict.get('num_heads')} heads")
    print()

    return backbone, multihead_lm, config_dict


def load_test_data(test_data_path: Path):
    """Load test sequences."""
    print(f"Loading test data: {test_data_path}")

    data = np.load(test_data_path, allow_pickle=True)

    # Check data format and adapt
    if 'sequences' in data:
        # Old format
        test_sequences = data['sequences']
        test_sensor = data['sensor_data']
    elif 'tokens' in data:
        # New format
        test_sequences = data['tokens']
        # Reconstruct sensor_data dict
        test_sensor = [
            {
                'continuous': data['continuous'][i],
                'categorical': data['categorical'][i]
            }
            for i in range(len(test_sequences))
        ]
    else:
        raise ValueError(f"Unknown data format. Keys: {list(data.keys())}")

    print(f"✓ Loaded {len(test_sequences)} test samples")
    print()

    return test_sequences, test_sensor


def evaluate(backbone, multihead_lm, decomposer, test_sequences, test_sensor, device, vocab_size):
    """
    Evaluate model on test set.

    Returns metrics for each prediction head.
    """
    print("Running evaluation...")

    total_samples = len(test_sequences)
    command_correct = 0
    param_type_correct = 0
    param_value_correct = 0
    overall_correct = 0

    with torch.no_grad():
        for idx, (seq, sensor) in enumerate(zip(test_sequences, test_sensor)):
            if (idx + 1) % 100 == 0:
                print(f"  Evaluated {idx + 1}/{total_samples} samples...")

            # Prepare input
            sensor_tensor = {
                'continuous': torch.FloatTensor(sensor['continuous']).unsqueeze(0).to(device),
                'categorical': torch.FloatTensor(sensor['categorical']).unsqueeze(0).to(device),
            }

            # Decompose target sequence using decompose_batch
            seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)  # [1, T]
            decomposed = decomposer.decompose_batch(seq_tensor)

            # Get embeddings using backbone forward pass
            mods = [sensor_tensor['continuous'], sensor_tensor['categorical']]
            lengths = torch.tensor([seq_tensor.shape[1]]).to(device)
            backbone_out = backbone(mods=mods, lengths=lengths, gcode_in=seq_tensor[:, :-1])
            embeddings = backbone_out['memory']  # [B, T, D]

            # Prepare targets (decompose_batch returns tensors, already on device)
            seq_len = seq_tensor.shape[1]
            type_gate = decomposed['type']  # [1, T]
            commands = decomposed['command_id']  # [1, T]
            param_types = decomposed['param_type_id']  # [1, T]
            param_values = decomposed['param_value_id']  # [1, T]

            # Forward pass (new API: just memory and target tokens)
            outputs = multihead_lm(
                memory=embeddings,
                tgt_tokens=seq_tensor[:, :-1]
            )

            # Get predictions
            cmd_pred = outputs['command_logits'].argmax(dim=-1)[0]
            param_type_pred = outputs['param_type_logits'].argmax(dim=-1)[0]
            # Note: param_value is now continuous regression, not classification
            param_value_pred = outputs['param_value_regression'].squeeze(-1)[0]

            # Compare with targets (shift by 1 for teacher forcing)
            cmd_target = commands[0, 1:]
            param_type_target = param_types[0, 1:]
            param_value_target = param_values[0, 1:]

            # Accuracy per head
            command_correct += (cmd_pred == cmd_target).float().mean().item()
            param_type_correct += (param_type_pred == param_type_target).float().mean().item()

            # For param values: check if rounded regression matches bucket ID
            param_value_pred_rounded = param_value_pred.round().long().clamp(0, 9)
            param_value_correct += (param_value_pred_rounded == param_value_target).float().mean().item()

            # Overall token accuracy
            overall_match = (cmd_pred == cmd_target) & \
                          (param_type_pred == param_type_target) & \
                          (param_value_pred_rounded == param_value_target)
            overall_correct += overall_match.float().mean().item()

    # Calculate averages
    metrics = {
        'command_acc': command_correct / total_samples,
        'param_type_acc': param_type_correct / total_samples,
        'param_value_acc': param_value_correct / total_samples,
        'overall_acc': overall_correct / total_samples,
        'num_samples': total_samples,
    }

    print()
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Test Samples: {metrics['num_samples']}")
    print(f"Command Accuracy: {metrics['command_acc']:.4f} ({metrics['command_acc']*100:.2f}%)")
    print(f"Parameter Type Accuracy: {metrics['param_type_acc']:.4f} ({metrics['param_type_acc']*100:.2f}%)")
    print(f"Parameter Value Accuracy: {metrics['param_value_acc']:.4f} ({metrics['param_value_acc']*100:.2f}%)")
    print(f"Overall Token Accuracy: {metrics['overall_acc']:.4f} ({metrics['overall_acc']*100:.2f}%)")
    print("=" * 80)
    print()

    return metrics


def save_results(metrics: Dict, output_dir: Path, checkpoint_path: Path):
    """Save evaluation results to JSON."""
    results = {
        'checkpoint': str(checkpoint_path),
        'metrics': metrics,
    }

    output_file = output_dir / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved evaluation results to: {output_file}")

    # Also save to CSV for easy comparison
    csv_file = output_dir / 'evaluation_results.csv'
    with open(csv_file, 'w') as f:
        f.write("checkpoint,command_acc,param_type_acc,param_value_acc,overall_acc,num_samples\n")
        f.write(f"{checkpoint_path},{metrics['command_acc']:.4f},{metrics['param_type_acc']:.4f},"
                f"{metrics['param_value_acc']:.4f},{metrics['overall_acc']:.4f},{metrics['num_samples']}\n")

    print(f"✓ Saved evaluation results to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoint on test set')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test sequences (.npz)')
    parser.add_argument('--vocab-path', type=str, required=True, help='Path to vocabulary file')
    parser.add_argument('--output', type=str, default='outputs/evaluation_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Setup device
        device = get_device()
        print(f"Using device: {device}")
        print()

        # Load vocabulary and create decomposer
        decomposer = TokenDecomposer(args.vocab_path)

        # Load checkpoint
        backbone, multihead_lm, config = load_checkpoint(
            Path(args.checkpoint), decomposer, device
        )

        # Load test data
        test_sequences, test_sensor = load_test_data(Path(args.test_data))

        # Evaluate
        metrics = evaluate(
            backbone, multihead_lm, decomposer,
            test_sequences, test_sensor, device, decomposer.vocab_size
        )

        # Save results
        save_results(metrics, output_dir, Path(args.checkpoint))

        print("✅ Evaluation complete!")
        return 0

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
