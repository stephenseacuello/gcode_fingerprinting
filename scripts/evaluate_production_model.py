#!/usr/bin/env python3
"""
Evaluate production model comprehensively.
"""

import torch
import numpy as np
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.multihead_lm import MultiHeadGCodeLM
from miracle.dataset.dataset import GCodeDataset, collate_fn
from miracle.dataset.target_utils import TokenDecomposer
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_production_model(checkpoint_path: str, test_data_path: str,
                              vocab_path: str, device: str = 'cpu'):
    """Comprehensive production model evaluation"""

    print("="*80)
    print("PRODUCTION MODEL EVALUATION")
    print("="*80)

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    print(f"Config: {config}")

    # Create decomposer
    decomposer = TokenDecomposer(vocab_path)

    # Infer dimensions from checkpoint
    if 'backbone_state_dict' in checkpoint:
        encoder0_weight = checkpoint['backbone_state_dict']['encoders.0.proj.0.weight']
        encoder1_weight = checkpoint['backbone_state_dict']['encoders.1.proj.0.weight']
        n_continuous = encoder0_weight.shape[1]
        n_categorical = encoder1_weight.shape[1]
    else:
        n_continuous, n_categorical = 155, 4

    # Infer vocab size from checkpoint (CRITICAL: must do before creating backbone)
    embed_weight = checkpoint['multihead_state_dict']['embed.weight']
    vocab_size = embed_weight.shape[0]

    # Infer n_param_values from checkpoint (to handle 100 vs 1000 mismatch)
    param_value_head_weight = checkpoint['multihead_state_dict']['param_value_head.4.weight']
    n_param_values = param_value_head_weight.shape[0]
    print(f"  Inferred n_param_values from checkpoint: {n_param_values}")

    # Create backbone
    backbone_config = ModelConfig(
        sensor_dims=[n_continuous, n_categorical],
        d_model=config.get('hidden_dim', 128),
        lstm_layers=config.get('num_layers', 2),
        gcode_vocab=vocab_size,  # Fixed: use actual vocab size from checkpoint
        n_heads=config.get('num_heads', 4),
    )
    backbone = MM_DTAE_LSTM(backbone_config).to(device)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])

    # Create multihead LM
    multihead_lm = MultiHeadGCodeLM(
        d_model=config.get('hidden_dim', 128),
        n_commands=decomposer.n_commands,
        n_param_types=decomposer.n_param_types,
        n_param_values=n_param_values,  # Use inferred from checkpoint, not decomposer
        nhead=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 2),
        vocab_size=vocab_size,
    ).to(device)
    multihead_lm.load_state_dict(checkpoint['multihead_state_dict'])

    backbone.eval()
    multihead_lm.eval()

    print(f"\n✓ Model loaded")
    print(f"  Sensor dims: continuous={n_continuous}, categorical={n_categorical}")
    print(f"  Hidden dim: {config.get('hidden_dim', 128)}")
    print(f"  Vocab size: {vocab_size}")

    # Load test dataset
    print(f"\nLoading test data: {test_data_path}")
    test_dataset = GCodeDataset(test_data_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    print(f"✓ Test dataset loaded: {len(test_dataset)} sequences")

    # Evaluation
    print(f"\n{'='*80}")
    print("RUNNING EVALUATION")
    print("="*80)

    all_type_correct = 0
    all_command_correct = 0
    all_param_type_correct = 0
    all_param_value_correct = 0
    all_overall_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            continuous = batch['continuous'].to(device)
            categorical = batch['categorical'].to(device).float()
            tokens = batch['tokens'].to(device)

            # Clip tokens to valid range
            tokens = torch.clamp(tokens, min=0, max=vocab_size-1)

            # Compute lengths
            batch_size = continuous.size(0)
            seq_len = continuous.size(1)
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

            # Prepare targets (matching training validation logic)
            tgt_in = tokens[:, :-1]  # Input: [B, T-1] - remove last token
            tgt_out = tokens[:, 1:]   # Target: [B, T-1] - remove first token

            # Forward backbone (pass gcode_in to match training)
            mods = [continuous, categorical]
            backbone_out = backbone(mods=mods, lengths=lengths, gcode_in=tgt_in)
            memory = backbone_out['memory']

            # Decompose targets
            tgt_decomposed = decomposer.decompose_batch(tgt_out)

            # Forward multihead LM
            logits = multihead_lm(memory, tgt_in)

            # Get predictions (no slicing needed - logits already [B, T-1])
            type_preds = torch.argmax(logits['type_logits'], dim=-1)
            command_preds = torch.argmax(logits['command_logits'], dim=-1)
            param_type_preds = torch.argmax(logits['param_type_logits'], dim=-1)
            param_value_preds = torch.argmax(logits['param_value_logits'], dim=-1)

            # Create valid mask (exclude PAD tokens where type=0)
            valid_mask = tgt_decomposed['type'] != 0

            # Compute accuracies (only on valid tokens)
            type_correct = (type_preds == tgt_decomposed['type'])[valid_mask].sum().item()
            command_correct = (command_preds == tgt_decomposed['command_id'])[valid_mask].sum().item()
            param_type_correct = (param_type_preds == tgt_decomposed['param_type_id'])[valid_mask].sum().item()
            param_value_correct = (param_value_preds == tgt_decomposed['param_value_id'])[valid_mask].sum().item()

            # Overall: all heads must be correct
            overall_correct = (
                (type_preds == tgt_decomposed['type']) &
                (command_preds == tgt_decomposed['command_id']) &
                (param_type_preds == tgt_decomposed['param_type_id']) &
                (param_value_preds == tgt_decomposed['param_value_id'])
            )[valid_mask].sum().item()

            n_tokens = valid_mask.sum().item()

            all_type_correct += type_correct
            all_command_correct += command_correct
            all_param_type_correct += param_type_correct
            all_param_value_correct += param_value_correct
            all_overall_correct += overall_correct
            total_tokens += n_tokens

    # Compute final accuracies
    type_acc = all_type_correct / total_tokens * 100
    command_acc = all_command_correct / total_tokens * 100
    param_type_acc = all_param_type_correct / total_tokens * 100
    param_value_acc = all_param_value_correct / total_tokens * 100
    overall_acc = all_overall_correct / total_tokens * 100

    print(f"\n{'='*80}")
    print("RESULTS")
    print("="*80)
    print(f"\nTest Set Performance:")
    print(f"  Type Accuracy:        {type_acc:.2f}%")
    print(f"  Command Accuracy:     {command_acc:.2f}%")
    print(f"  Param Type Accuracy:  {param_type_acc:.2f}%")
    print(f"  Param Value Accuracy: {param_value_acc:.2f}%")
    print(f"  Overall Accuracy:     {overall_acc:.2f}%")
    print(f"\nTotal tokens evaluated: {total_tokens:,}")

    # Save results
    results = {
        'checkpoint': str(checkpoint_path),
        'test_data': str(test_data_path),
        'config': config,
        'accuracies': {
            'type': type_acc,
            'command': command_acc,
            'param_type': param_type_acc,
            'param_value': param_value_acc,
            'overall': overall_acc,
        },
        'total_tokens': total_tokens,
    }

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test-data', required=True)
    parser.add_argument('--vocab', default='data/vocabulary.json')
    parser.add_argument('--output', default=None)

    args = parser.parse_args()

    # Set device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")

    # Run evaluation
    results = evaluate_production_model(
        args.checkpoint,
        args.test_data,
        args.vocab,
        device=device
    )

    # Save results if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {output_path}")
