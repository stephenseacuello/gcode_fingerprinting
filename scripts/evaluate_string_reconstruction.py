#!/usr/bin/env python3
"""
Evaluate model with G-code string reconstruction and grammar validation.

This script evaluates:
1. Standard multi-head prediction accuracy
2. G-code grammar validity (RS-274D compliance)
3. String reconstruction accuracy
4. Generates demo outputs

Usage:
    python scripts/evaluate_string_reconstruction.py \
        --checkpoint outputs/sweep_zay9vjt6_best/checkpoint_best.pt \
        --test-data outputs/processed_hybrid/test_sequences.npz \
        --vocab-path data/vocabulary_1digit_hybrid.json \
        --output outputs/string_reconstruction_eval \
        --n-samples 100 \
        --generate-demos
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.multihead_lm import MultiHeadGCodeLM
from miracle.dataset.target_utils import TokenDecomposer
from miracle.utilities.device import get_device
from miracle.inference.string_reconstructor import (
    GCodeStringReconstructor,
    GCodeValidator,
    compute_string_metrics,
)
from miracle.training.modal_groups import (
    get_modal_group,
    check_modal_conflict,
    COMMAND_PARAM_RULES,
)


def load_vocab(vocab_path: Path) -> Dict[str, int]:
    """Load vocabulary from JSON file."""
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    # Handle nested vocab structure
    if 'vocab' in vocab_data:
        return vocab_data['vocab']
    return vocab_data


def load_checkpoint(checkpoint_path: Path, decomposer: TokenDecomposer, device, test_data_path: Path = None):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load config
    config_dict = checkpoint.get('config', {
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 5,
        'dropout': 0.2,
    })

    # Infer dimensions from state dict
    backbone_state = checkpoint['backbone_state_dict']
    multihead_state = checkpoint['multihead_state_dict']

    # Get n_continuous from checkpoint weights (most reliable)
    if 'encoders.0.proj.0.weight' in backbone_state:
        n_continuous = backbone_state['encoders.0.proj.0.weight'].shape[1]
    else:
        # Fallback: try to get from test data
        if test_data_path and test_data_path.exists():
            test_data = np.load(test_data_path, allow_pickle=True)
            n_continuous = test_data['continuous'].shape[-1]
        else:
            n_continuous = 219  # Default for hybrid data

    n_categorical = 4

    if 'embed.weight' in multihead_state:
        vocab_size = multihead_state['embed.weight'].shape[0]
    else:
        vocab_size = decomposer.vocab_size

    # Create backbone
    backbone_config = ModelConfig(
        sensor_dims=[n_continuous, n_categorical],
        d_model=config_dict.get('hidden_dim', 256),
        lstm_layers=config_dict.get('num_layers', 5),
        gcode_vocab=vocab_size,
        n_heads=config_dict.get('num_heads', 8),
    )

    backbone = MM_DTAE_LSTM(backbone_config).to(device)

    # Create multi-head LM
    multihead_lm = MultiHeadGCodeLM(
        d_model=config_dict.get('hidden_dim', 256),
        n_commands=decomposer.n_commands,
        n_param_types=decomposer.n_param_types,
        n_param_values=decomposer.n_param_values,
        nhead=config_dict.get('num_heads', 8),
        num_layers=config_dict.get('num_layers', 5),
        dropout=config_dict.get('dropout', 0.2),
        vocab_size=vocab_size,
    ).to(device)

    # Load weights
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    multihead_lm.load_state_dict(checkpoint['multihead_state_dict'])

    backbone.eval()
    multihead_lm.eval()

    print(f"  Model: {config_dict.get('hidden_dim')}d, {config_dict.get('num_layers')} layers")
    print(f"  Vocab size: {vocab_size}")

    return backbone, multihead_lm, config_dict


def load_test_data(test_data_path: Path, n_samples: int = None):
    """Load test sequences."""
    print(f"Loading test data: {test_data_path}")

    data = np.load(test_data_path, allow_pickle=True)

    if 'tokens' in data:
        tokens = data['tokens']
        continuous = data['continuous']
        categorical = data['categorical']
        gcode_texts = data.get('gcode_texts', None)
    else:
        raise ValueError(f"Unknown data format. Keys: {list(data.keys())}")

    if n_samples:
        tokens = tokens[:n_samples]
        continuous = continuous[:n_samples]
        categorical = categorical[:n_samples]
        if gcode_texts is not None:
            gcode_texts = gcode_texts[:n_samples]

    print(f"  Loaded {len(tokens)} samples")

    return tokens, continuous, categorical, gcode_texts


def evaluate_predictions(
    backbone,
    multihead_lm,
    decomposer,
    tokens,
    continuous,
    categorical,
    device,
) -> Dict[str, Any]:
    """
    Evaluate model predictions on test data.

    Returns detailed metrics per head.
    """
    print("\nEvaluating predictions...")

    metrics = defaultdict(list)
    all_predictions = []

    with torch.no_grad():
        for idx in range(len(tokens)):
            if (idx + 1) % 50 == 0:
                print(f"  Processing {idx + 1}/{len(tokens)}...")

            # Prepare input
            seq = torch.LongTensor(tokens[idx]).unsqueeze(0).to(device)
            cont = torch.FloatTensor(continuous[idx]).unsqueeze(0).to(device)
            cat = torch.FloatTensor(categorical[idx]).unsqueeze(0).to(device)

            # Decompose target
            decomposed = decomposer.decompose_batch(seq)

            # Forward pass
            mods = [cont, cat]
            lengths = torch.tensor([seq.shape[1]]).to(device)
            backbone_out = backbone(mods=mods, lengths=lengths, gcode_in=seq[:, :-1])
            embeddings = backbone_out['memory']

            outputs = multihead_lm(memory=embeddings, tgt_tokens=seq[:, :-1])

            # Get predictions
            type_pred = outputs['type_logits'].argmax(dim=-1)[0]
            cmd_pred = outputs['command_logits'].argmax(dim=-1)[0]
            param_type_pred = outputs['param_type_logits'].argmax(dim=-1)[0]
            param_value_pred = outputs['param_value_regression'].squeeze(-1)[0]

            if 'operation_logits' in outputs:
                op_pred = outputs['operation_logits'].argmax(dim=-1)[0]
            else:
                op_pred = None

            # Targets
            type_target = decomposed['type'][0, 1:]
            cmd_target = decomposed['command_id'][0, 1:]
            param_type_target = decomposed['param_type_id'][0, 1:]
            param_value_target = decomposed['param_value_id'][0, 1:].float()  # Bucket ID as float

            if 'operation_id' in decomposed:
                op_target = decomposed['operation_id'][0, 1:]
            else:
                op_target = None

            # Calculate accuracies
            metrics['type_acc'].append((type_pred == type_target).float().mean().item())
            metrics['command_acc'].append((cmd_pred == cmd_target).float().mean().item())
            metrics['param_type_acc'].append((param_type_pred == param_type_target).float().mean().item())

            # Param value MAE
            mae = (param_value_pred - param_value_target).abs().mean().item()
            metrics['param_value_mae'].append(mae)

            # Tolerance accuracy (within 0.01)
            within_tol = ((param_value_pred - param_value_target).abs() < 0.01).float().mean().item()
            metrics['param_value_tolerance_acc'].append(within_tol)

            # Operation accuracy
            if op_pred is not None and op_target is not None:
                metrics['operation_acc'].append((op_pred == op_target).float().mean().item())

            # Store predictions for string reconstruction
            all_predictions.append({
                'type': type_pred.cpu().numpy(),
                'command': cmd_pred.cpu().numpy(),
                'param_type': param_type_pred.cpu().numpy(),
                'param_value': param_value_pred.cpu().numpy(),
                'operation': op_pred.cpu().numpy() if op_pred is not None else None,
            })

    # Aggregate metrics
    results = {}
    for key, values in metrics.items():
        results[key] = np.mean(values)
        results[f'{key}_std'] = np.std(values)

    return results, all_predictions


def evaluate_grammar(
    predictions: List[Dict],
    vocab: Dict[str, int],
    inv_vocab: Dict[int, str],
) -> Dict[str, Any]:
    """
    Evaluate grammar validity of predictions using RS-274D rules.
    """
    print("\nEvaluating grammar validity...")

    validator = GCodeValidator(vocab)
    reconstructor = GCodeStringReconstructor(vocab)

    grammar_metrics = {
        'total_sequences': len(predictions),
        'valid_sequences': 0,
        'modal_conflicts': 0,
        'arc_violations': 0,
        'rapid_feed_violations': 0,
        'single_letter_violations': 0,
        'total_violations': 0,
    }

    reconstructed_gcodes = []

    for pred in predictions:
        reconstructor.reset_modal_state()

        # Build prediction list for reconstruction
        pred_list = []
        current_param = None

        for t in range(len(pred['type'])):
            p = {
                'type': pred['type'][t],
                'command': pred['command'][t],
                'param_type': pred['param_type'][t],
                'param_value': pred['param_value'][t],
            }
            pred_list.append(p)

            # Track current param for value association
            if p['type'] == 2:  # PARAM type
                current_param = inv_vocab.get(p['param_type'], '')

        # Reconstruct G-code string
        gcode_line = reconstructor.reconstruct_line(pred_list)
        reconstructed_gcodes.append(gcode_line)

        # Validate
        is_valid, errors = validator.validate_line(gcode_line)

        if is_valid:
            grammar_metrics['valid_sequences'] += 1
        else:
            grammar_metrics['total_violations'] += len(errors)

            for error in errors:
                if 'modal' in error.lower():
                    grammar_metrics['modal_conflicts'] += 1
                elif 'arc' in error.lower() or 'G2' in error or 'G3' in error:
                    grammar_metrics['arc_violations'] += 1
                elif 'G0' in error and 'feed' in error.lower():
                    grammar_metrics['rapid_feed_violations'] += 1
                elif 'duplicate' in error.lower():
                    grammar_metrics['single_letter_violations'] += 1

    # Calculate rates
    n = grammar_metrics['total_sequences']
    grammar_metrics['validity_rate'] = grammar_metrics['valid_sequences'] / n if n > 0 else 0
    grammar_metrics['avg_violations_per_seq'] = grammar_metrics['total_violations'] / n if n > 0 else 0

    return grammar_metrics, reconstructed_gcodes


def generate_demo_outputs(
    predictions: List[Dict],
    reconstructed_gcodes: List[str],
    actual_gcodes: List[str],
    vocab: Dict[str, int],
    decomposer,
    output_dir: Path,
    n_demos: int = 10,
):
    """Generate demonstration outputs showing predicted vs actual G-code."""
    print(f"\nGenerating {n_demos} demo outputs...")

    # Use decomposer's mappings for command and param lookups
    # command_id -> command token
    id2cmd = {v: k for k, v in decomposer.command2id.items()}
    # param_id -> param token
    id2param = {v: k for k, v in decomposer.param2id.items()}

    demos = []

    for i in range(min(n_demos, len(predictions))):
        pred = predictions[i]
        reconstructed = reconstructed_gcodes[i]
        actual = actual_gcodes[i] if actual_gcodes is not None else "N/A"

        # Build token sequence for display using decomposer mappings
        tokens = []
        current_param = None

        for t in range(min(20, len(pred['type']))):
            token_type = int(pred['type'][t])
            cmd_id = int(pred['command'][t])
            param_id = int(pred['param_type'][t])
            value = pred['param_value'][t]

            # Get actual token names using decomposer mappings
            cmd_token = id2cmd.get(cmd_id, f'CMD{cmd_id}')
            param_token = id2param.get(param_id, f'P{param_id}')

            if token_type == 1:  # COMMAND
                tokens.append(cmd_token)
            elif token_type == 2:  # PARAM
                current_param = param_token
                tokens.append(param_token)
            elif token_type == 3:  # VALUE
                if current_param:
                    tokens.append(f"{current_param}{value:.3f}")
                    current_param = None
                else:
                    tokens.append(f"V{value:.3f}")
            # Skip type 0 (PAD)

        # Build simple reconstruction from predictions
        simple_recon_parts = []
        seen_types = set()

        for t in range(min(15, len(pred['type']))):
            token_type = int(pred['type'][t])
            cmd_id = int(pred['command'][t])
            param_id = int(pred['param_type'][t])
            value = pred['param_value'][t]

            if token_type == 0:  # Skip PAD
                continue

            if token_type == 1:  # COMMAND
                cmd = id2cmd.get(cmd_id, '')
                if cmd and cmd not in seen_types:
                    simple_recon_parts.append(cmd)
                    seen_types.add(cmd)
            elif token_type == 2:  # PARAM
                param = id2param.get(param_id, '')
                if param:
                    simple_recon_parts.append(param)
            elif token_type == 3:  # VALUE
                # Attach value to previous param if exists
                if simple_recon_parts and simple_recon_parts[-1] in id2param.values():
                    simple_recon_parts[-1] = f"{simple_recon_parts[-1]}{value:.3f}"
                else:
                    simple_recon_parts.append(f"{value:.3f}")

        simple_recon = ' '.join(simple_recon_parts[:10])

        demo = {
            'sample_id': i,
            'predicted_tokens': ' '.join(tokens) if tokens else '[empty]',
            'simple_reconstruction': simple_recon if simple_recon else '[empty]',
            'reconstructed_gcode': reconstructed if reconstructed else '[empty]',
            'actual_gcode': str(actual) if actual else "N/A",
            'match': reconstructed.strip() == str(actual).strip() if actual and actual != "N/A" else None,
        }
        demos.append(demo)

    # Save demos
    demo_file = output_dir / 'demo_outputs.json'
    with open(demo_file, 'w') as f:
        json.dump(demos, f, indent=2)

    # Also create a readable text file
    txt_file = output_dir / 'demo_outputs.txt'
    with open(txt_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("G-CODE STRING RECONSTRUCTION DEMO OUTPUTS\n")
        f.write("=" * 80 + "\n\n")

        for demo in demos:
            f.write(f"Sample {demo['sample_id']}:\n")
            f.write(f"  Predicted Tokens:     {demo['predicted_tokens']}\n")
            f.write(f"  Simple Recon:         {demo['simple_reconstruction']}\n")
            f.write(f"  Full Reconstructed:   {demo['reconstructed_gcode']}\n")
            f.write(f"  Actual G-code:        {demo['actual_gcode']}\n")
            if demo['match'] is not None:
                f.write(f"  Match:                {'✓ YES' if demo['match'] else '✗ NO'}\n")
            f.write("\n")

    print(f"  Saved demos to {demo_file}")
    print(f"  Saved readable demos to {txt_file}")

    return demos


def print_results(
    prediction_metrics: Dict,
    grammar_metrics: Dict,
    output_dir: Path,
):
    """Print and save comprehensive results."""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)

    print("\n--- PREDICTION ACCURACY ---")
    print(f"  Type Accuracy:           {prediction_metrics['type_acc']:.4f} ({prediction_metrics['type_acc']*100:.2f}%)")
    print(f"  Command Accuracy:        {prediction_metrics['command_acc']:.4f} ({prediction_metrics['command_acc']*100:.2f}%)")
    print(f"  Param Type Accuracy:     {prediction_metrics['param_type_acc']:.4f} ({prediction_metrics['param_type_acc']*100:.2f}%)")
    print(f"  Param Value MAE:         {prediction_metrics['param_value_mae']:.6f}")
    print(f"  Param Value Tolerance:   {prediction_metrics['param_value_tolerance_acc']:.4f} ({prediction_metrics['param_value_tolerance_acc']*100:.2f}%)")

    if 'operation_acc' in prediction_metrics:
        print(f"  Operation Accuracy:      {prediction_metrics['operation_acc']:.4f} ({prediction_metrics['operation_acc']*100:.2f}%)")

    print("\n--- GRAMMAR VALIDITY (RS-274D) ---")
    print(f"  Total Sequences:         {grammar_metrics['total_sequences']}")
    print(f"  Valid Sequences:         {grammar_metrics['valid_sequences']}")
    print(f"  Validity Rate:           {grammar_metrics['validity_rate']:.4f} ({grammar_metrics['validity_rate']*100:.2f}%)")
    print(f"  Avg Violations/Seq:      {grammar_metrics['avg_violations_per_seq']:.2f}")

    print("\n  Violation Breakdown:")
    print(f"    Modal Conflicts:       {grammar_metrics['modal_conflicts']}")
    print(f"    Arc Violations:        {grammar_metrics['arc_violations']}")
    print(f"    Rapid+Feed Violations: {grammar_metrics['rapid_feed_violations']}")
    print(f"    Single Letter Errors:  {grammar_metrics['single_letter_violations']}")

    # Quality assessment
    print("\n--- QUALITY ASSESSMENT ---")

    validity = grammar_metrics['validity_rate']
    if validity >= 0.95:
        print("  Grammar:    ✅ EXCELLENT (95%+ valid)")
    elif validity >= 0.85:
        print("  Grammar:    ⚠️  GOOD (85-95% valid)")
    elif validity >= 0.70:
        print("  Grammar:    ⚠️  FAIR (70-85% valid)")
    else:
        print("  Grammar:    ❌ NEEDS IMPROVEMENT (<70% valid)")

    cmd_acc = prediction_metrics['command_acc']
    if cmd_acc >= 0.98:
        print("  Commands:   ✅ EXCELLENT (98%+ accurate)")
    elif cmd_acc >= 0.90:
        print("  Commands:   ⚠️  GOOD (90-98% accurate)")
    else:
        print("  Commands:   ❌ NEEDS IMPROVEMENT (<90% accurate)")

    print("=" * 80)

    # Save results
    results = {
        'prediction_metrics': prediction_metrics,
        'grammar_metrics': grammar_metrics,
    }

    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate with string reconstruction')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to checkpoint')
    parser.add_argument('--test-data', type=Path, required=True, help='Path to test data')
    parser.add_argument('--vocab-path', type=Path, required=True, help='Path to vocabulary')
    parser.add_argument('--output', type=Path, default=Path('outputs/string_reconstruction_eval'),
                       help='Output directory')
    parser.add_argument('--n-samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--generate-demos', action='store_true', help='Generate demo outputs')
    parser.add_argument('--n-demos', type=int, default=10, help='Number of demos to generate')

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        # Setup
        device = get_device()
        print(f"Using device: {device}\n")

        # Load vocab
        vocab = load_vocab(args.vocab_path)
        inv_vocab = {v: k for k, v in vocab.items()}

        # Load decomposer
        decomposer = TokenDecomposer(args.vocab_path)

        # Load model
        backbone, multihead_lm, config = load_checkpoint(
            args.checkpoint, decomposer, device, args.test_data
        )

        # Load test data
        tokens, continuous, categorical, gcode_texts = load_test_data(
            args.test_data, args.n_samples
        )

        # Evaluate predictions
        prediction_metrics, all_predictions = evaluate_predictions(
            backbone, multihead_lm, decomposer,
            tokens, continuous, categorical, device
        )

        # Evaluate grammar
        grammar_metrics, reconstructed_gcodes = evaluate_grammar(
            all_predictions, vocab, inv_vocab
        )

        # Generate demos if requested
        if args.generate_demos:
            generate_demo_outputs(
                all_predictions,
                reconstructed_gcodes,
                gcode_texts,
                vocab,
                decomposer,
                args.output,
                args.n_demos
            )

        # Print and save results
        print_results(prediction_metrics, grammar_metrics, args.output)

        print("\n✅ Evaluation complete!")
        return 0

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
