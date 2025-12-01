"""
G-code String Assembly Pipeline

Assembles predicted G-code strings from model outputs by:
1. Using model's param_type predictions (X, Y, Z, F, R, S)
2. Using model's param_value predictions (numeric values)
3. Inferring command from first token type OR operation type
4. Combining into complete G-code strings

Architecture Note:
- Commands (G0, G1, G2, G3) are at position 0, sliced off in teacher-forcing
- Model predicts NEXT tokens, so command is inferred from context
- Parameter predictions (X, Y, Z, numeric) work well (~93% accuracy)
"""

import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Tuple

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.multihead_lm import MultiHeadGCodeLM
from miracle.dataset.target_utils import TokenDecomposer
from miracle.utilities.device import get_device

def get_device_for_eval():
    """Use CPU for evaluation to avoid MPS issues with non-float inputs."""
    return 'cpu'


def load_model(checkpoint_path: str, vocab_path: str, device: str):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load decomposer
    decomposer = TokenDecomposer(vocab_path)
    
    # Get config from checkpoint (key is 'config' not 'model_config')
    # Config uses: hidden_dim, num_layers, num_heads (not d_model, lstm_layers, n_heads)
    model_config = checkpoint.get('config', {})

    # Create backbone - map checkpoint config keys to ModelConfig parameter names
    # sensor_dims: [continuous_dim, categorical_dim] = [219, 4] from data
    backbone_config = ModelConfig(
        sensor_dims=[219, 4],
        d_model=model_config.get('hidden_dim', 256),
        lstm_layers=model_config.get('num_layers', 5),
        gcode_vocab=decomposer.vocab_size,
        n_heads=model_config.get('num_heads', 8),
        dropout=0.2
    )
    backbone = MM_DTAE_LSTM(backbone_config)
    
    # Create multi-head LM (use hidden_dim from config)
    # num_layers in MultiHeadGCodeLM matches backbone's num_layers
    multihead_lm = MultiHeadGCodeLM(
        d_model=model_config.get('hidden_dim', 256),
        n_commands=decomposer.n_commands,
        n_param_types=decomposer.n_param_types,
        n_param_values=decomposer.n_param_values,
        nhead=model_config.get('num_heads', 8),
        num_layers=model_config.get('num_layers', 6),  # Match checkpoint
        dropout=0.2,
        vocab_size=decomposer.vocab_size,
        n_operation_types=6  # 6 operation types in checkpoint
    )
    
    # Load weights (key is 'multihead_state_dict' not 'model_state_dict')
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    multihead_lm.load_state_dict(checkpoint['multihead_state_dict'])
    
    backbone = backbone.to(device)
    multihead_lm = multihead_lm.to(device)
    backbone.eval()
    multihead_lm.eval()
    
    return backbone, multihead_lm, decomposer


def assemble_gcode_string(
    decomposer: TokenDecomposer,
    type_pred: int,
    command_pred: int,
    param_type_pred: int,
    param_value_pred: float,
    include_command: bool = True
) -> str:
    """
    Assemble a single token into G-code string representation.

    Args:
        decomposer: TokenDecomposer for vocabulary lookup
        type_pred: Predicted token type (0=SPECIAL, 1=COMMAND, 2=PARAMETER, 3=NUMERIC)
        command_pred: Predicted command ID (index into command_tokens)
        param_type_pred: Predicted parameter type ID (index into param_tokens)
        param_value_pred: Predicted numeric value (continuous)
        include_command: Whether to include command in output

    Returns:
        G-code string fragment
    """
    if type_pred == 0:  # SPECIAL
        return ""

    if type_pred == 1:  # COMMAND
        if command_pred < len(decomposer.command_tokens):
            return decomposer.command_tokens[command_pred]
        return "G0"  # Default

    if type_pred == 2:  # PARAMETER (axis letter like X, Y, Z)
        # Skip standalone parameter letters - they're included with NUMERIC tokens
        # The model alternates: NUMERIC (has axis+value) -> PARAMETER -> NUMERIC...
        # To avoid duplicate axis letters, skip PARAMETER tokens
        return ""

    if type_pred == 3:  # NUMERIC
        # Get axis letter from param_type prediction
        if param_type_pred < len(decomposer.param_tokens):
            axis = decomposer.param_tokens[param_type_pred]
        else:
            axis = "X"  # Default

        # Format value with appropriate precision
        if abs(param_value_pred) < 0.001:
            value_str = f"{param_value_pred:.4f}"
        elif abs(param_value_pred) < 10:
            value_str = f"{param_value_pred:.3f}"
        else:
            value_str = f"{param_value_pred:.2f}"

        # Return axis letter + value combined (e.g., "X1.234")
        return f"{axis}{value_str}"

    return ""


def reconstruct_sequence(
    decomposer: TokenDecomposer,
    type_preds: List[int],
    command_preds: List[int],
    param_type_preds: List[int],
    param_value_preds: List[float],
    first_token_command: str = None
) -> str:
    """
    Reconstruct full G-code line from predicted tokens.

    Args:
        decomposer: TokenDecomposer
        type_preds: Predicted token types for each position
        command_preds: Predicted command IDs
        param_type_preds: Predicted param type IDs
        param_value_preds: Predicted numeric values
        first_token_command: Override for first command (since model can't predict it)

    Returns:
        Full G-code string
    """
    parts = []

    # Add first command if provided (model can't predict it from targets)
    if first_token_command:
        parts.append(first_token_command)

    # Process predicted tokens
    for i, (type_p, cmd_p, pt_p, pv_p) in enumerate(zip(
        type_preds, command_preds, param_type_preds, param_value_preds
    )):
        if type_p == 0:  # SPECIAL - stop
            break

        token_str = assemble_gcode_string(
            decomposer, type_p, cmd_p, pt_p, pv_p,
            include_command=(first_token_command is None)
        )

        if token_str:
            parts.append(token_str)

    # Parts are now already formatted (NUMERIC tokens include axis letter)
    # Just join with spaces
    return ' '.join(parts)


def evaluate_gcode_assembly(
    checkpoint_path: str,
    data_dir: str,
    vocab_path: str,
    split: str = 'val',
    num_samples: int = 20
):
    """
    Evaluate G-code assembly on a data split.
    """
    device = get_device_for_eval()  # Use CPU to avoid MPS issues
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    backbone, multihead_lm, decomposer = load_model(checkpoint_path, vocab_path, device)

    # Load data
    data_path = Path(data_dir) / f'{split}_sequences.npz'
    print(f"Loading data from: {data_path}")
    data = np.load(data_path, allow_pickle=True)

    continuous = torch.tensor(data['continuous'], dtype=torch.float32)
    categorical = torch.tensor(data['categorical'], dtype=torch.float32)  # Model expects float, not long
    tokens = torch.tensor(data['tokens'], dtype=torch.long)
    gcode_texts = data['gcode_texts']
    lengths = torch.tensor(data['lengths'], dtype=torch.long)

    # Load ground truth param values if available
    param_value_raw = data.get('param_value_raw', None)
    has_gt_values = param_value_raw is not None

    if has_gt_values:
        print("Ground truth param_value_raw available - will show GT reconstruction")
    else:
        print("WARNING: param_value_raw not in data - GT reconstruction not available")

    print(f"\nEvaluating {num_samples} samples from {split} set:")
    print("=" * 80)

    correct = 0
    correct_structure = 0  # Count samples where axis letters match
    total = 0

    for idx in range(min(num_samples, len(tokens))):
        # Prepare batch of 1
        cont = continuous[idx:idx+1].to(device)
        cat = categorical[idx:idx+1].to(device)
        toks = tokens[idx:idx+1].to(device)
        lens = lengths[idx:idx+1].to(device)

        # Teacher-forcing input
        tgt_in = toks[:, :-1]

        # Forward pass
        with torch.no_grad():
            backbone_out = backbone(mods=[cont, cat], lengths=lens, gcode_in=tgt_in)
            memory = backbone_out['memory']
            logits = multihead_lm(memory, tgt_in)

        # Get predictions
        type_logits = logits['type_logits']
        command_logits = logits['command_logits']
        param_type_logits = logits['param_type_logits']

        if 'param_value_regression' in logits:
            # Shape: [batch, seq_len, 1] -> squeeze last dim to [seq_len]
            param_value_preds = logits['param_value_regression'][0].squeeze(-1).cpu().numpy()
        else:
            param_value_logits = logits['param_value_logits']
            param_value_preds = param_value_logits[0].argmax(dim=-1).cpu().numpy()

        type_preds = type_logits[0].argmax(dim=-1).cpu().tolist()
        command_preds = command_logits[0].argmax(dim=-1).cpu().tolist()
        param_type_preds = param_type_logits[0].argmax(dim=-1).cpu().tolist()

        # Get actual first token (command) from input
        first_token_id = tokens[idx, 0].item()
        first_token = decomposer.id2token.get(first_token_id, '?')

        # Determine if first token is a command
        first_token_command = None
        if first_token in decomposer.command2id:
            first_token_command = first_token

        # Reconstruct G-code with model predictions
        reconstructed = reconstruct_sequence(
            decomposer,
            type_preds,
            command_preds,
            param_type_preds,
            param_value_preds.tolist() if hasattr(param_value_preds, 'tolist') else param_value_preds,
            first_token_command=first_token_command
        )

        # Reconstruct with ground truth values (if available)
        gt_reconstructed = None
        if has_gt_values:
            # param_value_raw is [N, seq_len], we need values for positions 1, 3, 5, ... (NUMERIC positions)
            gt_values = param_value_raw[idx][1:]  # Shift by 1 to align with model output
            gt_reconstructed = reconstruct_sequence(
                decomposer,
                type_preds,
                command_preds,
                param_type_preds,
                gt_values.tolist(),
                first_token_command=first_token_command
            )

        actual = str(gcode_texts[idx])

        # Compare
        match = "✓" if reconstructed.strip().lower() == actual.strip().lower() else "✗"
        if match == "✓":
            correct += 1
        total += 1

        # Check structure match (axis letters correct)
        axes_prefix_match = False
        if gt_reconstructed:
            gt_match = "✓" if gt_reconstructed.strip().lower() == actual.strip().lower() else "✗"
            if gt_match == "✓":
                correct_structure += 1

            # Also check if axis letters sequence matches (ignoring values and extra tokens)
            import re
            actual_axes = re.findall(r'([XYZFRS])', actual)
            gt_axes = re.findall(r'([XYZFRS])', gt_reconstructed)
            # Check if actual axes are a prefix of predicted axes (model might over-generate)
            axes_prefix_match = gt_axes[:len(actual_axes)] == actual_axes if actual_axes else False

        print(f"\n{idx:3d}. {match}")
        print(f"   Actual:           {actual}")
        print(f"   Model prediction: {reconstructed}")
        if gt_reconstructed:
            print(f"   With GT values:   {gt_reconstructed}")
            if axes_prefix_match:
                print(f"   Axes match:       {actual_axes} -> {gt_axes[:len(actual_axes)]} {'(+extra)' if len(gt_axes) > len(actual_axes) else ''}")
            else:
                print(f"   Axes MISMATCH:    {actual_axes} != {gt_axes[:len(actual_axes)] if len(gt_axes) >= len(actual_axes) else gt_axes}")
        print(f"   First token:      {first_token} (command: {first_token_command})")

    print("\n" + "=" * 80)
    print(f"Exact match (model values): {correct}/{total} = {100*correct/total:.1f}%")
    if has_gt_values:
        print(f"Structure match (GT values): {correct_structure}/{total} = {100*correct_structure/total:.1f}%")
        print("\nNote: 'Structure match' shows if axis prediction is correct")
        print("      'Exact match' shows if both axis AND values are correct")
        print("      Low exact match with high structure match = param_value_regression needs fixing")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='G-code String Assembly')
    parser.add_argument('--checkpoint', type=str, 
                        default='outputs/ultimate_model_v1/checkpoint_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, 
                        default='outputs/processed_hybrid',
                        help='Path to processed data')
    parser.add_argument('--vocab-path', type=str,
                        default='data/vocabulary_1digit_hybrid.json',
                        help='Path to vocabulary file')
    parser.add_argument('--split', type=str, default='val',
                        help='Data split to evaluate (train/val/test)')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of samples to evaluate')
    
    args = parser.parse_args()
    
    evaluate_gcode_assembly(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        vocab_path=args.vocab_path,
        split=args.split,
        num_samples=args.num_samples
    )
