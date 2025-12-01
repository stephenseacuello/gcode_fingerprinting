#!/usr/bin/env python3
"""
Load experimental results for visualization.

This module provides utilities to:
- Load model checkpoints (local or W&B)
- Run comprehensive evaluation on test set
- Compute bootstrap confidence intervals
- Extract embeddings and metrics for visualization

Reuses existing evaluation infrastructure from test_sweep_run.py and test_local_checkpoint.py
to avoid code duplication.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm

# Import existing evaluation infrastructure
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.multihead_lm import MultiHeadGCodeLM
from miracle.dataset.dataset import GCodeDataset, collate_fn
from miracle.dataset.target_utils import TokenDecomposer
from miracle.training.metrics import compute_confusion_matrix
from torch.utils.data import DataLoader


def load_checkpoint(checkpoint_path: str, vocab_path: str = 'data/gcode_vocab_v2.json',
                   device: str = 'cpu') -> dict:
    """
    Load model from local checkpoint file.

    Reuses logic from scripts/test_local_checkpoint.py

    Args:
        checkpoint_path: Path to checkpoint .pt file
        vocab_path: Path to vocabulary JSON file
        device: Device to load model on

    Returns:
        dict with keys: backbone, multihead_lm, decomposer, config
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    # Create decomposer
    decomposer = TokenDecomposer(vocab_path)

    # Infer vocab size from checkpoint BEFORE creating models
    # Check both backbone and multihead state dicts for embedding weights
    actual_vocab_size = None

    # First try backbone state dict (gcode_head.embed.weight)
    if 'backbone_state_dict' in checkpoint:
        for key in checkpoint['backbone_state_dict'].keys():
            if 'embed.weight' in key:
                embed_weight = checkpoint['backbone_state_dict'][key]
                actual_vocab_size = embed_weight.shape[0]
                print(f"  Inferred vocab size from backbone checkpoint: {actual_vocab_size}")
                break

    # If not found, try multihead state dict
    if actual_vocab_size is None and 'multihead_state_dict' in checkpoint:
        if 'embed.weight' in checkpoint['multihead_state_dict']:
            embed_weight = checkpoint['multihead_state_dict']['embed.weight']
            actual_vocab_size = embed_weight.shape[0]
            print(f"  Inferred vocab size from multihead checkpoint: {actual_vocab_size}")

    # Fallback to vocabulary file size
    if actual_vocab_size is None:
        actual_vocab_size = len(decomposer.vocab)
        print(f"  Using vocab size from file: {actual_vocab_size}")

    # Create backbone (MM_DTAE_LSTM)
    # Infer sensor dimensions from checkpoint if available
    if 'backbone_state_dict' in checkpoint:
        # Get dimensions from encoder weights
        encoder0_weight = checkpoint['backbone_state_dict']['encoders.0.proj.0.weight']
        encoder1_weight = checkpoint['backbone_state_dict']['encoders.1.proj.0.weight']
        n_continuous = encoder0_weight.shape[1]
        n_categorical = encoder1_weight.shape[1]
        print(f"  Inferred sensor dims from checkpoint: continuous={n_continuous}, categorical={n_categorical}")
    else:
        # Fallback to common dimensions
        n_continuous, n_categorical = 155, 4
        print(f"  Using default sensor dims: continuous={n_continuous}, categorical={n_categorical}")

    backbone_config = ModelConfig(
        sensor_dims=[n_continuous, n_categorical],
        d_model=config.get('hidden_dim', 128),
        lstm_layers=config.get('num_layers', 2),
        gcode_vocab=actual_vocab_size,  # Use inferred vocab size instead of hardcoded 170
        n_heads=config.get('num_heads', 4),
    )
    backbone = MM_DTAE_LSTM(backbone_config).to(device)

    # Load backbone state
    if 'backbone_state_dict' in checkpoint:
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
    elif 'model_state_dict' in checkpoint:
        backbone.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise KeyError("Checkpoint missing backbone_state_dict or model_state_dict")

    # Create multihead LM (using vocab size inferred earlier)
    multihead_lm = MultiHeadGCodeLM(
        d_model=config.get('hidden_dim', 128),
        n_commands=decomposer.n_commands,
        n_param_types=decomposer.n_param_types,
        n_param_values=decomposer.n_param_values,
        nhead=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 2),
        vocab_size=actual_vocab_size,
    ).to(device)

    # Load multihead state
    if 'multihead_state_dict' in checkpoint:
        multihead_lm.load_state_dict(checkpoint['multihead_state_dict'])
    else:
        raise KeyError("Checkpoint missing multihead_state_dict")

    backbone.eval()
    multihead_lm.eval()

    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Config: {config}")

    return {
        'backbone': backbone,
        'multihead_lm': multihead_lm,
        'decomposer': decomposer,
        'config': config,
        'device': device,
    }


def load_wandb_checkpoint(run_id: str, vocab_path: str = 'data/gcode_vocab_v2.json',
                         device: str = 'cpu') -> dict:
    """
    Load model from W&B run.

    Reuses logic from scripts/test_sweep_run.py

    Args:
        run_id: W&B run ID (e.g., "entity/project/run_id")
        vocab_path: Path to vocabulary JSON file
        device: Device to load model on

    Returns:
        dict with keys: backbone, multihead_lm, decomposer, config
    """
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb not installed. Install with: pip install wandb")

    print(f"Fetching checkpoint from W&B run: {run_id}...")

    api = wandb.Api()
    run = api.run(run_id)

    # Download checkpoint
    checkpoint_file = None
    for file in run.files():
        if 'checkpoint' in file.name and file.name.endswith('.pt'):
            print(f"  Downloading {file.name}...")
            checkpoint_file = file.download(replace=True)
            break

    if checkpoint_file is None:
        raise FileNotFoundError(f"No checkpoint file found in run {run_id}")

    # Load using local checkpoint loader
    result = load_checkpoint(checkpoint_file.name, vocab_path, device)

    # Add W&B run info
    result['wandb_run'] = run
    result['wandb_config'] = dict(run.config)
    result['wandb_summary'] = dict(run.summary)

    return result


def evaluate_model_on_test(model_dict: dict, test_npz_path: str,
                           batch_size: int = 8, max_batches: Optional[int] = None) -> dict:
    """
    Run comprehensive evaluation on test set.

    Reuses evaluation loop from test_sweep_run.py

    Args:
        model_dict: Dict from load_checkpoint() or load_wandb_checkpoint()
        test_npz_path: Path to test data .npz file
        batch_size: Batch size for evaluation
        max_batches: Optional limit on number of batches (for quick testing)

    Returns:
        dict with keys:
            - accuracies: Dict of mean accuracies per head
            - per_sample_accuracies: Dict of per-sample accuracy arrays
            - confusion_matrices: Dict of confusion matrices per head
            - predictions: List of prediction examples
    """
    print(f"\nEvaluating on test set: {test_npz_path}")

    backbone = model_dict['backbone']
    multihead_lm = model_dict['multihead_lm']
    decomposer = model_dict['decomposer']
    device = model_dict['device']

    # Load test dataset
    test_dataset = GCodeDataset(test_npz_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Storage for results
    all_type_preds = []
    all_command_preds = []
    all_param_type_preds = []
    all_param_value_preds = []

    all_type_targets = []
    all_command_targets = []
    all_param_type_targets = []
    all_param_value_targets = []

    per_sample_type_acc = []
    per_sample_command_acc = []
    per_sample_param_type_acc = []
    per_sample_param_value_acc = []
    per_sample_overall_acc = []

    predictions_examples = []

    # Evaluation loop
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=min(len(test_loader), max_batches or len(test_loader)),
                   desc="Evaluating")

        for batch_idx, batch in pbar:
            if max_batches and batch_idx >= max_batches:
                break

            continuous = batch['continuous'].to(device)
            categorical = batch['categorical'].to(device).float()  # Convert to float
            tokens = batch['tokens'].to(device)

            # Clip tokens to valid vocabulary range for the model
            # (test data may have tokens from full vocab, but model only supports subset)
            vocab_size = multihead_lm.embed.num_embeddings
            tokens = torch.clamp(tokens, min=0, max=vocab_size-1)

            # Compute sequence lengths (all timesteps are valid, no padding)
            batch_size = continuous.size(0)
            seq_len = continuous.size(1)
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

            # Forward backbone
            mods = [continuous, categorical]
            backbone_out = backbone(mods=mods, lengths=lengths)
            memory = backbone_out['memory']

            # Prepare target tokens (teacher forcing)
            bos_token = 1
            tgt_in = torch.cat([
                torch.full((tokens.size(0), 1), bos_token, dtype=tokens.dtype, device=device),
                tokens[:, :-1]
            ], dim=1)

            tgt_out = tokens[:, 1:]

            # Decompose targets
            tgt_decomposed = decomposer.decompose_batch(tgt_out)

            # Forward multihead LM
            logits = multihead_lm(memory, tgt_in)

            # Get predictions
            type_preds = torch.argmax(logits['type_logits'][:, :-1], dim=-1)
            command_preds = torch.argmax(logits['command_logits'][:, :-1], dim=-1)
            param_type_preds = torch.argmax(logits['param_type_logits'][:, :-1], dim=-1)

            # Handle both regression and classification modes for param_value
            if 'param_value_regression' in logits:
                # Regression mode: round continuous predictions to nearest integer bucket
                param_value_preds = torch.round(logits['param_value_regression'][:, :-1].squeeze(-1)).long()
                param_value_preds = torch.clamp(param_value_preds, 0, decomposer.n_param_values - 1)
            elif 'param_value_logits' in logits:
                # Classification mode (legacy)
                param_value_preds = torch.argmax(logits['param_value_logits'][:, :-1], dim=-1)
            else:
                raise KeyError(f"No param_value prediction found in logits. Available keys: {logits.keys()}")

            # Store for confusion matrices
            all_type_preds.append(type_preds.cpu())
            all_command_preds.append(command_preds.cpu())
            all_param_type_preds.append(param_type_preds.cpu())
            all_param_value_preds.append(param_value_preds.cpu())

            all_type_targets.append(tgt_decomposed['type'].cpu())
            all_command_targets.append(tgt_decomposed['command_id'].cpu())
            all_param_type_targets.append(tgt_decomposed['param_type_id'].cpu())
            all_param_value_targets.append(tgt_decomposed['param_value_id'].cpu())

            # Compute per-sample accuracies
            B = type_preds.size(0)
            for b in range(B):
                type_acc = (type_preds[b] == tgt_decomposed['type'][b]).float().mean().item()
                command_acc = (command_preds[b] == tgt_decomposed['command_id'][b]).float().mean().item()
                param_type_acc = (param_type_preds[b] == tgt_decomposed['param_type_id'][b]).float().mean().item()
                param_value_acc = (param_value_preds[b] == tgt_decomposed['param_value_id'][b]).float().mean().item()

                # Overall accuracy (all heads must match)
                overall_correct = (
                    (type_preds[b] == tgt_decomposed['type'][b]) &
                    (command_preds[b] == tgt_decomposed['command_id'][b]) &
                    (param_type_preds[b] == tgt_decomposed['param_type_id'][b]) &
                    (param_value_preds[b] == tgt_decomposed['param_value_id'][b])
                ).float().mean().item()

                per_sample_type_acc.append(type_acc)
                per_sample_command_acc.append(command_acc)
                per_sample_param_type_acc.append(param_type_acc)
                per_sample_param_value_acc.append(param_value_acc)
                per_sample_overall_acc.append(overall_correct)

            # Store first few examples for visualization
            if batch_idx < 2:
                for b in range(min(2, B)):
                    predictions_examples.append({
                        'continuous': continuous[b].cpu().numpy(),
                        'categorical': categorical[b].cpu().numpy(),
                        'predicted_tokens': {
                            'type': type_preds[b].cpu().numpy(),
                            'command': command_preds[b].cpu().numpy(),
                            'param_type': param_type_preds[b].cpu().numpy(),
                            'param_value': param_value_preds[b].cpu().numpy(),
                        },
                        'target_tokens': {
                            'type': tgt_decomposed['type'][b].cpu().numpy(),
                            'command': tgt_decomposed['command_id'][b].cpu().numpy(),
                            'param_type': tgt_decomposed['param_type_id'][b].cpu().numpy(),
                            'param_value': tgt_decomposed['param_value_id'][b].cpu().numpy(),
                        },
                        'gcode_text': batch['gcode_texts'][b] if 'gcode_texts' in batch else None,
                    })

    # Concatenate all predictions/targets
    all_type_preds = torch.cat(all_type_preds, dim=0).flatten()
    all_command_preds = torch.cat(all_command_preds, dim=0).flatten()
    all_param_type_preds = torch.cat(all_param_type_preds, dim=0).flatten()
    all_param_value_preds = torch.cat(all_param_value_preds, dim=0).flatten()

    all_type_targets = torch.cat(all_type_targets, dim=0).flatten()
    all_command_targets = torch.cat(all_command_targets, dim=0).flatten()
    all_param_type_targets = torch.cat(all_param_type_targets, dim=0).flatten()
    all_param_value_targets = torch.cat(all_param_value_targets, dim=0).flatten()

    # Compute mean accuracies
    type_acc = (all_type_preds == all_type_targets).float().mean().item() * 100
    command_acc = (all_command_preds == all_command_targets).float().mean().item() * 100
    param_type_acc = (all_param_type_preds == all_param_type_targets).float().mean().item() * 100
    param_value_acc = (all_param_value_preds == all_param_value_targets).float().mean().item() * 100

    overall_acc = np.mean(per_sample_overall_acc) * 100

    print(f"\n✓ Evaluation complete!")
    print(f"  Type:        {type_acc:.2f}%")
    print(f"  Command:     {command_acc:.2f}%")
    print(f"  Param Type:  {param_type_acc:.2f}%")
    print(f"  Param Value: {param_value_acc:.2f}%")
    print(f"  Overall:     {overall_acc:.2f}%")

    # Compute confusion matrices (for command and param_type heads)
    command_cm = compute_confusion_matrix(
        all_command_targets.numpy(),
        all_command_preds.numpy(),
        num_classes=decomposer.n_commands
    )

    param_type_cm = compute_confusion_matrix(
        all_param_type_targets.numpy(),
        all_param_type_preds.numpy(),
        num_classes=decomposer.n_param_types
    )

    return {
        'accuracies': {
            'type': type_acc,
            'command': command_acc,
            'param_type': param_type_acc,
            'param_value': param_value_acc,
            'overall': overall_acc,
        },
        'per_sample_accuracies': {
            'type': np.array(per_sample_type_acc) * 100,
            'command': np.array(per_sample_command_acc) * 100,
            'param_type': np.array(per_sample_param_type_acc) * 100,
            'param_value': np.array(per_sample_param_value_acc) * 100,
            'overall': np.array(per_sample_overall_acc) * 100,
        },
        'confusion_matrices': {
            'command': command_cm,
            'param_type': param_type_cm,
        },
        'predictions': predictions_examples,
    }


def compute_bootstrap_ci(samples: np.ndarray, n_bootstrap: int = 1000,
                        ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals.

    Args:
        samples: Array of per-sample values (e.g., accuracies)
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence interval level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    n = len(samples)
    bootstrap_means = []

    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        resample = rng.choice(samples, size=n, replace=True)
        bootstrap_means.append(resample.mean())

    bootstrap_means = np.array(bootstrap_means)
    mean = samples.mean()

    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_means, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return mean, ci_lower, ci_upper


def extract_token_embeddings(multihead_lm, vocab_path: str = 'data/gcode_vocab_v2.json') -> dict:
    """
    Extract token embeddings from trained model.

    Args:
        multihead_lm: Trained MultiHeadGCodeLM model
        vocab_path: Path to vocabulary file

    Returns:
        dict with keys:
            - embeddings: [vocab_size, d_model] numpy array
            - token_to_idx: Dict mapping token string to index
            - token_types: Dict mapping token to type (COMMAND/PARAM/etc)
    """
    import json

    # Get embeddings from model
    embeddings = multihead_lm.embed.weight.detach().cpu().numpy()

    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    token_to_idx = vocab_data.get('vocab', vocab_data)

    # Infer token types (basic heuristic)
    token_types = {}
    for token, idx in token_to_idx.items():
        if token in ['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<MASK>']:
            token_types[token] = 'Special'
        elif token.startswith('G') or token.startswith('M'):
            token_types[token] = 'Command'
        elif token.startswith('NUM_'):
            token_types[token] = 'Param Value'
        elif len(token) == 1 and token.isalpha():
            token_types[token] = 'Param Type'
        else:
            token_types[token] = 'Other'

    return {
        'embeddings': embeddings,
        'token_to_idx': token_to_idx,
        'token_types': token_types,
    }


if __name__ == '__main__':
    # Quick test
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test-data', default='data/test_sequences.npz')
    parser.add_argument('--vocab', default='data/gcode_vocab_v2.json')

    args = parser.parse_args()

    # Load and evaluate
    model_dict = load_checkpoint(args.checkpoint, args.vocab)
    results = evaluate_model_on_test(model_dict, args.test_data, batch_size=8, max_batches=5)

    # Compute CIs
    print("\nBootstrap 95% Confidence Intervals:")
    for head in ['type', 'command', 'param_type', 'param_value', 'overall']:
        samples = results['per_sample_accuracies'][head]
        mean, lower, upper = compute_bootstrap_ci(samples)
        print(f"  {head:12s}: {mean:.2f}% [{lower:.2f}%, {upper:.2f}%]")

    # Extract embeddings
    embeddings = extract_token_embeddings(model_dict['multihead_lm'], args.vocab)
    print(f"\nExtracted embeddings: shape {embeddings['embeddings'].shape}")
