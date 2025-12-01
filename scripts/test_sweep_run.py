#!/usr/bin/env python3
"""
Test inference on models from W&B sweep runs.

Usage:
    # Test a specific run
    python scripts/test_sweep_run.py --run-id ofx0nhdx

    # Test best run from sweep
    python scripts/test_sweep_run.py --sweep-id e3brf5ss --best

    # Compare top 5 runs
    python scripts/test_sweep_run.py --sweep-id e3brf5ss --top 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from miracle.dataset.dataset import GCodeDataset, collate_fn
from miracle.dataset.target_utils import TokenDecomposer
from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.multihead_lm import MultiHeadGCodeLM


def load_model_from_wandb(run_path: str, vocab_path: str, device='cpu'):
    """
    Load model checkpoint from W&B run.

    Args:
        run_path: W&B run path (entity/project/run_id)
        vocab_path: Path to vocabulary file
        device: Device to load model on

    Returns:
        backbone, multihead_lm, config, decomposer
    """
    print(f"Loading model from W&B run: {run_path}")

    api = wandb.Api()
    run = api.run(run_path)

    # Download checkpoint
    print("Downloading checkpoint...")
    checkpoint_file = None
    for file in run.files():
        if file.name.endswith('checkpoint_best.pt'):
            checkpoint_file = file
            break

    if checkpoint_file is None:
        print("⚠️  No checkpoint found in run. Run may still be training.")
        return None, None, None, None

    # Download to temp location
    checkpoint_path = checkpoint_file.download(replace=True, root='./temp_checkpoints')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path.name, map_location=device)
    config = checkpoint.get('config', {})

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

    print(f"✓ Loaded model from run: {run.name}")
    print(f"  Config: {config}")
    print(f"  Val accuracy (from training): {checkpoint.get('val_acc', 0):.2%}")

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


def get_sweep_runs(sweep_id: str, top_k: int = None):
    """
    Get runs from a sweep, optionally filtered to top K.

    Args:
        sweep_id: W&B sweep ID
        top_k: Return only top K runs by val/overall_accuracy

    Returns:
        List of run objects
    """
    api = wandb.Api()
    sweep = api.sweep(sweep_id)

    # Get all runs
    runs = list(sweep.runs)

    # Filter finished runs with metrics
    finished_runs = [
        r for r in runs
        if r.state == 'finished' and 'val/overall_accuracy' in r.summary
    ]

    print(f"\nSweep: {sweep_id}")
    print(f"  Total runs: {len(runs)}")
    print(f"  Finished runs: {len(finished_runs)}")

    if not finished_runs:
        print("⚠️  No finished runs yet. Wait for some runs to complete.")
        return []

    # Sort by accuracy
    finished_runs.sort(key=lambda r: r.summary.get('val/overall_accuracy', 0), reverse=True)

    if top_k:
        finished_runs = finished_runs[:top_k]

    return finished_runs


def compare_runs(runs, test_loader, vocab_path, device='cpu'):
    """
    Compare multiple runs.

    Args:
        runs: List of W&B run objects
        test_loader: Test data loader
        vocab_path: Path to vocabulary file
        device: Device to run on
    """
    print("\n" + "=" * 80)
    print("COMPARING SWEEP RUNS")
    print("=" * 80)

    results = []

    for i, run in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] Testing run: {run.name}")
        print(f"  Run ID: {run.id}")
        print(f"  Val accuracy (from training): {run.summary.get('val/overall_accuracy', 0):.2f}%")
        print(f"  Config: {run.config}")

        # Load model
        run_path = f"{run.entity}/{run.project}/{run.id}"
        backbone, multihead_lm, config, decomposer = load_model_from_wandb(
            run_path, vocab_path, device
        )

        if backbone is None:
            print("  ⚠️  Skipping (no checkpoint)")
            continue

        # Evaluate
        metrics = evaluate_model(backbone, multihead_lm, decomposer, test_loader, device)

        # Store results
        result = {
            'run_id': run.id,
            'run_name': run.name,
            'config': run.config,
            'val_accuracy_training': run.summary.get('val/overall_accuracy', 0),
            'metrics': metrics,
        }
        results.append(result)

        # Print results
        print(f"\n  Test Set Results:")
        print(f"    Overall:       {metrics['overall_accuracy']:.2f}%")
        print(f"    Type:          {metrics['type_accuracy']:.2f}%")
        print(f"    Command:       {metrics['command_accuracy']:.2f}%")
        print(f"    Param Type:    {metrics['param_type_accuracy']:.2f}%")
        print(f"    Param Value:   {metrics['param_value_accuracy']:.2f}%")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Run':<20} | {'Val (train)':<12} | {'Test Overall':<12} | {'Command':<8} | {'Param Val':<10}")
    print("-" * 80)

    for result in results:
        print(
            f"{result['run_name']:<20} | "
            f"{result['val_accuracy_training']:>11.2f}% | "
            f"{result['metrics']['overall_accuracy']:>11.2f}% | "
            f"{result['metrics']['command_accuracy']:>7.2f}% | "
            f"{result['metrics']['param_value_accuracy']:>9.2f}%"
        )

    print("=" * 80)

    # Best run
    if results:
        best = max(results, key=lambda r: r['metrics']['overall_accuracy'])
        print(f"\n🏆 Best Test Accuracy: {best['run_name']}")
        print(f"   Overall: {best['metrics']['overall_accuracy']:.2f}%")
        print(f"   Config: {best['config']}")


def main():
    parser = argparse.ArgumentParser(description="Test models from W&B sweep runs")

    parser.add_argument(
        '--run-id',
        type=str,
        help='Specific W&B run ID to test'
    )
    parser.add_argument(
        '--sweep-id',
        type=str,
        help='W&B sweep ID (entity/project/sweep_id or just sweep_id)'
    )
    parser.add_argument(
        '--best',
        action='store_true',
        help='Test only the best run from sweep'
    )
    parser.add_argument(
        '--top',
        type=int,
        help='Test top N runs from sweep'
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

    # Validate arguments
    if not args.run_id and not args.sweep_id:
        parser.error("Must provide either --run-id or --sweep-id")

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

    # Test specific run
    if args.run_id:
        run_path = f"seacuello-university-of-rhode-island/uncategorized/{args.run_id}"
        backbone, multihead_lm, config, decomposer = load_model_from_wandb(
            run_path, vocab_path, args.device
        )

        if backbone:
            metrics = evaluate_model(
                backbone, multihead_lm, decomposer, test_loader, args.device
            )

            print("\n" + "=" * 60)
            print("TEST RESULTS")
            print("=" * 60)
            print(f"Overall Accuracy:       {metrics['overall_accuracy']:.2f}%")
            print(f"Type Accuracy:          {metrics['type_accuracy']:.2f}%")
            print(f"Command Accuracy:       {metrics['command_accuracy']:.2f}%")
            print(f"Param Type Accuracy:    {metrics['param_type_accuracy']:.2f}%")
            print(f"Param Value Accuracy:   {metrics['param_value_accuracy']:.2f}%")
            print("=" * 60)

    # Test sweep runs
    elif args.sweep_id:
        # Add full path if needed
        if '/' not in args.sweep_id:
            sweep_id = f"seacuello-university-of-rhode-island/uncategorized/{args.sweep_id}"
        else:
            sweep_id = args.sweep_id

        # Get runs
        if args.best:
            runs = get_sweep_runs(sweep_id, top_k=1)
        elif args.top:
            runs = get_sweep_runs(sweep_id, top_k=args.top)
        else:
            runs = get_sweep_runs(sweep_id)

        if runs:
            compare_runs(runs, test_loader, vocab_path, args.device)


if __name__ == "__main__":
    main()
