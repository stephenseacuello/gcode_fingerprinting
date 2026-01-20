#!/usr/bin/env python3
"""
Run Architecture Ablation Studies for G-code Fingerprinting.

This script runs systematic ablation studies to measure the contribution
of each architectural component:

1. Cross-attention ablation
2. Positional encoding ablation
3. Operation conditioning ablation
4. Multi-head vs single-head vocabulary
5. Model size scaling
6. Dropout sensitivity
7. Learning rate sensitivity
8. Augmentation probability sweep

Usage:
    python scripts/run_architecture_ablations.py --output-dir outputs/ablations
    python scripts/run_architecture_ablations.py --ablation cross_attention --seed 42
    python scripts/run_architecture_ablations.py --all --seeds 42,123,456

Author: Claude Code
Date: December 2025
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def run_training(
    output_dir: str,
    run_name: str,
    extra_args: List[str],
    base_args: Dict,
    use_wandb: bool = False,
    wandb_project: str = 'gcode-ablations',
) -> Dict:
    """Run a single training configuration."""

    run_output = Path(output_dir) / run_name
    run_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, 'scripts/training/train_sensor_multihead.py',
        '--seed', str(base_args['seed']),
        '--split-dir', base_args['split_dir'],
        '--vocab-path', base_args['vocab_path'],
        '--encoder-path', base_args['encoder_path'],
        '--output-dir', str(run_output),
        '--max-epochs', str(base_args['max_epochs']),
        '--patience', str(base_args['patience']),
        '--use-enhanced-encoder',
        '--use-multiscale-encoder',
        '--use-multihead-pooling',
    ]

    # Add extra args
    cmd.extend(extra_args)

    if use_wandb:
        cmd.extend([
            '--use-wandb',
            '--wandb-project', wandb_project,
            '--run-name', run_name,
        ])

    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"Extra args: {' '.join(extra_args)}")
    print('='*60)

    result = subprocess.run(cmd, capture_output=False, text=True)

    # Load results
    results_path = run_output / 'results.json'
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {'error': f'No results found, return code: {result.returncode}'}


def run_cross_attention_ablation(base_args: Dict, output_dir: str, use_wandb: bool = False) -> Dict:
    """Ablate cross-attention mechanism."""
    results = {}

    # Baseline (with cross-attention)
    results['with_cross_attention'] = run_training(
        output_dir=output_dir,
        run_name=f"cross_attn_ON_seed{base_args['seed']}",
        extra_args=[
            '--d-model', '192',
            '--n-layers', '4',
            '--learning-rate', '0.0002',
        ],
        base_args=base_args,
        use_wandb=use_wandb,
    )

    # Without cross-attention
    results['without_cross_attention'] = run_training(
        output_dir=output_dir,
        run_name=f"cross_attn_OFF_seed{base_args['seed']}",
        extra_args=[
            '--d-model', '192',
            '--n-layers', '4',
            '--learning-rate', '0.0002',
            '--no-cross-attention',
        ],
        base_args=base_args,
        use_wandb=use_wandb,
    )

    return results


def run_positional_encoding_ablation(base_args: Dict, output_dir: str, use_wandb: bool = False) -> Dict:
    """Ablate positional encoding."""
    results = {}

    # Baseline (with positional encoding)
    results['with_positional'] = run_training(
        output_dir=output_dir,
        run_name=f"pos_enc_ON_seed{base_args['seed']}",
        extra_args=[
            '--d-model', '192',
            '--n-layers', '4',
            '--learning-rate', '0.0002',
        ],
        base_args=base_args,
        use_wandb=use_wandb,
    )

    # Without positional encoding
    results['without_positional'] = run_training(
        output_dir=output_dir,
        run_name=f"pos_enc_OFF_seed{base_args['seed']}",
        extra_args=[
            '--d-model', '192',
            '--n-layers', '4',
            '--learning-rate', '0.0002',
            '--no-positional-encoding',
        ],
        base_args=base_args,
        use_wandb=use_wandb,
    )

    return results


def run_operation_conditioning_ablation(base_args: Dict, output_dir: str, use_wandb: bool = False) -> Dict:
    """Ablate operation conditioning."""
    results = {}

    # Baseline (with operation conditioning)
    results['with_op_cond'] = run_training(
        output_dir=output_dir,
        run_name=f"op_cond_ON_seed{base_args['seed']}",
        extra_args=[
            '--d-model', '192',
            '--n-layers', '4',
            '--learning-rate', '0.0002',
        ],
        base_args=base_args,
        use_wandb=use_wandb,
    )

    # Without operation conditioning
    results['without_op_cond'] = run_training(
        output_dir=output_dir,
        run_name=f"op_cond_OFF_seed{base_args['seed']}",
        extra_args=[
            '--d-model', '192',
            '--n-layers', '4',
            '--learning-rate', '0.0002',
            '--no-operation-conditioning',
        ],
        base_args=base_args,
        use_wandb=use_wandb,
    )

    return results


def run_model_size_scaling(base_args: Dict, output_dir: str, use_wandb: bool = False) -> Dict:
    """Study model size scaling."""
    results = {}

    configs = [
        {'d_model': 128, 'n_layers': 2, 'name': 'tiny'},
        {'d_model': 128, 'n_layers': 3, 'name': 'small'},
        {'d_model': 192, 'n_layers': 4, 'name': 'base'},
        {'d_model': 256, 'n_layers': 4, 'name': 'medium'},
        {'d_model': 256, 'n_layers': 5, 'name': 'large'},
        {'d_model': 320, 'n_layers': 6, 'name': 'xlarge'},
    ]

    for config in configs:
        results[config['name']] = run_training(
            output_dir=output_dir,
            run_name=f"size_{config['name']}_seed{base_args['seed']}",
            extra_args=[
                '--d-model', str(config['d_model']),
                '--n-layers', str(config['n_layers']),
                '--learning-rate', '0.0002',
            ],
            base_args=base_args,
            use_wandb=use_wandb,
        )

    return results


def run_dropout_sensitivity(base_args: Dict, output_dir: str, use_wandb: bool = False) -> Dict:
    """Study dropout sensitivity."""
    results = {}

    for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]:
        results[f'dropout_{dropout}'] = run_training(
            output_dir=output_dir,
            run_name=f"dropout_{dropout}_seed{base_args['seed']}",
            extra_args=[
                '--d-model', '192',
                '--n-layers', '4',
                '--learning-rate', '0.0002',
                '--dropout', str(dropout),
            ],
            base_args=base_args,
            use_wandb=use_wandb,
        )

    return results


def run_learning_rate_sensitivity(base_args: Dict, output_dir: str, use_wandb: bool = False) -> Dict:
    """Study learning rate sensitivity."""
    results = {}

    for lr in [0.00005, 0.0001, 0.0002, 0.0005, 0.001]:
        results[f'lr_{lr}'] = run_training(
            output_dir=output_dir,
            run_name=f"lr_{lr}_seed{base_args['seed']}",
            extra_args=[
                '--d-model', '192',
                '--n-layers', '4',
                '--learning-rate', str(lr),
            ],
            base_args=base_args,
            use_wandb=use_wandb,
        )

    return results


def run_augmentation_ablation(base_args: Dict, output_dir: str, use_wandb: bool = False) -> Dict:
    """Study data augmentation impact."""
    results = {}

    # No augmentation
    results['no_augment'] = run_training(
        output_dir=output_dir,
        run_name=f"aug_OFF_seed{base_args['seed']}",
        extra_args=[
            '--d-model', '192',
            '--n-layers', '4',
            '--learning-rate', '0.0002',
            # No --augment flag
        ],
        base_args=base_args,
        use_wandb=use_wandb,
    )

    # With different augmentation probabilities
    for aug_prob in [0.1, 0.2, 0.3, 0.4, 0.5]:
        results[f'aug_prob_{aug_prob}'] = run_training(
            output_dir=output_dir,
            run_name=f"aug_prob_{aug_prob}_seed{base_args['seed']}",
            extra_args=[
                '--d-model', '192',
                '--n-layers', '4',
                '--learning-rate', '0.0002',
                '--augment',
                '--augment-prob', str(aug_prob),
            ],
            base_args=base_args,
            use_wandb=use_wandb,
        )

    return results


def run_curriculum_ablation(base_args: Dict, output_dir: str, use_wandb: bool = False) -> Dict:
    """Study curriculum learning impact."""
    results = {}

    # Without curriculum
    results['no_curriculum'] = run_training(
        output_dir=output_dir,
        run_name=f"curriculum_OFF_seed{base_args['seed']}",
        extra_args=[
            '--d-model', '192',
            '--n-layers', '4',
            '--learning-rate', '0.0002',
            '--augment',
        ],
        base_args=base_args,
        use_wandb=use_wandb,
    )

    # With curriculum
    results['with_curriculum'] = run_training(
        output_dir=output_dir,
        run_name=f"curriculum_ON_seed{base_args['seed']}",
        extra_args=[
            '--d-model', '192',
            '--n-layers', '4',
            '--learning-rate', '0.0002',
            '--augment',
            '--curriculum',
        ],
        base_args=base_args,
        use_wandb=use_wandb,
    )

    return results


def run_scheduled_sampling_ablation(base_args: Dict, output_dir: str, use_wandb: bool = False) -> Dict:
    """Study scheduled sampling impact."""
    results = {}

    # Without scheduled sampling
    results['no_scheduled_sampling'] = run_training(
        output_dir=output_dir,
        run_name=f"sched_samp_OFF_seed{base_args['seed']}",
        extra_args=[
            '--d-model', '192',
            '--n-layers', '4',
            '--learning-rate', '0.0002',
            '--augment',
        ],
        base_args=base_args,
        use_wandb=use_wandb,
    )

    # With scheduled sampling
    results['with_scheduled_sampling'] = run_training(
        output_dir=output_dir,
        run_name=f"sched_samp_ON_seed{base_args['seed']}",
        extra_args=[
            '--d-model', '192',
            '--n-layers', '4',
            '--learning-rate', '0.0002',
            '--augment',
            '--scheduled-sampling',
            '--teacher-forcing-end', '0.5',
        ],
        base_args=base_args,
        use_wandb=use_wandb,
    )

    return results


def generate_ablation_report(all_results: Dict, output_dir: str):
    """Generate comprehensive ablation report."""

    report_path = Path(output_dir) / 'ablation_summary.json'

    # Add metadata
    all_results['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'output_dir': output_dir,
    }

    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)

    for ablation_name, results in all_results.items():
        if ablation_name == 'metadata':
            continue

        print(f"\n{ablation_name.upper()}")
        print("-" * 40)

        if isinstance(results, dict):
            for config_name, config_results in results.items():
                if isinstance(config_results, dict):
                    token_acc = config_results.get('val_token_accuracy',
                                config_results.get('token_accuracy', 'N/A'))
                    if isinstance(token_acc, (int, float)):
                        token_acc = f"{token_acc*100:.2f}%"
                    print(f"  {config_name}: {token_acc}")

    print("\n" + "="*80)
    print(f"Full report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Run architecture ablations')
    parser.add_argument('--output-dir', type=str, default='outputs/ablations',
                        help='Output directory for ablation results')
    parser.add_argument('--split-dir', type=str, default='outputs/processed_v3',
                        help='Directory with train/val/test splits')
    parser.add_argument('--vocab-path', type=str, default='data/vocabulary_4digit_full.json',
                        help='Path to vocabulary file')
    parser.add_argument('--encoder-path', type=str, default='outputs/mm_dtae_lstm_v2/best_model.pt',
                        help='Path to pretrained encoder')
    parser.add_argument('--ablation', type=str, default='all',
                        choices=[
                            'cross_attention', 'positional', 'operation',
                            'model_size', 'dropout', 'learning_rate',
                            'augmentation', 'curriculum', 'scheduled_sampling',
                            'all'
                        ],
                        help='Which ablation to run')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max-epochs', type=int, default=80,
                        help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Log to Weights & Biases')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base arguments
    base_args = {
        'seed': args.seed,
        'split_dir': args.split_dir,
        'vocab_path': args.vocab_path,
        'encoder_path': args.encoder_path,
        'max_epochs': args.max_epochs,
        'patience': args.patience,
    }

    # Ablation functions
    ablation_funcs = {
        'cross_attention': ('Cross-Attention', run_cross_attention_ablation),
        'positional': ('Positional Encoding', run_positional_encoding_ablation),
        'operation': ('Operation Conditioning', run_operation_conditioning_ablation),
        'model_size': ('Model Size Scaling', run_model_size_scaling),
        'dropout': ('Dropout Sensitivity', run_dropout_sensitivity),
        'learning_rate': ('Learning Rate Sensitivity', run_learning_rate_sensitivity),
        'augmentation': ('Data Augmentation', run_augmentation_ablation),
        'curriculum': ('Curriculum Learning', run_curriculum_ablation),
        'scheduled_sampling': ('Scheduled Sampling', run_scheduled_sampling_ablation),
    }

    # Determine which ablations to run
    if args.ablation == 'all':
        ablations_to_run = list(ablation_funcs.keys())
    else:
        ablations_to_run = [args.ablation]

    all_results = {}

    for ablation_key in ablations_to_run:
        ablation_name, ablation_func = ablation_funcs[ablation_key]

        print(f"\n{'#'*60}")
        print(f"# Running {ablation_name} Ablation")
        print(f"{'#'*60}")

        ablation_output = str(output_dir / ablation_key)
        results = ablation_func(base_args, ablation_output, args.use_wandb)
        all_results[ablation_key] = results

    # Generate report
    generate_ablation_report(all_results, str(output_dir))


if __name__ == '__main__':
    main()
