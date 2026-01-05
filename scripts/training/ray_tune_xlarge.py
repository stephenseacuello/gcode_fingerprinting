#!/usr/bin/env python3
"""
Ray Tune hyperparameter sweep for XLarge G-code decoder models.
Designed for Lambda Labs GPU cluster with A100/H100 GPUs.

Usage:
    # Local test (1 GPU)
    python scripts/training/ray_tune_xlarge.py --num-samples 2 --max-concurrent 1

    # Lambda cluster (4 GPUs)
    python scripts/training/ray_tune_xlarge.py --num-samples 50 --max-concurrent 4

    # Resume from checkpoint
    python scripts/training/ray_tune_xlarge.py --resume --experiment-name gcode-xlarge-sweep
"""

import os
import sys
import json
import argparse
from pathlib import Path
from functools import partial

import yaml
import torch
import numpy as np

# Ray imports
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air import RunConfig, CheckpointConfig
from ray.air.integrations.wandb import WandbLoggerCallback

# Project root for runtime env
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = str(PROJECT_ROOT / 'src')


def load_config(config_path: str) -> dict:
    """Load sweep configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_search_space(config: dict) -> dict:
    """Convert config search space to Ray Tune format."""
    space = {}
    for param, spec in config.get('search_space', {}).items():
        if spec['type'] == 'choice':
            space[param] = tune.choice(spec['values'])
        elif spec['type'] == 'uniform':
            space[param] = tune.uniform(float(spec['min']), float(spec['max']))
        elif spec['type'] == 'loguniform':
            space[param] = tune.loguniform(float(spec['min']), float(spec['max']))
        elif spec['type'] == 'randint':
            space[param] = tune.randint(int(spec['min']), int(spec['max']))
    return space


def train_model(config: dict, data_dir: str, vocab_path: str, encoder_path: str):
    """
    Training function for Ray Tune.
    Each trial runs this function with different hyperparameters.
    """
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    from torch.utils.data import DataLoader

    # Import miracle modules (available via runtime_env)
    from miracle.dataset.decoder_dataset import (
        DecoderDatasetFromSplits,
        decoder_collate_fn,
    )
    from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
    from miracle.model.model import EnhancedEncoder

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set seed
    seed = config.get('seed', 123)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load vocab
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']

    # Load datasets
    train_dataset = DecoderDatasetFromSplits(
        split_dir=data_dir,
        split='train',
        max_token_len=32,
        augment=config.get('augment', True),
        augment_prob=config.get('augment_prob', 0.25),
    )
    val_dataset = DecoderDatasetFromSplits(
        split_dir=data_dir,
        split='val',
        max_token_len=32,
        augment=False,
    )

    batch_size = config.get('batch_size', 16)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=decoder_collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=2, pin_memory=True
    )

    # Create encoder
    encoder = EnhancedEncoder(
        input_dim=155,
        hidden_dim=256,
        latent_dim=128,
        n_operations=9,
        use_multiscale=True,
        n_scales=4,
    )

    # Load pretrained encoder weights
    if os.path.exists(encoder_path):
        ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)

    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Create decoder with hyperparameters from config
    d_model = int(config['d_model'])
    n_heads = int(config['n_heads'])

    # Ensure d_model is divisible by n_heads
    if d_model % n_heads != 0:
        # Adjust n_heads to be compatible
        for nh in [32, 24, 16, 8]:
            if d_model % nh == 0:
                n_heads = nh
                break

    decoder = SensorMultiHeadDecoder(
        sensor_dim=128,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=int(config['n_layers']),
        vocab_size=len(vocab),
        n_operations=9,
        n_types=4,
        n_commands=6,
        n_param_types=10,
        dropout=config['dropout'],
        max_seq_len=32,
        n_decimal_digits=4,
        max_int_digits=2,
        embed_dropout=config['embed_dropout'],
        drop_path_rate=config['drop_path_rate'],
        use_sensor_prior=config.get('use_sensor_prior', True),
        sensor_prior_weight=config['sensor_prior_weight'],
    )

    if hasattr(decoder, 'set_vocab'):
        decoder.set_vocab(vocab)

    decoder.to(device)

    # Optimizer
    optimizer = AdamW(
        decoder.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    # Scheduler
    warmup_epochs = int(config['warmup_epochs'])
    restart_period = int(config['restart_period'])
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=restart_period,
        T_mult=1,
        eta_min=1e-6,
    )

    # Loss
    label_smoothing = config['label_smoothing']
    criterion = nn.CrossEntropyLoss(
        ignore_index=0,  # PAD token
        label_smoothing=label_smoothing,
    )

    # Training loop
    max_epochs = config.get('max_epochs', 150)
    best_val_acc = 0.0
    patience_counter = 0
    patience = config.get('patience', 40)

    for epoch in range(max_epochs):
        # Training
        decoder.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            sensors = batch['sensors'].to(device)
            operations = batch['operations'].to(device)
            targets = batch['targets'].to(device)

            # Encode sensors
            with torch.no_grad():
                sensor_features, _ = encoder(sensors, operations)

            # Forward pass
            optimizer.zero_grad()
            outputs = decoder(
                sensor_features=sensor_features,
                operation_ids=operations,
                target_tokens=targets[:, :-1],
            )

            # Loss
            logits = outputs['logits']
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1)
            )

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            # Accuracy
            preds = logits.argmax(dim=-1)
            mask = targets[:, 1:] != 0
            train_correct += ((preds == targets[:, 1:]) & mask).sum().item()
            train_total += mask.sum().item()

        # Warmup
        if epoch < warmup_epochs:
            warmup_lr = config['learning_rate'] * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
        else:
            scheduler.step()

        # Validation
        decoder.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                sensors = batch['sensors'].to(device)
                operations = batch['operations'].to(device)
                targets = batch['targets'].to(device)

                sensor_features, _ = encoder(sensors, operations)
                outputs = decoder(
                    sensor_features=sensor_features,
                    operation_ids=operations,
                    target_tokens=targets[:, :-1],
                )

                logits = outputs['logits']
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets[:, 1:].reshape(-1)
                )

                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                mask = targets[:, 1:] != 0
                val_correct += ((preds == targets[:, 1:]) & mask).sum().item()
                val_total += mask.sum().item()

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Report to Ray Tune
        tune.report(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            train_token_acc=train_acc,
            val_token_acc=val_acc,
        )

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break


def main():
    parser = argparse.ArgumentParser(description='Ray Tune XLarge Sweep')
    parser.add_argument('--config', default='configs/lambda_sweeps/xlarge_sweep_config.yaml')
    parser.add_argument('--data-dir', default='outputs/production/data_splits')
    parser.add_argument('--vocab-path', default='data/vocabulary_4digit_full.json')
    parser.add_argument('--encoder-path', default='outputs/production/encoder/best_model.pt')
    parser.add_argument('--num-samples', type=int, default=50)
    parser.add_argument('--max-concurrent', type=int, default=4)
    parser.add_argument('--experiment-name', default='gcode-xlarge-sweep')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--local-dir', default=str(Path(__file__).resolve().parent.parent.parent / 'ray_results'))
    parser.add_argument('--wandb-project', default='gcode-xlarge-sweep')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Build search space
    search_space = build_search_space(config)

    # Add fixed params to search space
    for key, value in config.get('fixed', {}).items():
        search_space[key] = value

    # Initialize Ray with runtime environment including src path
    # Exclude large directories to stay under 512MB limit
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "working_dir": str(PROJECT_ROOT),
            "excludes": [
                ".git",
                "outputs",
                "wandb",
                "ray_results",
                "*.pt",
                "*.npz",
                "htmlcov",
                "figures",
                "sweeps",
                "archive",
                "node_red",
                "gcode_fingerprinting",  # nested repo
            ],
            "env_vars": {"PYTHONPATH": SRC_PATH}
        }
    )

    # Scheduler (ASHA for early stopping)
    scheduler = ASHAScheduler(
        metric='val_token_acc',
        mode='max',
        max_t=config['fixed'].get('max_epochs', 150),
        grace_period=20,
        reduction_factor=3,
    )

    # Search algorithm (Optuna)
    search_alg = OptunaSearch(
        metric='val_token_acc',
        mode='max',
    )

    # Callbacks
    callbacks = []
    if not args.no_wandb:
        callbacks.append(
            WandbLoggerCallback(
                project=args.wandb_project,
                group=args.experiment_name,
                log_config=True,
            )
        )

    # Reporter
    reporter = CLIReporter(
        metric_columns=['train_loss', 'val_loss', 'val_token_acc'],
        max_report_frequency=60,
    )

    # Training function with fixed ABSOLUTE paths
    # Ray workers run in temp directories, so paths must be absolute
    train_fn = partial(
        train_model,
        data_dir=str(PROJECT_ROOT / args.data_dir),
        vocab_path=str(PROJECT_ROOT / args.vocab_path),
        encoder_path=str(PROJECT_ROOT / args.encoder_path),
    )

    # Run sweep
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn,
            resources={'cpu': 4, 'gpu': 1}
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent,
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=RunConfig(
            name=args.experiment_name,
            storage_path=args.local_dir,
            callbacks=callbacks,
            progress_reporter=reporter,
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,
            ),
        ),
    )

    if args.resume:
        tuner = tune.Tuner.restore(
            os.path.join(args.local_dir, args.experiment_name),
            trainable=train_fn,
            resume_errored=True,
        )

    results = tuner.fit()

    # Print best result
    best_result = results.get_best_result(metric='val_token_acc', mode='max')
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"Val Token Acc: {best_result.metrics['val_token_acc']:.4f}")
    print(f"Config: {best_result.config}")
    print(f"Path: {best_result.path}")

    # Save best config
    best_config_path = Path(args.local_dir) / 'best_config.json'
    with open(best_config_path, 'w') as f:
        json.dump({
            'metrics': best_result.metrics,
            'config': best_result.config,
        }, f, indent=2)
    print(f"\nBest config saved to: {best_config_path}")


if __name__ == '__main__':
    main()
