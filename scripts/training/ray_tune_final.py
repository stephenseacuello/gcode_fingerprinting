#!/usr/bin/env python3
"""
Ray Tune Final Comprehensive Sweep for G-code Fingerprinting.

Explores:
- Architecture (wide/shallow vs narrow/deep)
- Multiple optimizers (AdamW, Adam, SGD, RAdam, NAdam)
- Multiple schedulers (cosine, cosine_restarts, linear, step, one_cycle)
- Data augmentation and noise injection

Usage:
    # Lambda cluster (8 GPUs)
    python scripts/training/ray_tune_final.py \
        --config configs/lambda_sweeps/final_comprehensive_sweep.yaml \
        --num-samples 300 --max-concurrent 8

    # Resume from checkpoint
    python scripts/training/ray_tune_final.py --resume --experiment-name gcode-final-sweep
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from functools import partial

import yaml
import torch
import torch.nn as nn
import numpy as np

# Ray imports are deferred to main() so the script can run without Ray
# for single-trial standalone mode (--no-ray flag).

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = str(PROJECT_ROOT / 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


def load_config(config_path: str) -> dict:
    """Load sweep configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_search_space(config: dict, use_ray: bool = True) -> dict:
    """Convert config search space to Ray Tune format or plain values."""
    import random
    space = {}
    for param, spec in config.get('search_space', {}).items():
        if use_ray:
            from ray import tune
            if spec['type'] == 'choice':
                space[param] = tune.choice(spec['values'])
            elif spec['type'] == 'uniform':
                space[param] = tune.uniform(float(spec['min']), float(spec['max']))
            elif spec['type'] == 'loguniform':
                space[param] = tune.loguniform(float(spec['min']), float(spec['max']))
            elif spec['type'] == 'randint':
                space[param] = tune.randint(int(spec['min']), int(spec['max']))
        else:
            # Without Ray: pick first/default value
            if spec['type'] == 'choice':
                space[param] = spec['values'][0]
            elif spec['type'] == 'uniform':
                space[param] = float(spec['min'])
            elif spec['type'] == 'loguniform':
                space[param] = float(spec['min'])
            elif spec['type'] == 'randint':
                space[param] = int(spec['min'])
    return space


def get_optimizer(name: str, params, lr: float, weight_decay: float, **kwargs):
    """Create optimizer by name."""
    from torch.optim import AdamW, Adam, SGD

    if name == 'adamw':
        return AdamW(
            params, lr=lr, weight_decay=weight_decay,
            betas=(kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999))
        )
    elif name == 'adam':
        return Adam(
            params, lr=lr, weight_decay=weight_decay,
            betas=(kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999))
        )
    elif name == 'sgd':
        return SGD(
            params, lr=lr, weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9), nesterov=True
        )
    elif name == 'radam':
        try:
            from torch.optim import RAdam
            return RAdam(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            # Fallback to AdamW
            return AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == 'nadam':
        try:
            from torch.optim import NAdam
            return NAdam(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            return AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        return AdamW(params, lr=lr, weight_decay=weight_decay)


def get_scheduler(name: str, optimizer, config: dict, num_steps_per_epoch: int):
    """Create scheduler by name."""
    from torch.optim.lr_scheduler import (
        CosineAnnealingWarmRestarts,
        CosineAnnealingLR,
        StepLR,
        ExponentialLR,
        OneCycleLR,
        LambdaLR,
    )

    max_epochs = config.get('max_epochs', 300)

    if name == 'cosine_restarts':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get('restart_period', 25),
            T_mult=1,
            eta_min=1e-6,
        )
    elif name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    elif name == 'step':
        return StepLR(
            optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.3),
        )
    elif name == 'exponential':
        return ExponentialLR(optimizer, gamma=config.get('gamma', 0.95))
    elif name == 'one_cycle':
        return OneCycleLR(
            optimizer,
            max_lr=config.get('learning_rate', 1e-3),
            epochs=max_epochs,
            steps_per_epoch=num_steps_per_epoch,
        )
    elif name == 'linear_warmup':
        # Linear warmup then constant
        warmup_epochs = config.get('warmup_epochs', 10)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0

        return LambdaLR(optimizer, lr_lambda)
    else:
        return CosineAnnealingWarmRestarts(optimizer, T_0=25)


def apply_augmentation(sensor_data: torch.Tensor, config: dict) -> torch.Tensor:
    """Apply data augmentation to sensor data."""
    if not config.get('use_augmentation', False):
        return sensor_data

    augment_prob = config.get('augment_prob', 0.3)
    noise_std = config.get('noise_std', 0.01)
    mixup_alpha = config.get('mixup_alpha', 0.0)

    batch_size = sensor_data.size(0)

    # Noise injection
    if noise_std > 0 and torch.rand(1).item() < augment_prob:
        noise = torch.randn_like(sensor_data) * noise_std
        sensor_data = sensor_data + noise

    # Mixup (blend with another sample in the batch)
    if mixup_alpha > 0 and batch_size > 1 and torch.rand(1).item() < augment_prob:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        rand_idx = torch.randperm(batch_size)
        sensor_data = lam * sensor_data + (1 - lam) * sensor_data[rand_idx]

    # Time masking (zero out random time segments)
    if torch.rand(1).item() < augment_prob * 0.5:
        T = sensor_data.size(1)
        mask_len = int(T * 0.1)  # Mask 10% of time
        start = torch.randint(0, max(1, T - mask_len), (1,)).item()
        sensor_data[:, start:start + mask_len, :] = 0

    return sensor_data


def build_modality_indices(metadata_path: str):
    """Build per-modality feature index groups from metadata.json."""
    with open(metadata_path) as f:
        metadata = json.load(f)
    columns = metadata['continuous_columns']

    sensor_patterns = {
        'accelerometer': ['Ax', 'Ay', 'Az'],
        'gyroscope': ['Gx', 'Gy', 'Gz'],
        'magnetometer': ['Mx', 'My', 'Mz'],
        'environmental': ['Pressure', 'Temperature', 'Proximity'],
        'color': ['ColorR', 'ColorG', 'ColorB', 'ColorA'],
        'rms': ['RMS'],
    }

    groups = {name: [] for name in sensor_patterns}
    groups['machine'] = []

    for idx, col in enumerate(columns):
        matched = False
        if '.' in col:
            _, feat = col.rsplit('.', 1)
            for group_name, patterns in sensor_patterns.items():
                if feat in patterns:
                    groups[group_name].append(idx)
                    matched = True
                    break
        if not matched:
            groups['machine'].append(idx)

    group_names = list(groups.keys())
    sensor_dims = [len(groups[name]) for name in group_names]
    return group_names, groups, sensor_dims


def split_modalities(sensor_data, group_indices):
    """Split [B, T, D] tensor into list of [B, T, Cm] per modality."""
    return [sensor_data[:, :, indices] for indices in group_indices]


def train_model(config: dict, data_dir: str, vocab_path: str, encoder_path: str = None,
                freeze_encoder: bool = False, encoder_lr_scale: float = 0.1):
    """Training function for Ray Tune with full optimizer/scheduler support."""
    from torch.utils.data import DataLoader
    from miracle.dataset.decoder_dataset import (
        DecoderDatasetFromSplits,
        decoder_collate_fn,
    )
    from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
    from miracle.model.model import MM_DTAE_LSTM, ModelConfig
    from miracle.training.metrics import compute_comprehensive_metrics

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set seed
    seed = config.get('seed', 123)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load vocab
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']

    # Build modality indices from metadata
    metadata_path = Path(data_dir) / 'metadata.json'
    group_names, groups, sensor_dims = build_modality_indices(str(metadata_path))
    group_indices = [groups[name] for name in group_names]

    max_seq_len = config.get('max_seq_len', 32)

    # Load datasets
    train_dataset = DecoderDatasetFromSplits(
        split_dir=data_dir, split='train', max_token_len=max_seq_len,
    )
    val_dataset = DecoderDatasetFromSplits(
        split_dir=data_dir, split='val', max_token_len=max_seq_len,
    )

    batch_size = config.get('batch_size', 16)
    # Python 3.14+ changed multiprocessing start method; datasets with open file
    # handles (NpzFile) can't be pickled for forkserver. Use 0 workers on
    # non-Linux or Python 3.14+.
    import platform
    num_workers = 4
    if platform.system() == 'Windows':
        num_workers = 0
    elif sys.version_info >= (3, 14):
        num_workers = 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=decoder_collate_fn, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=num_workers, pin_memory=True
    )

    # Load test set for periodic evaluation
    test_dataset = DecoderDatasetFromSplits(
        split_dir=data_dir, split='test', max_token_len=max_seq_len,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=num_workers, pin_memory=True
    )

    # Load MM_DTAE_LSTM encoder from checkpoint
    encoder = None
    encoder_d_model = 256  # default
    if encoder_path and os.path.exists(encoder_path):
        ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
        enc_config = ckpt['config']
        if not isinstance(enc_config, ModelConfig):
            enc_config = ModelConfig(**{k: v for k, v in enc_config.items()
                                        if k in ModelConfig.__dataclass_fields__})
        encoder = MM_DTAE_LSTM(enc_config)
        # Resize head_cls to match checkpoint (model defaults to 5, checkpoint may have 9)
        sd = ckpt['model_state_dict']
        if 'head_cls.weight' in sd:
            n_cls = sd['head_cls.weight'].shape[0]
            if encoder.head_cls.out_features != n_cls:
                encoder.head_cls = nn.Linear(enc_config.d_model, n_cls)
        encoder.load_state_dict(sd, strict=False)
        encoder_d_model = enc_config.d_model
    else:
        # No encoder path — create fresh MM_DTAE_LSTM
        enc_config = ModelConfig(sensor_dims=sensor_dims)
        encoder = MM_DTAE_LSTM(enc_config)
        encoder_d_model = enc_config.d_model

    encoder.to(device)
    if freeze_encoder:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
    else:
        encoder.train()

    # Create decoder
    d_model = int(config['d_model'])
    n_heads = int(config['n_heads'])

    if d_model % n_heads != 0:
        for nh in [32, 24, 16, 8]:
            if d_model % nh == 0:
                n_heads = nh
                break

    ffn_multiplier = config.get('ffn_multiplier', 4)
    d_ff = d_model * ffn_multiplier

    decoder = SensorMultiHeadDecoder(
        sensor_dim=encoder_d_model,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=int(config['n_layers']),
        vocab_size=len(vocab),
        n_operations=9,
        n_types=4,
        n_commands=6,
        n_param_types=10,
        d_ff=d_ff,
        dropout=config['dropout'],
        max_seq_len=max_seq_len,
        n_decimal_digits=4,
        max_int_digits=2,
        embed_dropout=config['embed_dropout'],
        drop_path_rate=config['drop_path_rate'],
        use_sensor_prior=config.get('use_sensor_prior', True),
        sensor_prior_weight=config.get('sensor_prior_weight', 0.5),
    )

    if hasattr(decoder, 'set_vocab'):
        decoder.set_vocab(vocab)

    decoder.to(device)

    # Get optimizer
    optimizer_name = config.get('optimizer', 'adamw')
    lr = config['learning_rate']
    if freeze_encoder:
        opt_params = decoder.parameters()
    else:
        opt_params = [
            {'params': decoder.parameters(), 'lr': lr},
            {'params': encoder.parameters(), 'lr': lr * encoder_lr_scale},
        ]
    optimizer = get_optimizer(
        optimizer_name,
        opt_params,
        lr=lr,
        weight_decay=config['weight_decay'],
        beta1=config.get('beta1', 0.9),
        beta2=config.get('beta2', 0.999),
        momentum=config.get('momentum', 0.9),
    )

    # Get scheduler
    scheduler_name = config.get('scheduler', 'cosine_restarts')
    scheduler = get_scheduler(scheduler_name, optimizer, config, len(train_loader))
    use_step_per_batch = scheduler_name == 'one_cycle'

    # Loss
    criterion = nn.CrossEntropyLoss(
        ignore_index=0,
        label_smoothing=config.get('label_smoothing', 0.1),
    )

    # Training
    max_epochs = config.get('max_epochs', 300)
    patience = config.get('patience', 60)
    warmup_epochs = config.get('warmup_epochs', 10)
    best_val_acc = 0.0
    best_test_acc = 0.0
    patience_counter = 0
    test_eval_interval = 25  # Evaluate test set every 25 epochs

    all_model_params = list(decoder.parameters()) + list(encoder.parameters())
    n_params = sum(p.numel() for p in all_model_params)
    n_trainable = sum(p.numel() for p in all_model_params if p.requires_grad)

    for epoch in range(max_epochs):
        epoch_start = time.time()

        # Training
        decoder.train()
        if not freeze_encoder:
            encoder.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            # Apply augmentation
            sensor_data = apply_augmentation(sensor_data, config)

            # Split into per-modality tensors for MM_DTAE_LSTM
            mods = split_modalities(sensor_data, group_indices)
            sensor_lengths = torch.full((sensor_data.size(0),), sensor_data.size(1),
                                        dtype=torch.long, device=device)

            if freeze_encoder:
                with torch.no_grad():
                    encoder_out = encoder(mods, sensor_lengths)
                    sensor_memory = encoder_out['memory']
            else:
                encoder_out = encoder(mods, sensor_lengths)
                sensor_memory = encoder_out['memory']

            optimizer.zero_grad()
            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=sensor_memory,
                operation_type=operations,
            )

            logits = outputs['legacy_logits']
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1)
            )

            # Encoder operation classification loss
            if not freeze_encoder and 'cls' in encoder_out:
                op_loss_weight = config.get('op_loss_weight', 1.0)
                op_cls_loss = nn.CrossEntropyLoss()(encoder_out['cls'], operations)
                loss = loss + op_loss_weight * op_cls_loss

            loss.backward()
            grad_clip = config.get('grad_clip', 1.0)
            all_params = list(decoder.parameters())
            if not freeze_encoder:
                all_params += list(encoder.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
            optimizer.step()

            if use_step_per_batch:
                scheduler.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = target_tokens != 0
            train_correct += ((preds == target_tokens) & mask).sum().item()
            train_total += mask.sum().item()

        # Warmup (manual for non-one_cycle schedulers)
        if not use_step_per_batch:
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
                sensor_data = batch['sensor_features'].to(device)
                operations = batch['operation_type'].to(device)
                input_tokens = batch['input_tokens'].to(device)
                target_tokens = batch['target_tokens'].to(device)

                mods = split_modalities(sensor_data, group_indices)
                sensor_lengths = torch.full((sensor_data.size(0),), sensor_data.size(1),
                                            dtype=torch.long, device=device)
                encoder_out = encoder(mods, sensor_lengths)
                sensor_memory = encoder_out['memory']
                outputs = decoder(
                    tokens=input_tokens,
                    sensor_embeddings=sensor_memory,
                    operation_type=operations,
                )

                logits = outputs['legacy_logits']
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1)
                )

                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                mask = target_tokens != 0
                val_correct += ((preds == target_tokens) & mask).sum().item()
                val_total += mask.sum().item()

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        gpu_memory_mb = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()

        # Evaluate on test set periodically or when val improves
        test_acc = best_test_acc  # Default to last known test_acc
        test_metrics = {}
        run_test_eval = (epoch % test_eval_interval == 0) or (val_acc > best_val_acc)
        if run_test_eval:
            test_correct = 0
            test_total = 0
            all_test_preds = []
            all_test_targets = []
            all_test_pred_seqs = []
            all_test_target_seqs = []
            encoder.eval()
            with torch.no_grad():
                for batch in test_loader:
                    sensor_data = batch['sensor_features'].to(device)
                    operations = batch['operation_type'].to(device)
                    input_tokens = batch['input_tokens'].to(device)
                    target_tokens = batch['target_tokens'].to(device)

                    mods = split_modalities(sensor_data, group_indices)
                    sensor_lengths = torch.full((sensor_data.size(0),), sensor_data.size(1),
                                                dtype=torch.long, device=device)
                    encoder_out = encoder(mods, sensor_lengths)
                    sensor_memory = encoder_out['memory']
                    outputs = decoder(
                        tokens=input_tokens,
                        sensor_embeddings=sensor_memory,
                        operation_type=operations,
                    )

                    logits = outputs['legacy_logits']
                    preds = logits.argmax(dim=-1)
                    mask = target_tokens != 0
                    test_correct += ((preds == target_tokens) & mask).sum().item()
                    test_total += mask.sum().item()

                    all_test_preds.append(preds)
                    all_test_targets.append(target_tokens)
                    # Collect sequences for BLEU/ED
                    for i in range(preds.size(0)):
                        m = mask[i]
                        all_test_pred_seqs.append(preds[i][m].cpu().tolist())
                        all_test_target_seqs.append(target_tokens[i][m].cpu().tolist())

            if not freeze_encoder:
                encoder.train()

            test_acc = test_correct / test_total if test_total > 0 else 0
            if test_acc > best_test_acc:
                best_test_acc = test_acc

            # Compute comprehensive metrics
            all_test_preds_t = torch.cat(all_test_preds, dim=0)
            all_test_targets_t = torch.cat(all_test_targets, dim=0)
            test_metrics = compute_comprehensive_metrics(
                all_test_preds_t, all_test_targets_t,
                all_test_pred_seqs, all_test_target_seqs,
                pad_token=0,
            )

        # Report metrics
        report_dict = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_token_acc': train_acc,
            'val_token_acc': val_acc,
            'test_token_acc': test_acc,
            'best_test_acc': best_test_acc,
            'learning_rate': current_lr,
            'best_val_acc': best_val_acc,
            'overfitting_gap': train_acc - val_acc,
            'patience_counter': patience_counter,
            'epoch_time_sec': epoch_time,
            'samples_per_sec': len(train_dataset) / epoch_time if epoch_time > 0 else 0,
            'gpu_memory_mb': gpu_memory_mb,
            'n_params_millions': n_params / 1e6,
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': int(config['n_layers']),
            'ffn_multiplier': ffn_multiplier,
            'batch_size': batch_size,
            'optimizer': optimizer_name,
            'scheduler': scheduler_name,
            'use_augmentation': config.get('use_augmentation', False),
            'noise_std': config.get('noise_std', 0),
            'use_sensor_prior': config.get('use_sensor_prior', True),
            'freeze_encoder': freeze_encoder,
        }
        # Add comprehensive test metrics when available
        for k, v in test_metrics.items():
            report_dict[f'test_{k}'] = v

        # Report to Ray Tune if running under Ray, otherwise print
        try:
            from ray.air import session
            session.report(report_dict)
        except (ImportError, RuntimeError):
            print(f"  Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} | "
                  f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f} | "
                  f"best_test={best_test_acc:.4f} lr={current_lr:.6f} patience={patience_counter}/{patience}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break


def main():
    parser = argparse.ArgumentParser(description='Ray Tune Final Comprehensive Sweep')
    parser.add_argument('--config', default='configs/lambda_sweeps/final_comprehensive_sweep.yaml')
    parser.add_argument('--split-dir', '--data-dir', dest='data_dir',
                        default='outputs/jan30/encoder_pipeline/data')
    parser.add_argument('--vocab-path', default='data/vocabulary_4digit_full.json')
    parser.add_argument('--encoder-path',
                        default='outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt',
                        help='Path to pretrained MM_DTAE_LSTM encoder')
    parser.add_argument('--freeze-encoder', action='store_true',
                        help='Freeze encoder weights (default: train jointly)')
    parser.add_argument('--encoder-lr-scale', type=float, default=0.1,
                        help='Encoder LR multiplier relative to decoder LR (default: 0.1)')
    parser.add_argument('--num-samples', type=int, default=300)
    parser.add_argument('--max-concurrent', type=int, default=8)
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Override max_epochs from config')
    parser.add_argument('--patience', type=int, default=None,
                        help='Override patience from config')
    parser.add_argument('--experiment-name', default='gcode-final-sweep')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-dir', '--local-dir', dest='local_dir',
                        default=str(PROJECT_ROOT / 'outputs' / 'ray_sweep_final'))
    parser.add_argument('--wandb-project', default='gcode-jan30-sweep')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--no-ray', action='store_true',
                        help='Run a single trial without Ray (standalone mode)')
    args = parser.parse_args()

    config = load_config(args.config)
    search_space = build_search_space(config, use_ray=not args.no_ray)

    for key, value in config.get('fixed', {}).items():
        search_space[key] = value

    # CLI overrides for max_epochs and patience
    if args.max_epochs is not None:
        search_space['max_epochs'] = args.max_epochs
    if args.patience is not None:
        search_space['patience'] = args.patience

    encoder_path = str(PROJECT_ROOT / args.encoder_path) if args.encoder_path else None

    if args.no_ray:
        # Standalone mode: run a single trial directly without Ray
        print("Running in standalone mode (no Ray)")

        print(f"Config: { {k: v for k, v in search_space.items() if k not in ('seed',)} }")

        train_model(
            config=search_space,
            data_dir=str(PROJECT_ROOT / args.data_dir),
            vocab_path=str(PROJECT_ROOT / args.vocab_path),
            encoder_path=encoder_path,
            freeze_encoder=args.freeze_encoder,
            encoder_lr_scale=args.encoder_lr_scale,
        )
        print("\nStandalone training complete.")
        return

    # --- Ray Tune mode ---
    import ray
    from ray import tune as ray_tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
    from ray.air import RunConfig, CheckpointConfig
    from ray.air.integrations.wandb import WandbLoggerCallback

    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "working_dir": str(PROJECT_ROOT),
            "excludes": [
                ".git", "outputs", "wandb", "ray_results", "*.pt", "*.npz",
                "htmlcov", "figures", "sweeps", "archive", "node_red",
            ],
            "env_vars": {"PYTHONPATH": SRC_PATH}
        }
    )

    asha_max_t = args.max_epochs if args.max_epochs else config['fixed'].get('max_epochs', 300)
    scheduler = ASHAScheduler(
        metric='val_token_acc',
        mode='max',
        max_t=asha_max_t,
        grace_period=25,
        reduction_factor=3,
    )

    search_alg = OptunaSearch(metric='val_token_acc', mode='max')

    callbacks = []
    if not args.no_wandb:
        callbacks.append(
            WandbLoggerCallback(
                project=args.wandb_project,
                group=args.experiment_name,
                log_config=True,
            )
        )

    reporter = CLIReporter(
        metric_columns=['train_loss', 'val_loss', 'val_token_acc', 'optimizer', 'scheduler'],
        max_report_frequency=60,
    )

    train_fn = partial(
        train_model,
        data_dir=str(PROJECT_ROOT / args.data_dir),
        vocab_path=str(PROJECT_ROOT / args.vocab_path),
        encoder_path=encoder_path,
        freeze_encoder=args.freeze_encoder,
        encoder_lr_scale=args.encoder_lr_scale,
    )

    tuner = ray_tune.Tuner(
        ray_tune.with_resources(train_fn, resources={'cpu': 4, 'gpu': 1}),
        param_space=search_space,
        tune_config=ray_tune.TuneConfig(
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
            checkpoint_config=CheckpointConfig(num_to_keep=3),
        ),
    )

    if args.resume:
        tuner = ray_tune.Tuner.restore(
            os.path.join(args.local_dir, args.experiment_name),
            trainable=train_fn,
            resume_errored=True,
        )

    results = tuner.fit()

    best_result = results.get_best_result(metric='val_token_acc', mode='max')
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    m = best_result.metrics
    c = best_result.config
    print(f"  Token Acc:      train={m.get('train_token_acc', 0):.2%}  val={m['val_token_acc']:.2%}  test={m.get('best_test_acc', 0):.2%}")
    print(f"  Optimizer:      {c.get('optimizer', 'adamw')}")
    print(f"  Scheduler:      {c.get('scheduler', 'cosine_restarts')}")
    print(f"  Architecture:   d_model={c['d_model']}, n_layers={c['n_layers']}, n_heads={c['n_heads']}")
    print(f"  Config: {c}")

    best_config_path = Path(args.local_dir) / 'best_config.json'
    with open(best_config_path, 'w') as f:
        json.dump({
            'metrics': best_result.metrics,
            'config': best_result.config,
        }, f, indent=2)
    print(f"\nBest config saved to: {best_config_path}")


if __name__ == '__main__':
    main()
