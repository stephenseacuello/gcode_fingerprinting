#!/usr/bin/env python3
"""Standalone encoder training for operation classification.

Trains the EnhancedEncoder on the operation classification task only,
without any decoder overhead. Goal: maximize encoder op accuracy before
using the trained encoder in the full pipeline.

Supports:
- Data augmentation (noise, time warp, feature dropout)
- Multiple optimizers (adamw, adam, sgd)
- Cosine annealing with warm restarts
- Mixup augmentation
- Label smoothing
- Class-weighted loss for imbalanced classes
"""
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import EnhancedEncoder
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn


def apply_augmentation(sensor_data, args):
    """Apply sensor data augmentation."""
    if not args.augment:
        return sensor_data

    if torch.rand(1).item() > args.augment_prob:
        return sensor_data

    # Gaussian noise
    if args.noise_std > 0:
        noise = torch.randn_like(sensor_data) * args.noise_std
        sensor_data = sensor_data + noise

    # Feature dropout
    if args.feature_dropout > 0 and torch.rand(1).item() < 0.5:
        mask = torch.rand(sensor_data.shape[-1], device=sensor_data.device) > args.feature_dropout
        sensor_data = sensor_data * mask.float()

    # Time shift
    if args.time_shift > 0 and torch.rand(1).item() < 0.3:
        shift = torch.randint(-args.time_shift, args.time_shift + 1, (1,)).item()
        if shift != 0:
            sensor_data = torch.roll(sensor_data, shifts=shift, dims=1)

    # Random scaling
    if args.scale_range > 0 and torch.rand(1).item() < 0.3:
        scale = 1.0 + (torch.rand(1).item() * 2 - 1) * args.scale_range
        sensor_data = sensor_data * scale

    # Temporal masking (cutout)
    if args.cutout_len > 0 and torch.rand(1).item() < 0.3:
        T = sensor_data.size(1)
        mask_len = min(args.cutout_len, T // 4)
        start = torch.randint(0, max(1, T - mask_len), (1,)).item()
        sensor_data[:, start:start + mask_len, :] = 0

    return sensor_data


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation."""
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def evaluate(encoder, dataloader, device, criterion=None):
    """Evaluate encoder on a dataset split."""
    encoder.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    n_batches = 0

    per_class_correct = Counter()
    per_class_total = Counter()

    with torch.no_grad():
        for batch in dataloader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)

            out = encoder(sensor_data)
            logits = out['cls_logits']
            preds = logits.argmax(-1)

            correct += (preds == operations).sum().item()
            total += operations.size(0)

            if criterion is not None:
                loss = criterion(logits, operations)
                total_loss += loss.item()
                n_batches += 1

            # Per-class accuracy
            for p, t in zip(preds.cpu().tolist(), operations.cpu().tolist()):
                per_class_total[t] += 1
                if p == t:
                    per_class_correct[t] += 1

    acc = correct / total if total > 0 else 0
    avg_loss = total_loss / n_batches if n_batches > 0 else 0

    per_class_acc = {}
    for cls in sorted(per_class_total.keys()):
        pc = per_class_correct[cls] / per_class_total[cls] if per_class_total[cls] > 0 else 0
        per_class_acc[cls] = pc

    return {
        'accuracy': acc,
        'loss': avg_loss,
        'per_class_accuracy': per_class_acc,
        'per_class_total': dict(per_class_total),
    }


def main():
    parser = argparse.ArgumentParser(description='Standalone Encoder Training')
    parser.add_argument('--split-dir', required=True)
    parser.add_argument('--vocab-path', required=True)
    parser.add_argument('--output-dir', required=True)

    # Architecture
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--n-operations', type=int, default=9)
    parser.add_argument('--n-scales', type=int, default=4)
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--use-multihead-pooling', action='store_true')
    parser.add_argument('--pooling-n-heads', type=int, default=4)
    parser.add_argument('--pooling-n-queries', type=int, default=8)

    # Regularization
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--use-class-weights', action='store_true')

    # Training
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=60)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--optimizer', choices=['adamw', 'adam', 'sgd'], default='adamw')
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)

    # Scheduler
    parser.add_argument('--scheduler', choices=['cosine_restarts', 'cosine', 'step'], default='cosine_restarts')
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--restart-period', type=int, default=40)
    parser.add_argument('--restart-mult', type=int, default=2)

    # Augmentation
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--augment-prob', type=float, default=0.5)
    parser.add_argument('--noise-std', type=float, default=0.02)
    parser.add_argument('--feature-dropout', type=float, default=0.05)
    parser.add_argument('--time-shift', type=int, default=2)
    parser.add_argument('--scale-range', type=float, default=0.02)
    parser.add_argument('--cutout-len', type=int, default=3)
    parser.add_argument('--mixup-alpha', type=float, default=0.0)

    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Redirect stdout to log file
    import io
    log_file = open(output_dir / 'train.log', 'w')

    class TeeOutput:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = TeeOutput(sys.__stdout__, log_file)
    sys.stderr = TeeOutput(sys.__stderr__, log_file)

    print(f"Device: {device}")
    print(f"Args: {vars(args)}")

    # Load data
    train_dataset = DecoderDatasetFromSplits(
        split_dir=args.split_dir, split='train', max_token_len=32,
    )
    val_dataset = DecoderDatasetFromSplits(
        split_dir=args.split_dir, split='val', max_token_len=32,
    )
    test_dataset = DecoderDatasetFromSplits(
        split_dir=args.split_dir, split='test', max_token_len=32,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=decoder_collate_fn, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=0,
    )

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Get input dim from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['sensor_features'].shape[-1]
    print(f"Sensor input dim: {input_dim}")

    # Count class distribution
    all_ops = []
    for batch in train_loader:
        all_ops.extend(batch['operation_type'].tolist())
    class_counts = Counter(all_ops)
    print(f"Class distribution (train): {dict(sorted(class_counts.items()))}")

    # Class weights for imbalanced data
    class_weights = None
    if args.use_class_weights:
        total_samples = sum(class_counts.values())
        n_classes = args.n_operations
        weights = []
        for c in range(n_classes):
            count = class_counts.get(c, 1)
            w = total_samples / (n_classes * count)
            weights.append(w)
        class_weights = torch.FloatTensor(weights).to(device)
        print(f"Class weights: {[f'{w:.2f}' for w in weights]}")

    # Build encoder
    encoder = EnhancedEncoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_operations=args.n_operations,
        use_multiscale=True,
        n_scales=args.n_scales,
        lstm_layers=args.lstm_layers,
        use_multihead_pooling=args.use_multihead_pooling,
        pooling_n_heads=args.pooling_n_heads,
        pooling_n_queries=args.pooling_n_queries,
        dropout=args.dropout,
    )
    encoder.to(device)

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {n_params:,}")

    # Loss
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    # Optimizer
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            encoder.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            encoder.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            encoder.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay, momentum=0.9,
        )

    # Scheduler
    if args.scheduler == 'cosine_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.restart_period, T_mult=args.restart_mult,
        )
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs,
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5,
        )

    # Warmup
    warmup_epochs = args.warmup_epochs

    # Training
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"\nStarting training...")
    print(f"{'='*60}")

    for epoch in range(args.max_epochs):
        epoch_start = time.time()
        encoder.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0
        n_batches = 0

        # Warmup LR
        if epoch < warmup_epochs:
            warmup_lr = args.learning_rate * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        for batch in train_loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)

            # Augmentation
            sensor_data = apply_augmentation(sensor_data, args)

            # Mixup
            use_mixup = args.mixup_alpha > 0 and torch.rand(1).item() < 0.5
            if use_mixup:
                sensor_data, ops_a, ops_b, lam = mixup_data(
                    sensor_data, operations, args.mixup_alpha
                )

            optimizer.zero_grad()
            out = encoder(sensor_data)
            logits = out['cls_logits']

            if use_mixup:
                loss = mixup_criterion(criterion, logits, ops_a, ops_b, lam)
            else:
                loss = criterion(logits, operations)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

            # Track accuracy (no mixup for accuracy)
            with torch.no_grad():
                preds = logits.argmax(-1)
                train_correct += (preds == operations).sum().item()
                train_total += operations.size(0)

        # Step scheduler (after warmup)
        if epoch >= warmup_epochs:
            scheduler.step()

        avg_train_loss = train_loss / n_batches
        train_acc = train_correct / train_total if train_total > 0 else 0

        # Validate
        val_result = evaluate(encoder, val_loader, device, criterion)
        val_acc = val_result['accuracy']
        val_loss = val_result['loss']

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']

        # Print summary
        print(f"\nEpoch {epoch+1}/{args.max_epochs}  ({epoch_time:.1f}s)")
        print(f"  Loss:     train={avg_train_loss:.4f}  val={val_loss:.4f}")
        print(f"  Op Acc:   train={train_acc:.2%}  val={val_acc:.2%}")
        print(f"  LR: {lr:.2e}")

        # Check best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0

            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'best_val_acc': best_val_acc,
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }, output_dir / 'best_model.pt')
            print(f"  NEW BEST: op_acc = {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={args.patience})")
                break

    # Final test evaluation
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val op acc: {best_val_acc:.2%}")

    # Load best checkpoint for test eval
    ckpt = torch.load(output_dir / 'best_model.pt', map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt['encoder_state_dict'])

    test_result = evaluate(encoder, test_loader, device, criterion)
    val_result = evaluate(encoder, val_loader, device, criterion)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS (from best checkpoint, epoch {best_epoch})")
    print(f"{'='*60}")
    print(f"  Val Op Acc:  {val_result['accuracy']:.2%}")
    print(f"  Test Op Acc: {test_result['accuracy']:.2%}")
    print(f"\n  Per-class test accuracy:")
    for cls in sorted(test_result['per_class_accuracy'].keys()):
        acc = test_result['per_class_accuracy'][cls]
        n = test_result['per_class_total'].get(cls, 0)
        print(f"    Class {cls}: {acc:.2%} ({n} samples)")

    # Save results
    results = {
        'best_epoch': best_epoch,
        'best_val_op_acc': best_val_acc,
        'val_op_acc': val_result['accuracy'],
        'test_op_acc': test_result['accuracy'],
        'test_per_class_accuracy': {str(k): v for k, v in test_result['per_class_accuracy'].items()},
        'test_per_class_total': {str(k): v for k, v in test_result['per_class_total'].items()},
    }
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {output_dir / 'test_results.json'}")


if __name__ == '__main__':
    main()
