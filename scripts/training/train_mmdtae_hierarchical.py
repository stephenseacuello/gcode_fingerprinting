#!/usr/bin/env python3
"""Hierarchical MM_DTAE_LSTM training for operation classification.

Two-stage approach:
  Stage 1: 3-class base operation classifier (adaptive / face / pocket)
  Stage 2: 3 binary classifiers (normal vs damage) within each base op

Class mapping:
  adaptive group: c0 (adaptive), c1 (adaptive150025), c6 (damageadaptive)
  face group:     c2 (face),     c3 (face150025),     c7 (damageface)
  pocket group:   c4 (pocket),   c5 (pocket150025),   c8 (damagepocket)

Stage 1 labels: 0=adaptive, 1=face, 2=pocket
Stage 2 labels: 0=normal, 1=damage
"""
import sys
import json
import argparse
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig, make_pad_mask
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn

# Class mappings: original 9-class → base operation + damage flag
BASE_OP_MAP = {0: 0, 1: 0, 6: 0,   # adaptive group
               2: 1, 3: 1, 7: 1,   # face group
               4: 2, 5: 2, 8: 2}   # pocket group

DAMAGE_MAP = {0: 0, 1: 0, 6: 1,    # adaptive: normal/normal/damage
              2: 0, 3: 0, 7: 1,    # face: normal/normal/damage
              4: 0, 5: 0, 8: 1}    # pocket: normal/normal/damage

BASE_OP_NAMES = {0: 'adaptive', 1: 'face', 2: 'pocket'}
GROUP_CLASSES = {0: [0, 1, 6], 1: [2, 3, 7], 2: [4, 5, 8]}
DAMAGE_CLASSES = {0: 6, 1: 7, 2: 8}  # damage class ID per base op


def build_modality_indices(metadata_path: str):
    """Build column index lists for each modality group from metadata."""
    with open(metadata_path) as f:
        meta = json.load(f)
    columns = meta['continuous_columns']

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


def get_subset_indices(dataset, target_classes):
    """Get indices of samples belonging to target_classes."""
    indices = []
    for i in range(len(dataset)):
        op = dataset.operation_type[i].item()
        if op in target_classes:
            indices.append(i)
    return indices


def train_one_stage(model, train_loader, val_loader, test_loader,
                    group_indices, device, label_fn, n_classes,
                    args, stage_name, output_dir, log_fn):
    """Train a single model (either stage 1 or stage 2 binary)."""

    # Override classification head
    d_model = args.d_model
    model.head_cls = nn.Linear(d_model, n_classes)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_fn(f"\n  Model: {n_params:,} params, {n_classes} output classes")

    # Class weights from training data
    class_counts = Counter()
    for batch in train_loader:
        ops = batch['operation_type']
        for op in ops:
            class_counts[label_fn(op.item())] += 1

    total = sum(class_counts.values())
    weights = torch.zeros(n_classes)
    for c, count in class_counts.items():
        weights[c] = total / (n_classes * count)
    weights = weights / weights.sum() * n_classes
    log_fn(f"  Class distribution: {dict(sorted(class_counts.items()))}")
    log_fn(f"  Class weights: {[f'{w:.2f}' for w in weights.tolist()]}")

    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device), label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.restart_period, T_mult=args.restart_mult)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])

    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for batch in train_loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            B = operations.size(0)
            T = sensor_data.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)

            # Map labels
            labels = torch.tensor([label_fn(op.item()) for op in operations],
                                  dtype=torch.long, device=device)

            if args.augment and args.noise_std > 0:
                sensor_data = sensor_data + torch.randn_like(sensor_data) * args.noise_std

            mods = split_modalities(sensor_data, group_indices)
            out = model(mods, lengths, modality_dropout_p=args.modality_dropout if model.training else 0.0)

            loss = criterion(out['cls'], labels)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            preds = out['cls'].argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += B
            epoch_loss += loss.item() * B

        scheduler.step()
        train_acc = epoch_correct / epoch_total
        train_loss = epoch_loss / epoch_total
        elapsed = time.time() - t0

        # Validate
        val_acc, val_per_class = evaluate_stage(
            model, val_loader, group_indices, device, label_fn, n_classes)

        improved = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            improved = ' ** NEW BEST **'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'n_classes': n_classes,
                'stage_name': stage_name,
            }, output_dir / f'{stage_name}_best.pt')
        else:
            no_improve += 1

        lr = optimizer.param_groups[0]['lr']
        per_class_str = ' '.join([f"c{c}:{a:.0%}" for c, a in sorted(val_per_class.items())])
        log_fn(f"  Epoch {epoch:3d}/{args.max_epochs} | "
               f"loss={train_loss:.4f} train={train_acc:.2%} | "
               f"val={val_acc:.2%} [{per_class_str}] | lr={lr:.2e} | {elapsed:.1f}s{improved}")

        if no_improve >= args.patience:
            log_fn(f"  Early stopping at epoch {epoch}")
            break

    # Load best and evaluate test
    ckpt = torch.load(output_dir / f'{stage_name}_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_acc, test_per_class = evaluate_stage(
        model, test_loader, group_indices, device, label_fn, n_classes)

    log_fn(f"  Best epoch {best_epoch}: val={best_val_acc:.2%}, test={test_acc:.2%}")
    for c in sorted(test_per_class.keys()):
        log_fn(f"    Class {c}: {test_per_class[c]:.2%}")

    return model, test_acc, test_per_class


def evaluate_stage(model, loader, group_indices, device, label_fn, n_classes):
    """Evaluate a stage model."""
    model.eval()
    total_correct = 0
    total = 0
    per_class_correct = Counter()
    per_class_total = Counter()

    with torch.no_grad():
        for batch in loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            B = operations.size(0)
            T = sensor_data.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)

            labels = torch.tensor([label_fn(op.item()) for op in operations],
                                  dtype=torch.long, device=device)

            mods = split_modalities(sensor_data, group_indices)
            out = model(mods, lengths)
            preds = out['cls'].argmax(dim=-1)

            total_correct += (preds == labels).sum().item()
            total += B

            for i in range(B):
                c = labels[i].item()
                per_class_total[c] += 1
                if preds[i].item() == c:
                    per_class_correct[c] += 1

    acc = total_correct / total if total > 0 else 0
    per_class_acc = {}
    for c in sorted(per_class_total.keys()):
        per_class_acc[c] = per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0

    model.train()
    return acc, per_class_acc


def hierarchical_predict(base_model, damage_models, sensor_data, group_indices, device, soft_routing=False):
    """Run full hierarchical prediction: base op → damage detection.

    Args:
        soft_routing: If True, run ALL damage classifiers and weight by stage 1
            softmax probs instead of hard argmax routing.

    Returns predicted 9-class labels.
    """
    B = sensor_data.size(0)
    T = sensor_data.size(1)
    lengths = torch.full((B,), T, dtype=torch.long, device=device)
    mods = split_modalities(sensor_data, group_indices)

    # Stage 1: predict base operation
    base_out = base_model(mods, lengths)
    base_probs = torch.softmax(base_out['cls'], dim=-1)  # [B, 3]
    base_preds = base_probs.argmax(dim=-1)  # [B] in {0,1,2}

    final_preds = torch.zeros(B, dtype=torch.long, device=device)

    if soft_routing:
        # Run ALL 3 damage classifiers on ALL samples
        # If ANY classifier says "damage" with high confidence, predict damage for that group
        damage_scores = torch.zeros(B, 3, dtype=torch.float, device=device)

        for base_op in range(3):
            sub_mods = split_modalities(sensor_data, group_indices)
            damage_out = damage_models[base_op](sub_mods, lengths)
            damage_probs = torch.softmax(damage_out['cls'], dim=-1)  # [B, 2]
            damage_scores[:, base_op] = damage_probs[:, 1]  # P(damage)

        for i in range(B):
            # Check if any damage classifier fires with high confidence
            max_damage_score, max_damage_group = damage_scores[i].max(dim=0)
            if max_damage_score > 0.5:
                # Some classifier thinks it's damage — use that group
                final_preds[i] = DAMAGE_CLASSES[max_damage_group.item()]
            else:
                # No damage detected — use hard base op pred for normal class
                best_group = base_preds[i].item()
                normal_classes = [c for c in GROUP_CLASSES[best_group] if c != DAMAGE_CLASSES[best_group]]
                final_preds[i] = normal_classes[0]
    else:
        # Hard routing (original)
        for base_op in range(3):
            mask = base_preds == base_op
            if not mask.any():
                continue

            indices = mask.nonzero(as_tuple=True)[0]
            sub_sensor = sensor_data[indices]
            sub_T = sub_sensor.size(1)
            sub_lengths = torch.full((indices.size(0),), sub_T, dtype=torch.long, device=device)
            sub_mods = split_modalities(sub_sensor, group_indices)

            damage_out = damage_models[base_op](sub_mods, sub_lengths)
            damage_preds = damage_out['cls'].argmax(dim=-1)  # [N] in {0, 1}

            group_classes = GROUP_CLASSES[base_op]
            normal_classes = [c for c in group_classes if c != DAMAGE_CLASSES[base_op]]

            for j, idx in enumerate(indices):
                if damage_preds[j].item() == 1:
                    final_preds[idx] = DAMAGE_CLASSES[base_op]
                else:
                    final_preds[idx] = normal_classes[0]

    return final_preds, base_preds


def evaluate_hierarchical(base_model, damage_models, loader, group_indices, device, soft_routing=False):
    """Evaluate full hierarchical pipeline, reporting damage detection accuracy."""
    base_model.eval()
    for dm in damage_models.values():
        dm.eval()

    # Track per original-class accuracy
    per_class_correct = Counter()
    per_class_total = Counter()
    # Track damage detection accuracy per group
    damage_correct = Counter()
    damage_total = Counter()

    with torch.no_grad():
        for batch in loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            B = operations.size(0)

            final_preds, base_preds = hierarchical_predict(
                base_model, damage_models, sensor_data, group_indices, device,
                soft_routing=soft_routing)

            for i in range(B):
                true_op = operations[i].item()
                true_base = BASE_OP_MAP[true_op]
                true_damage = DAMAGE_MAP[true_op]
                pred_base = base_preds[i].item()

                # Track base op accuracy
                per_class_total[f'base_{true_base}'] += 1
                if pred_base == true_base:
                    per_class_correct[f'base_{true_base}'] += 1

                # Track damage detection (only if base op was correct)
                if pred_base == true_base:
                    damage_total[true_base] += 1
                    pred_damage = 1 if final_preds[i].item() == DAMAGE_CLASSES[true_base] else 0
                    if pred_damage == true_damage:
                        damage_correct[true_base] += 1

                # Track original 9-class (for damage classes only)
                if true_op in [6, 7, 8]:
                    per_class_total[f'orig_{true_op}'] += 1
                    if soft_routing:
                        # With soft routing, just check if final pred matches true damage class
                        if final_preds[i].item() == true_op:
                            per_class_correct[f'orig_{true_op}'] += 1
                    else:
                        pred_damage = 1 if final_preds[i].item() == DAMAGE_CLASSES[true_base] else 0
                        if pred_base == true_base and pred_damage == 1:
                            per_class_correct[f'orig_{true_op}'] += 1

    return per_class_correct, per_class_total, damage_correct, damage_total


def main():
    parser = argparse.ArgumentParser(description='Hierarchical MM_DTAE_LSTM training')
    parser.add_argument('--split-dir', required=True)
    parser.add_argument('--vocab-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default=None)

    # Architecture
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)

    # Training
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--modality-dropout', type=float, default=0.1)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--noise-std', type=float, default=0.01)

    # Scheduler
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--restart-period', type=int, default=30)
    parser.add_argument('--restart-mult', type=int, default=2)
    parser.add_argument('--soft-routing', action='store_true',
                        help='Use soft probability-weighted routing instead of hard argmax')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training, load existing models and re-evaluate')
    parser.add_argument('--stage1-damage-oversample', type=float, default=1.0,
                        help='Oversample factor for damage classes in stage 1 (e.g. 5.0 = 5x)')
    parser.add_argument('--stage1-weight-boost', type=float, default=1.0,
                        help='Extra class weight multiplier for damage samples in stage 1 loss')
    parser.add_argument('--finetune-stage1', type=str, default=None,
                        help='Path to pre-trained stage1 checkpoint to fine-tune (e.g. outputs/jan30/mmdtae_hierarchical_v1/stage1_base_best.pt)')
    parser.add_argument('--finetune-lr', type=float, default=1e-4,
                        help='Learning rate for stage 1 fine-tuning')
    parser.add_argument('--finetune-epochs', type=int, default=20,
                        help='Max epochs for stage 1 fine-tuning')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build modality indices
    metadata_path = Path(args.split_dir) / 'metadata.json'
    group_names, groups, sensor_dims = build_modality_indices(str(metadata_path))
    group_indices = [groups[name] for name in group_names]

    # Load full datasets
    max_seq_len = 32
    train_dataset = DecoderDatasetFromSplits(split_dir=args.split_dir, split='train', max_token_len=max_seq_len)
    val_dataset = DecoderDatasetFromSplits(split_dir=args.split_dir, split='val', max_token_len=max_seq_len)
    test_dataset = DecoderDatasetFromSplits(split_dir=args.split_dir, split='test', max_token_len=max_seq_len)

    log_path = output_dir / 'training_log.txt'
    # Clear log
    with open(log_path, 'w') as f:
        pass

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    log(f"{'='*70}")
    log(f"Hierarchical MM_DTAE_LSTM Op Classification")
    log(f"{'='*70}")
    log(f"Device: {device}")
    log(f"Dataset: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    log(f"Modalities: {list(zip(group_names, sensor_dims))}")

    def make_model(n_classes):
        config = ModelConfig(
            sensor_dims=sensor_dims,
            d_model=args.d_model,
            lstm_layers=args.lstm_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            gcode_vocab=128,
            future_len=8,
        )
        model = MM_DTAE_LSTM(config)
        model.head_cls = nn.Linear(args.d_model, n_classes)
        return model

    # Full-dataset loaders (needed for both training and eval-only)
    test_loader_full = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=decoder_collate_fn, num_workers=0)

    # ===================== EVAL-ONLY MODE =====================
    if args.eval_only:
        log(f"\n{'='*70}")
        log(f"EVAL-ONLY MODE — loading existing models from {output_dir}")
        log(f"Soft routing: {args.soft_routing}")
        log(f"{'='*70}")

        base_model = make_model(3)
        ckpt = torch.load(output_dir / 'stage1_base_best.pt', map_location=device, weights_only=True)
        base_model.load_state_dict(ckpt['model_state_dict'])
        base_model.to(device)

        damage_models = {}
        for base_op, op_name in BASE_OP_NAMES.items():
            dm = make_model(2)
            ckpt = torch.load(output_dir / f'stage2_{op_name}_best.pt', map_location=device, weights_only=True)
            dm.load_state_dict(ckpt['model_state_dict'])
            dm.to(device)
            damage_models[base_op] = dm

        per_class_correct, per_class_total, damage_correct, damage_total = \
            evaluate_hierarchical(base_model, damage_models, test_loader_full, group_indices, device,
                                  soft_routing=args.soft_routing)

        routing_label = " (SOFT ROUTING)" if args.soft_routing else " (HARD ROUTING)"
        log(f"\nBase operation accuracy (test){routing_label}:")
        for base_op in range(3):
            key = f'base_{base_op}'
            acc = per_class_correct[key] / per_class_total[key] if per_class_total[key] > 0 else 0
            log(f"  {BASE_OP_NAMES[base_op]}: {acc:.2%} ({per_class_correct[key]}/{per_class_total[key]})")

        log(f"\nDamage detection accuracy (test, given correct base op):")
        for base_op in range(3):
            acc = damage_correct[base_op] / damage_total[base_op] if damage_total[base_op] > 0 else 0
            log(f"  {BASE_OP_NAMES[base_op]}: {acc:.2%} ({damage_correct[base_op]}/{damage_total[base_op]})")

        log(f"\nDamage class recovery (the hard problem){routing_label}:")
        for orig_cls in [6, 7, 8]:
            key = f'orig_{orig_cls}'
            total = per_class_total.get(key, 0)
            correct = per_class_correct.get(key, 0)
            acc = correct / total if total > 0 else 0
            log(f"  Class {orig_cls}: {acc:.2%} ({correct}/{total})")

        return

    # ===================== STAGE 1: Base Operation =====================
    log(f"\n{'='*70}")
    log(f"STAGE 1: Base Operation Classification (3 classes)")
    log(f"  adaptive={[0,1,6]}, face={[2,3,7]}, pocket={[4,5,8]}")
    log(f"{'='*70}")

    # Stage 1 train loader — optionally oversample damage classes
    if args.stage1_damage_oversample > 1.0:
        damage_ops = {6, 7, 8}
        sample_weights = []
        for i in range(len(train_dataset)):
            op = train_dataset.operation_type[i].item()
            if op in damage_ops:
                sample_weights.append(args.stage1_damage_oversample)
            else:
                sample_weights.append(1.0)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
        train_loader_full = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                       collate_fn=decoder_collate_fn, num_workers=0, drop_last=True)
        log(f"  Stage 1 oversampling: damage classes at {args.stage1_damage_oversample}x")
    else:
        train_loader_full = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                       collate_fn=decoder_collate_fn, num_workers=0, drop_last=True)

    val_loader_full = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=decoder_collate_fn, num_workers=0)

    base_model = make_model(3)
    base_label_fn = lambda op: BASE_OP_MAP[op]

    if args.finetune_stage1:
        # Load pre-trained stage 1 and fine-tune
        log(f"\n  Loading pre-trained stage 1 from {args.finetune_stage1}")
        ckpt = torch.load(args.finetune_stage1, map_location=device, weights_only=True)
        base_model.load_state_dict(ckpt['model_state_dict'])
        base_model.to(device)

        # Fine-tune with lower LR, shorter training
        finetune_args = argparse.Namespace(**vars(args))
        finetune_args.learning_rate = args.finetune_lr
        finetune_args.max_epochs = args.finetune_epochs
        finetune_args.patience = max(10, args.finetune_epochs // 2)
        log(f"  Fine-tuning: lr={args.finetune_lr}, epochs={args.finetune_epochs}")

        base_model, base_test_acc, base_test_per_class = train_one_stage(
            base_model, train_loader_full, val_loader_full, test_loader_full,
            group_indices, device, base_label_fn, 3,
            finetune_args, 'stage1_base', output_dir, log)
    else:
        base_model, base_test_acc, base_test_per_class = train_one_stage(
            base_model, train_loader_full, val_loader_full, test_loader_full,
            group_indices, device, base_label_fn, 3,
            args, 'stage1_base', output_dir, log)

    # ===================== STAGE 2: Damage Detection =====================
    damage_models = {}
    damage_results = {}

    for base_op, op_name in BASE_OP_NAMES.items():
        log(f"\n{'='*70}")
        target_classes = GROUP_CLASSES[base_op]
        damage_cls = DAMAGE_CLASSES[base_op]
        normal_cls = [c for c in target_classes if c != damage_cls]
        log(f"STAGE 2: Damage detection for '{op_name}' (classes {target_classes})")
        log(f"  Normal: {normal_cls}, Damage: [{damage_cls}]")
        log(f"{'='*70}")

        # Create subset dataloaders for this base operation's classes
        train_idx = get_subset_indices(train_dataset, target_classes)
        val_idx = get_subset_indices(val_dataset, target_classes)
        test_idx = get_subset_indices(test_dataset, target_classes)

        log(f"  Subset sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        # Count damage vs normal
        train_damage = sum(1 for i in train_idx if train_dataset.operation_type[i].item() == damage_cls)
        log(f"  Damage samples in train: {train_damage}/{len(train_idx)} ({train_damage/len(train_idx):.1%})")

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)
        test_subset = Subset(test_dataset, test_idx)

        bs = min(args.batch_size, len(train_idx))
        train_sub_loader = DataLoader(train_subset, batch_size=bs, shuffle=True,
                                      collate_fn=decoder_collate_fn, num_workers=0, drop_last=True)
        val_sub_loader = DataLoader(val_subset, batch_size=bs, shuffle=False,
                                    collate_fn=decoder_collate_fn, num_workers=0)
        test_sub_loader = DataLoader(test_subset, batch_size=bs, shuffle=False,
                                     collate_fn=decoder_collate_fn, num_workers=0)

        damage_label_fn = lambda op, dc=damage_cls: 1 if op == dc else 0
        damage_model = make_model(2)

        damage_model, dmg_test_acc, dmg_test_per_class = train_one_stage(
            damage_model, train_sub_loader, val_sub_loader, test_sub_loader,
            group_indices, device, damage_label_fn, 2,
            args, f'stage2_{op_name}', output_dir, log)

        damage_models[base_op] = damage_model
        damage_results[op_name] = {
            'test_acc': dmg_test_acc,
            'test_per_class': dmg_test_per_class,
        }

    # ===================== COMBINED EVALUATION =====================
    log(f"\n{'='*70}")
    log(f"COMBINED HIERARCHICAL EVALUATION")
    log(f"{'='*70}")

    per_class_correct, per_class_total, damage_correct, damage_total = \
        evaluate_hierarchical(base_model, damage_models, test_loader_full, group_indices, device,
                              soft_routing=args.soft_routing)

    routing_label = " (SOFT ROUTING)" if args.soft_routing else ""
    log(f"\nBase operation accuracy (test){routing_label}:")
    for base_op in range(3):
        key = f'base_{base_op}'
        acc = per_class_correct[key] / per_class_total[key] if per_class_total[key] > 0 else 0
        log(f"  {BASE_OP_NAMES[base_op]}: {acc:.2%} ({per_class_correct[key]}/{per_class_total[key]})")

    log(f"\nDamage detection accuracy (test, given correct base op):")
    for base_op in range(3):
        acc = damage_correct[base_op] / damage_total[base_op] if damage_total[base_op] > 0 else 0
        log(f"  {BASE_OP_NAMES[base_op]}: {acc:.2%} ({damage_correct[base_op]}/{damage_total[base_op]})")

    log(f"\nDamage class recovery (the hard problem):")
    for orig_cls in [6, 7, 8]:
        key = f'orig_{orig_cls}'
        total = per_class_total.get(key, 0)
        correct = per_class_correct.get(key, 0)
        acc = correct / total if total > 0 else 0
        log(f"  Class {orig_cls}: {acc:.2%} ({correct}/{total})")

    # Save results
    results = {
        'base_test_acc': base_test_acc,
        'base_test_per_class': {str(k): v for k, v in base_test_per_class.items()},
        'damage_results': {k: {
            'test_acc': v['test_acc'],
            'test_per_class': {str(kk): vv for kk, vv in v['test_per_class'].items()},
        } for k, v in damage_results.items()},
        'damage_class_recovery': {
            str(c): per_class_correct.get(f'orig_{c}', 0) / per_class_total.get(f'orig_{c}', 1)
            for c in [6, 7, 8]
        },
        'args': vars(args),
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    log(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == '__main__':
    main()
