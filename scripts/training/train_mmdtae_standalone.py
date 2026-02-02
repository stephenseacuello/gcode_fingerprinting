#!/usr/bin/env python3
"""Standalone MM_DTAE_LSTM training for operation classification.

Trains the multimodal DTAE+LSTM encoder on op classification only.
Splits the 296 concatenated features into per-modality groups and
feeds them as separate tensors to MM_DTAE_LSTM.

Modality grouping (by sensor type across all 16 locations):
  - Accelerometer (Ax,Ay,Az): 3×16 = 48
  - Gyroscope (Gx,Gy,Gz): 3×16 = 48
  - Magnetometer (Mx,My,Mz): 3×16 = 48
  - Environmental (Pressure,Temperature,Proximity): 3×16 = 48
  - Color (ColorR,ColorG,ColorB,ColorA): 4×16 = 64
  - RMS: 1×16 = 16
  - Machine (coor,dist,feed,...): 24
Total: 296
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

from miracle.model.model import MM_DTAE_LSTM, ModelConfig, make_pad_mask
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma=0, equivalent to cross-entropy.
    Higher gamma focuses more on hard/misclassified examples.
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def build_modality_indices(metadata_path: str):
    """Build column index lists for each modality group from metadata."""
    with open(metadata_path) as f:
        meta = json.load(f)
    columns = meta['continuous_columns']

    # Define modality patterns (sensor_type -> substrings in column name)
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
            # sensor feature: e.g. "frame_b1.Ax"
            _, feat = col.rsplit('.', 1)
            for group_name, patterns in sensor_patterns.items():
                if feat in patterns:
                    groups[group_name].append(idx)
                    matched = True
                    break
        if not matched:
            groups['machine'].append(idx)

    # Print summary
    group_names = list(groups.keys())
    sensor_dims = []
    for name in group_names:
        dim = len(groups[name])
        sensor_dims.append(dim)
        print(f"  Modality '{name}': {dim} features")
    print(f"  Total: {sum(sensor_dims)} features, {len(group_names)} modalities")

    return group_names, groups, sensor_dims


def split_modalities(sensor_data, group_indices):
    """Split [B, T, D] tensor into list of [B, T, Cm] per modality."""
    mods = []
    for indices in group_indices:
        mods.append(sensor_data[:, :, indices])
    return mods


def evaluate(model, loader, group_indices, device, n_operations, compute_anomaly=False):
    """Evaluate model on a dataset split."""
    model.eval()
    total_correct = 0
    total = 0
    per_class_correct = Counter()
    per_class_total = Counter()
    anom_tp = anom_fp = anom_tn = anom_fn = 0

    with torch.no_grad():
        for batch in loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            B = operations.size(0)
            T = sensor_data.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)

            mods = split_modalities(sensor_data, group_indices)
            out = model(mods, lengths)
            preds = out['cls'].argmax(dim=-1)

            total_correct += (preds == operations).sum().item()
            total += B

            for i in range(B):
                c = operations[i].item()
                per_class_total[c] += 1
                if preds[i].item() == c:
                    per_class_correct[c] += 1

            if compute_anomaly:
                anom_labels = ((operations == 6) | (operations == 7) | (operations == 8))
                anom_preds = out['anom'].squeeze(-1) > 0  # logit > 0 = anomaly
                anom_tp += (anom_preds & anom_labels).sum().item()
                anom_fp += (anom_preds & ~anom_labels).sum().item()
                anom_tn += (~anom_preds & ~anom_labels).sum().item()
                anom_fn += (~anom_preds & anom_labels).sum().item()

    acc = total_correct / total if total > 0 else 0
    per_class_acc = {}
    for c in sorted(per_class_total.keys()):
        per_class_acc[c] = per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0

    anom_metrics = None
    if compute_anomaly:
        anom_prec = anom_tp / (anom_tp + anom_fp) if (anom_tp + anom_fp) > 0 else 0
        anom_rec = anom_tp / (anom_tp + anom_fn) if (anom_tp + anom_fn) > 0 else 0
        anom_f1 = 2 * anom_prec * anom_rec / (anom_prec + anom_rec) if (anom_prec + anom_rec) > 0 else 0
        anom_metrics = {'precision': anom_prec, 'recall': anom_rec, 'f1': anom_f1,
                        'tp': anom_tp, 'fp': anom_fp, 'tn': anom_tn, 'fn': anom_fn}

    model.train()
    return acc, per_class_acc, anom_metrics


def evaluate_per_mod_pipeline(model, loader, group_indices, device, active_mods, n_operations):
    """Evaluate combined pipeline: v1 cls + per-modality damage heads.

    For each sample:
    1. Get cls prediction (9-class)
    2. Get per-modality damage head predictions (4-class each)
    3. If any active modality's damage head predicts c6/c7/c8 with prob > threshold, override cls
    """
    model.eval()
    # Collect all predictions and labels
    all_cls_preds = []
    all_mod_probs = {m: [] for m in active_mods}  # modality → list of [B, 4] probs
    all_ops = []

    with torch.no_grad():
        for batch in loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            B = operations.size(0)
            T = sensor_data.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)

            mods = split_modalities(sensor_data, group_indices)
            out = model(mods, lengths)

            all_cls_preds.append(out['cls'].argmax(dim=-1).cpu())
            all_ops.append(operations.cpu())
            for m_idx in active_mods:
                probs = torch.softmax(out['per_mod_damage'][m_idx], dim=-1)
                all_mod_probs[m_idx].append(probs.cpu())

    all_cls_preds = torch.cat(all_cls_preds).numpy()
    all_ops = torch.cat(all_ops).numpy()
    for m_idx in active_mods:
        all_mod_probs[m_idx] = torch.cat(all_mod_probs[m_idx]).numpy()

    N = len(all_ops)
    per_class_total = Counter()
    for i in range(N):
        per_class_total[all_ops[i]] += 1

    # Strategy: best single modality per damage class
    # Machine(6)→c6, Gyro(1)→c7, Mag(2)→c8
    DAMAGE_MOD_MAP = {1: (6, 1), 2: (1, 2), 3: (2, 3)}  # 4class_label → (mod_idx, 4class_label)

    results = {}
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pc_correct = Counter()
        for i in range(N):
            true_op = all_ops[i]
            cls_pred = all_cls_preds[i]

            # Check per-modality damage heads
            best_damage = None
            best_prob = 0
            for damage_4c, (m_idx, _) in DAMAGE_MOD_MAP.items():
                if m_idx in active_mods:
                    prob = all_mod_probs[m_idx][i, damage_4c]
                    if prob > thresh and prob > best_prob:
                        best_prob = prob
                        best_damage = [None, 6, 7, 8][damage_4c]

            final_pred = best_damage if best_damage is not None else cls_pred

            if final_pred == true_op:
                pc_correct[true_op] += 1

        total = sum(pc_correct.values())
        c6 = pc_correct[6] / per_class_total[6] if per_class_total[6] > 0 else 0
        c7 = pc_correct[7] / per_class_total[7] if per_class_total[7] > 0 else 0
        c8 = pc_correct[8] / per_class_total[8] if per_class_total[8] > 0 else 0
        rest = sum(pc_correct[c] for c in range(6))
        rest_total = sum(per_class_total[c] for c in range(6))
        results[thresh] = {
            'overall': total / N, 'c0_c5': rest / rest_total if rest_total > 0 else 0,
            'c6': c6, 'c7': c7, 'c8': c8
        }

    model.train()
    return results


def main():
    parser = argparse.ArgumentParser(description='Train MM_DTAE_LSTM for operation classification')
    parser.add_argument('--split-dir', required=True)
    parser.add_argument('--vocab-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default=None)

    # Architecture
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--n-operations', type=int, default=9)

    # Training
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

    # Scheduler
    parser.add_argument('--scheduler', choices=['cosine', 'cosine_restarts', 'step'], default='cosine_restarts')
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--restart-period', type=int, default=30)
    parser.add_argument('--restart-mult', type=int, default=2)

    # Class weighting
    parser.add_argument('--class-weights', action='store_true', help='Use inverse frequency class weights')

    # Modality dropout (training regularization)
    parser.add_argument('--modality-dropout', type=float, default=0.1)

    # Cls head pooling strategy
    parser.add_argument('--cls-pooling', choices=['last', 'mean'], default='last',
                        help='Pooling for cls head: last (last timestep) or mean (temporal mean)')

    # Focal loss
    parser.add_argument('--focal-loss', action='store_true', help='Use focal loss instead of CE')
    parser.add_argument('--focal-gamma', type=float, default=3.0, help='Focal loss gamma')

    # Manual class weight overrides (comma-separated class:weight pairs)
    parser.add_argument('--weight-overrides', type=str, default=None,
                        help='Manual class weight overrides, e.g. "6:8.0,7:8.0"')

    # Augmentation
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--noise-std', type=float, default=0.01)
    parser.add_argument('--train-anomaly', action='store_true',
                        help='Train anomaly scoring head (damage classes = anomaly)')
    parser.add_argument('--anomaly-weight', type=float, default=0.1,
                        help='Weight for anomaly BCE loss relative to classification loss')
    parser.add_argument('--train-per-mod-damage', action='store_true',
                        help='Train per-modality 4-class damage heads (normal/c6/c7/c8) before fusion')
    parser.add_argument('--per-mod-damage-weight', type=float, default=0.3,
                        help='Weight for per-modality damage loss')
    parser.add_argument('--per-mod-damage-modalities', type=str, default='6,1,2',
                        help='Comma-separated modality indices to train damage heads on (default: 6=machine,1=gyro,2=mag)')

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
    print(f"Device: {device}")

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build modality indices from metadata
    metadata_path = Path(args.split_dir) / 'metadata.json'
    print(f"\nBuilding modality groups from {metadata_path}:")
    group_names, groups, sensor_dims = build_modality_indices(str(metadata_path))
    group_indices = [groups[name] for name in group_names]

    # Load datasets
    max_seq_len = 32  # not used for encoder-only but required by dataset
    train_dataset = DecoderDatasetFromSplits(split_dir=args.split_dir, split='train', max_token_len=max_seq_len)
    val_dataset = DecoderDatasetFromSplits(split_dir=args.split_dir, split='val', max_token_len=max_seq_len)
    test_dataset = DecoderDatasetFromSplits(split_dir=args.split_dir, split='test', max_token_len=max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=decoder_collate_fn, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=decoder_collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=decoder_collate_fn, num_workers=0)

    print(f"\nDataset: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Class distribution
    op_counts = Counter()
    for i in range(len(train_dataset)):
        op_counts[train_dataset.operation_type[i].item()] += 1
    print(f"Train class distribution: {dict(sorted(op_counts.items()))}")

    # Build model config
    config = ModelConfig(
        sensor_dims=sensor_dims,
        d_model=args.d_model,
        lstm_layers=args.lstm_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        gcode_vocab=128,  # not used, but required
        future_len=8,     # not used
        cls_pooling=args.cls_pooling,
    )

    model = MM_DTAE_LSTM(config)

    # Override head_cls to have correct number of classes
    model.head_cls = nn.Linear(args.d_model, args.n_operations)

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: MM_DTAE_LSTM, {n_params:,} trainable parameters")
    print(f"  d_model={args.d_model}, n_heads={args.n_heads}, lstm_layers={args.lstm_layers}")
    print(f"  cls_pooling={args.cls_pooling}")
    print(f"  {len(sensor_dims)} modalities: {sensor_dims}")

    # Loss
    weights = None
    if args.class_weights:
        total = sum(op_counts.values())
        weights = torch.zeros(args.n_operations)
        for c, count in op_counts.items():
            weights[c] = total / (args.n_operations * count)
        weights = weights / weights.sum() * args.n_operations

    # Apply manual weight overrides
    if args.weight_overrides:
        if weights is None:
            weights = torch.ones(args.n_operations)
        for pair in args.weight_overrides.split(','):
            cls_id, w = pair.split(':')
            weights[int(cls_id)] = float(w)
        print(f"Class weights (with overrides): {weights.tolist()}")
    elif weights is not None:
        print(f"Class weights: {weights.tolist()}")

    if args.focal_loss:
        print(f"Using Focal Loss (gamma={args.focal_gamma})")
        criterion = FocalLoss(
            weight=weights.to(device) if weights is not None else None,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=weights.to(device) if weights is not None else None,
            label_smoothing=args.label_smoothing)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Scheduler
    if args.scheduler == 'cosine_restarts':
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.restart_period, T_mult=args.restart_mult)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    log_path = output_dir / 'training_log.txt'

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    # Per-modality damage head setup
    if args.train_per_mod_damage:
        active_damage_mods = [int(x) for x in args.per_mod_damage_modalities.split(',')]
        # Balanced weights for 4-class damage (normal is ~85% of data)
        n_normal = sum(c for op, c in op_counts.items() if op < 6)
        n_damage = sum(c for op, c in op_counts.items() if op >= 6)
        damage_w = torch.ones(4, device=device)
        if n_damage > 0:
            # Weight damage classes proportionally
            for damage_op, damage_4c in [(6, 1), (7, 2), (8, 3)]:
                n_cls = op_counts.get(damage_op, 1)
                damage_w[damage_4c] = n_normal / (3 * n_cls) if n_cls > 0 else 1.0
            damage_w[0] = 1.0  # normal class weight = 1
        per_mod_damage_criterion = nn.CrossEntropyLoss(weight=damage_w)
        log(f"Per-modality damage heads active on modalities: {active_damage_mods}")
        log(f"  Damage class weights: {damage_w.tolist()}")

    log(f"\n{'='*70}")
    log(f"MM_DTAE_LSTM Standalone Op Classification Training")
    log(f"{'='*70}")

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

            # Optional noise augmentation
            if args.augment and args.noise_std > 0:
                sensor_data = sensor_data + torch.randn_like(sensor_data) * args.noise_std

            mods = split_modalities(sensor_data, group_indices)
            out = model(mods, lengths, modality_dropout_p=args.modality_dropout if model.training else 0.0)

            cls_loss = criterion(out['cls'], operations)

            # Anomaly scoring: damage classes (6,7,8) = anomaly
            if args.train_anomaly:
                anom_labels = ((operations == 6) | (operations == 7) | (operations == 8)).float()
                anom_loss = nn.functional.binary_cross_entropy_with_logits(
                    out['anom'].squeeze(-1), anom_labels)
                loss = cls_loss + args.anomaly_weight * anom_loss
            else:
                loss = cls_loss

            # Per-modality damage heads: 4-class (0=normal, 1=c6, 2=c7, 3=c8)
            if args.train_per_mod_damage:
                damage_labels = torch.zeros(B, dtype=torch.long, device=device)
                damage_labels[operations == 6] = 1
                damage_labels[operations == 7] = 2
                damage_labels[operations == 8] = 3

                per_mod_loss = 0.0
                for m_idx in active_damage_mods:
                    mod_logits = out['per_mod_damage'][m_idx]  # [B, 4]
                    per_mod_loss = per_mod_loss + per_mod_damage_criterion(mod_logits, damage_labels)
                per_mod_loss = per_mod_loss / len(active_damage_mods)
                loss = loss + args.per_mod_damage_weight * per_mod_loss

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            preds = out['cls'].argmax(dim=-1)
            epoch_correct += (preds == operations).sum().item()
            epoch_total += B
            epoch_loss += loss.item() * B

        scheduler.step()
        train_acc = epoch_correct / epoch_total
        train_loss = epoch_loss / epoch_total
        elapsed = time.time() - t0

        # Validate
        val_acc, val_per_class, _ = evaluate(model, val_loader, group_indices, device, args.n_operations)

        # Check improvement
        improved = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            improved = ' ** NEW BEST **'
            # Save
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'best_val_acc': best_val_acc,
                'sensor_dims': sensor_dims,
                'group_names': group_names,
                'n_operations': args.n_operations,
                'd_model': args.d_model,
                'cls_pooling': args.cls_pooling,
            }, output_dir / 'best_model.pt')
        else:
            no_improve += 1

        lr = optimizer.param_groups[0]['lr']
        per_class_str = ' '.join([f"c{c}:{a:.0%}" for c, a in sorted(val_per_class.items())])
        log(f"Epoch {epoch:3d}/{args.max_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} | "
            f"val_acc={val_acc:.2%} | lr={lr:.2e} | {elapsed:.1f}s{improved}")
        log(f"  Val per-class: {per_class_str}")

        if no_improve >= args.patience:
            log(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # Test evaluation
    log(f"\n{'='*70}")
    log(f"Loading best model from epoch {best_epoch} (val_acc={best_val_acc:.2%})")

    ckpt = torch.load(output_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    compute_anom = args.train_anomaly
    test_acc, test_per_class, test_anom = evaluate(model, test_loader, group_indices, device, args.n_operations, compute_anomaly=compute_anom)
    train_acc_final, train_per_class, train_anom = evaluate(model, train_loader, group_indices, device, args.n_operations, compute_anomaly=compute_anom)
    val_acc_final, val_per_class_final, val_anom = evaluate(model, val_loader, group_indices, device, args.n_operations, compute_anomaly=compute_anom)

    log(f"\nFINAL RESULTS (best epoch {best_epoch}):")
    log(f"  Train Acc: {train_acc_final:.2%}")
    log(f"  Val Acc:   {val_acc_final:.2%}")
    log(f"  Test Acc:  {test_acc:.2%}")
    log(f"\n  Test per-class accuracy:")
    for c in sorted(test_per_class.keys()):
        log(f"    Class {c}: {test_per_class[c]:.2%}")

    if test_anom:
        log(f"\n  Anomaly Detection (test):")
        log(f"    Precision: {test_anom['precision']:.2%}")
        log(f"    Recall:    {test_anom['recall']:.2%}")
        log(f"    F1:        {test_anom['f1']:.2%}")
        log(f"    TP={test_anom['tp']} FP={test_anom['fp']} TN={test_anom['tn']} FN={test_anom['fn']}")

    # Per-modality damage pipeline evaluation
    pipeline_results = None
    if args.train_per_mod_damage:
        log(f"\n  Per-modality damage pipeline (test):")
        pipeline_results = evaluate_per_mod_pipeline(
            model, test_loader, group_indices, device, active_damage_mods, args.n_operations)
        for thresh, r in sorted(pipeline_results.items()):
            log(f"    t={thresh:.1f}: {r['overall']:.2%}  c0-c5={r['c0_c5']:.2%}  "
                f"c6={r['c6']:.0%} c7={r['c7']:.0%} c8={r['c8']:.0%}")

    # Save results
    results = {
        'best_epoch': best_epoch,
        'train_accuracy': train_acc_final,
        'val_accuracy': val_acc_final,
        'test_accuracy': test_acc,
        'train_per_class': {str(k): v for k, v in train_per_class.items()},
        'val_per_class': {str(k): v for k, v in val_per_class_final.items()},
        'test_per_class': {str(k): v for k, v in test_per_class.items()},
        'anomaly_metrics': {
            'train': train_anom, 'val': val_anom, 'test': test_anom
        } if test_anom else None,
        'per_mod_damage_pipeline': {str(k): v for k, v in pipeline_results.items()} if pipeline_results else None,
        'sensor_dims': sensor_dims,
        'group_names': group_names,
        'n_params': n_params,
        'args': vars(args),
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    log(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == '__main__':
    main()
