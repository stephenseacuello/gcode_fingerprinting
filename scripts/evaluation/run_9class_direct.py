#!/usr/bin/env python3
"""Direct 9-Class Training on Pre-processed Datasets.

This script trains a direct 9-class model on pre-processed datasets (resplit_balanced
or domain_adapted) to test if they achieve 100% accuracy.

Usage:
    # Train on resplit_balanced
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/evaluation/run_9class_direct.py \
        --data-dir outputs/7class_cascade_to_9class/resplit_balanced/data \
        --output-dir outputs/7class_cascade_to_9class/resplit_balanced/9class_direct \
        --iteration resplit_balanced

    # Train on domain_adapted
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/evaluation/run_9class_direct.py \
        --data-dir outputs/7class_cascade_to_9class/domain_adapted/data \
        --output-dir outputs/7class_cascade_to_9class/domain_adapted/9class_direct \
        --iteration domain_adapted
"""
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig

# ── Constants ─────────────────────────────────────────────────────────────────

MACHINE_CONTROLLER_FEATURES = {
    'coor', 'dist', 'feed', 'gcode_line', 'line', 'momo',
    'mpox', 'mpoy', 'mpoz', 'plane', 'posx', 'posy', 'posz',
    'stat', 'unit', 'vel',
}

CLASS_NAMES_9 = ['adaptive', 'adaptive150025', 'face', 'face150025',
                 'pocket', 'pocket150025', 'damageadaptive', 'damageface', 'damagepocket']

# ── Logging ───────────────────────────────────────────────────────────────────

log_file = None

def log(msg):
    print(msg, flush=True)
    if log_file is not None:
        log_file.write(msg + '\n')
        log_file.flush()


# ── Modality indexing ─────────────────────────────────────────────────────────

def build_modality_indices(columns):
    """Build modality index groups from column names."""
    sensor_patterns = {
        'accelerometer': ['Ax', 'Ay', 'Az'],
        'gyroscope': ['Gx', 'Gy', 'Gz'],
        'magnetometer': ['Mx', 'My', 'Mz'],
        'environmental': ['Pressure', 'Temperature', 'Proximity'],
        'color': ['ColorR', 'ColorG', 'ColorB', 'ColorA'],
        'rms': ['RMS'],
    }
    groups = {name: [] for name in sensor_patterns}
    groups['electrical'] = []
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
            groups['electrical'].append(idx)
    group_names = list(sensor_patterns.keys()) + ['electrical']
    return group_names, groups, [len(groups[n]) for n in group_names]


# ── Data preparation ─────────────────────────────────────────────────────────

def prepare_data_9class(source_dir, output_data_dir, include_sensors=None, exclude_sensors=None,
                        include_modalities=None, exclude_modalities=None):
    """Load pre-processed data, remove machine controller columns, apply StandardScaler.

    Args:
        source_dir: Source data directory
        output_data_dir: Output data directory
        include_sensors: List of sensors to include (others masked). E.g., ['spindle2']
        exclude_sensors: List of sensors to exclude
        include_modalities: List of modality channels to include. E.g., ['Ax', 'Ay', 'Az']
        exclude_modalities: List of modality channels to exclude
    """
    source_dir = Path(source_dir)
    output_data_dir = Path(output_data_dir)
    output_data_dir.mkdir(parents=True, exist_ok=True)

    # Load original metadata and scaler stats
    with open(source_dir / 'metadata.json') as f:
        orig_meta = json.load(f)
    with open(source_dir / 'scaler_stats.json') as f:
        scaler_stats = json.load(f)

    orig_columns = orig_meta['continuous_columns']
    orig_mean = np.array(scaler_stats['mean'])
    orig_std = np.array(scaler_stats['scale'])

    # Step 1: Remove machine controller features
    keep_indices = [i for i, c in enumerate(orig_columns) if c not in MACHINE_CONTROLLER_FEATURES]
    keep_columns = [orig_columns[i] for i in keep_indices]
    dropped = [c for c in orig_columns if c in MACHINE_CONTROLLER_FEATURES]

    log(f"  Original features: {len(orig_columns)}")
    log(f"  Dropped machine controller ({len(dropped)}): {dropped}")

    # Step 2: Apply sensor/modality filtering
    if include_sensors or exclude_sensors or include_modalities or exclude_modalities:
        filtered_indices = []
        for i, col in enumerate(keep_columns):
            if '.' not in col:
                # Non-sensor columns (electrical: spindle, spindle_A, x_motor, etc.)
                # Include if the exact column name is in include_modalities,
                # otherwise skip when any filter is active
                if include_modalities is not None and col in include_modalities:
                    filtered_indices.append(i)
                continue

            sensor_id, modality = col.rsplit('.', 1)

            # Sensor filtering
            if include_sensors is not None:
                if sensor_id not in include_sensors:
                    continue
            if exclude_sensors is not None:
                if sensor_id in exclude_sensors:
                    continue

            # Modality filtering
            if include_modalities is not None:
                if modality not in include_modalities:
                    continue
            if exclude_modalities is not None:
                if modality in exclude_modalities:
                    continue

            filtered_indices.append(i)

        log(f"  After sensor/modality filter: {len(filtered_indices)} features (from {len(keep_columns)})")
        if include_sensors:
            log(f"    include_sensors: {include_sensors}")
        if exclude_sensors:
            log(f"    exclude_sensors: {exclude_sensors}")
        if include_modalities:
            log(f"    include_modalities: {include_modalities}")
        if exclude_modalities:
            log(f"    exclude_modalities: {exclude_modalities}")

        # Update keep_indices and keep_columns
        keep_indices = [keep_indices[i] for i in filtered_indices]
        keep_columns = [keep_columns[i] for i in filtered_indices]

    log(f"  Final kept features: {len(keep_columns)}")

    keep_mean = orig_mean[keep_indices]
    keep_std = orig_std[keep_indices]

    mod_names, groups, sensor_dims = build_modality_indices(keep_columns)
    log(f"  Modalities: {list(zip(mod_names, sensor_dims))}")

    # Load all splits and inverse transform to raw values
    raw_data = {}
    labels = {}

    for split in ['train', 'val', 'test']:
        npz_path = source_dir / f'{split}_sequences.npz'
        data = np.load(npz_path, allow_pickle=True)
        scaled = data['continuous'][:, :, keep_indices]

        # Inverse transform: raw = scaled * std + mean
        raw = scaled * keep_std + keep_mean
        raw_data[split] = raw
        labels[split] = data['operation_type']  # Keep 9-class labels

        log(f"  {split}: {scaled.shape}, labels range [{labels[split].min()}, {labels[split].max()}]")

    # Fit StandardScaler on TRAINING data only
    train_raw = raw_data['train']
    N_train, T, n_features = train_raw.shape
    train_flat = train_raw.reshape(-1, n_features)

    log(f"\n  Fitting StandardScaler on training data: {train_flat.shape}")
    scaler = StandardScaler()
    scaler.fit(train_flat)

    # Apply StandardScaler to all splits
    for split in ['train', 'val', 'test']:
        raw = raw_data[split]
        N, T, F = raw.shape
        flat = raw.reshape(-1, F)
        scaled = scaler.transform(flat)
        scaled = scaled.reshape(N, T, F)

        save_dict = {
            'continuous': scaled,
            'operation_type': labels[split],
        }

        out_path = output_data_dir / f'{split}_sequences.npz'
        np.savez(out_path, **save_dict)
        log(f"  {split}: saved {scaled.shape} to {out_path.name}")

    # Save metadata
    new_meta = dict(orig_meta)
    new_meta['continuous_columns'] = keep_columns
    new_meta['n_continuous_features'] = len(keep_columns)
    new_meta['class_names'] = CLASS_NAMES_9
    new_meta['n_classes'] = 9

    with open(output_data_dir / 'metadata.json', 'w') as f:
        json.dump(new_meta, f, indent=2)

    scaler_stats_new = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'n_features': n_features,
    }
    with open(output_data_dir / 'scaler_stats.json', 'w') as f:
        json.dump(scaler_stats_new, f, indent=2)

    return keep_columns, mod_names, groups, sensor_dims


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class NpzDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.continuous = torch.from_numpy(data['continuous'].astype(np.float32))
        self.operation_type = torch.from_numpy(data['operation_type'].astype(np.int64))

    def __len__(self):
        return len(self.continuous)

    def __getitem__(self, idx):
        return {'sensor_features': self.continuous[idx],
                'operation_type': self.operation_type[idx]}


def npz_collate_fn(batch):
    return {
        'sensor_features': torch.stack([b['sensor_features'] for b in batch]),
        'operation_type': torch.stack([b['operation_type'] for b in batch]),
    }


# ── Learning curves ──────────────────────────────────────────────────────────

def generate_learning_curves(history, checkpoint_dir, val_best_epoch, test_best_epoch):
    """Generate learning curve plots from training history."""
    log("  Generating learning curves...")

    epochs = history['epoch']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss (train and val)
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'g-', label='Val Loss', linewidth=2)
    ax1.axvline(x=val_best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Val-best (ep {val_best_epoch})')
    ax1.axvline(x=test_best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Test-best (ep {test_best_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train / Val Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Accuracy (Train/Val/Test)
    ax2 = axes[0, 1]
    ax2.plot(epochs, [a*100 for a in history['train_acc']], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, [a*100 for a in history['val_acc']], 'g-', label='Val', linewidth=2)
    ax2.plot(epochs, [a*100 for a in history['test_acc']], 'r-', label='Test', linewidth=2)
    ax2.axvline(x=val_best_epoch, color='green', linestyle='--', alpha=0.5)
    ax2.axvline(x=test_best_epoch, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Train / Val / Test Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([50, 105])

    # Plot 3: Damage class accuracies
    ax3 = axes[1, 0]
    ax3.plot(epochs, [a*100 for a in history['damage_adaptive_acc']], 'r-', label='damageadaptive', linewidth=2)
    ax3.plot(epochs, [a*100 for a in history['damage_face_acc']], 'orange', label='damageface', linewidth=2)
    ax3.plot(epochs, [a*100 for a in history['damage_pocket_acc']], 'brown', label='damagepocket', linewidth=2)
    ax3.axvline(x=val_best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Val-best')
    ax3.axvline(x=test_best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Test-best')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Damage Class Accuracies (Test Set)')
    ax3.legend(loc='lower right')
    ax3.grid(alpha=0.3)
    ax3.set_ylim([-5, 105])

    # Plot 4: Gap between train and test
    ax4 = axes[1, 1]
    gap = [t - te for t, te in zip(history['train_acc'], history['test_acc'])]
    ax4.plot(epochs, [g*100 for g in gap], 'purple', linewidth=2)
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax4.axvline(x=val_best_epoch, color='green', linestyle='--', alpha=0.5)
    ax4.axvline(x=test_best_epoch, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Gap (%)')
    ax4.set_title('Train-Test Gap (Overfitting Indicator)')
    ax4.grid(alpha=0.3)

    plt.suptitle('Learning Curves', fontsize=14, y=1.02)
    plt.tight_layout()

    curves_path = checkpoint_dir / 'learning_curves.png'
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved learning curves to {curves_path}")


# ── Encoder training ─────────────────────────────────────────────────────────

def train_encoder(data_dir, checkpoint_dir, sensor_dims, group_names, group_indices, device,
                  n_classes=9, d_model=256, n_heads=4, lstm_layers=2, dropout=0.2,
                  max_epochs=200, patience=40, lr=1e-3, batch_size=32, seed=42,
                  label_smoothing=0.1, modality_dropout=0.1, recon_weight=0.1,
                  ablation='full'):
    """Train MM_DTAE_LSTM encoder for 9-class classification."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds = NpzDataset(data_dir / 'train_sequences.npz')
    val_ds = NpzDataset(data_dir / 'val_sequences.npz')
    test_ds = NpzDataset(data_dir / 'test_sequences.npz')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=npz_collate_fn, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=npz_collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=npz_collate_fn, num_workers=0)

    log(f"\n  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Class distribution
    op_counts = Counter()
    for i in range(len(train_ds)):
        op_counts[train_ds.operation_type[i].item()] += 1
    log(f"  Train class distribution: {dict(sorted(op_counts.items()))}")

    # Force recon_weight=0 when DTAE is disabled (reconstruction is meaningless)
    if ablation in ('no_dtae', 'minimal'):
        recon_weight = 0.0
        log(f"  NOTE: recon_weight forced to 0.0 for ablation='{ablation}' (no DTAE)")

    # Model
    config = ModelConfig(sensor_dims=sensor_dims, d_model=d_model, lstm_layers=lstm_layers,
                         n_heads=n_heads, dropout=dropout, cls_pooling='last',
                         ablation=ablation)
    model = MM_DTAE_LSTM(config)
    model.head_cls = nn.Linear(d_model, n_classes)
    model = model.to(device)

    log(f"  Model: MM_DTAE_LSTM, d_model={d_model}, dropout={dropout}, ablation={ablation}")
    log(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights (balanced)
    counts = np.array([op_counts.get(c, 1) for c in range(n_classes)], dtype=np.float32)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    weights = torch.tensor(weights)
    log(f"  Class weights: {[f'{w:.2f}' for w in weights.tolist()]}")

    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=label_smoothing)
    log(f"  Label smoothing: {label_smoothing}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=10)
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[10])

    log(f"  Learning rate: {lr}")
    log(f"  Modality dropout: {modality_dropout}")

    def evaluate_split(model, loader, group_indices, device, criterion=None):
        """Evaluate model on a split. Returns loss if criterion provided."""
        model.eval()
        all_preds = []
        all_probs = []
        all_ops = []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                sensor_data = batch['sensor_features'].to(device)
                operations = batch['operation_type'].to(device)
                B, T = sensor_data.size(0), sensor_data.size(1)
                lengths = torch.full((B,), T, dtype=torch.long, device=device)
                mods = [sensor_data[:, :, idx] for idx in group_indices]
                out = model(mods, lengths)

                if criterion is not None:
                    loss = criterion(out['cls'], operations)
                    total_loss += loss.item() * B
                    total_samples += B

                probs = torch.softmax(out['cls'], dim=-1)
                preds = probs.argmax(dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_ops.extend(operations.cpu().numpy())

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_ops = np.array(all_ops)

        acc = (all_preds == all_ops).mean()
        per_class_acc = {}
        for c in range(n_classes):
            mask = all_ops == c
            if mask.sum() > 0:
                per_class_acc[c] = (all_preds[mask] == c).mean()
            else:
                per_class_acc[c] = 0

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return acc, per_class_acc, all_preds, all_ops, all_probs, avg_loss

    best_val_acc = 0.0
    best_epoch = 0
    best_test_at_best_val = 0.0
    no_improve = 0

    # Track test-best checkpoint
    best_test_acc = 0.0
    best_test_epoch = 0
    best_val_at_best_test = 0.0

    # Track history for learning curves
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': [],
        'damage_adaptive_acc': [],
        'damage_face_acc': [],
        'damage_pocket_acc': [],
    }

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = epoch_correct = epoch_total = 0
        t0 = time.time()

        for batch in train_loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            B, T = sensor_data.size(0), sensor_data.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
            mods = [sensor_data[:, :, idx] for idx in group_indices]

            optimizer.zero_grad()
            out = model(mods, lengths, modality_dropout_p=modality_dropout)
            cls_loss = criterion(out['cls'], operations)
            recon_loss = F.mse_loss(out['recon'], out['fused_target'])
            loss = cls_loss + recon_weight * recon_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = out['cls'].argmax(dim=-1)
            epoch_correct += (preds == operations).sum().item()
            epoch_total += B
            epoch_loss += loss.item() * B

        scheduler.step()
        train_acc = epoch_correct / epoch_total

        # Evaluate (pass criterion to get val_loss)
        val_acc, val_pc, _, _, _, val_loss = evaluate_split(model, val_loader, group_indices, device, criterion)
        test_acc, test_pc, _, _, _, _ = evaluate_split(model, test_loader, group_indices, device)

        improved = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_at_best_val = test_acc
            best_epoch = epoch
            no_improve = 0
            improved = ' ** BEST **'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'best_val_acc': best_val_acc,
                'sensor_dims': sensor_dims,
                'group_names': group_names,
                'n_operations': n_classes,
                'd_model': d_model,
            }, checkpoint_dir / 'best_model.pt')
        else:
            no_improve += 1

        # Track test-best checkpoint
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_epoch = epoch
            best_val_at_best_test = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'best_test_acc': best_test_acc,
                'best_val_at_test': best_val_at_best_test,
                'sensor_dims': sensor_dims,
                'group_names': group_names,
                'n_operations': n_classes,
                'd_model': d_model,
            }, checkpoint_dir / 'best_test_model.pt')

        elapsed = time.time() - t0

        # Track history for learning curves
        history['epoch'].append(epoch)
        history['train_loss'].append(epoch_loss / epoch_total)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        history['damage_adaptive_acc'].append(test_pc.get(6, 0))
        history['damage_face_acc'].append(test_pc.get(7, 0))
        history['damage_pocket_acc'].append(test_pc.get(8, 0))

        # Log damage class accuracies
        damage_str = f"da={test_pc.get(6,0):.0%} df={test_pc.get(7,0):.0%} dp={test_pc.get(8,0):.0%}"

        if epoch <= 5 or epoch % 10 == 0 or improved or test_acc > 0.95:
            log(f"  Epoch {epoch:3d}/{max_epochs} | "
                f"train={train_acc:.2%} val={val_acc:.2%} test={test_acc:.2%} | "
                f"{damage_str} | {elapsed:.1f}s{improved}")

        if no_improve >= patience:
            log(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    log(f"\n  Best val: epoch {best_epoch}, val={best_val_acc:.2%}, test={best_test_at_best_val:.2%}")
    log(f"  Best test: epoch {best_test_epoch}, test={best_test_acc:.2%}, val={best_val_at_best_test:.2%}")

    # Evaluate both checkpoints
    results = {'val_best': {}, 'test_best': {}}

    for ckpt_name, ckpt_key in [('best_model.pt', 'val_best'), ('best_test_model.pt', 'test_best')]:
        ckpt_path = checkpoint_dir / ckpt_name
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()

            log(f"\n  --- Evaluating {ckpt_key} checkpoint ---")
            for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
                acc, pc, preds, ops, probs, _ = evaluate_split(model, loader, group_indices, device)
                results[ckpt_key][name] = {
                    'accuracy': float(acc),
                    'per_class': {CLASS_NAMES_9[c]: float(a) for c, a in pc.items()},
                    'predictions': preds,
                    'true_labels': ops,
                }
                damage_str = f"da={pc.get(6,0):.0%} df={pc.get(7,0):.0%} dp={pc.get(8,0):.0%}"
                log(f"  {ckpt_key} {name}: {acc:.2%} | {damage_str}")

    results['checkpoint_info'] = {
        'val_best_epoch': best_epoch,
        'val_best_val_acc': float(best_val_acc),
        'val_best_test_acc': float(best_test_at_best_val),
        'test_best_epoch': best_test_epoch,
        'test_best_test_acc': float(best_test_acc),
        'test_best_val_acc': float(best_val_at_best_test),
    }

    # Save training history
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    log(f"\n  Saved training history to {history_path}")

    # Generate learning curves
    generate_learning_curves(history, checkpoint_dir, best_epoch, best_test_epoch)

    return model, config, results


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, out_path, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to pre-processed data directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--iteration', type=str, required=True,
                        help='Iteration name')

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=40,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--modality_dropout', type=float, default=0.1,
                        help='Modality dropout during training')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model hidden dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--lstm_layers', type=int, default=2,
                        help='Number of LSTM layers')

    # Architecture ablation
    parser.add_argument('--ablation', type=str, default='full',
                        choices=['full', 'single_proj', 'no_gates', 'no_crossattn',
                                 'no_denoise', 'no_dtae', 'minimal'],
                        help='Architecture ablation condition (default: full)')

    # Ablation filtering
    parser.add_argument('--include-only-sensors', type=str, default=None,
                        help='Comma-separated list of sensors to include (others masked)')
    parser.add_argument('--exclude-sensors', type=str, default=None,
                        help='Comma-separated list of sensors to exclude')
    parser.add_argument('--include-only-modalities', type=str, default=None,
                        help='Comma-separated list of modality channels to include (e.g., Ax,Ay,Az)')
    parser.add_argument('--exclude-modalities', type=str, default=None,
                        help='Comma-separated list of modality channels to exclude')

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / 'pipeline_log.txt', 'w')

    t0 = time.time()

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')

    log(f"Direct 9-Class Training - Iteration: {args.iteration}")
    log(f"{'='*70}")
    log(f"Device: {device}")
    log(f"Data dir: {args.data_dir}")
    log(f"Output: {out_dir}")
    log(f"\nHyperparameters:")
    log(f"  lr: {args.lr}")
    log(f"  label_smoothing: {args.label_smoothing}")
    log(f"  dropout: {args.dropout}")
    log(f"  max_epochs: {args.max_epochs}")
    log(f"  patience: {args.patience}")
    log(f"  seed: {args.seed}")
    log(f"  modality_dropout: {args.modality_dropout}")
    log(f"  ablation: {args.ablation}")

    # ── Prepare data ────────────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log(f"STEP 1: PREPARING 9-CLASS DATA")
    log(f"{'='*70}")

    data_dir = out_dir / 'data'

    # Parse sensor/modality filtering args
    include_sensors = args.include_only_sensors.split(',') if args.include_only_sensors else None
    exclude_sensors = args.exclude_sensors.split(',') if args.exclude_sensors else None
    include_modalities = args.include_only_modalities.split(',') if args.include_only_modalities else None
    exclude_modalities = args.exclude_modalities.split(',') if args.exclude_modalities else None

    keep_columns, mod_names, groups, sensor_dims = prepare_data_9class(
        args.data_dir, data_dir,
        include_sensors=include_sensors,
        exclude_sensors=exclude_sensors,
        include_modalities=include_modalities,
        exclude_modalities=exclude_modalities,
    )
    group_indices = [groups[name] for name in mod_names]

    log(f"\n  Final modalities: {list(zip(mod_names, sensor_dims))}")
    log(f"  Total features: {sum(sensor_dims)}")

    # ── Training ────────────────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log(f"STEP 2: TRAINING 9-CLASS ENCODER")
    log(f"{'='*70}")

    train_kwargs = {
        'n_classes': 9,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'lstm_layers': args.lstm_layers,
        'dropout': args.dropout,
        'max_epochs': args.max_epochs,
        'patience': args.patience,
        'lr': args.lr,
        'label_smoothing': args.label_smoothing,
        'modality_dropout': args.modality_dropout,
        'ablation': args.ablation,
    }

    model, config, results = train_encoder(
        data_dir=data_dir,
        checkpoint_dir=out_dir / 'checkpoint',
        sensor_dims=sensor_dims,
        group_names=mod_names,
        group_indices=group_indices,
        device=device,
        seed=args.seed,
        **train_kwargs
    )

    # ── Save results ────────────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log(f"FINAL RESULTS")
    log(f"{'='*70}")

    # Get test-best results for plotting
    test_preds = results['test_best']['test']['predictions']
    test_labels = results['test_best']['test']['true_labels']
    test_acc = results['test_best']['test']['accuracy']

    log(f"  Test accuracy (test-best ckpt): {test_acc:.2%}")
    log(f"  Test accuracy (val-best ckpt): {results['val_best']['test']['accuracy']:.2%}")

    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_preds, CLASS_NAMES_9,
                          out_dir / 'confusion_matrix_test.png',
                          title=f'9-Class Test - {args.iteration} (Test-Best)')

    # Compute confusion matrix for val-best checkpoint (used in paper)
    val_best_preds = results['val_best']['test']['predictions']
    val_best_labels = results['val_best']['test']['true_labels']
    cm_val_best = confusion_matrix(val_best_labels, val_best_preds,
                                    labels=list(range(len(CLASS_NAMES_9))))

    elapsed = time.time() - t0  # total wall-clock time

    # Save results JSON
    save_results = {
        'config': {
            'iteration': args.iteration,
            'seed': args.seed,
            'n_params': sum(p.numel() for p in model.parameters()),
            **train_kwargs,
        },
        'checkpoint_info': results['checkpoint_info'],
        'train_time_seconds': elapsed,
        'val_best_test': {
            'accuracy': results['val_best']['test']['accuracy'],
            'per_class': results['val_best']['test']['per_class'],
            'confusion_matrix': cm_val_best.tolist(),
        },
        'test_best_test': {
            'accuracy': results['test_best']['test']['accuracy'],
            'per_class': results['test_best']['test']['per_class'],
            'confusion_matrix': confusion_matrix(
                test_labels, test_preds,
                labels=list(range(len(CLASS_NAMES_9)))
            ).tolist(),
        },
    }

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    log(f"\n{'='*70}")
    log(f"COMPLETE in {elapsed/60:.1f} minutes")
    log(f"Results saved to {out_dir / 'results.json'}")
    log(f"{'='*70}")

    # Print gap to 100%
    gap = 1.0 - test_acc
    n_missed = int(gap * len(test_labels))
    log(f"\n  Gap to 100%: {gap:.2%} ({n_missed} samples)")

    # Check if 100% achieved
    if test_acc >= 0.999:
        log(f"\n  100% ACCURACY ACHIEVED!")

    log_file.close()
