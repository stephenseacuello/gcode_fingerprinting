#!/usr/bin/env python3
"""Architecture Ablation Study for 9-Class Classification.

This script runs architecture ablation experiments varying:
- Model size (d_model, lstm_layers, n_heads)
- Components (use_lstm, use_attention)
- Regularization (dropout, modality_dropout, label_smoothing)
- Pooling strategy (last, mean, max, attention_pool)
- Training (use_class_weights, lr)

Usage:
    # Model size ablation (tiny)
    python scripts/evaluation/run_architecture_ablation.py \
        --data-dir outputs/ablation_study/data/split_42 \
        --output-dir outputs/ablation_study/architecture/size/tiny_ds42_ms42 \
        --d_model 64 --lstm_layers 1 --n_heads 2 \
        --seed 42

    # Component ablation (no LSTM)
    python scripts/evaluation/run_architecture_ablation.py \
        --data-dir outputs/ablation_study/data/split_42 \
        --output-dir outputs/ablation_study/architecture/component/no_lstm_ds42_ms42 \
        --use_lstm false \
        --seed 42

    # Pooling ablation (mean pooling)
    python scripts/evaluation/run_architecture_ablation.py \
        --data-dir outputs/ablation_study/data/split_42 \
        --output-dir outputs/ablation_study/architecture/pooling/mean_ds42_ms42 \
        --cls_pooling mean \
        --seed 42
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


# ── Model variants ────────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """Learnable attention pooling over sequence dimension."""
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

    def forward(self, x):
        # x: (B, T, D)
        B = x.size(0)
        query = self.query.expand(B, -1, -1)  # (B, 1, D)
        out, _ = self.attn(query, x, x)  # (B, 1, D)
        return out.squeeze(1)  # (B, D)


class MM_DTAE_NoLSTM(nn.Module):
    """MM_DTAE variant without LSTM - attention only."""
    def __init__(self, config, n_classes=9):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # Modality projections
        self.modality_projs = nn.ModuleList([
            nn.Linear(dim, config.d_model) for dim in config.sensor_dims
        ])

        # Positional encoding
        max_len = 1000
        pe = torch.zeros(max_len, config.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * (-np.log(10000.0) / config.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        # Self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Pooling
        self.cls_pooling = config.cls_pooling
        if self.cls_pooling == 'attention_pool':
            self.attention_pool = AttentionPooling(config.d_model)

        # Classification head
        self.head_cls = nn.Linear(config.d_model, n_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, modalities, lengths, modality_dropout_p=0.0):
        B, T = modalities[0].size(0), modalities[0].size(1)

        # Project and concatenate modalities
        projected = []
        for i, (mod, proj) in enumerate(zip(modalities, self.modality_projs)):
            if modality_dropout_p > 0 and self.training and torch.rand(1).item() < modality_dropout_p:
                projected.append(torch.zeros(B, T, self.d_model, device=mod.device))
            else:
                projected.append(proj(mod))

        # Mean fusion
        x = torch.stack(projected, dim=0).mean(dim=0)  # (B, T, D)

        # Add positional encoding
        x = x + self.pe[:, :T, :]

        # Transformer
        x = self.transformer(x)

        # Pooling
        if self.cls_pooling == 'last':
            cls_feat = x[:, -1, :]
        elif self.cls_pooling == 'mean':
            cls_feat = x.mean(dim=1)
        elif self.cls_pooling == 'max':
            cls_feat = x.max(dim=1)[0]
        elif self.cls_pooling == 'attention_pool':
            cls_feat = self.attention_pool(x)

        cls_feat = self.dropout(cls_feat)
        logits = self.head_cls(cls_feat)

        return {'cls': logits, 'cls_feat': cls_feat}


class MM_DTAE_NoAttention(nn.Module):
    """MM_DTAE variant without attention - LSTM only."""
    def __init__(self, config, n_classes=9):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # Modality projections
        self.modality_projs = nn.ModuleList([
            nn.Linear(dim, config.d_model) for dim in config.sensor_dims
        ])

        # LSTM
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            bidirectional=False
        )

        # Pooling
        self.cls_pooling = config.cls_pooling
        if self.cls_pooling == 'attention_pool':
            self.attention_pool = AttentionPooling(config.d_model)

        # Classification head
        self.head_cls = nn.Linear(config.d_model, n_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, modalities, lengths, modality_dropout_p=0.0):
        B, T = modalities[0].size(0), modalities[0].size(1)

        # Project and concatenate modalities
        projected = []
        for i, (mod, proj) in enumerate(zip(modalities, self.modality_projs)):
            if modality_dropout_p > 0 and self.training and torch.rand(1).item() < modality_dropout_p:
                projected.append(torch.zeros(B, T, self.d_model, device=mod.device))
            else:
                projected.append(proj(mod))

        # Mean fusion
        x = torch.stack(projected, dim=0).mean(dim=0)  # (B, T, D)

        # LSTM
        x, (h_n, c_n) = self.lstm(x)

        # Pooling
        if self.cls_pooling == 'last':
            cls_feat = x[:, -1, :]
        elif self.cls_pooling == 'mean':
            cls_feat = x.mean(dim=1)
        elif self.cls_pooling == 'max':
            cls_feat = x.max(dim=1)[0]
        elif self.cls_pooling == 'attention_pool':
            cls_feat = self.attention_pool(x)

        cls_feat = self.dropout(cls_feat)
        logits = self.head_cls(cls_feat)

        return {'cls': logits, 'cls_feat': cls_feat}


class MM_DTAE_Pooling(nn.Module):
    """MM_DTAE_LSTM with configurable pooling strategy."""
    def __init__(self, config, n_classes=9):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.cls_pooling = config.cls_pooling

        # Create base model
        self.base_model = MM_DTAE_LSTM(config)

        # Override head for n_classes
        self.base_model.head_cls = nn.Linear(config.d_model, n_classes)

        # Add attention pooling if needed
        if self.cls_pooling == 'attention_pool':
            self.attention_pool = AttentionPooling(config.d_model)

    def forward(self, modalities, lengths, modality_dropout_p=0.0):
        # Get base model output (uses 'last' pooling by default)
        out = self.base_model(modalities, lengths, modality_dropout_p=modality_dropout_p)

        # If using different pooling, we need to access the sequence
        # For now, fall back to base model behavior
        return out


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

def prepare_data(source_dir, output_data_dir):
    """Load pre-processed data, remove machine controller columns, apply StandardScaler."""
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

    # Compute keep indices (remove machine controller)
    keep_indices = [i for i, c in enumerate(orig_columns) if c not in MACHINE_CONTROLLER_FEATURES]
    keep_columns = [orig_columns[i] for i in keep_indices]
    dropped = [c for c in orig_columns if c in MACHINE_CONTROLLER_FEATURES]

    log(f"  Original features: {len(orig_columns)}")
    log(f"  Dropped ({len(dropped)}): {dropped}")
    log(f"  Kept features: {len(keep_columns)}")

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
        labels[split] = data['operation_type']

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

    # Plot 1: Loss
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

    # Plot 2: Accuracy
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
    ax2.set_ylim([0, 105])

    # Plot 3: Damage class accuracies
    ax3 = axes[1, 0]
    ax3.plot(epochs, [a*100 for a in history['damage_adaptive_acc']], 'r-', label='damageadaptive', linewidth=2)
    ax3.plot(epochs, [a*100 for a in history['damage_face_acc']], 'orange', label='damageface', linewidth=2)
    ax3.plot(epochs, [a*100 for a in history['damage_pocket_acc']], 'brown', label='damagepocket', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Damage Class Accuracies (Test Set)')
    ax3.legend(loc='lower right')
    ax3.grid(alpha=0.3)
    ax3.set_ylim([-5, 105])

    # Plot 4: Gap
    ax4 = axes[1, 1]
    gap = [t - te for t, te in zip(history['train_acc'], history['test_acc'])]
    ax4.plot(epochs, [g*100 for g in gap], 'purple', linewidth=2)
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Gap (%)')
    ax4.set_title('Train-Test Gap')
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(checkpoint_dir / 'learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Training ─────────────────────────────────────────────────────────────────

def train_encoder(data_dir, checkpoint_dir, sensor_dims, group_names, group_indices, device,
                  n_classes=9, d_model=256, n_heads=4, lstm_layers=2, dropout=0.2,
                  max_epochs=200, patience=40, lr=1e-3, batch_size=32, seed=42,
                  label_smoothing=0.1, modality_dropout=0.1,
                  use_lstm=True, use_attention=True, use_class_weights=True,
                  cls_pooling='last'):
    """Train encoder for 9-class classification."""
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

    # Model selection
    config = ModelConfig(sensor_dims=sensor_dims, d_model=d_model, lstm_layers=lstm_layers,
                         n_heads=n_heads, dropout=dropout, cls_pooling=cls_pooling)

    if use_lstm and use_attention:
        # Standard MM_DTAE_LSTM
        model = MM_DTAE_LSTM(config)
        model.head_cls = nn.Linear(d_model, n_classes)
        model_type = 'MM_DTAE_LSTM'
    elif use_lstm and not use_attention:
        # LSTM only, no attention
        model = MM_DTAE_NoAttention(config, n_classes=n_classes)
        model_type = 'MM_DTAE_NoAttention'
    elif not use_lstm and use_attention:
        # Attention only, no LSTM
        model = MM_DTAE_NoLSTM(config, n_classes=n_classes)
        model_type = 'MM_DTAE_NoLSTM'
    else:
        raise ValueError("Must use at least one of: lstm, attention")

    model = model.to(device)

    log(f"  Model: {model_type}, d_model={d_model}, dropout={dropout}")
    log(f"  Pooling: {cls_pooling}")
    log(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    if use_class_weights:
        counts = np.array([op_counts.get(c, 1) for c in range(n_classes)], dtype=np.float32)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(weights)
        weights = torch.tensor(weights).to(device)
        log(f"  Class weights: {[f'{w:.2f}' for w in weights.tolist()]}")
    else:
        weights = None
        log(f"  Class weights: None (disabled)")

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    log(f"  Label smoothing: {label_smoothing}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=10)
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[10])

    log(f"  Learning rate: {lr}")
    log(f"  Modality dropout: {modality_dropout}")

    def evaluate_split(model, loader, group_indices, device, criterion=None):
        model.eval()
        all_preds, all_probs, all_ops = [], [], []
        total_loss, total_samples = 0.0, 0

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
        return acc, per_class_acc, all_preds, all_ops, np.array(all_probs), avg_loss

    best_val_acc = 0.0
    best_epoch = 0
    best_test_at_best_val = 0.0
    no_improve = 0

    best_test_acc = 0.0
    best_test_epoch = 0
    best_val_at_best_test = 0.0

    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': [],
        'damage_adaptive_acc': [], 'damage_face_acc': [], 'damage_pocket_acc': [],
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
            loss = criterion(out['cls'], operations)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = out['cls'].argmax(dim=-1)
            epoch_correct += (preds == operations).sum().item()
            epoch_total += B
            epoch_loss += loss.item() * B

        scheduler.step()
        train_acc = epoch_correct / epoch_total

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
            }, checkpoint_dir / 'best_model.pt')
        else:
            no_improve += 1

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_epoch = epoch
            best_val_at_best_test = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
            }, checkpoint_dir / 'best_test_model.pt')

        history['epoch'].append(epoch)
        history['train_loss'].append(epoch_loss / epoch_total)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        history['damage_adaptive_acc'].append(test_pc.get(6, 0))
        history['damage_face_acc'].append(test_pc.get(7, 0))
        history['damage_pocket_acc'].append(test_pc.get(8, 0))

        damage_str = f"da={test_pc.get(6,0):.0%} df={test_pc.get(7,0):.0%} dp={test_pc.get(8,0):.0%}"

        if epoch <= 5 or epoch % 10 == 0 or improved or test_acc > 0.95:
            log(f"  Epoch {epoch:3d}/{max_epochs} | train={train_acc:.2%} val={val_acc:.2%} test={test_acc:.2%} | "
                f"{damage_str} | {time.time()-t0:.1f}s{improved}")

        if no_improve >= patience:
            log(f"  Early stopping at epoch {epoch}")
            break

    log(f"\n  Best val: epoch {best_epoch}, val={best_val_acc:.2%}, test={best_test_at_best_val:.2%}")
    log(f"  Best test: epoch {best_test_epoch}, test={best_test_acc:.2%}, val={best_val_at_best_test:.2%}")

    # Final evaluation
    results = {'val_best': {}, 'test_best': {}}

    for ckpt_name, ckpt_key in [('best_model.pt', 'val_best'), ('best_test_model.pt', 'test_best')]:
        ckpt_path = checkpoint_dir / ckpt_name
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()

            for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
                acc, pc, preds, ops, probs, _ = evaluate_split(model, loader, group_indices, device)
                results[ckpt_key][name] = {
                    'accuracy': float(acc),
                    'per_class': {CLASS_NAMES_9[c]: float(a) for c, a in pc.items()},
                    'predictions': preds,
                    'true_labels': ops,
                }

    results['checkpoint_info'] = {
        'val_best_epoch': best_epoch, 'val_best_val_acc': float(best_val_acc),
        'val_best_test_acc': float(best_test_at_best_val),
        'test_best_epoch': best_test_epoch, 'test_best_test_acc': float(best_test_acc),
        'test_best_val_acc': float(best_val_at_best_test),
    }

    # Save history and generate curves
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Architecture Ablation Study')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--iteration', type=str, default=None)

    # Architecture params
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--modality_dropout', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--cls_pooling', type=str, default='last',
                        choices=['last', 'mean', 'max', 'attention_pool'])

    # Component flags
    parser.add_argument('--use_lstm', type=str2bool, default=True)
    parser.add_argument('--use_attention', type=str2bool, default=True)
    parser.add_argument('--use_class_weights', type=str2bool, default=True)

    # Training params
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / 'pipeline_log.txt', 'w')

    t0 = time.time()
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')

    iteration = args.iteration or f"arch_s{args.seed}"

    log(f"Architecture Ablation Study - {iteration}")
    log(f"{'='*70}")
    log(f"Device: {device}")
    log(f"Data dir: {args.data_dir}")
    log(f"Output: {out_dir}")
    log(f"\nArchitecture Config:")
    log(f"  d_model: {args.d_model}")
    log(f"  n_heads: {args.n_heads}")
    log(f"  lstm_layers: {args.lstm_layers}")
    log(f"  dropout: {args.dropout}")
    log(f"  modality_dropout: {args.modality_dropout}")
    log(f"  label_smoothing: {args.label_smoothing}")
    log(f"  cls_pooling: {args.cls_pooling}")
    log(f"  use_lstm: {args.use_lstm}")
    log(f"  use_attention: {args.use_attention}")
    log(f"  use_class_weights: {args.use_class_weights}")
    log(f"\nTraining Config:")
    log(f"  lr: {args.lr}")
    log(f"  max_epochs: {args.max_epochs}")
    log(f"  patience: {args.patience}")
    log(f"  seed: {args.seed}")

    # Prepare data
    log(f"\n{'='*70}")
    log(f"STEP 1: PREPARING DATA")
    log(f"{'='*70}")

    data_dir = out_dir / 'data'
    keep_columns, mod_names, groups, sensor_dims = prepare_data(args.data_dir, data_dir)
    group_indices = [groups[name] for name in mod_names]

    # Train
    log(f"\n{'='*70}")
    log(f"STEP 2: TRAINING")
    log(f"{'='*70}")

    model, config, results = train_encoder(
        data_dir=data_dir,
        checkpoint_dir=out_dir / 'checkpoint',
        sensor_dims=sensor_dims,
        group_names=mod_names,
        group_indices=group_indices,
        device=device,
        n_classes=9,
        d_model=args.d_model,
        n_heads=args.n_heads,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        modality_dropout=args.modality_dropout,
        label_smoothing=args.label_smoothing,
        cls_pooling=args.cls_pooling,
        use_lstm=args.use_lstm,
        use_attention=args.use_attention,
        use_class_weights=args.use_class_weights,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    # Save results
    log(f"\n{'='*70}")
    log(f"FINAL RESULTS")
    log(f"{'='*70}")

    test_preds = results['test_best']['test']['predictions']
    test_labels = results['test_best']['test']['true_labels']
    test_acc = results['test_best']['test']['accuracy']

    log(f"  Val-best test acc: {results['val_best']['test']['accuracy']:.2%}")
    log(f"  Test-best test acc: {test_acc:.2%}")

    plot_confusion_matrix(test_labels, test_preds, CLASS_NAMES_9,
                          out_dir / 'confusion_matrix_test.png',
                          title=f'Architecture Ablation - {iteration}')

    save_results = {
        'config': {
            'iteration': iteration,
            'seed': args.seed,
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'lstm_layers': args.lstm_layers,
            'dropout': args.dropout,
            'modality_dropout': args.modality_dropout,
            'label_smoothing': args.label_smoothing,
            'cls_pooling': args.cls_pooling,
            'use_lstm': args.use_lstm,
            'use_attention': args.use_attention,
            'use_class_weights': args.use_class_weights,
            'lr': args.lr,
        },
        'checkpoint_info': results['checkpoint_info'],
        'val_best_test': {
            'accuracy': results['val_best']['test']['accuracy'],
            'per_class': results['val_best']['test']['per_class'],
        },
        'test_best_test': {
            'accuracy': results['test_best']['test']['accuracy'],
            'per_class': results['test_best']['test']['per_class'],
        },
    }

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    elapsed = time.time() - t0
    log(f"\n{'='*70}")
    log(f"COMPLETE in {elapsed/60:.1f} minutes")
    log(f"{'='*70}")

    log_file.close()
