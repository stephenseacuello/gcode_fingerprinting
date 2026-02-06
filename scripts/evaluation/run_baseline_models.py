#!/usr/bin/env python3
"""Baseline Model Comparisons for 9-Class Classification.

Implements traditional ML and neural network baselines:

Traditional ML (on flattened features):
- XGBoost
- RandomForest
- SVM (RBF kernel)
- LogisticRegression (L2)
- KNN

Neural Network:
- MLP (3-layer)
- CNN_1D (1D convolutions over time)
- LSTM_simple (basic 2-layer LSTM)
- Transformer (encoder only)
- TCN (Temporal Convolutional Network)

Usage:
    # XGBoost baseline
    python scripts/evaluation/run_baseline_models.py \
        --data-dir outputs/ablation_study/data/split_42 \
        --output-dir outputs/ablation_study/baseline/ml/xgboost_ds42_ms42 \
        --model xgboost \
        --seed 42

    # Simple LSTM baseline
    python scripts/evaluation/run_baseline_models.py \
        --data-dir outputs/ablation_study/data/split_42 \
        --output-dir outputs/ablation_study/baseline/nn/lstm_simple_ds42_ms42 \
        --model lstm_simple \
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from collections import Counter

# ML imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# ── Constants ─────────────────────────────────────────────────────────────────

MACHINE_CONTROLLER_FEATURES = {
    'coor', 'dist', 'feed', 'gcode_line', 'line', 'momo',
    'mpox', 'mpoy', 'mpoz', 'plane', 'posx', 'posy', 'posz',
    'stat', 'unit', 'vel',
}

CLASS_NAMES_9 = ['adaptive', 'adaptive150025', 'face', 'face150025',
                 'pocket', 'pocket150025', 'damageadaptive', 'damageface', 'damagepocket']

ML_MODELS = ['xgboost', 'random_forest', 'svm_rbf', 'logistic_regression', 'knn']
NN_MODELS = ['mlp', 'cnn_1d', 'lstm_simple', 'transformer', 'tcn']
ALL_MODELS = ML_MODELS + NN_MODELS

# ── Logging ───────────────────────────────────────────────────────────────────

log_file = None

def log(msg):
    print(msg, flush=True)
    if log_file is not None:
        log_file.write(msg + '\n')
        log_file.flush()


# ── Neural Network Baselines ──────────────────────────────────────────────────

class MLP(nn.Module):
    """3-layer MLP on flattened features."""
    def __init__(self, input_dim, n_classes=9, hidden_dims=(512, 256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)

    def forward(self, x):
        # x: (B, T, F) -> flatten to (B, T*F)
        B = x.size(0)
        x = x.view(B, -1)
        feat = self.features(x)
        return self.classifier(feat)


class CNN_1D(nn.Module):
    """1D CNN over time dimension."""
    def __init__(self, n_features, n_classes=9, hidden_channels=128, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels*2)
        self.conv3 = nn.Conv1d(hidden_channels*2, hidden_channels*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_channels*4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels*4, n_classes)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T) for Conv1d
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2) if x.size(2) >= 2 else x
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2) if x.size(2) >= 2 else x
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.classifier(x)


class LSTM_Simple(nn.Module):
    """Basic 2-layer LSTM without attention."""
    def __init__(self, n_features, n_classes=9, hidden_size=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=n_layers,
                           batch_first=True, dropout=dropout if n_layers > 1 else 0,
                           bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # x: (B, T, F)
        out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        feat = out[:, -1, :]
        feat = self.dropout(feat)
        return self.classifier(feat)


class TransformerEncoder(nn.Module):
    """Standard transformer encoder."""
    def __init__(self, n_features, n_classes=9, d_model=256, n_heads=4, n_layers=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding
        max_len = 1000
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (B, T, F)
        T = x.size(1)
        x = self.input_proj(x)
        x = x + self.pe[:, :T, :]
        x = self.transformer(x)
        # Mean pooling
        feat = x.mean(dim=1)
        feat = self.dropout(feat)
        return self.classifier(feat)


class TCN(nn.Module):
    """Temporal Convolutional Network with dilated convolutions."""
    def __init__(self, n_features, n_classes=9, n_channels=256, kernel_size=3, n_layers=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Conv1d(n_features, n_channels, 1)

        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            layers.append(nn.Conv1d(n_channels, n_channels, kernel_size,
                                   padding=padding, dilation=dilation))
            layers.append(nn.BatchNorm1d(n_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.tcn = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


# ── Data preparation ─────────────────────────────────────────────────────────

def load_data(source_dir):
    """Load pre-processed data, remove machine controller columns."""
    source_dir = Path(source_dir)

    with open(source_dir / 'metadata.json') as f:
        orig_meta = json.load(f)
    with open(source_dir / 'scaler_stats.json') as f:
        scaler_stats = json.load(f)

    orig_columns = orig_meta['continuous_columns']
    orig_mean = np.array(scaler_stats['mean'])
    orig_std = np.array(scaler_stats['scale'])

    # Remove machine controller features
    keep_indices = [i for i, c in enumerate(orig_columns) if c not in MACHINE_CONTROLLER_FEATURES]
    keep_columns = [orig_columns[i] for i in keep_indices]

    log(f"  Original features: {len(orig_columns)}")
    log(f"  Kept features: {len(keep_columns)}")

    keep_mean = orig_mean[keep_indices]
    keep_std = orig_std[keep_indices]

    # Load splits
    data = {}
    labels = {}

    for split in ['train', 'val', 'test']:
        npz_path = source_dir / f'{split}_sequences.npz'
        npz = np.load(npz_path, allow_pickle=True)
        scaled = npz['continuous'][:, :, keep_indices]

        # Inverse transform
        raw = scaled * keep_std + keep_mean
        data[split] = raw
        labels[split] = npz['operation_type']

        log(f"  {split}: {raw.shape}")

    # Re-fit StandardScaler on training data
    train_raw = data['train']
    N, T, F = train_raw.shape
    train_flat = train_raw.reshape(-1, F)

    scaler = StandardScaler()
    scaler.fit(train_flat)

    # Apply to all splits
    for split in ['train', 'val', 'test']:
        raw = data[split]
        N, T, F = raw.shape
        flat = raw.reshape(-1, F)
        scaled = scaler.transform(flat)
        data[split] = scaled.reshape(N, T, F)

    return data, labels, keep_columns


# ── Training functions ───────────────────────────────────────────────────────

def train_ml_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test, seed=42):
    """Train traditional ML model on flattened features."""
    # Flatten: (N, T, F) -> (N, T*F)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    log(f"  Flattened shape: {X_train_flat.shape}")

    t0 = time.time()

    if model_name == 'xgboost':
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)],
                 verbose=False)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=seed,
            n_jobs=-1
        )
        model.fit(X_train_flat, y_train)
    elif model_name == 'svm_rbf':
        model = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            random_state=seed,
            probability=True
        )
        model.fit(X_train_flat, y_train)
    elif model_name == 'logistic_regression':
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=seed,
            n_jobs=-1
        )
        model.fit(X_train_flat, y_train)
    elif model_name == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        )
        model.fit(X_train_flat, y_train)
    else:
        raise ValueError(f"Unknown ML model: {model_name}")

    train_time = time.time() - t0
    log(f"  Training time: {train_time:.1f}s")

    # Evaluate
    results = {}
    for name, X, y in [('train', X_train_flat, y_train),
                        ('val', X_val_flat, y_val),
                        ('test', X_test_flat, y_test)]:
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        per_class_acc = {}
        for c in range(9):
            mask = y == c
            if mask.sum() > 0:
                per_class_acc[CLASS_NAMES_9[c]] = float((preds[mask] == c).mean())
            else:
                per_class_acc[CLASS_NAMES_9[c]] = 0.0

        results[name] = {
            'accuracy': float(acc),
            'per_class': per_class_acc,
            'predictions': preds,
            'true_labels': y,
        }
        log(f"  {name}: {acc:.2%}")

    return model, results, train_time


def train_nn_model(model_name, data, labels, device, seed=42, max_epochs=200, patience=40,
                   lr=1e-3, batch_size=32, label_smoothing=0.1):
    """Train neural network baseline."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_val, X_test = data['train'], data['val'], data['test']
    y_train, y_val, y_test = labels['train'], labels['val'], labels['test']

    n_features = X_train.shape[2]
    T = X_train.shape[1]

    # Create datasets
    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64))
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.int64))
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test.astype(np.float32)),
        torch.from_numpy(y_test.astype(np.int64))
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Create model
    if model_name == 'mlp':
        model = MLP(T * n_features, n_classes=9)
    elif model_name == 'cnn_1d':
        model = CNN_1D(n_features, n_classes=9)
    elif model_name == 'lstm_simple':
        model = LSTM_Simple(n_features, n_classes=9)
    elif model_name == 'transformer':
        model = TransformerEncoder(n_features, n_classes=9)
    elif model_name == 'tcn':
        model = TCN(n_features, n_classes=9)
    else:
        raise ValueError(f"Unknown NN model: {model_name}")

    model = model.to(device)
    log(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    op_counts = Counter(y_train)
    counts = np.array([op_counts.get(c, 1) for c in range(9)], dtype=np.float32)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    weights = torch.tensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    def evaluate(model, loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = (all_preds == all_labels).mean()
        return acc, all_preds, all_labels

    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    no_improve = 0

    history = {'epoch': [], 'train_acc': [], 'val_acc': [], 'test_acc': []}

    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = epoch_correct = epoch_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = out.argmax(dim=-1)
            epoch_correct += (preds == y).sum().item()
            epoch_total += y.size(0)
            epoch_loss += loss.item() * y.size(0)

        scheduler.step()
        train_acc = epoch_correct / epoch_total

        val_acc, _, _ = evaluate(model, val_loader)
        test_acc, _, _ = evaluate(model, test_loader)

        history['epoch'].append(epoch)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        improved = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            improved = ' ** BEST **'
        else:
            no_improve += 1

        if epoch <= 5 or epoch % 20 == 0 or improved:
            log(f"  Epoch {epoch:3d}/{max_epochs} | train={train_acc:.2%} val={val_acc:.2%} test={test_acc:.2%}{improved}")

        if no_improve >= patience:
            log(f"  Early stopping at epoch {epoch}")
            break

    train_time = time.time() - t0
    log(f"  Training time: {train_time:.1f}s")
    log(f"  Best val epoch: {best_epoch}, val_acc={best_val_acc:.2%}")

    # Load best model and evaluate
    model.load_state_dict(best_model_state)

    results = {}
    for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        acc, preds, true_labels = evaluate(model, loader)
        per_class_acc = {}
        for c in range(9):
            mask = true_labels == c
            if mask.sum() > 0:
                per_class_acc[CLASS_NAMES_9[c]] = float((preds[mask] == c).mean())
            else:
                per_class_acc[CLASS_NAMES_9[c]] = 0.0

        results[name] = {
            'accuracy': float(acc),
            'per_class': per_class_acc,
            'predictions': preds,
            'true_labels': true_labels,
        }

    log(f"  Final: train={results['train']['accuracy']:.2%} "
        f"val={results['val']['accuracy']:.2%} test={results['test']['accuracy']:.2%}")

    results['checkpoint_info'] = {
        'val_best_epoch': best_epoch,
        'val_best_val_acc': float(best_val_acc),
    }
    results['history'] = history

    return model, results, train_time


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
    parser = argparse.ArgumentParser(description='Baseline Model Comparisons')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--model', type=str, required=True, choices=ALL_MODELS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / 'pipeline_log.txt', 'w')

    t0 = time.time()
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')

    log(f"Baseline Model: {args.model}")
    log(f"{'='*70}")
    log(f"Device: {device}")
    log(f"Data dir: {args.data_dir}")
    log(f"Output: {out_dir}")
    log(f"Seed: {args.seed}")

    # Load data
    log(f"\n{'='*70}")
    log(f"LOADING DATA")
    log(f"{'='*70}")

    data, labels, columns = load_data(args.data_dir)

    # Train model
    log(f"\n{'='*70}")
    log(f"TRAINING {args.model.upper()}")
    log(f"{'='*70}")

    if args.model in ML_MODELS:
        model, results, train_time = train_ml_model(
            args.model,
            data['train'], labels['train'],
            data['val'], labels['val'],
            data['test'], labels['test'],
            seed=args.seed
        )
        model_type = 'ml'
    else:
        model, results, train_time = train_nn_model(
            args.model, data, labels, device,
            seed=args.seed,
            max_epochs=args.max_epochs,
            patience=args.patience,
            lr=args.lr,
            batch_size=args.batch_size
        )
        model_type = 'nn'

    # Save results
    log(f"\n{'='*70}")
    log(f"FINAL RESULTS")
    log(f"{'='*70}")

    test_preds = results['test']['predictions']
    test_labels = results['test']['true_labels']
    test_acc = results['test']['accuracy']

    log(f"  Test accuracy: {test_acc:.2%}")

    # Per-class report
    for cls_name, acc in results['test']['per_class'].items():
        log(f"    {cls_name}: {acc:.2%}")

    # Confusion matrix
    plot_confusion_matrix(test_labels, test_preds, CLASS_NAMES_9,
                          out_dir / 'confusion_matrix_test.png',
                          title=f'{args.model} - Test Set')

    # Save results JSON
    save_results = {
        'config': {
            'model': args.model,
            'model_type': model_type,
            'seed': args.seed,
        },
        'train_time_seconds': train_time,
        'train': {
            'accuracy': results['train']['accuracy'],
            'per_class': results['train']['per_class'],
        },
        'val': {
            'accuracy': results['val']['accuracy'],
            'per_class': results['val']['per_class'],
        },
        'test': {
            'accuracy': results['test']['accuracy'],
            'per_class': results['test']['per_class'],
        },
    }

    if 'checkpoint_info' in results:
        save_results['checkpoint_info'] = results['checkpoint_info']

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    elapsed = time.time() - t0
    log(f"\n{'='*70}")
    log(f"COMPLETE in {elapsed/60:.1f} minutes")
    log(f"{'='*70}")

    log_file.close()
