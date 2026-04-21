#!/usr/bin/env python3
"""Post-hoc re-evaluation of baseline models to compute precision, recall, F1.

Since the original run_baseline_models.py did not save model checkpoints,
this script re-trains each model with the same seed (deterministic) and
computes full classification metrics via sklearn.

Outputs:
  - baseline_full_metrics.csv   (one row per model×seed)
  - baseline_summary.csv        (aggregated mean ± std per model)
  - Updates each results.json in-place with precision/recall/f1 fields

Usage:
    python scripts/evaluation/posthoc_baseline_metrics.py
"""
import sys
import json
import time
import numpy as np
import torch
import warnings
from pathlib import Path
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = ROOT / "outputs" / "ablation_study_2026_02_06"
DATA_SPLITS_DIR = ABLATION_DIR / "data_splits"
BASELINE_DIR = ABLATION_DIR / "baseline"

MACHINE_CONTROLLER_FEATURES = {
    'coor', 'dist', 'feed', 'gcode_line', 'line', 'momo',
    'mpox', 'mpoy', 'mpoz', 'plane', 'posx', 'posy', 'posz',
    'stat', 'unit', 'vel',
}

CLASS_NAMES_9 = [
    'adaptive', 'adaptive150025', 'face', 'face150025',
    'pocket', 'pocket150025', 'damageadaptive', 'damageface', 'damagepocket',
]

ML_MODELS = ['xgboost', 'random_forest', 'logistic_regression', 'knn']
NN_MODELS = ['mlp', 'cnn_1d', 'lstm_simple', 'transformer', 'tcn']

DATA_SEEDS = [42, 123, 456]
MODEL_SEEDS = [42, 123, 456]


# ── NN architectures (copied from run_baseline_models.py) ───────────────────

class MLP(nn.Module):
    def __init__(self, input_dim, n_classes=9, hidden_dims=(512, 256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim),
                nn.ReLU(), nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x.view(x.size(0), -1)))


class CNN_1D(nn.Module):
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
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2) if x.size(2) >= 2 else x
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2) if x.size(2) >= 2 else x
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        return self.classifier(self.dropout(x))


class LSTM_Simple(nn.Module):
    def __init__(self, n_features, n_classes=9, hidden_size=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=n_layers,
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(self.dropout(out[:, -1, :]))


class TransformerEncoder(nn.Module):
    def __init__(self, n_features, n_classes=9, d_model=256, n_heads=4, n_layers=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        pe = torch.zeros(1000, d_model)
        position = torch.arange(0, 1000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        T = x.size(1)
        x = self.input_proj(x) + self.pe[:, :T, :]
        x = self.transformer(x)
        return self.classifier(self.dropout(x.mean(dim=1)))


class TCN(nn.Module):
    def __init__(self, n_features, n_classes=9, n_channels=256, kernel_size=3, n_layers=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Conv1d(n_features, n_channels, 1)
        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            layers.extend([
                nn.Conv1d(n_channels, n_channels, kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(n_channels), nn.ReLU(), nn.Dropout(dropout),
            ])
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(self.input_proj(x))
        return self.classifier(self.pool(x).squeeze(-1))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(data_seed):
    """Load pre-processed data for a given data split seed."""
    source_dir = DATA_SPLITS_DIR / f"split_{data_seed}"
    with open(source_dir / 'metadata.json') as f:
        orig_meta = json.load(f)
    with open(source_dir / 'scaler_stats.json') as f:
        scaler_stats = json.load(f)

    orig_columns = orig_meta['continuous_columns']
    orig_mean = np.array(scaler_stats['mean'])
    orig_std = np.array(scaler_stats['scale'])
    keep_indices = [i for i, c in enumerate(orig_columns) if c not in MACHINE_CONTROLLER_FEATURES]
    keep_mean = orig_mean[keep_indices]
    keep_std = orig_std[keep_indices]

    data, labels = {}, {}
    for split in ['train', 'val', 'test']:
        npz = np.load(source_dir / f'{split}_sequences.npz', allow_pickle=True)
        scaled = npz['continuous'][:, :, keep_indices]
        data[split] = scaled * keep_std + keep_mean
        labels[split] = npz['operation_type']

    # Re-fit scaler on training data
    N, T, Feat = data['train'].shape
    scaler = StandardScaler()
    scaler.fit(data['train'].reshape(-1, Feat))
    for split in ['train', 'val', 'test']:
        N, T, Feat = data[split].shape
        data[split] = scaler.transform(data[split].reshape(-1, Feat)).reshape(N, T, Feat)

    return data, labels


# ── Training + evaluation ────────────────────────────────────────────────────

def train_and_evaluate_ml(model_name, data, labels, seed):
    """Train ML model, return test y_pred and y_true."""
    X_train = data['train'].reshape(data['train'].shape[0], -1)
    X_test = data['test'].reshape(data['test'].shape[0], -1)
    y_train, y_test = labels['train'], labels['test']

    if model_name == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=10, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            n_jobs=-1, eval_metric='mlogloss',
            tree_method='hist', device='cuda')
    elif model_name == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            random_state=seed, n_jobs=-1)
    elif model_name == 'logistic_regression':
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, n_jobs=-1)
    elif model_name == 'knn':
        model = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    else:
        raise ValueError(f"Unknown ML model: {model_name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, y_test


def train_and_evaluate_nn(model_name, data, labels, device, seed,
                          max_epochs=200, patience=40, lr=1e-3, batch_size=32):
    """Train NN model, return test y_pred and y_true."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_val, X_test = data['train'], data['val'], data['test']
    y_train, y_val, y_test = labels['train'], labels['val'], labels['test']
    n_features, T = X_train.shape[2], X_train.shape[1]

    train_ds = TensorDataset(torch.from_numpy(X_train.astype(np.float32)),
                             torch.from_numpy(y_train.astype(np.int64)))
    val_ds = TensorDataset(torch.from_numpy(X_val.astype(np.float32)),
                           torch.from_numpy(y_val.astype(np.int64)))
    test_ds = TensorDataset(torch.from_numpy(X_test.astype(np.float32)),
                            torch.from_numpy(y_test.astype(np.int64)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if model_name == 'mlp':
        model = MLP(T * n_features)
    elif model_name == 'cnn_1d':
        model = CNN_1D(n_features)
    elif model_name == 'lstm_simple':
        model = LSTM_Simple(n_features)
    elif model_name == 'transformer':
        model = TransformerEncoder(n_features)
    elif model_name == 'tcn':
        model = TCN(n_features)
    else:
        raise ValueError(f"Unknown NN model: {model_name}")

    model = model.to(device)

    op_counts = Counter(y_train)
    counts = np.array([op_counts.get(c, 1) for c in range(9)], dtype=np.float32)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    weights = torch.tensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    def evaluate(loader):
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                out = model(x)
                preds_all.extend(out.argmax(dim=-1).cpu().numpy())
                labels_all.extend(y.numpy())
        return np.array(preds_all), np.array(labels_all)

    best_val_acc, best_state, no_improve = 0.0, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_preds, val_labels = evaluate(val_loader)
        val_acc = (val_preds == val_labels).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    y_pred, y_true = evaluate(test_loader)
    return y_pred, y_true


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    all_models = ML_MODELS + NN_MODELS
    rows = []

    # Cache loaded data per data_seed
    data_cache = {}

    total = len(all_models) * len(DATA_SEEDS) * len(MODEL_SEEDS)
    done = 0

    for model_name in all_models:
        for ds in DATA_SEEDS:
            # Load data once per data seed
            if ds not in data_cache:
                print(f"Loading data split {ds}...")
                data_cache[ds] = load_data(ds)
            data, labels = data_cache[ds]

            for ms in MODEL_SEEDS:
                done += 1
                tag = f"{model_name}_ds{ds}_ms{ms}"
                print(f"[{done}/{total}] {tag} ...", end=" ", flush=True)

                t0 = time.time()
                if model_name in ML_MODELS:
                    y_pred, y_true = train_and_evaluate_ml(model_name, data, labels, seed=ms)
                else:
                    y_pred, y_true = train_and_evaluate_nn(model_name, data, labels, device, seed=ms)
                elapsed = time.time() - t0

                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
                rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

                print(f"Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  ({elapsed:.1f}s)")

                rows.append({
                    'model': model_name,
                    'model_type': 'ml' if model_name in ML_MODELS else 'nn',
                    'data_seed': ds,
                    'model_seed': ms,
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'train_time_s': elapsed,
                })

                # Update the existing results.json in-place
                if model_name in ML_MODELS:
                    result_dir = BASELINE_DIR / "ml" / tag
                else:
                    result_dir = BASELINE_DIR / "nn" / tag
                result_json = result_dir / "results.json"
                if result_json.exists():
                    with open(result_json) as f:
                        existing = json.load(f)
                    existing['test']['precision_macro'] = prec
                    existing['test']['recall_macro'] = rec
                    existing['test']['f1_macro'] = f1
                    # Per-class breakdown
                    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES_9,
                                                   output_dict=True, zero_division=0)
                    per_class_detail = {}
                    for cls_name in CLASS_NAMES_9:
                        if cls_name in report:
                            per_class_detail[cls_name] = {
                                'precision': report[cls_name]['precision'],
                                'recall': report[cls_name]['recall'],
                                'f1': report[cls_name]['f1-score'],
                                'support': report[cls_name]['support'],
                            }
                    existing['test']['per_class_detail'] = per_class_detail
                    existing['test']['confusion_matrix'] = confusion_matrix(
                        y_true, y_pred, labels=list(range(9))).tolist()
                    with open(result_json, 'w') as f:
                        json.dump(existing, f, indent=2)

    # ── Write CSV outputs ────────────────────────────────────────────────────
    import csv

    out_dir = ABLATION_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full per-run CSV
    full_csv = out_dir / "baseline_full_metrics.csv"
    with open(full_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {full_csv}")

    # Summary CSV (mean ± std per model)
    summary_csv = out_dir / "baseline_summary.csv"
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'type', 'n_runs',
                         'accuracy_mean', 'accuracy_std',
                         'precision_mean', 'precision_std',
                         'recall_mean', 'recall_std',
                         'f1_mean', 'f1_std'])
        for model_name in all_models:
            model_rows = [r for r in rows if r['model'] == model_name]
            n = len(model_rows)
            mtype = model_rows[0]['model_type']
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                vals = [r[metric] for r in model_rows]
                if metric == 'accuracy':
                    acc_mean, acc_std = np.mean(vals), np.std(vals)
                elif metric == 'precision':
                    prec_mean, prec_std = np.mean(vals), np.std(vals)
                elif metric == 'recall':
                    rec_mean, rec_std = np.mean(vals), np.std(vals)
                elif metric == 'f1':
                    f1_mean, f1_std = np.mean(vals), np.std(vals)
            writer.writerow([model_name, mtype, n,
                             f"{acc_mean:.6f}", f"{acc_std:.6f}",
                             f"{prec_mean:.6f}", f"{prec_std:.6f}",
                             f"{rec_mean:.6f}", f"{rec_std:.6f}",
                             f"{f1_mean:.6f}", f"{f1_std:.6f}"])
    print(f"Wrote {summary_csv}")

    # ── Print summary table ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Model':<22} {'Type':<5} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print(f"{'='*80}")
    for model_name in all_models:
        model_rows = [r for r in rows if r['model'] == model_name]
        mtype = model_rows[0]['model_type']
        acc_vals = [r['accuracy'] for r in model_rows]
        prec_vals = [r['precision'] for r in model_rows]
        rec_vals = [r['recall'] for r in model_rows]
        f1_vals = [r['f1'] for r in model_rows]
        print(f"{model_name:<22} {mtype:<5} "
              f"{np.mean(acc_vals)*100:>5.2f}±{np.std(acc_vals)*100:.2f}  "
              f"{np.mean(prec_vals)*100:>5.2f}±{np.std(prec_vals)*100:.2f}  "
              f"{np.mean(rec_vals)*100:>5.2f}±{np.std(rec_vals)*100:.2f}  "
              f"{np.mean(f1_vals)*100:>5.2f}±{np.std(f1_vals)*100:.2f}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
