#!/usr/bin/env python3
"""Evaluate post-hoc pipeline on the mean-pooled cls v2 encoder embeddings.

Tests if the new encoder (trained with cls_pooling=mean) produces better
per-modality embeddings for the damage router pipeline.
"""
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn
from frozen_mlp_damage_heads import DamageMLPHead, train_mlp_head

MODALITY_NAMES = ['accelerometer', 'gyroscope', 'magnetometer', 'environmental',
                  'color', 'rms', 'machine']
CLASS_NAMES = ['adaptive', 'adaptive150025', 'face', 'face150025',
               'pocket', 'pocket150025', 'damageadaptive', 'damageface', 'damagepocket']
MACHINE_MOD_IDX = 6
GYRO_IDX = 1
MAG_IDX = 2


def build_modality_indices(metadata_path):
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
    group_names = list(sensor_patterns.keys()) + ['machine']
    return group_names, groups, [len(groups[n]) for n in group_names]


def extract_embeddings(model, loader, group_indices, device):
    model.eval()
    all_last, all_mean, all_cls_preds, all_ops = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type']
            B, T = sensor_data.size(0), sensor_data.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
            mods = [sensor_data[:, :, idx] for idx in group_indices]
            out = model(mods, lengths)
            enc_mods = out['encoded_mods']
            last_idx = (lengths.clamp(min=1) - 1).view(-1)
            all_last.append(enc_mods[torch.arange(B, device=device), last_idx].cpu())
            all_mean.append(enc_mods.mean(dim=1).cpu())
            all_cls_preds.append(out['cls'].argmax(dim=-1).cpu())
            all_ops.append(operations)
    return {
        'last': torch.cat(all_last).numpy(),
        'mean': torch.cat(all_mean).numpy(),
        'cls_preds': torch.cat(all_cls_preds).numpy(),
        'ops': torch.cat(all_ops).numpy(),
    }


def map_4class(ops):
    y = np.zeros(len(ops), dtype=int)
    y[ops == 6] = 1
    y[ops == 7] = 2
    y[ops == 8] = 3
    return y


def run_pipeline(test_ops, test_cls_preds, router_probs, data, clf_c7, clf_c8, mt, pool='last'):
    N = len(test_ops)
    preds = np.zeros(N, dtype=int)
    for i in range(N):
        cls_pred = test_cls_preds[i]
        rp = router_probs[i]
        max_dc = np.argmax(rp[1:]) + 1
        max_dp = rp[max_dc]
        if max_dp > mt:
            if max_dc == 1:
                preds[i] = 6
            elif max_dc == 2:
                x_g = data['test'][pool][i, GYRO_IDX, :].reshape(1, -1)
                gp = clf_c7.predict_proba(x_g)[0, 1]
                preds[i] = 7 if gp > 0.5 else cls_pred
            elif max_dc == 3:
                x_m = data['test'][pool][i, MAG_IDX, :].reshape(1, -1)
                mp = clf_c8.predict_proba(x_m)[0, 1]
                preds[i] = 8 if mp > 0.5 else cls_pred
        else:
            preds[i] = cls_pred
    return preds


def report(test_ops, preds, label):
    N = len(test_ops)
    overall = (preds == test_ops).mean()
    c6 = (preds[test_ops == 6] == 6).mean() if (test_ops == 6).sum() > 0 else 0
    c7 = (preds[test_ops == 7] == 7).mean() if (test_ops == 7).sum() > 0 else 0
    c8 = (preds[test_ops == 8] == 8).mean() if (test_ops == 8).sum() > 0 else 0
    c05 = sum((preds[test_ops == c] == c).sum() for c in range(6)) / sum((test_ops == c).sum() for c in range(6))
    missed = sum(1 for i in range(N) if preds[i] != test_ops[i])
    print(f"  {label:50s} {overall:.2%}  c0-5={c05:.2%} c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}  miss={missed}")
    return overall


def main():
    split_dir = 'outputs/jan23_followup/no_leakage'

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    loaders = {}
    for split in ['train', 'val', 'test']:
        ds = DecoderDatasetFromSplits(split_dir=split_dir, split=split)
        loaders[split] = DataLoader(ds, batch_size=32, shuffle=False,
                                     collate_fn=decoder_collate_fn, num_workers=0)

    metadata_path = str(Path(split_dir) / 'metadata.json')
    mod_names, groups, sensor_dims = build_modality_indices(metadata_path)
    group_indices = [groups[name] for name in mod_names]

    # Test both v1 (last-step) and v2 (mean-pooled) models
    models = {
        'v1 (last-step cls)': 'outputs/jan30/mmdtae_standalone_v1/best_model.pt',
        'v2 (mean-pooled cls)': 'outputs/jan30/mmdtae_v1_mean_pooled/best_model.pt',
    }

    for model_name, model_path in models.items():
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        config = ckpt['config']
        model = MM_DTAE_LSTM(config)
        model.head_cls = nn.Linear(config.d_model, 9)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.to(device)
        model.eval()
        print(f"  Loaded, d_model={config.d_model}, cls_pooling={getattr(config, 'cls_pooling', 'last')}")

        print(f"\n  Extracting embeddings...")
        data = {}
        for split in ['train', 'val', 'test']:
            data[split] = extract_embeddings(model, loaders[split], group_indices, device)

        train_ops = data['train']['ops']
        test_ops = data['test']['ops']
        y_train_4c = map_4class(train_ops)
        y_test_4c = map_4class(test_ops)
        y_val_4c = map_4class(data['val']['ops'])

        counts_4c = np.bincount(y_train_4c)
        total = len(y_train_4c)
        cw_4c = [total / (4 * c) for c in counts_4c]

        # Cls head accuracy
        cls_acc = (data['test']['cls_preds'] == test_ops).mean()
        print(f"  Cls head test accuracy: {cls_acc:.2%}")
        for c in range(9):
            mask = test_ops == c
            if mask.sum() > 0:
                acc = (data['test']['cls_preds'][mask] == c).mean()
                print(f"    c{c}: {acc:.2%}")

        # Test both pooling strategies for the router
        for pool in ['last', 'mean']:
            print(f"\n  --- Router pooling: {pool} ---")

            X_tr = data['train'][pool][:, MACHINE_MOD_IDX, :]
            X_val = data['val'][pool][:, MACHINE_MOD_IDX, :]
            X_te = data['test'][pool][:, MACHINE_MOD_IDX, :]

            # LogReg C=10
            clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=10.0)
            clf.fit(X_tr, y_train_4c)
            probs_lr = clf.predict_proba(X_te)
            preds_lr = probs_lr.argmax(axis=1)
            from sklearn.metrics import accuracy_score
            acc_lr = accuracy_score(y_test_4c, preds_lr)
            per_cls_lr = {c: accuracy_score(y_test_4c[y_test_4c==c], preds_lr[y_test_4c==c])
                          for c in range(4) if (y_test_4c==c).sum()>0}
            print(f"    LogReg C=10 standalone: {acc_lr:.2%}  c6={per_cls_lr.get(1,0):.0%} c7={per_cls_lr.get(2,0):.0%} c8={per_cls_lr.get(3,0):.0%}")

            # MLP(64) - best of 3 seeds
            best_acc = 0
            best_probs = None
            for seed in [42, 123, 777]:
                torch.manual_seed(seed)
                np.random.seed(seed)
                mlp, _ = train_mlp_head(
                    X_tr, y_train_4c, X_val, y_val_4c,
                    n_classes=4, hidden_dims=(64,), dropout=0.2, lr=1e-3,
                    epochs=300, patience=50, class_weight=cw_4c, device='cpu')
                with torch.no_grad():
                    probs = torch.softmax(mlp(torch.tensor(X_te, dtype=torch.float32)), dim=1).numpy()
                acc = accuracy_score(y_test_4c, probs.argmax(axis=1))
                if acc > best_acc:
                    best_acc = acc
                    best_probs = probs
            per_cls_mlp = {c: accuracy_score(y_test_4c[y_test_4c==c], best_probs.argmax(axis=1)[y_test_4c==c])
                           for c in range(4) if (y_test_4c==c).sum()>0}
            print(f"    MLP(64) standalone:     {best_acc:.2%}  c6={per_cls_mlp.get(1,0):.0%} c7={per_cls_mlp.get(2,0):.0%} c8={per_cls_mlp.get(3,0):.0%}")

            # Specialists
            face_mask = np.isin(train_ops, [2, 3, 7])
            clf_c7 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
            clf_c7.fit(data['train'][pool][face_mask, GYRO_IDX, :],
                       (train_ops[face_mask] == 7).astype(int))

            pocket_mask = np.isin(train_ops, [4, 5, 8])
            clf_c8 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
            clf_c8.fit(data['train'][pool][pocket_mask, MAG_IDX, :],
                       (train_ops[pocket_mask] == 8).astype(int))

            # Pipeline
            for mt in [0.3, 0.4, 0.5, 0.55]:
                preds = run_pipeline(test_ops, data['test']['cls_preds'], best_probs, data,
                                     clf_c7, clf_c8, mt, pool)
                report(test_ops, preds, f"MLP(64) pipeline mt={mt:.2f}")


if __name__ == '__main__':
    main()
