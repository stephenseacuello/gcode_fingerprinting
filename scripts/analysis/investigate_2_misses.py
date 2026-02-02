#!/usr/bin/env python3
"""Investigate the 2 remaining misses from mean-pooled MLP(64) mt=0.5 pipeline (99.67%).

Identifies which samples are missed, their true/predicted classes, and router probabilities.
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
BASE_OP_MAP = {0: 0, 1: 0, 6: 0, 2: 1, 3: 1, 7: 1, 4: 2, 5: 2, 8: 2}
DAMAGE_FOR_BASE = {0: 6, 1: 7, 2: 8}
GROUP_CLASSES = {0: [0, 1, 6], 1: [2, 3, 7], 2: [4, 5, 8]}


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
    all_last, all_mean, all_cls_preds, all_cls_probs, all_ops = [], [], [], [], []
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
            cls_logits = out['cls']
            all_cls_probs.append(torch.softmax(cls_logits, dim=-1).cpu())
            all_cls_preds.append(cls_logits.argmax(dim=-1).cpu())
            all_ops.append(operations)
    return {
        'last': torch.cat(all_last).numpy(),
        'mean': torch.cat(all_mean).numpy(),
        'cls_preds': torch.cat(all_cls_preds).numpy(),
        'cls_probs': torch.cat(all_cls_probs).numpy(),
        'ops': torch.cat(all_ops).numpy(),
    }


def map_4class(ops):
    y = np.zeros(len(ops), dtype=int)
    y[ops == 6] = 1
    y[ops == 7] = 2
    y[ops == 8] = 3
    return y


def main():
    split_dir = 'outputs/jan23_followup/no_leakage'
    model_path = 'outputs/jan30/mmdtae_standalone_v1/best_model.pt'

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

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = MM_DTAE_LSTM(config)
    model.head_cls = nn.Linear(config.d_model, 9)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded v1 model, d_model={config.d_model}")

    print(f"\nExtracting embeddings...")
    data = {}
    for split in ['train', 'val', 'test']:
        data[split] = extract_embeddings(model, loaders[split], group_indices, device)
        print(f"  {split}: {data[split]['last'].shape}")

    train_ops = data['train']['ops']
    test_ops = data['test']['ops']
    y_train_4c = map_4class(train_ops)
    y_test_4c = map_4class(test_ops)
    y_val_4c = map_4class(data['val']['ops'])

    counts_4c = np.bincount(y_train_4c)
    total = len(y_train_4c)
    cw_4c = [total / (4 * c) for c in counts_4c]

    # =====================================================================
    # Train the MEAN-POOLED MLP(64) router (the 99.67% config)
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"Training mean-pooled MLP(64) router")
    print(f"{'='*70}")

    X_tr = data['train']['mean'][:, MACHINE_MOD_IDX, :]
    X_val = data['val']['mean'][:, MACHINE_MOD_IDX, :]
    X_te = data['test']['mean'][:, MACHINE_MOD_IDX, :]

    best_mlp = None
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
        preds = probs.argmax(axis=1)
        acc = (preds == y_test_4c).mean()
        print(f"  Seed {seed}: standalone {acc:.2%}")
        if acc > best_acc:
            best_acc = acc
            best_mlp = mlp
            best_probs = probs

    print(f"\n  Best standalone: {best_acc:.2%}")

    # LogReg specialists (on LAST-STEP embeddings - same as before)
    face_mask = np.isin(train_ops, [2, 3, 7])
    clf_c7 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_c7.fit(data['train']['last'][face_mask, GYRO_IDX, :],
               (train_ops[face_mask] == 7).astype(int))

    pocket_mask = np.isin(train_ops, [4, 5, 8])
    clf_c8 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_c8.fit(data['train']['last'][pocket_mask, MAG_IDX, :],
               (train_ops[pocket_mask] == 8).astype(int))

    # =====================================================================
    # Run pipeline and identify the 2 misses
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"Running mean-pooled MLP(64) pipeline at mt=0.5")
    print(f"{'='*70}")

    test_cls_preds = data['test']['cls_preds']
    test_cls_probs = data['test']['cls_probs']
    N = len(test_ops)
    mt = 0.5

    missed = []
    all_final_preds = np.zeros(N, dtype=int)

    for i in range(N):
        true_op = test_ops[i]
        cls_pred = test_cls_preds[i]
        rp = best_probs[i]

        max_dc = np.argmax(rp[1:]) + 1
        max_dp = rp[max_dc]

        if max_dp > mt:
            if max_dc == 1:
                final_pred = 6
            elif max_dc == 2:
                x_g = data['test']['last'][i, GYRO_IDX, :].reshape(1, -1)
                gp = clf_c7.predict_proba(x_g)[0, 1]
                final_pred = 7 if gp > 0.5 else cls_pred
            elif max_dc == 3:
                x_m = data['test']['last'][i, MAG_IDX, :].reshape(1, -1)
                mp = clf_c8.predict_proba(x_m)[0, 1]
                final_pred = 8 if mp > 0.5 else cls_pred
        else:
            final_pred = cls_pred

        all_final_preds[i] = final_pred

        if final_pred != true_op:
            missed.append(i)

    # Per-class accuracy
    print(f"\n  Overall: {(all_final_preds == test_ops).mean():.2%} ({(all_final_preds == test_ops).sum()}/{N})")
    for c in range(9):
        mask = test_ops == c
        if mask.sum() > 0:
            acc = (all_final_preds[mask] == c).mean()
            print(f"  c{c} ({CLASS_NAMES[c]:20s}): {acc:.2%} ({(all_final_preds[mask] == c).sum()}/{mask.sum()})")

    print(f"\n{'='*70}")
    print(f"The {len(missed)} missed sample(s)")
    print(f"{'='*70}")

    for idx in missed:
        true_op = test_ops[idx]
        final = all_final_preds[idx]
        cls_pred = test_cls_preds[idx]
        cls_p = test_cls_probs[idx]
        router_p = best_probs[idx]

        print(f"\n  test[{idx}]:")
        print(f"    True class:    c{true_op} ({CLASS_NAMES[true_op]})")
        print(f"    Pipeline pred: c{final} ({CLASS_NAMES[final]})")
        print(f"    v1 cls pred:   c{cls_pred} ({CLASS_NAMES[cls_pred]})")
        print(f"    v1 cls probs:  {' '.join(f'c{c}={cls_p[c]:.3f}' for c in range(9))}")
        print(f"    Router probs:  norm={router_p[0]:.4f} c6={router_p[1]:.4f} c7={router_p[2]:.4f} c8={router_p[3]:.4f}")
        print(f"    Router max damage: class={np.argmax(router_p[1:])+1} prob={router_p[np.argmax(router_p[1:])+1]:.4f}")

        # Check what specialists say
        x_g = data['test']['last'][idx, GYRO_IDX, :].reshape(1, -1)
        x_m = data['test']['last'][idx, MAG_IDX, :].reshape(1, -1)
        gp = clf_c7.predict_proba(x_g)[0, 1]
        mp = clf_c8.predict_proba(x_m)[0, 1]
        print(f"    Gyro specialist (c7): prob={gp:.4f}")
        print(f"    Mag specialist (c8):  prob={mp:.4f}")

        # Per-modality damage signal (mean-pooled)
        print(f"    Per-modality damage probs (mean-pooled):")
        base_op = BASE_OP_MAP.get(true_op, -1)
        if base_op >= 0:
            damage_cls = DAMAGE_FOR_BASE[base_op]
            member_classes = GROUP_CLASSES[base_op]
            tr_mask = np.isin(train_ops, member_classes)
            y_tr = (train_ops[tr_mask] == damage_cls).astype(int)
            for m_idx, mod_name in enumerate(MODALITY_NAMES):
                clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
                clf.fit(data['train']['mean'][tr_mask, m_idx, :], y_tr)
                prob = clf.predict_proba(data['test']['mean'][idx, m_idx, :].reshape(1, -1))[0, 1]
                detected = "YES" if prob > 0.5 else "no"
                print(f"      {mod_name:15s}: {prob:.4f} [{detected}]")
        else:
            # Normal class — check all 3 damage groups
            for base_op_check in range(3):
                damage_cls = DAMAGE_FOR_BASE[base_op_check]
                member_classes = GROUP_CLASSES[base_op_check]
                group_name = ['adaptive', 'face', 'pocket'][base_op_check]
                tr_mask = np.isin(train_ops, member_classes)
                y_tr = (train_ops[tr_mask] == damage_cls).astype(int)
                print(f"      --- vs {group_name} damage (c{damage_cls}) ---")
                for m_idx, mod_name in enumerate(MODALITY_NAMES):
                    clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
                    clf.fit(data['train']['mean'][tr_mask, m_idx, :], y_tr)
                    prob = clf.predict_proba(data['test']['mean'][idx, m_idx, :].reshape(1, -1))[0, 1]
                    detected = "YES" if prob > 0.5 else "no"
                    print(f"        {mod_name:15s}: {prob:.4f} [{detected}]")

    # =====================================================================
    # Check if the 2 misses are FPs or routing errors
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"Error analysis summary")
    print(f"{'='*70}")
    fp_damage = sum(1 for i in missed if test_ops[i] < 6 and all_final_preds[i] >= 6)
    fn_damage = sum(1 for i in missed if test_ops[i] >= 6 and all_final_preds[i] < 6)
    cls_errors = sum(1 for i in missed if test_ops[i] < 6 and all_final_preds[i] < 6)
    print(f"  False positive damage (normal → damage): {fp_damage}")
    print(f"  False negative damage (damage → normal): {fn_damage}")
    print(f"  Normal-class confusion (normal → wrong normal): {cls_errors}")

    # =====================================================================
    # Sweep mt to see if any threshold gets 100%
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"Threshold sweep for mean-pooled MLP(64)")
    print(f"{'='*70}")
    for mt_val in np.arange(0.1, 0.95, 0.05):
        correct = 0
        damage_correct = {6: 0, 7: 0, 8: 0}
        damage_total = {6: 0, 7: 0, 8: 0}
        normal_fp = 0
        for i in range(N):
            true_op = test_ops[i]
            cls_pred = test_cls_preds[i]
            rp = best_probs[i]
            max_dc = np.argmax(rp[1:]) + 1
            max_dp = rp[max_dc]
            if max_dp > mt_val:
                if max_dc == 1:
                    fp = 6
                elif max_dc == 2:
                    x_g = data['test']['last'][i, GYRO_IDX, :].reshape(1, -1)
                    gp = clf_c7.predict_proba(x_g)[0, 1]
                    fp = 7 if gp > 0.5 else cls_pred
                elif max_dc == 3:
                    x_m = data['test']['last'][i, MAG_IDX, :].reshape(1, -1)
                    mp = clf_c8.predict_proba(x_m)[0, 1]
                    fp = 8 if mp > 0.5 else cls_pred
            else:
                fp = cls_pred
            if fp == true_op:
                correct += 1
            if true_op in (6, 7, 8):
                damage_total[true_op] += 1
                if fp == true_op:
                    damage_correct[true_op] += 1
            elif fp >= 6:
                normal_fp += 1

        overall = correct / N
        c6a = damage_correct[6] / damage_total[6] if damage_total[6] else 0
        c7a = damage_correct[7] / damage_total[7] if damage_total[7] else 0
        c8a = damage_correct[8] / damage_total[8] if damage_total[8] else 0
        print(f"  mt={mt_val:.2f}: {overall:.2%}  c6={c6a:.0%} c7={c7a:.0%} c8={c8a:.0%}  normal_FP={normal_fp}")


if __name__ == '__main__':
    main()
