#!/usr/bin/env python3
"""Diagnose the 5 missed samples from the MLP(64) mt=0.5 pipeline.

For each missed sample:
- Which test index is it?
- What does the MLP router predict? (prob per class)
- What does each of the 7 modality embeddings look like through per-modality LogReg classifiers?
- Is there ANY modality that correctly flags this sample as damage?
- What does the raw sensor feature look like (time-averaged)?

Also tests: mean-pooled embeddings vs last-step, and combined (last+mean=512d) router.
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
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn

MODALITY_NAMES = ['accelerometer', 'gyroscope', 'magnetometer', 'environmental',
                  'color', 'rms', 'machine']
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


def extract_embeddings_multi_pool(model, loader, group_indices, device):
    """Extract both last-step and mean-pooled per-modality embeddings."""
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
            enc_mods = out['encoded_mods']  # [B, T, M, D]

            # Last step
            last_idx = (lengths.clamp(min=1) - 1).view(-1)
            per_mod_last = enc_mods[torch.arange(B, device=device), last_idx]  # [B, M, D]
            all_last.append(per_mod_last.cpu())

            # Mean pool over time
            per_mod_mean = enc_mods.mean(dim=1)  # [B, M, D]
            all_mean.append(per_mod_mean.cpu())

            all_cls_preds.append(out['cls'].argmax(dim=-1).cpu())
            all_ops.append(operations)

    return (torch.cat(all_last).numpy(), torch.cat(all_mean).numpy(),
            torch.cat(all_cls_preds).numpy(), torch.cat(all_ops).numpy())


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

    print(f"\nExtracting embeddings (last-step + mean-pooled)...")
    split_data = {}
    for split in ['train', 'val', 'test']:
        last_embeds, mean_embeds, cls_preds, ops = extract_embeddings_multi_pool(
            model, loaders[split], group_indices, device)
        split_data[split] = {
            'last': last_embeds, 'mean': mean_embeds,
            'cls_preds': cls_preds, 'ops': ops
        }
        print(f"  {split}: last={last_embeds.shape}, mean={mean_embeds.shape}")

    train_ops = split_data['train']['ops']
    test_ops = split_data['test']['ops']
    y_train_4c = map_4class(train_ops)
    y_test_4c = map_4class(test_ops)

    # =====================================================================
    # PART 1: Reproduce the MLP(64) pipeline to identify the 5 misses
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"PART 1: Identify the 5 missed samples")
    print(f"{'='*70}")

    from frozen_mlp_damage_heads import DamageMLPHead, train_mlp_head

    X_train_machine = split_data['train']['last'][:, MACHINE_MOD_IDX, :]
    X_val_machine = split_data['val']['last'][:, MACHINE_MOD_IDX, :]
    X_test_machine = split_data['test']['last'][:, MACHINE_MOD_IDX, :]

    counts_4c = np.bincount(y_train_4c)
    total = len(y_train_4c)
    cw_4c = [total / (4 * c) for c in counts_4c]
    y_val_4c = map_4class(split_data['val']['ops'])

    # Train MLP(64) router (best seed)
    best_mlp = None
    best_acc = 0
    for seed in [42, 123, 777]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        mlp, _ = train_mlp_head(
            X_train_machine, y_train_4c, X_val_machine, y_val_4c,
            n_classes=4, hidden_dims=(64,), dropout=0.2, lr=1e-3,
            epochs=300, patience=50, class_weight=cw_4c, device='cpu')
        with torch.no_grad():
            probs = torch.softmax(mlp(torch.tensor(X_test_machine, dtype=torch.float32)), dim=1).numpy()
        preds = probs.argmax(axis=1)
        acc = accuracy_score(y_test_4c, preds)
        if acc > best_acc:
            best_acc = acc
            best_mlp = mlp
            best_probs = probs

    print(f"  MLP(64) standalone: {best_acc:.2%}")

    # LogReg specialists
    face_mask = np.isin(train_ops, [2, 3, 7])
    clf_c7 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_c7.fit(split_data['train']['last'][face_mask, GYRO_IDX, :],
               (train_ops[face_mask] == 7).astype(int))

    pocket_mask = np.isin(train_ops, [4, 5, 8])
    clf_c8 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_c8.fit(split_data['train']['last'][pocket_mask, MAG_IDX, :],
               (train_ops[pocket_mask] == 8).astype(int))

    # Run pipeline at mt=0.5
    mt = 0.5
    test_cls_preds = split_data['test']['cls_preds']
    N = len(test_ops)
    missed_indices = []

    for i in range(N):
        true_op = test_ops[i]
        cls_pred = test_cls_preds[i]
        machine_probs = best_probs[i]

        max_damage_class = np.argmax(machine_probs[1:]) + 1
        max_damage_prob = machine_probs[max_damage_class]

        if max_damage_prob > mt:
            if max_damage_class == 1:
                final_pred = 6
            elif max_damage_class == 2:
                x_gyro = split_data['test']['last'][i, GYRO_IDX, :].reshape(1, -1)
                gyro_prob = clf_c7.predict_proba(x_gyro)[0, 1]
                final_pred = 7 if gyro_prob > 0.5 else cls_pred
            elif max_damage_class == 3:
                x_mag = split_data['test']['last'][i, MAG_IDX, :].reshape(1, -1)
                mag_prob = clf_c8.predict_proba(x_mag)[0, 1]
                final_pred = 8 if mag_prob > 0.5 else cls_pred
        else:
            final_pred = cls_pred

        if final_pred != true_op and true_op in (6, 7, 8):
            missed_indices.append(i)

    print(f"\n  Missed {len(missed_indices)} damage samples at mt={mt}:")
    for idx in missed_indices:
        true_op = test_ops[idx]
        print(f"    test[{idx}]: true=c{true_op}, cls_pred=c{test_cls_preds[idx]}, "
              f"router_probs={best_probs[idx].round(4)}")

    # =====================================================================
    # PART 2: Per-modality analysis of missed samples
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"PART 2: Per-modality damage signal for missed samples")
    print(f"{'='*70}")

    # Train per-modality binary damage classifiers (within-group)
    for pool_type in ['last', 'mean']:
        print(f"\n--- Pooling: {pool_type} ---")

        for base_op in range(3):
            damage_cls = DAMAGE_FOR_BASE[base_op]
            member_classes = GROUP_CLASSES[base_op]
            group_name = ['adaptive', 'face', 'pocket'][base_op]

            tr_mask = np.isin(train_ops, member_classes)
            y_tr = (train_ops[tr_mask] == damage_cls).astype(int)

            missed_in_group = [idx for idx in missed_indices if test_ops[idx] == damage_cls]
            if not missed_in_group:
                continue

            print(f"\n  {group_name} (c{damage_cls}) — {len(missed_in_group)} missed samples:")

            for m_idx in range(7):
                mod_name = MODALITY_NAMES[m_idx]
                X_tr = split_data['train'][pool_type][tr_mask, m_idx, :]
                clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
                clf.fit(X_tr, y_tr)

                for idx in missed_in_group:
                    x = split_data['test'][pool_type][idx, m_idx, :].reshape(1, -1)
                    prob = clf.predict_proba(x)[0, 1]
                    detected = "YES" if prob > 0.5 else "no"
                    print(f"    test[{idx}] c{test_ops[idx]} | {mod_name:15s}: damage_prob={prob:.4f} [{detected}]")

    # =====================================================================
    # PART 3: Mean-pooled vs last-step vs combined router
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"PART 3: Alternative pooling strategies for the router")
    print(f"{'='*70}")

    for pool_label, get_features in [
        ('last-step (current)', lambda s, m: split_data[s]['last'][:, m, :]),
        ('mean-pooled', lambda s, m: split_data[s]['mean'][:, m, :]),
        ('last+mean concat', lambda s, m: np.hstack([
            split_data[s]['last'][:, m, :], split_data[s]['mean'][:, m, :]])),
    ]:
        print(f"\n--- {pool_label} ---")
        X_tr = get_features('train', MACHINE_MOD_IDX)
        X_val = get_features('val', MACHINE_MOD_IDX)
        X_te = get_features('test', MACHINE_MOD_IDX)

        # LogReg C=10
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=10.0)
        clf.fit(X_tr, y_train_4c)
        probs = clf.predict_proba(X_te)
        preds = probs.argmax(axis=1)
        acc = accuracy_score(y_test_4c, preds)
        per_cls = {c: accuracy_score(y_test_4c[y_test_4c == c], preds[y_test_4c == c])
                   for c in range(4) if (y_test_4c == c).sum() > 0}
        print(f"  LogReg C=10: {acc:.2%}  c6={per_cls.get(1,0):.0%} c7={per_cls.get(2,0):.0%} c8={per_cls.get(3,0):.0%}")

        # MLP(64)
        best_mlp_acc = 0
        best_mlp_probs = None
        for seed in [42, 123, 777]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            y_val = map_4class(split_data['val']['ops'])
            mlp, _ = train_mlp_head(
                X_tr, y_train_4c, X_val, y_val,
                n_classes=4, hidden_dims=(64,), dropout=0.2, lr=1e-3,
                epochs=300, patience=50, class_weight=cw_4c, device='cpu')
            with torch.no_grad():
                test_probs = torch.softmax(mlp(torch.tensor(X_te, dtype=torch.float32)), dim=1).numpy()
            acc = accuracy_score(y_test_4c, test_probs.argmax(axis=1))
            if acc > best_mlp_acc:
                best_mlp_acc = acc
                best_mlp_probs = test_probs

        per_cls = {c: accuracy_score(y_test_4c[y_test_4c == c], best_mlp_probs.argmax(axis=1)[y_test_4c == c])
                   for c in range(4) if (y_test_4c == c).sum() > 0}
        print(f"  MLP(64):    {best_mlp_acc:.2%}  c6={per_cls.get(1,0):.0%} c7={per_cls.get(2,0):.0%} c8={per_cls.get(3,0):.0%}")

        # Full pipeline
        for mt_val in [0.3, 0.4, 0.5]:
            pc_correct = Counter()
            pc_total = Counter()
            for i in range(N):
                true_op = test_ops[i]
                pc_total[true_op] += 1
                cls_pred = test_cls_preds[i]
                rp = best_mlp_probs[i]

                max_dc = np.argmax(rp[1:]) + 1
                max_dp = rp[max_dc]

                if max_dp > mt_val:
                    if max_dc == 1:
                        fp = 6
                    elif max_dc == 2:
                        x_g = split_data['test']['last'][i, GYRO_IDX, :].reshape(1, -1)
                        gp = clf_c7.predict_proba(x_g)[0, 1]
                        fp = 7 if gp > 0.5 else cls_pred
                    elif max_dc == 3:
                        x_m = split_data['test']['last'][i, MAG_IDX, :].reshape(1, -1)
                        mp = clf_c8.predict_proba(x_m)[0, 1]
                        fp = 8 if mp > 0.5 else cls_pred
                else:
                    fp = cls_pred

                if fp == true_op:
                    pc_correct[true_op] += 1

            overall = sum(pc_correct.values()) / N
            c6 = pc_correct[6] / pc_total[6]
            c7 = pc_correct[7] / pc_total[7]
            c8 = pc_correct[8] / pc_total[8]
            print(f"    pipeline mt={mt_val}: {overall:.2%}  c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    # =====================================================================
    # PART 4: Gray-zone fallback cascade
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"PART 4: Gray-zone fallback — check other modalities when router is uncertain")
    print(f"{'='*70}")

    # When machine router max_damage_prob is in [low, high), also check
    # gyroscope and magnetometer classifiers
    # Train global (all-sample) per-modality damage classifiers
    global_clfs = {}
    for m_idx, mod_name in enumerate(MODALITY_NAMES):
        for base_op in range(3):
            damage_cls = DAMAGE_FOR_BASE[base_op]
            member_classes = GROUP_CLASSES[base_op]
            tr_mask = np.isin(train_ops, member_classes)
            y_tr = (train_ops[tr_mask] == damage_cls).astype(int)
            clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
            clf.fit(split_data['train']['last'][tr_mask, m_idx, :], y_tr)
            global_clfs[(m_idx, base_op)] = clf

    # Reproduce MLP router
    torch.manual_seed(42)
    np.random.seed(42)
    mlp, _ = train_mlp_head(
        split_data['train']['last'][:, MACHINE_MOD_IDX, :], y_train_4c,
        split_data['val']['last'][:, MACHINE_MOD_IDX, :], y_val_4c,
        n_classes=4, hidden_dims=(64,), dropout=0.2, lr=1e-3,
        epochs=300, patience=50, class_weight=cw_4c, device='cpu')
    with torch.no_grad():
        router_probs = torch.softmax(
            mlp(torch.tensor(split_data['test']['last'][:, MACHINE_MOD_IDX, :],
                             dtype=torch.float32)), dim=1).numpy()

    for high_thresh in [0.5, 0.4, 0.3]:
        for low_thresh in [0.1, 0.15, 0.2, 0.25]:
            if low_thresh >= high_thresh:
                continue

            pc_correct = Counter()
            pc_total = Counter()
            for i in range(N):
                true_op = test_ops[i]
                pc_total[true_op] += 1
                cls_pred = test_cls_preds[i]
                rp = router_probs[i]
                max_dc = np.argmax(rp[1:]) + 1
                max_dp = rp[max_dc]

                if max_dp > high_thresh:
                    # High confidence damage
                    if max_dc == 1:
                        fp = 6
                    elif max_dc == 2:
                        gp = clf_c7.predict_proba(
                            split_data['test']['last'][i, GYRO_IDX, :].reshape(1, -1))[0, 1]
                        fp = 7 if gp > 0.5 else cls_pred
                    elif max_dc == 3:
                        mp = clf_c8.predict_proba(
                            split_data['test']['last'][i, MAG_IDX, :].reshape(1, -1))[0, 1]
                        fp = 8 if mp > 0.5 else cls_pred
                elif max_dp > low_thresh:
                    # Gray zone — check ALL modalities for this damage class
                    base_op = max_dc - 1  # 0=adaptive, 1=face, 2=pocket
                    damage_cls = DAMAGE_FOR_BASE[base_op]
                    votes = 0
                    for m_idx in range(7):
                        prob = global_clfs[(m_idx, base_op)].predict_proba(
                            split_data['test']['last'][i, m_idx, :].reshape(1, -1))[0, 1]
                        if prob > 0.5:
                            votes += 1
                    # Majority vote (4+ of 7 modalities agree)
                    if votes >= 4:
                        fp = damage_cls
                    else:
                        fp = cls_pred
                else:
                    fp = cls_pred

                if fp == true_op:
                    pc_correct[true_op] += 1

            overall = sum(pc_correct.values()) / N
            c6 = pc_correct[6] / pc_total[6]
            c7 = pc_correct[7] / pc_total[7]
            c8 = pc_correct[8] / pc_total[8]
            c05 = np.mean([pc_correct[c] / pc_total[c] for c in range(6)])
            if overall >= 0.99:
                print(f"  high={high_thresh} low={low_thresh}: {overall:.2%}  "
                      f"c0-5={c05:.0%} c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    print(f"\n{'='*70}")
    print(f"REFERENCE: MLP(64) mt=0.5 = 99.19%  c6=93% c7=97% c8=94%")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
