#!/usr/bin/env python3
"""Option C: Per-modality damage detection BEFORE fusion.

Extract per-modality encoded representations from v1 (before CrossModalFusion),
train per-modality damage classifiers, and test if the damage signal survives
the per-modality encoders but gets destroyed by fusion.

If per-modality embeddings preserve the damage signal (especially for c6),
this validates adding per-modality damage heads to the architecture.
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

BASE_OP_MAP = {0: 0, 1: 0, 6: 0, 2: 1, 3: 1, 7: 1, 4: 2, 5: 2, 8: 2}
DAMAGE_FOR_BASE = {0: 6, 1: 7, 2: 8}
GROUP_NAMES_MAP = {0: 'adaptive', 1: 'face', 2: 'pocket'}
GROUP_CLASSES = {0: [0, 1, 6], 1: [2, 3, 7], 2: [4, 5, 8]}

MODALITY_NAMES = ['accelerometer', 'gyroscope', 'magnetometer', 'environmental',
                  'color', 'rms', 'machine']


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
    sensor_dims = [len(groups[n]) for n in group_names]
    return group_names, groups, sensor_dims


def extract_per_modality_embeddings(model, loader, group_indices, device):
    """Extract per-modality encoded representations before fusion, and post-fusion last-step."""
    model.eval()
    all_mod_embeds = []  # list of [N, M, D]
    all_fused_embeds = []  # list of [N, D]
    all_ops = []
    all_cls_preds = []

    with torch.no_grad():
        for batch in loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type']
            B, T = sensor_data.size(0), sensor_data.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
            mods = [sensor_data[:, :, idx] for idx in group_indices]

            out = model(mods, lengths)

            # encoded_mods: [B, T, M, D] — per-modality before fusion
            # Take last timestep for each modality
            enc_mods = out['encoded_mods']  # [B, T, M, D]
            last_idx = (lengths.clamp(min=1) - 1).view(-1)
            # Get last timestep: [B, M, D]
            per_mod_last = enc_mods[torch.arange(B, device=device), last_idx]
            all_mod_embeds.append(per_mod_last.cpu())

            # Also get time-averaged per-modality embeddings
            all_cls_preds.append(out['cls'].argmax(dim=-1).cpu())
            all_ops.append(operations)

    mod_embeds = torch.cat(all_mod_embeds).numpy()  # [N, M, D]
    cls_preds = torch.cat(all_cls_preds).numpy()
    ops = torch.cat(all_ops).numpy()
    return mod_embeds, cls_preds, ops


def main():
    split_dir = 'outputs/jan23_followup/no_leakage'
    model_path = 'outputs/jan30/mmdtae_standalone_v1/best_model.pt'

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    loaders = {}
    for split in ['train', 'val', 'test']:
        ds = DecoderDatasetFromSplits(split_dir=split_dir, split=split)
        loaders[split] = DataLoader(ds, batch_size=32, shuffle=False,
                                     collate_fn=decoder_collate_fn, num_workers=0)

    metadata_path = str(Path(split_dir) / 'metadata.json')
    mod_names, groups, sensor_dims = build_modality_indices(metadata_path)
    group_indices = [groups[name] for name in mod_names]

    # Load v1 model
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = MM_DTAE_LSTM(config)
    model.head_cls = nn.Linear(config.d_model, 9)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded v1 model, d_model={config.d_model}, {len(mod_names)} modalities")

    # Extract per-modality embeddings
    print(f"\nExtracting per-modality embeddings (before fusion)...")
    split_data = {}
    for split in ['train', 'val', 'test']:
        mod_embeds, cls_preds, ops = extract_per_modality_embeddings(
            model, loaders[split], group_indices, device)
        split_data[split] = {
            'mod_embeds': mod_embeds,  # [N, M, D]
            'cls_preds': cls_preds,
            'ops': ops
        }
        print(f"  {split}: {mod_embeds.shape}")

    # Also get raw features for comparison
    print(f"\nExtracting raw features...")
    for split in ['train', 'val', 'test']:
        raw = []
        for batch in loaders[split]:
            raw.append(batch['sensor_features'].mean(dim=1).numpy())
        split_data[split]['raw'] = np.concatenate(raw)

    M = split_data['train']['mod_embeds'].shape[1]
    D = split_data['train']['mod_embeds'].shape[2]
    print(f"Per-modality embedding: {M} modalities × {D} dims")

    results = {}

    # ===== PER-MODALITY DAMAGE DETECTION =====
    print(f"\n{'='*70}")
    print(f"PER-MODALITY DAMAGE DETECTION (before fusion)")
    print(f"{'='*70}")

    for base_op in range(3):
        group_name = GROUP_NAMES_MAP[base_op]
        damage_cls = DAMAGE_FOR_BASE[base_op]
        member_classes = GROUP_CLASSES[base_op]

        print(f"\n--- {group_name.upper()} (c{damage_cls}) ---")

        train_mask = np.isin(split_data['train']['ops'], member_classes)
        test_mask = np.isin(split_data['test']['ops'], member_classes)
        y_train = (split_data['train']['ops'][train_mask] == damage_cls).astype(int)
        y_test = (split_data['test']['ops'][test_mask] == damage_cls).astype(int)

        print(f"  Train: {train_mask.sum()} samples ({y_train.sum()} damage)")
        print(f"  Test:  {test_mask.sum()} samples ({y_test.sum()} damage)")

        group_results = {}

        # Test each modality individually
        for m_idx in range(M):
            mod_name = MODALITY_NAMES[m_idx]
            X_train = split_data['train']['mod_embeds'][train_mask, m_idx, :]  # [N, D]
            X_test = split_data['test']['mod_embeds'][test_mask, m_idx, :]

            clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
            clf.fit(X_train, y_train)
            test_preds = clf.predict(X_test)
            test_probs = clf.predict_proba(X_test)[:, 1]
            test_acc = accuracy_score(y_test, test_preds)
            test_f1 = f1_score(y_test, test_preds, zero_division=0)
            tp = ((test_preds == 1) & (y_test == 1)).sum()
            fp = ((test_preds == 1) & (y_test == 0)).sum()

            print(f"  {mod_name:15s}: acc={test_acc:.2%} F1={test_f1:.2%} TP={tp} FP={fp}")
            group_results[mod_name] = {
                'test_acc': float(test_acc), 'test_f1': float(test_f1),
                'tp': int(tp), 'fp': int(fp)
            }

        # Test ALL modalities concatenated (before fusion)
        X_train_all = split_data['train']['mod_embeds'][train_mask].reshape(train_mask.sum(), -1)
        X_test_all = split_data['test']['mod_embeds'][test_mask].reshape(test_mask.sum(), -1)
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
        clf.fit(X_train_all, y_train)
        test_preds = clf.predict(X_test_all)
        test_probs = clf.predict_proba(X_test_all)[:, 1]
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, zero_division=0)
        tp = ((test_preds == 1) & (y_test == 1)).sum()
        fp = ((test_preds == 1) & (y_test == 0)).sum()
        print(f"  {'ALL CONCAT':15s}: acc={test_acc:.2%} F1={test_f1:.2%} TP={tp} FP={fp}")
        group_results['all_concat'] = {
            'test_acc': float(test_acc), 'test_f1': float(test_f1),
            'tp': int(tp), 'fp': int(fp)
        }

        # Compare with raw features (within group)
        X_train_raw = split_data['train']['raw'][train_mask]
        X_test_raw = split_data['test']['raw'][test_mask]
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
        clf.fit(X_train_raw, y_train)
        test_preds = clf.predict(X_test_raw)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, zero_division=0)
        tp = ((test_preds == 1) & (y_test == 1)).sum()
        fp = ((test_preds == 1) & (y_test == 0)).sum()
        print(f"  {'RAW FEATURES':15s}: acc={test_acc:.2%} F1={test_f1:.2%} TP={tp} FP={fp}")
        group_results['raw_features'] = {
            'test_acc': float(test_acc), 'test_f1': float(test_f1),
            'tp': int(tp), 'fp': int(fp)
        }

        results[group_name] = group_results

    # ===== FULL PIPELINE: v1 cls + per-modality damage override =====
    print(f"\n{'='*70}")
    print(f"FULL PIPELINE: v1 cls + per-modality damage classifiers")
    print(f"{'='*70}")

    # Train per-group damage classifiers using best per-modality features
    # Strategy: use ALL concat modalities (before fusion) for each group
    damage_clfs = {}
    for base_op in range(3):
        member_classes = GROUP_CLASSES[base_op]
        damage_cls = DAMAGE_FOR_BASE[base_op]
        train_mask = np.isin(split_data['train']['ops'], member_classes)
        X_train = split_data['train']['mod_embeds'][train_mask].reshape(train_mask.sum(), -1)
        y_train = (split_data['train']['ops'][train_mask] == damage_cls).astype(int)
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
        clf.fit(X_train, y_train)
        damage_clfs[base_op] = clf

    test_ops = split_data['test']['ops']
    test_cls_preds = split_data['test']['cls_preds']
    test_mod_embeds = split_data['test']['mod_embeds']
    N = len(test_ops)

    per_class_total = Counter()
    for i in range(N):
        per_class_total[test_ops[i]] += 1

    # Routed approach: use v1 cls for base op, per-modality for damage
    print(f"\n--- Routed: v1 cls + ALL CONCAT per-modality ---")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pc_correct = Counter()
        for i in range(N):
            true_op = test_ops[i]
            cls_pred = test_cls_preds[i]
            base_op = BASE_OP_MAP[cls_pred]

            x = test_mod_embeds[i].reshape(1, -1)
            damage_prob = damage_clfs[base_op].predict_proba(x)[0, 1]

            if damage_prob > thresh:
                final_pred = DAMAGE_FOR_BASE[base_op]
            else:
                final_pred = cls_pred

            if final_pred == true_op:
                pc_correct[true_op] += 1

        total = sum(pc_correct.values())
        c6 = pc_correct[6] / per_class_total[6] if per_class_total[6] > 0 else 0
        c7 = pc_correct[7] / per_class_total[7] if per_class_total[7] > 0 else 0
        c8 = pc_correct[8] / per_class_total[8] if per_class_total[8] > 0 else 0
        rest = sum(pc_correct[c] for c in range(6))
        rest_total = sum(per_class_total[c] for c in range(6))
        print(f"  t={thresh:.1f}: {total}/{N}={total/N:.2%}  "
              f"c0-c5={rest}/{rest_total}={rest/rest_total:.2%}  "
              f"c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    # Strategy 2: Use BEST SINGLE modality per group (not concat)
    # adaptive→machine(6), face→gyroscope(1), pocket→magnetometer(2)
    BEST_MOD_PER_GROUP = {0: 6, 1: 1, 2: 2}  # base_op → modality index

    damage_clfs_single = {}
    for base_op in range(3):
        member_classes = GROUP_CLASSES[base_op]
        damage_cls = DAMAGE_FOR_BASE[base_op]
        m_idx = BEST_MOD_PER_GROUP[base_op]
        train_mask = np.isin(split_data['train']['ops'], member_classes)
        X_train = split_data['train']['mod_embeds'][train_mask, m_idx, :]
        y_train = (split_data['train']['ops'][train_mask] == damage_cls).astype(int)
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
        clf.fit(X_train, y_train)
        damage_clfs_single[base_op] = (clf, m_idx)

    print(f"\n--- Routed: v1 cls + BEST SINGLE modality per group ---")
    print(f"  adaptive→{MODALITY_NAMES[BEST_MOD_PER_GROUP[0]]}, "
          f"face→{MODALITY_NAMES[BEST_MOD_PER_GROUP[1]]}, "
          f"pocket→{MODALITY_NAMES[BEST_MOD_PER_GROUP[2]]}")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pc_correct = Counter()
        for i in range(N):
            true_op = test_ops[i]
            cls_pred = test_cls_preds[i]
            base_op = BASE_OP_MAP[cls_pred]

            clf, m_idx = damage_clfs_single[base_op]
            x = test_mod_embeds[i, m_idx, :].reshape(1, -1)
            damage_prob = clf.predict_proba(x)[0, 1]

            if damage_prob > thresh:
                final_pred = DAMAGE_FOR_BASE[base_op]
            else:
                final_pred = cls_pred

            if final_pred == true_op:
                pc_correct[true_op] += 1

        total = sum(pc_correct.values())
        c6 = pc_correct[6] / per_class_total[6] if per_class_total[6] > 0 else 0
        c7 = pc_correct[7] / per_class_total[7] if per_class_total[7] > 0 else 0
        c8 = pc_correct[8] / per_class_total[8] if per_class_total[8] > 0 else 0
        rest = sum(pc_correct[c] for c in range(6))
        rest_total = sum(per_class_total[c] for c in range(6))
        print(f"  t={thresh:.1f}: {total}/{N}={total/N:.2%}  "
              f"c0-c5={rest}/{rest_total}={rest/rest_total:.2%}  "
              f"c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    # Strategy 3: GLOBAL per-modality — run all 3 best-modality classifiers on ALL samples
    # No routing dependency — each classifier uses its own modality embedding
    print(f"\n--- GLOBAL: run all 3 per-modality classifiers on ALL samples ---")
    print(f"  adaptive→{MODALITY_NAMES[BEST_MOD_PER_GROUP[0]]}, "
          f"face→{MODALITY_NAMES[BEST_MOD_PER_GROUP[1]]}, "
          f"pocket→{MODALITY_NAMES[BEST_MOD_PER_GROUP[2]]}")

    # Retrain classifiers on ALL samples (not just within-group)
    # Each classifier: is this sample damage-X? (binary, trained within-group)
    # But applied globally
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pc_correct = Counter()
        for i in range(N):
            true_op = test_ops[i]
            cls_pred = test_cls_preds[i]

            # Check each damage classifier on its best modality
            damage_hits = {}
            for base_op in range(3):
                clf, m_idx = damage_clfs_single[base_op]
                x = test_mod_embeds[i, m_idx, :].reshape(1, -1)
                prob = clf.predict_proba(x)[0, 1]
                if prob > thresh:
                    damage_hits[base_op] = prob

            if damage_hits:
                # Pick the one with highest confidence
                best_base = max(damage_hits, key=damage_hits.get)
                final_pred = DAMAGE_FOR_BASE[best_base]
            else:
                final_pred = cls_pred

            if final_pred == true_op:
                pc_correct[true_op] += 1

        total = sum(pc_correct.values())
        c6 = pc_correct[6] / per_class_total[6] if per_class_total[6] > 0 else 0
        c7 = pc_correct[7] / per_class_total[7] if per_class_total[7] > 0 else 0
        c8 = pc_correct[8] / per_class_total[8] if per_class_total[8] > 0 else 0
        rest = sum(pc_correct[c] for c in range(6))
        rest_total = sum(per_class_total[c] for c in range(6))
        print(f"  t={thresh:.1f}: {total}/{N}={total/N:.2%}  "
              f"c0-c5={rest}/{rest_total}={rest/rest_total:.2%}  "
              f"c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    # Also try per-modality individually for each group
    # Use the best single modality per group
    print(f"\n--- Per-group best single modality ---")
    for base_op in range(3):
        group_name = GROUP_NAMES_MAP[base_op]
        member_classes = GROUP_CLASSES[base_op]
        damage_cls = DAMAGE_FOR_BASE[base_op]
        train_mask = np.isin(split_data['train']['ops'], member_classes)
        test_mask = np.isin(split_data['test']['ops'], member_classes)
        y_train = (split_data['train']['ops'][train_mask] == damage_cls).astype(int)
        y_test = (split_data['test']['ops'][test_mask] == damage_cls).astype(int)

        best_f1 = -1
        best_mod = -1
        for m_idx in range(M):
            X_train = split_data['train']['mod_embeds'][train_mask, m_idx, :]
            X_test = split_data['test']['mod_embeds'][test_mask, m_idx, :]
            clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            f1 = f1_score(y_test, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_mod = m_idx

        print(f"  {group_name}: best modality = {MODALITY_NAMES[best_mod]} (F1={best_f1:.2%})")

    # ===== HYBRID: Machine 4-class router + specialized per-modality classifiers =====
    print(f"\n{'='*70}")
    print(f"HYBRID: Machine 4-class router + specialized per-modality overrides")
    print(f"{'='*70}")

    # Step 1: Train machine-only 4-class classifier (normal/c6/c7/c8)
    MACHINE_MOD_IDX = 6
    train_ops = split_data['train']['ops']
    test_ops_arr = split_data['test']['ops']

    # Map ops to 4 classes: 0=normal(c0-c5), 1=c6, 2=c7, 3=c8
    def map_4class(ops):
        y = np.zeros(len(ops), dtype=int)
        y[ops == 6] = 1
        y[ops == 7] = 2
        y[ops == 8] = 3
        return y

    y_train_4c = map_4class(train_ops)
    y_test_4c = map_4class(test_ops_arr)

    X_train_machine = split_data['train']['mod_embeds'][:, MACHINE_MOD_IDX, :]
    X_test_machine = split_data['test']['mod_embeds'][:, MACHINE_MOD_IDX, :]

    clf_4c = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_4c.fit(X_train_machine, y_train_4c)
    preds_4c = clf_4c.predict(X_test_machine)
    probs_4c = clf_4c.predict_proba(X_test_machine)

    print(f"\nMachine 4-class standalone: {accuracy_score(y_test_4c, preds_4c):.2%}")

    # Step 2: Train specialized per-modality binary classifiers for c7 and c8
    # c7: gyroscope (mod 1), within face group
    # c8: magnetometer (mod 2), within pocket group
    GYRO_IDX = 1
    MAG_IDX = 2

    # c7 classifier: gyroscope, face group (c2, c3, c7)
    face_train_mask = np.isin(train_ops, [2, 3, 7])
    face_test_mask = np.isin(test_ops_arr, [2, 3, 7])
    clf_c7 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_c7.fit(
        split_data['train']['mod_embeds'][face_train_mask, GYRO_IDX, :],
        (train_ops[face_train_mask] == 7).astype(int)
    )

    # c8 classifier: magnetometer, pocket group (c4, c5, c8)
    pocket_train_mask = np.isin(train_ops, [4, 5, 8])
    pocket_test_mask = np.isin(test_ops_arr, [4, 5, 8])
    clf_c8 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_c8.fit(
        split_data['train']['mod_embeds'][pocket_train_mask, MAG_IDX, :],
        (train_ops[pocket_train_mask] == 8).astype(int)
    )

    # Step 3: Hybrid pipeline
    # - Machine 4-class gives initial prediction
    # - If machine says c7 → confirm with gyroscope classifier (within-group)
    # - If machine says c8 → confirm with magnetometer classifier (within-group)
    # - If machine says normal → use v1 cls for sub-class
    # - If machine says c6 → trust it (machine is best for c6)
    print(f"\n--- Hybrid pipeline: machine router + specialist overrides ---")
    print(f"  c6: machine only (93%)")
    print(f"  c7: machine router → gyroscope specialist")
    print(f"  c8: machine router → magnetometer specialist")

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pc_correct = Counter()
        for i in range(N):
            true_op = test_ops[i]
            cls_pred = test_cls_preds[i]
            machine_probs = probs_4c[i]  # [normal, c6, c7, c8]

            # Check if machine detects damage (any damage class > thresh)
            max_damage_class = np.argmax(machine_probs[1:]) + 1  # 1=c6, 2=c7, 3=c8
            max_damage_prob = machine_probs[max_damage_class]

            if max_damage_prob > thresh:
                if max_damage_class == 1:
                    # c6: trust machine
                    final_pred = 6
                elif max_damage_class == 2:
                    # c7: use gyroscope specialist
                    x_gyro = split_data['test']['mod_embeds'][i, GYRO_IDX, :].reshape(1, -1)
                    gyro_prob = clf_c7.predict_proba(x_gyro)[0, 1]
                    if gyro_prob > 0.5:
                        final_pred = 7
                    else:
                        final_pred = cls_pred  # gyroscope says no damage
                elif max_damage_class == 3:
                    # c8: use magnetometer specialist
                    x_mag = split_data['test']['mod_embeds'][i, MAG_IDX, :].reshape(1, -1)
                    mag_prob = clf_c8.predict_proba(x_mag)[0, 1]
                    if mag_prob > 0.5:
                        final_pred = 8
                    else:
                        final_pred = cls_pred  # magnetometer says no damage
            else:
                final_pred = cls_pred

            if final_pred == true_op:
                pc_correct[true_op] += 1

        total = sum(pc_correct.values())
        c6 = pc_correct[6] / per_class_total[6] if per_class_total[6] > 0 else 0
        c7 = pc_correct[7] / per_class_total[7] if per_class_total[7] > 0 else 0
        c8 = pc_correct[8] / per_class_total[8] if per_class_total[8] > 0 else 0
        rest = sum(pc_correct[c] for c in range(6))
        rest_total = sum(per_class_total[c] for c in range(6))
        print(f"  t={thresh:.1f}: {total}/{N}={total/N:.2%}  "
              f"c0-c5={rest}/{rest_total}={rest/rest_total:.2%}  "
              f"c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    # Also try: machine 4-class for ALL damage routing (no specialist override)
    print(f"\n--- Machine 4-class only (no specialists) ---")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pc_correct = Counter()
        for i in range(N):
            true_op = test_ops[i]
            cls_pred = test_cls_preds[i]
            machine_probs = probs_4c[i]

            max_damage_class = np.argmax(machine_probs[1:]) + 1
            max_damage_prob = machine_probs[max_damage_class]

            if max_damage_prob > thresh:
                final_pred = [None, 6, 7, 8][max_damage_class]
            else:
                final_pred = cls_pred

            if final_pred == true_op:
                pc_correct[true_op] += 1

        total = sum(pc_correct.values())
        c6 = pc_correct[6] / per_class_total[6] if per_class_total[6] > 0 else 0
        c7 = pc_correct[7] / per_class_total[7] if per_class_total[7] > 0 else 0
        c8 = pc_correct[8] / per_class_total[8] if per_class_total[8] > 0 else 0
        rest = sum(pc_correct[c] for c in range(6))
        rest_total = sum(per_class_total[c] for c in range(6))
        print(f"  t={thresh:.1f}: {total}/{N}={total/N:.2%}  "
              f"c0-c5={rest}/{rest_total}={rest/rest_total:.2%}  "
              f"c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    # Strategy: Machine 4-class + fallback specialist scan on "normal" predictions
    # For samples machine says "normal": also run gyroscope c7 and magnetometer c8
    # classifiers to catch what the machine missed
    print(f"\n--- Hybrid + specialist fallback scan on normal predictions ---")
    print(f"  If machine says damage → trust it")
    print(f"  If machine says normal → also check gyroscope(c7) and magnetometer(c8)")

    # Need within-group classifiers trained on ALL data (not just within-group)
    # Actually, use the within-group ones but apply them globally
    # Re-train c7 gyro on all data (face-damage vs everything else)
    clf_c7_global = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_c7_global.fit(
        split_data['train']['mod_embeds'][:, GYRO_IDX, :],
        (train_ops == 7).astype(int)
    )
    clf_c8_global = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_c8_global.fit(
        split_data['train']['mod_embeds'][:, MAG_IDX, :],
        (train_ops == 8).astype(int)
    )

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for specialist_thresh in [0.5, 0.7, 0.9]:
            pc_correct = Counter()
            for i in range(N):
                true_op = test_ops[i]
                cls_pred = test_cls_preds[i]
                machine_probs = probs_4c[i]

                max_damage_class = np.argmax(machine_probs[1:]) + 1
                max_damage_prob = machine_probs[max_damage_class]

                if max_damage_prob > thresh:
                    # Machine detected damage
                    if max_damage_class == 1:
                        final_pred = 6
                    elif max_damage_class == 2:
                        final_pred = 7
                    elif max_damage_class == 3:
                        final_pred = 8
                else:
                    # Machine says normal — run specialists as fallback
                    x_gyro = split_data['test']['mod_embeds'][i, GYRO_IDX, :].reshape(1, -1)
                    x_mag = split_data['test']['mod_embeds'][i, MAG_IDX, :].reshape(1, -1)
                    c7_prob = clf_c7_global.predict_proba(x_gyro)[0, 1]
                    c8_prob = clf_c8_global.predict_proba(x_mag)[0, 1]

                    if c7_prob > specialist_thresh and c7_prob > c8_prob:
                        final_pred = 7
                    elif c8_prob > specialist_thresh:
                        final_pred = 8
                    else:
                        final_pred = cls_pred

                if final_pred == true_op:
                    pc_correct[true_op] += 1

            total = sum(pc_correct.values())
            c6 = pc_correct[6] / per_class_total[6] if per_class_total[6] > 0 else 0
            c7 = pc_correct[7] / per_class_total[7] if per_class_total[7] > 0 else 0
            c8 = pc_correct[8] / per_class_total[8] if per_class_total[8] > 0 else 0
            rest = sum(pc_correct[c] for c in range(6))
            rest_total = sum(per_class_total[c] for c in range(6))
            print(f"  t={thresh:.1f} s={specialist_thresh:.1f}: {total}/{N}={total/N:.2%}  "
                  f"c0-c5={rest}/{rest_total}={rest/rest_total:.2%}  "
                  f"c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    # Save
    output_path = Path('outputs/jan30/per_modality_damage_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
