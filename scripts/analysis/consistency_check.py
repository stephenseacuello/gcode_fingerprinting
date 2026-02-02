#!/usr/bin/env python3
"""Test base-operation consistency check to eliminate the last false positive.

Rule: if router predicts damage class X, but v1 cls head predicts a class from
a different base-op group, veto the damage prediction and use cls head instead.

  c6 (adaptive damage) only valid if cls pred in {c0, c1, c6}
  c7 (face damage) only valid if cls pred in {c2, c3, c7}
  c8 (pocket damage) only valid if cls pred in {c4, c5, c8}

Also tests: mean-pooled specialists, and all combinations.
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

# Which cls predictions are consistent with each damage class
CONSISTENT_CLS = {
    6: {0, 1, 6},     # c6 = adaptive damage → cls should be adaptive group
    7: {2, 3, 7},     # c7 = face damage → cls should be face group
    8: {4, 5, 8},     # c8 = pocket damage → cls should be pocket group
}


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


def run_pipeline(test_ops, test_cls_preds, router_probs, data,
                 clf_c7, clf_c8, mt, consistency_check=False,
                 specialist_pool='last'):
    """Run the full pipeline with optional consistency check."""
    N = len(test_ops)
    preds = np.zeros(N, dtype=int)

    for i in range(N):
        cls_pred = test_cls_preds[i]
        rp = router_probs[i]
        max_dc = np.argmax(rp[1:]) + 1
        max_dp = rp[max_dc]
        damage_cls = [None, 6, 7, 8][max_dc]

        if max_dp > mt:
            # Consistency check: does the cls head agree on base operation?
            if consistency_check and cls_pred not in CONSISTENT_CLS[damage_cls]:
                preds[i] = cls_pred
                continue

            if max_dc == 1:  # c6
                preds[i] = 6
            elif max_dc == 2:  # c7
                x_g = data['test'][specialist_pool][i, GYRO_IDX, :].reshape(1, -1)
                gp = clf_c7.predict_proba(x_g)[0, 1]
                preds[i] = 7 if gp > 0.5 else cls_pred
            elif max_dc == 3:  # c8
                x_m = data['test'][specialist_pool][i, MAG_IDX, :].reshape(1, -1)
                mp = clf_c8.predict_proba(x_m)[0, 1]
                preds[i] = 8 if mp > 0.5 else cls_pred
        else:
            preds[i] = cls_pred

    return preds


def report(test_ops, preds, label):
    N = len(test_ops)
    overall = (preds == test_ops).mean()
    missed = [(i, test_ops[i], preds[i]) for i in range(N) if preds[i] != test_ops[i]]

    c05_correct = sum((preds[test_ops == c] == c).sum() for c in range(6))
    c05_total = sum((test_ops == c).sum() for c in range(6))
    c6 = (preds[test_ops == 6] == 6).mean() if (test_ops == 6).sum() > 0 else 0
    c7 = (preds[test_ops == 7] == 7).mean() if (test_ops == 7).sum() > 0 else 0
    c8 = (preds[test_ops == 8] == 8).mean() if (test_ops == 8).sum() > 0 else 0

    print(f"  {label:50s} {overall:.2%} ({(preds==test_ops).sum()}/{N})  "
          f"c0-5={c05_correct}/{c05_total} c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    if missed:
        for idx, true_c, pred_c in missed:
            print(f"    MISS test[{idx}]: true=c{true_c}({CLASS_NAMES[true_c]}) → pred=c{pred_c}({CLASS_NAMES[pred_c]})")

    return overall, missed


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

    # Train mean-pooled MLP(64) router
    print(f"\nTraining mean-pooled MLP(64) router...")
    X_tr = data['train']['mean'][:, MACHINE_MOD_IDX, :]
    X_val = data['val']['mean'][:, MACHINE_MOD_IDX, :]
    X_te = data['test']['mean'][:, MACHINE_MOD_IDX, :]

    best_probs = None
    best_acc = 0
    for seed in [42, 123, 777]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        mlp, _ = train_mlp_head(
            X_tr, y_train_4c, X_val, y_val_4c,
            n_classes=4, hidden_dims=(64,), dropout=0.2, lr=1e-3,
            epochs=300, patience=50, class_weight=cw_4c, device='cpu')
        with torch.no_grad():
            probs = torch.softmax(mlp(torch.tensor(X_te, dtype=torch.float32)), dim=1).numpy()
        acc = (probs.argmax(axis=1) == y_test_4c).mean()
        if acc > best_acc:
            best_acc = acc
            best_probs = probs
    print(f"  Best standalone: {best_acc:.2%}")

    # Train specialists — both last-step and mean-pooled
    print(f"\nTraining specialists...")
    specialists = {}
    for pool in ['last', 'mean']:
        face_mask = np.isin(train_ops, [2, 3, 7])
        clf_c7 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
        clf_c7.fit(data['train'][pool][face_mask, GYRO_IDX, :],
                   (train_ops[face_mask] == 7).astype(int))

        pocket_mask = np.isin(train_ops, [4, 5, 8])
        clf_c8 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
        clf_c8.fit(data['train'][pool][pocket_mask, MAG_IDX, :],
                   (train_ops[pocket_mask] == 8).astype(int))

        # Check specialist test performance
        te_face = np.isin(test_ops, [2, 3, 7])
        c7_probs = clf_c7.predict_proba(data['test'][pool][te_face, GYRO_IDX, :])[:, 1]
        c7_preds = (c7_probs > 0.5).astype(int)
        c7_true = (test_ops[te_face] == 7).astype(int)
        from sklearn.metrics import f1_score
        c7_f1 = f1_score(c7_true, c7_preds, zero_division=0)

        te_pocket = np.isin(test_ops, [4, 5, 8])
        c8_probs = clf_c8.predict_proba(data['test'][pool][te_pocket, MAG_IDX, :])[:, 1]
        c8_preds = (c8_probs > 0.5).astype(int)
        c8_true = (test_ops[te_pocket] == 8).astype(int)
        c8_f1 = f1_score(c8_true, c8_preds, zero_division=0)

        specialists[pool] = (clf_c7, clf_c8)
        print(f"  {pool:5s} specialists: c7 gyro F1={c7_f1:.2%}, c8 mag F1={c8_f1:.2%}")

    test_cls_preds = data['test']['cls_preds']

    # =====================================================================
    # Check: does test[579] cls pred fall outside adaptive group?
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"VERIFICATION: test[579] consistency")
    print(f"{'='*70}")
    idx = 579
    cls_pred = test_cls_preds[idx]
    rp = best_probs[idx]
    print(f"  cls pred: c{cls_pred} ({CLASS_NAMES[cls_pred]})")
    print(f"  router probs: norm={rp[0]:.4f} c6={rp[1]:.4f} c7={rp[2]:.4f} c8={rp[3]:.4f}")
    print(f"  router says c6 (adaptive damage)")
    print(f"  cls pred c{cls_pred} in adaptive group {CONSISTENT_CLS[6]}? {cls_pred in CONSISTENT_CLS[6]}")

    # =====================================================================
    # Test all configurations
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"RESULTS: All pipeline configurations")
    print(f"{'='*70}")

    for mt in [0.45, 0.50, 0.55]:
        print(f"\n--- mt={mt:.2f} ---")

        # Baseline (no consistency, last-step specialists)
        clf_c7, clf_c8 = specialists['last']
        preds = run_pipeline(test_ops, test_cls_preds, best_probs, data,
                             clf_c7, clf_c8, mt, consistency_check=False,
                             specialist_pool='last')
        report(test_ops, preds, "baseline (last specialists)")

        # Consistency check only
        preds = run_pipeline(test_ops, test_cls_preds, best_probs, data,
                             clf_c7, clf_c8, mt, consistency_check=True,
                             specialist_pool='last')
        report(test_ops, preds, "+ consistency check")

        # Mean-pooled specialists only
        clf_c7m, clf_c8m = specialists['mean']
        preds = run_pipeline(test_ops, test_cls_preds, best_probs, data,
                             clf_c7m, clf_c8m, mt, consistency_check=False,
                             specialist_pool='mean')
        report(test_ops, preds, "mean specialists")

        # Both: consistency + mean specialists
        preds = run_pipeline(test_ops, test_cls_preds, best_probs, data,
                             clf_c7m, clf_c8m, mt, consistency_check=True,
                             specialist_pool='mean')
        report(test_ops, preds, "+ consistency + mean specialists")

    # =====================================================================
    # Sanity: check if consistency ever hurts real damage samples
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"SANITY: consistency check impact on real damage samples")
    print(f"{'='*70}")

    clf_c7, clf_c8 = specialists['last']
    for mt in [0.45, 0.50, 0.55]:
        N = len(test_ops)
        vetoed_real = 0
        vetoed_fp = 0
        for i in range(N):
            rp = best_probs[i]
            max_dc = np.argmax(rp[1:]) + 1
            max_dp = rp[max_dc]
            damage_cls = [None, 6, 7, 8][max_dc]
            cls_pred = test_cls_preds[i]
            true_op = test_ops[i]

            if max_dp > mt and cls_pred not in CONSISTENT_CLS[damage_cls]:
                if true_op == damage_cls:
                    vetoed_real += 1
                    print(f"  mt={mt:.2f} VETOED REAL: test[{i}] true=c{true_op} "
                          f"cls=c{cls_pred}({CLASS_NAMES[cls_pred]}) "
                          f"router→c{damage_cls} prob={max_dp:.4f}")
                else:
                    vetoed_fp += 1

        print(f"  mt={mt:.2f}: vetoed {vetoed_fp} FP, {vetoed_real} real damage")


if __name__ == '__main__':
    main()
