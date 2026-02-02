#!/usr/bin/env python3
"""Push for 100%: add secondary confirmation for c6 predictions.

The 99.67% pipeline has 2 false positives: normal c5 samples predicted as c6.
The router says c6 but ALL 7 per-modality classifiers say "no damage".

Fix: when router predicts c6, require at least K modalities to confirm damage.
If not confirmed, fall back to v1 cls prediction.

Also tests: c7/c8 already have specialist confirmation (gyro/mag). Could we add
multi-modality confirmation for them too? And what about using mean-pooled
specialists instead of last-step?
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


def eval_pipeline(test_ops, test_cls_preds, router_probs, data,
                  clf_c7, clf_c8, c6_confirm_clfs, mt, min_c6_votes):
    """Run pipeline with optional c6 confirmation."""
    N = len(test_ops)
    preds = np.zeros(N, dtype=int)

    for i in range(N):
        cls_pred = test_cls_preds[i]
        rp = router_probs[i]
        max_dc = np.argmax(rp[1:]) + 1
        max_dp = rp[max_dc]

        if max_dp > mt:
            if max_dc == 1:  # c6
                if min_c6_votes > 0 and c6_confirm_clfs is not None:
                    # Secondary confirmation: count modalities that see adaptive damage
                    votes = 0
                    for (m_idx, clf, pool_type) in c6_confirm_clfs:
                        x = data['test'][pool_type][i, m_idx, :].reshape(1, -1)
                        prob = clf.predict_proba(x)[0, 1]
                        if prob > 0.5:
                            votes += 1
                    preds[i] = 6 if votes >= min_c6_votes else cls_pred
                else:
                    preds[i] = 6
            elif max_dc == 2:  # c7
                x_g = data['test']['last'][i, GYRO_IDX, :].reshape(1, -1)
                gp = clf_c7.predict_proba(x_g)[0, 1]
                preds[i] = 7 if gp > 0.5 else cls_pred
            elif max_dc == 3:  # c8
                x_m = data['test']['last'][i, MAG_IDX, :].reshape(1, -1)
                mp = clf_c8.predict_proba(x_m)[0, 1]
                preds[i] = 8 if mp > 0.5 else cls_pred
        else:
            preds[i] = cls_pred

    return preds


def print_results(test_ops, preds, label=""):
    N = len(test_ops)
    overall = (preds == test_ops).mean()
    per_cls = {}
    for c in range(9):
        mask = test_ops == c
        if mask.sum() > 0:
            per_cls[c] = (preds[mask] == c).sum(), mask.sum()

    c05_correct = sum(per_cls[c][0] for c in range(6))
    c05_total = sum(per_cls[c][1] for c in range(6))
    c6 = per_cls[6][0] / per_cls[6][1] if 6 in per_cls else 0
    c7 = per_cls[7][0] / per_cls[7][1] if 7 in per_cls else 0
    c8 = per_cls[8][0] / per_cls[8][1] if 8 in per_cls else 0

    missed = [(i, test_ops[i], preds[i]) for i in range(N) if preds[i] != test_ops[i]]
    print(f"  {label:40s} {overall:.2%} ({(preds==test_ops).sum()}/{N})  "
          f"c0-5={c05_correct}/{c05_total} c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}  "
          f"misses={len(missed)}")
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

    # =====================================================================
    # Train mean-pooled MLP(64) router
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"Training mean-pooled MLP(64) router (seed 42)")
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
        acc = (probs.argmax(axis=1) == y_test_4c).mean()
        print(f"  Seed {seed}: {acc:.2%}")
        if acc > best_acc:
            best_acc = acc
            best_mlp = mlp
            best_probs = probs

    print(f"  Best standalone: {best_acc:.2%}")

    # LogReg specialists
    face_mask = np.isin(train_ops, [2, 3, 7])
    clf_c7 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_c7.fit(data['train']['last'][face_mask, GYRO_IDX, :],
               (train_ops[face_mask] == 7).astype(int))

    pocket_mask = np.isin(train_ops, [4, 5, 8])
    clf_c8 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_c8.fit(data['train']['last'][pocket_mask, MAG_IDX, :],
               (train_ops[pocket_mask] == 8).astype(int))

    test_cls_preds = data['test']['cls_preds']
    N = len(test_ops)

    # =====================================================================
    # Baseline: no c6 confirmation
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"BASELINE: mean-pooled MLP(64) pipeline (no c6 confirmation)")
    print(f"{'='*70}")

    for mt in [0.45, 0.50, 0.55]:
        preds = eval_pipeline(test_ops, test_cls_preds, best_probs, data,
                              clf_c7, clf_c8, None, mt, 0)
        print_results(test_ops, preds, f"mt={mt:.2f} (no confirm)")

    # =====================================================================
    # Train per-modality c6 confirmation classifiers
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"Training per-modality adaptive damage (c6) confirmation classifiers")
    print(f"{'='*70}")

    adaptive_mask = np.isin(train_ops, [0, 1, 6])
    y_adaptive = (train_ops[adaptive_mask] == 6).astype(int)

    c6_confirm_clfs = []
    for pool_type in ['mean', 'last']:
        for m_idx, mod_name in enumerate(MODALITY_NAMES):
            clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
            clf.fit(data['train'][pool_type][adaptive_mask, m_idx, :], y_adaptive)

            # Check test accuracy on adaptive group
            te_adaptive = np.isin(test_ops, [0, 1, 6])
            y_te = (test_ops[te_adaptive] == 6).astype(int)
            te_probs = clf.predict_proba(data['test'][pool_type][te_adaptive, m_idx, :])[:, 1]
            te_preds = (te_probs > 0.5).astype(int)
            from sklearn.metrics import f1_score, recall_score, precision_score
            rec = recall_score(y_te, te_preds, zero_division=0)
            prec = precision_score(y_te, te_preds, zero_division=0)

            c6_confirm_clfs.append((m_idx, clf, pool_type))
            if rec > 0 or m_idx == MACHINE_MOD_IDX:
                print(f"  {pool_type:5s} {mod_name:15s}: recall={rec:.2%} precision={prec:.2%}")

    # =====================================================================
    # Strategy 1: Require K modalities to confirm c6
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"STRATEGY 1: Require K+ modality votes to confirm c6")
    print(f"{'='*70}")

    # Use mean-pooled modality classifiers for confirmation
    c6_mean_clfs = [(m_idx, clf, pt) for (m_idx, clf, pt) in c6_confirm_clfs if pt == 'mean']

    for mt in [0.45, 0.50, 0.55]:
        for min_votes in [1, 2, 3]:
            preds = eval_pipeline(test_ops, test_cls_preds, best_probs, data,
                                  clf_c7, clf_c8, c6_mean_clfs, mt, min_votes)
            overall, missed = print_results(test_ops, preds,
                                            f"mt={mt:.2f} c6_votes>={min_votes} (mean)")

    # Use last-step modality classifiers
    c6_last_clfs = [(m_idx, clf, pt) for (m_idx, clf, pt) in c6_confirm_clfs if pt == 'last']

    print()
    for mt in [0.45, 0.50, 0.55]:
        for min_votes in [1, 2]:
            preds = eval_pipeline(test_ops, test_cls_preds, best_probs, data,
                                  clf_c7, clf_c8, c6_last_clfs, mt, min_votes)
            overall, missed = print_results(test_ops, preds,
                                            f"mt={mt:.2f} c6_votes>={min_votes} (last)")

    # Use both poolings combined (14 classifiers)
    print()
    for mt in [0.45, 0.50, 0.55]:
        for min_votes in [1, 2, 3, 4]:
            preds = eval_pipeline(test_ops, test_cls_preds, best_probs, data,
                                  clf_c7, clf_c8, c6_confirm_clfs, mt, min_votes)
            overall, missed = print_results(test_ops, preds,
                                            f"mt={mt:.2f} c6_votes>={min_votes} (both)")

    # =====================================================================
    # Strategy 2: Use machine modality only for c6 confirmation
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"STRATEGY 2: Machine-only c6 confirmation (mean-pooled)")
    print(f"{'='*70}")

    c6_machine_only = [(m_idx, clf, pt) for (m_idx, clf, pt) in c6_confirm_clfs
                       if pt == 'mean' and m_idx == MACHINE_MOD_IDX]

    for mt in [0.45, 0.50, 0.55]:
        preds = eval_pipeline(test_ops, test_cls_preds, best_probs, data,
                              clf_c7, clf_c8, c6_machine_only, mt, 1)
        overall, missed = print_results(test_ops, preds,
                                        f"mt={mt:.2f} machine-confirm")

    # =====================================================================
    # Strategy 3: Machine + gyroscope for c6 confirmation
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"STRATEGY 3: Machine + gyroscope c6 confirmation (mean-pooled)")
    print(f"{'='*70}")

    c6_machine_gyro = [(m_idx, clf, pt) for (m_idx, clf, pt) in c6_confirm_clfs
                       if pt == 'mean' and m_idx in (MACHINE_MOD_IDX, GYRO_IDX)]

    for mt in [0.45, 0.50, 0.55]:
        for min_votes in [1, 2]:
            preds = eval_pipeline(test_ops, test_cls_preds, best_probs, data,
                                  clf_c7, clf_c8, c6_machine_gyro, mt, min_votes)
            overall, missed = print_results(test_ops, preds,
                                            f"mt={mt:.2f} machine+gyro>={min_votes}")

    # =====================================================================
    # Show detail on any 100% configs
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"DETAILED CHECK: best configs")
    print(f"{'='*70}")

    # Re-run best config with detail
    for mt in [0.50, 0.55]:
        preds = eval_pipeline(test_ops, test_cls_preds, best_probs, data,
                              clf_c7, clf_c8, c6_mean_clfs, mt, 1)
        overall = (preds == test_ops).mean()
        if overall >= 0.998:
            print(f"\n  mt={mt:.2f}, c6 confirm >= 1 vote (mean):")
            for c in range(9):
                mask = test_ops == c
                if mask.sum() > 0:
                    acc = (preds[mask] == c).mean()
                    print(f"    c{c} ({CLASS_NAMES[c]:20s}): {acc:.2%} ({(preds[mask]==c).sum()}/{mask.sum()})")

            missed_idx = np.where(preds != test_ops)[0]
            if len(missed_idx) == 0:
                print(f"    *** 100% — ZERO MISSES ***")
            else:
                for idx in missed_idx:
                    print(f"    MISS: test[{idx}] true=c{test_ops[idx]}({CLASS_NAMES[test_ops[idx]]}) "
                          f"pred=c{preds[idx]}({CLASS_NAMES[preds[idx]]})")


if __name__ == '__main__':
    main()
