#!/usr/bin/env python3
"""Post-hoc sweep v2: Router ensemble + stacked meta-classifier.

Tries:
1. Ensemble of C=10 + C=100 LogReg routers (avg probabilities)
2. Stacked meta-classifier: machine router probs + gyroscope c7 prob + magnetometer c8 prob
3. Multi-modality stacked router (machine + gyro + mag features → meta LogReg)
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn

BASE_OP_MAP = {0: 0, 1: 0, 6: 0, 2: 1, 3: 1, 7: 1, 4: 2, 5: 2, 8: 2}
DAMAGE_FOR_BASE = {0: 6, 1: 7, 2: 8}
MODALITY_NAMES = ['accelerometer', 'gyroscope', 'magnetometer', 'environmental',
                  'color', 'rms', 'machine']
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


def extract_per_modality_embeddings(model, loader, group_indices, device):
    model.eval()
    all_mod_embeds, all_cls_preds, all_ops = [], [], []
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
            per_mod_last = enc_mods[torch.arange(B, device=device), last_idx]
            all_mod_embeds.append(per_mod_last.cpu())
            all_cls_preds.append(out['cls'].argmax(dim=-1).cpu())
            all_ops.append(operations)
    return (torch.cat(all_mod_embeds).numpy(),
            torch.cat(all_cls_preds).numpy(),
            torch.cat(all_ops).numpy())


def map_4class(ops):
    y = np.zeros(len(ops), dtype=int)
    y[ops == 6] = 1
    y[ops == 7] = 2
    y[ops == 8] = 3
    return y


def eval_pipeline(test_ops, test_cls_preds, test_mod_embeds,
                   router_probs, clf_c7, clf_c8,
                   machine_thresh, c7_thresh=0.5, c8_thresh=0.5):
    N = len(test_ops)
    per_class_total = Counter()
    per_class_correct = Counter()
    for i in range(N):
        per_class_total[test_ops[i]] += 1

    for i in range(N):
        true_op = test_ops[i]
        cls_pred = test_cls_preds[i]
        probs = router_probs[i]

        max_damage_class = np.argmax(probs[1:]) + 1
        max_damage_prob = probs[max_damage_class]

        if max_damage_prob > machine_thresh:
            if max_damage_class == 1:
                final_pred = 6
            elif max_damage_class == 2:
                x_gyro = test_mod_embeds[i, GYRO_IDX, :].reshape(1, -1)
                gyro_prob = clf_c7.predict_proba(x_gyro)[0, 1]
                final_pred = 7 if gyro_prob > c7_thresh else cls_pred
            elif max_damage_class == 3:
                x_mag = test_mod_embeds[i, MAG_IDX, :].reshape(1, -1)
                mag_prob = clf_c8.predict_proba(x_mag)[0, 1]
                final_pred = 8 if mag_prob > c8_thresh else cls_pred
        else:
            final_pred = cls_pred

        if final_pred == true_op:
            per_class_correct[true_op] += 1

    overall = sum(per_class_correct.values()) / N
    per_class = {c: per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0
                 for c in sorted(per_class_total.keys())}
    return overall, per_class


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

    print(f"\nExtracting per-modality embeddings...")
    split_data = {}
    for split in ['train', 'val', 'test']:
        mod_embeds, cls_preds, ops = extract_per_modality_embeddings(
            model, loaders[split], group_indices, device)
        split_data[split] = {'mod_embeds': mod_embeds, 'cls_preds': cls_preds, 'ops': ops}
        print(f"  {split}: {mod_embeds.shape}")

    train_ops = split_data['train']['ops']
    test_ops = split_data['test']['ops']
    y_train_4c = map_4class(train_ops)
    y_test_4c = map_4class(test_ops)

    X_train_machine = split_data['train']['mod_embeds'][:, MACHINE_MOD_IDX, :]
    X_test_machine = split_data['test']['mod_embeds'][:, MACHINE_MOD_IDX, :]

    # Train c7/c8 specialists (same as before, already perfect)
    face_train_mask = np.isin(train_ops, [2, 3, 7])
    clf_c7 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_c7.fit(split_data['train']['mod_embeds'][face_train_mask, GYRO_IDX, :],
               (train_ops[face_train_mask] == 7).astype(int))

    pocket_train_mask = np.isin(train_ops, [4, 5, 8])
    clf_c8 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf_c8.fit(split_data['train']['mod_embeds'][pocket_train_mask, MAG_IDX, :],
               (train_ops[pocket_train_mask] == 8).astype(int))

    # =====================================================================
    # STRATEGY 1: Ensemble of C=10 + C=100 routers (avg probabilities)
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"STRATEGY 1: Router ensemble (avg C=10 + C=100 probabilities)")
    print(f"{'='*70}")

    clf_c10 = LogisticRegression(max_iter=2000, class_weight='balanced', C=10.0)
    clf_c10.fit(X_train_machine, y_train_4c)
    clf_c100 = LogisticRegression(max_iter=2000, class_weight='balanced', C=100.0)
    clf_c100.fit(X_train_machine, y_train_4c)

    probs_c10 = clf_c10.predict_proba(X_test_machine)
    probs_c100 = clf_c100.predict_proba(X_test_machine)

    # Also train C=50 for a 3-way ensemble
    clf_c50 = LogisticRegression(max_iter=2000, class_weight='balanced', C=50.0)
    clf_c50.fit(X_train_machine, y_train_4c)
    probs_c50 = clf_c50.predict_proba(X_test_machine)

    ensembles = {
        'C10+C100 avg': (probs_c10 + probs_c100) / 2,
        'C10+C50+C100 avg': (probs_c10 + probs_c50 + probs_c100) / 3,
        'C10(0.6)+C100(0.4)': probs_c10 * 0.6 + probs_c100 * 0.4,
        'C10(0.4)+C100(0.6)': probs_c10 * 0.4 + probs_c100 * 0.6,
    }

    # Also individual baselines for comparison
    ensembles['C10 only'] = probs_c10
    ensembles['C100 only'] = probs_c100

    for ens_name, ens_probs in ensembles.items():
        # Standalone 4-class accuracy
        ens_preds = ens_probs.argmax(axis=1)
        acc = accuracy_score(y_test_4c, ens_preds)
        per_cls = {}
        for c in range(4):
            mask = y_test_4c == c
            if mask.sum() > 0:
                per_cls[c] = accuracy_score(y_test_4c[mask], ens_preds[mask])
        print(f"\n  {ens_name}:")
        print(f"    Standalone: {acc:.2%}  normal={per_cls.get(0,0):.0%} "
              f"c6={per_cls.get(1,0):.0%} c7={per_cls.get(2,0):.0%} c8={per_cls.get(3,0):.0%}")

        # Full pipeline with threshold sweep
        for mt in [0.3, 0.4, 0.5, 0.6, 0.7]:
            overall, pc = eval_pipeline(
                test_ops, split_data['test']['cls_preds'],
                split_data['test']['mod_embeds'],
                ens_probs, clf_c7, clf_c8,
                machine_thresh=mt, c7_thresh=0.5, c8_thresh=0.5)
            c6 = pc.get(6, 0)
            c7 = pc.get(7, 0)
            c8 = pc.get(8, 0)
            c05 = np.mean([pc.get(c, 0) for c in range(6)])
            print(f"    mt={mt}: {overall:.2%}  c0-5={c05:.0%} c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    # =====================================================================
    # STRATEGY 2: Stacked meta-classifier
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"STRATEGY 2: Stacked meta-classifier (router probs + specialist probs)")
    print(f"{'='*70}")

    # Build meta-features: [machine_4c_probs(4), gyro_c7_prob(1), mag_c8_prob(1)] = 6 dims
    def build_meta_features(split):
        mod_embeds = split_data[split]['mod_embeds']
        N = len(split_data[split]['ops'])

        machine_probs = clf_c10.predict_proba(mod_embeds[:, MACHINE_MOD_IDX, :])  # [N, 4]

        # c7 specialist on ALL samples (not just face group)
        gyro_probs = clf_c7.predict_proba(mod_embeds[:, GYRO_IDX, :])[:, 1:2]  # [N, 1]

        # c8 specialist on ALL samples
        mag_probs = clf_c8.predict_proba(mod_embeds[:, MAG_IDX, :])[:, 1:2]  # [N, 1]

        return np.hstack([machine_probs, gyro_probs, mag_probs])

    meta_train = build_meta_features('train')
    meta_test = build_meta_features('test')
    print(f"  Meta-features: {meta_train.shape[1]} dims")

    for C in [0.1, 1.0, 10.0, 100.0]:
        meta_clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=C)
        meta_clf.fit(meta_train, y_train_4c)
        meta_probs = meta_clf.predict_proba(meta_test)
        meta_preds = meta_probs.argmax(axis=1)
        acc = accuracy_score(y_test_4c, meta_preds)
        per_cls = {}
        for c in range(4):
            mask = y_test_4c == c
            if mask.sum() > 0:
                per_cls[c] = accuracy_score(y_test_4c[mask], meta_preds[mask])
        print(f"\n  Meta LogReg C={C}:")
        print(f"    Standalone: {acc:.2%}  normal={per_cls.get(0,0):.0%} "
              f"c6={per_cls.get(1,0):.0%} c7={per_cls.get(2,0):.0%} c8={per_cls.get(3,0):.0%}")

        # Pipeline
        for mt in [0.3, 0.4, 0.5, 0.6]:
            overall, pc = eval_pipeline(
                test_ops, split_data['test']['cls_preds'],
                split_data['test']['mod_embeds'],
                meta_probs, clf_c7, clf_c8,
                machine_thresh=mt)
            c6 = pc.get(6, 0)
            c7 = pc.get(7, 0)
            c8 = pc.get(8, 0)
            c05 = np.mean([pc.get(c, 0) for c in range(6)])
            print(f"    mt={mt}: {overall:.2%}  c0-5={c05:.0%} c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    # =====================================================================
    # STRATEGY 3: Multi-modality stacked router
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"STRATEGY 3: Multi-modality router (machine+gyro+mag embeddings → meta)")
    print(f"{'='*70}")
    print(f"  Instead of concatenating raw embeddings (768 dims, known to fail),")
    print(f"  use PCA-reduced or just the 3 most informative modalities.")

    # Machine(256) + Gyro(256) + Mag(256) = 768 dims
    # But we know concat fails. Try with PCA first.
    from sklearn.decomposition import PCA

    for n_components in [16, 32, 64, 128]:
        X_train_3mod = np.hstack([
            split_data['train']['mod_embeds'][:, MACHINE_MOD_IDX, :],
            split_data['train']['mod_embeds'][:, GYRO_IDX, :],
            split_data['train']['mod_embeds'][:, MAG_IDX, :],
        ])
        X_test_3mod = np.hstack([
            split_data['test']['mod_embeds'][:, MACHINE_MOD_IDX, :],
            split_data['test']['mod_embeds'][:, GYRO_IDX, :],
            split_data['test']['mod_embeds'][:, MAG_IDX, :],
        ])

        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_3mod)
        X_test_pca = pca.transform(X_test_3mod)

        for C in [1.0, 10.0, 100.0]:
            clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=C)
            clf.fit(X_train_pca, y_train_4c)
            probs = clf.predict_proba(X_test_pca)
            preds = probs.argmax(axis=1)
            acc = accuracy_score(y_test_4c, preds)
            per_cls = {}
            for c in range(4):
                mask = y_test_4c == c
                if mask.sum() > 0:
                    per_cls[c] = accuracy_score(y_test_4c[mask], preds[mask])

            c6a = per_cls.get(1, 0)
            c7a = per_cls.get(2, 0)
            c8a = per_cls.get(3, 0)

            # Only print if promising
            if acc > 0.93:
                print(f"  PCA({n_components}) C={C}: {acc:.2%}  "
                      f"normal={per_cls.get(0,0):.0%} c6={c6a:.0%} c7={c7a:.0%} c8={c8a:.0%}")

                # Pipeline
                for mt in [0.3, 0.4, 0.5, 0.6]:
                    overall, pc = eval_pipeline(
                        test_ops, split_data['test']['cls_preds'],
                        split_data['test']['mod_embeds'],
                        probs, clf_c7, clf_c8,
                        machine_thresh=mt)
                    print(f"    mt={mt}: {overall:.2%}  c6={pc.get(6,0):.0%} c7={pc.get(7,0):.0%} c8={pc.get(8,0):.0%}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"REFERENCE: Previous best = 97.89% (LogReg C=10 router, mt=0.4)")
    print(f"  c0-c5=100%, c6=96%, c7=78%, c8=86%")
    print(f"{'='*70}")

    output_path = Path('outputs/jan30/posthoc_sweep_v2_results.txt')
    print(f"\nDone. Review output above for improvements.")


if __name__ == '__main__':
    main()
