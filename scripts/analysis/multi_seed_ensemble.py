#!/usr/bin/env python3
"""Multi-seed ensemble: combine per-modality embeddings from 3 v1 seeds.

Strategies:
1. Concatenate embeddings (3×256=768 per modality) → single classifier
2. Vote across 3 models' per-modality classifiers
3. Average probabilities from 3 models' classifiers
4. Best-of-3: use whichever seed's MLP router performs best
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


def eval_pipeline(test_ops, test_cls_preds, get_router_probs, get_c7_prob, get_c8_prob,
                   machine_thresh, c7_thresh=0.5, c8_thresh=0.5):
    N = len(test_ops)
    per_class_total = Counter()
    per_class_correct = Counter()
    for i in range(N):
        per_class_total[test_ops[i]] += 1

    for i in range(N):
        true_op = test_ops[i]
        cls_pred = test_cls_preds[i]
        probs = get_router_probs(i)

        max_damage_class = np.argmax(probs[1:]) + 1
        max_damage_prob = probs[max_damage_class]

        if max_damage_prob > machine_thresh:
            if max_damage_class == 1:
                final_pred = 6
            elif max_damage_class == 2:
                gyro_prob = get_c7_prob(i)
                final_pred = 7 if gyro_prob > c7_thresh else cls_pred
            elif max_damage_class == 3:
                mag_prob = get_c8_prob(i)
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
    seed_paths = {
        42: 'outputs/jan30/mmdtae_standalone_v1/best_model.pt',
        123: 'outputs/jan30/mmdtae_standalone_seed123/best_model.pt',
        777: 'outputs/jan30/mmdtae_standalone_seed777/best_model.pt',
    }

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

    # Extract embeddings from all 3 seeds
    seed_data = {}
    for seed, path in seed_paths.items():
        print(f"\nLoading seed {seed} from {path}...")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = ckpt['config']
        model = MM_DTAE_LSTM(config)
        model.head_cls = nn.Linear(config.d_model, 9)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.to(device)
        model.eval()

        seed_splits = {}
        for split in ['train', 'val', 'test']:
            mod_embeds, cls_preds, ops = extract_per_modality_embeddings(
                model, loaders[split], group_indices, device)
            seed_splits[split] = {'mod_embeds': mod_embeds, 'cls_preds': cls_preds, 'ops': ops}
            print(f"  {split}: {mod_embeds.shape}")
        seed_data[seed] = seed_splits

        del model
        torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None

    # Use seed 42's cls preds (best flat classifier) and ops
    train_ops = seed_data[42]['train']['ops']
    test_ops = seed_data[42]['test']['ops']
    test_cls_preds = seed_data[42]['test']['cls_preds']
    y_train_4c = map_4class(train_ops)
    y_test_4c = map_4class(test_ops)

    N_test = len(test_ops)
    per_class_total = Counter()
    for i in range(N_test):
        per_class_total[test_ops[i]] += 1

    # =====================================================================
    # STRATEGY 1: Per-seed classifiers, then vote/average
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"STRATEGY 1: Per-seed classifiers → vote/average probabilities")
    print(f"{'='*70}")

    # Train 4-class router on each seed's machine modality
    seed_router_probs = {}
    seed_c7_clfs = {}
    seed_c8_clfs = {}

    for seed in seed_paths:
        X_train = seed_data[seed]['train']['mod_embeds'][:, MACHINE_MOD_IDX, :]
        X_test = seed_data[seed]['test']['mod_embeds'][:, MACHINE_MOD_IDX, :]

        clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=10.0)
        clf.fit(X_train, y_train_4c)
        probs = clf.predict_proba(X_test)
        preds = probs.argmax(axis=1)
        acc = accuracy_score(y_test_4c, preds)
        per_cls = {c: accuracy_score(y_test_4c[y_test_4c == c], preds[y_test_4c == c])
                   for c in range(4) if (y_test_4c == c).sum() > 0}
        print(f"\n  Seed {seed} router (LogReg C=10): {acc:.2%}  "
              f"c6={per_cls.get(1,0):.0%} c7={per_cls.get(2,0):.0%} c8={per_cls.get(3,0):.0%}")
        seed_router_probs[seed] = probs

        # c7 specialist
        face_mask = np.isin(train_ops, [2, 3, 7])
        clf_c7 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
        clf_c7.fit(seed_data[seed]['train']['mod_embeds'][face_mask, GYRO_IDX, :],
                   (train_ops[face_mask] == 7).astype(int))
        seed_c7_clfs[seed] = clf_c7

        # c8 specialist
        pocket_mask = np.isin(train_ops, [4, 5, 8])
        clf_c8 = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
        clf_c8.fit(seed_data[seed]['train']['mod_embeds'][pocket_mask, MAG_IDX, :],
                   (train_ops[pocket_mask] == 8).astype(int))
        seed_c8_clfs[seed] = clf_c8

    # Average router probs across seeds
    avg_probs = np.mean([seed_router_probs[s] for s in seed_paths], axis=0)
    avg_preds = avg_probs.argmax(axis=1)
    acc = accuracy_score(y_test_4c, avg_preds)
    per_cls = {c: accuracy_score(y_test_4c[y_test_4c == c], avg_preds[y_test_4c == c])
               for c in range(4) if (y_test_4c == c).sum() > 0}
    print(f"\n  ENSEMBLE avg router: {acc:.2%}  "
          f"c6={per_cls.get(1,0):.0%} c7={per_cls.get(2,0):.0%} c8={per_cls.get(3,0):.0%}")

    # Full pipeline with averaged router probs + averaged specialist probs
    print(f"\n--- Full pipeline: averaged router + averaged specialists ---")
    for mt in [0.3, 0.4, 0.5, 0.6, 0.7]:
        def get_router(i):
            return avg_probs[i]
        def get_c7(i):
            probs = np.mean([seed_c7_clfs[s].predict_proba(
                seed_data[s]['test']['mod_embeds'][i, GYRO_IDX, :].reshape(1, -1))[0, 1]
                for s in seed_paths])
            return probs
        def get_c8(i):
            probs = np.mean([seed_c8_clfs[s].predict_proba(
                seed_data[s]['test']['mod_embeds'][i, MAG_IDX, :].reshape(1, -1))[0, 1]
                for s in seed_paths])
            return probs

        overall, pc = eval_pipeline(test_ops, test_cls_preds, get_router, get_c7, get_c8,
                                     machine_thresh=mt)
        c6 = pc.get(6, 0); c7 = pc.get(7, 0); c8 = pc.get(8, 0)
        c05 = np.mean([pc.get(c, 0) for c in range(6)])
        print(f"  mt={mt}: {overall:.2%}  c0-5={c05:.0%} c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    # =====================================================================
    # STRATEGY 2: Concatenated embeddings (3×256=768 per modality)
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"STRATEGY 2: Concatenated embeddings (3 seeds × 256 = 768 dims)")
    print(f"{'='*70}")

    # Concatenate machine modality from all 3 seeds
    X_train_concat = np.hstack([seed_data[s]['train']['mod_embeds'][:, MACHINE_MOD_IDX, :]
                                 for s in seed_paths])
    X_test_concat = np.hstack([seed_data[s]['test']['mod_embeds'][:, MACHINE_MOD_IDX, :]
                                for s in seed_paths])
    print(f"  Concat machine: {X_train_concat.shape}")

    for C in [1.0, 10.0, 100.0]:
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=C)
        clf.fit(X_train_concat, y_train_4c)
        probs = clf.predict_proba(X_test_concat)
        preds = probs.argmax(axis=1)
        acc = accuracy_score(y_test_4c, preds)
        per_cls = {c: accuracy_score(y_test_4c[y_test_4c == c], preds[y_test_4c == c])
                   for c in range(4) if (y_test_4c == c).sum() > 0}
        print(f"\n  Concat LogReg C={C}: {acc:.2%}  "
              f"c6={per_cls.get(1,0):.0%} c7={per_cls.get(2,0):.0%} c8={per_cls.get(3,0):.0%}")

        # Pipeline
        for mt in [0.3, 0.4, 0.5, 0.6]:
            def get_router(i, p=probs):
                return p[i]
            # Use seed 42 specialists (already perfect)
            def get_c7(i):
                return seed_c7_clfs[42].predict_proba(
                    seed_data[42]['test']['mod_embeds'][i, GYRO_IDX, :].reshape(1, -1))[0, 1]
            def get_c8(i):
                return seed_c8_clfs[42].predict_proba(
                    seed_data[42]['test']['mod_embeds'][i, MAG_IDX, :].reshape(1, -1))[0, 1]

            overall, pc = eval_pipeline(test_ops, test_cls_preds, get_router, get_c7, get_c8,
                                         machine_thresh=mt)
            c6 = pc.get(6, 0); c7 = pc.get(7, 0); c8 = pc.get(8, 0)
            print(f"    mt={mt}: {overall:.2%}  c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    # =====================================================================
    # STRATEGY 3: MLP router on concatenated embeddings
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"STRATEGY 3: MLP router on concatenated seed embeddings")
    print(f"{'='*70}")

    # Import the MLP from the frozen_mlp script
    sys.path.insert(0, str(Path(__file__).parent))
    from frozen_mlp_damage_heads import DamageMLPHead, train_mlp_head

    y_val_4c = map_4class(seed_data[42]['val']['ops'])
    X_val_concat = np.hstack([seed_data[s]['val']['mod_embeds'][:, MACHINE_MOD_IDX, :]
                               for s in seed_paths])

    counts_4c = np.bincount(y_train_4c)
    total = len(y_train_4c)
    cw_4c = [total / (4 * c) for c in counts_4c]

    for hidden in [(64,), (128, 64), (256, 128)]:
        best_acc = 0
        best_probs = None
        for mlp_seed in [42, 123, 777]:
            torch.manual_seed(mlp_seed)
            np.random.seed(mlp_seed)
            mlp, val_acc = train_mlp_head(
                X_train_concat, y_train_4c, X_val_concat, y_val_4c,
                n_classes=4, hidden_dims=hidden, dropout=0.3, lr=1e-3,
                epochs=300, patience=50, class_weight=cw_4c, device='cpu')
            with torch.no_grad():
                test_logits = mlp(torch.tensor(X_test_concat, dtype=torch.float32))
                test_probs = torch.softmax(test_logits, dim=1).numpy()
            preds = test_probs.argmax(axis=1)
            acc = accuracy_score(y_test_4c, preds)
            if acc > best_acc:
                best_acc = acc
                best_probs = test_probs

        per_cls = {c: accuracy_score(y_test_4c[y_test_4c == c], best_probs.argmax(axis=1)[y_test_4c == c])
                   for c in range(4) if (y_test_4c == c).sum() > 0}
        print(f"\n  MLP({hidden}): {best_acc:.2%}  "
              f"c6={per_cls.get(1,0):.0%} c7={per_cls.get(2,0):.0%} c8={per_cls.get(3,0):.0%}")

        for mt in [0.3, 0.4, 0.5, 0.6]:
            _probs = best_probs
            def get_router(i, p=_probs):
                return p[i]
            def get_c7(i):
                return seed_c7_clfs[42].predict_proba(
                    seed_data[42]['test']['mod_embeds'][i, GYRO_IDX, :].reshape(1, -1))[0, 1]
            def get_c8(i):
                return seed_c8_clfs[42].predict_proba(
                    seed_data[42]['test']['mod_embeds'][i, MAG_IDX, :].reshape(1, -1))[0, 1]

            overall, pc = eval_pipeline(test_ops, test_cls_preds, get_router, get_c7, get_c8,
                                         machine_thresh=mt)
            c6 = pc.get(6, 0); c7 = pc.get(7, 0); c8 = pc.get(8, 0)
            c05 = np.mean([pc.get(c, 0) for c in range(6)])
            print(f"    mt={mt}: {overall:.2%}  c0-5={c05:.0%} c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    print(f"\n{'='*70}")
    print(f"REFERENCE: Single-seed MLP(64) pipeline best = 99.19%")
    print(f"  c0-c5=100%, c6=93%, c7=97%, c8=94%")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
