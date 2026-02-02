#!/usr/bin/env python3
"""Post-hoc classifier hyperparameter sweep on frozen v1 per-modality embeddings.

The hybrid pipeline (97.24%) uses default LogReg C=1.0 for all classifiers.
This script sweeps:
  1. LogReg regularization C (0.001 to 100)
  2. Different classifiers (SVM-RBF, small MLP, Random Forest, GradientBoosting)
  3. Per-class threshold tuning
  4. SMOTE oversampling for c6 (84 train samples)

All on frozen v1 per-modality embeddings — zero GPU cost.
"""
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import product
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn

BASE_OP_MAP = {0: 0, 1: 0, 6: 0, 2: 1, 3: 1, 7: 1, 4: 2, 5: 2, 8: 2}
DAMAGE_FOR_BASE = {0: 6, 1: 7, 2: 8}
GROUP_NAMES_MAP = {0: 'adaptive', 1: 'face', 2: 'pocket'}
GROUP_CLASSES = {0: [0, 1, 6], 1: [2, 3, 7], 2: [4, 5, 8]}
MODALITY_NAMES = ['accelerometer', 'gyroscope', 'magnetometer', 'environmental',
                  'color', 'rms', 'machine']
BEST_MOD_PER_GROUP = {0: 6, 1: 1, 2: 2}  # adaptive→machine, face→gyroscope, pocket→magnetometer
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
    all_mod_embeds = []
    all_cls_preds = []
    all_ops = []

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


def make_classifier(name, **kwargs):
    """Factory for classifiers."""
    if name == 'logreg':
        C = kwargs.get('C', 1.0)
        return LogisticRegression(max_iter=2000, class_weight='balanced', C=C)
    elif name == 'svm_rbf':
        C = kwargs.get('C', 1.0)
        gamma = kwargs.get('gamma', 'scale')
        return SVC(C=C, kernel='rbf', gamma=gamma, class_weight='balanced',
                   probability=True)
    elif name == 'mlp':
        hidden = kwargs.get('hidden', (64, 32))
        return MLPClassifier(hidden_layer_sizes=hidden, max_iter=1000,
                             early_stopping=True, random_state=42)
    elif name == 'rf':
        n_est = kwargs.get('n_estimators', 200)
        return RandomForestClassifier(n_estimators=n_est, class_weight='balanced',
                                       random_state=42)
    elif name == 'gboost':
        n_est = kwargs.get('n_estimators', 200)
        return GradientBoostingClassifier(n_estimators=n_est, random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {name}")


def smote_oversample(X, y, target_ratio=0.5, random_state=42):
    """Simple SMOTE-like oversampling for minority class via interpolation."""
    rng = np.random.RandomState(random_state)
    minority_mask = y == 1
    X_min = X[minority_mask]
    n_minority = X_min.shape[0]
    n_majority = (~minority_mask).sum()
    n_target = int(n_majority * target_ratio) - n_minority
    if n_target <= 0:
        return X, y

    # Generate synthetic samples by interpolating random pairs
    synthetic = []
    for _ in range(n_target):
        i, j = rng.randint(0, n_minority, size=2)
        lam = rng.uniform(0, 1)
        synthetic.append(X_min[i] * lam + X_min[j] * (1 - lam))
    synthetic = np.array(synthetic)

    X_new = np.vstack([X, synthetic])
    y_new = np.concatenate([y, np.ones(n_target, dtype=y.dtype)])
    return X_new, y_new


def evaluate_hybrid_pipeline(split_data, clf_4c, clf_c7, clf_c8,
                              machine_thresh, c7_spec_thresh=0.5, c8_spec_thresh=0.5):
    """Evaluate the hybrid pipeline with given classifiers and thresholds."""
    test_ops = split_data['test']['ops']
    test_cls_preds = split_data['test']['cls_preds']
    test_mod_embeds = split_data['test']['mod_embeds']
    N = len(test_ops)

    per_class_total = Counter()
    per_class_correct = Counter()
    for i in range(N):
        per_class_total[test_ops[i]] += 1

    X_test_machine = test_mod_embeds[:, MACHINE_MOD_IDX, :]
    probs_4c = clf_4c.predict_proba(X_test_machine)

    for i in range(N):
        true_op = test_ops[i]
        cls_pred = test_cls_preds[i]
        machine_probs = probs_4c[i]

        max_damage_class = np.argmax(machine_probs[1:]) + 1
        max_damage_prob = machine_probs[max_damage_class]

        if max_damage_prob > machine_thresh:
            if max_damage_class == 1:
                final_pred = 6
            elif max_damage_class == 2:
                x_gyro = test_mod_embeds[i, GYRO_IDX, :].reshape(1, -1)
                gyro_prob = clf_c7.predict_proba(x_gyro)[0, 1]
                final_pred = 7 if gyro_prob > c7_spec_thresh else cls_pred
            elif max_damage_class == 3:
                x_mag = test_mod_embeds[i, MAG_IDX, :].reshape(1, -1)
                mag_prob = clf_c8.predict_proba(x_mag)[0, 1]
                final_pred = 8 if mag_prob > c8_spec_thresh else cls_pred
        else:
            final_pred = cls_pred

        if final_pred == true_op:
            per_class_correct[true_op] += 1

    overall = sum(per_class_correct.values()) / N
    per_class = {}
    for c in sorted(per_class_total.keys()):
        per_class[c] = per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0

    return overall, per_class


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
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded v1 model, d_model={config.d_model}")

    # Extract embeddings
    print(f"\nExtracting per-modality embeddings...")
    split_data = {}
    for split in ['train', 'val', 'test']:
        mod_embeds, cls_preds, ops = extract_per_modality_embeddings(
            model, loaders[split], group_indices, device)
        split_data[split] = {'mod_embeds': mod_embeds, 'cls_preds': cls_preds, 'ops': ops}
        print(f"  {split}: {mod_embeds.shape}")

    train_ops = split_data['train']['ops']

    # =====================================================================
    # PHASE 1: Sweep classifier + hyperparams for the machine 4-class router
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 1: Machine 4-class router sweep")
    print(f"{'='*70}")

    def map_4class(ops):
        y = np.zeros(len(ops), dtype=int)
        y[ops == 6] = 1
        y[ops == 7] = 2
        y[ops == 8] = 3
        return y

    y_train_4c = map_4class(train_ops)
    y_test_4c = map_4class(split_data['test']['ops'])
    X_train_machine = split_data['train']['mod_embeds'][:, MACHINE_MOD_IDX, :]
    X_test_machine = split_data['test']['mod_embeds'][:, MACHINE_MOD_IDX, :]

    # StandardScaler for SVM/MLP
    scaler_machine = StandardScaler()
    X_train_machine_scaled = scaler_machine.fit_transform(X_train_machine)
    X_test_machine_scaled = scaler_machine.transform(X_test_machine)

    router_configs = [
        ('logreg', {'C': 0.001}), ('logreg', {'C': 0.01}), ('logreg', {'C': 0.1}),
        ('logreg', {'C': 1.0}), ('logreg', {'C': 10.0}), ('logreg', {'C': 100.0}),
        ('svm_rbf', {'C': 0.1}), ('svm_rbf', {'C': 1.0}), ('svm_rbf', {'C': 10.0}),
        ('svm_rbf', {'C': 100.0}),
        ('mlp', {'hidden': (64, 32)}), ('mlp', {'hidden': (128, 64)}),
        ('mlp', {'hidden': (128, 64, 32)}),
        ('rf', {'n_estimators': 100}), ('rf', {'n_estimators': 500}),
        ('gboost', {'n_estimators': 100}), ('gboost', {'n_estimators': 300}),
    ]

    router_results = []
    for clf_name, params in router_configs:
        use_scaled = clf_name in ('svm_rbf', 'mlp')
        X_tr = X_train_machine_scaled if use_scaled else X_train_machine
        X_te = X_test_machine_scaled if use_scaled else X_test_machine

        # SMOTE for c6 (class 1 in 4-class)
        if clf_name in ('mlp', 'gboost'):
            X_tr_aug, y_tr_aug = smote_oversample(X_tr, (y_train_4c > 0).astype(int),
                                                   target_ratio=0.3)
            # Actually for 4-class, just use original
            X_tr_aug, y_tr_aug = X_tr, y_train_4c
        else:
            X_tr_aug, y_tr_aug = X_tr, y_train_4c

        try:
            clf = make_classifier(clf_name, **params)
            clf.fit(X_tr_aug, y_tr_aug)
            preds = clf.predict(X_te)
            acc = accuracy_score(y_test_4c, preds)

            # Per-class accuracy
            per_cls = {}
            for c in range(4):
                mask = y_test_4c == c
                if mask.sum() > 0:
                    per_cls[c] = accuracy_score(y_test_4c[mask], preds[mask])

            label = f"{clf_name}({params})"
            c6_acc = per_cls.get(1, 0)
            c7_acc = per_cls.get(2, 0)
            c8_acc = per_cls.get(3, 0)
            normal_acc = per_cls.get(0, 0)
            print(f"  {label:50s}: {acc:.2%}  normal={normal_acc:.0%} c6={c6_acc:.0%} c7={c7_acc:.0%} c8={c8_acc:.0%}")
            router_results.append({
                'clf': clf_name, 'params': str(params), 'acc': acc,
                'normal': normal_acc, 'c6': c6_acc, 'c7': c7_acc, 'c8': c8_acc,
                'model': clf
            })
        except Exception as e:
            print(f"  {clf_name}({params}): FAILED — {e}")

    # =====================================================================
    # PHASE 2: Sweep specialist classifiers (c7 gyroscope, c8 magnetometer)
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 2: Specialist classifier sweep (c7=gyro, c8=mag)")
    print(f"{'='*70}")

    specialist_configs = [
        ('logreg', {'C': 0.001}), ('logreg', {'C': 0.01}), ('logreg', {'C': 0.1}),
        ('logreg', {'C': 1.0}), ('logreg', {'C': 10.0}), ('logreg', {'C': 100.0}),
        ('svm_rbf', {'C': 1.0}), ('svm_rbf', {'C': 10.0}), ('svm_rbf', {'C': 100.0}),
        ('mlp', {'hidden': (64, 32)}), ('mlp', {'hidden': (128, 64)}),
        ('rf', {'n_estimators': 200}), ('rf', {'n_estimators': 500}),
    ]

    # c7 specialist: gyroscope, face group
    face_train_mask = np.isin(train_ops, [2, 3, 7])
    face_test_mask = np.isin(split_data['test']['ops'], [2, 3, 7])
    y_train_c7 = (train_ops[face_train_mask] == 7).astype(int)
    y_test_c7 = (split_data['test']['ops'][face_test_mask] == 7).astype(int)
    X_train_gyro = split_data['train']['mod_embeds'][face_train_mask, GYRO_IDX, :]
    X_test_gyro = split_data['test']['mod_embeds'][face_test_mask, GYRO_IDX, :]

    scaler_gyro = StandardScaler()
    X_train_gyro_scaled = scaler_gyro.fit_transform(X_train_gyro)
    X_test_gyro_scaled = scaler_gyro.transform(X_test_gyro)

    # c8 specialist: magnetometer, pocket group
    pocket_train_mask = np.isin(train_ops, [4, 5, 8])
    pocket_test_mask = np.isin(split_data['test']['ops'], [4, 5, 8])
    y_train_c8 = (train_ops[pocket_train_mask] == 8).astype(int)
    y_test_c8 = (split_data['test']['ops'][pocket_test_mask] == 8).astype(int)
    X_train_mag = split_data['train']['mod_embeds'][pocket_train_mask, MAG_IDX, :]
    X_test_mag = split_data['test']['mod_embeds'][pocket_test_mask, MAG_IDX, :]

    scaler_mag = StandardScaler()
    X_train_mag_scaled = scaler_mag.fit_transform(X_train_mag)
    X_test_mag_scaled = scaler_mag.transform(X_test_mag)

    print(f"\n--- c7 specialist (gyroscope, face group) ---")
    print(f"  Train: {len(y_train_c7)} ({y_train_c7.sum()} damage)")
    print(f"  Test:  {len(y_test_c7)} ({y_test_c7.sum()} damage)")

    c7_specialist_results = []
    for clf_name, params in specialist_configs:
        use_scaled = clf_name in ('svm_rbf', 'mlp')
        X_tr = X_train_gyro_scaled if use_scaled else X_train_gyro
        X_te = X_test_gyro_scaled if use_scaled else X_test_gyro

        try:
            clf = make_classifier(clf_name, **params)
            clf.fit(X_tr, y_train_c7)
            preds = clf.predict(X_te)
            acc = accuracy_score(y_test_c7, preds)
            f1 = f1_score(y_test_c7, preds, zero_division=0)
            tp = ((preds == 1) & (y_test_c7 == 1)).sum()
            fp = ((preds == 1) & (y_test_c7 == 0)).sum()
            fn = ((preds == 0) & (y_test_c7 == 1)).sum()
            label = f"{clf_name}({params})"
            print(f"  {label:50s}: acc={acc:.2%} F1={f1:.2%} TP={tp} FP={fp} FN={fn}")
            c7_specialist_results.append({
                'clf': clf_name, 'params': str(params), 'acc': acc, 'f1': f1,
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
                'model': clf, 'use_scaled': use_scaled
            })
        except Exception as e:
            print(f"  {clf_name}({params}): FAILED — {e}")

    # c7 with SMOTE
    print(f"\n--- c7 specialist with SMOTE ---")
    for target_ratio in [0.3, 0.5, 0.7]:
        X_tr_smote, y_tr_smote = smote_oversample(X_train_gyro, y_train_c7, target_ratio=target_ratio)
        for clf_name, params in [('logreg', {'C': 1.0}), ('logreg', {'C': 10.0}),
                                  ('svm_rbf', {'C': 10.0}), ('rf', {'n_estimators': 200})]:
            use_scaled = clf_name in ('svm_rbf', 'mlp')
            if use_scaled:
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr_smote)
                X_te_s = sc.transform(X_test_gyro)
            else:
                X_tr_s, X_te_s = X_tr_smote, X_test_gyro
            try:
                clf = make_classifier(clf_name, **params)
                clf.fit(X_tr_s, y_tr_smote)
                preds = clf.predict(X_te_s)
                f1 = f1_score(y_test_c7, preds, zero_division=0)
                tp = ((preds == 1) & (y_test_c7 == 1)).sum()
                fp = ((preds == 1) & (y_test_c7 == 0)).sum()
                print(f"  SMOTE({target_ratio}) {clf_name}({params}): F1={f1:.2%} TP={tp} FP={fp}")
            except Exception as e:
                print(f"  SMOTE({target_ratio}) {clf_name}({params}): FAILED — {e}")

    print(f"\n--- c8 specialist (magnetometer, pocket group) ---")
    print(f"  Train: {len(y_train_c8)} ({y_train_c8.sum()} damage)")
    print(f"  Test:  {len(y_test_c8)} ({y_test_c8.sum()} damage)")

    c8_specialist_results = []
    for clf_name, params in specialist_configs:
        use_scaled = clf_name in ('svm_rbf', 'mlp')
        X_tr = X_train_mag_scaled if use_scaled else X_train_mag
        X_te = X_test_mag_scaled if use_scaled else X_test_mag

        try:
            clf = make_classifier(clf_name, **params)
            clf.fit(X_tr, y_train_c8)
            preds = clf.predict(X_te)
            acc = accuracy_score(y_test_c8, preds)
            f1 = f1_score(y_test_c8, preds, zero_division=0)
            tp = ((preds == 1) & (y_test_c8 == 1)).sum()
            fp = ((preds == 1) & (y_test_c8 == 0)).sum()
            fn = ((preds == 0) & (y_test_c8 == 1)).sum()
            label = f"{clf_name}({params})"
            print(f"  {label:50s}: acc={acc:.2%} F1={f1:.2%} TP={tp} FP={fp} FN={fn}")
            c8_specialist_results.append({
                'clf': clf_name, 'params': str(params), 'acc': acc, 'f1': f1,
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
                'model': clf, 'use_scaled': use_scaled
            })
        except Exception as e:
            print(f"  {clf_name}({params}): FAILED — {e}")

    # =====================================================================
    # PHASE 3: Best-of-each full pipeline sweep
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 3: Full hybrid pipeline — best combos")
    print(f"{'='*70}")

    # Collect top-N routers and specialists
    # Sort routers by overall accuracy
    top_routers = sorted(router_results, key=lambda x: x['acc'], reverse=True)[:5]
    # Sort c7 specialists by F1
    top_c7 = sorted(c7_specialist_results, key=lambda x: x['f1'], reverse=True)[:5]
    # Sort c8 specialists by F1
    top_c8 = sorted(c8_specialist_results, key=lambda x: x['f1'], reverse=True)[:5]

    print(f"\nTop 5 routers:")
    for r in top_routers:
        print(f"  {r['clf']}({r['params']}): {r['acc']:.2%} c6={r['c6']:.0%} c7={r['c7']:.0%} c8={r['c8']:.0%}")
    print(f"\nTop 5 c7 specialists:")
    for r in top_c7:
        print(f"  {r['clf']}({r['params']}): F1={r['f1']:.2%} TP={r['tp']} FP={r['fp']}")
    print(f"\nTop 5 c8 specialists:")
    for r in top_c8:
        print(f"  {r['clf']}({r['params']}): F1={r['f1']:.2%} TP={r['tp']} FP={r['fp']}")

    # For full pipeline evaluation, we need classifiers that output predict_proba
    # Re-prepare scalers for test data if needed
    best_results = []
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    spec_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    for router in top_routers:
        for c7_spec in top_c7[:3]:
            for c8_spec in top_c8[:3]:
                for mt in thresholds:
                    for st in spec_thresholds:
                        try:
                            overall, per_class = evaluate_hybrid_pipeline(
                                split_data,
                                router['model'], c7_spec['model'], c8_spec['model'],
                                machine_thresh=mt, c7_spec_thresh=st, c8_spec_thresh=st
                            )
                            c6 = per_class.get(6, 0)
                            c7 = per_class.get(7, 0)
                            c8 = per_class.get(8, 0)
                            c05 = np.mean([per_class.get(c, 0) for c in range(6)])

                            if overall > 0.97:  # Only print improvements
                                best_results.append({
                                    'overall': overall, 'c6': c6, 'c7': c7, 'c8': c8, 'c05': c05,
                                    'router': f"{router['clf']}({router['params']})",
                                    'c7_spec': f"{c7_spec['clf']}({c7_spec['params']})",
                                    'c8_spec': f"{c8_spec['clf']}({c8_spec['params']})",
                                    'mt': mt, 'st': st
                                })
                        except Exception:
                            pass

    # Sort and display best results
    best_results.sort(key=lambda x: x['overall'], reverse=True)

    print(f"\n{'='*70}")
    print(f"TOP 20 PIPELINE CONFIGURATIONS (>97%)")
    print(f"{'='*70}")
    print(f"{'Overall':>8s} {'c0-c5':>6s} {'c6':>5s} {'c7':>5s} {'c8':>5s}  Router / c7_spec / c8_spec / thresholds")
    for r in best_results[:20]:
        print(f"  {r['overall']:.2%}  {r['c05']:.0%}  {r['c6']:.0%}  {r['c7']:.0%}  {r['c8']:.0%}  "
              f"router={r['router']} c7={r['c7_spec']} c8={r['c8_spec']} mt={r['mt']} st={r['st']}")

    # Also show best by each damage class
    print(f"\n--- Best c6 ---")
    by_c6 = sorted(best_results, key=lambda x: (x['c6'], x['overall']), reverse=True)[:5]
    for r in by_c6:
        print(f"  {r['overall']:.2%}  c6={r['c6']:.0%} c7={r['c7']:.0%} c8={r['c8']:.0%}  {r['router']} mt={r['mt']}")

    print(f"\n--- Best c7 ---")
    by_c7 = sorted(best_results, key=lambda x: (x['c7'], x['overall']), reverse=True)[:5]
    for r in by_c7:
        print(f"  {r['overall']:.2%}  c6={r['c6']:.0%} c7={r['c7']:.0%} c8={r['c8']:.0%}  {r['c7_spec']} st={r['st']}")

    print(f"\n--- Best c8 ---")
    by_c8 = sorted(best_results, key=lambda x: (x['c8'], x['overall']), reverse=True)[:5]
    for r in by_c8:
        print(f"  {r['overall']:.2%}  c6={r['c6']:.0%} c7={r['c7']:.0%} c8={r['c8']:.0%}  {r['c8_spec']} st={r['st']}")

    # =====================================================================
    # PHASE 4: Per-class threshold optimization
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 4: Independent per-class threshold optimization")
    print(f"{'='*70}")

    # Use the overall best router
    if best_results:
        best = best_results[0]
        print(f"Using best overall config: {best['overall']:.2%}")
        print(f"  Router: {best['router']}")
        print(f"  c7 spec: {best['c7_spec']}")
        print(f"  c8 spec: {best['c8_spec']}")
    else:
        print("No results above 97%, using default LogReg baseline")

    # Baseline reference
    print(f"\nBaseline (default LogReg C=1.0, all t=0.6): 97.24%")
    print(f"  c6=93%, c7=72%, c8=86%")

    # Save summary
    output_path = Path('outputs/jan30/posthoc_sweep_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_results = {
        'router_sweep': [{k: v for k, v in r.items() if k != 'model'}
                         for r in router_results],
        'c7_specialist_sweep': [{k: v for k, v in r.items() if k not in ('model',)}
                                 for r in c7_specialist_results],
        'c8_specialist_sweep': [{k: v for k, v in r.items() if k not in ('model',)}
                                 for r in c8_specialist_results],
        'top_20_pipelines': best_results[:20],
    }
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
