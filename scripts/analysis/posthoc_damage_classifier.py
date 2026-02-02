#!/usr/bin/env python3
"""Post-hoc damage classifier on v1 frozen embeddings.

Load the best flat model (v1), freeze it, extract CLS embeddings for all samples,
then train 3 lightweight binary classifiers (one per op group) to detect damage.

At inference: v1 cls head for 9-class prediction → if base op is in {adaptive, face, pocket},
check the post-hoc damage classifier for that group → override to damage class if positive.
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
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn

BASE_OP_MAP = {0: 0, 1: 0, 6: 0, 2: 1, 3: 1, 7: 1, 4: 2, 5: 2, 8: 2}
DAMAGE_FOR_BASE = {0: 6, 1: 7, 2: 8}
GROUP_NAMES = {0: 'adaptive', 1: 'face', 2: 'pocket'}
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
    sensor_dims = [len(groups[n]) for n in group_names]
    return group_names, groups, sensor_dims


def extract_embeddings(model, loader, group_indices, device):
    """Extract CLS embeddings, cls predictions, and true labels from frozen model."""
    model.eval()
    all_embeddings = []
    all_cls_preds = []
    all_true_ops = []

    with torch.no_grad():
        for batch in loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type']
            B, T = sensor_data.size(0), sensor_data.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
            mods = [sensor_data[:, :, idx] for idx in group_indices]

            out = model(mods, lengths)

            # Get the CLS representation (before the classification head)
            # We need to hook into the model to get the pre-head embedding
            # The cls head is: head_cls(cls_rep) where cls_rep is the pooled output
            # Re-run to get intermediate representation
            all_cls_preds.append(out['cls'].argmax(dim=-1).cpu())
            all_true_ops.append(operations)

    cls_preds = torch.cat(all_cls_preds).numpy()
    true_ops = torch.cat(all_true_ops).numpy()
    return cls_preds, true_ops


def extract_pre_head_embeddings(model, loader, group_indices, device):
    """Extract the d_model-dimensional embedding before the cls head."""
    model.eval()
    all_embeds = []
    all_ops = []

    # Register a hook on head_cls to capture its input
    hook_output = []

    def hook_fn(module, input, output):
        hook_output.append(input[0].detach().cpu())

    handle = model.head_cls.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type']
            B, T = sensor_data.size(0), sensor_data.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
            mods = [sensor_data[:, :, idx] for idx in group_indices]

            _ = model(mods, lengths)
            all_embeds.append(hook_output[-1])
            all_ops.append(operations)

    handle.remove()

    embeddings = torch.cat(all_embeds).numpy()
    true_ops = torch.cat(all_ops).numpy()
    return embeddings, true_ops


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

    # Load v1 model (frozen)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = MM_DTAE_LSTM(config)
    model.head_cls = nn.Linear(config.d_model, 9)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded v1 model from {model_path}")
    print(f"  Config: d_model={config.d_model}")

    # Extract embeddings from all splits
    print(f"\nExtracting pre-head embeddings...")
    split_data = {}
    for split in ['train', 'val', 'test']:
        embeds, ops = extract_pre_head_embeddings(model, loaders[split], group_indices, device)
        cls_preds, _ = extract_embeddings(model, loaders[split], group_indices, device)
        split_data[split] = {'embeddings': embeds, 'ops': ops, 'cls_preds': cls_preds}
        print(f"  {split}: {embeds.shape[0]} samples, embedding dim={embeds.shape[1]}")

    # Also extract raw sensor features (time-averaged) for comparison
    print(f"\nExtracting raw sensor features for comparison...")
    for split in ['train', 'val', 'test']:
        raw_feats = []
        for batch in loaders[split]:
            # Average over time dimension
            raw_feats.append(batch['sensor_features'].mean(dim=1).numpy())
        split_data[split]['raw_features'] = np.concatenate(raw_feats)

    # Train per-group damage classifiers
    print(f"\n{'='*70}")
    print(f"POST-HOC DAMAGE CLASSIFIERS")
    print(f"{'='*70}")

    results = {}

    for base_op in range(3):
        group_name = GROUP_NAMES[base_op]
        damage_cls = DAMAGE_FOR_BASE[base_op]
        member_classes = GROUP_CLASSES[base_op]

        print(f"\n--- {group_name.upper()} (damage class = c{damage_cls}) ---")

        # Filter to samples in this group
        for feature_type in ['embeddings', 'raw_features']:
            feat_label = 'v1 embeddings' if feature_type == 'embeddings' else 'raw features'

            train_mask = np.isin(split_data['train']['ops'], member_classes)
            test_mask = np.isin(split_data['test']['ops'], member_classes)

            X_train = split_data['train'][feature_type][train_mask]
            y_train = (split_data['train']['ops'][train_mask] == damage_cls).astype(int)
            X_test = split_data['test'][feature_type][test_mask]
            y_test = (split_data['test']['ops'][test_mask] == damage_cls).astype(int)

            print(f"\n  [{feat_label}]")
            print(f"    Train: {len(X_train)} samples ({y_train.sum()} damage, {(~y_train.astype(bool)).sum()} normal)")
            print(f"    Test:  {len(X_test)} samples ({y_test.sum()} damage, {(~y_test.astype(bool)).sum()} normal)")

            # Train LogisticRegression
            clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
            clf.fit(X_train, y_train)

            train_preds = clf.predict(X_train)
            test_preds = clf.predict(X_test)
            train_probs = clf.predict_proba(X_train)[:, 1]
            test_probs = clf.predict_proba(X_test)[:, 1]

            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            train_f1 = f1_score(y_train, train_preds)
            test_f1 = f1_score(y_test, test_preds)

            print(f"    Train: acc={train_acc:.2%} F1={train_f1:.2%}")
            print(f"    Test:  acc={test_acc:.2%} F1={test_f1:.2%}")

            # Detailed test metrics
            print(f"    Test damage detection: "
                  f"TP={((test_preds == 1) & (y_test == 1)).sum()} "
                  f"FP={((test_preds == 1) & (y_test == 0)).sum()} "
                  f"TN={((test_preds == 0) & (y_test == 0)).sum()} "
                  f"FN={((test_preds == 0) & (y_test == 1)).sum()}")

            # Probability distribution
            normal_probs = test_probs[y_test == 0]
            damage_probs = test_probs[y_test == 1]
            print(f"    Prob dist (test normal):  mean={normal_probs.mean():.4f} std={normal_probs.std():.4f} max={normal_probs.max():.4f}")
            if len(damage_probs) > 0:
                print(f"    Prob dist (test damage):  mean={damage_probs.mean():.4f} std={damage_probs.std():.4f} min={damage_probs.min():.4f}")

            key = f"{group_name}_{feature_type}"
            results[key] = {
                'train_acc': float(train_acc), 'test_acc': float(test_acc),
                'train_f1': float(train_f1), 'test_f1': float(test_f1),
                'test_tp': int(((test_preds == 1) & (y_test == 1)).sum()),
                'test_fp': int(((test_preds == 1) & (y_test == 0)).sum()),
                'test_tn': int(((test_preds == 0) & (y_test == 0)).sum()),
                'test_fn': int(((test_preds == 0) & (y_test == 1)).sum()),
            }

    # Now do the full pipeline: v1 cls + post-hoc damage override
    print(f"\n{'='*70}")
    print(f"FULL PIPELINE: v1 cls + post-hoc damage classifiers")
    print(f"{'='*70}")

    for feature_type in ['embeddings', 'raw_features']:
        feat_label = 'v1 embeddings' if feature_type == 'embeddings' else 'raw features'
        print(f"\n--- Using {feat_label} for damage detection ---")

        # Train all 3 damage classifiers
        damage_clfs = {}
        for base_op in range(3):
            member_classes = GROUP_CLASSES[base_op]
            damage_cls = DAMAGE_FOR_BASE[base_op]

            train_mask = np.isin(split_data['train']['ops'], member_classes)
            X_train = split_data['train'][feature_type][train_mask]
            y_train = (split_data['train']['ops'][train_mask] == damage_cls).astype(int)

            clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
            clf.fit(X_train, y_train)
            damage_clfs[base_op] = clf

        # Evaluate on test set
        test_ops = split_data['test']['ops']
        test_cls_preds = split_data['test']['cls_preds']
        test_features = split_data['test'][feature_type]
        N = len(test_ops)

        per_class_total = Counter()
        per_class_correct = Counter()

        for i in range(N):
            true_op = test_ops[i]
            cls_pred = test_cls_preds[i]
            per_class_total[true_op] += 1

            # Determine base op from cls prediction
            base_op = BASE_OP_MAP.get(cls_pred, -1)

            # Check post-hoc damage classifier for this group
            damage_prob = damage_clfs[base_op].predict_proba(test_features[i:i+1])[0, 1]

            if damage_prob > 0.5:
                final_pred = DAMAGE_FOR_BASE[base_op]
            else:
                final_pred = cls_pred

            if final_pred == true_op:
                per_class_correct[true_op] += 1

        total_correct = sum(per_class_correct.values())
        overall = total_correct / N
        print(f"  Overall: {total_correct}/{N} = {overall:.2%}")
        for c in sorted(per_class_total.keys()):
            acc = per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0
            print(f"    c{c}: {per_class_correct[c]}/{per_class_total[c]} = {acc:.2%}")

        results[f'pipeline_{feature_type}'] = {
            'overall': float(overall),
            'per_class': {str(c): float(per_class_correct[c] / per_class_total[c])
                          for c in sorted(per_class_total.keys())},
        }

        # Also try with multiple thresholds
        print(f"\n  Threshold sweep:")
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pc_correct = Counter()
            for i in range(N):
                true_op = test_ops[i]
                cls_pred = test_cls_preds[i]
                base_op = BASE_OP_MAP.get(cls_pred, -1)
                damage_prob = damage_clfs[base_op].predict_proba(test_features[i:i+1])[0, 1]

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
            print(f"    t={thresh:.1f}: {total}/{N}={total/N:.2%}  c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    # =========== GLOBAL DAMAGE DETECTION (no routing dependency) ===========
    print(f"\n{'='*70}")
    print(f"GLOBAL DAMAGE DETECTION: run ALL 3 classifiers on ALL samples")
    print(f"{'='*70}")

    for feature_type in ['embeddings', 'raw_features']:
        feat_label = 'v1 embeddings' if feature_type == 'embeddings' else 'raw features'
        print(f"\n--- Using {feat_label} ---")

        # Train all 3 damage classifiers (same as before)
        damage_clfs = {}
        for base_op in range(3):
            member_classes = GROUP_CLASSES[base_op]
            damage_cls = DAMAGE_FOR_BASE[base_op]
            train_mask = np.isin(split_data['train']['ops'], member_classes)
            X_train = split_data['train'][feature_type][train_mask]
            y_train = (split_data['train']['ops'][train_mask] == damage_cls).astype(int)
            clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
            clf.fit(X_train, y_train)
            damage_clfs[base_op] = clf

        # Also train a single global 4-class classifier: normal(0), c6(1), c7(2), c8(3)
        global_y_train = np.zeros(len(split_data['train']['ops']), dtype=int)
        global_y_train[split_data['train']['ops'] == 6] = 1
        global_y_train[split_data['train']['ops'] == 7] = 2
        global_y_train[split_data['train']['ops'] == 8] = 3
        global_clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
        global_clf.fit(split_data['train'][feature_type], global_y_train)

        test_ops = split_data['test']['ops']
        test_cls_preds = split_data['test']['cls_preds']
        test_features = split_data['test'][feature_type]
        N = len(test_ops)
        per_class_total = Counter()
        for i in range(N):
            per_class_total[test_ops[i]] += 1

        # Strategy A: Run all 3 per-group classifiers, take max damage prob
        print(f"\n  Strategy A: Max damage prob across all 3 group classifiers")
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pc_correct = Counter()
            for i in range(N):
                true_op = test_ops[i]
                cls_pred = test_cls_preds[i]

                # Get damage prob from ALL 3 classifiers
                damage_probs = {}
                for base_op in range(3):
                    damage_probs[base_op] = damage_clfs[base_op].predict_proba(
                        test_features[i:i+1])[0, 1]

                # Find max damage signal
                max_base = max(damage_probs, key=damage_probs.get)
                max_prob = damage_probs[max_base]

                if max_prob > thresh:
                    final_pred = DAMAGE_FOR_BASE[max_base]
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
            print(f"    t={thresh:.1f}: {total}/{N}={total/N:.2%}  "
                  f"c0-c5={rest}/{rest_total}={rest/rest_total:.2%}  "
                  f"c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

        # Strategy B: Global 4-class classifier (normal vs c6 vs c7 vs c8)
        print(f"\n  Strategy B: Global 4-class classifier (normal/c6/c7/c8)")
        global_preds = global_clf.predict(test_features)
        global_probs = global_clf.predict_proba(test_features)

        pc_correct = Counter()
        for i in range(N):
            true_op = test_ops[i]
            cls_pred = test_cls_preds[i]
            g_pred = global_preds[i]

            if g_pred == 0:
                final_pred = cls_pred  # normal — use v1 cls
            elif g_pred == 1:
                final_pred = 6
            elif g_pred == 2:
                final_pred = 7
            else:
                final_pred = 8

            if final_pred == true_op:
                pc_correct[true_op] += 1

        total = sum(pc_correct.values())
        print(f"  Overall: {total}/{N} = {total/N:.2%}")
        for c in sorted(per_class_total.keys()):
            acc = pc_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0
            print(f"    c{c}: {pc_correct[c]}/{per_class_total[c]} = {acc:.2%}")

        results[f'global_4class_{feature_type}'] = {
            'overall': float(total / N),
            'per_class': {str(c): float(pc_correct[c] / per_class_total[c])
                          for c in sorted(per_class_total.keys())},
        }

        # Strategy C: Global 4-class with threshold (only override if confident)
        print(f"\n  Strategy C: Global 4-class with confidence threshold")
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pc_correct = Counter()
            for i in range(N):
                true_op = test_ops[i]
                cls_pred = test_cls_preds[i]
                g_pred = global_preds[i]
                g_conf = global_probs[i].max()

                if g_pred > 0 and g_conf > thresh:
                    final_pred = [None, 6, 7, 8][g_pred]
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
            print(f"    t={thresh:.1f}: {total}/{N}={total/N:.2%}  "
                  f"c0-c5={rest}/{rest_total}={rest/rest_total:.2%}  "
                  f"c6={c6:.0%} c7={c7:.0%} c8={c8:.0%}")

    # Save
    output_path = Path('outputs/jan30/posthoc_damage_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
