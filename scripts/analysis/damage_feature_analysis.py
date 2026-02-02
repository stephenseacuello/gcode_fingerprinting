#!/usr/bin/env python3
"""Analyze per-modality feature differences between damage and normal samples.

For each operation group (adaptive, face, pocket), compute:
1. Per-modality mean/std differences between normal and damage samples
2. Per-modality classification accuracy using simple classifiers
3. Which specific sensors carry the strongest damage signal
"""
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits


def build_modality_indices(metadata_path):
    """Build column index lists for each modality group from metadata."""
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
    return group_names, groups

GROUP_CLASSES = {
    'adaptive': {'normal': [0, 1], 'damage': [6]},
    'face': {'normal': [2, 3], 'damage': [7]},
    'pocket': {'normal': [4, 5], 'damage': [8]},
}


def main():
    split_dir = 'outputs/jan23_followup/no_leakage'
    metadata_path = str(Path(split_dir) / 'metadata.json')
    group_names, groups = build_modality_indices(metadata_path)
    MODALITY_GROUPS = [(name, groups[name]) for name in group_names]

    # Load all splits
    results = {}
    for split_name in ['train', 'val', 'test']:
        dataset = DecoderDatasetFromSplits(split_dir=split_dir, split=split_name)

        # Collect per-class features (average over time dimension)
        class_features = defaultdict(list)
        for i in range(len(dataset)):
            sample = dataset[i]
            op = sample['operation_type'].item()
            # Average over time → [296]
            features = sample['sensor_features'].mean(dim=0).numpy()
            class_features[op].append(features)

        for op in class_features:
            class_features[op] = np.stack(class_features[op])

        if split_name != 'test':
            continue

        print(f"\n{'='*70}")
        print(f"DAMAGE FEATURE ANALYSIS ({split_name} set)")
        print(f"{'='*70}")

        for group_name, classes in GROUP_CLASSES.items():
            normal_ops = classes['normal']
            damage_ops = classes['damage']

            normal_feats = np.concatenate([class_features[c] for c in normal_ops if c in class_features])
            damage_feats = np.concatenate([class_features[c] for c in damage_ops if c in class_features])

            print(f"\n--- {group_name.upper()} ---")
            print(f"  Normal samples: {len(normal_feats)}, Damage samples: {len(damage_feats)}")

            group_results = {'modalities': {}}

            for mod_name, mod_indices in MODALITY_GROUPS:
                normal_mod = normal_feats[:, mod_indices]
                damage_mod = damage_feats[:, mod_indices]

                # Cohen's d effect size per feature, then average
                normal_mean = normal_mod.mean(axis=0)
                damage_mean = damage_mod.mean(axis=0)
                pooled_std = np.sqrt((normal_mod.std(axis=0)**2 + damage_mod.std(axis=0)**2) / 2)
                pooled_std = np.where(pooled_std < 1e-8, 1.0, pooled_std)
                cohens_d = np.abs(damage_mean - normal_mean) / pooled_std
                mean_d = cohens_d.mean()
                max_d = cohens_d.max()
                max_d_idx = cohens_d.argmax()

                # Simple threshold classifier: for each feature, find best threshold
                # to separate normal from damage
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score, f1_score

                X = np.vstack([normal_mod, damage_mod])
                y = np.array([0]*len(normal_mod) + [1]*len(damage_mod))

                # Logistic regression on this modality alone
                try:
                    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
                    clf.fit(X, y)
                    preds = clf.predict(X)
                    acc = accuracy_score(y, preds)
                    f1 = f1_score(y, preds)
                except Exception:
                    acc = 0.5
                    f1 = 0.0

                print(f"  {mod_name:15s}: Cohen's d mean={mean_d:.3f} max={max_d:.3f} "
                      f"| LogReg acc={acc:.2%} F1={f1:.2%}")

                group_results['modalities'][mod_name] = {
                    'cohens_d_mean': float(mean_d),
                    'cohens_d_max': float(max_d),
                    'max_d_feature_idx': int(max_d_idx),
                    'logreg_accuracy': float(acc),
                    'logreg_f1': float(f1),
                }

            # Also try all features combined
            X_all = np.vstack([normal_feats, damage_feats])
            y_all = np.array([0]*len(normal_feats) + [1]*len(damage_feats))
            try:
                clf_all = LogisticRegression(max_iter=1000, class_weight='balanced')
                clf_all.fit(X_all, y_all)
                preds_all = clf_all.predict(X_all)
                acc_all = accuracy_score(y_all, preds_all)
                f1_all = f1_score(y_all, preds_all)
            except Exception:
                acc_all = 0.5
                f1_all = 0.0

            print(f"  {'ALL FEATURES':15s}: LogReg acc={acc_all:.2%} F1={f1_all:.2%}")
            group_results['all_features'] = {'accuracy': float(acc_all), 'f1': float(f1_all)}

            # Top 10 individual features by Cohen's d
            all_d = np.abs(damage_feats.mean(axis=0) - normal_feats.mean(axis=0))
            all_std = np.sqrt((normal_feats.std(axis=0)**2 + damage_feats.std(axis=0)**2) / 2)
            all_std = np.where(all_std < 1e-8, 1.0, all_std)
            all_cohens = all_d / all_std
            top_10_idx = all_cohens.argsort()[-10:][::-1]

            print(f"\n  Top 10 features by Cohen's d:")
            for idx in top_10_idx:
                # Find which modality this belongs to
                mod = "unknown"
                for mn, mi in MODALITY_GROUPS:
                    if idx in mi:
                        mod = mn
                        break
                print(f"    Feature {idx:3d} ({mod:15s}): d={all_cohens[idx]:.3f}")

            group_results['top_features'] = [
                {'index': int(idx), 'cohens_d': float(all_cohens[idx])}
                for idx in top_10_idx
            ]
            results[group_name] = group_results

    # Save
    import json
    output_path = Path('outputs/jan30/damage_feature_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
