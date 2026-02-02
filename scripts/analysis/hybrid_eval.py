#!/usr/bin/env python3
"""Hybrid evaluation: combine hierarchical v1 (for face/c7) with flat v1 (for pocket/c8).

Strategy:
- Use flat v1 as the base 9-class classifier
- Override face-group predictions using hierarchical v1's stage1 + stage2_face
- This should give us c7=78% (from hier) + c8=100% (from flat)
"""
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig, make_pad_mask
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn

# Same mappings as hierarchical script
BASE_OP_MAP = {0: 0, 1: 0, 6: 0, 2: 1, 3: 1, 7: 1, 4: 2, 5: 2, 8: 2}
DAMAGE_CLASSES = {0: 6, 1: 7, 2: 8}
GROUP_CLASSES = {0: [0, 1, 6], 1: [2, 3, 7], 2: [4, 5, 8]}

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
    sensor_dims = [len(groups[n]) for n in group_names]
    return group_names, groups, sensor_dims


def split_modalities(sensor_data, group_indices):
    return [sensor_data[:, :, idx] for idx in group_indices]


def main():
    split_dir = 'outputs/jan23_followup/no_leakage'
    flat_model_path = 'outputs/jan30/mmdtae_standalone_v1/best_model.pt'
    hier_dir = Path('outputs/jan30/mmdtae_hierarchical_v1')

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    test_dataset = DecoderDatasetFromSplits(split_dir=split_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             collate_fn=decoder_collate_fn, num_workers=0)

    metadata_path = str(Path(split_dir) / 'metadata.json')
    group_names, groups, sensor_dims = build_modality_indices(metadata_path)
    group_indices = [groups[name] for name in group_names]

    def make_model(n_classes):
        config = ModelConfig(
            sensor_dims=sensor_dims, d_model=256, n_heads=4, lstm_layers=2,
            dropout=0.2, gcode_vocab=128, future_len=8,
        )
        model = MM_DTAE_LSTM(config)
        model.head_cls = nn.Linear(256, n_classes)
        return model

    # Load flat v1 (9-class)
    flat_model = make_model(9)
    ckpt = torch.load(flat_model_path, map_location=device, weights_only=False)
    flat_model.load_state_dict(ckpt['model_state_dict'])
    flat_model.to(device)
    flat_model.eval()

    # Load hierarchical v1 stage1 (3-class base op) + stage2_face (binary)
    hier_base = make_model(3)
    ckpt = torch.load(hier_dir / 'stage1_base_best.pt', map_location=device, weights_only=True)
    hier_base.load_state_dict(ckpt['model_state_dict'])
    hier_base.to(device)
    hier_base.eval()

    hier_face = make_model(2)
    ckpt = torch.load(hier_dir / 'stage2_face_best.pt', map_location=device, weights_only=True)
    hier_face.load_state_dict(ckpt['model_state_dict'])
    hier_face.to(device)
    hier_face.eval()

    # Evaluate with multiple strategies
    strategies = {
        'flat_only': {},       # Pure flat v1
        'hier_face': {},       # Flat v1 + hier face override
        'hier_all': {},        # Flat v1 + hier override for all 3 groups
    }

    # Also load adaptive and pocket damage models for hier_all
    hier_adaptive = make_model(2)
    ckpt = torch.load(hier_dir / 'stage2_adaptive_best.pt', map_location=device, weights_only=True)
    hier_adaptive.load_state_dict(ckpt['model_state_dict'])
    hier_adaptive.to(device)
    hier_adaptive.eval()

    hier_pocket = make_model(2)
    ckpt = torch.load(hier_dir / 'stage2_pocket_best.pt', map_location=device, weights_only=True)
    hier_pocket.load_state_dict(ckpt['model_state_dict'])
    hier_pocket.to(device)
    hier_pocket.eval()

    damage_models = {0: hier_adaptive, 1: hier_face, 2: hier_pocket}

    per_class_total = Counter()
    results = {s: Counter() for s in strategies}

    with torch.no_grad():
        for batch in test_loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            B = operations.size(0)
            T = sensor_data.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
            mods = split_modalities(sensor_data, group_indices)

            # Flat v1 predictions
            flat_out = flat_model(mods, lengths)
            flat_preds = flat_out['cls'].argmax(dim=-1)

            # Hierarchical predictions
            hier_base_out = hier_base(mods, lengths)
            hier_base_preds = hier_base_out['cls'].argmax(dim=-1)

            # Per-group damage predictions
            damage_preds = {}
            for g in range(3):
                d_out = damage_models[g](mods, lengths)
                damage_preds[g] = d_out['cls'].argmax(dim=-1)  # 0=normal, 1=damage

            for i in range(B):
                true_op = operations[i].item()
                per_class_total[true_op] += 1

                # Strategy 1: flat only
                if flat_preds[i].item() == true_op:
                    results['flat_only'][true_op] += 1

                # Strategy 2: flat + hier face override
                pred = flat_preds[i].item()
                # If hier says this is face group AND damage, override
                if hier_base_preds[i].item() == 1 and damage_preds[1][i].item() == 1:
                    pred = 7  # damageface
                # But if flat already predicted face group correctly, keep it
                # unless hier says damage
                if pred == true_op:
                    results['hier_face'][true_op] += 1
                elif true_op == 7 and hier_base_preds[i].item() == 1 and damage_preds[1][i].item() == 1:
                    results['hier_face'][true_op] += 1
                else:
                    # Re-check: did our hybrid get it right?
                    hybrid_pred = flat_preds[i].item()
                    if hier_base_preds[i].item() == 1 and damage_preds[1][i].item() == 1:
                        hybrid_pred = 7
                    if hybrid_pred == true_op:
                        results['hier_face'][true_op] += 1

                # Strategy 3: flat + hier override for ALL damage classes
                hybrid_all = flat_preds[i].item()
                base_pred = hier_base_preds[i].item()
                if damage_preds[base_pred][i].item() == 1:
                    hybrid_all = DAMAGE_CLASSES[base_pred]
                if hybrid_all == true_op:
                    results['hier_all'][true_op] += 1

    # Print results
    print("\n" + "=" * 70)
    print("HYBRID EVALUATION RESULTS")
    print("=" * 70)

    for strategy_name in strategies:
        correct = results[strategy_name]
        total_correct = sum(correct.values())
        total = sum(per_class_total.values())
        print(f"\n--- {strategy_name} ---")
        print(f"  Overall: {total_correct}/{total} = {total_correct/total:.2%}")
        print(f"  Per-class:")
        for c in sorted(per_class_total.keys()):
            acc = correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0
            print(f"    c{c}: {correct[c]}/{per_class_total[c]} = {acc:.2%}")

    # Save
    output = {
        'per_class_total': dict(per_class_total),
    }
    for s in strategies:
        output[s] = {
            'per_class_correct': dict(results[s]),
            'overall': sum(results[s].values()) / sum(per_class_total.values()),
        }
    with open('outputs/jan30/hybrid_eval_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to outputs/jan30/hybrid_eval_results.json")


if __name__ == '__main__':
    main()
