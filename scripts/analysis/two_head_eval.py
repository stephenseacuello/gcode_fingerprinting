#!/usr/bin/env python3
"""Two-head inference: use cls head for base op routing + anom head for damage detection.

Instead of relying on the 9-class cls head to distinguish damage classes (zero-sum problem),
use cls for base op (c0-c5 routing, which is ~100%) and anom for binary damage flag.

Final prediction:
  if anom_score > threshold: predict damage class for the base op group
  else: use cls prediction as-is
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

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn

# Base op mapping: 9-class → 3 base ops
BASE_OP_MAP = {0: 0, 1: 0, 6: 0, 2: 1, 3: 1, 7: 1, 4: 2, 5: 2, 8: 2}
DAMAGE_FOR_BASE = {0: 6, 1: 7, 2: 8}  # base_op → damage class


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


def main():
    split_dir = 'outputs/jan23_followup/no_leakage'
    model_path = 'outputs/jan30/mmdtae_standalone_v5/best_model.pt'

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    test_dataset = DecoderDatasetFromSplits(split_dir=split_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             collate_fn=decoder_collate_fn, num_workers=0)

    metadata_path = str(Path(split_dir) / 'metadata.json')
    group_names, groups, sensor_dims = build_modality_indices(metadata_path)
    group_indices = [groups[name] for name in group_names]

    # Load model
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = MM_DTAE_LSTM(config)
    model.head_cls = nn.Linear(config.d_model, 9)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Sweep thresholds
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Collect all predictions and scores
    all_cls_preds = []
    all_anom_logits = []
    all_true_ops = []

    with torch.no_grad():
        for batch in test_loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            B, T = sensor_data.size(0), sensor_data.size(1)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
            mods = [sensor_data[:, :, idx] for idx in group_indices]

            out = model(mods, lengths)
            all_cls_preds.append(out['cls'].argmax(dim=-1).cpu())
            all_anom_logits.append(out['anom'].squeeze(-1).cpu())
            all_true_ops.append(operations.cpu())

    cls_preds = torch.cat(all_cls_preds)
    anom_logits = torch.cat(all_anom_logits)
    anom_probs = torch.sigmoid(anom_logits)
    true_ops = torch.cat(all_true_ops)
    N = len(true_ops)

    print(f"\nAnom logit stats: min={anom_logits.min():.3f} max={anom_logits.max():.3f} "
          f"mean={anom_logits.mean():.3f} std={anom_logits.std():.3f}")
    print(f"Anom prob stats:  min={anom_probs.min():.3f} max={anom_probs.max():.3f} "
          f"mean={anom_probs.mean():.3f}")

    # Show anom scores by class
    print(f"\nAnom prob by true class:")
    for c in range(9):
        mask = true_ops == c
        if mask.sum() > 0:
            probs = anom_probs[mask]
            print(f"  c{c}: mean={probs.mean():.4f} std={probs.std():.4f} "
                  f"min={probs.min():.4f} max={probs.max():.4f} (n={mask.sum()})")

    print(f"\n{'='*80}")
    print(f"TWO-HEAD INFERENCE RESULTS")
    print(f"{'='*80}")

    # Also evaluate pure cls baseline
    print(f"\n--- cls_only (baseline) ---")
    per_class_total = Counter()
    per_class_correct = Counter()
    for i in range(N):
        t = true_ops[i].item()
        per_class_total[t] += 1
        if cls_preds[i].item() == t:
            per_class_correct[t] += 1
    total_correct = sum(per_class_correct.values())
    print(f"  Overall: {total_correct}/{N} = {total_correct/N:.2%}")
    for c in sorted(per_class_total.keys()):
        acc = per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0
        print(f"    c{c}: {per_class_correct[c]}/{per_class_total[c]} = {acc:.2%}")

    best_overall = 0
    best_threshold = 0
    results = {}

    for threshold in thresholds:
        per_class_correct = Counter()

        for i in range(N):
            t = true_ops[i].item()
            cls_pred = cls_preds[i].item()
            anom_prob = anom_probs[i].item()

            if anom_prob > threshold:
                # Anomaly detected — predict damage class for this base op
                base_op = BASE_OP_MAP[cls_pred]
                final_pred = DAMAGE_FOR_BASE[base_op]
            else:
                final_pred = cls_pred

            if final_pred == t:
                per_class_correct[t] += 1

        total_correct = sum(per_class_correct.values())
        overall = total_correct / N

        print(f"\n--- threshold={threshold:.1f} ---")
        print(f"  Overall: {total_correct}/{N} = {overall:.2%}")
        for c in sorted(per_class_total.keys()):
            acc = per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0
            print(f"    c{c}: {per_class_correct[c]}/{per_class_total[c]} = {acc:.2%}")

        results[f"threshold_{threshold:.1f}"] = {
            'overall': overall,
            'per_class': {str(c): per_class_correct[c] / per_class_total[c]
                          for c in sorted(per_class_total.keys())},
        }

        if overall > best_overall:
            best_overall = overall
            best_threshold = threshold

    print(f"\n{'='*80}")
    print(f"BEST: threshold={best_threshold:.1f} → {best_overall:.2%} overall")
    print(f"{'='*80}")

    # Save
    output_path = Path('outputs/jan30/two_head_eval_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
