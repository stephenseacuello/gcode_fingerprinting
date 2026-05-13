#!/usr/bin/env python3
"""Linear / MLP probe of the frozen V7-era encoder on V8 per_row fold 1.

Reads the cached encoder memory (4.8 GB tensors saved during decoder training),
mean-pools to one feature vector per sample, parses the G-code text into
structured fields, and trains a tiny probe per field. The probe tells us:

  * What information does the frozen encoder ACTUALLY carry?
  * Does the decoder's plateau match the probe's ceiling?

If decoder ≈ probe, the encoder is the bottleneck (Phase F is critical).
If decoder << probe, the decoder is leaving useful info on the table.

Writes: outputs/decoder20260511/audit/encoder_probe_v8.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "outputs/decoder20260511/preprocessed_f98/per_row/fold_1"
MEMORY_ROOT = REPO / "outputs/decoder20260511/checkpoints/hp_sweep_stage2/scheduled_sampling_0.5/fold_1/encoder_memory"
OUT_JSON = REPO / "outputs/decoder20260511/audit/encoder_probe_v8.json"


# -------- G-code text parsing (axis-presence + sign + value) --------

_FIELD_RE = re.compile(r'([XYZFSRIJ])(-?\d+\.?\d*)')

def parse_fields(text: str) -> dict:
    """Parse a G-code line into structured fields.

    Returns dict with: cmd (G0/G1/G2/G3/none),
    has_X/Y/Z/F/S/R/I/J (bool),
    sign_X/Y/Z (int: -1/0/+1, where 0 = axis not present),
    val_X/Y/Z/F (float: NaN if absent).
    """
    out = {
        'cmd': 'none',
        'has_X': 0, 'has_Y': 0, 'has_Z': 0, 'has_F': 0, 'has_S': 0,
        'has_R': 0, 'has_I': 0, 'has_J': 0,
        'sign_X': 0, 'sign_Y': 0, 'sign_Z': 0,
        'val_X': np.nan, 'val_Y': np.nan, 'val_Z': np.nan, 'val_F': np.nan,
    }
    # Command
    m = re.search(r'G([0-3])\b', text)
    if m:
        out['cmd'] = f'G{m.group(1)}'
    # Fields
    for axis, val in _FIELD_RE.findall(text):
        out[f'has_{axis}'] = 1
        v = float(val)
        if axis in ('X', 'Y', 'Z'):
            out[f'sign_{axis}'] = 1 if v >= 0 else -1
        if axis in ('X', 'Y', 'Z', 'F'):
            out[f'val_{axis}'] = v
    return out


def parse_split(texts) -> dict:
    """Convert array of G-code text strings to a struct-of-arrays of fields."""
    parsed = [parse_fields(str(t)) for t in texts]
    out = {}
    for key in parsed[0].keys():
        if key == 'cmd':
            out[key] = np.array([p[key] for p in parsed])
        else:
            out[key] = np.array([p[key] for p in parsed], dtype=float)
    return out


# -------- Probe model --------

class MLPProbe(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def train_probe(X_tr, y_tr, X_val, y_val, X_te, y_te,
                task='classification', n_classes=None, epochs=50, lr=1e-3,
                device='cuda'):
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)

    if task == 'classification':
        y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
        y_te_t = torch.tensor(y_te, dtype=torch.long).to(device)
        model = MLPProbe(X_tr.shape[1], n_classes).to(device)
        loss_fn = nn.CrossEntropyLoss()
    else:  # regression
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device).unsqueeze(1)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device).unsqueeze(1)
        y_te_t = torch.tensor(y_te, dtype=torch.float32).to(device).unsqueeze(1)
        model = MLPProbe(X_tr.shape[1], 1).to(device)
        loss_fn = nn.MSELoss()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float('-inf') if task == 'classification' else float('inf')
    best_state = None
    patience_left = 10

    for ep in range(epochs):
        model.train()
        # full-batch (probe is small enough)
        opt.zero_grad()
        loss = loss_fn(model(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()
        # Val
        model.eval()
        with torch.no_grad():
            preds = model(X_val_t)
            if task == 'classification':
                metric = (preds.argmax(1) == y_val_t).float().mean().item()
                is_better = metric > best_val
            else:
                metric = ((preds - y_val_t) ** 2).mean().item() ** 0.5  # RMSE
                is_better = metric < best_val
        if is_better:
            best_val = metric
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = 10
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_te_t)
        if task == 'classification':
            test_metric = (preds.argmax(1) == y_te_t).float().mean().item()
        else:
            test_metric = ((preds - y_te_t) ** 2).mean().item() ** 0.5
    return {'val_metric': best_val, 'test_metric': test_metric, 'final_epoch': ep + 1}


# -------- Main --------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--memory-root", type=Path, default=MEMORY_ROOT)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--out", type=Path, default=OUT_JSON)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # Load V8 splits
    print("Loading V8 NPZ splits...")
    splits = {}
    for split in ('train', 'val', 'test'):
        d = np.load(args.data_root / f"{split}_sequences.npz", allow_pickle=True)
        splits[split] = {
            'gcode_texts': d['gcode_texts'],
            'operation_type': d['operation_type'],
        }
        print(f"  {split}: {len(d['gcode_texts'])} samples")

    # Load cached encoder memory (mean-pool over seq dim immediately)
    print("\nLoading + mean-pooling encoder memory (could be slow)...")
    features = {}
    for split in ('train', 'val', 'test'):
        mem_path = args.memory_root / f"{'memory.pt' if split == 'train' else split + '_memory.pt'}"
        if split == 'train':
            # 'memory.pt' is the train memory in this trainer
            mem_path = args.memory_root / "train_memory.pt"
        mem = torch.load(mem_path, map_location='cpu')
        # mem shape: (N, seq_len, d_model)
        print(f"  {split}: loaded {mem.shape} from {mem_path.name}")
        pooled = mem.mean(dim=1).numpy()  # (N, d_model)
        features[split] = pooled.astype(np.float32)
        del mem

    # Parse G-code fields per split
    print("\nParsing G-code fields...")
    labels = {split: parse_split(splits[split]['gcode_texts']) for split in splits}

    # Build probe tasks
    print("\nRunning probes...")
    out_results = {}
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"  device = {device}")

    # ---- Classification probes ----
    # 1) Command (G0/G1/G2/G3/none) — 5-class
    cmd_classes = ['none', 'G0', 'G1', 'G2', 'G3']
    cmd_idx = {c: i for i, c in enumerate(cmd_classes)}
    for split in ('train', 'val', 'test'):
        labels[split]['cmd_idx'] = np.array([cmd_idx.get(c, 0) for c in labels[split]['cmd']])

    print("  > probe: command (5-class)")
    r = train_probe(
        features['train'], labels['train']['cmd_idx'],
        features['val'], labels['val']['cmd_idx'],
        features['test'], labels['test']['cmd_idx'],
        task='classification', n_classes=5, epochs=args.epochs, device=device,
    )
    out_results['probe_command_acc'] = r['test_metric']
    print(f"    test acc = {r['test_metric']:.4f} (val {r['val_metric']:.4f}, ep {r['final_epoch']})")

    # 2) has_X / has_Y / has_Z / has_F (binary)
    for field in ('has_X', 'has_Y', 'has_Z', 'has_F', 'has_S'):
        print(f"  > probe: {field}")
        r = train_probe(
            features['train'], labels['train'][field].astype(int),
            features['val'], labels['val'][field].astype(int),
            features['test'], labels['test'][field].astype(int),
            task='classification', n_classes=2, epochs=args.epochs, device=device,
        )
        out_results[f'probe_{field}_acc'] = r['test_metric']
        print(f"    test acc = {r['test_metric']:.4f}")

    # 3) sign_X / sign_Y / sign_Z (ternary: -1, 0, +1 → 0, 1, 2)
    for field in ('sign_X', 'sign_Y', 'sign_Z'):
        sign_map = {-1: 0, 0: 1, 1: 2}
        y_tr = np.array([sign_map[int(v)] for v in labels['train'][field]])
        y_val = np.array([sign_map[int(v)] for v in labels['val'][field]])
        y_te = np.array([sign_map[int(v)] for v in labels['test'][field]])
        print(f"  > probe: {field}")
        r = train_probe(
            features['train'], y_tr, features['val'], y_val, features['test'], y_te,
            task='classification', n_classes=3, epochs=args.epochs, device=device,
        )
        out_results[f'probe_{field}_acc'] = r['test_metric']
        print(f"    test acc = {r['test_metric']:.4f}")

    # 4) operation_type (the encoder's training target — sanity check)
    print(f"  > probe: operation_type (9-class, ENCODER'S OWN TRAINING TARGET)")
    op_tr = splits['train']['operation_type'].astype(int)
    op_val = splits['val']['operation_type'].astype(int)
    op_te = splits['test']['operation_type'].astype(int)
    r = train_probe(
        features['train'], op_tr, features['val'], op_val, features['test'], op_te,
        task='classification', n_classes=int(op_tr.max()) + 1, epochs=args.epochs, device=device,
    )
    out_results['probe_operation_type_acc'] = r['test_metric']
    print(f"    test acc = {r['test_metric']:.4f}")

    # ---- Regression probes (only on rows where axis present) ----
    for field in ('val_X', 'val_Y', 'val_Z'):
        mask_tr = ~np.isnan(labels['train'][field])
        mask_val = ~np.isnan(labels['val'][field])
        mask_te = ~np.isnan(labels['test'][field])
        if mask_tr.sum() < 100:
            continue
        print(f"  > probe: {field} (regression, n_train={mask_tr.sum()})")
        r = train_probe(
            features['train'][mask_tr], labels['train'][field][mask_tr],
            features['val'][mask_val], labels['val'][field][mask_val],
            features['test'][mask_te], labels['test'][field][mask_te],
            task='regression', epochs=args.epochs, device=device,
        )
        out_results[f'probe_{field}_rmse'] = r['test_metric']
        print(f"    test RMSE = {r['test_metric']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        'config': {
            'memory_root': str(args.memory_root),
            'data_root': str(args.data_root),
            'epochs': args.epochs,
        },
        'probe_results': out_results,
        'n_train': len(features['train']),
        'n_val': len(features['val']),
        'n_test': len(features['test']),
    }, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
