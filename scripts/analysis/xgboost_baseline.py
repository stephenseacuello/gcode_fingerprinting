#!/usr/bin/env python3
"""Non-neural baseline: XGBoost classifiers on mean-pooled encoder memory.

Reviewer-anticipated baseline for the G-Code Decoder paper. The encoder paper
compared the multi-modal Transformer against five baseline families; the
decoder paper had no non-neural baseline until now. This script trains an
XGBoost classifier per output head (command, has-axis, sign) on the same
frozen encoder memory the decoder consumes, and reports per-head accuracy
+ macro F1.

The baseline answers a specific reviewer question:
    "What does a non-deep-learning classifier on the SAME encoder features
     get? Is the decoder's lift over this baseline meaningful?"

Inputs:
  - encoder memory:  outputs/decoder20260511/checkpoints/hp_sweep_stage2/
                     scheduled_sampling_0.5/fold_1/encoder_memory/{train,val,test}_memory.pt
  - V8 NPZ:           outputs/decoder20260511/preprocessed_f98/per_row/fold_1/
                      {train,val,test}_sequences.npz

Output:
  - outputs/decoder20260511/audit/xgboost_baseline_v8.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "outputs/decoder20260511/preprocessed_f98/per_row/fold_1"
MEMORY_ROOT = REPO / "outputs/decoder20260511/checkpoints/hp_sweep_stage2/scheduled_sampling_0.5/fold_1/encoder_memory"
OUT_JSON = REPO / "outputs/decoder20260511/audit/xgboost_baseline_v8.json"

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


_FIELD_RE = re.compile(r'([XYZFSRIJ])(-?\d+\.?\d*)')


def parse_targets(texts) -> dict[str, np.ndarray]:
    """Parse G-code text array into structured field labels.

    Returns dict with int arrays:
      cmd_idx   — 5-class command index (none/G0/G1/G2/G3)
      has_X..S  — binary axis-presence
      sign_X..Z — ternary (-1=neg, 0=absent, +1=pos)
    """
    cmd_classes = ['none', 'G0', 'G1', 'G2', 'G3']
    cmd_idx_map = {c: i for i, c in enumerate(cmd_classes)}
    out = {
        'cmd_idx': [], 'has_X': [], 'has_Y': [], 'has_Z': [], 'has_F': [],
        'sign_X': [], 'sign_Y': [], 'sign_Z': [],
    }
    for t in texts:
        s = str(t)
        m = re.search(r'G([0-3])\b', s)
        out['cmd_idx'].append(cmd_idx_map.get(f'G{m.group(1)}', 0) if m else 0)
        fields = {ax: val for ax, val in _FIELD_RE.findall(s)}
        for ax in 'XYZF':
            out[f'has_{ax}'].append(1 if ax in fields else 0)
        for ax in 'XYZ':
            if ax in fields:
                v = float(fields[ax])
                out[f'sign_{ax}'].append(0 if v < 0 else 2)  # 0=neg, 1=absent, 2=pos
            else:
                out[f'sign_{ax}'].append(1)
    return {k: np.asarray(v, dtype=int) for k, v in out.items()}


def pool_memory(path: Path) -> np.ndarray:
    """Load encoder memory and mean-pool over the sequence dim."""
    mem = torch.load(path, map_location='cpu')
    pooled = mem.mean(dim=1).numpy().astype(np.float32)
    del mem
    return pooled


def train_xgb_classifier(X_tr, y_tr, X_val, y_val, X_te, y_te, n_classes, n_jobs=8):
    """Train an XGBoost classifier and return (test_acc, test_macro_f1, n_classes_used)."""
    if n_classes == 2:
        objective = 'binary:logistic'
        params = dict(objective=objective, eval_metric='error', max_depth=6,
                      learning_rate=0.1, n_estimators=300, n_jobs=n_jobs,
                      tree_method='hist', verbosity=0)
        model = xgb.XGBClassifier(**params)
    else:
        params = dict(objective='multi:softmax', num_class=n_classes,
                      eval_metric='mlogloss', max_depth=6, learning_rate=0.1,
                      n_estimators=300, n_jobs=n_jobs, tree_method='hist',
                      verbosity=0)
        model = xgb.XGBClassifier(**params)

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_te)
    acc = float(accuracy_score(y_te, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, average='macro', zero_division=0,
    )
    return {
        'accuracy': acc,
        'macro_precision': float(prec),
        'macro_recall': float(rec),
        'macro_f1': float(f1),
        'support': int(len(y_te)),
        'class_counts_test': dict(Counter(y_te.tolist())),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--memory-root", type=Path, default=MEMORY_ROOT)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--out", type=Path, default=OUT_JSON)
    p.add_argument("--n-jobs", type=int, default=24)
    args = p.parse_args()

    if not HAS_XGB:
        print("xgboost not installed")
        return 1

    t0 = time.time()

    print("Loading + pooling encoder memory...")
    X = {}
    for split, fname in [('train', 'train_memory.pt'), ('val', 'val_memory.pt'), ('test', 'test_memory.pt')]:
        path = args.memory_root / fname
        print(f"  {split}: {path.name}")
        X[split] = pool_memory(path)
        print(f"    shape = {X[split].shape}")

    print("\nParsing G-code targets...")
    y = {}
    for split in ('train', 'val', 'test'):
        npz = np.load(args.data_root / f"{split}_sequences.npz", allow_pickle=True)
        y[split] = parse_targets(npz['gcode_texts'])
        print(f"  {split}: {len(npz['gcode_texts'])} rows")

    # Heads to evaluate
    tasks = {
        'command_5class':  ('cmd_idx',  5),
        'has_X_binary':    ('has_X',    2),
        'has_Y_binary':    ('has_Y',    2),
        'has_Z_binary':    ('has_Z',    2),
        'has_F_binary':    ('has_F',    2),
        'sign_X_3class':   ('sign_X',   3),
        'sign_Y_3class':   ('sign_Y',   3),
        'sign_Z_3class':   ('sign_Z',   3),
    }

    results = {}
    print("\nTraining XGBoost classifiers per head...")
    for task_name, (key, n_classes) in tasks.items():
        print(f"  > {task_name} (n_classes={n_classes})")
        r = train_xgb_classifier(
            X['train'], y['train'][key],
            X['val'],   y['val'][key],
            X['test'],  y['test'][key],
            n_classes=n_classes, n_jobs=args.n_jobs,
        )
        results[task_name] = r
        print(f"    accuracy = {r['accuracy']:.4f}, macro_f1 = {r['macro_f1']:.4f}")

    elapsed = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        'config': {
            'memory_root': str(args.memory_root),
            'data_root': str(args.data_root),
            'baseline': 'XGBoost on mean-pooled encoder memory',
            'n_train': int(X['train'].shape[0]),
            'n_val':   int(X['val'].shape[0]),
            'n_test':  int(X['test'].shape[0]),
            'feature_dim': int(X['train'].shape[1]),
            'elapsed_seconds': elapsed,
        },
        'results_per_head': results,
    }, indent=2))
    print(f"\nWrote {args.out}")
    print(f"Total elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
