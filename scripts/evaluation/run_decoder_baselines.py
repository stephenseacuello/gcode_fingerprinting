#!/usr/bin/env python3
"""
Decoder Baselines: Quantify how much the autoregressive decoder adds over simpler approaches.

Baselines:
    1A. Majority-class lookup table (position → most common G-code)
    1B. Nearest-neighbor retrieval on encoder embeddings (k=1, k=5)
    1C. MLP 34-class sequence classifier on CLS embedding
    1D. Random Forest / XGBoost 34-class on encoder embeddings

All baselines use the same V7 preprocessed data and frozen encoder as the decoder.
Metrics are computed identically to decoder evaluation (token_acc, seq_acc, cmd_acc, num_acc).

Usage:
    python scripts/evaluation/run_decoder_baselines.py \
        --encoder_config v7_f110_w256_s64 \
        --fold 1 \
        --vocab data/gcode_vocab_712.json \
        --output_dir outputs/decoder20260304/baselines \
        --baselines 1A 1B 1C 1D

Author: Claude Code
Date: March 2026
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.utilities.gcode_tokenizer import GCodeTokenizer

# Reuse infrastructure from decoder quick test
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_decoder_quick_test import (
    PAD, BOS, EOS, UNK, TYPE_SPECIAL, TYPE_COMMAND, TYPE_PARAM, TYPE_NUMERIC,
    ENCODER_BASE, V7_DATA_BASE, ENCODER_CONFIGS, V7_ENCODER_CONFIGS,
    DecoderQuickTestDataset, decoder_collate_fn,
    load_frozen_encoder, build_modality_indices, classify_token,
)


def log(msg, log_file=None):
    print(msg, flush=True)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


# ── Metric Computation ────────────────────────────────────────────────────────

def compute_metrics(pred_token_ids, true_token_ids, type_targets, tokenizer):
    """Compute decoder-style metrics from predicted and true token ID arrays.

    Args:
        pred_token_ids: list of lists of int (predicted token IDs per sample)
        true_token_ids: list of lists of int (true token IDs per sample, no BOS)
        type_targets: list of lists of int (token type per position, matching true_token_ids)
        tokenizer: GCodeTokenizer for decoding

    Returns:
        dict with token_accuracy, sequence_accuracy, command_accuracy,
        param_type_accuracy, numeric_accuracy
    """
    token_correct = 0
    token_total = 0
    seq_match = 0
    seq_total = 0
    num_correct = 0
    num_total = 0
    cmd_correct = 0
    cmd_total = 0
    param_correct = 0
    param_total = 0
    type_correct = 0
    type_total = 0

    for pred, true, types in zip(pred_token_ids, true_token_ids, type_targets):
        # Align lengths
        max_len = max(len(pred), len(true))
        pred_padded = pred + [PAD] * (max_len - len(pred))
        true_padded = true + [PAD] * (max_len - len(true))
        types_padded = types + [-1] * (max_len - len(types))

        seq_total += 1
        all_match = True

        for p, t, tp in zip(pred_padded, true_padded, types_padded):
            if t == PAD:
                if p != PAD:
                    all_match = False
                continue

            token_total += 1
            if p == t:
                token_correct += 1
            else:
                all_match = False

            if tp == TYPE_COMMAND:
                cmd_total += 1
                if p == t:
                    cmd_correct += 1
            elif tp == TYPE_PARAM:
                param_total += 1
                if p == t:
                    param_correct += 1
            elif tp == TYPE_NUMERIC:
                num_total += 1
                if p == t:
                    num_correct += 1
            elif tp >= 0:
                type_total += 1
                if p == t:
                    type_correct += 1

        if all_match:
            seq_match += 1

    return {
        'token_accuracy': token_correct / max(token_total, 1),
        'sequence_accuracy': seq_match / max(seq_total, 1),
        'command_accuracy': cmd_correct / max(cmd_total, 1),
        'param_type_accuracy': param_correct / max(param_total, 1),
        'numeric_accuracy': num_correct / max(num_total, 1),
        'n_samples': seq_total,
    }


# ── Helper: Extract Token Data ───────────────────────────────────────────────

def extract_token_data(dataset):
    """Extract parallel lists of token IDs, gcode text, types, and metadata from dataset."""
    all_target_ids = []
    all_type_targets = []
    all_gcode_texts = []
    all_window_idx = []
    all_op_types = []

    for i in range(len(dataset)):
        item = dataset[i]
        target = item['target_tokens'].tolist()
        types = item['type_targets'].tolist()
        # Strip trailing PAD
        tokens = []
        ttypes = []
        for t, tp in zip(target, types):
            if t == PAD:
                break
            tokens.append(t)
            ttypes.append(tp)
        all_target_ids.append(tokens)
        all_type_targets.append(ttypes)
        all_gcode_texts.append(dataset.gcode_texts_list[i])
        all_window_idx.append(item['window_index'].item())
        all_op_types.append(item['operation_type'].item())

    return all_target_ids, all_type_targets, all_gcode_texts, all_window_idx, all_op_types


# ── 1A. Majority-Class Lookup Table ──────────────────────────────────────────

def run_baseline_1A(train_dataset, test_dataset, tokenizer, lprint):
    """For each (window_position, operation_type), predict the most common G-code sequence."""
    lprint("\n[1A] Majority-Class Lookup Table")
    lprint("  Building lookup from training set...")

    train_ids, train_types, train_texts, train_widx, train_ops = extract_token_data(train_dataset)

    # Build lookup: (window_position_bin, op_type) → most common gcode text
    # Bin window positions to handle slight variations
    lookup = defaultdict(list)
    for text, widx, op in zip(train_texts, train_widx, train_ops):
        key = (widx, op)
        lookup[key].append(text)

    # Resolve to majority
    majority = {}
    for key, texts in lookup.items():
        counter = Counter(texts)
        majority[key] = counter.most_common(1)[0][0]

    # Also build position-only fallback
    pos_lookup = defaultdict(list)
    for text, widx in zip(train_texts, train_widx):
        pos_lookup[widx].append(text)
    pos_majority = {}
    for widx, texts in pos_lookup.items():
        counter = Counter(texts)
        pos_majority[widx] = counter.most_common(1)[0][0]

    # Global fallback
    global_majority = Counter(train_texts).most_common(1)[0][0]

    lprint(f"  Lookup entries: {len(majority)} (pos+op), {len(pos_majority)} (pos only)")

    # Predict on test set
    test_ids, test_types, test_texts, test_widx, test_ops = extract_token_data(test_dataset)
    precision = tokenizer.cfg.precision

    pred_ids = []
    hits = {'pos_op': 0, 'pos': 0, 'global': 0}
    for widx, op in zip(test_widx, test_ops):
        key = (widx, op)
        if key in majority:
            pred_text = majority[key]
            hits['pos_op'] += 1
        elif widx in pos_majority:
            pred_text = pos_majority[widx]
            hits['pos'] += 1
        else:
            pred_text = global_majority
            hits['global'] += 1

        # Tokenize predicted text
        canon = tokenizer.canonicalize([pred_text])
        tok_strings = tokenizer.tokenize_canonical(canon)
        tok_ids = [tokenizer._tok2id(t) for t in tok_strings]
        pred_ids.append(tok_ids + [EOS])

    lprint(f"  Fallback stats: pos+op={hits['pos_op']}, pos_only={hits['pos']}, global={hits['global']}")

    # Add EOS to true targets for fair comparison
    true_ids_with_eos = [ids for ids in test_ids]

    metrics = compute_metrics(pred_ids, true_ids_with_eos, test_types, tokenizer)
    lprint(f"  Results: seq={metrics['sequence_accuracy']:.1%} tok={metrics['token_accuracy']:.1%} "
           f"cmd={metrics['command_accuracy']:.1%} num={metrics['numeric_accuracy']:.1%}")
    return metrics


# ── 1B. Nearest-Neighbor Retrieval ───────────────────────────────────────────

def run_baseline_1B(train_dataset, test_dataset, train_memory, test_memory, tokenizer, lprint):
    """For each test window, find k-nearest training windows by cosine similarity on encoder CLS.

    Uses mean-pooled encoder memory as embedding.
    """
    lprint("\n[1B] Nearest-Neighbor Retrieval on Encoder Embeddings")

    # Mean-pool encoder memory to get CLS-like embedding
    lprint("  Computing mean-pooled embeddings...")
    train_emb = train_memory.mean(dim=1).numpy()  # [N_train, d_model]
    test_emb = test_memory.mean(dim=1).numpy()     # [N_test, d_model]

    # Normalize for cosine similarity
    train_norm = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-8)
    test_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-8)

    train_ids, train_types, train_texts, _, _ = extract_token_data(train_dataset)
    test_ids, test_types, test_texts, _, _ = extract_token_data(test_dataset)

    results = {}
    for k in [1, 5]:
        lprint(f"  k={k}: computing cosine similarities...")
        # Compute similarity matrix in chunks to save memory
        chunk_size = 100
        pred_ids = []
        for start in range(0, len(test_norm), chunk_size):
            end = min(start + chunk_size, len(test_norm))
            sim = test_norm[start:end] @ train_norm.T  # [chunk, N_train]
            topk_idx = np.argsort(-sim, axis=1)[:, :k]

            for row_idx in range(end - start):
                if k == 1:
                    nn_text = train_texts[topk_idx[row_idx, 0]]
                else:
                    # Majority vote among k neighbors
                    nn_texts = [train_texts[j] for j in topk_idx[row_idx]]
                    nn_text = Counter(nn_texts).most_common(1)[0][0]

                canon = tokenizer.canonicalize([nn_text])
                tok_strings = tokenizer.tokenize_canonical(canon)
                tok_ids_pred = [tokenizer._tok2id(t) for t in tok_strings]
                pred_ids.append(tok_ids_pred + [EOS])

        metrics = compute_metrics(pred_ids, test_ids, test_types, tokenizer)
        lprint(f"  k={k}: seq={metrics['sequence_accuracy']:.1%} tok={metrics['token_accuracy']:.1%} "
               f"cmd={metrics['command_accuracy']:.1%} num={metrics['numeric_accuracy']:.1%}")
        results[f'1B_kNN_k{k}'] = metrics

    return results


# ── 1C. MLP Sequence Classifier ──────────────────────────────────────────────

class MLPSequenceClassifier(nn.Module):
    """Classify encoder embedding → one of N unique G-code sequences."""

    def __init__(self, input_dim, n_classes, hidden_dim=256, n_layers=2, dropout=0.3):
        super().__init__()
        layers = []
        in_d = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_baseline_1C(train_dataset, val_dataset, test_dataset,
                    train_memory, val_memory, test_memory,
                    tokenizer, device, lprint, epochs=200, patience=30):
    """Train MLP classifier: encoder embedding → G-code sequence class."""
    lprint("\n[1C] MLP Sequence Classifier (34-class)")

    # Build class mapping from all unique G-code texts
    train_ids, train_types, train_texts, _, _ = extract_token_data(train_dataset)
    val_ids, val_types, val_texts, _, _ = extract_token_data(val_dataset)
    test_ids, test_types, test_texts, _, _ = extract_token_data(test_dataset)

    all_unique = sorted(set(train_texts + val_texts + test_texts))
    text2class = {t: i for i, t in enumerate(all_unique)}
    n_classes = len(all_unique)
    lprint(f"  Unique sequences: {n_classes}")

    # Mean-pool memory for embeddings
    train_emb = train_memory.mean(dim=1)  # [N, d_model]
    val_emb = val_memory.mean(dim=1)
    test_emb = test_memory.mean(dim=1)

    train_labels = torch.tensor([text2class[t] for t in train_texts], dtype=torch.long)
    val_labels = torch.tensor([text2class[t] for t in val_texts], dtype=torch.long)
    test_labels = torch.tensor([text2class[t] for t in test_texts], dtype=torch.long)

    input_dim = train_emb.shape[1]

    results = {}
    for n_layers, name in [(1, '1C_MLP_1layer'), (2, '1C_MLP_2layer')]:
        lprint(f"\n  Training {name} ({n_layers} hidden layers)...")
        model = MLPSequenceClassifier(input_dim, n_classes, hidden_dim=256,
                                       n_layers=n_layers, dropout=0.3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        best_val_acc = 0
        best_state = None
        no_improve = 0

        for ep in range(epochs):
            model.train()
            # Mini-batch training
            perm = torch.randperm(len(train_emb))
            total_loss = 0
            n_batch = 0
            for start in range(0, len(perm), 64):
                idx = perm[start:start+64]
                x = train_emb[idx].to(device)
                y = train_labels[idx].to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batch += 1
            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(val_emb.to(device))
                val_preds = val_logits.argmax(-1).cpu()
                val_acc = (val_preds == val_labels).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                lprint(f"    Early stopping at epoch {ep+1} (best val: {best_val_acc:.1%})")
                break

        if (ep + 1) == epochs:
            lprint(f"    Finished {epochs} epochs (best val: {best_val_acc:.1%})")

        # Load best and evaluate on test
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_logits = model(test_emb.to(device))
            test_preds = test_logits.argmax(-1).cpu().tolist()

        # Convert predicted class back to token IDs
        class2text = {v: k for k, v in text2class.items()}
        pred_token_ids = []
        for pred_class in test_preds:
            pred_text = class2text[pred_class]
            canon = tokenizer.canonicalize([pred_text])
            tok_strings = tokenizer.tokenize_canonical(canon)
            tok_ids = [tokenizer._tok2id(t) for t in tok_strings]
            pred_token_ids.append(tok_ids + [EOS])

        metrics = compute_metrics(pred_token_ids, test_ids, test_types, tokenizer)
        metrics['val_accuracy'] = best_val_acc
        lprint(f"    Test: seq={metrics['sequence_accuracy']:.1%} tok={metrics['token_accuracy']:.1%} "
               f"cmd={metrics['command_accuracy']:.1%} num={metrics['numeric_accuracy']:.1%}")
        results[name] = metrics

    return results


# ── 1D. Random Forest / XGBoost ──────────────────────────────────────────────

def run_baseline_1D(train_dataset, test_dataset,
                    train_memory, test_memory,
                    tokenizer, lprint):
    """XGBoost and Random Forest: encoder embedding → G-code sequence class."""
    lprint("\n[1D] Random Forest / XGBoost (34-class)")

    train_ids, train_types, train_texts, _, _ = extract_token_data(train_dataset)
    test_ids, test_types, test_texts, _, _ = extract_token_data(test_dataset)

    # Use only training set texts for class mapping (contiguous labels)
    # Test-only sequences get mapped to nearest training class at prediction time
    train_unique = sorted(set(train_texts))
    text2class = {t: i for i, t in enumerate(train_unique)}
    class2text = {v: k for k, v in text2class.items()}

    train_emb = train_memory.mean(dim=1).numpy()
    test_emb = test_memory.mean(dim=1).numpy()
    train_labels = np.array([text2class[t] for t in train_texts])
    # For test labels, map unknown sequences to -1 (they can't be predicted correctly anyway)
    test_labels = np.array([text2class.get(t, -1) for t in test_texts])
    n_classes = len(train_unique)
    lprint(f"  Training classes: {n_classes}, test-only sequences: {(test_labels == -1).sum()}")

    results = {}

    # Random Forest
    lprint("  Training RandomForest...")
    try:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
        rf.fit(train_emb, train_labels)
        rf_preds = rf.predict(test_emb)
        rf_acc = (rf_preds == test_labels).mean()
        lprint(f"    RF raw class accuracy: {rf_acc:.1%}")

        pred_token_ids = []
        for pred_class in rf_preds:
            pred_text = class2text[pred_class]
            canon = tokenizer.canonicalize([pred_text])
            tok_strings = tokenizer.tokenize_canonical(canon)
            tok_ids = [tokenizer._tok2id(t) for t in tok_strings]
            pred_token_ids.append(tok_ids + [EOS])

        metrics = compute_metrics(pred_token_ids, test_ids, test_types, tokenizer)
        lprint(f"    RF: seq={metrics['sequence_accuracy']:.1%} tok={metrics['token_accuracy']:.1%} "
               f"cmd={metrics['command_accuracy']:.1%} num={metrics['numeric_accuracy']:.1%}")
        results['1D_RandomForest'] = metrics
    except ImportError:
        lprint("    sklearn not available, skipping RF")

    # XGBoost
    lprint("  Training XGBoost...")
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            n_jobs=-1, random_state=42, tree_method='hist',
            eval_metric='mlogloss',
        )
        xgb.fit(train_emb, train_labels)
        xgb_preds = xgb.predict(test_emb)
        xgb_acc = (xgb_preds == test_labels).mean()
        lprint(f"    XGB raw class accuracy: {xgb_acc:.1%}")

        pred_token_ids = []
        for pred_class in xgb_preds:
            pred_text = class2text[pred_class]
            canon = tokenizer.canonicalize([pred_text])
            tok_strings = tokenizer.tokenize_canonical(canon)
            tok_ids = [tokenizer._tok2id(t) for t in tok_strings]
            pred_token_ids.append(tok_ids + [EOS])

        metrics = compute_metrics(pred_token_ids, test_ids, test_types, tokenizer)
        lprint(f"    XGB: seq={metrics['sequence_accuracy']:.1%} tok={metrics['token_accuracy']:.1%} "
               f"cmd={metrics['command_accuracy']:.1%} num={metrics['numeric_accuracy']:.1%}")
        results['1D_XGBoost'] = metrics
    except ImportError:
        lprint("    xgboost not available, skipping XGBoost")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Decoder baselines (POST_SWEEP_PLAN 1A-1D)")
    parser.add_argument("--encoder_config", type=str, default="v7_f110_w256_s64",
                        choices=list(ENCODER_CONFIGS.keys()) + list(V7_ENCODER_CONFIGS.keys()))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--vocab", type=str, default="data/gcode_vocab_712.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--baselines", nargs='+', default=['1A', '1B', '1C', '1D'],
                        choices=['1A', '1B', '1C', '1D'])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_token_len", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Resolve paths
    if args.encoder_config in V7_ENCODER_CONFIGS:
        v7cfg = V7_ENCODER_CONFIGS[args.encoder_config]
        data_dir = Path(v7cfg['data_dir']) / f"fold_{args.fold}"
        encoder_ckpt = ENCODER_BASE / v7cfg['encoder_dir'] / f"fold_{args.fold}" / "encoder" / "checkpoint" / "best_model.pt"
    elif args.encoder_config in ENCODER_CONFIGS:
        config_dir = ENCODER_CONFIGS[args.encoder_config]
        data_dir = ENCODER_BASE / config_dir / f"fold_{args.fold}" / "preprocessed"
        encoder_ckpt = ENCODER_BASE / config_dir / f"fold_{args.fold}" / "encoder" / "checkpoint" / "best_model.pt"

    output_dir = Path(args.output_dir) / f"fold_{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / 'baselines.log'
    log_fh = open(log_path, 'w')
    def lprint(msg):
        log(msg, log_fh)

    lprint(f"{'='*70}")
    lprint(f"Decoder Baselines — Fold {args.fold}")
    lprint(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lprint(f"Device: {device}")
    lprint(f"{'='*70}")

    # Load tokenizer
    lprint(f"\nLoading vocabulary from {args.vocab}")
    tokenizer = GCodeTokenizer.load(Path(args.vocab))

    # Load datasets
    lprint(f"\nLoading datasets from {data_dir}")
    datasets = {}
    for split in ['train', 'val', 'test']:
        npz_path = data_dir / f'{split}_sequences.npz'
        ds = DecoderQuickTestDataset(npz_path, tokenizer, max_token_len=args.max_token_len)
        datasets[split] = ds
        lprint(f"  {split}: {ds.stats['n_samples']} samples")

    # Load encoder and cache memory (needed for 1B, 1C, 1D)
    needs_encoder = bool(set(args.baselines) & {'1B', '1C', '1D'})
    memories = {}
    if needs_encoder:
        lprint(f"\nLoading encoder from {encoder_ckpt}")
        encoder, enc_config, ckpt = load_frozen_encoder(str(encoder_ckpt), device)

        # Get modality indices
        meta_path = data_dir / 'metadata.json'
        with open(meta_path) as f:
            meta = json.load(f)
        columns = meta['continuous_columns']
        _, group_indices, sensor_dims = build_modality_indices(columns)

        lprint(f"  Caching encoder memory for all splits...")
        for split in ['train', 'val', 'test']:
            loader = DataLoader(datasets[split], batch_size=32, shuffle=False,
                                collate_fn=decoder_collate_fn, num_workers=0)
            all_memory = []
            with torch.no_grad():
                for batch in loader:
                    sensor_data = batch['sensor_features'].to(device)
                    B, T = sensor_data.shape[:2]
                    lengths = torch.full((B,), T, dtype=torch.long, device=device)
                    mods = [sensor_data[:, :, idx] for idx in group_indices]
                    out = encoder(mods, lengths)
                    all_memory.append(out['memory'].cpu())
            memories[split] = torch.cat(all_memory, dim=0)
            lprint(f"    {split}: {memories[split].shape}")

    # Run baselines
    all_results = {}

    if '1A' in args.baselines:
        metrics = run_baseline_1A(datasets['train'], datasets['test'], tokenizer, lprint)
        all_results['1A_lookup'] = metrics

    if '1B' in args.baselines:
        results_1b = run_baseline_1B(
            datasets['train'], datasets['test'],
            memories['train'], memories['test'],
            tokenizer, lprint
        )
        all_results.update(results_1b)

    if '1C' in args.baselines:
        results_1c = run_baseline_1C(
            datasets['train'], datasets['val'], datasets['test'],
            memories['train'], memories['val'], memories['test'],
            tokenizer, device, lprint,
        )
        all_results.update(results_1c)

    if '1D' in args.baselines:
        results_1d = run_baseline_1D(
            datasets['train'], datasets['test'],
            memories['train'], memories['test'],
            tokenizer, lprint,
        )
        all_results.update(results_1d)

    # Save results
    results_path = output_dir / 'baseline_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    lprint(f"\nResults saved to {results_path}")

    # Print summary table
    lprint(f"\n{'='*70}")
    lprint(f"SUMMARY — Fold {args.fold}")
    lprint(f"{'='*70}")
    lprint(f"{'Baseline':<25} {'Seq':>8} {'Tok':>8} {'Cmd':>8} {'Num':>8}")
    lprint(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, m in all_results.items():
        lprint(f"{name:<25} {m['sequence_accuracy']:>7.1%} {m['token_accuracy']:>7.1%} "
               f"{m['command_accuracy']:>7.1%} {m['numeric_accuracy']:>7.1%}")
    lprint(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    lprint(f"{'V7 decoder (reference)':<25} {'86.0%':>8} {'95.1%':>8} {'98.7%':>8} {'91.5%':>8}")

    log_fh.close()


if __name__ == '__main__':
    main()
