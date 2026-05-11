#!/usr/bin/env python3
"""
Hybrid Position-Gated Decoding: Route unambiguous windows to lookup table,
ambiguous windows to decoder OR a coordinate disambiguation model (RF on
sensor statistics).

Modes:
    --disambiguator decoder   Use the trained decoder for ambiguous windows
    --disambiguator rf        Use a Random Forest on sensor statistics
    --disambiguator both      Run both and compare

Usage:
    python scripts/evaluation/run_hybrid_position_decoding.py \
        --folds 1 2 3 4 5 \
        --vocab data/gcode_vocab_712.json \
        --device auto \
        --disambiguator both

Author: Claude Code
Date: March 2026
"""

import sys
import os
import json
import time
import math
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.utilities.gcode_tokenizer import GCodeTokenizer

from run_decoder_quick_test import (
    PAD, BOS, EOS, UNK, TYPE_SPECIAL, TYPE_COMMAND, TYPE_PARAM, TYPE_NUMERIC,
    DecoderQuickTestDataset, CachedDecoderDataset, decoder_collate_fn,
    beam_search_decode, classify_token,
)
from run_decoder_baselines import extract_token_data, compute_metrics


# ── Logging ──────────────────────────────────────────────────────────────────

log_fh = None

def lprint(msg):
    print(msg, flush=True)
    if log_fh is not None:
        log_fh.write(msg + "\n")
        log_fh.flush()


# ── Ambiguity Map ────────────────────────────────────────────────────────────

def build_ambiguity_map(train_dataset):
    """Build position ambiguity map from training data.

    Returns dict with:
        pos_op_sequences: {(widx, op): Counter of gcode texts}
        pos_op_majority: {(widx, op): most common gcode text}
        pos_sequences: {widx: Counter of gcode texts}
        pos_majority: {widx: most common gcode text}
        global_majority: str
        ambiguity: {(widx, op): {'n_unique', 'entropy', 'majority_frac', 'is_ambiguous'}}
    """
    _, _, train_texts, train_widx, train_ops = extract_token_data(train_dataset)

    # (widx, op) → Counter
    pos_op_seqs = defaultdict(list)
    for text, widx, op in zip(train_texts, train_widx, train_ops):
        pos_op_seqs[(widx, op)].append(text)

    pos_op_counters = {}
    pos_op_majority = {}
    for key, texts in pos_op_seqs.items():
        c = Counter(texts)
        pos_op_counters[key] = c
        pos_op_majority[key] = c.most_common(1)[0][0]

    # widx → Counter (fallback)
    pos_seqs = defaultdict(list)
    for text, widx in zip(train_texts, train_widx):
        pos_seqs[widx].append(text)

    pos_counters = {}
    pos_majority = {}
    for widx, texts in pos_seqs.items():
        c = Counter(texts)
        pos_counters[widx] = c
        pos_majority[widx] = c.most_common(1)[0][0]

    global_majority = Counter(train_texts).most_common(1)[0][0]

    # Ambiguity stats per (widx, op)
    ambiguity = {}
    for key, counter in pos_op_counters.items():
        total = sum(counter.values())
        n_unique = len(counter)
        majority_frac = counter.most_common(1)[0][1] / total
        # Shannon entropy
        entropy = 0.0
        if n_unique > 1:
            for count in counter.values():
                p = count / total
                entropy -= p * math.log2(p)
        ambiguity[key] = {
            'n_unique': n_unique,
            'entropy': round(entropy, 4),
            'majority_frac': round(majority_frac, 4),
            'total_samples': total,
            'is_ambiguous': n_unique > 1,
        }

    return {
        'pos_op_counters': pos_op_counters,
        'pos_op_majority': pos_op_majority,
        'pos_counters': pos_counters,
        'pos_majority': pos_majority,
        'global_majority': global_majority,
        'ambiguity': ambiguity,
    }


# ── Lookup Predictions ───────────────────────────────────────────────────────

def get_lookup_predictions(amb_map, test_dataset, tokenizer):
    """Get lookup table predictions for all test samples."""
    _, test_types, _, test_widx, test_ops = extract_token_data(test_dataset)

    pred_ids = []
    fallback_levels = []

    for widx, op in zip(test_widx, test_ops):
        key = (widx, op)
        if key in amb_map['pos_op_majority']:
            pred_text = amb_map['pos_op_majority'][key]
            fallback_levels.append('pos_op')
        elif widx in amb_map['pos_majority']:
            pred_text = amb_map['pos_majority'][widx]
            fallback_levels.append('pos')
        else:
            pred_text = amb_map['global_majority']
            fallback_levels.append('global')

        canon = tokenizer.canonicalize([pred_text])
        tok_strings = tokenizer.tokenize_canonical(canon)
        tok_ids = [tokenizer._tok2id(t) for t in tok_strings]
        pred_ids.append(tok_ids + [EOS])

    return pred_ids, fallback_levels


# ── Decoder Predictions ──────────────────────────────────────────────────────

@torch.no_grad()
def get_decoder_predictions(decoder, cached_dataset, device, beam_width=1,
                            batch_size=32):
    """Run decoder inference on all samples, return per-sample predicted token IDs."""
    loader = DataLoader(cached_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=decoder_collate_fn, num_workers=0)
    decoder.eval()

    all_preds = []
    for batch in loader:
        memory = batch['memory'].to(device)
        op_pred = batch['op_pred'].to(device)
        B = memory.shape[0]

        extra_kwargs = {}
        if 'window_index' in batch:
            extra_kwargs['window_index'] = batch['window_index'].to(device)
        if 'total_windows' in batch:
            extra_kwargs['total_windows'] = batch['total_windows'].to(device)

        if beam_width >= 1:
            # Autoregressive decoding
            pred_tokens = beam_search_decode(
                decoder, memory, op_pred, device,
                beam_width=beam_width, max_len=16,
                extra_kwargs=extra_kwargs,
            )  # [B, max_len] with BOS at position 0
            # Strip BOS, collect per-sample
            for b in range(B):
                seq = pred_tokens[b, 1:].tolist()  # skip BOS
                # Trim at EOS or PAD
                trimmed = []
                for t in seq:
                    if t == EOS:
                        trimmed.append(t)
                        break
                    if t == PAD:
                        break
                    trimmed.append(t)
                if not trimmed or trimmed[-1] != EOS:
                    trimmed.append(EOS)
                all_preds.append(trimmed)
        else:
            # Teacher-forced (beam_width=0)
            input_tokens = batch['input_tokens'].to(device)
            padding_mask = (input_tokens == PAD)
            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=memory,
                operation_type=op_pred,
                tgt_key_padding_mask=padding_mask,
                **extra_kwargs,
            )
            logits = outputs.get('legacy_logits', outputs.get('raw_legacy_logits'))
            preds = logits.argmax(-1)  # [B, L]
            target_tokens = batch['target_tokens'].to(device)
            for b in range(B):
                seq = preds[b].tolist()
                true_len = (target_tokens[b] != PAD).sum().item()
                trimmed = seq[:true_len]
                if not trimmed or trimmed[-1] != EOS:
                    trimmed.append(EOS)
                all_preds.append(trimmed)

    return all_preds


# ── RF Coordinate Disambiguation ─────────────────────────────────────────────

def compute_sensor_features(continuous):
    """Compute per-window summary statistics from raw sensor data.
    Args: continuous (N, 256, C). Returns: (N, C*5) — mean, std, max, min, RMS.
    """
    mean = np.mean(continuous, axis=1)
    std = np.std(continuous, axis=1)
    mx = np.max(continuous, axis=1)
    mn = np.min(continuous, axis=1)
    rms = np.sqrt(np.mean(continuous ** 2, axis=1))
    return np.concatenate([mean, std, mx, mn, rms], axis=1)


def train_rf_disambiguator(data_dir, fold, amb_map, lprint):
    """Train RF on sensor statistics for ambiguous (widx, op) positions.

    Returns: (rf_model, label_encoder, features_test) or (None, None, None).
    """
    fold_dir = Path(data_dir) / f"fold_{fold}"
    train_npz = np.load(fold_dir / "train_sequences.npz", allow_pickle=True)
    test_npz = np.load(fold_dir / "test_sequences.npz", allow_pickle=True)

    train_texts = list(train_npz['gcode_texts'])
    train_widx = train_npz['window_index'].astype(int)
    train_ops = train_npz['operation_type'].astype(int)
    train_cont = train_npz['continuous']

    test_cont = test_npz['continuous']

    # Filter training to ambiguous positions only
    amb_keys = {k for k, v in amb_map['ambiguity'].items() if v['is_ambiguous']}
    train_mask = np.array([
        (int(widx), int(op)) in amb_keys
        for widx, op in zip(train_widx, train_ops)
    ])

    n_amb = train_mask.sum()
    if n_amb == 0:
        lprint("  RF: No ambiguous training samples")
        return None, None, None

    amb_texts = [t for t, m in zip(train_texts, train_mask) if m]
    lprint(f"  RF: {n_amb} ambiguous training samples, "
           f"{len(set(amb_texts))} distinct G-code sequences")

    # Compute features
    train_features = compute_sensor_features(train_cont[train_mask])
    test_features = compute_sensor_features(test_cont)

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(amb_texts)

    # Train
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1,
    )
    rf.fit(train_features, labels)
    train_acc = (rf.predict(train_features) == labels).mean()
    lprint(f"  RF: train accuracy {train_acc:.1%}, {len(le.classes_)} classes")

    return rf, le, test_features


# ── Hybrid Gate ──────────────────────────────────────────────────────────────

def hybrid_gate(amb_map, test_widx, test_ops, lookup_preds, decoder_preds,
                threshold=1.0):
    """Route each sample to lookup or decoder based on position ambiguity.

    Args:
        threshold: majority fraction threshold. 1.0 = binary (any ambiguity → decoder).
                   0.9 = positions with >=90% majority treated as unambiguous.
    """
    hybrid_preds = []
    routing = []

    for i, (widx, op) in enumerate(zip(test_widx, test_ops)):
        key = (widx, op)
        if key in amb_map['ambiguity']:
            info = amb_map['ambiguity'][key]
            if info['n_unique'] == 1 or info['majority_frac'] >= threshold:
                hybrid_preds.append(lookup_preds[i])
                routing.append('lookup')
            else:
                hybrid_preds.append(decoder_preds[i])
                routing.append('decoder')
        else:
            # Unseen position — decoder is our only option
            hybrid_preds.append(decoder_preds[i])
            routing.append('decoder_unseen')

    return hybrid_preds, routing


# ── Decoder Reconstruction ───────────────────────────────────────────────────

def reconstruct_decoder(checkpoint_path, tokenizer, max_windows, device):
    """Reconstruct decoder model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get('args', {})

    # Load axis value tables if pointer network was used
    axis_value_tables = None
    if args.get('use_pointer_network', False):
        avt_path = Path(args.get('axis_value_tables', 'data/axis_value_tables.json'))
        if avt_path.exists():
            with open(avt_path) as f:
                axis_value_tables = json.load(f)

    decoder = SensorMultiHeadDecoder(
        vocab_size=args.get('vocab_size', len(tokenizer.vocab)),
        d_model=args.get('d_model', 384),
        sensor_dim=256,
        n_operations=9,
        n_heads=args.get('n_heads', 12),
        n_layers=args.get('n_layers', 8),
        max_int_digits=2,
        n_decimal_digits=4,
        dropout=args.get('dropout', 0.1),
        max_seq_len=args.get('max_token_len', 16),
        hierarchical=args.get('hierarchical', False),
        memory_pos_encoding=args.get('memory_pos_encoding', True),
        use_regression_head=args.get('use_regression_head', False),
        use_window_position=args.get('use_window_position', True),
        max_windows_per_file=max_windows + 4,
        use_sequence_classifier=args.get('use_sequence_classifier', False),
        n_unique_sequences=args.get('n_unique_sequences', 34),
        use_pointer_network=args.get('use_pointer_network', False),
        axis_value_tables=axis_value_tables,
        use_sensor_prior=args.get('use_sensor_prior', True),
        drop_path_rate=args.get('drop_path_rate', 0.0),
    )
    decoder.use_grammar_constraint = args.get('grammar_constraint', False)
    decoder.set_vocab(tokenizer.vocab)

    # Load state dict with size mismatch handling
    state_dict = ckpt['decoder_state_dict']
    model_state = decoder.state_dict()
    missing, unexpected = decoder.load_state_dict(state_dict, strict=False)
    # Handle window_pos_embed size mismatch
    for key in list(state_dict.keys()):
        if key in model_state and state_dict[key].shape != model_state[key].shape:
            ckpt_w = state_dict[key]
            n_copy = min(ckpt_w.shape[0], model_state[key].shape[0])
            with torch.no_grad():
                parts = key.split('.')
                obj = decoder
                for attr in parts[:-1]:
                    obj = getattr(obj, attr)
                param = getattr(obj, parts[-1])
                param[:n_copy] = ckpt_w[:n_copy].to(param.device)

    decoder = decoder.to(device)
    decoder.eval()
    return decoder, ckpt.get('epoch', '?')


# ── Per-Fold Runner ──────────────────────────────────────────────────────────

def run_fold(fold, args):
    """Run hybrid evaluation for one fold."""
    lprint(f"\n{'='*70}")
    lprint(f"FOLD {fold}")
    lprint(f"{'='*70}")

    device = args.device
    data_dir = Path(args.data_dir) / f'fold_{fold}'
    ckpt_dir = Path(args.decoder_dir) / f'fold_{fold}'
    output_dir = Path(args.output_dir) / f'fold_{fold}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = GCodeTokenizer.load(Path(args.vocab))
    vocab_size = len(tokenizer.vocab)

    # Load datasets
    lprint(f"  Loading data from {data_dir}")
    # Phase-3 (decoder20260511): None auto-resolves from NPZ; V7 → ~16, V8 → larger.
    train_ds = DecoderQuickTestDataset(
        data_dir / 'train_sequences.npz', tokenizer, max_token_len=None)
    test_ds = DecoderQuickTestDataset(
        data_dir / 'test_sequences.npz', tokenizer, max_token_len=None)
    lprint(f"  Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")

    # Build ambiguity map
    lprint(f"  Building ambiguity map...")
    amb_map = build_ambiguity_map(train_ds)

    n_positions = len(amb_map['ambiguity'])
    n_ambiguous_positions = sum(1 for v in amb_map['ambiguity'].values() if v['is_ambiguous'])
    lprint(f"  Positions: {n_positions} total, {n_ambiguous_positions} ambiguous, "
           f"{n_positions - n_ambiguous_positions} unambiguous")

    # Get test metadata
    _, test_types, test_texts, test_widx, test_ops = extract_token_data(test_ds)
    test_ids, _, _, _, _ = extract_token_data(test_ds)

    # Classify test samples
    n_test = len(test_ds)
    sample_class = []  # 'unambiguous', 'ambiguous', 'unseen'
    for widx, op in zip(test_widx, test_ops):
        key = (widx, op)
        if key not in amb_map['ambiguity']:
            sample_class.append('unseen')
        elif amb_map['ambiguity'][key]['is_ambiguous']:
            sample_class.append('ambiguous')
        else:
            sample_class.append('unambiguous')

    n_unamb = sum(1 for s in sample_class if s == 'unambiguous')
    n_amb = sum(1 for s in sample_class if s == 'ambiguous')
    n_unseen = sum(1 for s in sample_class if s == 'unseen')
    lprint(f"  Test samples: {n_unamb} unambiguous, {n_amb} ambiguous, {n_unseen} unseen")

    # Get lookup predictions
    lprint(f"  Getting lookup predictions...")
    lookup_preds, fallback_levels = get_lookup_predictions(amb_map, test_ds, tokenizer)

    # Decoder path (if needed)
    use_decoder = args.disambiguator in ('decoder', 'both')
    use_rf = args.disambiguator in ('rf', 'both')

    decoder_preds = None
    if use_decoder:
        lprint(f"  Loading cached encoder memory...")
        mem_dir = ckpt_dir / 'encoder_memory'
        test_memory = torch.load(mem_dir / 'test_memory.pt', map_location='cpu', weights_only=False)
        test_op_pred = torch.load(mem_dir / 'test_op_pred.pt', map_location='cpu', weights_only=False)
        cached_test = CachedDecoderDataset(
            test_ds, test_memory, test_op_pred,
            multi_window_context=args.multi_window_context,
        )
        lprint(f"  Memory shape: {test_memory.shape}, MWC={args.multi_window_context}")

        ckpt_path = ckpt_dir / 'decoder_checkpoint' / 'best_decoder.pt'
        lprint(f"  Loading decoder from {ckpt_path}")
        max_windows = int(train_ds.total_windows.max().item())
        decoder, best_epoch = reconstruct_decoder(ckpt_path, tokenizer, max_windows, device)
        n_params = sum(p.numel() for p in decoder.parameters())
        lprint(f"  Decoder: {n_params:,} params, best epoch {best_epoch}")

        lprint(f"  Running decoder inference (beam={args.beam_width})...")
        t0 = time.time()
        decoder_preds = get_decoder_predictions(
            decoder, cached_test, device,
            beam_width=args.beam_width, batch_size=args.batch_size)
        lprint(f"  Decoder inference: {time.time()-t0:.1f}s")

    # RF path (if needed)
    rf_preds = None
    if use_rf:
        lprint(f"\n  Training RF coordinate disambiguator...")
        rf_model, rf_le, test_features = train_rf_disambiguator(
            args.data_dir, fold, amb_map, lprint)

        if rf_model is not None:
            # Build RF predictions: lookup for unambiguous, RF for ambiguous
            rf_preds = []
            amb_keys = {k for k, v in amb_map['ambiguity'].items() if v['is_ambiguous']}
            key_to_gcode_set = {}
            for k, v in amb_map['ambiguity'].items():
                if v['is_ambiguous']:
                    key_to_gcode_set[k] = set(
                        text for text, c in amb_map['pos_op_counters'][k].items()
                    )

            for i in range(n_test):
                widx, op = test_widx[i], test_ops[i]
                key = (widx, op)
                if key in amb_keys:
                    feat = test_features[i:i+1]
                    pred_label = rf_model.predict(feat)[0]
                    pred_text = rf_le.inverse_transform([pred_label])[0]
                    # Ensure prediction is valid for this position
                    if key in key_to_gcode_set and pred_text not in key_to_gcode_set[key]:
                        pred_text = amb_map['pos_op_majority'][key]
                    canon = tokenizer.canonicalize([pred_text])
                    tok_strings = tokenizer.tokenize_canonical(canon)
                    tok_ids = [tokenizer._tok2id(t) for t in tok_strings]
                    rf_preds.append(tok_ids + [EOS])
                else:
                    rf_preds.append(lookup_preds[i])

    # Apply hybrid gates
    if decoder_preds is not None:
        hybrid_decoder_preds, routing_decoder = hybrid_gate(
            amb_map, test_widx, test_ops, lookup_preds, decoder_preds,
            threshold=args.ambiguity_threshold)
    else:
        hybrid_decoder_preds, routing_decoder = None, None

    hybrid_rf_preds = rf_preds  # already gated (lookup for unambiguous, RF for ambiguous)

    # Compute metrics for all configurations
    lprint(f"\n  Computing metrics...")
    results = {
        'fold': fold,
        'n_test': n_test,
        'n_unambiguous': n_unamb,
        'n_ambiguous': n_amb,
        'n_unseen': n_unseen,
        'ambiguity_threshold': args.ambiguity_threshold,
        'beam_width': args.beam_width,
        'disambiguator': args.disambiguator,
    }

    # Lookup (always computed)
    results['lookup_all'] = compute_metrics(lookup_preds, test_ids, test_types, tokenizer)
    lprint(f"  lookup:       seq={results['lookup_all']['sequence_accuracy']:.1%}  "
           f"tok={results['lookup_all']['token_accuracy']:.1%}  "
           f"num={results['lookup_all']['numeric_accuracy']:.1%}")

    # Decoder (if available)
    if decoder_preds is not None:
        results['decoder_all'] = compute_metrics(decoder_preds, test_ids, test_types, tokenizer)
        results['hybrid_decoder'] = compute_metrics(hybrid_decoder_preds, test_ids, test_types, tokenizer)
        lprint(f"  decoder:      seq={results['decoder_all']['sequence_accuracy']:.1%}  "
               f"tok={results['decoder_all']['token_accuracy']:.1%}  "
               f"num={results['decoder_all']['numeric_accuracy']:.1%}")
        lprint(f"  hybrid+dec:   seq={results['hybrid_decoder']['sequence_accuracy']:.1%}  "
               f"tok={results['hybrid_decoder']['token_accuracy']:.1%}  "
               f"num={results['hybrid_decoder']['numeric_accuracy']:.1%}")

    # RF (if available)
    if rf_preds is not None:
        results['hybrid_rf'] = compute_metrics(rf_preds, test_ids, test_types, tokenizer)
        lprint(f"  hybrid+RF:    seq={results['hybrid_rf']['sequence_accuracy']:.1%}  "
               f"tok={results['hybrid_rf']['token_accuracy']:.1%}  "
               f"num={results['hybrid_rf']['numeric_accuracy']:.1%}")

    # Subset metrics for ambiguous windows
    amb_indices = [i for i in range(n_test) if sample_class[i] in ('ambiguous', 'unseen')]
    unamb_indices = [i for i in range(n_test) if sample_class[i] == 'unambiguous']

    if amb_indices:
        sub_true = [test_ids[i] for i in amb_indices]
        sub_types = [test_types[i] for i in amb_indices]

        results['lookup_ambiguous'] = compute_metrics(
            [lookup_preds[i] for i in amb_indices], sub_true, sub_types, tokenizer)
        lprint(f"\n  AMBIGUOUS subset ({len(amb_indices)} samples):")
        lprint(f"    lookup:  seq={results['lookup_ambiguous']['sequence_accuracy']:.1%}  "
               f"num={results['lookup_ambiguous']['numeric_accuracy']:.1%}")

        if decoder_preds is not None:
            results['decoder_ambiguous'] = compute_metrics(
                [decoder_preds[i] for i in amb_indices], sub_true, sub_types, tokenizer)
            lprint(f"    decoder: seq={results['decoder_ambiguous']['sequence_accuracy']:.1%}  "
                   f"num={results['decoder_ambiguous']['numeric_accuracy']:.1%}")

        if rf_preds is not None:
            results['rf_ambiguous'] = compute_metrics(
                [rf_preds[i] for i in amb_indices], sub_true, sub_types, tokenizer)
            lprint(f"    RF:      seq={results['rf_ambiguous']['sequence_accuracy']:.1%}  "
                   f"num={results['rf_ambiguous']['numeric_accuracy']:.1%}")

    if unamb_indices:
        sub_true = [test_ids[i] for i in unamb_indices]
        sub_types = [test_types[i] for i in unamb_indices]
        results['lookup_unambiguous'] = compute_metrics(
            [lookup_preds[i] for i in unamb_indices], sub_true, sub_types, tokenizer)
        lprint(f"  UNAMBIGUOUS subset ({len(unamb_indices)} samples):")
        lprint(f"    lookup:  seq={results['lookup_unambiguous']['sequence_accuracy']:.1%}  "
               f"num={results['lookup_unambiguous']['numeric_accuracy']:.1%}")

    # Routing stats
    if routing_decoder is not None:
        routing_counts = Counter(routing_decoder)
        results['routing_decoder'] = dict(routing_counts)
        lprint(f"\n  Routing (decoder): {dict(routing_counts)}")
    if rf_preds is not None:
        n_rf_amb = sum(1 for i in range(n_test)
                       if sample_class[i] in ('ambiguous', 'unseen'))
        results['routing_rf'] = {'lookup': n_test - n_rf_amb, 'rf': n_rf_amb}
        lprint(f"  Routing (RF): lookup={n_test - n_rf_amb}, rf={n_rf_amb}")

    # Ambiguity stats
    amb_stats = {
        'n_positions_total': n_positions,
        'n_positions_ambiguous': n_ambiguous_positions,
    }
    if n_ambiguous_positions > 0:
        amb_entropies = [v['entropy'] for v in amb_map['ambiguity'].values() if v['is_ambiguous']]
        amb_uniques = [v['n_unique'] for v in amb_map['ambiguity'].values() if v['is_ambiguous']]
        amb_stats['mean_entropy'] = round(np.mean(amb_entropies), 4)
        amb_stats['mean_n_unique'] = round(np.mean(amb_uniques), 2)
    results['ambiguity_stats'] = amb_stats

    # Save results
    with open(output_dir / 'hybrid_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save ambiguity map (JSON-serializable version)
    amb_json = {}
    for key, info in amb_map['ambiguity'].items():
        amb_json[f"{key[0]}_{key[1]}"] = info
    with open(output_dir / 'ambiguity_map.json', 'w') as f:
        json.dump(amb_json, f, indent=2)

    # Save per-sample predictions
    def tokens_to_text(token_ids):
        """Convert token IDs to readable G-code string."""
        toks = tokenizer.decode([t for t in token_ids if t not in (PAD, BOS, EOS)])
        return ' '.join(toks)

    per_sample = []
    for i in range(n_test):
        entry = {
            'window_index': test_widx[i],
            'op_type': test_ops[i],
            'true_text': test_texts[i],
            'lookup_text': tokens_to_text(lookup_preds[i]),
            'class': sample_class[i],
        }
        if decoder_preds is not None:
            entry['decoder_text'] = tokens_to_text(decoder_preds[i])
            entry['hybrid_decoder_text'] = tokens_to_text(hybrid_decoder_preds[i])
        if rf_preds is not None:
            entry['hybrid_rf_text'] = tokens_to_text(rf_preds[i])
        per_sample.append(entry)
    with open(output_dir / 'per_sample_predictions.json', 'w') as f:
        json.dump(per_sample, f, indent=2)

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global log_fh

    parser = argparse.ArgumentParser(description="Hybrid Position-Gated Decoding")
    parser.add_argument("--folds", nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--vocab", type=str, default="data/gcode_vocab_712.json")
    parser.add_argument("--decoder_dir", type=str,
                        default="outputs/decoder20260304/v8_optimized_encoder")
    parser.add_argument("--data_dir", type=str,
                        default="outputs/decoder20260304/preprocessed_v8_f98")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/decoder20260304/hybrid_position_decoding")
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--multi_window_context", type=int, default=2)
    parser.add_argument("--ambiguity_threshold", type=float, default=1.0,
                        help="Majority fraction threshold. 1.0=binary gate, 0.9=soft gate")
    parser.add_argument("--disambiguator", type=str, default="both",
                        choices=["decoder", "rf", "both"],
                        help="What to use for ambiguous windows: decoder, rf, or both")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_fh = open(output_dir / 'hybrid_eval.log', 'w')

    lprint(f"Hybrid Position-Gated Decoding")
    lprint(f"Started: {datetime.now()}")
    lprint(f"Device: {args.device}")
    lprint(f"Folds: {args.folds}")
    lprint(f"Beam width: {args.beam_width}")
    lprint(f"Ambiguity threshold: {args.ambiguity_threshold}")
    lprint(f"MWC: {args.multi_window_context}")

    all_results = []
    for fold in args.folds:
        result = run_fold(fold, args)
        all_results.append(result)

    # Aggregate
    lprint(f"\n{'='*70}")
    lprint(f"CROSS-FOLD SUMMARY")
    lprint(f"{'='*70}")

    # Determine which metrics exist
    metrics_to_agg = ['lookup_all']
    if any('decoder_all' in r for r in all_results):
        metrics_to_agg.extend(['decoder_all', 'hybrid_decoder'])
    if any('hybrid_rf' in r for r in all_results):
        metrics_to_agg.append('hybrid_rf')
    metrics_to_agg.extend(['lookup_ambiguous'])
    if any('decoder_ambiguous' in r for r in all_results):
        metrics_to_agg.append('decoder_ambiguous')
    if any('rf_ambiguous' in r for r in all_results):
        metrics_to_agg.append('rf_ambiguous')

    summary = {}
    for config in metrics_to_agg:
        seq_accs = [r[config]['sequence_accuracy'] for r in all_results
                    if config in r and r[config].get('n_samples', 0) > 0]
        tok_accs = [r[config]['token_accuracy'] for r in all_results
                    if config in r and r[config].get('n_samples', 0) > 0]
        num_accs = [r[config]['numeric_accuracy'] for r in all_results
                    if config in r and r[config].get('n_samples', 0) > 0]
        cmd_accs = [r[config]['command_accuracy'] for r in all_results
                    if config in r and r[config].get('n_samples', 0) > 0]

        summary[config] = {
            'seq_mean': round(np.mean(seq_accs), 4) if seq_accs else None,
            'seq_std': round(np.std(seq_accs), 4) if seq_accs else None,
            'tok_mean': round(np.mean(tok_accs), 4) if tok_accs else None,
            'num_mean': round(np.mean(num_accs), 4) if num_accs else None,
            'cmd_mean': round(np.mean(cmd_accs), 4) if cmd_accs else None,
            'per_fold_seq': [round(s, 4) for s in seq_accs],
        }

    # Print table
    lprint(f"\n{'Method':<25} {'Seq%':>8} {'Tok%':>8} {'Num%':>8} {'Cmd%':>8}")
    lprint(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for config in metrics_to_agg:
        s = summary[config]
        if s['seq_mean'] is None:
            continue
        lprint(f"{config:<25} {s['seq_mean']*100:7.1f}% {s['tok_mean']*100:7.1f}% "
               f"{s['num_mean']*100:7.1f}% {s['cmd_mean']*100:7.1f}%")

    # Per-fold breakdown
    lprint(f"\nPer-fold sequence accuracy:")
    lprint(f"{'Method':<25} {'F1':>7} {'F2':>7} {'F3':>7} {'F4':>7} {'F5':>7} {'Mean':>8}")
    lprint(f"{'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    fold_configs = ['lookup_all']
    if 'decoder_all' in metrics_to_agg:
        fold_configs.extend(['decoder_all', 'hybrid_decoder'])
    if 'hybrid_rf' in metrics_to_agg:
        fold_configs.append('hybrid_rf')
    for config in fold_configs:
        vals = summary[config]['per_fold_seq']
        row = f"{config:<25}"
        for v in vals:
            row += f" {v*100:6.1f}%"
        row += f" {summary[config]['seq_mean']*100:7.1f}%"
        lprint(row)

    # Ambiguity summary
    total_amb = sum(r['n_ambiguous'] + r['n_unseen'] for r in all_results)
    total_test = sum(r['n_test'] for r in all_results)
    lprint(f"\nAmbiguous + unseen windows: {total_amb}/{total_test} "
           f"({total_amb/total_test:.1%} of test set)")

    # Save summary
    summary['per_fold_results'] = all_results
    summary['total_ambiguous_windows'] = total_amb
    summary['total_test_windows'] = total_test
    with open(output_dir / 'hybrid_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    lprint(f"\nAll outputs saved to {output_dir}")
    lprint(f"Completed: {datetime.now()}")
    log_fh.close()


if __name__ == "__main__":
    main()
