#!/usr/bin/env python3
"""
Quick Test: SensorMultiHeadDecoder on Frozen 20260225 Encoder.

Attaches the SensorMultiHeadDecoder to the frozen MM-DTAE-LSTM encoder from
experiment 20260225, and trains it to generate G-code from sensor embeddings.

Usage:
    # Direct paths:
    python scripts/evaluation/run_decoder_quick_test.py \
        --data_dir outputs/experiments_2026_02_25/full_w128_s32_cv/fold_1/preprocessed \
        --encoder_ckpt outputs/experiments_2026_02_25/full_w128_s32_cv/fold_1/encoder/checkpoint/best_model.pt \
        --vocab data/gcode_vocab_712.json \
        --output_dir outputs/decoder20260304/fold_1 \
        --epochs 250 --patience 50 --batch_size 32

    # With encoder config mapping (for wandb sweep):
    python scripts/evaluation/run_decoder_quick_test.py \
        --encoder_config f110_w128_s32 \
        --vocab data/gcode_vocab_712.json \
        --output_dir outputs/decoder20260304/sweep/run_abc \
        --epochs 250 --patience 50 --wandb

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
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.model.digit_value_head import DigitByDigitLoss
from miracle.utilities.gcode_tokenizer import GCodeTokenizer
from miracle.dataset.target_utils import decompose_value_to_digits, SIGN_PAD, DIGIT_PAD

# ── Constants ──────────────────────────────────────────────────────────────────
PAD = 0
BOS = 1
EOS = 2
UNK = 3

# Token types
TYPE_SPECIAL = 0
TYPE_COMMAND = 1
TYPE_PARAM = 2
TYPE_NUMERIC = 3

# Commands and params (matching SensorMultiHeadDecoder expectations)
COMMANDS = ['G0', 'G1', 'G2', 'G3', 'G53', 'M30']
PARAMS = ['X', 'Y', 'Z', 'F', 'R', 'S', 'I', 'J', 'K']
CMD2ID = {c: i for i, c in enumerate(COMMANDS)}
PARAM2ID = {p: i for i, p in enumerate(PARAMS)}

# ── Encoder Config Mapping ────────────────────────────────────────────────────
ENCODER_BASE = Path("outputs/experiments_2026_02_25")
V7_DATA_BASE = Path("outputs/decoder20260304/preprocessed_v7")

ENCODER_CONFIGS = {
    'f110_w256_s64':  'full_w256_s64_cv',
    'f110_w256_s128': 'full_w256_s128_cv',
    'f110_w128_s32':  'full_w128_s32_cv',
    'f110_w128_s64':  'full_w128_s64_cv',
    'f110_w64_s16':   'full_w64_s16_cv',
    'f110_w64_s32':   'full_w64_s32_cv',
    'f110_w64_s64':   'full_w64_s64_cv',
    'f110_w32_s8':    'full_w32_s8_cv',
    'f110_w16_s4':    'full_w16_s4_cv',
    'f110_w16_s8':    'full_w16_s8_cv',
    'f104_w64_s16':   'no_proximity_w64_s16_cv',
    'f98_w256_s64':   'no_proximity_no_pressure_w256_s64_cv',
    'f98_w64_s16':    'no_proximity_no_pressure_w64_s16_cv',
    'f74_w64_s16':    'no_proximity_no_pressure_no_color_w64_s16_cv',
    'f56_w256_s128':  'no_proximity_no_pressure_no_color_no_magnetometer_w256_s128_cv',
    'f56_w256_s64':   'no_proximity_no_pressure_no_color_no_magnetometer_w256_s64_cv',
    'f56_w128_s32':   'no_proximity_no_pressure_no_color_no_magnetometer_w128_s32_cv',
    'f56_w128_s64':   'no_proximity_no_pressure_no_color_no_magnetometer_w128_s64_cv',
    'f56_w64_s16':    'no_proximity_no_pressure_no_color_no_magnetometer_w64_s16_cv',
    'f56_w64_s32':    'no_proximity_no_pressure_no_color_no_magnetometer_w64_s32_cv',
    'f56_w64_s64':    'no_proximity_no_pressure_no_color_no_magnetometer_w64_s64_cv',
    'f56_w32_s8':     'no_proximity_no_pressure_no_color_no_magnetometer_w32_s8_cv',
    'f56_w16_s4':     'no_proximity_no_pressure_no_color_no_magnetometer_w16_s4_cv',
    'f56_w16_s8':     'no_proximity_no_pressure_no_color_no_magnetometer_w16_s8_cv',
}

# V7 configs: use re-preprocessed data from data_clean/ with window position metadata
# Encoder checkpoints are reused from the original experiments
V7_ENCODER_CONFIGS = {
    'v7_f110_w256_s64': {
        'data_dir': V7_DATA_BASE,   # fold_N appended at runtime
        'encoder_dir': 'full_w256_s64_cv',  # encoder checkpoint from original
    },
    'v8_f98_w256_s64': {
        'data_dir': Path('outputs/decoder20260304/preprocessed_v8_f98'),  # fold_N appended at runtime
        'encoder_dir': 'no_proximity_no_pressure_w256_s64_cv',
    },
    'v8_f98_w64_s16': {
        'data_dir': Path('outputs/decoder20260304/preprocessed_v8_f98_w64'),  # fold_N appended at runtime
        'encoder_dir': 'no_proximity_no_pressure_w64_s16_cv',
    },
}


def log(msg, log_file=None):
    print(msg, flush=True)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


# ── Modality Indexing ──────────────────────────────────────────────────────────

def build_modality_indices(columns):
    """Build modality index groups from column names.
    Replicates logic from run_9class_direct.py:67-91.
    """
    sensor_patterns = {
        'accelerometer': ['Ax', 'Ay', 'Az'],
        'gyroscope': ['Gx', 'Gy', 'Gz'],
        'magnetometer': ['Mx', 'My', 'Mz'],
        'environmental': ['Pressure', 'Temperature', 'Proximity'],
        'color': ['ColorR', 'ColorG', 'ColorB', 'ColorA'],
        'rms': ['RMS'],
    }
    groups = {name: [] for name in sensor_patterns}
    groups['electrical'] = []
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
            groups['electrical'].append(idx)
    group_names = list(sensor_patterns.keys()) + ['electrical']
    group_indices = [groups[n] for n in group_names]
    sensor_dims = [len(groups[n]) for n in group_names]
    return group_names, group_indices, sensor_dims


# ── Token Parsing for Structured Targets ───────────────────────────────────────

def classify_token(tok_str, precision):
    """Classify a token string and extract structured targets.

    Returns:
        (type_id, command_id, param_type_id, sign, digits, value)
        - type_id: 0=SPECIAL, 1=COMMAND, 2=PARAM, 3=NUMERIC
        - command_id: 0-5 for commands, -1 otherwise
        - param_type_id: 0-8 for params/numerics, -1 otherwise
        - sign: 0=+, 1=-, 2=pad (for non-numeric)
        - digits: list of 6 ints (0-9 or DIGIT_PAD=10 for non-numeric)
        - value: float (0.0 for non-numeric)
    """
    n_digits = 6  # max_int_digits(2) + n_decimal_digits(4)
    pad_digits = [DIGIT_PAD] * n_digits

    # Special tokens
    if tok_str in ('PAD', 'BOS', 'EOS', 'UNK', 'MASK'):
        return TYPE_SPECIAL, -1, -1, SIGN_PAD, pad_digits, 0.0

    # Command tokens
    if tok_str in CMD2ID:
        return TYPE_COMMAND, CMD2ID[tok_str], -1, SIGN_PAD, pad_digits, 0.0

    # Parameter letter tokens
    if tok_str in PARAM2ID:
        return TYPE_PARAM, -1, PARAM2ID[tok_str], SIGN_PAD, pad_digits, 0.0

    # Numeric tokens: NUM_X_1492 → addr='X', bucket=1492
    if tok_str.startswith('NUM_'):
        parts = tok_str.split('_')
        if len(parts) >= 3:
            addr = parts[1]
            try:
                bucket = int(parts[2])
            except ValueError:
                return TYPE_NUMERIC, -1, PARAM2ID.get(addr, -1), SIGN_PAD, pad_digits, 0.0
            step = precision.get(addr, 0.001)
            value = bucket * step
            param_type_id = PARAM2ID.get(addr, -1)
            sign, digits = decompose_value_to_digits(value, max_int_digits=2, n_decimal_digits=4)
            return TYPE_NUMERIC, -1, param_type_id, sign, digits, value

    # Fallback: treat as special (could be compound tokens like "F22.", "Z0.", etc.)
    # Check if it starts with a known command prefix
    for cmd in COMMANDS:
        if tok_str.startswith(cmd[0]) and tok_str[1:].replace('.', '').isdigit():
            return TYPE_COMMAND, CMD2ID.get(cmd, -1), -1, SIGN_PAD, pad_digits, 0.0

    return TYPE_SPECIAL, -1, -1, SIGN_PAD, pad_digits, 0.0


# ── FocalLoss ──────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss: (1-pt)^gamma * CE. Downweights easy/frequent tokens."""

    def __init__(self, gamma=2.0, ignore_index=0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, label_smoothing=label_smoothing, reduction='none'
        )

    def forward(self, input, target):
        ce_loss = self.ce(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        return focal_loss.mean()


# ── Dataset ────────────────────────────────────────────────────────────────────

class DecoderQuickTestDataset(Dataset):
    """Loads .npz sensor data, re-tokenizes with vocab, builds structured targets."""

    def __init__(self, npz_path, tokenizer, max_token_len=16, sequence_class_map=None):
        data = np.load(npz_path, allow_pickle=True)
        self.continuous = torch.from_numpy(data['continuous'].astype(np.float32))
        self.operation_type = torch.from_numpy(data['operation_type'].astype(np.int64))
        gcode_texts = data['gcode_texts']
        precision = tokenizer.cfg.precision

        # Window position metadata (V6)
        if 'window_index' in data:
            self.window_index = torch.from_numpy(data['window_index'].astype(np.int64))
            self.total_windows = torch.from_numpy(data['total_windows'].astype(np.int64))
        else:
            n = len(self.continuous)
            self.window_index = torch.zeros(n, dtype=torch.long)
            self.total_windows = torch.ones(n, dtype=torch.long)

        # Source file for file-level MWC (V6)
        if 'source_file' in data:
            self.source_files = list(data['source_file'])
        else:
            self.source_files = [f'unknown_{i}' for i in range(len(self.continuous))]

        # G-code text strings for oversampling / sequence classification
        self.gcode_texts_list = [str(t) for t in gcode_texts]

        # Sequence class mapping (V6: two-stage classifier)
        if sequence_class_map is not None:
            self.sequence_class = torch.tensor(
                [sequence_class_map.get(str(t), 0) for t in gcode_texts], dtype=torch.long
            )
        else:
            self.sequence_class = torch.zeros(len(self.continuous), dtype=torch.long)

        n_samples = len(self.continuous)
        self.input_tokens = torch.full((n_samples, max_token_len), PAD, dtype=torch.long)
        self.target_tokens = torch.full((n_samples, max_token_len), PAD, dtype=torch.long)
        self.type_targets = torch.full((n_samples, max_token_len), -1, dtype=torch.long)
        self.command_targets = torch.full((n_samples, max_token_len), -1, dtype=torch.long)
        self.param_type_targets = torch.full((n_samples, max_token_len), -1, dtype=torch.long)
        self.sign_targets = torch.full((n_samples, max_token_len), SIGN_PAD, dtype=torch.long)
        self.digit_targets = torch.full((n_samples, max_token_len, 6), DIGIT_PAD, dtype=torch.long)
        self.values = torch.zeros((n_samples, max_token_len), dtype=torch.float32)
        self.numeric_mask = torch.zeros((n_samples, max_token_len), dtype=torch.bool)

        unk_count = 0
        total_tokens = 0
        token_lengths = []

        for i, text in enumerate(gcode_texts):
            text_str = str(text)
            # Support multi-line G-code (newline-separated)
            lines = [l for l in text_str.split('\n') if l.strip()]
            canon = tokenizer.canonicalize(lines)
            tok_strings = tokenizer.tokenize_canonical(canon)
            tok_ids = [tokenizer._tok2id(t) for t in tok_strings]

            # Count UNKs
            for tid in tok_ids:
                total_tokens += 1
                if tid == UNK:
                    unk_count += 1
            token_lengths.append(len(tok_ids))

            # Truncate to max_token_len - 1 (leave room for BOS/EOS)
            tok_ids = tok_ids[:max_token_len - 1]
            tok_strings = tok_strings[:max_token_len - 1]

            # Input: [BOS, t1, t2, ..., tn]
            inp = [BOS] + tok_ids
            # Target: [t1, t2, ..., tn, EOS]
            tgt = tok_ids + [EOS]

            inp_len = len(inp)
            tgt_len = len(tgt)
            self.input_tokens[i, :inp_len] = torch.tensor(inp, dtype=torch.long)
            self.target_tokens[i, :tgt_len] = torch.tensor(tgt, dtype=torch.long)

            # Structured targets (aligned with TARGET tokens, not input)
            target_strs = tok_strings + ['EOS']
            for j, ts in enumerate(target_strs):
                if j >= max_token_len:
                    break
                type_id, cmd_id, pt_id, sign, digits, value = classify_token(ts, precision)
                self.type_targets[i, j] = type_id
                self.command_targets[i, j] = cmd_id
                self.param_type_targets[i, j] = pt_id
                self.sign_targets[i, j] = sign
                self.digit_targets[i, j] = torch.tensor(digits, dtype=torch.long)
                self.values[i, j] = value
                if type_id == TYPE_NUMERIC:
                    self.numeric_mask[i, j] = True

        self.stats = {
            'n_samples': n_samples,
            'total_tokens': total_tokens,
            'unk_count': unk_count,
            'unk_rate': unk_count / max(total_tokens, 1),
            'mean_token_len': float(np.mean(token_lengths)),
            'max_token_len': int(np.max(token_lengths)),
            'min_token_len': int(np.min(token_lengths)),
        }

    def __len__(self):
        return len(self.continuous)

    def __getitem__(self, idx):
        return {
            'sensor_features': self.continuous[idx],
            'operation_type': self.operation_type[idx],
            'input_tokens': self.input_tokens[idx],
            'target_tokens': self.target_tokens[idx],
            'type_targets': self.type_targets[idx],
            'command_targets': self.command_targets[idx],
            'param_type_targets': self.param_type_targets[idx],
            'sign_targets': self.sign_targets[idx],
            'digit_targets': self.digit_targets[idx],
            'values': self.values[idx],
            'numeric_mask': self.numeric_mask[idx],
            'window_index': self.window_index[idx],
            'total_windows': self.total_windows[idx],
            'sequence_class': self.sequence_class[idx],
        }


def decoder_collate_fn(batch):
    """Collate with variable-length memory padding for multi-window context."""
    result = {}
    for k in batch[0]:
        tensors = [b[k] for b in batch]
        if k == 'memory' and tensors[0].dim() >= 1:
            # Pad memory to max length in batch (handles variable MWC neighbors)
            max_len = max(t.shape[0] for t in tensors)
            if any(t.shape[0] != max_len for t in tensors):
                padded = []
                for t in tensors:
                    pad_len = max_len - t.shape[0]
                    if pad_len > 0:
                        padded.append(torch.nn.functional.pad(t, (0, 0, 0, pad_len)))
                    else:
                        padded.append(t)
                result[k] = torch.stack(padded)
            else:
                result[k] = torch.stack(tensors)
        else:
            result[k] = torch.stack(tensors)
    return result


# ── Encoder Loading ────────────────────────────────────────────────────────────

def load_frozen_encoder(ckpt_path, device):
    """Load the MM-DTAE-LSTM encoder from 20260225 checkpoint, frozen."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = MM_DTAE_LSTM(config)
    # CRITICAL: override head_cls from 5 to 9 classes before loading state dict
    model.head_cls = nn.Linear(config.d_model, 9)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config, ckpt


# ── Memory Caching ─────────────────────────────────────────────────────────────

def cache_encoder_memory(encoder, dataset, group_indices, device, batch_size=32, cache_dir=None):
    """Pre-compute encoder memory for all samples, optionally caching to disk."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=decoder_collate_fn, num_workers=0)

    all_memory = []
    all_op_pred = []
    all_cls_correct = 0
    all_cls_total = 0

    with torch.no_grad():
        for batch in loader:
            sensor_data = batch['sensor_features'].to(device)
            gt_ops = batch['operation_type'].to(device)
            B, T = sensor_data.shape[:2]
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
            mods = [sensor_data[:, :, idx] for idx in group_indices]

            out = encoder(mods, lengths)
            memory = out['memory']  # [B, T, 256]
            op_pred = out['cls'].argmax(1)  # [B]

            all_memory.append(memory.cpu())
            all_op_pred.append(op_pred.cpu())
            all_cls_correct += (op_pred == gt_ops).sum().item()
            all_cls_total += B

    all_memory = torch.cat(all_memory, dim=0)
    all_op_pred = torch.cat(all_op_pred, dim=0)
    cls_acc = all_cls_correct / max(all_cls_total, 1)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(all_memory, cache_dir / 'memory.pt')
        torch.save(all_op_pred, cache_dir / 'op_pred.pt')

    return all_memory, all_op_pred, cls_acc


class CachedDecoderDataset(Dataset):
    """Wraps a DecoderQuickTestDataset with pre-cached encoder memory.

    If multi_window_context > 0, concatenates neighboring windows' memory
    using file-level grouping (not index-based) to give the decoder
    temporal context from actual neighboring windows in the same file.
    """

    def __init__(self, base_dataset, memory, op_pred, multi_window_context=0,
                 noise_scale=0.0, window_dropout=0.0, training=False):
        self.base = base_dataset
        self.memory = memory
        self.op_pred = op_pred
        self.multi_window_context = multi_window_context
        self.noise_scale = noise_scale
        self.window_dropout = window_dropout
        self.training = training
        assert len(memory) == len(base_dataset)

        # Build file-level grouping for proper MWC
        self.file_groups = {}  # source_file -> sorted list of dataset indices
        if hasattr(base_dataset, 'source_files'):
            from collections import defaultdict
            groups = defaultdict(list)
            for i, sf in enumerate(base_dataset.source_files):
                groups[sf].append(i)
            # Sort each group by window_index for correct ordering
            for sf in groups:
                groups[sf].sort(key=lambda i: base_dataset.window_index[i].item()
                                if hasattr(base_dataset, 'window_index') else i)
            self.file_groups = dict(groups)
            self.idx_to_file = base_dataset.source_files

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if self.multi_window_context > 0 and self.file_groups:
            # Use file-level neighbors (not index-based)
            sf = self.idx_to_file[idx]
            file_idxs = self.file_groups[sf]
            try:
                my_pos = file_idxs.index(idx)
            except ValueError:
                my_pos = 0
            ctx = self.multi_window_context
            start = max(0, my_pos - ctx)
            end = min(len(file_idxs), my_pos + ctx + 1)
            neighbor_idxs = [file_idxs[i] for i in range(start, end)]

            # Window dropout during training: randomly drop context windows
            if self.training and self.window_dropout > 0 and len(neighbor_idxs) > 1:
                import random
                kept = [ni for ni in neighbor_idxs
                        if ni == idx or random.random() > self.window_dropout]
                if not kept:
                    kept = [idx]
                neighbor_idxs = kept

            memories = [self.memory[ni] for ni in neighbor_idxs]
            item['memory'] = torch.cat(memories, dim=0)
        elif self.multi_window_context > 0:
            # Fallback: index-based (legacy behavior)
            ctx = self.multi_window_context
            memories = []
            for offset in range(-ctx, ctx + 1):
                neighbor = max(0, min(len(self.memory) - 1, idx + offset))
                memories.append(self.memory[neighbor])
            item['memory'] = torch.cat(memories, dim=0)
        else:
            item['memory'] = self.memory[idx]

        # Noise injection during training
        if self.training and self.noise_scale > 0:
            item['memory'] = item['memory'] + torch.randn_like(item['memory']) * self.noise_scale

        item['op_pred'] = self.op_pred[idx]
        return item


# ── Training ───────────────────────────────────────────────────────────────────

def get_active_losses(curriculum, epoch):
    """Return set of active loss names for this epoch given curriculum mode."""
    # Regression, sequence, and pointer losses are always active when enabled
    always_on = {'regression', 'sequence', 'pointer'}
    if curriculum == '3phase':
        if epoch <= 30:
            return {'type', 'command', 'param_type'} | always_on
        elif epoch <= 80:
            return {'type', 'command', 'param_type', 'digit'} | always_on
        else:
            return {'type', 'command', 'param_type', 'digit', 'legacy'} | always_on
    else:  # 'none'
        return {'type', 'command', 'param_type', 'digit', 'legacy'} | always_on


def train_epoch(decoder, loader, optimizer, device, loss_fns, loss_weights, epoch,
                curriculum='none', scheduled_sampling=0.0, total_epochs=250,
                pointer_target_map=None):
    """Train one epoch with full multi-head loss."""
    active_losses = get_active_losses(curriculum, epoch)
    decoder.train()
    total_loss = 0.0
    loss_accum = defaultdict(float)
    n_batches = 0
    token_correct = 0
    token_total = 0
    type_correct = 0
    type_total = 0

    # Scheduled sampling ratio (linearly ramp over 70% of training)
    ss_ratio = 0.0
    if scheduled_sampling > 0:
        ss_ratio = scheduled_sampling * min(1.0, epoch / (total_epochs * 0.7))

    for batch in loader:
        input_tokens = batch['input_tokens'].to(device)
        target_tokens = batch['target_tokens'].to(device)
        memory = batch['memory'].to(device)
        op_pred = batch['op_pred'].to(device)
        padding_mask = (input_tokens == PAD)

        # Scheduled sampling: replace some teacher-forced tokens with model predictions
        if ss_ratio > 0:
            with torch.no_grad():
                ss_outputs = decoder(
                    tokens=input_tokens,
                    sensor_embeddings=memory,
                    operation_type=op_pred,
                    tgt_key_padding_mask=padding_mask,
                )
                ss_logits = ss_outputs.get('raw_legacy_logits', ss_outputs.get('legacy_logits'))
                if ss_logits is not None:
                    ss_preds = ss_logits.argmax(-1)
                    B, L = input_tokens.shape
                    ss_mask = torch.rand(B, L, device=device) < ss_ratio
                    ss_mask[:, 0] = False  # keep BOS
                    ss_mask = ss_mask & ~padding_mask  # don't replace padding
                    input_tokens = torch.where(ss_mask, ss_preds, input_tokens)

        # V6: pass window position and sequence class
        extra_kwargs = {}
        if 'window_index' in batch:
            extra_kwargs['window_index'] = batch['window_index'].to(device)
        if 'total_windows' in batch:
            extra_kwargs['total_windows'] = batch['total_windows'].to(device)
        if 'sequence_class' in batch:
            extra_kwargs['sequence_class'] = batch['sequence_class'].to(device)

        outputs = decoder(
            tokens=input_tokens,
            sensor_embeddings=memory,
            operation_type=op_pred,
            tgt_key_padding_mask=padding_mask,
            **extra_kwargs,
        )

        # ── Structured losses ──
        losses = {}

        # Type loss
        type_targets = batch['type_targets'].to(device)
        losses['type'] = loss_fns['type'](
            outputs['type_logits'].reshape(-1, 4),
            type_targets.reshape(-1)
        )

        # Command loss
        cmd_targets = batch['command_targets'].to(device)
        losses['command'] = loss_fns['command'](
            outputs['command_logits'].reshape(-1, outputs['command_logits'].size(-1)),
            cmd_targets.reshape(-1)
        )

        # Param type loss
        pt_targets = batch['param_type_targets'].to(device)
        losses['param_type'] = loss_fns['param_type'](
            outputs['param_type_logits'].reshape(-1, outputs['param_type_logits'].size(-1)),
            pt_targets.reshape(-1)
        )

        # Digit loss
        sign_targets = batch['sign_targets'].to(device)
        digit_targets = batch['digit_targets'].to(device)
        value_targets = batch['values'].to(device)
        numeric_mask = batch['numeric_mask'].to(device)
        digit_loss, digit_metrics = loss_fns['digit'](
            outputs, sign_targets, digit_targets, value_targets, numeric_mask
        )
        losses['digit'] = digit_loss

        # Legacy token loss (use grammar-constrained logits for loss if grammar is on)
        logits = outputs.get('legacy_logits', outputs.get('raw_legacy_logits'))
        if logits is not None:
            losses['legacy'] = loss_fns['legacy'](
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1)
            )

        # Regression loss (MSE on numeric values)
        if 'regression_value' in outputs and 'regression' in loss_fns:
            reg_pred = outputs['regression_value']  # [B, L]
            reg_target = batch['values'].to(device)  # [B, L]
            reg_mask = batch['numeric_mask'].to(device)  # [B, L]
            if reg_mask.any():
                reg_loss = nn.functional.mse_loss(
                    reg_pred[reg_mask], reg_target[reg_mask]
                )
                losses['regression'] = reg_loss

        # V6: Sequence classification loss
        if 'sequence_logits' in outputs and 'sequence' in loss_fns:
            seq_logits = outputs['sequence_logits']  # [B, n_sequences]
            seq_targets = batch['sequence_class'].to(device)  # [B]
            losses['sequence'] = loss_fns['sequence'](seq_logits, seq_targets)

        # V7: Pointer network loss (per-axis classification)
        if 'pointer_logits' in outputs and pointer_target_map is not None and 'pointer' in loss_fns:
            ptr_logits = outputs['pointer_logits']  # Dict[axis, [B, L, n_values]]
            # Build per-axis targets from target_tokens
            B_ptr, L_ptr = target_tokens.shape
            total_ptr_loss = 0.0
            n_ptr_axes = 0
            for axis_name, axis_logits in ptr_logits.items():
                # Build targets for this axis: -1 for non-applicable positions
                axis_targets = torch.full((B_ptr, L_ptr), -1, dtype=torch.long, device=device)
                for b in range(B_ptr):
                    for t in range(L_ptr):
                        tok_id = target_tokens[b, t].item()
                        if tok_id in pointer_target_map:
                            ax, ax_idx = pointer_target_map[tok_id]
                            if ax == axis_name:
                                axis_targets[b, t] = ax_idx
                # Only compute loss if there are valid targets
                valid = axis_targets >= 0
                if valid.any():
                    ptr_loss = loss_fns['pointer'](
                        axis_logits[valid],  # [N_valid, n_values]
                        axis_targets[valid],  # [N_valid]
                    )
                    total_ptr_loss = total_ptr_loss + ptr_loss
                    n_ptr_axes += 1
            if n_ptr_axes > 0:
                losses['pointer'] = total_ptr_loss / n_ptr_axes

        # Weighted total (only active losses per curriculum)
        loss = sum(loss_weights.get(k, 1.0) * v for k, v in losses.items()
                   if v is not None and k in active_losses)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in losses.items():
            loss_accum[k] += v.item() if torch.is_tensor(v) else v
        n_batches += 1

        # Token accuracy (legacy)
        if logits is not None:
            mask = target_tokens != PAD
            preds = logits.argmax(-1)
            token_correct += ((preds == target_tokens) & mask).sum().item()
            token_total += mask.sum().item()

        # Type accuracy
        type_mask = type_targets >= 0
        type_preds = outputs['type_logits'].argmax(-1)
        type_correct += ((type_preds == type_targets) & type_mask).sum().item()
        type_total += type_mask.sum().item()

    avg_loss = total_loss / max(n_batches, 1)
    avg_losses = {k: v / max(n_batches, 1) for k, v in loss_accum.items()}
    token_acc = token_correct / max(token_total, 1)
    type_acc = type_correct / max(type_total, 1)
    return avg_loss, avg_losses, token_acc, type_acc


# ── End-to-End Training (Approach A) ──────────────────────────────────────────

def train_epoch_e2e(encoder, decoder, loader, optimizer, device, loss_fns,
                    loss_weights, epoch, group_indices, curriculum='none',
                    scheduled_sampling=0.0, total_epochs=250,
                    pointer_target_map=None, grad_accum=1, cls_weight=0.1):
    """Train one epoch end-to-end: decoder loss backprops through encoder."""
    active_losses = get_active_losses(curriculum, epoch)
    encoder.train()
    decoder.train()
    total_loss = 0.0
    loss_accum = defaultdict(float)
    n_batches = 0
    token_correct = 0
    token_total = 0
    type_correct = 0
    type_total = 0

    cls_loss_fn = nn.CrossEntropyLoss()

    # Scheduled sampling ratio
    ss_ratio = 0.0
    if scheduled_sampling > 0:
        ss_ratio = scheduled_sampling * min(1.0, epoch / (total_epochs * 0.7))

    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        sensor_data = batch['sensor_features'].to(device)
        input_tokens = batch['input_tokens'].to(device)
        target_tokens = batch['target_tokens'].to(device)
        gt_ops = batch['operation_type'].to(device)
        padding_mask = (input_tokens == PAD)

        B, T = sensor_data.shape[:2]
        lengths = torch.full((B,), T, dtype=torch.long, device=device)
        mods = [sensor_data[:, :, idx] for idx in group_indices]

        # LIVE encoder forward — gradients flow through memory
        enc_out = encoder(mods, lengths)
        memory = enc_out['memory']  # [B, T, 256] WITH gradients
        op_pred = enc_out['cls'].argmax(1)  # [B] detached (argmax not differentiable)

        # Auxiliary CLS loss to prevent catastrophic forgetting
        aux_cls_loss = cls_loss_fn(enc_out['cls'], gt_ops)

        # Scheduled sampling
        if ss_ratio > 0:
            with torch.no_grad():
                ss_outputs = decoder(
                    tokens=input_tokens,
                    sensor_embeddings=memory.detach(),
                    operation_type=op_pred,
                    tgt_key_padding_mask=padding_mask,
                )
                ss_logits = ss_outputs.get('raw_legacy_logits', ss_outputs.get('legacy_logits'))
                if ss_logits is not None:
                    ss_preds = ss_logits.argmax(-1)
                    B_ss, L_ss = input_tokens.shape
                    ss_mask = torch.rand(B_ss, L_ss, device=device) < ss_ratio
                    ss_mask[:, 0] = False
                    ss_mask = ss_mask & ~padding_mask
                    input_tokens = torch.where(ss_mask, ss_preds, input_tokens)

        # Window position kwargs
        extra_kwargs = {}
        if 'window_index' in batch:
            extra_kwargs['window_index'] = batch['window_index'].to(device)
        if 'total_windows' in batch:
            extra_kwargs['total_windows'] = batch['total_windows'].to(device)
        if 'sequence_class' in batch:
            extra_kwargs['sequence_class'] = batch['sequence_class'].to(device)

        # Decoder forward — receives live encoder memory
        outputs = decoder(
            tokens=input_tokens,
            sensor_embeddings=memory,
            operation_type=op_pred,
            tgt_key_padding_mask=padding_mask,
            **extra_kwargs,
        )

        # ── Compute losses (same as train_epoch) ──
        losses = {}

        type_targets = batch['type_targets'].to(device)
        losses['type'] = loss_fns['type'](
            outputs['type_logits'].reshape(-1, 4), type_targets.reshape(-1))

        cmd_targets = batch['command_targets'].to(device)
        losses['command'] = loss_fns['command'](
            outputs['command_logits'].reshape(-1, outputs['command_logits'].size(-1)),
            cmd_targets.reshape(-1))

        pt_targets = batch['param_type_targets'].to(device)
        losses['param_type'] = loss_fns['param_type'](
            outputs['param_type_logits'].reshape(-1, outputs['param_type_logits'].size(-1)),
            pt_targets.reshape(-1))

        sign_targets = batch['sign_targets'].to(device)
        digit_targets = batch['digit_targets'].to(device)
        value_targets = batch['values'].to(device)
        numeric_mask = batch['numeric_mask'].to(device)
        digit_loss, digit_metrics = loss_fns['digit'](
            outputs, sign_targets, digit_targets, value_targets, numeric_mask)
        losses['digit'] = digit_loss

        logits = outputs.get('legacy_logits', outputs.get('raw_legacy_logits'))
        if logits is not None:
            losses['legacy'] = loss_fns['legacy'](
                logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))

        if 'regression_value' in outputs and 'regression' in loss_fns:
            reg_pred = outputs['regression_value']
            reg_target = batch['values'].to(device)
            reg_mask = batch['numeric_mask'].to(device)
            if reg_mask.any():
                losses['regression'] = nn.functional.mse_loss(
                    reg_pred[reg_mask], reg_target[reg_mask])

        if 'sequence_logits' in outputs and 'sequence' in loss_fns:
            losses['sequence'] = loss_fns['sequence'](
                outputs['sequence_logits'], batch['sequence_class'].to(device))

        if 'pointer_logits' in outputs and pointer_target_map is not None and 'pointer' in loss_fns:
            ptr_logits = outputs['pointer_logits']
            B_ptr, L_ptr = target_tokens.shape
            total_ptr_loss = 0.0
            n_ptr_axes = 0
            for axis_name, axis_logits in ptr_logits.items():
                axis_targets = torch.full((B_ptr, L_ptr), -1, dtype=torch.long, device=device)
                for b in range(B_ptr):
                    for t in range(L_ptr):
                        tok_id = target_tokens[b, t].item()
                        if tok_id in pointer_target_map:
                            ax, ax_idx = pointer_target_map[tok_id]
                            if ax == axis_name:
                                axis_targets[b, t] = ax_idx
                valid = axis_targets >= 0
                if valid.any():
                    total_ptr_loss = total_ptr_loss + loss_fns['pointer'](
                        axis_logits[valid], axis_targets[valid])
                    n_ptr_axes += 1
            if n_ptr_axes > 0:
                losses['pointer'] = total_ptr_loss / n_ptr_axes

        # Weighted decoder loss + auxiliary encoder CLS loss
        decoder_loss = sum(loss_weights.get(k, 1.0) * v for k, v in losses.items()
                          if v is not None and k in active_losses)
        loss = decoder_loss + cls_weight * aux_cls_loss
        loss = loss / grad_accum

        loss.backward()

        # Step every grad_accum batches
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        for k, v in losses.items():
            loss_accum[k] += v.item() if torch.is_tensor(v) else v
        loss_accum['aux_cls'] = loss_accum.get('aux_cls', 0) + aux_cls_loss.item()
        n_batches += 1

        # Token accuracy
        if logits is not None:
            mask = target_tokens != PAD
            preds = logits.argmax(-1)
            token_correct += ((preds == target_tokens) & mask).sum().item()
            token_total += mask.sum().item()

        # Type accuracy
        type_mask = type_targets >= 0
        type_preds = outputs['type_logits'].argmax(-1)
        type_correct += ((type_preds == type_targets) & type_mask).sum().item()
        type_total += type_mask.sum().item()

    # Handle leftover gradients
    if n_batches % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(n_batches, 1)
    avg_losses = {k: v / max(n_batches, 1) for k, v in loss_accum.items()}
    token_acc = token_correct / max(token_total, 1)
    type_acc = type_correct / max(type_total, 1)
    return avg_loss, avg_losses, token_acc, type_acc


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def beam_search_decode(decoder, memory, op_pred, device, beam_width=3, max_len=16,
                       extra_kwargs=None, length_penalty=0.6):
    """Autoregressive beam search decoding with grammar constraints.

    Args:
        decoder: SensorMultiHeadDecoder
        memory: Encoder memory [B, T_s, sensor_dim]
        op_pred: Operation type predictions [B]
        device: torch device
        beam_width: Number of beams (1 = greedy autoregressive)
        max_len: Maximum sequence length
        extra_kwargs: Dict with window_index, total_windows, etc. per sample [B]
        length_penalty: Length normalization alpha (0=no norm, 1=full)

    Returns:
        best_sequences: [B, max_len] tensor of predicted token IDs
    """
    B = memory.size(0)
    all_best = []

    for b in range(B):
        mem_b = memory[b:b+1]  # [1, T_s, dim]
        op_b = op_pred[b:b+1]  # [1]

        # Build per-sample extra kwargs (window_index, total_windows, etc.)
        sample_extra = {}
        if extra_kwargs:
            for k, v in extra_kwargs.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    sample_extra[k] = v[b:b+1]

        # Each beam: (log_prob, token_sequence)
        beams = [(0.0, [BOS])]

        for step in range(max_len - 1):
            candidates = []
            for score, seq in beams:
                if seq[-1] == EOS or seq[-1] == PAD:
                    candidates.append((score, seq))
                    continue

                # Forward pass with current sequence
                tokens_t = torch.tensor([seq], device=device)  # [1, len]
                padding = torch.zeros(1, len(seq), dtype=torch.bool, device=device)

                outputs = decoder(
                    tokens=tokens_t,
                    sensor_embeddings=mem_b,
                    operation_type=op_b,
                    tgt_key_padding_mask=padding,
                    **sample_extra,
                )

                # Get logits at the last position
                logits = outputs['legacy_logits'][0, -1]  # [vocab_size]
                log_probs = torch.log_softmax(logits, dim=-1)

                # Get top-k tokens
                topk_probs, topk_ids = log_probs.topk(beam_width)
                for lp, tid in zip(topk_probs.tolist(), topk_ids.tolist()):
                    new_score = score + lp
                    candidates.append((new_score, seq + [tid]))

            # Length-normalize scores for ranking
            def norm_score(item):
                s, seq = item
                length = len(seq)
                return s / ((5 + length) / 6) ** length_penalty

            candidates.sort(key=lambda x: -norm_score(x))
            beams = candidates[:beam_width]

            # Early stop if all beams ended
            if all(s[-1] in (EOS, PAD) for _, s in beams):
                break

        # Pad best beam to max_len
        best_seq = beams[0][1]
        if len(best_seq) < max_len:
            best_seq = best_seq + [PAD] * (max_len - len(best_seq))
        all_best.append(best_seq[:max_len])

    return torch.tensor(all_best, device=device)


@torch.no_grad()
def evaluate(decoder, loader, device, loss_fns, tokenizer, beam_width=1):
    """Evaluate on a dataset split. Returns metrics and sample predictions.

    Args:
        beam_width: If > 1, use beam search for token prediction. Grammar
                    constraints are applied automatically if the decoder has them.
    """
    decoder.eval()
    total_loss = 0.0
    loss_accum = defaultdict(float)
    n_batches = 0

    all_preds = []
    all_targets = []
    all_type_preds = []
    all_type_targets = []
    all_cmd_preds = []
    all_cmd_targets = []
    all_pt_preds = []
    all_pt_targets = []
    token_correct = 0
    token_total = 0
    seq_match = 0
    seq_total = 0
    num_correct = 0
    num_total = 0

    for batch in loader:
        input_tokens = batch['input_tokens'].to(device)
        target_tokens = batch['target_tokens'].to(device)
        memory = batch['memory'].to(device)
        op_pred = batch['op_pred'].to(device)
        padding_mask = (input_tokens == PAD)

        # V6: pass extra fields
        extra_kwargs = {}
        if 'window_index' in batch:
            extra_kwargs['window_index'] = batch['window_index'].to(device)
        if 'total_windows' in batch:
            extra_kwargs['total_windows'] = batch['total_windows'].to(device)
        if 'sequence_class' in batch:
            extra_kwargs['sequence_class'] = batch['sequence_class'].to(device)

        outputs = decoder(
            tokens=input_tokens,
            sensor_embeddings=memory,
            operation_type=op_pred,
            tgt_key_padding_mask=padding_mask,
            **extra_kwargs,
        )

        # Legacy logits
        logits = outputs.get('legacy_logits', outputs.get('raw_legacy_logits'))
        if logits is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1),
                ignore_index=PAD,
            )
            total_loss += loss.item()
            n_batches += 1

            # Predictions: teacher-forced (default) or autoregressive beam/greedy
            if beam_width >= 1 and beam_width != 0:
                # beam_width >= 1: autoregressive decoding (beam=1 is greedy AR)
                # beam_width == 0: teacher-forced (legacy behavior, use logits.argmax)
                beam_preds = beam_search_decode(
                    decoder, memory, op_pred, device,
                    beam_width=max(beam_width, 1), max_len=target_tokens.size(1),
                    extra_kwargs=extra_kwargs,
                )
                # Beam search returns BOS-prefixed sequences; shift to match targets
                # targets are shifted left by 1 relative to inputs
                preds = beam_preds[:, 1:]  # drop BOS
                # Pad if needed
                if preds.size(1) < target_tokens.size(1):
                    preds = torch.nn.functional.pad(preds, (0, target_tokens.size(1) - preds.size(1)), value=PAD)
                preds = preds[:, :target_tokens.size(1)]
            else:
                # beam_width == 0: teacher-forced evaluation (legacy)
                preds = logits.argmax(-1)

            # Token accuracy
            mask = target_tokens != PAD
            token_correct += ((preds == target_tokens) & mask).sum().item()
            token_total += mask.sum().item()

            # Sequence match
            seq_match += ((preds == target_tokens) | ~mask).all(dim=1).sum().item()
            seq_total += target_tokens.size(0)

            # Numeric accuracy
            type_targets = batch['type_targets'].to(device)
            numeric_mask = (type_targets == TYPE_NUMERIC) & mask
            num_correct += ((preds == target_tokens) & numeric_mask).sum().item()
            num_total += numeric_mask.sum().item()

            all_preds.append(preds.cpu())
            all_targets.append(target_tokens.cpu())

        # Type accuracy
        type_targets_eval = batch['type_targets'].to(device)
        type_preds = outputs['type_logits'].argmax(-1)
        all_type_preds.append(type_preds.cpu())
        all_type_targets.append(type_targets_eval.cpu())

        # Command accuracy
        cmd_targets = batch['command_targets'].to(device)
        cmd_preds = outputs['command_logits'].argmax(-1)
        all_cmd_preds.append(cmd_preds.cpu())
        all_cmd_targets.append(cmd_targets.cpu())

        # Param type accuracy
        pt_targets = batch['param_type_targets'].to(device)
        pt_preds = outputs['param_type_logits'].argmax(-1)
        all_pt_preds.append(pt_preds.cpu())
        all_pt_targets.append(pt_targets.cpu())

    # Compute per-head accuracy
    type_p = torch.cat(all_type_preds)
    type_t = torch.cat(all_type_targets)
    type_mask = type_t >= 0
    type_acc = ((type_p == type_t) & type_mask).sum().item() / max(type_mask.sum().item(), 1)

    cmd_p = torch.cat(all_cmd_preds)
    cmd_t = torch.cat(all_cmd_targets)
    cmd_mask = cmd_t >= 0
    cmd_acc = ((cmd_p == cmd_t) & cmd_mask).sum().item() / max(cmd_mask.sum().item(), 1)

    pt_p = torch.cat(all_pt_preds)
    pt_t = torch.cat(all_pt_targets)
    pt_mask = pt_t >= 0
    pt_acc = ((pt_p == pt_t) & pt_mask).sum().item() / max(pt_mask.sum().item(), 1)

    token_acc = token_correct / max(token_total, 1)
    seq_acc = seq_match / max(seq_total, 1)
    num_acc = num_correct / max(num_total, 1)
    avg_loss = total_loss / max(n_batches, 1)

    # Sample predictions (all)
    samples = []
    if all_preds:
        preds_all = torch.cat(all_preds)
        targets_all = torch.cat(all_targets)
        for i in range(len(preds_all)):
            mask = targets_all[i] != PAD
            pred_ids = preds_all[i][mask].tolist()
            true_ids = targets_all[i][mask].tolist()
            pred_toks = tokenizer.decode(pred_ids)
            true_toks = tokenizer.decode(true_ids)
            samples.append({
                'true': ' '.join(true_toks),
                'pred': ' '.join(pred_toks),
                'match': pred_ids == true_ids,
            })

    metrics = {
        'loss': avg_loss,
        'token_accuracy': token_acc,
        'sequence_accuracy': seq_acc,
        'type_accuracy': type_acc,
        'command_accuracy': cmd_acc,
        'param_type_accuracy': pt_acc,
        'numeric_accuracy': num_acc,
    }
    return metrics, samples


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SensorMultiHeadDecoder on frozen 20260225 encoder")
    # Data/model paths
    parser.add_argument("--data_dir", type=str, default=None, help="Preprocessed data dir")
    parser.add_argument("--encoder_ckpt", type=str, default=None, help="Encoder checkpoint path")
    parser.add_argument("--encoder_config", type=str, default=None,
                        choices=list(ENCODER_CONFIGS.keys()) + list(V7_ENCODER_CONFIGS.keys()),
                        help="Encoder config name (overrides data_dir/encoder_ckpt)")
    parser.add_argument("--fold", type=int, default=1, help="Fold number (used with --encoder_config)")
    parser.add_argument("--vocab", type=str, required=True, help="Vocab JSON path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_token_len", type=int, default=16, help="Max token sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0=disabled)")
    parser.add_argument("--curriculum", type=str, default="none", choices=["none", "3phase"],
                        help="Curriculum learning mode")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")

    # Loss / training strategy
    parser.add_argument("--digit_weight", type=float, default=2.0, help="Digit loss weight")
    parser.add_argument("--legacy_weight", type=float, default=1.0, help="Legacy token loss weight")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing for legacy CE")
    parser.add_argument("--scheduled_sampling", type=float, default=0.0,
                        help="Max scheduled sampling ratio (0=pure teacher forcing)")
    parser.add_argument("--focal_gamma", type=float, default=0.0,
                        help="Focal loss gamma (0=standard CE)")

    # Decoder architecture
    parser.add_argument("--d_model", type=int, default=192, help="Decoder hidden dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Decoder transformer layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--hierarchical", type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                        default=False, help="Enable hierarchical head conditioning")
    parser.add_argument("--memory_pos_encoding", type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                        default=False, help="Enable memory positional encoding")

    # V5 features
    parser.add_argument("--grammar_constraint", type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                        default=False, help="Enable grammar-constrained decoding")
    parser.add_argument("--use_regression_head", type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                        default=False, help="Enable numeric regression head")
    parser.add_argument("--regression_weight", type=float, default=1.0,
                        help="Weight for regression loss")
    parser.add_argument("--beam_width", type=int, default=1,
                        help="Beam search width for evaluation (1=greedy)")
    parser.add_argument("--multi_window_context", type=int, default=0,
                        help="Number of adjacent windows to include as context (0=disabled)")

    # Optimizer
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Linear warmup epochs")

    # V6 features
    parser.add_argument("--use_window_position", type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                        default=False, help="Enable window position input")
    parser.add_argument("--use_sequence_classifier", type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                        default=False, help="Enable two-stage sequence classifier")
    parser.add_argument("--sequence_class_weight", type=float, default=1.0,
                        help="Weight for sequence classification loss")
    parser.add_argument("--oversample_rare", type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                        default=False, help="Oversample rare sequences during training")
    parser.add_argument("--noise_scale", type=float, default=0.0,
                        help="Gaussian noise scale for encoder memory augmentation")
    parser.add_argument("--window_dropout", type=float, default=0.0,
                        help="Dropout rate for MWC context windows during training")

    # V7 features
    parser.add_argument("--use_pointer_network", type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                        default=False, help="Enable pointer network for per-axis numeric classification")
    parser.add_argument("--pointer_weight", type=float, default=2.0,
                        help="Weight for pointer network loss")
    parser.add_argument("--axis_value_tables", type=str, default="data/axis_value_tables.json",
                        help="Path to axis value tables JSON for pointer network")
    parser.add_argument("--use_sensor_prior", type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                        default=False, help="Enable SensorValuePrior bypass path")
    parser.add_argument("--drop_path_rate", type=float, default=0.0,
                        help="Stochastic depth / DropPath rate")
    parser.add_argument("--use_distillation", type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                        default=False, help="Enable self-distillation from teacher model")
    parser.add_argument("--distillation_alpha", type=float, default=0.3,
                        help="Weight for distillation KD loss (1-alpha for hard loss)")
    parser.add_argument("--distillation_temp", type=float, default=2.0,
                        help="Temperature for distillation softmax")
    parser.add_argument("--teacher_checkpoint", type=str, default=None,
                        help="Path to teacher decoder checkpoint for distillation")
    parser.add_argument("--cross_window_attention", type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                        default=False, help="Enable cross-window self-attention for MWC")

    # Eval-only mode
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, load checkpoint and evaluate with beam search")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to decoder checkpoint to load for eval_only mode")

    # Encoder fine-tuning
    parser.add_argument("--finetune_encoder", action="store_true",
                        help="Unfreeze encoder and fine-tune end-to-end with decoder")
    parser.add_argument("--encoder_lr", type=float, default=3e-5,
                        help="Learning rate for encoder parameters (when fine-tuning)")
    parser.add_argument("--finetune_after_epoch", type=int, default=0,
                        help="Freeze encoder for this many epochs before unfreezing")
    parser.add_argument("--finetune_layers", type=str, default="lstm_only",
                        choices=["all", "lstm_only", "last_layer"],
                        help="Which encoder layers to fine-tune")
    parser.add_argument("--recache_interval", type=int, default=50,
                        help="Re-cache encoder memory every N epochs during fine-tuning")

    # Approach A: end-to-end training
    parser.add_argument("--e2e", action="store_true",
                        help="End-to-end training: decoder loss backprops through encoder")
    parser.add_argument("--grad_accum", type=int, default=2,
                        help="Gradient accumulation steps for e2e (effective batch = batch_size * grad_accum)")
    parser.add_argument("--e2e_recache_interval", type=int, default=20,
                        help="Re-cache val/test memory every N epochs during e2e training")
    parser.add_argument("--e2e_cls_weight", type=float, default=0.1,
                        help="Auxiliary classification loss weight to prevent encoder forgetting")

    # Wandb
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="decoder-sweep", help="Wandb project name")

    args = parser.parse_args()

    # ── Resolve encoder config ──
    if args.encoder_config is not None:
        if args.encoder_config in V7_ENCODER_CONFIGS:
            # V7 config: new preprocessed data, old encoder checkpoint
            v7cfg = V7_ENCODER_CONFIGS[args.encoder_config]
            args.data_dir = str(v7cfg['data_dir'] / f"fold_{args.fold}")
            args.encoder_ckpt = str(ENCODER_BASE / v7cfg['encoder_dir'] / f"fold_{args.fold}" / "encoder" / "checkpoint" / "best_model.pt")
        elif args.encoder_config in ENCODER_CONFIGS:
            config_dir = ENCODER_CONFIGS[args.encoder_config]
            args.data_dir = str(ENCODER_BASE / config_dir / f"fold_{args.fold}" / "preprocessed")
            args.encoder_ckpt = str(ENCODER_BASE / config_dir / f"fold_{args.fold}" / "encoder" / "checkpoint" / "best_model.pt")
        else:
            parser.error(f"Unknown encoder config: {args.encoder_config}")
    elif args.data_dir is None or args.encoder_ckpt is None:
        parser.error("Either --encoder_config or both --data_dir and --encoder_ckpt are required")

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ['encoder_memory', 'decoder_checkpoint', 'results', 'figures']:
        (output_dir / sub).mkdir(exist_ok=True)

    log_path = output_dir / 'results' / 'training_log.txt'
    log_fh = open(log_path, 'w')

    def lprint(msg):
        log(msg, log_fh)

    # ── Wandb init ──
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))
        # Append wandb run ID to output_dir for unique per-run output
        output_dir = output_dir / wandb.run.id
        output_dir.mkdir(parents=True, exist_ok=True)
        for sub in ['encoder_memory', 'decoder_checkpoint', 'results', 'figures']:
            (output_dir / sub).mkdir(exist_ok=True)
        log_path = output_dir / 'results' / 'training_log.txt'
        log_fh = open(log_path, 'w')

    lprint(f"=" * 70)
    lprint(f"SensorMultiHeadDecoder Training")
    lprint(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lprint(f"Device: {device}")
    if args.encoder_config:
        lprint(f"Encoder config: {args.encoder_config}")
    lprint(f"=" * 70)

    # ── 1. Load tokenizer ──
    lprint(f"\n[1/7] Loading vocabulary from {args.vocab}")
    tokenizer = GCodeTokenizer.load(Path(args.vocab))
    vocab_size = len(tokenizer.vocab)
    lprint(f"  Vocab size: {vocab_size}")

    # ── 2. Load datasets with re-tokenization ──
    lprint(f"\n[2/7] Loading datasets and re-tokenizing")

    # Build sequence class mapping (V6: two-stage classifier)
    sequence_class_map = None
    if args.use_sequence_classifier:
        all_texts = set()
        for split in ['train', 'val', 'test']:
            npz_path = data_dir / f'{split}_sequences.npz'
            d = np.load(npz_path, allow_pickle=True)
            all_texts.update(str(t) for t in d['gcode_texts'])
        sequence_class_map = {t: i for i, t in enumerate(sorted(all_texts))}
        lprint(f"  Sequence classifier: {len(sequence_class_map)} unique sequences")

    datasets = {}
    for split in ['train', 'val', 'test']:
        npz_path = data_dir / f'{split}_sequences.npz'
        lprint(f"  Loading {split}: {npz_path}")
        ds = DecoderQuickTestDataset(
            npz_path, tokenizer, max_token_len=args.max_token_len,
            sequence_class_map=sequence_class_map,
        )
        datasets[split] = ds
        lprint(f"    Samples: {ds.stats['n_samples']}, Tokens: {ds.stats['total_tokens']}, "
               f"UNK: {ds.stats['unk_count']} ({ds.stats['unk_rate']:.4%}), "
               f"Avg len: {ds.stats['mean_token_len']:.1f}, Max len: {ds.stats['max_token_len']}")
        if ds.stats['unk_count'] > 0:
            lprint(f"    WARNING: {ds.stats['unk_count']} UNK tokens in {split} ({ds.stats['unk_rate']:.2%})")

    # ── 3. Load frozen encoder ──
    lprint(f"\n[3/7] Loading encoder from {args.encoder_ckpt}")
    encoder, enc_config, ckpt = load_frozen_encoder(args.encoder_ckpt, device)
    lprint(f"  Config: sensor_dims={list(enc_config.sensor_dims)}, d_model={enc_config.d_model}")
    lprint(f"  Checkpoint epoch: {ckpt.get('epoch')}, best_val_acc: {ckpt.get('best_val_acc', 'N/A')}")
    lprint(f"  Total encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    if args.e2e:
        # Unfreeze encoder for end-to-end training
        for p in encoder.parameters():
            p.requires_grad_(True)
        n_enc_params = sum(p.numel() for p in encoder.parameters())
        lprint(f"  ** E2E mode: encoder unfrozen ({n_enc_params:,} params), encoder_lr={args.encoder_lr}")
        lprint(f"     grad_accum={args.grad_accum}, e2e_cls_weight={args.e2e_cls_weight}")
        lprint(f"     Val/test recache every {args.e2e_recache_interval} epochs")
    elif args.finetune_encoder:
        lprint(f"  ** Fine-tuning enabled: layers={args.finetune_layers}, encoder_lr={args.encoder_lr}")
        lprint(f"     Unfreeze after epoch {args.finetune_after_epoch}, recache every {args.recache_interval} epochs")

    # ── 4. Build group indices from metadata ──
    lprint(f"\n[4/7] Building modality group indices from metadata")
    metadata_path = data_dir / 'metadata.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
    columns = metadata['continuous_columns']
    group_names, group_indices, sensor_dims = build_modality_indices(columns)
    lprint(f"  Groups: {dict(zip(group_names, sensor_dims))}")
    assert sensor_dims == list(enc_config.sensor_dims), \
        f"Sensor dims mismatch: {sensor_dims} vs {list(enc_config.sensor_dims)}"
    lprint(f"  Sensor dims verified: {sensor_dims}")

    # ── 5. Cache encoder memory ──
    if args.e2e:
        lprint(f"\n[5/7] E2E mode: caching val/test only (train uses live encoder)")
        cached_datasets = {}
        encoder.eval()  # eval mode for caching
        for split in ['val', 'test']:
            lprint(f"  Caching {split}...")
            t0 = time.time()
            memory, op_pred, cls_acc = cache_encoder_memory(
                encoder, datasets[split], group_indices, device,
                batch_size=args.batch_size,
            )
            cached_datasets[split] = CachedDecoderDataset(
                datasets[split], memory, op_pred,
                multi_window_context=0,  # no MWC for e2e initially
            )
            lprint(f"    {split}: memory shape={memory.shape}, cls_acc={cls_acc:.4f} ({time.time()-t0:.1f}s)")
        # Train uses raw dataset directly
        cached_datasets['train'] = None  # placeholder, train_loader uses datasets['train']
    else:
        lprint(f"\n[5/7] Caching encoder memory for all splits")
        cached_datasets = {}
        for split in ['train', 'val', 'test']:
            lprint(f"  Caching {split}...")
            t0 = time.time()
            cache_path = output_dir / 'encoder_memory'
            memory, op_pred, cls_acc = cache_encoder_memory(
                encoder, datasets[split], group_indices, device,
                batch_size=args.batch_size,
                cache_dir=cache_path if split == 'train' else None,
            )
            torch.save(memory, output_dir / 'encoder_memory' / f'{split}_memory.pt')
            torch.save(op_pred, output_dir / 'encoder_memory' / f'{split}_op_pred.pt')
            is_train = (split == 'train')
            cached_datasets[split] = CachedDecoderDataset(
                datasets[split], memory, op_pred,
                multi_window_context=args.multi_window_context,
                noise_scale=args.noise_scale if is_train else 0.0,
                window_dropout=args.window_dropout if is_train else 0.0,
                training=is_train,
            )
            lprint(f"    {split}: memory shape={memory.shape}, cls_acc={cls_acc:.4f} ({time.time()-t0:.1f}s)")

    # ── 6. Create decoder ──
    lprint(f"\n[6/7] Creating SensorMultiHeadDecoder")
    lprint(f"  d_model={args.d_model}, n_layers={args.n_layers}, n_heads={args.n_heads}, dropout={args.dropout}")
    lprint(f"  hierarchical={args.hierarchical}, memory_pos_encoding={args.memory_pos_encoding}")
    lprint(f"  grammar_constraint={args.grammar_constraint}, regression_head={args.use_regression_head}")
    lprint(f"  beam_width={args.beam_width}, multi_window_context={args.multi_window_context}")
    lprint(f"  V6: window_pos={args.use_window_position}, seq_classifier={args.use_sequence_classifier}")
    lprint(f"  V6: oversample={args.oversample_rare}, noise={args.noise_scale}, win_dropout={args.window_dropout}")
    lprint(f"  V7: pointer_net={args.use_pointer_network}, sensor_prior={args.use_sensor_prior}, "
           f"drop_path={args.drop_path_rate}, distill={args.use_distillation}")

    # Load axis value tables for pointer network
    axis_value_tables = None
    if args.use_pointer_network:
        avt_path = Path(args.axis_value_tables)
        if avt_path.exists():
            with open(avt_path) as f:
                axis_value_tables = json.load(f)
            lprint(f"  Pointer network: loaded {len(axis_value_tables)} axes from {avt_path}")
        else:
            lprint(f"  WARNING: axis_value_tables not found at {avt_path}, disabling pointer network")
            args.use_pointer_network = False

    n_unique_seqs = len(sequence_class_map) if sequence_class_map else 34
    decoder = SensorMultiHeadDecoder(
        vocab_size=vocab_size,
        d_model=args.d_model,
        sensor_dim=enc_config.d_model,  # 256 from encoder
        n_operations=9,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_int_digits=2,
        n_decimal_digits=4,
        dropout=args.dropout,
        max_seq_len=args.max_token_len,
        hierarchical=args.hierarchical,
        memory_pos_encoding=args.memory_pos_encoding,
        use_regression_head=args.use_regression_head,
        use_window_position=args.use_window_position,
        max_windows_per_file=int(datasets['train'].total_windows.max().item()) + 4,  # dynamic: actual max + margin
        use_sequence_classifier=args.use_sequence_classifier,
        n_unique_sequences=n_unique_seqs,
        use_pointer_network=args.use_pointer_network,
        axis_value_tables=axis_value_tables,
        use_sensor_prior=args.use_sensor_prior,
        drop_path_rate=args.drop_path_rate,
    )
    decoder.use_grammar_constraint = args.grammar_constraint
    decoder.set_vocab(tokenizer.vocab)
    decoder = decoder.to(device)
    n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    lprint(f"  Decoder params: {n_params:,}")

    # ── Loss functions ──
    loss_fns = {
        'type': nn.CrossEntropyLoss(ignore_index=-1),
        'command': nn.CrossEntropyLoss(ignore_index=-1),
        'param_type': nn.CrossEntropyLoss(ignore_index=-1),
        'digit': DigitByDigitLoss(n_digit_positions=6, aux_loss_weight=0.1),
    }
    # Legacy loss: focal or standard CE with optional label smoothing
    if args.focal_gamma > 0:
        loss_fns['legacy'] = FocalLoss(
            gamma=args.focal_gamma, ignore_index=PAD, label_smoothing=args.label_smoothing
        )
        lprint(f"  Legacy loss: FocalLoss(gamma={args.focal_gamma}, label_smoothing={args.label_smoothing})")
    else:
        loss_fns['legacy'] = nn.CrossEntropyLoss(
            ignore_index=PAD, label_smoothing=args.label_smoothing
        )
        if args.label_smoothing > 0:
            lprint(f"  Legacy loss: CE(label_smoothing={args.label_smoothing})")

    loss_weights = {
        'type': 1.0, 'command': 1.0, 'param_type': 1.0,
        'digit': args.digit_weight, 'legacy': args.legacy_weight,
    }
    if args.use_regression_head:
        loss_fns['regression'] = nn.MSELoss()  # placeholder, actual loss computed inline
        loss_weights['regression'] = args.regression_weight
        lprint(f"  Regression head enabled, weight={args.regression_weight}")
    if args.use_sequence_classifier:
        loss_fns['sequence'] = nn.CrossEntropyLoss()
        loss_weights['sequence'] = args.sequence_class_weight
        lprint(f"  Sequence classifier enabled, weight={args.sequence_class_weight}")

    # V7: Pointer network loss
    pointer_target_map = None
    if args.use_pointer_network and axis_value_tables is not None:
        # Build target mapping: vocab_token_id -> (axis_name, axis_index)
        pointer_target_map = {}
        for axis_name, axis_info in axis_value_tables.items():
            n_values = axis_info.get('n_values', 0)
            if n_values < 2:
                continue
            token_ids = axis_info.get('token_ids', [])
            for axis_idx, tok_id in enumerate(token_ids):
                pointer_target_map[tok_id] = (axis_name, axis_idx)
        loss_fns['pointer'] = nn.CrossEntropyLoss(ignore_index=-1)
        loss_weights['pointer'] = args.pointer_weight
        lprint(f"  Pointer network enabled: {len(pointer_target_map)} numeric tokens mapped, weight={args.pointer_weight}")

    lprint(f"  Loss weights: {loss_weights}")

    # ── Optimizer + scheduler ──
    if args.e2e:
        # Two param groups: decoder (higher LR) + encoder (lower LR)
        optimizer = torch.optim.AdamW([
            {'params': decoder.parameters(), 'lr': args.lr},
            {'params': encoder.parameters(), 'lr': args.encoder_lr},
        ], weight_decay=args.weight_decay)
        lprint(f"  E2E optimizer: decoder_lr={args.lr}, encoder_lr={args.encoder_lr}")
    elif args.finetune_encoder:
        # Two param groups: decoder (higher LR) + encoder (lower LR, added when unfrozen)
        optimizer = torch.optim.AdamW([
            {'params': decoder.parameters(), 'lr': args.lr},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_epochs])

    # ── Data loaders ──
    if args.e2e:
        # E2E: train uses raw dataset (live encoder forward), val/test use cached
        e2e_batch_size = args.batch_size // args.grad_accum if args.batch_size > args.grad_accum else args.batch_size
        train_loader = DataLoader(datasets['train'], batch_size=e2e_batch_size, shuffle=True,
                                  collate_fn=decoder_collate_fn, num_workers=0, drop_last=True)
        lprint(f"  E2E train loader: batch_size={e2e_batch_size}, grad_accum={args.grad_accum}, "
               f"effective_batch={e2e_batch_size * args.grad_accum}")
    elif args.oversample_rare:
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler
        train_texts = datasets['train'].gcode_texts_list
        seq_counts = Counter(train_texts)
        sample_weights = [1.0 / seq_counts[t] for t in train_texts]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(cached_datasets['train'], batch_size=args.batch_size, sampler=sampler,
                                  collate_fn=decoder_collate_fn, num_workers=0, drop_last=True)
        lprint(f"  Oversampling: {len(seq_counts)} unique sequences, min_count={min(seq_counts.values())}")
    else:
        train_loader = DataLoader(cached_datasets['train'], batch_size=args.batch_size, shuffle=True,
                                  collate_fn=decoder_collate_fn, num_workers=0, drop_last=True)
    val_loader = DataLoader(cached_datasets['val'], batch_size=args.batch_size, shuffle=False,
                            collate_fn=decoder_collate_fn, num_workers=0)
    test_loader = DataLoader(cached_datasets['test'], batch_size=args.batch_size, shuffle=False,
                             collate_fn=decoder_collate_fn, num_workers=0)

    # ── Eval-only mode ──
    if args.eval_only:
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            # Default: look for best_decoder.pt in the output_dir
            ckpt_path = str(output_dir / 'decoder_checkpoint' / 'best_decoder.pt')
        lprint(f"\n[EVAL-ONLY] Loading checkpoint: {ckpt_path}")
        ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt_data['decoder_state_dict']
        # Filter out size-mismatched keys (e.g., window_pos_embed with different max_windows)
        model_state = decoder.state_dict()
        filtered_state = {}
        size_mismatched = []
        for k, v in state_dict.items():
            if k in model_state and v.shape != model_state[k].shape:
                size_mismatched.append(k)
            else:
                filtered_state[k] = v
        missing, unexpected = decoder.load_state_dict(filtered_state, strict=False)
        if size_mismatched:
            lprint(f"  Size-mismatched keys (handled manually): {size_mismatched}")
        if missing:
            lprint(f"  Missing keys (random init): {list(set(k.split('.')[0] for k in missing if k not in [sm.split('.')[0] for sm in size_mismatched]))[:5]}")
        if unexpected:
            lprint(f"  Unexpected keys (ignored): {list(set(k.split('.')[0] for k in unexpected))[:5]}")
        # Copy window_pos_embed rows if size changed (dynamic max_windows_per_file)
        for key in size_mismatched:
            if key in state_dict and key in model_state:
                ckpt_w = state_dict[key]
                cur_w = model_state[key]
                n_copy = min(ckpt_w.shape[0], cur_w.shape[0])
                with torch.no_grad():
                    getattr_nested = key.split('.')
                    obj = decoder
                    for attr in getattr_nested[:-1]:
                        obj = getattr(obj, attr)
                    param = getattr(obj, getattr_nested[-1])
                    param[:n_copy] = ckpt_w[:n_copy].to(param.device)
                lprint(f"  Copied {key}: {ckpt_w.shape} -> {cur_w.shape} ({n_copy} rows)")
        lprint(f"  Loaded from epoch {ckpt_data.get('epoch', '?')}")

        for bw in ([args.beam_width] if args.beam_width > 1 else [0, 1, 3, 5]):
            bw_label = {0: "teacher_forced", 1: "greedy_AR"}.get(bw, f"beam_{bw}")
            lprint(f"\n  Evaluating with {bw_label} (beam_width={bw})...")
            test_metrics, test_samples = evaluate(decoder, test_loader, device, loss_fns, tokenizer,
                                                  beam_width=bw)
            lprint(f"\n  Test Results ({bw_label}):")
            for k, v in test_metrics.items():
                lprint(f"    {k}: {v:.4f}")

            lprint(f"\n  Sample Generations ({bw_label}):")
            for i, s in enumerate(test_samples[:10]):
                match_str = "MATCH" if s['match'] else "MISS"
                lprint(f"    [{i+1:2d}] [{match_str}]")
                lprint(f"         TRUE: {s['true']}")
                lprint(f"         PRED: {s['pred']}")

            # Save results
            beam_results = {
                'beam_width': bw,
                'checkpoint': ckpt_path,
                'test_metrics': test_metrics,
            }
            with open(output_dir / 'results' / f'beam_{bw}_metrics.json', 'w') as f:
                json.dump(beam_results, f, indent=2)

            # Save ALL predictions for per-class analysis
            all_preds_path = output_dir / 'results' / f'beam_{bw}_all_predictions.json'
            with open(all_preds_path, 'w') as f:
                json.dump(test_samples, f, indent=2)
            lprint(f"  Saved {len(test_samples)} predictions to {all_preds_path}")

        lprint(f"\nAll outputs saved to {output_dir}")
        lprint(f"Done!")
        log_fh.close()
        if args.wandb:
            wandb.finish()
        return

    # ── 7. Training loop ──
    lprint(f"\n[7/7] Training decoder for {args.epochs} epochs")
    lprint(f"  LR: {args.lr}, batch_size: {args.batch_size}, weight_decay: {args.weight_decay}")
    lprint(f"  Curriculum: {args.curriculum}")
    if args.scheduled_sampling > 0:
        lprint(f"  Scheduled sampling: max_ratio={args.scheduled_sampling}")
    if args.curriculum == '3phase':
        lprint(f"    Phase 1 (epochs 1-30): structure only (type+command+param_type)")
        lprint(f"    Phase 2 (epochs 31-80): + digit loss")
        lprint(f"    Phase 3 (epochs 81+): + legacy loss (all heads)")
    if args.patience > 0:
        lprint(f"  Early stopping: patience={args.patience}")

    history = defaultdict(list)
    best_val_token_acc = -1.0
    best_epoch = 1
    patience_counter = 0
    encoder_unfrozen = False

    def _unfreeze_encoder_layers():
        """Selectively unfreeze encoder layers based on --finetune_layers."""
        nonlocal encoder_unfrozen
        if encoder_unfrozen:
            return
        encoder.train()
        if args.finetune_layers == 'all':
            for p in encoder.parameters():
                p.requires_grad_(True)
            n_unfrozen = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        elif args.finetune_layers == 'lstm_only':
            for name, p in encoder.named_parameters():
                if 'temporal' in name or 'lstm' in name:
                    p.requires_grad_(True)
            n_unfrozen = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        elif args.finetune_layers == 'last_layer':
            # Unfreeze last LSTM layer + layer norm + classification head
            for name, p in encoder.named_parameters():
                if any(k in name for k in ['temporal.weight_hh_l1', 'temporal.weight_ih_l1',
                                            'temporal.bias_hh_l1', 'temporal.bias_ih_l1',
                                            'lstm.weight_hh_l1', 'lstm.weight_ih_l1',
                                            'lstm.bias_hh_l1', 'lstm.bias_ih_l1',
                                            'layer_norm', 'norm', 'head_cls']):
                    p.requires_grad_(True)
            n_unfrozen = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        # Add encoder params to optimizer
        encoder_params = [p for p in encoder.parameters() if p.requires_grad]
        optimizer.add_param_group({'params': encoder_params, 'lr': args.encoder_lr})
        encoder_unfrozen = True
        lprint(f"  ** Encoder unfrozen: {n_unfrozen:,} params, lr={args.encoder_lr} **")

    def _finetune_encoder_one_round():
        """Fine-tune encoder for 1 epoch on classification loss, then re-cache (Approach B)."""
        encoder.train()
        enc_optimizer = torch.optim.AdamW(
            [p for p in encoder.parameters() if p.requires_grad],
            lr=args.encoder_lr, weight_decay=0.01
        )
        cls_loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True,
                            collate_fn=decoder_collate_fn, num_workers=0, drop_last=True)
        total_cls_loss = 0.0
        n_b = 0
        for batch in loader:
            sensor_data = batch['sensor_features'].to(device)
            gt_ops = batch['operation_type'].to(device)
            B, T = sensor_data.shape[:2]
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
            mods = [sensor_data[:, :, idx] for idx in group_indices]
            out = encoder(mods, lengths)
            cls_logits = out['cls']  # [B, 9]
            loss = cls_loss_fn(cls_logits, gt_ops)
            # Add reconstruction loss if available
            if 'recon' in out and 'fused_target' in out:
                recon_loss = F.mse_loss(out['recon'], out['fused_target'].detach())
                loss = loss + 0.1 * recon_loss
            enc_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in encoder.parameters() if p.requires_grad], 0.5)
            enc_optimizer.step()
            total_cls_loss += loss.item()
            n_b += 1
        avg_cls_loss = total_cls_loss / max(n_b, 1)
        lprint(f"    Encoder fine-tune round: avg_loss={avg_cls_loss:.4f}")

    def _recache_all_splits():
        """Re-cache encoder memory after encoder updates (Approach B)."""
        nonlocal cached_datasets, train_loader, val_loader, test_loader
        lprint(f"  ** Re-caching encoder memory after fine-tuning update **")
        encoder.eval()
        new_cached = {}
        for split in ['train', 'val', 'test']:
            t0_rc = time.time()
            memory, op_pred_new, cls_acc = cache_encoder_memory(
                encoder, datasets[split], group_indices, device,
                batch_size=args.batch_size,
            )
            is_train = (split == 'train')
            new_cached[split] = CachedDecoderDataset(
                datasets[split], memory, op_pred_new,
                multi_window_context=args.multi_window_context,
                noise_scale=args.noise_scale if is_train else 0.0,
                window_dropout=args.window_dropout if is_train else 0.0,
                training=is_train,
            )
            lprint(f"    {split}: cls_acc={cls_acc:.4f} ({time.time()-t0_rc:.1f}s)")
        cached_datasets = new_cached
        # Rebuild data loaders
        if args.oversample_rare:
            train_loader = DataLoader(cached_datasets['train'], batch_size=args.batch_size, sampler=sampler,
                                      collate_fn=decoder_collate_fn, num_workers=0, drop_last=True)
        else:
            train_loader = DataLoader(cached_datasets['train'], batch_size=args.batch_size, shuffle=True,
                                      collate_fn=decoder_collate_fn, num_workers=0, drop_last=True)
        val_loader = DataLoader(cached_datasets['val'], batch_size=args.batch_size, shuffle=False,
                                collate_fn=decoder_collate_fn, num_workers=0)
        test_loader = DataLoader(cached_datasets['test'], batch_size=args.batch_size, shuffle=False,
                                 collate_fn=decoder_collate_fn, num_workers=0)

    prev_active = None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Fine-tuning: unfreeze encoder at the right epoch
        if args.finetune_encoder and epoch == args.finetune_after_epoch + 1:
            _unfreeze_encoder_layers()

        # Fine-tuning: re-cache encoder memory periodically
        if (args.finetune_encoder and encoder_unfrozen
                and args.recache_interval > 0
                and epoch > args.finetune_after_epoch + 1
                and (epoch - args.finetune_after_epoch - 1) % args.recache_interval == 0):
            _finetune_encoder_one_round()
            _recache_all_splits()

        # Log curriculum phase transitions
        if args.curriculum != 'none':
            active = get_active_losses(args.curriculum, epoch)
            if active != prev_active:
                lprint(f"  --- Curriculum phase change at epoch {epoch}: active losses = {sorted(active)} ---")
                prev_active = active

        # Train
        if args.e2e:
            train_loss, train_losses, train_token_acc, train_type_acc = train_epoch_e2e(
                encoder, decoder, train_loader, optimizer, device, loss_fns, loss_weights, epoch,
                group_indices=group_indices,
                curriculum=args.curriculum,
                scheduled_sampling=args.scheduled_sampling,
                total_epochs=args.epochs,
                pointer_target_map=pointer_target_map,
                grad_accum=args.grad_accum,
                cls_weight=args.e2e_cls_weight,
            )
        else:
            train_loss, train_losses, train_token_acc, train_type_acc = train_epoch(
                decoder, train_loader, optimizer, device, loss_fns, loss_weights, epoch,
                curriculum=args.curriculum,
                scheduled_sampling=args.scheduled_sampling,
                total_epochs=args.epochs,
                pointer_target_map=pointer_target_map,
            )
        scheduler.step()

        # E2E: re-cache val/test periodically as encoder changes
        if args.e2e and epoch % args.e2e_recache_interval == 0:
            encoder.eval()
            for split in ['val', 'test']:
                memory, op_pred_new, cls_acc = cache_encoder_memory(
                    encoder, datasets[split], group_indices, device,
                    batch_size=args.batch_size)
                cached_datasets[split] = CachedDecoderDataset(
                    datasets[split], memory, op_pred_new, multi_window_context=0)
            val_loader = DataLoader(cached_datasets['val'], batch_size=args.batch_size, shuffle=False,
                                    collate_fn=decoder_collate_fn, num_workers=0)
            test_loader = DataLoader(cached_datasets['test'], batch_size=args.batch_size, shuffle=False,
                                     collate_fn=decoder_collate_fn, num_workers=0)
            lprint(f"    [E2E] Re-cached val/test (val cls_acc={cls_acc:.4f})")

        # Validate (teacher-forced for speed during training)
        if args.e2e:
            encoder.eval()  # eval mode for validation
        val_metrics, _ = evaluate(decoder, val_loader, device, loss_fns, tokenizer, beam_width=0)

        # Log
        elapsed = time.time() - t0
        lprint(f"  Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | "
               f"Train loss={train_loss:.4f} tok_acc={train_token_acc:.4f} type_acc={train_type_acc:.4f} | "
               f"Val loss={val_metrics['loss']:.4f} tok_acc={val_metrics['token_accuracy']:.4f} "
               f"type={val_metrics['type_accuracy']:.4f} cmd={val_metrics['command_accuracy']:.4f} "
               f"param={val_metrics['param_type_accuracy']:.4f} seq={val_metrics['sequence_accuracy']:.4f} "
               f"num={val_metrics['numeric_accuracy']:.4f}")

        # Wandb logging
        if args.wandb:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/token_accuracy': train_token_acc,
                'train/type_accuracy': train_type_acc,
            }
            for k, v in train_losses.items():
                log_dict[f'train/{k}_loss'] = v
            for k, v in val_metrics.items():
                log_dict[f'val/{k}'] = v
            log_dict['lr'] = optimizer.param_groups[0]['lr']
            wandb.log(log_dict)

        # History
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_token_acc'].append(train_token_acc)
        history['train_type_acc'].append(train_type_acc)
        for k, v in train_losses.items():
            history[f'train_{k}_loss'].append(v)
        for k, v in val_metrics.items():
            history[f'val_{k}'].append(v)

        # Best checkpoint
        # For curriculum: only track best/patience after all losses are active
        all_losses_active = {'type', 'command', 'param_type', 'digit', 'legacy'}.issubset(
                             get_active_losses(args.curriculum, epoch))
        if val_metrics['token_accuracy'] > best_val_token_acc:
            best_val_token_acc = val_metrics['token_accuracy']
            best_epoch = epoch
            patience_counter = 0
            ckpt_dict = {
                'epoch': epoch,
                'decoder_state_dict': decoder.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args),
            }
            if args.e2e or (args.finetune_encoder and encoder_unfrozen):
                ckpt_dict['encoder_state_dict'] = encoder.state_dict()
            torch.save(ckpt_dict, output_dir / 'decoder_checkpoint' / 'best_decoder.pt')
            lprint(f"    ** New best val token_acc: {best_val_token_acc:.4f} **")
        elif all_losses_active:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                lprint(f"  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # Save final checkpoint
    torch.save({
        'epoch': epoch,
        'decoder_state_dict': decoder.state_dict(),
        'args': vars(args),
    }, output_dir / 'decoder_checkpoint' / 'final_decoder.pt')

    # ── Final evaluation on test set ──
    lprint(f"\n{'='*70}")
    lprint(f"Final Evaluation (best checkpoint from epoch {best_epoch})")
    lprint(f"{'='*70}")

    # Reload best
    best_ckpt = torch.load(output_dir / 'decoder_checkpoint' / 'best_decoder.pt',
                           map_location=device, weights_only=False)
    decoder.load_state_dict(best_ckpt['decoder_state_dict'])

    # If e2e or fine-tuning, reload encoder and re-cache test data
    if 'encoder_state_dict' in best_ckpt:
        encoder.load_state_dict(best_ckpt['encoder_state_dict'])
        lprint(f"  Reloaded {'e2e' if args.e2e else 'fine-tuned'} encoder from best checkpoint")
        if args.e2e:
            encoder.eval()
            for split in ['val', 'test']:
                memory, op_pred_new, cls_acc = cache_encoder_memory(
                    encoder, datasets[split], group_indices, device,
                    batch_size=args.batch_size)
                cached_datasets[split] = CachedDecoderDataset(
                    datasets[split], memory, op_pred_new, multi_window_context=0)
                lprint(f"    {split}: cls_acc={cls_acc:.4f}")
            val_loader = DataLoader(cached_datasets['val'], batch_size=args.batch_size, shuffle=False,
                                    collate_fn=decoder_collate_fn, num_workers=0)
            test_loader = DataLoader(cached_datasets['test'], batch_size=args.batch_size, shuffle=False,
                                     collate_fn=decoder_collate_fn, num_workers=0)
        else:
            _recache_all_splits()

    # Teacher-forced evaluation (standard metric for training/sweeps)
    test_metrics, test_samples = evaluate(decoder, test_loader, device, loss_fns, tokenizer,
                                          beam_width=0)

    lprint(f"\nTest Results:")
    for k, v in test_metrics.items():
        lprint(f"  {k}: {v:.4f}")

    lprint(f"\nSample Generations (test set):")
    for i, s in enumerate(test_samples):
        match_str = "MATCH" if s['match'] else "MISS"
        lprint(f"  [{i+1:2d}] [{match_str}]")
        lprint(f"       TRUE: {s['true']}")
        lprint(f"       PRED: {s['pred']}")

    # Also evaluate on val (teacher-forced)
    val_metrics_final, val_samples = evaluate(decoder, val_loader, device, loss_fns, tokenizer, beam_width=0)

    # Wandb log test metrics
    if args.wandb:
        for k, v in test_metrics.items():
            wandb.log({f'test/{k}': v})
        for k, v in val_metrics_final.items():
            wandb.log({f'val_final/{k}': v})

    # ── Save results ──
    results = {
        'best_epoch': best_epoch,
        'best_val_token_accuracy': best_val_token_acc,
        'test_metrics': test_metrics,
        'val_metrics': val_metrics_final,
        'encoder_ckpt': args.encoder_ckpt,
        'vocab_path': args.vocab,
        'vocab_size': vocab_size,
        'encoder_config': {
            'sensor_dims': list(enc_config.sensor_dims),
            'd_model': enc_config.d_model,
            'n_heads': enc_config.n_heads,
        },
        'decoder_config': {
            'vocab_size': vocab_size,
            'd_model': args.d_model,
            'sensor_dim': enc_config.d_model,
            'n_operations': 9,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
        },
        'training_config': vars(args),
        'loss_weights': loss_weights,
        'curriculum': args.curriculum,
    }

    with open(output_dir / 'results' / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(output_dir / 'results' / 'training_history.json', 'w') as f:
        json.dump(dict(history), f, indent=2)

    # Save sample generations
    with open(output_dir / 'results' / 'sample_generations.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Test Set Sample Generations\n")
        f.write("=" * 70 + "\n\n")
        for i, s in enumerate(test_samples):
            f.write(f"[{i+1:2d}] {'MATCH' if s['match'] else 'MISS'}\n")
            f.write(f"     TRUE: {s['true']}\n")
            f.write(f"     PRED: {s['pred']}\n\n")

    lprint(f"\nAll outputs saved to {output_dir}")
    lprint(f"Done!")
    log_fh.close()

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
