#!/usr/bin/env python3
"""
Compute per-(head, op_class) Shannon entropy (in bits) over the 6 prediction heads
of the decoder, restricted to each of the 9 operation classes in the training corpus.

Heads:
  - token:      full vocab token id (non-PAD positions)
  - type:       {SPECIAL, COMMAND, PARAMETER, NUMERIC} (non-PAD positions)
  - command:    command id (positions where type=COMMAND)
  - param_type: param type id (positions where type in {PARAMETER, NUMERIC})
  - sign:       0=positive, 1=negative (positions where type=NUMERIC; sign parsed from token string)
  - digit:      digit value 0-9 pooled across the 6 digit slots (positions where type=NUMERIC)

H = -sum(p * log2(p)) over non-zero p.
"""

import json
import math
from pathlib import Path

import numpy as np

REPO = Path('/home/seacuello/Documents/gcode_fingerprinting')
TRAIN_NPZ = REPO / 'outputs/decoder20260511/preprocessed_f98/full_window/fold_1/train_sequences.npz'
VOCAB_JSON = REPO / 'data/gcode_vocab_v8.json'
OUT_JSON = REPO / 'outputs/decoder20260511/audit/per_head_per_opclass_entropy.json'

# Token type ids (mirror src/miracle/dataset/target_utils.py)
TYPE_SPECIAL = 0
TYPE_COMMAND = 1
TYPE_PARAMETER = 2
TYPE_NUMERIC = 3

SPECIAL_TOKENS = {'PAD', 'BOS', 'EOS', 'UNK', 'MASK',
                  '<PAD>', '<BOS>', '<EOS>', '<UNK>', '<MASK>'}


def build_decomposition_tables(vocab_path: Path):
    """For every token id in the vocab, precompute:
       type_id, command_id, param_type_id, sign_id, digit_list (6 digits)."""
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']
    id2token = {idx: tok for tok, idx in vocab.items()}
    vocab_size = max(id2token) + 1

    # First pass: gather command and param tokens
    command_tokens = []
    param_tokens = []
    for tok in vocab:
        if tok in SPECIAL_TOKENS:
            continue
        if tok.startswith('G') and len(tok) > 1 and tok[1:].isdigit():
            command_tokens.append(tok)
        elif tok.startswith('M') and len(tok) > 1 and tok[1:].isdigit():
            command_tokens.append(tok)
        elif tok.startswith('NUM_'):
            continue
        else:
            if len(tok) <= 2:
                param_tokens.append(tok)
    command_tokens = sorted(command_tokens)
    param_tokens = sorted(param_tokens)
    command2id = {c: i for i, c in enumerate(command_tokens)}
    param2id = {p: i for i, p in enumerate(param_tokens)}

    cfg = vocab_data.get('config', {})
    bucket_digits = cfg.get('bucket_digits', 4)
    canonical_dec = cfg.get('canonical_decimal_places', 4)
    # The digit head predicts ±XX.XXXX -> 2 int digits + 4 dec digits = 6 slots
    MAX_INT = 2
    N_DEC = 4
    N_DIG = MAX_INT + N_DEC  # 6

    PAD_ID = 10  # digit pad value (per decompose_value_to_digits)

    type_arr = np.zeros(vocab_size, dtype=np.int16)
    cmd_arr = np.full(vocab_size, -1, dtype=np.int32)
    pt_arr = np.full(vocab_size, -1, dtype=np.int32)
    sign_arr = np.full(vocab_size, -1, dtype=np.int16)
    digit_arr = np.full((vocab_size, N_DIG), -1, dtype=np.int16)

    for tid, tok in id2token.items():
        if tok in SPECIAL_TOKENS:
            type_arr[tid] = TYPE_SPECIAL
            continue
        if tok in command2id:
            type_arr[tid] = TYPE_COMMAND
            cmd_arr[tid] = command2id[tok]
            continue
        if tok in param2id:
            type_arr[tid] = TYPE_PARAMETER
            pt_arr[tid] = param2id[tok]
            continue
        if tok.startswith('NUM_'):
            type_arr[tid] = TYPE_NUMERIC
            parts = tok.split('_')
            if len(parts) >= 3:
                ptype = parts[1]
                pval_str = parts[2]
                if ptype in param2id:
                    pt_arr[tid] = param2id[ptype]
                # parse the bucketed integer (signed); convert to ±XX.XXXX representation.
                try:
                    bucket = int(pval_str)
                except ValueError:
                    bucket = 0
                sign = 1 if bucket < 0 else 0  # 0=pos, 1=neg
                sign_arr[tid] = sign
                # Reconstruct value: bucket / 10**bucket_digits gives the canonical value
                # then decompose into 6 digits via ±XX.XXXX representation.
                # We follow decompose_value_to_digits(value).
                value = abs(bucket) / (10 ** bucket_digits)
                # Clip to representable range
                max_value = (10 ** MAX_INT) - (10 ** -N_DEC)
                if value > max_value:
                    value = max_value
                # Integer part digits, most significant first
                int_part = int(value)
                int_digits = []
                rem = int_part
                for _ in range(MAX_INT):
                    int_digits.append(rem % 10)
                    rem //= 10
                int_digits.reverse()
                # Decimal digits
                dec_part = value - int_part
                dec_digits = []
                for _ in range(N_DEC):
                    dec_part *= 10
                    d = int(dec_part)
                    if d > 9:
                        d = 9
                    dec_digits.append(d)
                    dec_part -= d
                all_digits = int_digits + dec_digits
                digit_arr[tid] = all_digits
            continue
        # Unknown token -> treat as special
        type_arr[tid] = TYPE_SPECIAL

    return {
        'type': type_arr,
        'command': cmd_arr,
        'param_type': pt_arr,
        'sign': sign_arr,
        'digit': digit_arr,
        'vocab_size': vocab_size,
        'n_commands': len(command_tokens),
        'n_param_types': len(param_tokens),
        'n_digits_per_token': N_DIG,
        'pad_token_id': vocab.get('PAD', 0),
    }


def shannon_entropy_bits(values: np.ndarray) -> float:
    """Compute Shannon entropy (bits) of the empirical distribution over `values`.
    PAD/invalid entries should be filtered before calling."""
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def main():
    print(f"Loading {TRAIN_NPZ}")
    z = np.load(TRAIN_NPZ, allow_pickle=True)
    tokens = z['tokens']  # [N, T]
    op_names = z['operation_type_names']  # [N]
    N, T = tokens.shape
    print(f"  N={N}, T={T}")

    print(f"Building decomposition tables from {VOCAB_JSON}")
    tables = build_decomposition_tables(VOCAB_JSON)
    type_arr = tables['type']
    cmd_arr = tables['command']
    pt_arr = tables['param_type']
    sign_arr = tables['sign']
    digit_arr = tables['digit']
    pad_id = tables['pad_token_id']
    print(f"  vocab_size={tables['vocab_size']}, n_commands={tables['n_commands']}, "
          f"n_param_types={tables['n_param_types']}")

    # Op classes (sorted deterministically)
    op_classes = sorted(np.unique(op_names).tolist())
    print(f"  op_classes ({len(op_classes)}): {op_classes}")

    heads = ['token', 'type', 'command', 'param_type', 'sign', 'digit']

    entropy_matrix = []  # 6 rows x 9 cols
    counts_matrix = []   # for diagnostic

    for head in heads:
        row_entropy = []
        row_counts = []
        for op in op_classes:
            mask = (op_names == op)
            sample_tokens = tokens[mask]  # [n_op, T]
            flat = sample_tokens.reshape(-1)  # [n_op * T]
            # Filter out padding
            non_pad = flat[flat != pad_id]

            if head == 'token':
                vals = non_pad
            elif head == 'type':
                vals = type_arr[non_pad]
            elif head == 'command':
                t_ids = type_arr[non_pad]
                sel = non_pad[t_ids == TYPE_COMMAND]
                vals = cmd_arr[sel]
                # all entries should be >=0 by construction
                vals = vals[vals >= 0]
            elif head == 'param_type':
                t_ids = type_arr[non_pad]
                sel = non_pad[(t_ids == TYPE_PARAMETER) | (t_ids == TYPE_NUMERIC)]
                vals = pt_arr[sel]
                vals = vals[vals >= 0]
            elif head == 'sign':
                t_ids = type_arr[non_pad]
                sel = non_pad[t_ids == TYPE_NUMERIC]
                vals = sign_arr[sel]
                vals = vals[vals >= 0]
            elif head == 'digit':
                t_ids = type_arr[non_pad]
                sel = non_pad[t_ids == TYPE_NUMERIC]
                # Pool digits across all 6 slots
                vals = digit_arr[sel].reshape(-1)
                vals = vals[vals >= 0]
            else:
                raise ValueError(head)

            H = shannon_entropy_bits(vals)
            row_entropy.append(H)
            row_counts.append(int(vals.size))
        entropy_matrix.append(row_entropy)
        counts_matrix.append(row_counts)
        print(f"  head={head:11s}  entropies={[f'{h:.3f}' for h in row_entropy]}")

    out = {
        'op_classes': op_classes,
        'heads': heads,
        'entropy_matrix': entropy_matrix,
        'observation_counts': counts_matrix,
        'meta': {
            'source_npz': str(TRAIN_NPZ.relative_to(REPO)),
            'vocab': str(VOCAB_JSON.relative_to(REPO)),
            'fold': 1,
            'split': 'train',
            'units': 'bits (log2)',
            'description': (
                'Per-(head, op_class) Shannon entropy over training tokens, '
                'filtered to non-PAD positions and to positions where the head '
                'is well-defined (command: type=COMMAND; param_type: type in '
                '{PARAMETER, NUMERIC}; sign/digit: type=NUMERIC). Digit head '
                'pools all 6 digit slots.'
            ),
            'n_samples_per_op': {
                op: int(np.sum(op_names == op)) for op in op_classes
            },
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_JSON}")


if __name__ == '__main__':
    main()
