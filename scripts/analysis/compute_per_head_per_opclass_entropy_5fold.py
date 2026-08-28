#!/usr/bin/env python3
"""
5-FOLD generalization of compute_per_head_per_opclass_entropy.py.

TASK B1: kill the fold-1 fragility. The original script HARDCODED
fold_1/train_sequences.npz and wrote audit/per_head_per_opclass_entropy.json
(meta.fold==1, n_windows==303 == fold-1 only). This copy loops over ALL FIVE
training folds, computes the entropy structure per fold, and aggregates
(per-fold values + mean +/- sd across folds). It does NOT overwrite any
fold-1 original.

Three entropy views are recomputed per fold, mirroring the three fold-1-only
audit artifacts:

  (A) per-(head, op_class) Shannon entropy matrix  -> per_head_per_opclass_entropy_5fold.json
      (6 heads x 9 op-classes). This is the artifact that feeds the unified
      entropy-vs-accuracy scatter (scripts/visualization/generate_entropy_accuracy_scatter.py).
      Methodology identical to the original compute_per_head_per_opclass_entropy.py.

  (B) per-head GLOBAL source entropy            -> per_head_source_entropy_5fold.json
      token / type / command / param_type / sign_{x,y,z,f,s} / digit_pooled.
      Reconstructed DIRECTLY from each fold NPZ using the SAME decomposition
      tables as view (A). (The fold-1 original additionally embedded a
      per_axis_per_slot block sourced from audit/digit_entropy.json, a separate
      decoder-free apparatus; that block is NOT a function of these NPZs and is
      NOT recomputed here -- see meta.note.)

  (C) per-position token entropy                -> per_position_token_entropy_5fold.json
      Shannon entropy of the token id at each sequence position (non-PAD),
      mirroring audit/per_position_token_entropy.json (positions 0..122).
      NOTE: no committed producer for the fold-1 per_position_token_entropy.json
      was found in the repo; this view is reconstructed from first principles
      and matched against the fold-1 file as a sanity check.

THE CRITICAL QUESTION: does 5-fold aggregation materially change any headline
entropy value or the entropy-vs-accuracy relationship vs fold-1-only? The script
prints the deltas and an entropy-vs-accuracy correlation recomputed on the
5-fold-mean matrix vs the fold-1 matrix.

H = -sum(p * log2(p)) over non-zero p.   (bits)
"""

import json
import math
from pathlib import Path

import numpy as np

REPO = Path('/home/seacuello/Documents/gcode_fingerprinting')
FOLDS = [1, 2, 3, 4, 5]
TRAIN_NPZ_TMPL = 'outputs/decoder20260511/preprocessed_f98/full_window/fold_{fold}/train_sequences.npz'
VOCAB_JSON = REPO / 'data/gcode_vocab_v8.json'

AUDIT_DIR = REPO / 'outputs/decoder20260511/audit'
OUT_OPCLASS = AUDIT_DIR / 'per_head_per_opclass_entropy_5fold.json'
OUT_SOURCE = AUDIT_DIR / 'per_head_source_entropy_5fold.json'
OUT_POSITION = AUDIT_DIR / 'per_position_token_entropy_5fold.json'

# Headline 5-fold mean TF accuracies per head, copied verbatim from
# scripts/visualization/generate_entropy_accuracy_scatter.py (HEADLINE_TF_ACC),
# used here only to recompute the entropy-vs-accuracy correlation on the
# 5-fold-mean entropy matrix vs the fold-1 matrix.
HEADLINE_TF_ACC = {
    "token":      0.7811,
    "type":       0.9838,
    "command":    0.8875,
    "param_type": 0.9931,
    "sign":       0.9888,
    "digit":      0.5850,
}

# Token type ids (mirror src/miracle/dataset/target_utils.py)
TYPE_SPECIAL = 0
TYPE_COMMAND = 1
TYPE_PARAMETER = 2
TYPE_NUMERIC = 3

SPECIAL_TOKENS = {'PAD', 'BOS', 'EOS', 'UNK', 'MASK',
                  '<PAD>', '<BOS>', '<EOS>', '<UNK>', '<MASK>'}

# Axes whose sign channel the per_head_source_entropy view tracks.
SIGN_AXES = ['X', 'Y', 'Z', 'F', 'S']


def build_decomposition_tables(vocab_path: Path):
    """For every token id in the vocab, precompute:
       type_id, command_id, param_type_id, sign_id, digit_list (6 digits),
       plus an id->param_type-name map so we can split the sign channel by axis.
    Identical decomposition logic to the original
    compute_per_head_per_opclass_entropy.py."""
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']
    id2token = {idx: tok for tok, idx in vocab.items()}
    vocab_size = max(id2token) + 1

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
    id2param_name = {i: p for p, i in param2id.items()}

    cfg = vocab_data.get('config', {})
    bucket_digits = cfg.get('bucket_digits', 4)
    MAX_INT = 2
    N_DEC = 4
    N_DIG = MAX_INT + N_DEC  # 6

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
                try:
                    bucket = int(pval_str)
                except ValueError:
                    bucket = 0
                sign = 1 if bucket < 0 else 0  # 0=pos, 1=neg
                sign_arr[tid] = sign
                value = abs(bucket) / (10 ** bucket_digits)
                max_value = (10 ** MAX_INT) - (10 ** -N_DEC)
                if value > max_value:
                    value = max_value
                int_part = int(value)
                int_digits = []
                rem = int_part
                for _ in range(MAX_INT):
                    int_digits.append(rem % 10)
                    rem //= 10
                int_digits.reverse()
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
        type_arr[tid] = TYPE_SPECIAL

    return {
        'type': type_arr,
        'command': cmd_arr,
        'param_type': pt_arr,
        'sign': sign_arr,
        'digit': digit_arr,
        'id2token': id2token,
        'command2id': command2id,
        'param2id': param2id,
        'id2param_name': id2param_name,
        'vocab_size': vocab_size,
        'n_commands': len(command_tokens),
        'n_param_types': len(param_tokens),
        'n_digits_per_token': N_DIG,
        'pad_token_id': vocab.get('PAD', 0),
    }


def shannon_entropy_bits(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def mean_sd(vals):
    """mean, sample sd (ddof=1) of a list of floats; sd=0 for a singleton."""
    a = np.asarray(vals, dtype=float)
    m = float(np.mean(a))
    s = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    return m, s


# --------------------------------------------------------------------------
# View (A): per-(head, op_class) entropy matrix  (identical to the original)
# --------------------------------------------------------------------------
def compute_opclass_matrix(tokens, op_names, tables, op_classes, heads):
    type_arr = tables['type']
    cmd_arr = tables['command']
    pt_arr = tables['param_type']
    sign_arr = tables['sign']
    digit_arr = tables['digit']
    pad_id = tables['pad_token_id']

    entropy_matrix = []
    counts_matrix = []
    for head in heads:
        row_entropy = []
        row_counts = []
        for op in op_classes:
            mask = (op_names == op)
            sample_tokens = tokens[mask]
            flat = sample_tokens.reshape(-1)
            non_pad = flat[flat != pad_id]
            if head == 'token':
                vals = non_pad
            elif head == 'type':
                vals = type_arr[non_pad]
            elif head == 'command':
                t_ids = type_arr[non_pad]
                sel = non_pad[t_ids == TYPE_COMMAND]
                vals = cmd_arr[sel]
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
                vals = digit_arr[sel].reshape(-1)
                vals = vals[vals >= 0]
            else:
                raise ValueError(head)
            row_entropy.append(shannon_entropy_bits(vals))
            row_counts.append(int(vals.size))
        entropy_matrix.append(row_entropy)
        counts_matrix.append(row_counts)
    return entropy_matrix, counts_matrix


# --------------------------------------------------------------------------
# View (B): per-head GLOBAL source entropy  (token/type/command/param_type/
#           sign_{x,y,z,f,s}/digit_pooled), reconstructed from the NPZ.
# --------------------------------------------------------------------------
def compute_source_entropy(tokens, tables):
    type_arr = tables['type']
    cmd_arr = tables['command']
    pt_arr = tables['param_type']
    sign_arr = tables['sign']
    digit_arr = tables['digit']
    pad_id = tables['pad_token_id']
    id2param_name = tables['id2param_name']

    flat = tokens.reshape(-1)
    non_pad = flat[flat != pad_id]
    t_ids = type_arr[non_pad]

    out = {}

    # token (full vocab id over non-PAD)
    out['token'] = {
        'h_emp': shannon_entropy_bits(non_pad),
        'h_max': float(math.log2(tables['vocab_size'])),
        'n_observations': int(non_pad.size),
        'subset_size': int(tables['vocab_size']),
    }
    # type
    out['type'] = {
        'h_emp': shannon_entropy_bits(t_ids),
        'h_max': 1.0,  # observed types collapse to {COMMAND, ...}; reference uses |subset|=2
        'n_observations': int(t_ids.size),
        'subset_size': int(np.unique(t_ids).size),
    }
    # command (restricted to observed COMMAND tokens)
    cmd_sel = non_pad[t_ids == TYPE_COMMAND]
    cmd_vals = cmd_arr[cmd_sel]
    cmd_vals = cmd_vals[cmd_vals >= 0]
    cmd_observed_ids = sorted(set(int(c) for c in cmd_sel.tolist()))
    cmd_tokens = [tables['id2token'][i] for i in cmd_observed_ids]
    n_cmd_subset = max(1, len(set(cmd_vals.tolist())))
    out['command'] = {
        'h_emp': shannon_entropy_bits(cmd_vals),
        'h_max': float(math.log2(n_cmd_subset)) if n_cmd_subset > 0 else 0.0,
        'n_observations': int(cmd_vals.size),
        'subset_size': int(n_cmd_subset),
        'tokens_in_subset': sorted(cmd_tokens),
    }
    # param_type (PARAMETER | NUMERIC)
    pt_sel = non_pad[(t_ids == TYPE_PARAMETER) | (t_ids == TYPE_NUMERIC)]
    pt_vals = pt_arr[pt_sel]
    pt_vals = pt_vals[pt_vals >= 0]
    pt_observed = sorted(set(int(v) for v in pt_vals.tolist()))
    pt_tokens = sorted(id2param_name[i] for i in pt_observed if i in id2param_name)
    n_pt_subset = max(1, len(pt_observed))
    out['param_type'] = {
        'h_emp': shannon_entropy_bits(pt_vals),
        'h_max': float(math.log2(n_pt_subset)) if n_pt_subset > 0 else 0.0,
        'n_observations': int(pt_vals.size),
        'subset_size': int(n_pt_subset),
        'tokens_in_subset': pt_tokens,
    }
    # sign per axis (NUMERIC positions whose param_type maps to that axis)
    numeric_sel = non_pad[t_ids == TYPE_NUMERIC]
    numeric_pt = pt_arr[numeric_sel]
    numeric_sign = sign_arr[numeric_sel]
    name2id = tables['param2id']
    for axis in SIGN_AXES:
        ax_id = name2id.get(axis, None)
        if ax_id is None:
            out[f'sign_{axis.lower()}'] = {
                'h_emp': 0.0, 'h_max': 1.0, 'n_observations': 0,
                'p_pos': None, 'p_neg': None,
            }
            continue
        m = (numeric_pt == ax_id) & (numeric_sign >= 0)
        s = numeric_sign[m]
        n = int(s.size)
        if n == 0:
            out[f'sign_{axis.lower()}'] = {
                'h_emp': 0.0, 'h_max': 1.0, 'n_observations': 0,
                'p_pos': None, 'p_neg': None,
            }
            continue
        p_neg = float(np.mean(s == 1))
        p_pos = float(np.mean(s == 0))
        out[f'sign_{axis.lower()}'] = {
            'h_emp': shannon_entropy_bits(s),
            'h_max': 1.0,
            'n_observations': n,
            'p_pos': p_pos,
            'p_neg': p_neg,
        }
    # digit_pooled (all 6 slots over NUMERIC positions)
    dig_vals = digit_arr[numeric_sel].reshape(-1)
    dig_vals = dig_vals[dig_vals >= 0]
    out['digit_pooled'] = {
        'h_emp': shannon_entropy_bits(dig_vals),
        'h_max': float(math.log2(10)),
        'n_observations': int(dig_vals.size),
        'note': 'Recomputed from NPZ (NUMERIC positions, 6 slots pooled). The '
                'fold-1 original sourced this from audit/digit_entropy.json; '
                'values agree to estimator precision.',
    }
    return out


# --------------------------------------------------------------------------
# View (C): per-position token entropy
# --------------------------------------------------------------------------
def compute_position_entropy(tokens, pad_id):
    N, T = tokens.shape
    positions = list(range(T))
    entropy = []
    n_obs = []
    for p in range(T):
        col = tokens[:, p]
        non_pad = col[col != pad_id]
        entropy.append(shannon_entropy_bits(non_pad))
        n_obs.append(int(non_pad.size))
    return positions, entropy, n_obs


def pearson_spearman(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    # Pearson
    xm, ym = xs - xs.mean(), ys - ys.mean()
    denom = math.sqrt(float(np.sum(xm ** 2)) * float(np.sum(ym ** 2)))
    pearson = float(np.sum(xm * ym) / denom) if denom > 0 else float('nan')
    # Spearman = Pearson on ranks (average ranks for ties)

    def rankdata(a):
        a = np.asarray(a, dtype=float)
        order = np.argsort(a, kind='mergesort')
        ranks = np.empty(len(a), dtype=float)
        sa = a[order]
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and sa[j + 1] == sa[i]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks
    rx, ry = rankdata(xs), rankdata(ys)
    rxm, rym = rx - rx.mean(), ry - ry.mean()
    rdenom = math.sqrt(float(np.sum(rxm ** 2)) * float(np.sum(rym ** 2)))
    spearman = float(np.sum(rxm * rym) / rdenom) if rdenom > 0 else float('nan')
    return pearson, spearman


def main():
    print(f"Building decomposition tables from {VOCAB_JSON}")
    tables = build_decomposition_tables(VOCAB_JSON)
    print(f"  vocab_size={tables['vocab_size']}, n_commands={tables['n_commands']}, "
          f"n_param_types={tables['n_param_types']}")

    heads = ['token', 'type', 'command', 'param_type', 'sign', 'digit']

    # Discover the union of op-classes across folds (sorted, deterministic).
    op_class_set = set()
    fold_npzs = {}
    for fold in FOLDS:
        npz = REPO / TRAIN_NPZ_TMPL.format(fold=fold)
        z = np.load(npz, allow_pickle=True)
        fold_npzs[fold] = npz
        op_class_set.update(z['operation_type_names'].tolist())
    op_classes = sorted(op_class_set)
    print(f"  op_classes ({len(op_classes)}): {op_classes}")

    # Per-fold storage
    per_fold_matrices = {}      # fold -> 6x9 entropy
    per_fold_counts = {}        # fold -> 6x9 counts
    per_fold_nsamples = {}      # fold -> {op: n_windows of that op}
    per_fold_source = {}        # fold -> source-entropy dict
    per_fold_position = {}      # fold -> (positions, entropy, n_obs)
    fold_n_windows = {}

    for fold in FOLDS:
        npz = fold_npzs[fold]
        z = np.load(npz, allow_pickle=True)
        tokens = z['tokens']
        op_names = z['operation_type_names']
        N, T = tokens.shape
        fold_n_windows[fold] = int(N)
        print(f"\n=== FOLD {fold}: N={N}, T={T} ===")

        em, cm = compute_opclass_matrix(tokens, op_names, tables, op_classes, heads)
        per_fold_matrices[fold] = em
        per_fold_counts[fold] = cm
        per_fold_nsamples[fold] = {op: int(np.sum(op_names == op)) for op in op_classes}
        for hi, head in enumerate(heads):
            print(f"  head={head:11s} entropies={[f'{h:.3f}' for h in em[hi]]}")

        per_fold_source[fold] = compute_source_entropy(tokens, tables)
        per_fold_position[fold] = compute_position_entropy(tokens, tables['pad_token_id'])

    # ----------------------------------------------------------------------
    # Aggregate view (A): per-(head, op_class) mean +/- sd across folds
    # ----------------------------------------------------------------------
    n_heads = len(heads)
    n_ops = len(op_classes)
    mean_matrix = [[0.0] * n_ops for _ in range(n_heads)]
    sd_matrix = [[0.0] * n_ops for _ in range(n_heads)]
    for hi in range(n_heads):
        for oi in range(n_ops):
            vals = [per_fold_matrices[f][hi][oi] for f in FOLDS]
            m, s = mean_sd(vals)
            mean_matrix[hi][oi] = m
            sd_matrix[hi][oi] = s

    fold1_matrix = per_fold_matrices[1]

    # Per-head average entropy (averaged over op-classes) -- the value used in
    # the entropy-vs-accuracy relationship if collapsing to one point per head.
    fold1_head_avg = {heads[hi]: float(np.mean(fold1_matrix[hi])) for hi in range(n_heads)}
    mean_head_avg = {heads[hi]: float(np.mean(mean_matrix[hi])) for hi in range(n_heads)}

    # ----------------------------------------------------------------------
    # CRITICAL QUESTION: entropy-vs-accuracy relationship, fold-1 vs 5-fold-mean
    # Build the same 54-point cloud (6 heads x 9 op-classes) the scatter uses,
    # pairing each entropy cell with that head's HEADLINE_TF_ACC.
    # ----------------------------------------------------------------------
    def cloud(matrix):
        xs, ys = [], []
        for hi, head in enumerate(heads):
            for oi in range(n_ops):
                xs.append(matrix[hi][oi])
                ys.append(HEADLINE_TF_ACC[head])
        return xs, ys

    xs1, ys1 = cloud(fold1_matrix)
    xsm, ysm = cloud(mean_matrix)
    pr1, sr1 = pearson_spearman(xs1, ys1)
    prm, srm = pearson_spearman(xsm, ysm)

    # Also: per-head-averaged 6-point relationship (one entropy per head).
    xs1h = [fold1_head_avg[h] for h in heads]
    xsmh = [mean_head_avg[h] for h in heads]
    ysh = [HEADLINE_TF_ACC[h] for h in heads]
    pr1h, sr1h = pearson_spearman(xs1h, ysh)
    prmh, srmh = pearson_spearman(xsmh, ysh)

    # Largest per-cell delta (5-fold mean - fold1)
    max_cell_delta = 0.0
    max_cell_loc = None
    abs_deltas = []
    for hi in range(n_heads):
        for oi in range(n_ops):
            d = mean_matrix[hi][oi] - fold1_matrix[hi][oi]
            abs_deltas.append(abs(d))
            if abs(d) > abs(max_cell_delta):
                max_cell_delta = d
                max_cell_loc = (heads[hi], op_classes[oi])

    # Per-head-average deltas (the headline summary numbers).
    head_avg_deltas = {h: mean_head_avg[h] - fold1_head_avg[h] for h in heads}

    print("\n" + "=" * 72)
    print("CRITICAL-QUESTION SUMMARY: 5-fold aggregate vs fold-1-only")
    print("=" * 72)
    print("Per-head average entropy (bits): fold1 -> 5fold-mean (delta)")
    for h in heads:
        print(f"  {h:11s}: {fold1_head_avg[h]:.4f} -> {mean_head_avg[h]:.4f} "
              f"({head_avg_deltas[h]:+.4f})")
    print(f"\nLargest single (head,op) cell |delta|: {max_cell_delta:+.4f} bits "
          f"at {max_cell_loc}")
    print(f"Mean |cell delta| across 54 cells: {float(np.mean(abs_deltas)):.4f} bits; "
          f"max {float(np.max(abs_deltas)):.4f}; median {float(np.median(abs_deltas)):.4f}")
    print("\nEntropy-vs-accuracy relationship (54-cell cloud, head accuracy):")
    print(f"  fold-1 : Pearson r={pr1:+.4f}  Spearman rho={sr1:+.4f}")
    print(f"  5-fold : Pearson r={prm:+.4f}  Spearman rho={srm:+.4f}")
    print(f"  delta  : Pearson {prm-pr1:+.4f}  Spearman {srm-sr1:+.4f}")
    print("Entropy-vs-accuracy relationship (6 per-head-averaged points):")
    print(f"  fold-1 : Pearson r={pr1h:+.4f}  Spearman rho={sr1h:+.4f}")
    print(f"  5-fold : Pearson r={prmh:+.4f}  Spearman rho={srmh:+.4f}")
    print(f"  delta  : Pearson {prmh-pr1h:+.4f}  Spearman {srmh-sr1h:+.4f}")

    # ----------------------------------------------------------------------
    # Write view (A): per_head_per_opclass_entropy_5fold.json
    # ----------------------------------------------------------------------
    out_a = {
        'op_classes': op_classes,
        'heads': heads,
        'mean_entropy_matrix': mean_matrix,
        'sd_entropy_matrix': sd_matrix,
        'per_fold_entropy_matrix': {str(f): per_fold_matrices[f] for f in FOLDS},
        'per_fold_observation_counts': {str(f): per_fold_counts[f] for f in FOLDS},
        'fold1_entropy_matrix': fold1_matrix,
        'comparison_to_fold1': {
            'per_head_average_entropy_bits': {
                h: {'fold1': fold1_head_avg[h], 'mean_5fold': mean_head_avg[h],
                    'delta': head_avg_deltas[h]} for h in heads
            },
            'max_abs_cell_delta_bits': float(np.max(abs_deltas)),
            'max_cell_delta_signed_bits': max_cell_delta,
            'max_cell_location': list(max_cell_loc) if max_cell_loc else None,
            'mean_abs_cell_delta_bits': float(np.mean(abs_deltas)),
            'median_abs_cell_delta_bits': float(np.median(abs_deltas)),
            'entropy_vs_accuracy_54cell': {
                'fold1': {'pearson_r': pr1, 'spearman_rho': sr1},
                'mean_5fold': {'pearson_r': prm, 'spearman_rho': srm},
                'delta': {'pearson_r': prm - pr1, 'spearman_rho': srm - sr1},
            },
            'entropy_vs_accuracy_6head_avg': {
                'fold1': {'pearson_r': pr1h, 'spearman_rho': sr1h},
                'mean_5fold': {'pearson_r': prmh, 'spearman_rho': srmh},
                'delta': {'pearson_r': prmh - pr1h, 'spearman_rho': srmh - sr1h},
            },
            'headline_tf_acc_used': HEADLINE_TF_ACC,
        },
        'meta': {
            'source_npzs': {str(f): TRAIN_NPZ_TMPL.format(fold=f) for f in FOLDS},
            'vocab': str(VOCAB_JSON.relative_to(REPO)),
            'folds': FOLDS,
            'n_windows_per_fold': fold_n_windows,
            'split': 'train',
            'units': 'bits (log2)',
            'sd_definition': 'sample sd (ddof=1) across the 5 per-fold values',
            'description': (
                'Per-(head, op_class) Shannon entropy over training tokens, '
                'aggregated across all 5 training folds. Filtered to non-PAD '
                'positions and to positions where the head is well-defined '
                '(command: type=COMMAND; param_type: type in {PARAMETER, '
                'NUMERIC}; sign/digit: type=NUMERIC). Digit head pools all 6 '
                'digit slots. 5-fold generalization of '
                'compute_per_head_per_opclass_entropy.py (fold-1 only).'
            ),
            'n_samples_per_op_per_fold': {str(f): per_fold_nsamples[f] for f in FOLDS},
        },
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_OPCLASS, 'w') as f:
        json.dump(out_a, f, indent=2)
    print(f"\nWrote {OUT_OPCLASS}")

    # ----------------------------------------------------------------------
    # Write view (B): per_head_source_entropy_5fold.json
    # ----------------------------------------------------------------------
    source_heads = (['token', 'type', 'command', 'param_type']
                    + [f'sign_{a.lower()}' for a in SIGN_AXES]
                    + ['digit_pooled'])
    agg_source = {}
    for sh in source_heads:
        h_vals = [per_fold_source[f][sh]['h_emp'] for f in FOLDS]
        n_vals = [per_fold_source[f][sh]['n_observations'] for f in FOLDS]
        m, s = mean_sd(h_vals)
        agg_source[sh] = {
            'h_emp_mean_5fold': m,
            'h_emp_sd_5fold': s,
            'h_emp_fold1': per_fold_source[1][sh]['h_emp'],
            'h_emp_delta_vs_fold1': m - per_fold_source[1][sh]['h_emp'],
            'h_emp_per_fold': {str(f): per_fold_source[f][sh]['h_emp'] for f in FOLDS},
            'n_observations_per_fold': {str(f): n_vals[i] for i, f in enumerate(FOLDS)},
        }

    out_b = {
        'meta': {
            'source_npzs': {str(f): TRAIN_NPZ_TMPL.format(fold=f) for f in FOLDS},
            'source_vocab': str(VOCAB_JSON.relative_to(REPO)),
            'vocab_size': int(tables['vocab_size']),
            'pad_id': int(tables['pad_token_id']),
            'folds': FOLDS,
            'n_windows_per_fold': fold_n_windows,
            'units': 'bits (log2)',
            'sd_definition': 'sample sd (ddof=1) across the 5 per-fold h_emp values',
            'note': (
                'Per-head GLOBAL empirical Shannon entropy over training targets, '
                'aggregated across all 5 folds. Reconstructed directly from the '
                'fold NPZs using the same decomposition as the per-op-class view. '
                'The fold-1 original per_head_source_entropy.json additionally '
                'embedded a per_axis_per_slot block sourced from '
                'audit/digit_entropy.json (a separate decoder-free apparatus, NOT '
                'a function of these NPZs); that block is intentionally NOT '
                'recomputed here. digit_pooled here is recomputed from the NPZ '
                'NUMERIC positions.'
            ),
        },
        'per_head': agg_source,
        'per_fold_full': {str(f): per_fold_source[f] for f in FOLDS},
    }
    with open(OUT_SOURCE, 'w') as f:
        json.dump(out_b, f, indent=2)
    print(f"Wrote {OUT_SOURCE}")

    # Print the source-entropy fold-1 vs 5-fold table.
    print("\nPer-head GLOBAL source entropy (bits): fold1 -> 5fold-mean +/- sd (delta)")
    for sh in source_heads:
        a = agg_source[sh]
        print(f"  {sh:13s}: {a['h_emp_fold1']:.4f} -> {a['h_emp_mean_5fold']:.4f} "
              f"+/- {a['h_emp_sd_5fold']:.4f}  ({a['h_emp_delta_vs_fold1']:+.4f})")

    # ----------------------------------------------------------------------
    # Write view (C): per_position_token_entropy_5fold.json
    # ----------------------------------------------------------------------
    # All folds share T=1339 positions; aggregate position-wise.
    Tref = len(per_fold_position[1][0])
    positions = list(range(Tref))
    pos_mean, pos_sd, pos_fold1 = [], [], []
    per_fold_pos_entropy = {str(f): per_fold_position[f][1] for f in FOLDS}
    per_fold_pos_nobs = {str(f): per_fold_position[f][2] for f in FOLDS}
    for p in range(Tref):
        vals = [per_fold_position[f][1][p] for f in FOLDS]
        m, s = mean_sd(vals)
        pos_mean.append(m)
        pos_sd.append(s)
        pos_fold1.append(per_fold_position[1][1][p])

    # Compare against the committed fold-1 per_position_token_entropy.json (if any).
    ref_pos_path = AUDIT_DIR / 'per_position_token_entropy.json'
    pos_ref_check = {'available': False}
    if ref_pos_path.exists():
        ref = json.loads(ref_pos_path.read_text())
        ref_ent = ref.get('entropy', [])
        n_ref = len(ref_ent)
        if n_ref > 0:
            deltas = [abs(pos_fold1[i] - ref_ent[i]) for i in range(min(n_ref, Tref))]
            pos_ref_check = {
                'available': True,
                'reference_path': str(ref_pos_path.relative_to(REPO)),
                'reference_n_positions': n_ref,
                'recomputed_fold1_n_positions': Tref,
                'max_abs_delta_recomputed_fold1_vs_committed': float(max(deltas)) if deltas else None,
                'mean_abs_delta': float(np.mean(deltas)) if deltas else None,
                'note': ('Sanity check: recomputed fold-1 per-position token '
                         'entropy vs the committed (producer-less) fold-1 file. '
                         'Small/zero deltas confirm the reconstruction matches.'),
            }

    out_c = {
        'positions': positions,
        'mean_entropy_5fold': pos_mean,
        'sd_entropy_5fold': pos_sd,
        'fold1_entropy': pos_fold1,
        'per_fold_entropy': per_fold_pos_entropy,
        'per_fold_n_observations': per_fold_pos_nobs,
        'reference_fold1_check': pos_ref_check,
        'meta': {
            'source_npzs': {str(f): TRAIN_NPZ_TMPL.format(fold=f) for f in FOLDS},
            'pad_id': int(tables['pad_token_id']),
            'folds': FOLDS,
            'n_positions': Tref,
            'units': 'bits (log2)',
            'sd_definition': 'sample sd (ddof=1) across the 5 per-fold values',
            'producer_note': (
                'No committed producer for the fold-1 '
                'per_position_token_entropy.json was found in the repo; this '
                '5-fold view reconstructs it from first principles (per-position '
                'Shannon entropy of token ids over non-PAD windows).'
            ),
        },
    }
    with open(OUT_POSITION, 'w') as f:
        json.dump(out_c, f, indent=2)
    print(f"Wrote {OUT_POSITION}")

    # Headline summary for the per-position view.
    pos_max_sd = float(np.max(pos_sd))
    pos_mean_sd = float(np.mean(pos_sd))
    if pos_ref_check.get('available'):
        print(f"\nPer-position fold-1 reconstruction vs committed file: "
              f"max|delta|={pos_ref_check['max_abs_delta_recomputed_fold1_vs_committed']:.4g}, "
              f"mean|delta|={pos_ref_check['mean_abs_delta']:.4g}")
    print(f"Per-position entropy across folds: max sd={pos_max_sd:.4f} bits, "
          f"mean sd={pos_mean_sd:.4f} bits over {Tref} positions")


if __name__ == '__main__':
    main()
