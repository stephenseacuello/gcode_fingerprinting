#!/usr/bin/env python3
"""Generate ROC curves and score distribution figures for the anomaly paper."""

import sys
sys.path.insert(0, '/home/seacuello/Documents/gcode_fingerprinting/src')
sys.path.insert(0, '/home/seacuello/Documents/gcode_fingerprinting')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

from scripts.anomaly.anomaly_scoring_utils import (
    load_cached_logits, load_cached_targets, load_vocab,
    compute_per_token_nll, compute_window_scores, compute_rank_scores,
    FOLDS, ATTACK_DIR,
)

OUT_DIR = Path('/home/seacuello/Documents/gcode_fingerprinting/outputs/anomaly20260319/anomalypaper/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
})

# ── Load attacks ──
print("Loading attack data...")
a1 = torch.load(ATTACK_DIR / 'attack_A1_command_swap.pt', weights_only=False, map_location='cpu')
a2 = torch.load(ATTACK_DIR / 'attack_A2_coord_graded.pt', weights_only=False, map_location='cpu')
a3 = torch.load(ATTACK_DIR / 'attack_A3_param_swap.pt', weights_only=False, map_location='cpu')
a4 = torch.load(ATTACK_DIR / 'attack_A4_cross_op.pt', weights_only=False, map_location='cpu')

vocab = load_vocab()
vocab_map = vocab['vocab']  # name -> id


def get_attack_info(attack_obj, fold, key='fold_attacks'):
    """Get attack info for a fold."""
    return attack_obj[key][fold]


NLL_CLIP = 50.0  # Clip extreme NLL values (model artifacts at BOS/EOS boundaries)


def _robust_window_scores(logits, targets, padding_mask):
    """Compute robust NLL S_mean: excludes PAD, EOS, and clips extreme values."""
    nll = compute_per_token_nll(logits, targets, padding_mask)
    # Content mask: exclude PAD (0) and EOS (2)
    content_mask = (targets != 0) & (targets != 2)
    # Clip extreme NLL values (BOS boundary artifacts)
    nll = nll.clamp(max=NLL_CLIP)
    content_lengths = content_mask.sum(dim=1).float().clamp(min=1)
    s_mean = (nll * content_mask.float()).sum(dim=1) / content_lengths
    s_max = nll.clone()
    s_max[~content_mask] = -float('inf')
    s_max = s_max.max(dim=1).values
    return s_mean.numpy(), s_max.numpy()


def score_fold(fold, attack_info):
    """Compute NLL S_mean for normal windows and attacked windows in a fold.

    Normal scores = original NLL for ALL windows (unmodified targets).
    Attacked scores = NLL when attacked tokens replace the originals at attacked positions.

    Returns (normal_scores, attacked_scores) as numpy arrays.
    """
    cached = load_cached_logits(fold)
    logits = cached['legacy_logits']
    tgt_data = load_cached_targets(fold)
    targets = tgt_data['target_tokens']
    padding_mask = tgt_data['padding_mask']

    N = logits.shape[0]

    # Normal NLL for ALL windows (unmodified)
    normal_scores, _ = _robust_window_scores(logits, targets, padding_mask)

    # Get attacked indices
    orig_idx = attack_info['orig_indices']
    if isinstance(orig_idx, torch.Tensor):
        orig_idx = orig_idx.numpy()
    orig_idx = np.array(orig_idx)
    valid_idx = orig_idx[orig_idx < N]

    # Build attacked targets: only replace at attacked positions
    attacked_tokens = attack_info['attacked_tokens']
    if not isinstance(attacked_tokens, torch.Tensor):
        attacked_tokens = torch.tensor(attacked_tokens)

    # Score attacked windows: compute NLL with substituted tokens
    attacked_logits = logits[valid_idx]
    attacked_pmask = padding_mask[valid_idx]
    n_valid = len(valid_idx)
    seq_len = min(attacked_tokens.shape[1], targets.shape[1])
    attacked_tgt = targets[valid_idx].clone()
    attacked_tgt[:n_valid, :seq_len] = attacked_tokens[:n_valid, :seq_len]

    attacked_scores, _ = _robust_window_scores(attacked_logits, attacked_tgt, attacked_pmask)

    return normal_scores, attacked_scores


def score_fold_rank(fold, attack_info):
    """Compute Rank S_rank_mean for normal and attacked windows."""
    cached = load_cached_logits(fold)
    logits = cached['legacy_logits']
    tgt_data = load_cached_targets(fold)
    targets = tgt_data['target_tokens']

    N = logits.shape[0]

    # Rank scores for ALL windows (normal)
    rank_scores = compute_rank_scores(logits, targets, vocab_map)
    normal_s_rank = rank_scores['S_rank_mean'].numpy()

    # Get attacked indices
    orig_idx = attack_info['orig_indices']
    if isinstance(orig_idx, torch.Tensor):
        orig_idx = orig_idx.numpy()
    orig_idx = np.array(orig_idx)
    valid_idx = orig_idx[orig_idx < N]

    # Build attacked targets for attacked windows only
    attacked_tokens = attack_info['attacked_tokens']
    if not isinstance(attacked_tokens, torch.Tensor):
        attacked_tokens = torch.tensor(attacked_tokens)

    n_valid = len(valid_idx)
    seq_len = min(attacked_tokens.shape[1], targets.shape[1])
    attacked_tgt = targets[valid_idx].clone()
    attacked_tgt[:n_valid, :seq_len] = attacked_tokens[:n_valid, :seq_len]

    attacked_rank = compute_rank_scores(logits[valid_idx], attacked_tgt, vocab_map)
    attacked_s_rank = attacked_rank['S_rank_mean'].numpy()

    return normal_s_rank, attacked_s_rank


# ══════════════════════════════════════════════════════════════
# Gather data across folds
# ══════════════════════════════════════════════════════════════

# A1: command swap - NLL per fold (for individual + mean ROC)
print("Computing A1 (command swap) scores...")
a1_per_fold = []
for f in FOLDS:
    info = get_attack_info(a1, f)
    n, a = score_fold(f, info)
    a1_per_fold.append((n, a))
    print(f"  Fold {f}: {len(n)} normal, {len(a)} attacked")

# A2: coord graded X delta=0.1 - NLL + Rank
print("Computing A2 (X delta=0.1) scores...")
a2_nll_normal_all, a2_nll_attack_all = [], []
a2_rank_normal_all, a2_rank_attack_all = [], []
for f in FOLDS:
    info = a2['attacks_by_delta_axis']['X_delta_0.1']['fold_attacks'][f]
    n, a = score_fold(f, info)
    a2_nll_normal_all.append(n)
    a2_nll_attack_all.append(a)
    rn, ra = score_fold_rank(f, info)
    a2_rank_normal_all.append(rn)
    a2_rank_attack_all.append(ra)
    print(f"  Fold {f}: {len(n)} normal, {len(a)} attacked")

# A3: param swap
print("Computing A3 (param swap) scores...")
a3_normal_all, a3_attack_all = [], []
for f in FOLDS:
    info = get_attack_info(a3, f)
    n, a = score_fold(f, info)
    a3_normal_all.append(n)
    a3_attack_all.append(a)
    print(f"  Fold {f}: {len(n)} normal, {len(a)} attacked")

# A4: cross-op (V7 only for now — V8 needs separate cache)
print("Computing A4 (cross-op) scores...")
a4_normal_all, a4_attack_all = [], []
for f in FOLDS:
    info = get_attack_info(a4, f)
    n, a = score_fold(f, info)
    a4_normal_all.append(n)
    a4_attack_all.append(a)
    print(f"  Fold {f}: {len(n)} normal, {len(a)} attacked")


def make_roc(normal, attacked):
    """Build ROC from normal (label=0) and attacked (label=1) scores."""
    labels = np.concatenate([np.zeros(len(normal)), np.ones(len(attacked))])
    scores = np.concatenate([normal, attacked])
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr, auc(fpr, tpr)


# ══════════════════════════════════════════════════════════════
# Figure 1: ROC Curves (2x2)
# ══════════════════════════════════════════════════════════════
print("\nGenerating ROC curves figure...")
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# --- A1: per-fold ROC + mean ---
ax = axes[0, 0]
mean_fpr = np.linspace(0, 1, 200)
tpr_interps = []
for i, (n, a) in enumerate(a1_per_fold):
    fpr, tpr, auroc = make_roc(n, a)
    ax.plot(fpr, tpr, color='tab:blue', alpha=0.3, linewidth=0.8)
    tpr_interps.append(np.interp(mean_fpr, fpr, tpr))

mean_tpr = np.mean(tpr_interps, axis=0)
mean_auroc = auc(mean_fpr, mean_tpr)
ax.plot(mean_fpr, mean_tpr, color='tab:blue', linewidth=2.0,
        label=f'NLL $S_{{mean}}$ (AUROC={mean_auroc:.3f})')
ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=0.8)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('(a) A1: Command Swap')
ax.legend(loc='lower right')
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])

# --- A2: NLL vs Rank ---
ax = axes[0, 1]
n_nll = np.concatenate(a2_nll_normal_all)
a_nll = np.concatenate(a2_nll_attack_all)
fpr_nll, tpr_nll, auroc_nll = make_roc(n_nll, a_nll)
ax.plot(fpr_nll, tpr_nll, color='tab:blue', linewidth=2.0,
        label=f'NLL $S_{{mean}}$ (AUROC={auroc_nll:.3f})')

n_rank = np.concatenate(a2_rank_normal_all)
a_rank = np.concatenate(a2_rank_attack_all)
fpr_rank, tpr_rank, auroc_rank = make_roc(n_rank, a_rank)
ax.plot(fpr_rank, tpr_rank, color='tab:orange', linewidth=2.0,
        label=f'Rank $S_{{rank\\_mean}}$ (AUROC={auroc_rank:.3f})')

ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=0.8)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(r'(b) A2: Coord $\Delta X$=0.1 mm')
ax.legend(loc='lower right')
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])

# --- A3: param swap ---
ax = axes[1, 0]
n3 = np.concatenate(a3_normal_all)
a3a = np.concatenate(a3_attack_all)
fpr3, tpr3, auroc3 = make_roc(n3, a3a)
ax.plot(fpr3, tpr3, color='tab:blue', linewidth=2.0,
        label=f'NLL $S_{{mean}}$ (AUROC={auroc3:.3f})')
ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=0.8)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('(c) A3: Parameter Swap')
ax.legend(loc='lower right')
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])

# --- A4: cross-op (V7 only) ---
ax = axes[1, 1]
n4 = np.concatenate(a4_normal_all)
a4a = np.concatenate(a4_attack_all)
fpr4, tpr4, auroc4 = make_roc(n4, a4a)
ax.plot(fpr4, tpr4, color='tab:blue', linewidth=2.0,
        label=f'NLL $S_{{mean}}$ V7 (AUROC={auroc4:.3f})')

# Check if V8 cache exists for A4
v8_cache = Path('/home/seacuello/Documents/gcode_fingerprinting/outputs/anomaly20260319/cached_inference_v8')
if v8_cache.exists() and (v8_cache / 'fold_1').exists():
    print("  V8 cache found, computing A4 V8 scores...")
    import os
    os.environ['ANOMALY_CACHE_DIR'] = str(v8_cache)
    # Reload to pick up V8 cache
    from scripts.anomaly.anomaly_scoring_utils import load_cached_v8
    v8_attacks = Path('/home/seacuello/Documents/gcode_fingerprinting/outputs/anomaly20260319/attacks_v8')
    if v8_attacks.exists() and (v8_attacks / 'attack_A4_cross_op.pt').exists():
        a4_v8 = torch.load(v8_attacks / 'attack_A4_cross_op.pt', weights_only=False, map_location='cpu')
        a4_v8_normal_all, a4_v8_attack_all = [], []
        for f in FOLDS:
            info = a4_v8['fold_attacks'][f]
            logits_v8_data = load_cached_v8(f, 'logits.pt')
            logits_v8 = logits_v8_data['legacy_logits'] if isinstance(logits_v8_data, dict) and 'legacy_logits' in logits_v8_data else logits_v8_data
            tgt_v8 = load_cached_v8(f, 'targets.pt')
            if isinstance(tgt_v8, dict):
                targets_v8 = tgt_v8['target_tokens']
                pmask_v8 = tgt_v8['padding_mask']
            else:
                targets_v8 = tgt_v8
                pmask_v8 = targets_v8 == 0

            N = logits_v8.shape[0]
            normal_s, _ = _robust_window_scores(logits_v8, targets_v8, pmask_v8)

            orig_idx = info['orig_indices']
            if isinstance(orig_idx, torch.Tensor):
                orig_idx = orig_idx.numpy()
            orig_idx = np.array(orig_idx)
            vi = orig_idx[orig_idx < N]
            at = info['attacked_tokens']
            if not isinstance(at, torch.Tensor):
                at = torch.tensor(at)

            nv = len(vi)
            sl = min(at.shape[1], targets_v8.shape[1])
            atgt = targets_v8[vi].clone()
            atgt[:nv, :sl] = at[:nv, :sl]
            attack_s, _ = _robust_window_scores(logits_v8[vi], atgt, pmask_v8[vi])

            a4_v8_normal_all.append(normal_s)
            a4_v8_attack_all.append(attack_s)

        n4v8 = np.concatenate(a4_v8_normal_all)
        a4v8 = np.concatenate(a4_v8_attack_all)
        fpr4v8, tpr4v8, auroc4v8 = make_roc(n4v8, a4v8)
        ax.plot(fpr4v8, tpr4v8, color='tab:red', linewidth=2.0,
                label=f'NLL $S_{{mean}}$ V8 (AUROC={auroc4v8:.3f})')
    else:
        print("  V8 attack file not found, showing V7 only")
else:
    print("  V8 cache not found, showing V7 only for A4")

ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=0.8)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('(d) A4: Cross-Operation')
ax.legend(loc='lower right')
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])

fig.tight_layout()
fig.savefig(OUT_DIR / 'roc_curves.pdf', dpi=300, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'roc_curves.pdf'}")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
# Figure 2: Score Distributions (2x1)
# ══════════════════════════════════════════════════════════════
print("\nGenerating score distributions figure...")
fig, axes = plt.subplots(2, 1, figsize=(8, 6))

# --- Top: NLL S_mean for normal vs A1 command swap ---
ax = axes[0]
all_a1_normal = np.concatenate([n for n, a in a1_per_fold])
all_a1_attack = np.concatenate([a for n, a in a1_per_fold])

# Determine common range
lo = min(all_a1_normal.min(), all_a1_attack.min())
hi = np.percentile(np.concatenate([all_a1_normal, all_a1_attack]), 99.5)
bins = np.linspace(lo, hi, 31)

ax.hist(all_a1_normal, bins=bins, density=True, alpha=0.5, color='tab:blue',
        label='Normal', edgecolor='none')
ax.hist(all_a1_attack, bins=bins, density=True, alpha=0.5, color='tab:red',
        label='A1 Command Swap', edgecolor='none')
ax.set_xlabel('NLL $S_{mean}$')
ax.set_ylabel('Density')
ax.set_title('(a) NLL Score Distribution: Normal vs. A1 Command Swap')
ax.legend()

# --- Bottom: Rank S_rank_mean for normal vs A2 X delta=0.1 ---
ax = axes[1]
all_a2_rank_normal = np.concatenate(a2_rank_normal_all)
all_a2_rank_attack = np.concatenate(a2_rank_attack_all)

lo2 = min(all_a2_rank_normal.min(), all_a2_rank_attack.min())
hi2 = max(np.percentile(all_a2_rank_normal, 99.5), np.percentile(all_a2_rank_attack, 99.5))
bins2 = np.linspace(lo2, hi2, 31)

ax.hist(all_a2_rank_normal, bins=bins2, density=True, alpha=0.5, color='tab:blue',
        label='Normal', edgecolor='none')
ax.hist(all_a2_rank_attack, bins=bins2, density=True, alpha=0.5, color='tab:orange',
        label=r'A2 $\Delta X$=0.1 mm', edgecolor='none')
ax.set_xlabel('Rank $S_{rank\\_mean}$')
ax.set_ylabel('Density')
ax.set_title(r'(b) Rank Score Distribution: Normal vs. A2 Coord $\Delta X$=0.1 mm')
ax.legend()

fig.tight_layout()
fig.savefig(OUT_DIR / 'score_distributions.pdf', dpi=300, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'score_distributions.pdf'}")
plt.close(fig)

print("\nDone!")
