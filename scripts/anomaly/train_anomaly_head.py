#!/usr/bin/env python3
"""Experiment 20: Trained Anomaly Head on Encoder/Decoder Features

Trains a binary anomaly classifier using features extracted from the
cached encoder memory and decoder outputs. Compares three classifiers:
  1. MLP (PyTorch): Linear(D,128) -> ReLU -> Dropout(0.3) -> Linear(128,1) -> sigmoid
  2. RandomForest (sklearn)
  3. XGBoost (if available)

Feature vector per window (~265 dims):
  - Mean-pooled encoder memory [256]
  - NLL statistics: S_mean, S_max, S_std [3]
  - Rank score: S_rank_mean, S_rank_max [2]
  - Per-head NLL: type, command, param, digit [4]

Training uses BCE loss with Adam lr=1e-4, early stopping on validation AUROC.
Synthetic attacks (A1-A4) provide positive labels; normal windows are negatives.
5-fold CV matches existing fold structure.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import (
    ATTACK_DIR,
    BASE_DIR,
    CACHE_DIR,
    FOLDS,
    PAD_ID,
    EOS_ID,
    load_cached_logits,
    load_cached_nll,
    load_cached_targets,
    load_cached_embeddings,
    load_vocab,
    compute_per_token_nll,
    compute_window_scores,
    compute_rank_scores,
    compute_auroc,
    compute_detection_at_fpr,
    bootstrap_auroc_ci,
    cohens_d,
    save_results,
    save_figure,
    append_to_summary,
    setup_logging,
    get_default_parser,
)

logger = logging.getLogger(__name__)

EXP_ID = 20
EXP_NAME = "Anomaly Head"
OUTPUT_DIR_NAME = "exp20_anomaly_head"


# ============================================================
# Feature Extraction
# ============================================================

def extract_features(fold: int) -> Tuple[np.ndarray, int]:
    """Extract feature vectors from cached inference data for one fold.

    Returns:
        features: [N, D] feature matrix
        n_features: dimensionality D
    """
    logger.info(f"Extracting features for fold {fold}")

    # --- Encoder memory: mean-pool [N, T, 256] -> [N, 256] ---
    embeddings = load_cached_embeddings(fold)
    test_memory = embeddings["test_memory"]  # [N, T, 256]
    encoder_feats = test_memory.mean(dim=1).numpy()  # [N, 256]

    # --- NLL statistics ---
    logits_data = load_cached_logits(fold)
    targets_data = load_cached_targets(fold)
    nll_data = load_cached_nll(fold)

    legacy_logits = logits_data["legacy_logits"]  # [N, L, 712]
    target_tokens = targets_data["target_tokens"]  # [N, L]
    padding_mask = targets_data["padding_mask"]  # [N, L]
    lengths = targets_data["lengths"]  # [N]

    # Per-token NLL (legacy)
    nll_legacy = nll_data["nll_legacy"]  # [N, L]

    # Window-level NLL aggregates
    window_scores = compute_window_scores(nll_legacy, lengths, target_tokens)
    s_mean = window_scores["S_mean"].numpy()  # [N]
    s_max = window_scores["S_max"].numpy()    # [N]

    # NLL std per window (over non-padded positions)
    content_mask = (target_tokens != PAD_ID) & (target_tokens != EOS_ID)
    content_lengths = content_mask.sum(dim=1).float().clamp(min=1)
    nll_masked = nll_legacy * content_mask.float()
    nll_mean_per = (nll_masked.sum(dim=1) / content_lengths)
    nll_var = ((nll_masked - nll_mean_per.unsqueeze(1)) ** 2 * content_mask.float()).sum(dim=1) / content_lengths
    s_std = torch.sqrt(nll_var).numpy()  # [N]

    nll_stats = np.stack([s_mean, s_max, s_std], axis=1)  # [N, 3]

    # --- Rank scores ---
    vocab = load_vocab()
    try:
        rank_scores = compute_rank_scores(legacy_logits, target_tokens, vocab["vocab"])
        s_rank_mean = rank_scores["S_rank_mean"].numpy()
        s_rank_max = rank_scores["S_rank_max"].numpy()
    except Exception as e:
        logger.warning(f"Rank score computation failed: {e}, using zeros")
        N = len(target_tokens)
        s_rank_mean = np.zeros(N)
        s_rank_max = np.zeros(N)
    rank_feats = np.stack([s_rank_mean, s_rank_max], axis=1)  # [N, 2]

    # --- Per-head NLL: type, command, param, digit ---
    nll_type = nll_data["nll_type"]  # [N, L]
    type_window = compute_window_scores(nll_type, lengths, target_tokens)
    nll_type_mean = type_window["S_mean"].numpy()

    # Compute command-head and param-head NLL from logits if available
    head_feats_list = [nll_type_mean]

    for head_name in ["command_logits", "param_type_logits"]:
        if head_name in logits_data:
            head_logits = logits_data[head_name]  # [N, L, C]
            # Use argmax of legacy target mapped to head categories as proxy
            # Simple approach: entropy of head distribution as feature
            probs = torch.softmax(head_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [N, L]
            entropy_masked = entropy * content_mask.float()
            head_mean = (entropy_masked.sum(dim=1) / content_lengths).numpy()
            head_feats_list.append(head_mean)

    # Digit head entropy
    if "digit_logits" in logits_data:
        digit_logits = logits_data["digit_logits"]  # [N, L, D, 10]
        probs = torch.softmax(digit_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [N, L, D]
        entropy_mean = entropy.mean(dim=-1)  # [N, L] mean over digit positions
        entropy_masked = entropy_mean * content_mask.float()
        digit_mean = (entropy_masked.sum(dim=1) / content_lengths).numpy()
        head_feats_list.append(digit_mean)

    head_feats = np.stack(head_feats_list, axis=1)  # [N, 2-4]

    # --- Concatenate all features ---
    features = np.concatenate([encoder_feats, nll_stats, rank_feats, head_feats], axis=1)

    logger.info(f"  Feature shape: {features.shape} "
                f"(encoder={encoder_feats.shape[1]}, nll={nll_stats.shape[1]}, "
                f"rank={rank_feats.shape[1]}, head={head_feats.shape[1]})")

    return features.astype(np.float32), features.shape[1]


def extract_attack_features(
    fold: int,
    attack_name: str,
    attacked_tokens: torch.Tensor,
    orig_indices: torch.Tensor,
    base_features: np.ndarray,
) -> np.ndarray:
    """Extract features for attacked windows.

    For encoder-based features, we reuse the original window's encoder memory
    (attacks only modify the G-code tokens, not sensor data).
    For decoder-based features, we recompute NLL/rank using attacked tokens
    against the original logits.

    Args:
        fold: fold number
        attack_name: attack identifier
        attacked_tokens: [N_atk, L] attacked token sequences
        orig_indices: [N_atk] indices into original test set
        base_features: [N_total, D] features for all normal windows

    Returns:
        attack_features: [N_atk, D] feature matrix for attacked windows
    """
    logits_data = load_cached_logits(fold)
    targets_data = load_cached_targets(fold)

    legacy_logits = logits_data["legacy_logits"]
    padding_mask = targets_data["padding_mask"]
    lengths = targets_data["lengths"]

    # Start from the normal features for these windows (encoder memory unchanged)
    attack_feats = base_features[orig_indices.numpy()].copy()

    # Recompute NLL-based features using attacked tokens
    attack_logits = legacy_logits[orig_indices]
    attack_mask = padding_mask[orig_indices]
    attack_lengths = lengths[orig_indices]

    nll_attack = compute_per_token_nll(attack_logits, attacked_tokens, attack_mask)
    content_mask = (attacked_tokens != PAD_ID) & (attacked_tokens != EOS_ID)
    content_lengths = content_mask.sum(dim=1).float().clamp(min=1)

    # Window scores
    atk_window = compute_window_scores(nll_attack, attack_lengths, attacked_tokens)
    s_mean = atk_window["S_mean"].numpy()
    s_max = atk_window["S_max"].numpy()

    # NLL std
    nll_masked = nll_attack * content_mask.float()
    nll_mean_per = nll_masked.sum(dim=1) / content_lengths
    nll_var = ((nll_masked - nll_mean_per.unsqueeze(1)) ** 2 * content_mask.float()).sum(dim=1) / content_lengths
    s_std = torch.sqrt(nll_var).numpy()

    # Rank scores for attacked tokens
    vocab = load_vocab()
    try:
        rank_scores = compute_rank_scores(attack_logits, attacked_tokens, vocab["vocab"])
        s_rank_mean = rank_scores["S_rank_mean"].numpy()
        s_rank_max = rank_scores["S_rank_max"].numpy()
    except Exception:
        s_rank_mean = np.zeros(len(orig_indices))
        s_rank_max = np.zeros(len(orig_indices))

    # Overwrite NLL and rank features (columns 256 onward)
    # encoder_feats = [:256], nll_stats = [256:259], rank = [259:261]
    enc_dim = 256
    attack_feats[:, enc_dim] = s_mean
    attack_feats[:, enc_dim + 1] = s_max
    attack_feats[:, enc_dim + 2] = s_std
    attack_feats[:, enc_dim + 3] = s_rank_mean
    attack_feats[:, enc_dim + 4] = s_rank_max

    # Head features stay from normal (they depend on model distribution, not target)
    # Actually, for attacked tokens the NLL changes but head entropy stays same
    # since entropy only depends on logits, not targets. Keep as-is.

    return attack_feats


# ============================================================
# Attack Loading
# ============================================================

def load_all_attacks(fold: int) -> Dict[str, dict]:
    """Load all pre-generated attacks for a fold.

    Returns dict mapping attack_name -> {attacked_tokens, orig_indices, n_attacks}
    """
    attacks = {}
    attack_dir = ATTACK_DIR

    # A1: Command swap
    a1_path = attack_dir / "attack_A1_command_swap.pt"
    if a1_path.exists():
        a1 = torch.load(a1_path, map_location="cpu", weights_only=False)
        if fold in a1["fold_attacks"] and a1["fold_attacks"][fold]["n_attacks"] > 0:
            attacks["A1_command_swap"] = a1["fold_attacks"][fold]

    # A2: Coordinate tampering (use largest delta variants)
    a2_path = attack_dir / "attack_A2_coord_graded.pt"
    if a2_path.exists():
        a2 = torch.load(a2_path, map_location="cpu", weights_only=False)
        for key, data in a2.get("attacks_by_delta_axis", {}).items():
            fold_data = data["fold_attacks"].get(fold, {})
            if fold_data.get("n_attacks", 0) > 0:
                attacks[f"A2_{key}"] = fold_data

    # A3: Param swap
    a3_path = attack_dir / "attack_A3_param_swap.pt"
    if a3_path.exists():
        a3 = torch.load(a3_path, map_location="cpu", weights_only=False)
        if fold in a3["fold_attacks"] and a3["fold_attacks"][fold]["n_attacks"] > 0:
            attacks["A3_param_swap"] = a3["fold_attacks"][fold]

    # A4: Cross-operation swap
    a4_path = attack_dir / "attack_A4_cross_op.pt"
    if a4_path.exists():
        a4 = torch.load(a4_path, map_location="cpu", weights_only=False)
        if fold in a4["fold_attacks"] and a4["fold_attacks"][fold]["n_attacks"] > 0:
            attacks["A4_cross_op"] = a4["fold_attacks"][fold]

    return attacks


def get_main_attack_keys(attacks: Dict[str, dict]) -> List[str]:
    """Select representative attacks (one per category, largest delta for A2)."""
    main = []
    if "A1_command_swap" in attacks:
        main.append("A1_command_swap")

    # For A2, pick largest-delta variant per axis
    a2_keys = [k for k in attacks if k.startswith("A2_")]
    if a2_keys:
        # Group by axis, pick largest delta
        by_axis = {}
        for k in a2_keys:
            # e.g. "A2_X_delta_0.05"
            parts = k.split("_")
            axis = parts[1]
            delta = float(parts[-1])
            if axis not in by_axis or delta > by_axis[axis][1]:
                by_axis[axis] = (k, delta)
        main.extend([v[0] for v in by_axis.values()])

    if "A3_param_swap" in attacks:
        main.append("A3_param_swap")
    if "A4_cross_op" in attacks:
        main.append("A4_cross_op")

    return sorted(main)


# ============================================================
# Dataset Construction
# ============================================================

def build_balanced_dataset(
    normal_feats: np.ndarray,
    attack_feats_dict: Dict[str, np.ndarray],
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a balanced binary dataset: 50% normal, 50% attack.

    Returns:
        X: [2*N, D] features
        y: [2*N] labels (0=normal, 1=attack)
        attack_ids: [2*N] attack type index (-1 for normal)
    """
    rng = np.random.RandomState(seed)

    # Pool all attack features
    all_attack = []
    all_attack_ids = []
    for i, (name, feats) in enumerate(sorted(attack_feats_dict.items())):
        all_attack.append(feats)
        all_attack_ids.extend([i] * len(feats))

    if not all_attack:
        logger.warning("No attack features available")
        return normal_feats, np.zeros(len(normal_feats)), np.full(len(normal_feats), -1)

    attack_pool = np.concatenate(all_attack, axis=0)
    attack_ids_pool = np.array(all_attack_ids)

    n_attack = len(attack_pool)
    n_normal = len(normal_feats)

    # Balance: subsample the larger set
    if n_attack >= n_normal:
        # Subsample attacks to match normals
        idx = rng.choice(n_attack, n_normal, replace=False)
        attack_selected = attack_pool[idx]
        attack_ids_selected = attack_ids_pool[idx]
        normal_selected = normal_feats
    else:
        # Subsample normals to match attacks
        idx = rng.choice(n_normal, n_attack, replace=False)
        normal_selected = normal_feats[idx]
        attack_selected = attack_pool
        attack_ids_selected = attack_ids_pool

    X = np.concatenate([normal_selected, attack_selected], axis=0)
    y = np.concatenate([np.zeros(len(normal_selected)), np.ones(len(attack_selected))])
    aids = np.concatenate([np.full(len(normal_selected), -1), attack_ids_selected])

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm], aids[perm]


def build_per_attack_datasets(
    normal_feats: np.ndarray,
    attack_feats_dict: Dict[str, np.ndarray],
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build per-attack-type balanced datasets for evaluation.

    Returns dict: attack_name -> (X, y) where y has 0=normal, 1=attack
    """
    rng = np.random.RandomState(seed)
    datasets = {}
    for name, atk_feats in sorted(attack_feats_dict.items()):
        n_atk = len(atk_feats)
        if n_atk == 0:
            continue
        n_norm = min(len(normal_feats), n_atk * 2)  # allow slight imbalance
        idx = rng.choice(len(normal_feats), n_norm, replace=n_norm > len(normal_feats))
        X = np.concatenate([normal_feats[idx], atk_feats], axis=0)
        y = np.concatenate([np.zeros(n_norm), np.ones(n_atk)])
        perm = rng.permutation(len(X))
        datasets[name] = (X[perm], y[perm])
    return datasets


# ============================================================
# MLP Classifier (PyTorch)
# ============================================================

class AnomalyMLP(nn.Module):
    """Binary anomaly classifier on extracted features."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [B]

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    lr: float = 1e-4,
    epochs: int = 200,
    batch_size: int = 64,
    patience: int = 20,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[AnomalyMLP, dict]:
    """Train MLP with early stopping on validation AUROC.

    Returns:
        model: trained AnomalyMLP
        history: dict with training metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = AnomalyMLP(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    # Normalize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    train_ds = TensorDataset(
        torch.tensor(X_train_norm, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)
    y_val_t = y_val

    best_auroc = 0.0
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_auroc": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # Validation AUROC
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val_t)).cpu().numpy()
        try:
            from sklearn.metrics import roc_auc_score
            val_auroc = roc_auc_score(y_val_t, val_probs)
        except ValueError:
            val_auroc = 0.5
        history["val_auroc"].append(val_auroc)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val_auroc={val_auroc:.4f}")

        if wait >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1} (best AUROC={best_auroc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Store normalization params
    model._norm_mean = mean
    model._norm_std = std

    history["best_auroc"] = best_auroc
    return model, history


def evaluate_mlp(
    model: AnomalyMLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
) -> dict:
    """Evaluate trained MLP on test data."""
    mean = model._norm_mean
    std = model._norm_std
    X_norm = (X_test - mean) / std

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            model(torch.tensor(X_norm, dtype=torch.float32).to(device))
        ).cpu().numpy()

    normal_scores = probs[y_test == 0]
    attack_scores = probs[y_test == 1]

    if len(normal_scores) == 0 or len(attack_scores) == 0:
        return {"auroc": 0.5, "n_normal": len(normal_scores), "n_attack": len(attack_scores)}

    try:
        auroc_info = bootstrap_auroc_ci(normal_scores, attack_scores, n_bootstrap=2000)
    except (ValueError, ZeroDivisionError):
        auroc_info = {"auroc": 0.5, "ci_lower": 0.0, "ci_upper": 1.0, "std": 0.0}

    det_01 = compute_detection_at_fpr(normal_scores, attack_scores, 0.01)
    det_05 = compute_detection_at_fpr(normal_scores, attack_scores, 0.05)
    d = cohens_d(attack_scores, normal_scores)

    return {
        "auroc": auroc_info["auroc"],
        "auroc_ci": [auroc_info["ci_lower"], auroc_info["ci_upper"]],
        "detection_at_fpr_01": det_01,
        "detection_at_fpr_05": det_05,
        "cohens_d": d,
        "n_normal": len(normal_scores),
        "n_attack": len(attack_scores),
    }


# ============================================================
# Sklearn Classifiers
# ============================================================

def train_sklearn_classifier(
    clf_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
):
    """Train a sklearn classifier. Returns (model, clf_name)."""
    if clf_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        return clf

    elif clf_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("XGBoost not available, skipping")
            return None
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=5,
            scale_pos_weight=1.0,
            random_state=seed,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        clf.fit(X_train, y_train)
        return clf

    else:
        raise ValueError(f"Unknown classifier: {clf_name}")


def evaluate_sklearn(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate a sklearn classifier."""
    probs = clf.predict_proba(X_test)[:, 1]

    normal_scores = probs[y_test == 0]
    attack_scores = probs[y_test == 1]

    if len(normal_scores) == 0 or len(attack_scores) == 0:
        return {"auroc": 0.5, "n_normal": len(normal_scores), "n_attack": len(attack_scores)}

    try:
        auroc_info = bootstrap_auroc_ci(normal_scores, attack_scores, n_bootstrap=2000)
    except (ValueError, ZeroDivisionError):
        auroc_info = {"auroc": 0.5, "ci_lower": 0.0, "ci_upper": 1.0, "std": 0.0}

    det_01 = compute_detection_at_fpr(normal_scores, attack_scores, 0.01)
    det_05 = compute_detection_at_fpr(normal_scores, attack_scores, 0.05)
    d = cohens_d(attack_scores, normal_scores)

    return {
        "auroc": auroc_info["auroc"],
        "auroc_ci": [auroc_info["ci_lower"], auroc_info["ci_upper"]],
        "detection_at_fpr_01": det_01,
        "detection_at_fpr_05": det_05,
        "cohens_d": d,
        "n_normal": len(normal_scores),
        "n_attack": len(attack_scores),
    }


# ============================================================
# Per-Fold Training and Evaluation
# ============================================================

def run_fold(fold: int, output_dir: Path, seed: int = 42) -> dict:
    """Train and evaluate all classifiers for one fold.

    Uses leave-one-fold-out: train on 4 folds, test on held-out fold.
    For the MLP, splits the training folds into train/val (80/20).
    """
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Fold {fold} ===")

    # --- Extract features and build datasets for all folds ---
    train_folds = [f for f in FOLDS if f != fold]
    test_fold = fold

    # Test features
    test_feats, n_dim = extract_features(test_fold)
    test_attacks = load_all_attacks(test_fold)
    main_attack_keys = get_main_attack_keys(test_attacks)
    logger.info(f"  Test: {len(test_feats)} windows, attacks: {list(test_attacks.keys())}")

    # Build attack features for test fold
    test_attack_feats = {}
    for aname in test_attacks:
        atk = test_attacks[aname]
        try:
            af = extract_attack_features(
                test_fold, aname,
                atk["attacked_tokens"], atk["orig_indices"],
                test_feats,
            )
            test_attack_feats[aname] = af
        except Exception as e:
            logger.warning(f"  Failed to extract features for {aname}: {e}")

    # Training features: pool from all train folds
    train_feats_list = []
    train_attack_feats_list = {}

    for tf in train_folds:
        feats, _ = extract_features(tf)
        train_feats_list.append(feats)

        tf_attacks = load_all_attacks(tf)
        for aname, atk in tf_attacks.items():
            try:
                af = extract_attack_features(
                    tf, aname,
                    atk["attacked_tokens"], atk["orig_indices"],
                    feats,
                )
                if aname not in train_attack_feats_list:
                    train_attack_feats_list[aname] = []
                train_attack_feats_list[aname].append(af)
            except Exception as e:
                logger.warning(f"  Train fold {tf} attack {aname} failed: {e}")

    train_normal = np.concatenate(train_feats_list, axis=0)
    train_attacks_pooled = {
        k: np.concatenate(v, axis=0) for k, v in train_attack_feats_list.items() if v
    }

    logger.info(f"  Train: {len(train_normal)} normal windows, "
                f"{sum(len(v) for v in train_attacks_pooled.values())} attack windows")

    # Build balanced training set
    X_all, y_all, _ = build_balanced_dataset(
        train_normal, train_attacks_pooled, seed=seed
    )

    # Split into train/val for MLP (80/20)
    rng = np.random.RandomState(seed)
    n_total = len(X_all)
    n_val = max(int(n_total * 0.2), 10)
    perm = rng.permutation(n_total)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    # Replace NaN/Inf
    for arr in [X_train, X_val, test_feats]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=100.0, neginf=-100.0)
    for arr in test_attack_feats.values():
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=100.0, neginf=-100.0)

    fold_results = {"fold": fold, "n_test": len(test_feats)}
    classifier_results = {}

    # --- 1. MLP ---
    logger.info("  Training MLP...")
    try:
        mlp_model, mlp_history = train_mlp(
            X_train, y_train, X_val, y_val,
            input_dim=n_dim, lr=1e-4, epochs=200,
            batch_size=64, patience=20, seed=seed,
        )

        # Evaluate per attack type
        mlp_per_attack = {}
        for aname, af in sorted(test_attack_feats.items()):
            X_test = np.concatenate([test_feats, af], axis=0)
            y_test = np.concatenate([np.zeros(len(test_feats)), np.ones(len(af))])
            np.nan_to_num(X_test, copy=False, nan=0.0, posinf=100.0, neginf=-100.0)
            result = evaluate_mlp(mlp_model, X_test, y_test)
            mlp_per_attack[aname] = result
            logger.info(f"    MLP {aname}: AUROC={result['auroc']:.4f}")

        # Also evaluate on combined test set
        if test_attack_feats:
            all_atk = np.concatenate(list(test_attack_feats.values()), axis=0)
            X_test_all = np.concatenate([test_feats, all_atk], axis=0)
            y_test_all = np.concatenate([np.zeros(len(test_feats)), np.ones(len(all_atk))])
            np.nan_to_num(X_test_all, copy=False, nan=0.0, posinf=100.0, neginf=-100.0)
            mlp_combined = evaluate_mlp(mlp_model, X_test_all, y_test_all)
        else:
            mlp_combined = {"auroc": 0.5}

        classifier_results["mlp"] = {
            "per_attack": mlp_per_attack,
            "combined": mlp_combined,
            "best_val_auroc": mlp_history["best_auroc"],
        }
    except Exception as e:
        logger.error(f"  MLP training failed: {e}")
        classifier_results["mlp"] = {"error": str(e)}

    # --- 2. Random Forest ---
    logger.info("  Training Random Forest...")
    try:
        rf = train_sklearn_classifier("random_forest", X_train, y_train, seed=seed)
        rf_per_attack = {}
        for aname, af in sorted(test_attack_feats.items()):
            X_test = np.concatenate([test_feats, af], axis=0)
            y_test = np.concatenate([np.zeros(len(test_feats)), np.ones(len(af))])
            np.nan_to_num(X_test, copy=False, nan=0.0, posinf=100.0, neginf=-100.0)
            result = evaluate_sklearn(rf, X_test, y_test)
            rf_per_attack[aname] = result
            logger.info(f"    RF {aname}: AUROC={result['auroc']:.4f}")

        if test_attack_feats:
            all_atk = np.concatenate(list(test_attack_feats.values()), axis=0)
            X_test_all = np.concatenate([test_feats, all_atk], axis=0)
            y_test_all = np.concatenate([np.zeros(len(test_feats)), np.ones(len(all_atk))])
            np.nan_to_num(X_test_all, copy=False, nan=0.0, posinf=100.0, neginf=-100.0)
            rf_combined = evaluate_sklearn(rf, X_test_all, y_test_all)
        else:
            rf_combined = {"auroc": 0.5}

        # Feature importances
        feat_imp = rf.feature_importances_
        top_k = min(20, len(feat_imp))
        top_idx = np.argsort(feat_imp)[-top_k:][::-1]
        feat_names = (
            [f"enc_{i}" for i in range(256)]
            + ["nll_mean", "nll_max", "nll_std"]
            + ["rank_mean", "rank_max"]
            + [f"head_{i}" for i in range(n_dim - 261)]
        )
        top_features = [
            {"feature": feat_names[i] if i < len(feat_names) else f"f{i}",
             "importance": float(feat_imp[i])}
            for i in top_idx
        ]

        classifier_results["random_forest"] = {
            "per_attack": rf_per_attack,
            "combined": rf_combined,
            "top_features": top_features,
        }
    except Exception as e:
        logger.error(f"  Random Forest failed: {e}")
        classifier_results["random_forest"] = {"error": str(e)}

    # --- 3. XGBoost ---
    logger.info("  Training XGBoost...")
    try:
        xgb = train_sklearn_classifier("xgboost", X_train, y_train, seed=seed)
        if xgb is not None:
            xgb_per_attack = {}
            for aname, af in sorted(test_attack_feats.items()):
                X_test = np.concatenate([test_feats, af], axis=0)
                y_test = np.concatenate([np.zeros(len(test_feats)), np.ones(len(af))])
                np.nan_to_num(X_test, copy=False, nan=0.0, posinf=100.0, neginf=-100.0)
                result = evaluate_sklearn(xgb, X_test, y_test)
                xgb_per_attack[aname] = result
                logger.info(f"    XGB {aname}: AUROC={result['auroc']:.4f}")

            if test_attack_feats:
                all_atk = np.concatenate(list(test_attack_feats.values()), axis=0)
                X_test_all = np.concatenate([test_feats, all_atk], axis=0)
                y_test_all = np.concatenate([np.zeros(len(test_feats)), np.ones(len(all_atk))])
                np.nan_to_num(X_test_all, copy=False, nan=0.0, posinf=100.0, neginf=-100.0)
                xgb_combined = evaluate_sklearn(xgb, X_test_all, y_test_all)
            else:
                xgb_combined = {"auroc": 0.5}

            classifier_results["xgboost"] = {
                "per_attack": xgb_per_attack,
                "combined": xgb_combined,
            }
        else:
            classifier_results["xgboost"] = {"error": "XGBoost not installed"}
    except Exception as e:
        logger.error(f"  XGBoost failed: {e}")
        classifier_results["xgboost"] = {"error": str(e)}

    fold_results["classifiers"] = classifier_results
    save_results(fold_results, fold_dir)
    return fold_results


# ============================================================
# Aggregation
# ============================================================

def aggregate(fold_results: list, output_dir: Path) -> dict:
    """Cross-fold aggregation of results."""
    clf_names = ["mlp", "random_forest", "xgboost"]
    agg = {}

    for clf_name in clf_names:
        clf_agg = {"combined_aurocs": [], "per_attack": {}}

        for fr in fold_results:
            clf_data = fr.get("classifiers", {}).get(clf_name, {})
            if "error" in clf_data:
                continue

            combined = clf_data.get("combined", {})
            if "auroc" in combined:
                clf_agg["combined_aurocs"].append(combined["auroc"])

            per_attack = clf_data.get("per_attack", {})
            for aname, metrics in per_attack.items():
                if aname not in clf_agg["per_attack"]:
                    clf_agg["per_attack"][aname] = []
                clf_agg["per_attack"][aname].append(metrics.get("auroc", 0.5))

        if clf_agg["combined_aurocs"]:
            clf_agg["mean_combined_auroc"] = float(np.mean(clf_agg["combined_aurocs"]))
            clf_agg["std_combined_auroc"] = float(np.std(clf_agg["combined_aurocs"]))

        for aname in clf_agg["per_attack"]:
            aurocs = clf_agg["per_attack"][aname]
            clf_agg["per_attack"][aname] = {
                "mean_auroc": float(np.mean(aurocs)),
                "std_auroc": float(np.std(aurocs)),
                "per_fold": aurocs,
            }

        agg[clf_name] = clf_agg

    # Find best classifier
    best_clf = None
    best_auroc = 0.0
    for clf_name, clf_agg in agg.items():
        auroc = clf_agg.get("mean_combined_auroc", 0.0)
        if auroc > best_auroc:
            best_auroc = auroc
            best_clf = clf_name

    results = {
        "per_classifier": agg,
        "best_classifier": best_clf,
        "best_combined_auroc": best_auroc,
        "n_folds": len(fold_results),
    }

    save_results(results, output_dir)
    return results


# ============================================================
# Figures
# ============================================================

def generate_figures(agg_results: dict, output_dir: Path):
    """Generate paper-ready figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping figures")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: AUROC per classifier, combined ---
    clf_names = []
    auroc_means = []
    auroc_stds = []
    for clf_name, clf_data in agg_results["per_classifier"].items():
        if "mean_combined_auroc" in clf_data:
            clf_names.append(clf_name.replace("_", " ").title())
            auroc_means.append(clf_data["mean_combined_auroc"])
            auroc_stds.append(clf_data.get("std_combined_auroc", 0))

    if clf_names:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(clf_names))
        bars = ax.bar(x, auroc_means, yerr=auroc_stds, capsize=5, color=["#4C72B0", "#55A868", "#C44E52"][:len(clf_names)])
        ax.set_ylabel("AUROC")
        ax.set_title("Anomaly Head: Combined AUROC by Classifier")
        ax.set_xticks(x)
        ax.set_xticklabels(clf_names)
        ax.set_ylim(0, 1.05)
        for i, (m, s) in enumerate(zip(auroc_means, auroc_stds)):
            ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=10)
        fig.tight_layout()
        save_figure(fig, fig_dir, "anomaly_head_combined_auroc.pdf")
        plt.close(fig)

    # --- Figure 2: Per-attack AUROC heatmap (best classifier) ---
    best_clf = agg_results.get("best_classifier")
    if best_clf and best_clf in agg_results["per_classifier"]:
        per_attack = agg_results["per_classifier"][best_clf].get("per_attack", {})
        # Filter to main attacks (skip per-delta A2 variants)
        main_attacks = {k: v for k, v in per_attack.items()
                        if not k.startswith("A2_") or "delta_1" in k or "delta_0.5" in k or "delta_0.1" in k}

        if main_attacks:
            names = sorted(main_attacks.keys())
            means = [main_attacks[n]["mean_auroc"] for n in names]
            stds = [main_attacks[n]["std_auroc"] for n in names]

            fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 4))
            x = np.arange(len(names))
            ax.bar(x, means, yerr=stds, capsize=3, color="#4C72B0")
            ax.set_ylabel("AUROC")
            ax.set_title(f"Anomaly Head ({best_clf}): Per-Attack AUROC")
            ax.set_xticks(x)
            ax.set_xticklabels([n.replace("_", "\n") for n in names], rotation=45, ha="right", fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
            ax.legend()
            fig.tight_layout()
            save_figure(fig, fig_dir, "anomaly_head_per_attack.pdf")
            plt.close(fig)

    # --- Figure 3: Comparison with individual scorers (if exp01 results exist) ---
    exp01_path = BASE_DIR / "exp01_nll_scoring" / "aggregate_results.json"
    if exp01_path.exists() and best_clf:
        try:
            with open(exp01_path) as f:
                exp01 = json.load(f)

            # Compare on main attack types
            per_attack_head = agg_results["per_classifier"][best_clf].get("per_attack", {})
            exp01_attacks = exp01.get("per_attack", {})

            common_attacks = sorted(set(per_attack_head.keys()) & set(exp01_attacks.keys()))
            if common_attacks:
                fig, ax = plt.subplots(figsize=(max(8, len(common_attacks) * 1.2), 5))
                x = np.arange(len(common_attacks))
                width = 0.35

                # Anomaly head
                head_aurocs = [per_attack_head[a]["mean_auroc"] for a in common_attacks]
                # Best individual scorer
                exp01_aurocs = []
                for a in common_attacks:
                    best_s = max(
                        exp01_attacks[a].get(s, {}).get("mean_auroc", 0)
                        for s in ["S_mean", "S_max", "S_sum", "perplexity"]
                    )
                    exp01_aurocs.append(best_s)

                ax.bar(x - width / 2, exp01_aurocs, width, label="Best NLL Scorer", color="#4C72B0")
                ax.bar(x + width / 2, head_aurocs, width, label=f"Anomaly Head ({best_clf})", color="#C44E52")
                ax.set_ylabel("AUROC")
                ax.set_title("Anomaly Head vs Individual NLL Scorers")
                ax.set_xticks(x)
                ax.set_xticklabels([a.replace("_", "\n") for a in common_attacks], rotation=45, ha="right", fontsize=8)
                ax.set_ylim(0, 1.05)
                ax.legend()
                fig.tight_layout()
                save_figure(fig, fig_dir, "anomaly_head_vs_nll.pdf")
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to generate comparison figure: {e}")

    logger.info(f"Figures saved to {fig_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = get_default_parser(EXP_NAME, EXP_ID)
    parser.add_argument("--lr", type=float, default=1e-4, help="MLP learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="MLP max epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="MLP batch size")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--skip-mlp", action="store_true", help="Skip MLP training")
    parser.add_argument("--skip-rf", action="store_true", help="Skip Random Forest")
    parser.add_argument("--skip-xgb", action="store_true", help="Skip XGBoost")
    args = parser.parse_args()

    setup_logging("exp20_anomaly_head")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")
    logger.info(f"Folds: {args.folds}")

    fold_results = []
    for fold in args.folds:
        result = run_fold(fold, output_dir, seed=args.seed)
        fold_results.append(result)

    agg = aggregate(fold_results, output_dir)
    generate_figures(agg, output_dir)

    # Summary
    best_clf = agg.get("best_classifier", "none")
    best_auroc = agg.get("best_combined_auroc", 0.0)
    finding = f"Best: {best_clf} at {best_auroc:.3f} combined AUROC"

    append_to_summary(
        exp_id=EXP_ID,
        name=EXP_NAME,
        status="done",
        auroc=f"{best_auroc:.3f}",
        finding=finding,
        output_dir=BASE_DIR,
    )

    logger.info(f"Experiment {EXP_ID} complete. {finding}")


if __name__ == "__main__":
    main()
