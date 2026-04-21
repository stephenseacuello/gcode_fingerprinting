#!/usr/bin/env python3
"""Experiment 19: Learned Ensemble Scorer for Anomaly Detection

Combines three base anomaly signals — NLL (S_mean), Rank (S_rank_mean),
and Mahalanobis embedding distance — using (a) logistic regression and
(b) grid-searched linear combination weights.  Evaluates on held-out test
windows for each of the 5 CV folds and reports AUROC per attack type.

Priority 1 in the anomaly-detection improvement plan.
"""

import argparse
import itertools
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import (
    ATTACK_DIR,
    BASE_DIR,
    CACHE_DIR,
    FOLDS,
    V8_CACHE_DIR,
    load_cached_logits,
    load_cached_targets,
    load_cached_embeddings,
    load_vocab,
    compute_per_token_nll,
    compute_window_scores,
    compute_rank_scores,
    compute_auroc,
    bootstrap_auroc_ci,
    compute_detection_at_fpr,
    cohens_d,
    save_results,
    save_figure,
    append_to_summary,
    setup_logging,
    get_default_parser,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Embedding helpers (adapted from exp11)
# ------------------------------------------------------------------

def _attention_pool(memory: torch.Tensor) -> np.ndarray:
    """Mean-pool encoder memory across time to get fingerprint vectors."""
    return memory.mean(dim=1).numpy()


def _fit_class_gaussians(fingerprints: np.ndarray, labels: np.ndarray):
    """Fit class-conditional Gaussians with tied covariance.

    Returns:
        centroids: dict  class_id -> [D] mean
        cov_inv:   [D, D] inverse of tied covariance
    """
    classes = np.unique(labels)
    D = fingerprints.shape[1]

    centroids = {}
    cov_sum = np.zeros((D, D))
    n_total = 0

    for c in classes:
        mask = labels == c
        class_data = fingerprints[mask]
        centroids[int(c)] = class_data.mean(axis=0)
        centered = class_data - centroids[int(c)]
        cov_sum += centered.T @ centered
        n_total += len(class_data)

    cov = cov_sum / max(n_total - len(classes), 1)
    cov += np.eye(D) * 1e-5
    cov_inv = np.linalg.inv(cov)
    return centroids, cov_inv


def _mahalanobis_scores_vectorized(
    fingerprints: np.ndarray, centroids: dict, cov_inv: np.ndarray
) -> np.ndarray:
    """Min Mahalanobis distance to any class centroid."""
    N = fingerprints.shape[0]
    scores = np.full(N, np.inf)
    for _c, mu in centroids.items():
        diff = fingerprints - mu[np.newaxis, :]
        m_dist = np.sqrt(np.sum((diff @ cov_inv) * diff, axis=1))
        scores = np.minimum(scores, m_dist)
    return scores


# ------------------------------------------------------------------
# Score normalisation
# ------------------------------------------------------------------

def _normalise_01(arr: np.ndarray) -> np.ndarray:
    """Scale array to [0, 1].  Handles constant arrays gracefully."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ------------------------------------------------------------------
# Attack loading helpers
# ------------------------------------------------------------------

def _load_all_attacks(fold: int):
    """Load all attack types for a given fold.

    Returns list of (attack_name, attacked_tokens, orig_indices).
    """
    attack_dir = ATTACK_DIR
    attacks = []

    # A1 command swap
    a1_path = attack_dir / "attack_A1_command_swap.pt"
    if a1_path.exists():
        a1 = torch.load(a1_path, weights_only=False)
        fold_data = a1.get("fold_attacks", {}).get(fold, {})
        if fold_data.get("n_attacks", 0) > 0:
            attacks.append(("A1_command_swap",
                            fold_data["attacked_tokens"],
                            fold_data["orig_indices"]))

    # A2 coordinate graded (multiple delta/axis combos)
    a2_path = attack_dir / "attack_A2_coord_graded.pt"
    if a2_path.exists():
        a2 = torch.load(a2_path, weights_only=False)
        for key, data in a2.get("attacks_by_delta_axis", {}).items():
            fold_data = data.get("fold_attacks", {}).get(fold, {})
            if fold_data.get("n_attacks", 0) > 0:
                attacks.append((f"A2_{key}",
                                fold_data["attacked_tokens"],
                                fold_data["orig_indices"]))

    # A3 param swap
    a3_path = attack_dir / "attack_A3_param_swap.pt"
    if a3_path.exists():
        a3 = torch.load(a3_path, weights_only=False)
        fold_data = a3.get("fold_attacks", {}).get(fold, {})
        if fold_data.get("n_attacks", 0) > 0:
            attacks.append(("A3_param_swap",
                            fold_data["attacked_tokens"],
                            fold_data["orig_indices"]))

    # A4 cross-op swap
    a4_path = attack_dir / "attack_A4_cross_op.pt"
    if a4_path.exists():
        a4 = torch.load(a4_path, weights_only=False)
        fold_data = a4.get("fold_attacks", {}).get(fold, {})
        if fold_data.get("n_attacks", 0) > 0:
            attacks.append(("A4_cross_op",
                            fold_data["attacked_tokens"],
                            fold_data["orig_indices"]))

    return attacks


# ------------------------------------------------------------------
# Per-fold pipeline
# ------------------------------------------------------------------

def _compute_base_scores(logits, targets_data, emb_data, vocab):
    """Compute three base anomaly scores for every test window.

    Returns:
        nll_scores:  np.ndarray [N]
        rank_scores: np.ndarray [N]
        maha_scores: np.ndarray [N]
        centroids:   dict (needed to score attack embeddings)
        cov_inv:     np.ndarray (needed to score attack embeddings)
    """
    target_tokens = targets_data["target_tokens"]
    padding_mask = targets_data["padding_mask"]
    lengths = targets_data["lengths"]

    legacy_logits = logits["legacy_logits"]

    # 1. NLL S_mean
    nll = compute_per_token_nll(legacy_logits, target_tokens, padding_mask)
    window = compute_window_scores(nll, lengths)
    nll_scores = window["S_mean"].numpy()

    # 2. Rank S_rank_mean
    vocab_map = vocab["vocab"]
    rank = compute_rank_scores(legacy_logits, target_tokens, vocab_map)
    rank_scores = rank["S_rank_mean"].numpy()

    # 3. Mahalanobis
    train_memory = emb_data["train_memory"]
    test_memory = emb_data["test_memory"]
    train_op_pred = emb_data["train_op_pred"]

    train_fp = _attention_pool(train_memory)
    test_fp = _attention_pool(test_memory)

    train_labels = train_op_pred.numpy() if isinstance(train_op_pred, torch.Tensor) else train_op_pred
    centroids, cov_inv = _fit_class_gaussians(train_fp, train_labels)
    maha_scores = _mahalanobis_scores_vectorized(test_fp, centroids, cov_inv)

    return nll_scores, rank_scores, maha_scores, centroids, cov_inv


def _compute_attack_scores(
    attack_tokens, orig_indices,
    logits, targets_data, emb_data, vocab,
    normal_maha, centroids, cov_inv,
):
    """Compute three base scores for attacked windows.

    For NLL and rank the attack changes tokens against the same logits.
    For Mahalanobis the encoder sees original sensor data, so the score
    is the same as the normal score for those windows (token-level attacks
    don't alter the encoder representation).
    """
    target_tokens = targets_data["target_tokens"]
    padding_mask = targets_data["padding_mask"]
    lengths = targets_data["lengths"]
    legacy_logits = logits["legacy_logits"]

    attack_logits = legacy_logits[orig_indices]
    attack_mask = padding_mask[orig_indices]
    attack_lengths = lengths[orig_indices]

    # 1. NLL
    nll = compute_per_token_nll(attack_logits, attack_tokens, attack_mask)
    window = compute_window_scores(nll, attack_lengths)
    atk_nll = window["S_mean"].numpy()

    # 2. Rank
    vocab_map = vocab["vocab"]
    rank = compute_rank_scores(attack_logits, attack_tokens, vocab_map)
    atk_rank = rank["S_rank_mean"].numpy()

    # 3. Mahalanobis — same as normal (encoder unchanged by token attack)
    atk_maha = normal_maha[orig_indices.numpy() if isinstance(orig_indices, torch.Tensor) else orig_indices]

    return atk_nll, atk_rank, atk_maha


# ------------------------------------------------------------------
# Ensemble methods
# ------------------------------------------------------------------

def _logistic_regression_ensemble(X_train, y_train, X_test, y_test):
    """Train LogisticRegression and return AUROC on test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    clf.fit(X_train_s, y_train)
    proba = clf.predict_proba(X_test_s)[:, 1]

    auroc = compute_auroc(
        proba[y_test == 0],
        proba[y_test == 1],
    )
    return auroc, clf.coef_[0].tolist(), proba


def _grid_search_alpha(
    normal_scores_3: np.ndarray,
    attack_scores_3: np.ndarray,
    resolution: int = 11,
):
    """Grid-search over alpha weights for S = a1*NLL + a2*Rank + a3*Maha.

    Each score column should already be normalised to [0, 1].
    Returns best (alpha tuple, auroc).
    """
    best_auroc = 0.0
    best_alpha = (1.0, 0.0, 0.0)

    alphas = np.linspace(0, 1, resolution)
    for a1 in alphas:
        for a2 in alphas:
            a3 = 1.0 - a1 - a2
            if a3 < -1e-9 or a3 > 1.0 + 1e-9:
                continue
            a3 = max(a3, 0.0)
            normal_combo = a1 * normal_scores_3[:, 0] + a2 * normal_scores_3[:, 1] + a3 * normal_scores_3[:, 2]
            attack_combo = a1 * attack_scores_3[:, 0] + a2 * attack_scores_3[:, 1] + a3 * attack_scores_3[:, 2]
            try:
                auroc = compute_auroc(normal_combo, attack_combo)
            except ValueError:
                continue
            if auroc > best_auroc:
                best_auroc = auroc
                best_alpha = (round(float(a1), 3), round(float(a2), 3), round(float(a3), 3))

    return best_alpha, best_auroc


# ------------------------------------------------------------------
# Fold runner
# ------------------------------------------------------------------

def run_fold(fold: int, output_dir: Path, use_v8: bool = False) -> dict:
    """Run the learned ensemble experiment for one fold."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Fold {fold} (v8={use_v8}) ---")

    # Load cached data
    logits = load_cached_logits(fold)
    targets_data = load_cached_targets(fold)
    emb_data = load_cached_embeddings(fold)
    vocab = load_vocab()

    # Compute base scores on normal test windows
    nll_scores, rank_scores, maha_scores, centroids, cov_inv = _compute_base_scores(
        logits, targets_data, emb_data, vocab
    )

    logger.info(
        f"  Normal base scores  NLL: {nll_scores.mean():.4f}  "
        f"Rank: {rank_scores.mean():.4f}  Maha: {maha_scores.mean():.4f}"
    )

    # Load attacks
    attacks = _load_all_attacks(fold)
    if not attacks:
        logger.warning(f"  No attacks found for fold {fold}")
        return {"fold": fold, "attacks": {}}

    # For each attack type, compute scores and evaluate ensemble
    fold_results = {"fold": fold, "attacks": {}}

    # Accumulate all attack data for a combined logistic regression
    all_normal_features = []
    all_attack_features = []

    for attack_name, attacked_tokens, orig_indices in attacks:
        logger.info(f"  Attack: {attack_name} ({len(orig_indices)} windows)")

        atk_nll, atk_rank, atk_maha = _compute_attack_scores(
            attacked_tokens, orig_indices,
            logits, targets_data, emb_data, vocab,
            maha_scores, centroids, cov_inv,
        )

        idx = orig_indices.numpy() if isinstance(orig_indices, torch.Tensor) else orig_indices
        norm_nll = nll_scores[idx]
        norm_rank = rank_scores[idx]
        norm_maha = maha_scores[idx]

        # --- Individual scorer AUROCs ---
        individual = {}
        for name_s, ns, ats in [
            ("NLL_S_mean", norm_nll, atk_nll),
            ("Rank_S_rank_mean", norm_rank, atk_rank),
            ("Mahalanobis", norm_maha, atk_maha),
        ]:
            try:
                auroc = compute_auroc(ns, ats)
            except ValueError:
                auroc = 0.5
            individual[name_s] = {"auroc": float(auroc)}

        # --- Grid-searched linear combination ---
        # Normalise to [0,1] using combined (normal+attack) stats
        all_nll = np.concatenate([norm_nll, atk_nll])
        all_rank = np.concatenate([norm_rank, atk_rank])
        all_maha_combined = np.concatenate([norm_maha, atk_maha])

        nll_lo, nll_hi = all_nll.min(), all_nll.max()
        rank_lo, rank_hi = all_rank.min(), all_rank.max()
        maha_lo, maha_hi = all_maha_combined.min(), all_maha_combined.max()

        def _norm(arr, lo, hi):
            if hi - lo < 1e-12:
                return np.zeros_like(arr)
            return (arr - lo) / (hi - lo)

        norm_3 = np.column_stack([
            _norm(norm_nll, nll_lo, nll_hi),
            _norm(norm_rank, rank_lo, rank_hi),
            _norm(norm_maha, maha_lo, maha_hi),
        ])
        atk_3 = np.column_stack([
            _norm(atk_nll, nll_lo, nll_hi),
            _norm(atk_rank, rank_lo, rank_hi),
            _norm(atk_maha, maha_lo, maha_hi),
        ])

        best_alpha, grid_auroc = _grid_search_alpha(norm_3, atk_3)

        # --- Logistic regression ---
        X = np.vstack([norm_3, atk_3])
        y = np.concatenate([np.zeros(len(norm_3)), np.ones(len(atk_3))])

        # Train/test split: use first 60% as train, rest as test
        n_total = len(y)
        n_train = max(int(0.6 * n_total), 2)

        # Stratified split
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        rng = np.random.RandomState(42 + fold)
        rng.shuffle(idx_0)
        rng.shuffle(idx_1)

        n_train_0 = max(int(0.6 * len(idx_0)), 1)
        n_train_1 = max(int(0.6 * len(idx_1)), 1)

        train_idx = np.concatenate([idx_0[:n_train_0], idx_1[:n_train_1]])
        test_idx = np.concatenate([idx_0[n_train_0:], idx_1[n_train_1:]])

        if len(test_idx) < 2 or len(np.unique(y[test_idx])) < 2:
            # Not enough test data for evaluation — use full set
            lr_auroc = 0.5
            lr_coefs = [0.0, 0.0, 0.0]
        else:
            lr_auroc, lr_coefs, _ = _logistic_regression_ensemble(
                X[train_idx], y[train_idx], X[test_idx], y[test_idx]
            )

        # Bootstrap CI on the grid-searched combo (full data, no train/test split)
        combo_normal = best_alpha[0] * norm_3[:, 0] + best_alpha[1] * norm_3[:, 1] + best_alpha[2] * norm_3[:, 2]
        combo_attack = best_alpha[0] * atk_3[:, 0] + best_alpha[1] * atk_3[:, 1] + best_alpha[2] * atk_3[:, 2]
        try:
            ci_info = bootstrap_auroc_ci(combo_normal, combo_attack, n_bootstrap=2000)
        except (ValueError, ZeroDivisionError):
            ci_info = {"auroc": grid_auroc, "ci_lower": 0.0, "ci_upper": 1.0, "std": 0.0}

        # Detection rates at standard FPR thresholds
        try:
            det_01 = compute_detection_at_fpr(combo_normal, combo_attack, 0.01)
            det_05 = compute_detection_at_fpr(combo_normal, combo_attack, 0.05)
            det_10 = compute_detection_at_fpr(combo_normal, combo_attack, 0.10)
        except (ValueError, ZeroDivisionError):
            det_01 = det_05 = det_10 = 0.0

        try:
            d = cohens_d(combo_attack, combo_normal)
        except (ValueError, ZeroDivisionError):
            d = 0.0

        fold_results["attacks"][attack_name] = {
            "individual_aurocs": individual,
            "grid_search": {
                "best_alpha": list(best_alpha),
                "auroc": float(grid_auroc),
                "auroc_ci": [ci_info["ci_lower"], ci_info["ci_upper"]],
                "detection_at_fpr_01": det_01,
                "detection_at_fpr_05": det_05,
                "detection_at_fpr_10": det_10,
                "cohens_d": d,
            },
            "logistic_regression": {
                "auroc": float(lr_auroc),
                "coefs": lr_coefs,
            },
            "n_normal": len(norm_nll),
            "n_attack": len(atk_nll),
        }

        # Accumulate for combined LR
        all_normal_features.append(norm_3)
        all_attack_features.append(atk_3)

    # Combined logistic regression across all attack types
    if all_normal_features and all_attack_features:
        X_norm_all = np.vstack(all_normal_features)
        X_atk_all = np.vstack(all_attack_features)
        X_all = np.vstack([X_norm_all, X_atk_all])
        y_all = np.concatenate([np.zeros(len(X_norm_all)), np.ones(len(X_atk_all))])

        idx_0 = np.where(y_all == 0)[0]
        idx_1 = np.where(y_all == 1)[0]
        rng = np.random.RandomState(42 + fold)
        rng.shuffle(idx_0)
        rng.shuffle(idx_1)
        n_train_0 = max(int(0.6 * len(idx_0)), 1)
        n_train_1 = max(int(0.6 * len(idx_1)), 1)
        train_idx = np.concatenate([idx_0[:n_train_0], idx_1[:n_train_1]])
        test_idx = np.concatenate([idx_0[n_train_0:], idx_1[n_train_1:]])

        if len(test_idx) >= 2 and len(np.unique(y_all[test_idx])) >= 2:
            combined_auroc, combined_coefs, _ = _logistic_regression_ensemble(
                X_all[train_idx], y_all[train_idx], X_all[test_idx], y_all[test_idx]
            )
        else:
            combined_auroc = 0.5
            combined_coefs = [0.0, 0.0, 0.0]

        fold_results["combined_lr"] = {
            "auroc": float(combined_auroc),
            "coefs": combined_coefs,
            "n_normal": len(X_norm_all),
            "n_attack": len(X_atk_all),
        }

    save_results(fold_results, fold_dir, "ensemble_results.json")
    return fold_results


# ------------------------------------------------------------------
# Cross-fold aggregation
# ------------------------------------------------------------------

def aggregate(fold_results: list, output_dir: Path) -> dict:
    """Aggregate results across folds."""
    # Collect per-attack stats
    attack_types = set()
    for fr in fold_results:
        attack_types.update(fr.get("attacks", {}).keys())

    per_attack = {}
    for attack in sorted(attack_types):
        grid_aurocs = []
        lr_aurocs = []
        indiv = {"NLL_S_mean": [], "Rank_S_rank_mean": [], "Mahalanobis": []}
        best_alphas = []

        for fr in fold_results:
            atk_data = fr.get("attacks", {}).get(attack)
            if atk_data is None:
                continue
            grid_aurocs.append(atk_data["grid_search"]["auroc"])
            lr_aurocs.append(atk_data["logistic_regression"]["auroc"])
            best_alphas.append(atk_data["grid_search"]["best_alpha"])
            for key in indiv:
                if key in atk_data["individual_aurocs"]:
                    indiv[key].append(atk_data["individual_aurocs"][key]["auroc"])

        per_attack[attack] = {
            "grid_search": {
                "mean_auroc": float(np.mean(grid_aurocs)) if grid_aurocs else 0.0,
                "std_auroc": float(np.std(grid_aurocs)) if grid_aurocs else 0.0,
                "per_fold": grid_aurocs,
                "mean_alpha": np.mean(best_alphas, axis=0).tolist() if best_alphas else [],
            },
            "logistic_regression": {
                "mean_auroc": float(np.mean(lr_aurocs)) if lr_aurocs else 0.0,
                "std_auroc": float(np.std(lr_aurocs)) if lr_aurocs else 0.0,
                "per_fold": lr_aurocs,
            },
            "individual": {
                key: {
                    "mean_auroc": float(np.mean(vals)) if vals else 0.0,
                    "std_auroc": float(np.std(vals)) if vals else 0.0,
                }
                for key, vals in indiv.items()
            },
        }

    # Combined LR stats
    combined_lr_aurocs = []
    combined_lr_coefs = []
    for fr in fold_results:
        clr = fr.get("combined_lr")
        if clr:
            combined_lr_aurocs.append(clr["auroc"])
            combined_lr_coefs.append(clr["coefs"])

    combined_lr = {
        "mean_auroc": float(np.mean(combined_lr_aurocs)) if combined_lr_aurocs else 0.0,
        "std_auroc": float(np.std(combined_lr_aurocs)) if combined_lr_aurocs else 0.0,
        "mean_coefs": np.mean(combined_lr_coefs, axis=0).tolist() if combined_lr_coefs else [],
        "per_fold": combined_lr_aurocs,
    }

    # Find best method per attack
    best_per_attack = {}
    for attack, data in per_attack.items():
        candidates = {
            "grid_search": data["grid_search"]["mean_auroc"],
            "logistic_regression": data["logistic_regression"]["mean_auroc"],
        }
        for key in data["individual"]:
            candidates[key] = data["individual"][key]["mean_auroc"]
        best_method = max(candidates, key=candidates.get)
        best_per_attack[attack] = {
            "method": best_method,
            "auroc": candidates[best_method],
        }

    results = {
        "per_attack": per_attack,
        "combined_lr": combined_lr,
        "best_per_attack": best_per_attack,
        "n_folds": len(fold_results),
    }

    save_results(results, output_dir)
    return results


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def generate_figures(agg_results: dict, output_dir: Path):
    """Generate comparison bar chart of AUROC by method."""
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

    per_attack = agg_results.get("per_attack", {})
    # Filter to main attack types (exclude per-delta A2 variants)
    main_attacks = sorted([a for a in per_attack if "delta" not in a])

    if not main_attacks:
        logger.info("No main attack types to plot")
        return

    methods = ["NLL_S_mean", "Rank_S_rank_mean", "Mahalanobis", "grid_search", "logistic_regression"]
    method_labels = ["NLL only", "Rank only", "Maha only", "Grid ensemble", "LR ensemble"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(main_attacks))
    width = 0.15

    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        vals = []
        errs = []
        for attack in main_attacks:
            data = per_attack[attack]
            if method in data.get("individual", {}):
                vals.append(data["individual"][method]["mean_auroc"])
                errs.append(data["individual"][method]["std_auroc"])
            elif method in data:
                vals.append(data[method]["mean_auroc"])
                errs.append(data[method]["std_auroc"])
            else:
                vals.append(0.0)
                errs.append(0.0)
        ax.bar(x + i * width, vals, width, yerr=errs, label=label,
               color=color, capsize=3, edgecolor="black", alpha=0.85)

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("AUROC")
    ax.set_title("Learned Ensemble vs Individual Scorers")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(main_attacks, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save_figure(fig, fig_dir, "ensemble_auroc_comparison.pdf")
    plt.close(fig)

    # Also plot A2 deltas if available
    a2_attacks = sorted([a for a in per_attack if a.startswith("A2_")])
    if len(a2_attacks) > 1:
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        x2 = np.arange(len(a2_attacks))
        for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
            vals = []
            errs = []
            for attack in a2_attacks:
                data = per_attack[attack]
                if method in data.get("individual", {}):
                    vals.append(data["individual"][method]["mean_auroc"])
                    errs.append(data["individual"][method]["std_auroc"])
                elif method in data:
                    vals.append(data[method]["mean_auroc"])
                    errs.append(data[method]["std_auroc"])
                else:
                    vals.append(0.0)
                    errs.append(0.0)
            ax2.bar(x2 + i * width, vals, width, yerr=errs, label=label,
                    color=color, capsize=3, edgecolor="black", alpha=0.85)

        ax2.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("A2 Coordinate Attack Variant")
        ax2.set_ylabel("AUROC")
        ax2.set_title("Ensemble vs Individual on Coordinate Tampering (A2)")
        ax2.set_xticks(x2 + width * 2)
        ax2.set_xticklabels(a2_attacks, rotation=45, ha="right", fontsize=7)
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=7, loc="lower right")
        fig2.tight_layout()
        save_figure(fig2, fig_dir, "ensemble_a2_deltas.pdf")
        plt.close(fig2)

    logger.info(f"Figures saved to {fig_dir}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = get_default_parser("Learned Ensemble", 19)
    parser.add_argument(
        "--v8", action="store_true",
        help="Use V8 cached data (multi-line decoder)",
    )
    args = parser.parse_args()

    # Allow environment variable override for V8
    use_v8 = args.v8 or bool(os.environ.get("ANOMALY_CACHE_DIR"))

    setup_logging("exp19_learned_ensemble")

    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Folds: {args.folds}")
    logger.info(f"V8 mode: {use_v8}")

    if use_v8 and V8_CACHE_DIR.exists():
        logger.info(f"V8 cache dir exists: {V8_CACHE_DIR}")

    fold_results = []
    for fold in args.folds:
        try:
            result = run_fold(fold, args.output_dir, use_v8=use_v8)
            fold_results.append(result)
        except FileNotFoundError as e:
            logger.error(f"Fold {fold} failed (missing data): {e}")
            continue
        except Exception as e:
            logger.error(f"Fold {fold} failed: {e}")
            raise

    if not fold_results:
        logger.error("No folds completed successfully")
        return

    agg = aggregate(fold_results, args.output_dir)
    generate_figures(agg, args.output_dir)

    # Summary update
    best = agg.get("best_per_attack", {})
    combined = agg.get("combined_lr", {})
    if best:
        top = max(best.values(), key=lambda x: x["auroc"])
        finding = (
            f"Best ensemble: {top['method']} at {top['auroc']:.3f} AUROC; "
            f"combined LR: {combined.get('mean_auroc', 0):.3f}"
        )
        auroc_str = f"{top['auroc']:.3f}"
    else:
        finding = "No attacks evaluated"
        auroc_str = "--"

    append_to_summary(
        exp_id=19,
        name="Learned Ensemble",
        status="done",
        auroc=auroc_str,
        finding=finding,
        output_dir=BASE_DIR,
    )

    logger.info("Experiment 19 (Learned Ensemble) complete.")


if __name__ == "__main__":
    main()
