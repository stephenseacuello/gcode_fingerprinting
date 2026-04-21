#!/usr/bin/env python3
"""Experiment 21: Baseline Anomaly Detection Comparisons

Implements four baseline anomaly detection methods from the literature for
direct comparison against our NLL-based decoder approach:

  Baseline 1: Kimmell et al. (2025) - Energy-based autoencoder on per-window
               energy features from accelerometer and electrical channels.
  Baseline 2: Coelho et al. (2024) - 1D-CNN classifier with MSP (Maximum
               Softmax Probability) anomaly scoring.
  Baseline 3: Isolation Forest on flattened sensor features per window.
  Baseline 4: One-Class SVM on mean-pooled sensor features.

Evaluates on:
  (a) Real damage anomalies (damageadaptive=6, damageface=7, damagepocket=8)
  (b) Synthetic attacks (A1-A4) from outputs/anomaly20260319/attacks/

Reports mean +/- std AUROC across 5-fold CV for each baseline and attack type.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import (
    ATTACK_DIR,
    BASE_DIR,
    FOLDS,
    PREPROC_TEMPLATE,
    compute_auroc,
    bootstrap_auroc_ci,
    compute_detection_at_fpr,
    cohens_d,
    save_results,
    save_figure,
    append_to_summary,
    setup_logging,
    get_default_parser,
    load_preprocessed_npz,
)

logger = logging.getLogger(__name__)

# Damage class IDs (real anomalies)
DAMAGE_CLASSES = {6: "damageadaptive", 7: "damageface", 8: "damagepocket"}
NORMAL_CLASSES = {0, 1, 2, 3, 4, 5}  # adaptive, face, pocket (with/without 150025)


# ============================================================
# Data Loading Helpers
# ============================================================


def load_fold_sensor_data(fold: int, split: str = "test") -> Tuple[np.ndarray, np.ndarray]:
    """Load sensor data and operation types from preprocessed NPZ.

    Returns:
        continuous: [N, 256, 110] sensor windows
        operation_type: [N] integer class labels
    """
    data = load_preprocessed_npz(fold, split)
    continuous = data["continuous"]  # [N, 256, 110]
    operation_type = data["operation_type"]  # [N]
    return continuous, operation_type


def split_normal_anomaly(
    continuous: np.ndarray,
    operation_type: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into normal and damage-anomaly subsets.

    Returns:
        normal_data, normal_labels, anomaly_data, anomaly_labels
    """
    normal_mask = np.isin(operation_type, list(NORMAL_CLASSES))
    anomaly_mask = np.isin(operation_type, list(DAMAGE_CLASSES.keys()))
    return (
        continuous[normal_mask],
        operation_type[normal_mask],
        continuous[anomaly_mask],
        operation_type[anomaly_mask],
    )


def load_attack_data(fold: int, continuous: np.ndarray) -> Dict[str, dict]:
    """Load synthetic attack data for a given fold.

    For sensor-based baselines, attacks modify token sequences but we need
    the original sensor windows (since the sensor data is unchanged in attacks).
    We return the original indices so baselines can flag those windows.

    Returns:
        dict mapping attack_name -> {"orig_indices": np.ndarray, "n_attacks": int}
    """
    attacks = {}
    attack_dir = ATTACK_DIR

    # A1: Command swap
    a1_path = attack_dir / "attack_A1_command_swap.pt"
    if a1_path.exists():
        a1 = torch.load(a1_path, map_location="cpu", weights_only=False)
        if fold in a1["fold_attacks"] and a1["fold_attacks"][fold]["n_attacks"] > 0:
            a1_data = a1["fold_attacks"][fold]
            attacks["A1_command_swap"] = {
                "orig_indices": np.array(a1_data["orig_indices"]),
                "n_attacks": a1_data["n_attacks"],
            }

    # A2: Coordinate tampering (multiple deltas)
    a2_path = attack_dir / "attack_A2_coord_graded.pt"
    if a2_path.exists():
        a2 = torch.load(a2_path, map_location="cpu", weights_only=False)
        for key, data in a2.get("attacks_by_delta_axis", {}).items():
            fold_data = data["fold_attacks"].get(fold, {})
            if fold_data.get("n_attacks", 0) > 0:
                attacks[f"A2_{key}"] = {
                    "orig_indices": np.array(fold_data["orig_indices"]),
                    "n_attacks": fold_data["n_attacks"],
                }

    # A3: Param swap
    a3_path = attack_dir / "attack_A3_param_swap.pt"
    if a3_path.exists():
        a3 = torch.load(a3_path, map_location="cpu", weights_only=False)
        if fold in a3["fold_attacks"] and a3["fold_attacks"][fold]["n_attacks"] > 0:
            a3_data = a3["fold_attacks"][fold]
            attacks["A3_param_swap"] = {
                "orig_indices": np.array(a3_data["orig_indices"]),
                "n_attacks": a3_data["n_attacks"],
            }

    # A4: Cross-operation swap
    a4_path = attack_dir / "attack_A4_cross_op.pt"
    if a4_path.exists():
        a4 = torch.load(a4_path, map_location="cpu", weights_only=False)
        if fold in a4["fold_attacks"] and a4["fold_attacks"][fold]["n_attacks"] > 0:
            a4_data = a4["fold_attacks"][fold]
            attacks["A4_cross_op"] = {
                "orig_indices": np.array(a4_data["orig_indices"]),
                "n_attacks": a4_data["n_attacks"],
            }

    return attacks


# ============================================================
# Feature Extraction
# ============================================================


def compute_energy_features(continuous: np.ndarray) -> np.ndarray:
    """Compute per-channel energy features (sum of squared values per window).

    Args:
        continuous: [N, 256, 110] raw sensor data

    Returns:
        energy: [N, 110] energy per channel per window
    """
    return np.sum(continuous ** 2, axis=1)  # [N, 110]


def compute_mean_features(continuous: np.ndarray) -> np.ndarray:
    """Compute per-channel mean features for One-Class SVM.

    Args:
        continuous: [N, 256, 110] raw sensor data

    Returns:
        mean_features: [N, 110] mean per channel per window
    """
    return np.mean(continuous, axis=1)  # [N, 110]


def compute_flat_features(continuous: np.ndarray, max_dim: int = 2000) -> np.ndarray:
    """Flatten sensor data for Isolation Forest, with optional downsampling.

    Args:
        continuous: [N, 256, 110] raw sensor data
        max_dim: maximum feature dimensionality (downsample time if exceeded)

    Returns:
        flat: [N, D] flattened features
    """
    N, T, C = continuous.shape
    if T * C > max_dim:
        # Downsample time dimension
        step = max(1, T * C // max_dim)
        flat = continuous[:, ::step, :].reshape(N, -1)
    else:
        flat = continuous.reshape(N, -1)
    return flat


# ============================================================
# Baseline 1: Kimmell et al. (2025) - Energy Autoencoder
# ============================================================


class EnergyAutoencoder(nn.Module):
    """Simple autoencoder for energy-based anomaly detection."""

    def __init__(self, n_features: int, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train_energy_autoencoder(
    normal_energy: np.ndarray,
    val_energy: np.ndarray,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 42,
) -> Tuple[EnergyAutoencoder, float]:
    """Train energy autoencoder on normal data, set threshold on val set.

    Returns:
        model: trained autoencoder
        threshold: reconstruction error threshold (95th percentile on val)
    """
    torch.manual_seed(seed)
    n_features = normal_energy.shape[1]
    model = EnergyAutoencoder(n_features)

    # Normalize features
    mean = normal_energy.mean(axis=0, keepdims=True)
    std = normal_energy.std(axis=0, keepdims=True) + 1e-8
    normal_normed = (normal_energy - mean) / std
    val_normed = (val_energy - mean) / std

    dataset = TensorDataset(torch.from_numpy(normal_normed).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

    model.eval()

    # Compute reconstruction errors on val set for threshold
    with torch.no_grad():
        val_tensor = torch.from_numpy(val_normed).float()
        val_recon = model(val_tensor)
        val_errors = ((val_tensor - val_recon) ** 2).mean(dim=1).numpy()

    threshold = np.percentile(val_errors, 95)

    return model, threshold, mean, std


def score_energy_autoencoder(
    model: EnergyAutoencoder,
    energy: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Compute reconstruction error scores for energy features.

    Returns:
        errors: [N] per-window reconstruction error (higher = more anomalous)
    """
    normed = (energy - mean) / std
    with torch.no_grad():
        x = torch.from_numpy(normed).float()
        recon = model(x)
        errors = ((x - recon) ** 2).mean(dim=1).numpy()
    return errors


# ============================================================
# Baseline 2: Coelho et al. (2024) - 1D-CNN with MSP
# ============================================================


class CNN1DClassifier(nn.Module):
    """1D-CNN classifier for sensor traces (Coelho et al. 2024)."""

    def __init__(self, n_channels: int, n_classes: int = 9):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [N, C, T]
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x).squeeze(-1)  # [N, 64]
        return self.fc(x)  # [N, n_classes]


def train_cnn_classifier(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    n_classes: int = 9,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 42,
    device: str = "cpu",
) -> CNN1DClassifier:
    """Train 1D-CNN classifier on all classes.

    Args:
        train_data: [N, 256, 110] sensor windows
        train_labels: [N] class labels (0-8)
        val_data, val_labels: validation set for early stopping

    Returns:
        model: trained CNN classifier
    """
    torch.manual_seed(seed)

    n_channels = train_data.shape[2]  # 110
    model = CNN1DClassifier(n_channels, n_classes).to(device)

    # Transpose to [N, C, T] for Conv1d
    train_tensor = torch.from_numpy(train_data).float().permute(0, 2, 1).to(device)
    train_label_tensor = torch.from_numpy(train_labels).long().to(device)
    val_tensor = torch.from_numpy(val_data).float().permute(0, 2, 1).to(device)
    val_label_tensor = torch.from_numpy(val_labels).long().to(device)

    dataset = TensorDataset(train_tensor, train_label_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Validation accuracy for early stopping
        model.eval()
        with torch.no_grad():
            val_logits = model(val_tensor)
            val_acc = (val_logits.argmax(dim=1) == val_label_tensor).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()
    logger.info(f"  CNN classifier best val accuracy: {best_val_acc:.4f}")
    return model


def score_msp(
    model: CNN1DClassifier,
    data: np.ndarray,
    device: str = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """Compute MSP anomaly scores (1 - max softmax probability).

    Higher score = more anomalous (lower classifier confidence).

    Returns:
        scores: [N] anomaly scores in [0, 1]
    """
    model.eval()
    # Transpose to [N, C, T]
    data_tensor = torch.from_numpy(data).float().permute(0, 2, 1).to(device)

    all_scores = []
    for i in range(0, len(data_tensor), batch_size):
        batch = data_tensor[i : i + batch_size]
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            max_prob = probs.max(dim=1).values
            all_scores.append((1.0 - max_prob).cpu().numpy())

    return np.concatenate(all_scores)


# ============================================================
# Baseline 3: Isolation Forest
# ============================================================


def run_isolation_forest(
    normal_features: np.ndarray,
    test_features: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Run Isolation Forest on flattened features.

    Returns:
        scores: [N_test] anomaly scores (higher = more anomalous)
    """
    from sklearn.ensemble import IsolationForest

    clf = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(normal_features)

    # score_samples returns negative scores (more negative = more anomalous)
    # Negate so higher = more anomalous
    return -clf.score_samples(test_features)


# ============================================================
# Baseline 4: One-Class SVM
# ============================================================


def run_ocsvm(
    normal_features: np.ndarray,
    test_features: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Run One-Class SVM on mean-pooled features.

    Returns:
        scores: [N_test] anomaly scores (higher = more anomalous)
    """
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    normal_scaled = scaler.fit_transform(normal_features)
    test_scaled = scaler.transform(test_features)

    # Subsample if training set is too large for SVM (O(n^2) memory)
    max_train = 5000
    if len(normal_scaled) > max_train:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(normal_scaled), max_train, replace=False)
        normal_scaled = normal_scaled[idx]

    clf = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    clf.fit(normal_scaled)

    # decision_function: positive = normal, negative = anomaly
    # Negate so higher = more anomalous
    return -clf.decision_function(test_scaled)


# ============================================================
# Evaluation
# ============================================================


def evaluate_baseline(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    name: str,
) -> dict:
    """Compute AUROC and detection metrics for a baseline.

    Returns:
        dict with auroc, auroc_ci, detection rates, cohens_d
    """
    if len(anomaly_scores) == 0 or len(normal_scores) == 0:
        return {
            "auroc": float("nan"),
            "auroc_ci": [float("nan"), float("nan")],
            "detection_at_fpr_01": float("nan"),
            "detection_at_fpr_05": float("nan"),
            "cohens_d": float("nan"),
            "n_normal": len(normal_scores),
            "n_anomaly": len(anomaly_scores),
        }

    try:
        auroc_info = bootstrap_auroc_ci(normal_scores, anomaly_scores, n_bootstrap=2000)
        det_01 = compute_detection_at_fpr(normal_scores, anomaly_scores, 0.01)
        det_05 = compute_detection_at_fpr(normal_scores, anomaly_scores, 0.05)
        d = cohens_d(anomaly_scores, normal_scores)
    except (ValueError, ZeroDivisionError):
        auroc_info = {"auroc": 0.5, "ci_lower": 0.0, "ci_upper": 1.0, "std": 0.0}
        det_01 = det_05 = 0.0
        d = 0.0

    return {
        "auroc": auroc_info["auroc"],
        "auroc_ci": [auroc_info["ci_lower"], auroc_info["ci_upper"]],
        "detection_at_fpr_01": det_01,
        "detection_at_fpr_05": det_05,
        "cohens_d": d,
        "n_normal": len(normal_scores),
        "n_anomaly": len(anomaly_scores),
    }


# ============================================================
# Per-Fold Runner
# ============================================================


def run_fold(fold: int, output_dir: Path, seed: int = 42) -> dict:
    """Run all baselines for one fold.

    Uses train split to fit baselines, test split for evaluation.
    """
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Fold {fold} ===")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load train and test data
    train_cont, train_ops = load_fold_sensor_data(fold, "train")
    test_cont, test_ops = load_fold_sensor_data(fold, "test")

    logger.info(f"  Train: {train_cont.shape}, Test: {test_cont.shape}")

    # Split test into normal vs real damage anomalies
    normal_test, normal_labels, anomaly_test, anomaly_labels = split_normal_anomaly(
        test_cont, test_ops
    )
    logger.info(
        f"  Test normal: {len(normal_test)}, Test damage anomalies: {len(anomaly_test)}"
    )

    # Normal train data (for unsupervised baselines)
    train_normal_mask = np.isin(train_ops, list(NORMAL_CLASSES))
    train_normal = train_cont[train_normal_mask]
    train_normal_labels = train_ops[train_normal_mask]

    # Use a portion of normal test as validation for threshold setting
    rng = np.random.RandomState(seed + fold)
    n_val = max(1, len(normal_test) // 5)
    val_indices = rng.choice(len(normal_test), n_val, replace=False)
    val_mask = np.zeros(len(normal_test), dtype=bool)
    val_mask[val_indices] = True
    eval_normal = normal_test[~val_mask]
    val_normal = normal_test[val_mask]

    fold_results = {"fold": fold, "baselines": {}}

    # ----------------------------------------------------------
    # Baseline 1: Energy Autoencoder (Kimmell et al. 2025)
    # ----------------------------------------------------------
    logger.info("  Baseline 1: Energy Autoencoder (Kimmell et al.)")

    train_energy = compute_energy_features(train_normal)
    val_energy = compute_energy_features(val_normal)
    eval_normal_energy = compute_energy_features(eval_normal)
    anomaly_energy = compute_energy_features(anomaly_test)

    ae_model, ae_threshold, ae_mean, ae_std = train_energy_autoencoder(
        train_energy, val_energy, seed=seed + fold
    )

    normal_ae_scores = score_energy_autoencoder(ae_model, eval_normal_energy, ae_mean, ae_std)
    anomaly_ae_scores = score_energy_autoencoder(ae_model, anomaly_energy, ae_mean, ae_std)

    fold_results["baselines"]["energy_ae"] = {
        "damage_all": evaluate_baseline(normal_ae_scores, anomaly_ae_scores, "Energy AE"),
    }

    # Per damage class
    for cls_id, cls_name in DAMAGE_CLASSES.items():
        cls_mask = anomaly_labels == cls_id
        if cls_mask.any():
            cls_scores = score_energy_autoencoder(
                ae_model, anomaly_test[cls_mask], ae_mean, ae_std
            )
            cls_energy = compute_energy_features(anomaly_test[cls_mask])
            cls_ae_scores = score_energy_autoencoder(ae_model, cls_energy, ae_mean, ae_std)
            fold_results["baselines"]["energy_ae"][cls_name] = evaluate_baseline(
                normal_ae_scores, cls_ae_scores, f"Energy AE ({cls_name})"
            )

    # ----------------------------------------------------------
    # Baseline 2: 1D-CNN MSP (Coelho et al. 2024)
    # ----------------------------------------------------------
    logger.info("  Baseline 2: 1D-CNN MSP (Coelho et al.)")

    cnn_model = train_cnn_classifier(
        train_cont,
        train_ops,
        val_normal,  # use normal val for early stopping accuracy
        np.full(len(val_normal), train_ops[train_normal_mask][0]),  # placeholder labels
        n_classes=9,
        n_epochs=50,
        seed=seed + fold,
        device=device,
    )
    # Actually, we need labeled val data for proper early stopping
    # Use a portion of train as val
    n_train_val = max(1, len(train_cont) // 10)
    train_val_idx = rng.choice(len(train_cont), n_train_val, replace=False)
    train_val_mask = np.zeros(len(train_cont), dtype=bool)
    train_val_mask[train_val_idx] = True

    cnn_model = train_cnn_classifier(
        train_cont[~train_val_mask],
        train_ops[~train_val_mask],
        train_cont[train_val_mask],
        train_ops[train_val_mask],
        n_classes=9,
        n_epochs=50,
        seed=seed + fold,
        device=device,
    )

    normal_msp_scores = score_msp(cnn_model, eval_normal, device=device)
    anomaly_msp_scores = score_msp(cnn_model, anomaly_test, device=device)

    fold_results["baselines"]["cnn_msp"] = {
        "damage_all": evaluate_baseline(normal_msp_scores, anomaly_msp_scores, "CNN MSP"),
    }

    for cls_id, cls_name in DAMAGE_CLASSES.items():
        cls_mask = anomaly_labels == cls_id
        if cls_mask.any():
            cls_msp = score_msp(cnn_model, anomaly_test[cls_mask], device=device)
            fold_results["baselines"]["cnn_msp"][cls_name] = evaluate_baseline(
                normal_msp_scores, cls_msp, f"CNN MSP ({cls_name})"
            )

    # ----------------------------------------------------------
    # Baseline 3: Isolation Forest
    # ----------------------------------------------------------
    logger.info("  Baseline 3: Isolation Forest")

    train_flat = compute_flat_features(train_normal)
    eval_normal_flat = compute_flat_features(eval_normal)
    anomaly_flat = compute_flat_features(anomaly_test)

    normal_if_scores = run_isolation_forest(train_flat, eval_normal_flat, seed=seed + fold)
    anomaly_if_scores = run_isolation_forest(train_flat, anomaly_flat, seed=seed + fold)

    fold_results["baselines"]["isolation_forest"] = {
        "damage_all": evaluate_baseline(normal_if_scores, anomaly_if_scores, "IsoForest"),
    }

    for cls_id, cls_name in DAMAGE_CLASSES.items():
        cls_mask = anomaly_labels == cls_id
        if cls_mask.any():
            cls_flat = compute_flat_features(anomaly_test[cls_mask])
            cls_if = run_isolation_forest(train_flat, cls_flat, seed=seed + fold)
            fold_results["baselines"]["isolation_forest"][cls_name] = evaluate_baseline(
                normal_if_scores, cls_if, f"IsoForest ({cls_name})"
            )

    # ----------------------------------------------------------
    # Baseline 4: One-Class SVM
    # ----------------------------------------------------------
    logger.info("  Baseline 4: One-Class SVM")

    train_mean = compute_mean_features(train_normal)
    eval_normal_mean = compute_mean_features(eval_normal)
    anomaly_mean = compute_mean_features(anomaly_test)

    normal_ocsvm_scores = run_ocsvm(train_mean, eval_normal_mean, seed=seed + fold)
    anomaly_ocsvm_scores = run_ocsvm(train_mean, anomaly_mean, seed=seed + fold)

    fold_results["baselines"]["ocsvm"] = {
        "damage_all": evaluate_baseline(
            normal_ocsvm_scores, anomaly_ocsvm_scores, "OC-SVM"
        ),
    }

    for cls_id, cls_name in DAMAGE_CLASSES.items():
        cls_mask = anomaly_labels == cls_id
        if cls_mask.any():
            cls_mean = compute_mean_features(anomaly_test[cls_mask])
            cls_ocsvm = run_ocsvm(train_mean, cls_mean, seed=seed + fold)
            fold_results["baselines"]["ocsvm"][cls_name] = evaluate_baseline(
                normal_ocsvm_scores, cls_ocsvm, f"OC-SVM ({cls_name})"
            )

    # ----------------------------------------------------------
    # Synthetic Attack Evaluation (all baselines)
    # ----------------------------------------------------------
    logger.info("  Evaluating on synthetic attacks...")

    attacks = load_attack_data(fold, test_cont)

    # For synthetic attacks, the sensor data is unchanged (attacks modify tokens).
    # We score the original sensor windows that were attacked.
    # A baseline that uses only sensors should score these same as normal,
    # demonstrating that sensor-only baselines miss G-code-level attacks.

    attack_results = {}
    for attack_name, attack_info in attacks.items():
        orig_idx = attack_info["orig_indices"]
        if len(orig_idx) == 0:
            continue

        # Get the test sensor data at attacked window indices
        # These indices refer to the full test set
        if orig_idx.max() >= len(test_cont):
            logger.warning(
                f"  Attack {attack_name}: indices exceed test set size, skipping"
            )
            continue

        attacked_windows = test_cont[orig_idx]

        attack_results[attack_name] = {}

        # Energy AE
        atk_energy = compute_energy_features(attacked_windows)
        atk_ae = score_energy_autoencoder(ae_model, atk_energy, ae_mean, ae_std)
        attack_results[attack_name]["energy_ae"] = evaluate_baseline(
            normal_ae_scores, atk_ae, f"Energy AE vs {attack_name}"
        )

        # CNN MSP
        atk_msp = score_msp(cnn_model, attacked_windows, device=device)
        attack_results[attack_name]["cnn_msp"] = evaluate_baseline(
            normal_msp_scores, atk_msp, f"CNN MSP vs {attack_name}"
        )

        # Isolation Forest
        atk_flat = compute_flat_features(attacked_windows)
        atk_if = run_isolation_forest(train_flat, atk_flat, seed=seed + fold)
        attack_results[attack_name]["isolation_forest"] = evaluate_baseline(
            normal_if_scores, atk_if, f"IsoForest vs {attack_name}"
        )

        # OC-SVM
        atk_mean_feat = compute_mean_features(attacked_windows)
        atk_ocsvm = run_ocsvm(train_mean, atk_mean_feat, seed=seed + fold)
        attack_results[attack_name]["ocsvm"] = evaluate_baseline(
            normal_ocsvm_scores, atk_ocsvm, f"OC-SVM vs {attack_name}"
        )

    fold_results["synthetic_attacks"] = attack_results

    # Save fold results
    save_results(fold_results, fold_dir, "baseline_results.json")
    logger.info(f"  Fold {fold} complete.")

    return fold_results


# ============================================================
# Cross-Fold Aggregation
# ============================================================


def aggregate(fold_results: List[dict], output_dir: Path) -> dict:
    """Aggregate results across folds. Reports mean +/- std AUROC."""
    baselines = ["energy_ae", "cnn_msp", "isolation_forest", "ocsvm"]
    baseline_names = {
        "energy_ae": "Kimmell Energy AE",
        "cnn_msp": "Coelho 1D-CNN MSP",
        "isolation_forest": "Isolation Forest",
        "ocsvm": "One-Class SVM",
    }

    # --- Real damage anomalies ---
    damage_agg = {}
    eval_targets = ["damage_all"] + list(DAMAGE_CLASSES.values())

    for bl in baselines:
        damage_agg[bl] = {"name": baseline_names[bl]}
        for target in eval_targets:
            aurocs = []
            for fr in fold_results:
                bl_data = fr["baselines"].get(bl, {})
                if target in bl_data and not np.isnan(bl_data[target].get("auroc", float("nan"))):
                    aurocs.append(bl_data[target]["auroc"])
            if aurocs:
                damage_agg[bl][target] = {
                    "mean_auroc": float(np.mean(aurocs)),
                    "std_auroc": float(np.std(aurocs)),
                    "per_fold": aurocs,
                    "n_folds": len(aurocs),
                }

    # --- Synthetic attacks ---
    attack_types = set()
    for fr in fold_results:
        attack_types.update(fr.get("synthetic_attacks", {}).keys())

    attack_agg = {}
    for attack in sorted(attack_types):
        attack_agg[attack] = {}
        for bl in baselines:
            aurocs = []
            for fr in fold_results:
                atk_data = fr.get("synthetic_attacks", {}).get(attack, {})
                bl_data = atk_data.get(bl, {})
                if bl_data and not np.isnan(bl_data.get("auroc", float("nan"))):
                    aurocs.append(bl_data["auroc"])
            if aurocs:
                attack_agg[attack][bl] = {
                    "mean_auroc": float(np.mean(aurocs)),
                    "std_auroc": float(np.std(aurocs)),
                    "per_fold": aurocs,
                }

    results = {
        "damage_detection": damage_agg,
        "synthetic_attacks": attack_agg,
        "n_folds": len(fold_results),
    }

    save_results(results, output_dir)

    # Print summary table
    logger.info("")
    logger.info("=" * 80)
    logger.info("DAMAGE DETECTION RESULTS (mean +/- std AUROC across folds)")
    logger.info("=" * 80)
    header = f"{'Baseline':<25s}"
    for target in eval_targets:
        header += f" | {target:<18s}"
    logger.info(header)
    logger.info("-" * 80)

    for bl in baselines:
        row = f"{baseline_names[bl]:<25s}"
        for target in eval_targets:
            data = damage_agg[bl].get(target, {})
            if data:
                row += f" | {data['mean_auroc']:.3f} +/- {data['std_auroc']:.3f}  "
            else:
                row += f" | {'N/A':>18s}"
        logger.info(row)

    if attack_agg:
        logger.info("")
        logger.info("=" * 80)
        logger.info("SYNTHETIC ATTACK RESULTS (mean +/- std AUROC)")
        logger.info("=" * 80)
        for attack in sorted(attack_agg.keys()):
            # Filter to main attacks for cleaner output
            if "delta" in attack:
                continue
            logger.info(f"  {attack}:")
            for bl in baselines:
                data = attack_agg[attack].get(bl, {})
                if data:
                    logger.info(
                        f"    {baseline_names[bl]:<25s}: "
                        f"{data['mean_auroc']:.3f} +/- {data['std_auroc']:.3f}"
                    )

    return results


# ============================================================
# Figures
# ============================================================


def generate_figures(agg_results: dict, output_dir: Path):
    """Generate comparison bar chart figures."""
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

    baselines = ["energy_ae", "cnn_msp", "isolation_forest", "ocsvm"]
    baseline_labels = {
        "energy_ae": "Energy AE\n(Kimmell)",
        "cnn_msp": "1D-CNN MSP\n(Coelho)",
        "isolation_forest": "Isolation\nForest",
        "ocsvm": "One-Class\nSVM",
    }

    # --- Figure 1: Damage detection AUROC comparison ---
    damage_data = agg_results.get("damage_detection", {})
    eval_targets = ["damage_all"] + list(DAMAGE_CLASSES.values())
    target_labels = {
        "damage_all": "All Damage",
        "damageadaptive": "Damage\nAdaptive",
        "damageface": "Damage\nFace",
        "damagepocket": "Damage\nPocket",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(eval_targets))
    width = 0.18
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, bl in enumerate(baselines):
        bl_data = damage_data.get(bl, {})
        means = []
        stds = []
        for target in eval_targets:
            data = bl_data.get(target, {})
            if data:
                means.append(data["mean_auroc"])
                stds.append(data["std_auroc"])
            else:
                means.append(0)
                stds.append(0)
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=baseline_labels[bl],
            color=colors[i],
            capsize=3,
            alpha=0.85,
        )

    ax.set_xlabel("Anomaly Type", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Baseline Anomaly Detection: Damage Classes", fontsize=13)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([target_labels.get(t, t) for t in eval_targets], fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    fig.tight_layout()
    save_figure(fig, fig_dir, "baseline_damage_auroc.pdf")
    plt.close(fig)

    # --- Figure 2: Synthetic attack AUROC comparison ---
    attack_data = agg_results.get("synthetic_attacks", {})
    # Filter to main attack types
    main_attacks = sorted(
        [a for a in attack_data.keys() if "delta" not in a]
    )

    if main_attacks:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(main_attacks))
        width = 0.18

        for i, bl in enumerate(baselines):
            means = []
            stds = []
            for attack in main_attacks:
                data = attack_data.get(attack, {}).get(bl, {})
                if data:
                    means.append(data["mean_auroc"])
                    stds.append(data["std_auroc"])
                else:
                    means.append(0)
                    stds.append(0)
            ax.bar(
                x + i * width,
                means,
                width,
                yerr=stds,
                label=baseline_labels[bl],
                color=colors[i],
                capsize=3,
                alpha=0.85,
            )

        ax.set_xlabel("Attack Type", fontsize=12)
        ax.set_ylabel("AUROC", fontsize=12)
        ax.set_title(
            "Baseline Anomaly Detection: Synthetic Attacks\n"
            "(sensor-only baselines on G-code-level attacks)",
            fontsize=13,
        )
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(main_attacks, rotation=20, ha="right", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        fig.tight_layout()
        save_figure(fig, fig_dir, "baseline_synthetic_auroc.pdf")
        plt.close(fig)

    logger.info(f"Figures saved to {fig_dir}")


# ============================================================
# Main
# ============================================================


def main():
    parser = get_default_parser("Baseline Comparisons", 21)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for neural network baselines",
    )
    args = parser.parse_args()

    setup_logging("exp21_baselines")

    # Override output dir default
    if "exp21" not in str(args.output_dir):
        args.output_dir = BASE_DIR / "exp21_baselines"

    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Folds: {args.folds}")
    logger.info(f"Device: {args.device}")

    fold_results = []
    for fold in args.folds:
        result = run_fold(fold, args.output_dir, seed=args.seed)
        fold_results.append(result)

    agg = aggregate(fold_results, args.output_dir)
    generate_figures(agg, args.output_dir)

    # Update summary
    damage_data = agg.get("damage_detection", {})
    best_bl = None
    best_auroc = 0
    for bl, bl_data in damage_data.items():
        if bl == "name":
            continue
        all_data = bl_data.get("damage_all", {})
        if all_data and all_data.get("mean_auroc", 0) > best_auroc:
            best_auroc = all_data["mean_auroc"]
            best_bl = bl_data.get("name", bl)

    if best_bl:
        finding = f"Best baseline: {best_bl} at {best_auroc:.3f} AUROC (damage)"
    else:
        finding = "No damage data available"

    append_to_summary(
        exp_id=21,
        name="Baseline Comparisons",
        status="done",
        auroc=f"{best_auroc:.3f}" if best_bl else "--",
        finding=finding,
        output_dir=BASE_DIR,
    )

    logger.info("Experiment 21 (Baseline Comparisons) complete.")


if __name__ == "__main__":
    main()
