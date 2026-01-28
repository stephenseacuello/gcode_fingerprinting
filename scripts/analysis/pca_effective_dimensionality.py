#!/usr/bin/env python3
"""
PCA analysis to determine effective dimensionality of sensor features.

This script supports the technical soundness claim that 296 features have
substantial redundancy, explaining why single sensors (41 features)
perform as well as or better than the full array.

Usage:
    python scripts/analysis/pca_effective_dimensionality.py
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def analyze_effective_dimensionality(data_path: str = "outputs/jan23_followup/no_leakage/train_sequences.npz"):
    """Run PCA analysis on training data."""

    # Load training data from npz
    data = np.load(data_path, allow_pickle=True)
    print("Keys in npz:", list(data.keys()))

    # Get sensor features
    X = data['continuous']
    print(f"Original shape: {X.shape}")

    # If 3D (samples, timesteps, features), reshape to 2D
    if len(X.shape) == 3:
        n_samples, n_timesteps, n_features = X.shape
        X = X.reshape(-1, n_features)  # Flatten time dimension
        print(f"Reshaped to: {X.shape}")

    # Remove any NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA for different variance thresholds
    results = {}
    for threshold in [0.90, 0.95, 0.99]:
        pca = PCA(n_components=threshold)
        pca.fit(X_scaled)
        results[threshold] = pca.n_components_
        print(f"\n{int(threshold*100)}% variance explained by {pca.n_components_} components (out of {X.shape[1]})")
        print(f"  Ratio: {pca.n_components_}/{X.shape[1]} = {pca.n_components_/X.shape[1]:.1%}")

    # Cumulative variance analysis
    pca_full = PCA(n_components=min(100, X.shape[1]))
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    print(f"\nCumulative variance by component count:")
    for n in [10, 20, 30, 40, 50, 60, 80, 100]:
        if n <= len(cumvar):
            print(f"  {n} components: {cumvar[n-1]*100:.1f}%")

    # Key finding
    idx_95 = np.argmax(cumvar >= 0.95) + 1
    print(f"\n*** KEY FINDING: 95% variance in {idx_95} components out of {X.shape[1]} ({idx_95/X.shape[1]*100:.0f}%) ***")

    return results, cumvar


if __name__ == "__main__":
    analyze_effective_dimensionality()
