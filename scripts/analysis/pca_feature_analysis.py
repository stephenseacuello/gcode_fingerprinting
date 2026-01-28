#!/usr/bin/env python3
"""
Comprehensive PCA and feature analysis for 296-feature input vector.

Extends pca_effective_dimensionality.py with:
  1. Variance contribution by feature group (operational sensors, non-functional, machine)
  2. Per-feature variance and zero-rate analysis (are non-functional sensors dead?)
  3. Scree plot + cumulative variance curve (saved as PNG)
  4. Feature importance ranking via PCA loadings
  5. Comparison: PCA on 228 features (excluding non-functional) vs full 296

Usage:
    python scripts/analysis/pca_feature_analysis.py
"""

import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Reproduce exact feature grouping from metadata
NON_FUNCTIONAL_SENSORS = ["frame_b1", "frame_l1", "z_gant_1", "z_gant_2"]
OPERATIONAL_SENSORS = [
    "xa_motor", "frame_r1", "frame_r2", "frame_b2", "frame_l2", "frame_l3",
    "spindle1", "spindle2", "y_bed__1", "y_bed__2", "y_bed__3", "y_bed__4",
]
MODALITIES = [
    "Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Mx", "My", "Mz",
    "Pressure", "Temperature", "Proximity",
    "ColorR", "ColorG", "ColorB", "ColorA", "RMS",
]

# Machine features (non-sensor)
MOTOR_ELECTRICAL = ["spindle", "spindle_A", "x_motor", "x_motor_A", "y_motor", "y_motor_A", "z_motor", "z_motor_A"]
POSITIONS = ["posx", "posy", "posz", "mpox", "mpoy", "mpoz", "line"]
CONTROLLER = ["coor", "dist", "feed", "gcode_line", "momo", "plane", "stat", "unit", "vel"]


def classify_feature(name):
    """Classify a feature name into a group."""
    for sensor in NON_FUNCTIONAL_SENSORS:
        if name.startswith(sensor + ".") or name.startswith(sensor + "_"):
            return "non_functional_sensor"
    for sensor in OPERATIONAL_SENSORS:
        if name.startswith(sensor + ".") or name.startswith(sensor + "_"):
            return "operational_sensor"
    if name in MOTOR_ELECTRICAL:
        return "motor_electrical"
    if name in POSITIONS:
        return "positions"
    if name in CONTROLLER:
        return "controller"
    return "unknown"


def run_analysis(data_path="outputs/jan23_followup/no_leakage/train_sequences.npz",
                 metadata_path="outputs/stratified_splits_2k_vocab/metadata.json",
                 output_dir="outputs/analysis/pca_feature_analysis"):
    """Run comprehensive PCA and feature analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata for column names
    with open(metadata_path) as f:
        metadata = json.load(f)
    columns = metadata["continuous_columns"]
    print(f"Feature columns: {len(columns)}")

    # Load training data
    data = np.load(data_path, allow_pickle=True)
    X = data["continuous"]
    print(f"Raw shape: {X.shape}")

    if len(X.shape) == 3:
        n_samples, n_timesteps, n_features = X.shape
        # Use time-averaged representation for feature-level analysis
        X_avg = np.nanmean(X, axis=1)  # (2966, 296)
        # Flattened for full PCA
        X_flat = X.reshape(-1, n_features)  # (189824, 296)
        print(f"Time-averaged: {X_avg.shape}, Flattened: {X_flat.shape}")
    else:
        X_avg = X
        X_flat = X

    X_avg = np.nan_to_num(X_avg, nan=0.0, posinf=0.0, neginf=0.0)
    X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)

    # ================================================================
    # 1. Feature group classification
    # ================================================================
    print("\n" + "=" * 70)
    print("1. FEATURE GROUP CLASSIFICATION")
    print("=" * 70)

    groups = {}
    feature_groups = []
    for col in columns:
        g = classify_feature(col)
        feature_groups.append(g)
        groups.setdefault(g, []).append(col)

    for g, feats in sorted(groups.items()):
        print(f"  {g}: {len(feats)} features")

    # ================================================================
    # 2. Per-feature statistics (variance, zero-rate, range)
    # ================================================================
    print("\n" + "=" * 70)
    print("2. PER-FEATURE STATISTICS (time-averaged)")
    print("=" * 70)

    variances = np.var(X_avg, axis=0)
    zero_rates = np.mean(X_avg == 0, axis=0)
    means = np.mean(X_avg, axis=0)
    stds = np.std(X_avg, axis=0)

    # Group-level statistics
    for group_name in ["operational_sensor", "non_functional_sensor", "motor_electrical", "positions", "controller"]:
        idxs = [i for i, g in enumerate(feature_groups) if g == group_name]
        if not idxs:
            continue
        g_var = variances[idxs]
        g_zero = zero_rates[idxs]
        print(f"\n  {group_name} ({len(idxs)} features):")
        print(f"    Variance: mean={g_var.mean():.4f}, median={np.median(g_var):.4f}, min={g_var.min():.6f}, max={g_var.max():.4f}")
        print(f"    Zero rate: mean={g_zero.mean():.2%}, max={g_zero.max():.2%}")
        n_near_zero = np.sum(g_var < 1e-6)
        print(f"    Near-zero variance (<1e-6): {n_near_zero}/{len(idxs)}")

    # Individual non-functional sensor features
    print("\n  Non-functional sensor features detail:")
    for col in groups.get("non_functional_sensor", []):
        idx = columns.index(col)
        print(f"    {col:30s}  var={variances[idx]:.6f}  zero={zero_rates[idx]:.2%}  mean={means[idx]:.4f}  std={stds[idx]:.4f}")

    # ================================================================
    # 3. Full PCA (all 296 features)
    # ================================================================
    print("\n" + "=" * 70)
    print("3. PCA ON ALL 296 FEATURES (time-averaged)")
    print("=" * 70)

    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X_avg)

    pca_full = PCA(n_components=min(X_scaled.shape))
    pca_full.fit(X_scaled)
    cumvar_full = np.cumsum(pca_full.explained_variance_ratio_)

    for thresh in [0.80, 0.90, 0.95, 0.99]:
        n = np.argmax(cumvar_full >= thresh) + 1
        print(f"  {thresh:.0%} variance: {n} components ({n / 296:.1%} of 296)")

    print(f"\n  Top 10 components variance:")
    for i in range(10):
        print(f"    PC{i+1}: {pca_full.explained_variance_ratio_[i]:.4f} ({cumvar_full[i]:.4f} cumulative)")

    # ================================================================
    # 4. PCA excluding non-functional sensors (228 features)
    # ================================================================
    print("\n" + "=" * 70)
    print("4. PCA EXCLUDING NON-FUNCTIONAL SENSORS (228 features)")
    print("=" * 70)

    keep_idxs = [i for i, g in enumerate(feature_groups) if g != "non_functional_sensor"]
    X_clean = X_avg[:, keep_idxs]
    print(f"  Reduced feature set: {X_clean.shape[1]} features")

    scaler_clean = StandardScaler()
    X_clean_scaled = scaler_clean.fit_transform(X_clean)

    pca_clean = PCA(n_components=min(X_clean_scaled.shape))
    pca_clean.fit(X_clean_scaled)
    cumvar_clean = np.cumsum(pca_clean.explained_variance_ratio_)

    for thresh in [0.80, 0.90, 0.95, 0.99]:
        n_full = np.argmax(cumvar_full >= thresh) + 1
        n_clean = np.argmax(cumvar_clean >= thresh) + 1
        print(f"  {thresh:.0%} variance: {n_clean} components (vs {n_full} with all 296)")

    # ================================================================
    # 5. Top PCA loadings (which features dominate each component)
    # ================================================================
    print("\n" + "=" * 70)
    print("5. TOP FEATURES BY PCA LOADING (first 5 components)")
    print("=" * 70)

    for pc_idx in range(5):
        loadings = np.abs(pca_full.components_[pc_idx])
        top_idxs = np.argsort(loadings)[::-1][:10]
        print(f"\n  PC{pc_idx + 1} (explains {pca_full.explained_variance_ratio_[pc_idx]:.2%}):")
        for rank, idx in enumerate(top_idxs):
            print(f"    {rank+1}. {columns[idx]:30s}  |loading|={loadings[idx]:.4f}  group={feature_groups[idx]}")

    # ================================================================
    # 6. Redundancy: features with near-zero PCA contribution
    # ================================================================
    print("\n" + "=" * 70)
    print("6. FEATURES WITH LOWEST PCA CONTRIBUTION")
    print("=" * 70)

    # Sum absolute loadings across top-95% components
    n95 = np.argmax(cumvar_full >= 0.95) + 1
    total_loading = np.sum(np.abs(pca_full.components_[:n95]), axis=0)
    bottom_idxs = np.argsort(total_loading)[:20]

    print(f"  Bottom 20 features by total |loading| across top {n95} PCs:")
    for rank, idx in enumerate(bottom_idxs):
        print(f"    {rank+1}. {columns[idx]:30s}  total_loading={total_loading[idx]:.4f}  group={feature_groups[idx]}  var={variances[idx]:.6f}")

    # Count by group
    bottom_groups = {}
    for idx in bottom_idxs:
        g = feature_groups[idx]
        bottom_groups[g] = bottom_groups.get(g, 0) + 1
    print(f"\n  Bottom 20 by group: {bottom_groups}")

    # ================================================================
    # 7. Save plots
    # ================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Scree plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: scree
        axes[0].bar(range(1, 51), pca_full.explained_variance_ratio_[:50], color="steelblue", alpha=0.8)
        axes[0].set_xlabel("Principal Component")
        axes[0].set_ylabel("Variance Explained")
        axes[0].set_title("Scree Plot (Top 50 Components)")
        axes[0].axhline(y=0.01, color="red", linestyle="--", alpha=0.5, label="1% threshold")
        axes[0].legend()

        # Right: cumulative
        axes[1].plot(range(1, len(cumvar_full) + 1), cumvar_full, color="steelblue", linewidth=2, label="All 296")
        axes[1].plot(range(1, len(cumvar_clean) + 1), cumvar_clean, color="orange", linewidth=2, label="228 (no non-functional)")
        axes[1].axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="95% threshold")
        axes[1].axhline(y=0.90, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
        n95_full = np.argmax(cumvar_full >= 0.95) + 1
        n95_clean = np.argmax(cumvar_clean >= 0.95) + 1
        axes[1].axvline(x=n95_full, color="steelblue", linestyle=":", alpha=0.5)
        axes[1].axvline(x=n95_clean, color="orange", linestyle=":", alpha=0.5)
        axes[1].set_xlabel("Number of Components")
        axes[1].set_ylabel("Cumulative Variance Explained")
        axes[1].set_title("Cumulative Variance: 296 vs 228 Features")
        axes[1].legend()
        axes[1].set_xlim(0, 150)

        plt.tight_layout()
        plot_path = output_dir / "pca_scree_and_cumulative.png"
        plt.savefig(plot_path, dpi=150)
        print(f"\n  Saved plot: {plot_path}")
        plt.close()

        # Group variance contribution bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        group_labels = []
        group_var_means = []
        group_counts = []
        for g in ["operational_sensor", "non_functional_sensor", "motor_electrical", "positions", "controller"]:
            idxs = [i for i, gg in enumerate(feature_groups) if gg == g]
            if idxs:
                group_labels.append(g.replace("_", " ").title())
                group_var_means.append(np.mean(variances[idxs]))
                group_counts.append(len(idxs))

        colors = ["steelblue", "salmon", "orange", "green", "purple"]
        bars = ax.bar(group_labels, group_var_means, color=colors[:len(group_labels)])
        for bar, count in zip(bars, group_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"n={count}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Mean Feature Variance (pre-scaling)")
        ax.set_title("Mean Variance by Feature Group")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plot_path = output_dir / "feature_group_variance.png"
        plt.savefig(plot_path, dpi=150)
        print(f"  Saved plot: {plot_path}")
        plt.close()

        # Per-sensor bar chart (individual sensor names)
        all_sensors = OPERATIONAL_SENSORS + NON_FUNCTIONAL_SENSORS
        sensor_var_means = []
        sensor_labels = []
        sensor_colors = []
        for sensor in all_sensors:
            idxs = [i for i, col in enumerate(columns)
                    if col.startswith(sensor + ".") or col.startswith(sensor + "_")]
            # Only include if we matched sensor features (not machine features with same prefix)
            sensor_idxs = [i for i in idxs if feature_groups[i] in ("operational_sensor", "non_functional_sensor")]
            if sensor_idxs:
                sensor_var_means.append(np.mean(variances[sensor_idxs]))
                sensor_labels.append(sensor)
                sensor_colors.append("salmon" if sensor in NON_FUNCTIONAL_SENSORS else "steelblue")

        fig, ax = plt.subplots(figsize=(14, 5))
        bars = ax.bar(range(len(sensor_labels)), sensor_var_means, color=sensor_colors)
        ax.set_xticks(range(len(sensor_labels)))
        ax.set_xticklabels(sensor_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Mean Feature Variance (pre-scaling)")
        ax.set_title("Mean Variance per Sensor (blue = operational, red = non-functional)")
        # Add legend
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="steelblue", label="Operational (12)"),
                           Patch(color="salmon", label="Non-functional (4)")],
                  loc="lower right")
        plt.tight_layout()
        plot_path = output_dir / "per_sensor_variance.png"
        plt.savefig(plot_path, dpi=150)
        print(f"  Saved plot: {plot_path}")
        plt.close()

        # Per-modality PCA analysis
        print("\n" + "=" * 70)
        print("7b. PCA BY MODALITY (across all 16 sensors)")
        print("=" * 70)

        modality_pca_results = []
        for mod in MODALITIES:
            mod_idxs = [i for i, col in enumerate(columns) if col.endswith("." + mod)]
            if len(mod_idxs) < 2:
                continue
            X_mod = X_avg[:, mod_idxs]
            X_mod = np.nan_to_num(X_mod, nan=0.0, posinf=0.0, neginf=0.0)
            # Skip if zero variance
            if np.all(np.var(X_mod, axis=0) < 1e-10):
                continue
            scaler_mod = StandardScaler()
            X_mod_scaled = scaler_mod.fit_transform(X_mod)
            pca_mod = PCA().fit(X_mod_scaled)
            cumvar_mod = np.cumsum(pca_mod.explained_variance_ratio_)
            n95_mod = np.argmax(cumvar_mod >= 0.95) + 1
            n90_mod = np.argmax(cumvar_mod >= 0.90) + 1
            pc1_var = pca_mod.explained_variance_ratio_[0]
            modality_pca_results.append((mod, len(mod_idxs), n90_mod, n95_mod, pc1_var))
            print(f"  {mod:15s}  sensors={len(mod_idxs):2d}  90%={n90_mod:2d} comps  95%={n95_mod:2d} comps  PC1={pc1_var:.1%}")

        # Bar chart: components needed for 95% variance per modality
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        mod_names = [r[0] for r in modality_pca_results]
        mod_n95 = [r[3] for r in modality_pca_results]
        mod_n_sensors = [r[1] for r in modality_pca_results]
        mod_pc1 = [r[4] for r in modality_pca_results]

        # Left: components for 95%
        bars = axes[0].bar(range(len(mod_names)), mod_n95, color="steelblue")
        for bar, ns in zip(bars, mod_n_sensors):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"/{ns}", ha="center", va="bottom", fontsize=8, color="gray")
        axes[0].set_xticks(range(len(mod_names)))
        axes[0].set_xticklabels(mod_names, rotation=45, ha="right", fontsize=9)
        axes[0].set_ylabel("Components for 95% Variance")
        axes[0].set_title("PCA per Modality: Components Needed for 95% Variance\n(label = /total sensors)")

        # Right: PC1 variance explained
        bars = axes[1].bar(range(len(mod_names)), [v * 100 for v in mod_pc1], color="orange")
        axes[1].set_xticks(range(len(mod_names)))
        axes[1].set_xticklabels(mod_names, rotation=45, ha="right", fontsize=9)
        axes[1].set_ylabel("PC1 Variance Explained (%)")
        axes[1].set_title("PCA per Modality: How Much PC1 Captures\n(higher = more redundant across sensors)")
        axes[1].axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50%")
        axes[1].legend()

        plt.tight_layout()
        plot_path = output_dir / "per_modality_pca.png"
        plt.savefig(plot_path, dpi=150)
        print(f"  Saved plot: {plot_path}")
        plt.close()

    except ImportError:
        print("\n  matplotlib not available, skipping plots")

    # ================================================================
    # 8. Summary JSON
    # ================================================================
    summary = {
        "total_features": 296,
        "feature_groups": {g: len(feats) for g, feats in groups.items()},
        "pca_all_296": {
            "90pct_components": int(np.argmax(cumvar_full >= 0.90) + 1),
            "95pct_components": int(np.argmax(cumvar_full >= 0.95) + 1),
            "99pct_components": int(np.argmax(cumvar_full >= 0.99) + 1),
        },
        "pca_228_no_nonfunctional": {
            "90pct_components": int(np.argmax(cumvar_clean >= 0.90) + 1),
            "95pct_components": int(np.argmax(cumvar_clean >= 0.95) + 1),
            "99pct_components": int(np.argmax(cumvar_clean >= 0.99) + 1),
        },
        "non_functional_sensor_stats": {
            "n_features": len(groups.get("non_functional_sensor", [])),
            "mean_variance": float(np.mean(variances[[i for i, g in enumerate(feature_groups) if g == "non_functional_sensor"]])),
            "mean_zero_rate": float(np.mean(zero_rates[[i for i, g in enumerate(feature_groups) if g == "non_functional_sensor"]])),
        },
    }

    summary_path = output_dir / "pca_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary: {summary_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    run_analysis()
