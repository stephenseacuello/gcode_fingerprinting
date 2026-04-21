"""Additional experiments to address reviewer concerns.

A. Trivial baseline classifier (mean axis position/velocity)
C. Repeated CV (20 repeats × 5-fold) for tighter statistics
D. Trivial feature sanity checks (position-only, gcode-line-range-only)
E. PCA ablation with SVM + explanation
G. ROCKET time-series classifier comparison
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolpath_signature import config as cfg
from toolpath_signature import utils

warnings.filterwarnings("ignore")


def cv_with_clf(X, y, clf, n_splits=5, random_state=42):
    """Single round of stratified k-fold CV."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = np.nan_to_num(scaler.fit_transform(X[train_idx]), nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(scaler.transform(X[test_idx]), nan=0.0, posinf=0.0, neginf=0.0)
        clf_copy = type(clf)(**clf.get_params())
        clf_copy.fit(X_train, y[train_idx])
        accs.append(accuracy_score(y[test_idx], clf_copy.predict(X_test)))
    return np.mean(accs), np.std(accs)


# ═══════════════════════════════════════════════════════════════════
# Experiment A: Trivial baselines
# ═══════════════════════════════════════════════════════════════════

def experiment_a_trivial_baselines():
    """Classify using trivially simple features to establish task difficulty."""
    print("=" * 70)
    print("EXPERIMENT A: Trivial Baseline Classifiers")
    print("=" * 70)

    le = LabelEncoder()
    results = {}

    # Load raw CSVs to extract trivial features
    trivial_rows = []
    for toolpath in cfg.TOOLPATH_TYPES:
        prefix = cfg.ACTIVE_PREFIX[toolpath]
        files = utils.discover_runs(prefix)
        for fpath in files:
            run_id = utils.extract_run_id(fpath)
            df = pd.read_csv(fpath)

            # Trivial feature sets
            row = {"toolpath": toolpath, "run_id": run_id}

            # A1: Mean axis positions
            for col in ["posx", "posy", "posz"]:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    row[f"{col}_mean"] = vals.mean()
                    row[f"{col}_std"] = vals.std()
                    row[f"{col}_min"] = vals.min()
                    row[f"{col}_max"] = vals.max()
                    row[f"{col}_range"] = vals.max() - vals.min()

            # A2: Mean velocity (approximate from feed column)
            if "feed" in df.columns:
                feed = pd.to_numeric(df["feed"], errors="coerce").dropna()
                row["feed_mean"] = feed.mean()
                row["feed_std"] = feed.std()

            # A3: G-code line range
            if "gcode_line" in df.columns:
                gl = df["gcode_line"].dropna()
                row["gcode_line_min"] = gl.min()
                row["gcode_line_max"] = gl.max()
                row["gcode_line_range"] = gl.max() - gl.min()
                row["gcode_line_nunique"] = gl.nunique()

            # A4: Row count (program duration)
            row["n_rows"] = len(df)

            trivial_rows.append(row)

    trivial_df = pd.DataFrame(trivial_rows)
    y = le.fit_transform(trivial_df["toolpath"].values)
    rf = RandomForestClassifier(**cfg.RF_PARAMS)

    # Test different trivial feature sets
    feature_sets = {
        "position_mean_only": ["posx_mean", "posy_mean", "posz_mean"],
        "position_stats": [c for c in trivial_df.columns if c.startswith("pos")],
        "gcode_line_range": ["gcode_line_min", "gcode_line_max", "gcode_line_range", "gcode_line_nunique"],
        "row_count_only": ["n_rows"],
        "feed_only": ["feed_mean", "feed_std"],
        "all_trivial": [c for c in trivial_df.columns if c not in ["toolpath", "run_id"]],
    }

    for name, cols in feature_sets.items():
        available = [c for c in cols if c in trivial_df.columns]
        if not available:
            print(f"  {name}: no features available, skipping")
            continue
        X = trivial_df[available].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0)
        acc, std = cv_with_clf(X, y, rf)
        results[name] = {"accuracy": float(acc), "std": float(std), "n_features": len(available),
                         "features": available}
        print(f"  {name}: {acc:.3f}±{std:.3f} ({len(available)} features: {available[:5]}...)")

    # Also test: random features (chance level sanity)
    X_random = np.random.RandomState(42).randn(len(y), 10)
    acc, std = cv_with_clf(X_random, y, rf)
    results["random_noise"] = {"accuracy": float(acc), "std": float(std), "n_features": 10}
    print(f"  random_noise: {acc:.3f}±{std:.3f} (10 random features)")

    return results


# ═══════════════════════════════════════════════════════════════════
# Experiment C: Repeated CV
# ═══════════════════════════════════════════════════════════════════

def experiment_c_repeated_cv():
    """20 repeats × 5-fold CV for all methods × classifiers."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Repeated CV (20 × 5-fold)")
    print("=" * 70)

    n_repeats = 20
    results = {}

    classifiers = {
        "RF": RandomForestClassifier(**cfg.RF_PARAMS),
        "SVM": SVC(**cfg.SVM_PARAMS),
        "MLP": MLPClassifier(**cfg.MLP_PARAMS),
    }

    for iso_method in cfg.ISOLATION_METHODS:
        df = utils.load_parquet(cfg.RUN_VECTORS_DIR / f"{iso_method}_temporal_summary.parquet")
        meta_cols = ["toolpath", "run_id", "prefix"]
        feature_cols = [c for c in df.columns if c not in meta_cols]
        X = np.nan_to_num(df[feature_cols].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        le = LabelEncoder()
        y = le.fit_transform(df["toolpath"].values)

        for clf_name, clf in classifiers.items():
            all_accs = []
            for rep in range(n_repeats):
                seed = cfg.RANDOM_STATE + rep
                skf = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=seed)
                for train_idx, test_idx in skf.split(X, y):
                    scaler = StandardScaler()
                    X_train = np.nan_to_num(scaler.fit_transform(X[train_idx]), nan=0.0, posinf=0.0, neginf=0.0)
                    X_test = np.nan_to_num(scaler.transform(X[test_idx]), nan=0.0, posinf=0.0, neginf=0.0)
                    c = type(clf)(**clf.get_params())
                    c.fit(X_train, y[train_idx])
                    all_accs.append(accuracy_score(y[test_idx], c.predict(X_test)))

            key = f"{iso_method}_{clf_name}"
            results[key] = {
                "accuracy_mean": float(np.mean(all_accs)),
                "accuracy_std": float(np.std(all_accs)),
                "accuracy_min": float(np.min(all_accs)),
                "accuracy_max": float(np.max(all_accs)),
                "n_evaluations": len(all_accs),
                "isolation": iso_method,
                "classifier": clf_name,
            }
            print(f"  {key}: {np.mean(all_accs):.3f}±{np.std(all_accs):.3f} "
                  f"(min={np.min(all_accs):.3f}, max={np.max(all_accs):.3f}, n={len(all_accs)})")

    return results


# ═══════════════════════════════════════════════════════════════════
# Experiment D: Trivial feature sanity checks
# ═══════════════════════════════════════════════════════════════════

def experiment_d_sanity_checks():
    """Check if position/gcode info alone can classify."""
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Feature Sanity Checks")
    print("=" * 70)

    results = {}
    le = LabelEncoder()

    # D1: Single best sensor feature
    df = utils.load_parquet(cfg.RUN_VECTORS_DIR / "raw_temporal_summary.parquet")
    meta_cols = ["toolpath", "run_id", "prefix"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X_full = np.nan_to_num(df[feature_cols].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = le.fit_transform(df["toolpath"].values)
    rf = RandomForestClassifier(**cfg.RF_PARAMS)

    # D2: Single feature classification (top 5 from importance)
    imp_df = pd.read_csv(cfg.IMPORTANCE_DIR / "raw_rf_importance.csv")
    for _, row in imp_df.head(5).iterrows():
        feat = row["feature"]
        if feat in feature_cols:
            idx = feature_cols.index(feat)
            X_single = X_full[:, idx:idx+1]
            acc, std = cv_with_clf(X_single, y, rf)
            results[f"single_{feat}"] = {"accuracy": float(acc), "std": float(std)}
            print(f"  Single feature '{feat}': {acc:.3f}±{std:.3f}")

    # D3: Just electrical sensors (motor currents only)
    elec_features = [f for f in feature_cols if any(e in f for e in cfg.ELECTRICAL_COLS[:4])]
    if elec_features:
        X_elec = df[elec_features].values.astype(np.float64)
        X_elec = np.nan_to_num(X_elec, nan=0.0, posinf=0.0, neginf=0.0)
        acc, std = cv_with_clf(X_elec, y, rf)
        results["motor_current_only"] = {"accuracy": float(acc), "std": float(std),
                                          "n_features": len(elec_features)}
        print(f"  Motor current only: {acc:.3f}±{std:.3f} ({len(elec_features)} features)")

    return results


# ═══════════════════════════════════════════════════════════════════
# Experiment E: PCA ablation with SVM + explanation
# ═══════════════════════════════════════════════════════════════════

def experiment_e_pca_with_svm():
    """Re-run PCA ablation with SVM where component count should matter more."""
    print("\n" + "=" * 70)
    print("EXPERIMENT E: PCA Ablation with SVM")
    print("=" * 70)

    from sklearn.decomposition import PCA

    df = utils.load_parquet(cfg.RUN_VECTORS_DIR / "raw_temporal_summary.parquet")
    meta_cols = ["toolpath", "run_id", "prefix"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = np.nan_to_num(df[feature_cols].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    le = LabelEncoder()
    y = le.fit_transform(df["toolpath"].values)

    results = {}
    max_train = X.shape[0] - X.shape[0] // cfg.CV_FOLDS

    for clf_name, clf in [("RF", RandomForestClassifier(**cfg.RF_PARAMS)),
                           ("SVM", SVC(**cfg.SVM_PARAMS)),
                           ("MLP", MLPClassifier(**cfg.MLP_PARAMS))]:
        clf_results = {}

        # No PCA
        acc, std = cv_with_clf(X, y, clf)
        clf_results["no_pca"] = {"accuracy": float(acc), "std": float(std), "n_components": X.shape[1]}
        print(f"  {clf_name} No PCA: {acc:.3f}±{std:.3f}")

        for n_comp in [2, 3, 5, 10, 15, 20, 30]:
            if n_comp >= max_train:
                continue
            skf = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
            accs = []
            for train_idx, test_idx in skf.split(X, y):
                scaler = StandardScaler()
                X_train = np.nan_to_num(scaler.fit_transform(X[train_idx]), nan=0.0, posinf=0.0, neginf=0.0)
                X_test = np.nan_to_num(scaler.transform(X[test_idx]), nan=0.0, posinf=0.0, neginf=0.0)
                pca = PCA(n_components=n_comp, random_state=cfg.RANDOM_STATE)
                X_train_p = pca.fit_transform(X_train)
                X_test_p = pca.transform(X_test)
                c = type(clf)(**clf.get_params())
                c.fit(X_train_p, y[train_idx])
                accs.append(accuracy_score(y[test_idx], c.predict(X_test_p)))

            clf_results[f"pca_{n_comp}"] = {
                "accuracy": float(np.mean(accs)), "std": float(np.std(accs)),
                "n_components": n_comp
            }
            print(f"  {clf_name} PCA {n_comp}: {np.mean(accs):.3f}±{np.std(accs):.3f}")

        results[clf_name] = clf_results

    return results


# ═══════════════════════════════════════════════════════════════════
# Experiment G: ROCKET time-series classifier
# ═══════════════════════════════════════════════════════════════════

def experiment_g_rocket():
    """Compare against ROCKET-style random convolutional features on raw time series."""
    print("\n" + "=" * 70)
    print("EXPERIMENT G: ROCKET-style Time-Series Classification")
    print("=" * 70)

    # ROCKET uses random convolutional kernels on raw time series.
    # We approximate ROCKET with random 1D convolutions since sktime may not be installed.
    from sklearn.linear_model import RidgeClassifierCV

    le = LabelEncoder()
    results = {}

    # Load raw time series for each run, pad/truncate to fixed length
    all_series = []
    all_labels = []

    for toolpath in cfg.TOOLPATH_TYPES:
        prefix = cfg.ACTIVE_PREFIX[toolpath]
        files = utils.discover_runs(prefix)
        for fpath in files:
            df = utils.load_run(fpath)
            sensor_data = df[[c for c in cfg.SENSOR_COLS if c in df.columns]].values
            all_series.append(sensor_data)
            all_labels.append(toolpath)

    y = le.fit_transform(all_labels)

    # Pad/truncate to common length
    max_len = max(s.shape[0] for s in all_series)
    n_channels = all_series[0].shape[1]

    # Strategy 1: Pad to max length with last value
    X_padded = np.zeros((len(all_series), max_len, n_channels))
    for i, s in enumerate(all_series):
        X_padded[i, :s.shape[0], :] = s
        if s.shape[0] < max_len:
            X_padded[i, s.shape[0]:, :] = s[-1, :]  # pad with last value

    # Generate random convolutional features (ROCKET-style)
    n_kernels = 1000
    rng = np.random.RandomState(cfg.RANDOM_STATE)

    kernel_lengths = rng.choice([7, 9, 11], size=n_kernels)
    kernel_channels = rng.randint(0, n_channels, size=n_kernels)
    kernel_dilations = rng.choice([1, 2, 4], size=n_kernels)
    kernel_biases = rng.uniform(-1, 1, size=n_kernels)

    # Apply random kernels and extract max + proportion of positive values (ppv)
    X_rocket = np.zeros((len(all_series), n_kernels * 2))

    for k in range(n_kernels):
        klen = kernel_lengths[k]
        ch = kernel_channels[k]
        dil = kernel_dilations[k]
        bias = kernel_biases[k]

        # Random kernel weights
        weights = rng.randn(klen)
        weights = weights - weights.mean()  # zero-mean

        for i in range(len(all_series)):
            signal = X_padded[i, :, ch]
            # Dilated convolution
            effective_len = (klen - 1) * dil + 1
            if effective_len > len(signal):
                conv_out = np.array([0.0])
            else:
                conv_out = np.zeros(len(signal) - effective_len + 1)
                for j in range(len(conv_out)):
                    indices = np.arange(klen) * dil + j
                    conv_out[j] = np.dot(signal[indices], weights) + bias

            X_rocket[i, k * 2] = np.max(conv_out)  # max pooling
            X_rocket[i, k * 2 + 1] = np.mean(conv_out > 0)  # ppv

    X_rocket = np.nan_to_num(X_rocket, nan=0.0, posinf=0.0, neginf=0.0)

    # Classify with Ridge (standard ROCKET approach)
    ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    acc_ridge, std_ridge = cv_with_clf(X_rocket, y, ridge)
    results["rocket_ridge"] = {"accuracy": float(acc_ridge), "std": float(std_ridge),
                                "n_features": n_kernels * 2}
    print(f"  ROCKET (1000 kernels) + Ridge: {acc_ridge:.3f}±{std_ridge:.3f}")

    # Also try ROCKET features with RF
    rf = RandomForestClassifier(**cfg.RF_PARAMS)
    acc_rf, std_rf = cv_with_clf(X_rocket, y, rf)
    results["rocket_rf"] = {"accuracy": float(acc_rf), "std": float(std_rf),
                             "n_features": n_kernels * 2}
    print(f"  ROCKET (1000 kernels) + RF: {acc_rf:.3f}±{std_rf:.3f}")

    # Baseline: flatten a downsampled version of the raw series
    # Downsample to 50 timesteps
    ds_len = 50
    X_flat = np.zeros((len(all_series), ds_len * n_channels))
    for i, s in enumerate(all_series):
        indices = np.linspace(0, s.shape[0] - 1, ds_len, dtype=int)
        X_flat[i] = s[indices].flatten()

    X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
    acc_flat, std_flat = cv_with_clf(X_flat, y, rf)
    results["raw_downsampled_rf"] = {"accuracy": float(acc_flat), "std": float(std_flat),
                                      "n_features": ds_len * n_channels}
    print(f"  Raw downsampled (50 steps) + RF: {acc_flat:.3f}±{std_flat:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════

def run_all_additional():
    all_results = {}
    all_results["trivial_baselines"] = experiment_a_trivial_baselines()
    all_results["repeated_cv"] = experiment_c_repeated_cv()
    all_results["sanity_checks"] = experiment_d_sanity_checks()
    all_results["pca_multiclf"] = experiment_e_pca_with_svm()
    all_results["rocket"] = experiment_g_rocket()

    # Save
    out_path = cfg.RESULTS_DIR / "additional_experiments.json"
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nAll additional results saved to {out_path}")


if __name__ == "__main__":
    run_all_additional()
