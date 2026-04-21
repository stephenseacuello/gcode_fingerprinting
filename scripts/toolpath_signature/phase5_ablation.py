"""Phase 5: Ablation studies — sensor groups, stats, learning curve, alignment, PCA."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolpath_signature import config as cfg
from toolpath_signature import utils

warnings.filterwarnings("ignore")


def cv_accuracy(X, y, n_splits=cfg.CV_FOLDS, random_state=cfg.RANDOM_STATE):
    """Quick 5-fold CV with RF, returns (mean_acc, std_acc, mean_f1)."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, f1s = [], []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        clf = RandomForestClassifier(**cfg.RF_PARAMS)
        clf.fit(X_train, y[train_idx])
        y_pred = clf.predict(X_test)
        accs.append(accuracy_score(y[test_idx], y_pred))
        f1s.append(f1_score(y[test_idx], y_pred, average="macro", zero_division=0))
    return np.mean(accs), np.std(accs), np.mean(f1s)


def load_best_method():
    """Load the best isolation method from Phase 4 results."""
    summary = pd.read_csv(cfg.CLASSIFICATION_DIR / "summary_table.csv")
    best_row = summary.loc[summary["accuracy_mean"].idxmax()]
    return best_row["isolation"]


def run_sensor_ablation(best_method):
    """5A: Sensor group ablation."""
    print("\n── 5A: Sensor Group Ablation ──")

    df = utils.load_parquet(cfg.RUN_VECTORS_DIR / f"{best_method}_temporal_summary.parquet")
    meta_cols = ["toolpath", "run_id", "prefix"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    le = LabelEncoder()
    y = le.fit_transform(df["toolpath"].values)

    results = {}

    # All sensors
    X_all = df[feature_cols].values
    acc, std, f1 = cv_accuracy(X_all, y)
    results["all_sensors"] = {"accuracy": acc, "std": std, "f1": f1, "n_features": X_all.shape[1]}
    print(f"  All sensors: {acc:.3f}±{std:.3f} ({X_all.shape[1]} features)")

    # Each sensor group
    for group_name, group_cols in cfg.SENSOR_GROUPS.items():
        # Find matching feature columns
        group_features = [f for f in feature_cols if any(
            f.startswith(f"binmean_{sc}_") or f.startswith(f"binstd_{sc}_")
            or f == f"binmean_{sc}" or f == f"binstd_{sc}"
            for sc in group_cols
        )]
        # More robust matching: check if any sensor col name is a substring
        group_features = []
        for f in feature_cols:
            for sc in group_cols:
                if sc in f:
                    group_features.append(f)
                    break

        if not group_features:
            print(f"  {group_name}: no matching features found, skipping")
            continue

        X_group = df[group_features].values
        acc, std, f1 = cv_accuracy(X_group, y)
        results[group_name] = {"accuracy": acc, "std": std, "f1": f1, "n_features": X_group.shape[1]}
        print(f"  {group_name}: {acc:.3f}±{std:.3f} ({X_group.shape[1]} features)")

    return results


def run_stat_ablation(best_method):
    """5B: Feature statistic ablation (2, 3, 7 stats)."""
    print("\n── 5B: Feature Statistic Ablation ──")

    df = utils.load_parquet(cfg.RUN_VECTORS_DIR / f"{best_method}_temporal_summary.parquet")
    meta_cols = ["toolpath", "run_id", "prefix"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    le = LabelEncoder()
    y = le.fit_transform(df["toolpath"].values)

    stat_sets = {
        "mean_rms": ["mean", "rms"],
        "mean_rms_std": ["mean", "rms", "std"],
        "all_7_stats": cfg.BLOCK_STATS,
    }

    results = {}
    for label, stat_list in stat_sets.items():
        selected = [f for f in feature_cols if any(f"_{s}" in f for s in stat_list)]
        if not selected:
            continue
        X_sub = df[selected].values
        acc, std, f1 = cv_accuracy(X_sub, y)
        results[label] = {"accuracy": acc, "std": std, "f1": f1, "n_features": X_sub.shape[1]}
        print(f"  {label}: {acc:.3f}±{std:.3f} ({X_sub.shape[1]} features)")

    return results


def run_learning_curve(best_method):
    """5C: Learning curve — accuracy vs training runs per class."""
    print("\n── 5C: Learning Curve ──")

    df = utils.load_parquet(cfg.RUN_VECTORS_DIR / f"{best_method}_temporal_summary.parquet")
    meta_cols = ["toolpath", "run_id", "prefix"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    le = LabelEncoder()
    y = le.fit_transform(df["toolpath"].values)

    # Min class size
    class_counts = np.bincount(y)
    min_class = class_counts.min()

    train_sizes = [3, 5, 8, 10, 13, min(15, min_class - 2)]
    train_sizes = [s for s in train_sizes if s < min_class]

    results = {}
    for n_train in train_sizes:
        accs = []
        for rep in range(10):
            rng = np.random.RandomState(cfg.RANDOM_STATE + rep)
            train_mask = np.zeros(len(y), dtype=bool)
            test_mask = np.zeros(len(y), dtype=bool)

            for cls in range(3):
                cls_idx = np.where(y == cls)[0]
                rng.shuffle(cls_idx)
                train_mask[cls_idx[:n_train]] = True
                test_mask[cls_idx[n_train:]] = True

            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]

            if len(X_test) == 0:
                continue

            scaler = StandardScaler()
            X_train_s = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0, posinf=0.0, neginf=0.0)
            X_test_s = np.nan_to_num(scaler.transform(X_test), nan=0.0, posinf=0.0, neginf=0.0)

            clf = RandomForestClassifier(**cfg.RF_PARAMS)
            clf.fit(X_train_s, y_train)
            accs.append(accuracy_score(y_test, clf.predict(X_test_s)))

        results[n_train] = {
            "accuracy_mean": np.mean(accs),
            "accuracy_std": np.std(accs),
            "n_repeats": len(accs),
        }
        print(f"  {n_train} runs/class: {np.mean(accs):.3f}±{np.std(accs):.3f}")

    return results


def run_alignment_ablation(best_method):
    """5D: Temporal binning vs whole-run summary."""
    print("\n── 5D: Alignment Ablation ──")

    le = LabelEncoder()
    results = {}

    for agg_type, suffix in [("temporal_binning", "temporal_summary"), ("whole_run", "wholerun")]:
        df = utils.load_parquet(cfg.RUN_VECTORS_DIR / f"{best_method}_{suffix}.parquet")
        meta_cols = ["toolpath", "run_id", "prefix"]
        feature_cols = [c for c in df.columns if c not in meta_cols]
        X = df[feature_cols].values.astype(np.float64)
        y = le.fit_transform(df["toolpath"].values)

        acc, std, f1 = cv_accuracy(X, y)
        results[agg_type] = {"accuracy": acc, "std": std, "f1": f1, "n_features": X.shape[1]}
        print(f"  {agg_type}: {acc:.3f}±{std:.3f} ({X.shape[1]} features)")

    return results


def run_pca_ablation(best_method):
    """5E: PCA ablation — various variance thresholds and fixed component counts."""
    print("\n── 5E: PCA Ablation ──")

    df = utils.load_parquet(cfg.RUN_VECTORS_DIR / f"{best_method}_temporal_summary.parquet")
    meta_cols = ["toolpath", "run_id", "prefix"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    le = LabelEncoder()
    y = le.fit_transform(df["toolpath"].values)

    results = {}

    # No PCA
    acc, std, f1 = cv_accuracy(X, y)
    results["no_pca"] = {"accuracy": acc, "std": std, "f1": f1, "n_components": X.shape[1]}
    print(f"  No PCA: {acc:.3f}±{std:.3f} ({X.shape[1]} features)")

    # Variance thresholds
    for var_thresh in [0.90, 0.95, 0.99]:
        skf = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
        accs = []
        n_comps = []
        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_train = np.nan_to_num(scaler.fit_transform(X[train_idx]), nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(scaler.transform(X[test_idx]), nan=0.0, posinf=0.0, neginf=0.0)
            pca = PCA(n_components=var_thresh, random_state=cfg.RANDOM_STATE)
            X_train_p = pca.fit_transform(X_train)
            X_test_p = pca.transform(X_test)
            n_comps.append(pca.n_components_)
            clf = RandomForestClassifier(**cfg.RF_PARAMS)
            clf.fit(X_train_p, y[train_idx])
            accs.append(accuracy_score(y[test_idx], clf.predict(X_test_p)))

        label = f"pca_{int(var_thresh*100)}pct"
        results[label] = {
            "accuracy": np.mean(accs), "std": np.std(accs),
            "n_components": int(np.mean(n_comps)),
        }
        print(f"  PCA {var_thresh*100:.0f}%: {np.mean(accs):.3f}±{np.std(accs):.3f} "
              f"({int(np.mean(n_comps))} components)")

    # Fixed component counts
    max_train = X.shape[0] - X.shape[0] // cfg.CV_FOLDS  # approx train size
    for n_comp in [5, 10, 20, 50]:
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
            clf = RandomForestClassifier(**cfg.RF_PARAMS)
            clf.fit(X_train_p, y[train_idx])
            accs.append(accuracy_score(y[test_idx], clf.predict(X_test_p)))

        results[f"pca_{n_comp}comp"] = {
            "accuracy": np.mean(accs), "std": np.std(accs), "n_components": n_comp,
        }
        print(f"  PCA {n_comp} comp: {np.mean(accs):.3f}±{np.std(accs):.3f}")

    return results


def run_phase5():
    print("=" * 70)
    print("PHASE 5: Ablation Studies")
    print("=" * 70)

    best_method = load_best_method()
    print(f"Using best isolation method: {best_method}")

    all_results = {}
    all_results["best_method"] = best_method
    all_results["sensor_ablation"] = run_sensor_ablation(best_method)
    all_results["stat_ablation"] = run_stat_ablation(best_method)
    all_results["learning_curve"] = run_learning_curve(best_method)
    all_results["alignment_ablation"] = run_alignment_ablation(best_method)
    all_results["pca_ablation"] = run_pca_ablation(best_method)

    # Save
    out_path = cfg.ABLATION_DIR / "all_ablation_results.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nAll ablation results saved to {out_path}")
    print("Phase 5 complete.")


if __name__ == "__main__":
    run_phase5()
