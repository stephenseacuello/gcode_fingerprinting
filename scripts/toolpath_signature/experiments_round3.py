"""Round 3 experiments:
1. PCA ablation: identify which samples are misclassified at each component count
2. Harder classification tasks:
   a. Leave-one-type-out (train on 2 types, predict 3rd)
   b. One-vs-rest binary discrimination within each toolpath type
   c. Air-vs-active discrimination (can sensor data tell if material is present?)
   d. Cross-condition transfer: train on air, predict active (or vice versa)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolpath_signature import config as cfg
from toolpath_signature import utils

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
# Issue 1: PCA misclassified sample analysis
# ═══════════════════════════════════════════════════════════════════

def pca_misclassification_analysis():
    """Identify which samples RF misclassifies under PCA at each component count."""
    print("=" * 70)
    print("PCA MISCLASSIFICATION ANALYSIS")
    print("=" * 70)

    df = utils.load_parquet(cfg.RUN_VECTORS_DIR / "raw_temporal_summary.parquet")
    meta_cols = ["toolpath", "run_id", "prefix"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = np.nan_to_num(df[feature_cols].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    le = LabelEncoder()
    y = le.fit_transform(df["toolpath"].values)

    results = {}

    for n_comp in [0, 2, 3, 5, 10, 20, 30]:
        skf = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
        misclassified_indices = []
        misclassified_details = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            scaler = StandardScaler()
            X_train = np.nan_to_num(scaler.fit_transform(X[train_idx]), nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(scaler.transform(X[test_idx]), nan=0.0, posinf=0.0, neginf=0.0)

            if n_comp > 0:
                pca = PCA(n_components=n_comp, random_state=cfg.RANDOM_STATE)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            clf = RandomForestClassifier(**cfg.RF_PARAMS)
            clf.fit(X_train, y[train_idx])
            y_pred = clf.predict(X_test)

            for i, (true, pred) in enumerate(zip(y[test_idx], y_pred)):
                if true != pred:
                    global_idx = test_idx[i]
                    misclassified_indices.append(int(global_idx))
                    misclassified_details.append({
                        "global_idx": int(global_idx),
                        "fold": fold,
                        "true_class": le.classes_[true],
                        "pred_class": le.classes_[pred],
                        "run_id": df.iloc[global_idx]["run_id"],
                        "prefix": df.iloc[global_idx]["prefix"],
                    })

        label = f"pca_{n_comp}" if n_comp > 0 else "no_pca"
        results[label] = {
            "n_misclassified": len(misclassified_indices),
            "misclassified_indices": misclassified_indices,
            "details": misclassified_details,
            "accuracy": 1.0 - len(misclassified_indices) / len(y),
        }

        if misclassified_details:
            print(f"\n  {label}: {len(misclassified_indices)} misclassified")
            for d in misclassified_details:
                print(f"    idx={d['global_idx']} {d['prefix']}_{d['run_id']}: "
                      f"true={d['true_class']}, pred={d['pred_class']} (fold {d['fold']})")
        else:
            print(f"  {label}: 0 misclassified (perfect)")

    # Check: are the same samples misclassified across PCA settings?
    all_pca_sets = {}
    for label, r in results.items():
        if label != "no_pca":
            all_pca_sets[label] = set(r["misclassified_indices"])

    if all_pca_sets:
        common = set.intersection(*all_pca_sets.values()) if all_pca_sets else set()
        union = set.union(*all_pca_sets.values()) if all_pca_sets else set()
        print(f"\n  Samples misclassified in ALL PCA configs: {sorted(common)}")
        print(f"  Samples misclassified in ANY PCA config: {sorted(union)}")
        print(f"  Same samples every time: {common == union and len(common) > 0}")

    return results


# ═══════════════════════════════════════════════════════════════════
# Issue 2: Harder classification tasks
# ═══════════════════════════════════════════════════════════════════

def harder_task_air_vs_active():
    """6-class classification: 3 toolpath types × 2 conditions (air/active).
    Then test: can we distinguish air from active WITHIN the same toolpath type?"""
    print("\n" + "=" * 70)
    print("HARDER TASK: Air vs Active Discrimination")
    print("=" * 70)

    results = {}

    # Build feature vectors for ALL runs (air + active)
    all_rows = []
    all_meta = []

    for toolpath in cfg.TOOLPATH_TYPES:
        for condition, prefix_map in [("air", cfg.AIR_PREFIX), ("active", cfg.ACTIVE_PREFIX)]:
            prefix = prefix_map[toolpath]
            files = utils.discover_runs(prefix)
            for fpath in files:
                run_id = utils.extract_run_id(fpath)
                df = utils.load_run(fpath)
                bin_stats = utils.temporal_bin(df, n_bins=cfg.N_BINS)
                run_vec = utils.aggregate_bins_to_run(bin_stats, method="summary")
                all_rows.append(run_vec)
                all_meta.append({
                    "toolpath": toolpath,
                    "condition": condition,
                    "label_6class": f"{toolpath}_{condition}",
                    "run_id": run_id,
                })

    X_all = np.stack(all_rows)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    meta_df = pd.DataFrame(all_meta)

    rf = RandomForestClassifier(**cfg.RF_PARAMS)

    # Task 1: 6-class (toolpath × condition)
    le6 = LabelEncoder()
    y6 = le6.fit_transform(meta_df["label_6class"].values)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
    accs = []
    all_cm = np.zeros((6, 6), dtype=int)
    for train_idx, test_idx in skf.split(X_all, y6):
        scaler = StandardScaler()
        X_train = np.nan_to_num(scaler.fit_transform(X_all[train_idx]), nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(scaler.transform(X_all[test_idx]), nan=0.0, posinf=0.0, neginf=0.0)
        clf = RandomForestClassifier(**cfg.RF_PARAMS)
        clf.fit(X_train, y6[train_idx])
        pred = clf.predict(X_test)
        accs.append(accuracy_score(y6[test_idx], pred))
        all_cm += confusion_matrix(y6[test_idx], pred, labels=range(6))

    results["six_class"] = {
        "accuracy": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "classes": le6.classes_.tolist(),
        "confusion_matrix": all_cm.tolist(),
    }
    print(f"  6-class (toolpath×condition): {np.mean(accs):.3f}±{np.std(accs):.3f}")
    print(f"  Classes: {le6.classes_.tolist()}")

    # Task 2: Binary air vs active (ignoring toolpath type)
    le_cond = LabelEncoder()
    y_cond = le_cond.fit_transform(meta_df["condition"].values)
    accs = []
    for train_idx, test_idx in skf.split(X_all, y_cond):
        scaler = StandardScaler()
        X_train = np.nan_to_num(scaler.fit_transform(X_all[train_idx]), nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(scaler.transform(X_all[test_idx]), nan=0.0, posinf=0.0, neginf=0.0)
        clf = RandomForestClassifier(**cfg.RF_PARAMS)
        clf.fit(X_train, y_cond[train_idx])
        accs.append(accuracy_score(y_cond[test_idx], clf.predict(X_test)))

    results["binary_air_vs_active"] = {
        "accuracy": float(np.mean(accs)),
        "std": float(np.std(accs)),
    }
    print(f"  Binary air vs active: {np.mean(accs):.3f}±{np.std(accs):.3f}")

    # Task 3: Per-toolpath air vs active
    for toolpath in cfg.TOOLPATH_TYPES:
        mask = meta_df["toolpath"] == toolpath
        X_sub = X_all[mask]
        y_sub = le_cond.transform(meta_df[mask]["condition"].values)
        n_per_class = np.bincount(y_sub)
        n_folds = min(5, min(n_per_class))
        if n_folds < 2:
            print(f"  Air vs active ({toolpath}): too few samples, skipping")
            continue
        skf_sub = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.RANDOM_STATE)
        accs = []
        for train_idx, test_idx in skf_sub.split(X_sub, y_sub):
            scaler = StandardScaler()
            X_train = np.nan_to_num(scaler.fit_transform(X_sub[train_idx]), nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(scaler.transform(X_sub[test_idx]), nan=0.0, posinf=0.0, neginf=0.0)
            clf = RandomForestClassifier(**cfg.RF_PARAMS)
            clf.fit(X_train, y_sub[train_idx])
            accs.append(accuracy_score(y_sub[test_idx], clf.predict(X_test)))

        results[f"air_vs_active_{toolpath}"] = {
            "accuracy": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "n_air": int(n_per_class[0]),
            "n_active": int(n_per_class[1]),
        }
        print(f"  Air vs active ({toolpath}): {np.mean(accs):.3f}±{np.std(accs):.3f} "
              f"(air={n_per_class[0]}, active={n_per_class[1]})")

    # Task 4: Cross-condition transfer — train on air, test on active
    for toolpath in cfg.TOOLPATH_TYPES:
        # Train: classify 3 toolpath types using AIR data only
        air_mask = meta_df["condition"] == "air"
        active_mask = meta_df["condition"] == "active"

        le_tp = LabelEncoder()
        y_air = le_tp.fit_transform(meta_df[air_mask]["toolpath"].values)
        y_active = le_tp.transform(meta_df[active_mask]["toolpath"].values)

        scaler = StandardScaler()
        X_air = np.nan_to_num(scaler.fit_transform(X_all[air_mask]), nan=0.0, posinf=0.0, neginf=0.0)
        X_active = np.nan_to_num(scaler.transform(X_all[active_mask]), nan=0.0, posinf=0.0, neginf=0.0)

        clf = RandomForestClassifier(**cfg.RF_PARAMS)
        clf.fit(X_air, y_air)
        y_pred = clf.predict(X_active)
        acc = accuracy_score(y_active, y_pred)

        results["cross_condition_air_to_active"] = {
            "accuracy": float(acc),
            "n_train": int(air_mask.sum()),
            "n_test": int(active_mask.sum()),
        }
        print(f"\n  Cross-condition (train air → test active): {acc:.3f} "
              f"({air_mask.sum()} train, {active_mask.sum()} test)")

        # Reverse: train active, test air
        clf2 = RandomForestClassifier(**cfg.RF_PARAMS)
        clf2.fit(np.nan_to_num(scaler.fit_transform(X_all[active_mask]), nan=0.0, posinf=0.0, neginf=0.0),
                 y_active)
        y_pred2 = clf2.predict(np.nan_to_num(scaler.transform(X_all[air_mask]), nan=0.0, posinf=0.0, neginf=0.0))
        acc2 = accuracy_score(y_air, y_pred2)
        results["cross_condition_active_to_air"] = {
            "accuracy": float(acc2),
            "n_train": int(active_mask.sum()),
            "n_test": int(air_mask.sum()),
        }
        print(f"  Cross-condition (train active → test air): {acc2:.3f}")
        break  # Only need to do this once

    return results


def harder_task_leave_one_type_out():
    """Train on 2 toolpath types, predict the 3rd (unknown type detection)."""
    print("\n" + "=" * 70)
    print("HARDER TASK: Leave-One-Type-Out")
    print("=" * 70)

    df = utils.load_parquet(cfg.RUN_VECTORS_DIR / "raw_temporal_summary.parquet")
    meta_cols = ["toolpath", "run_id", "prefix"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = np.nan_to_num(df[feature_cols].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    results = {}

    for held_out in cfg.TOOLPATH_TYPES:
        remaining = [t for t in cfg.TOOLPATH_TYPES if t != held_out]

        # Train binary classifier on 2 types
        train_mask = df["toolpath"].isin(remaining).values
        test_mask = ~train_mask

        le = LabelEncoder()
        y_train = le.fit_transform(df[train_mask]["toolpath"].values)

        scaler = StandardScaler()
        X_train = np.nan_to_num(scaler.fit_transform(X[train_mask]), nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(scaler.transform(X[test_mask]), nan=0.0, posinf=0.0, neginf=0.0)

        clf = RandomForestClassifier(**cfg.RF_PARAMS)
        clf.fit(X_train, y_train)

        # Get probabilities for unknown type
        probs = clf.predict_proba(X_test)
        max_probs = np.max(probs, axis=1)
        pred_classes = clf.predict(X_test)

        results[f"held_out_{held_out}"] = {
            "held_out_type": held_out,
            "trained_on": remaining,
            "n_held_out_samples": int(test_mask.sum()),
            "mean_max_confidence": float(np.mean(max_probs)),
            "std_max_confidence": float(np.std(max_probs)),
            "min_max_confidence": float(np.min(max_probs)),
            "predictions": {le.classes_[c]: int((pred_classes == c).sum()) for c in range(len(le.classes_))},
        }
        print(f"  Held out '{held_out}' ({test_mask.sum()} samples):")
        print(f"    Trained on: {remaining}")
        print(f"    Predictions: {results[f'held_out_{held_out}']['predictions']}")
        print(f"    Mean confidence: {np.mean(max_probs):.3f}±{np.std(max_probs):.3f} "
              f"(min={np.min(max_probs):.3f})")

    return results


# ═══════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════

def run_all_round3():
    all_results = {}
    all_results["pca_misclassification"] = pca_misclassification_analysis()
    all_results["air_vs_active"] = harder_task_air_vs_active()
    all_results["leave_one_type_out"] = harder_task_leave_one_type_out()

    out_path = cfg.RESULTS_DIR / "round3_experiments.json"
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nAll round 3 results saved to {out_path}")


if __name__ == "__main__":
    run_all_round3()
