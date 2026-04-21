"""Phase 6: Leave-K-runs-out cross-validation for generalization analysis."""

import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolpath_signature import config as cfg
from toolpath_signature import utils

warnings.filterwarnings("ignore")


def leave_k_out_cv(X, y, k=1, max_combos=100, random_state=42):
    """Leave-K-runs-out CV. For large K, sample random combinations."""
    n = len(y)
    indices = np.arange(n)

    if k == 1:
        # Standard leave-one-out
        combos = [(i,) for i in range(n)]
    else:
        all_combos = list(combinations(range(n), k))
        if len(all_combos) > max_combos:
            rng = np.random.RandomState(random_state)
            combo_idx = rng.choice(len(all_combos), max_combos, replace=False)
            combos = [all_combos[i] for i in combo_idx]
        else:
            combos = all_combos

    accs, f1s = [], []
    for test_indices in combos:
        test_mask = np.zeros(n, dtype=bool)
        test_mask[list(test_indices)] = True
        train_mask = ~test_mask

        # Need at least 2 classes in training
        if len(np.unique(y[train_mask])) < 3:
            continue

        scaler = StandardScaler()
        X_train = np.nan_to_num(scaler.fit_transform(X[train_mask]), nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(scaler.transform(X[test_mask]), nan=0.0, posinf=0.0, neginf=0.0)

        clf = RandomForestClassifier(**cfg.RF_PARAMS)
        clf.fit(X_train, y[train_mask])
        y_pred = clf.predict(X_test)

        accs.append(accuracy_score(y[test_mask], y_pred))
        f1s.append(f1_score(y[test_mask], y_pred, average="macro", zero_division=0))

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "accuracy_min": float(np.min(accs)),
        "accuracy_max": float(np.max(accs)),
        "f1_mean": float(np.mean(f1s)),
        "n_evaluations": len(accs),
    }


def run_phase6():
    print("=" * 70)
    print("PHASE 6: Generalization (Leave-K-Runs-Out)")
    print("=" * 70)

    results = {}

    for iso_method in cfg.ISOLATION_METHODS:
        print(f"\n── {iso_method} ──")
        df = utils.load_parquet(cfg.RUN_VECTORS_DIR / f"{iso_method}_temporal_summary.parquet")
        meta_cols = ["toolpath", "run_id", "prefix"]
        feature_cols = [c for c in df.columns if c not in meta_cols]

        X = df[feature_cols].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        le = LabelEncoder()
        y = le.fit_transform(df["toolpath"].values)

        method_results = {}
        for k in [1, 2, 3]:
            res = leave_k_out_cv(X, y, k=k)
            method_results[f"leave_{k}_out"] = res
            print(f"  Leave-{k}-out: Acc={res['accuracy_mean']:.3f}±{res['accuracy_std']:.3f} "
                  f"(min={res['accuracy_min']:.3f}, max={res['accuracy_max']:.3f}, "
                  f"n={res['n_evaluations']})")

        # Per-class analysis with leave-1-out
        per_class = {}
        for cls_idx, cls_name in enumerate(le.classes_):
            cls_mask = y == cls_idx
            cls_indices = np.where(cls_mask)[0]
            cls_accs = []
            for test_idx in cls_indices:
                train_mask = np.ones(len(y), dtype=bool)
                train_mask[test_idx] = False
                scaler = StandardScaler()
                X_train = np.nan_to_num(scaler.fit_transform(X[train_mask]), nan=0.0, posinf=0.0, neginf=0.0)
                X_test = np.nan_to_num(scaler.transform(X[[test_idx]]), nan=0.0, posinf=0.0, neginf=0.0)
                clf = RandomForestClassifier(**cfg.RF_PARAMS)
                clf.fit(X_train, y[train_mask])
                cls_accs.append(int(clf.predict(X_test)[0] == y[test_idx]))
            per_class[cls_name] = {
                "accuracy": float(np.mean(cls_accs)),
                "n_runs": len(cls_indices),
            }
            print(f"  Per-class L1O {cls_name}: {np.mean(cls_accs):.3f} ({len(cls_indices)} runs)")

        method_results["per_class_leave1out"] = per_class
        results[iso_method] = method_results

    # Save
    out_path = cfg.GENERALIZATION_DIR / "leave_k_out.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print("Phase 6 complete.")


if __name__ == "__main__":
    run_phase6()
