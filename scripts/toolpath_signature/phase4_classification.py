"""Phase 4: Classification with RF, SVM, MLP — 5-fold stratified CV at run level."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolpath_signature import config as cfg
from toolpath_signature import utils


def get_classifier(name: str):
    if name == "RF":
        return RandomForestClassifier(**cfg.RF_PARAMS)
    elif name == "SVM":
        return SVC(**cfg.SVM_PARAMS)
    elif name == "MLP":
        return MLPClassifier(**cfg.MLP_PARAMS)
    else:
        raise ValueError(f"Unknown classifier: {name}")


def run_classification(X, y, clf_name, use_pca=False, pca_variance=0.95):
    """Run stratified k-fold CV, return metrics dict."""
    skf = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)

    fold_metrics = []
    all_cm = np.zeros((3, 3), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Handle NaN/Inf from scaling
        X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

        # Optional PCA
        if use_pca:
            pca = PCA(n_components=pca_variance, random_state=cfg.RANDOM_STATE)
            X_train_s = pca.fit_transform(X_train_s)
            X_test_s = pca.transform(X_test_s)

        clf = get_classifier(clf_name)
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        fold_metrics.append({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        })
        all_cm += confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    # Aggregate
    metrics_df = pd.DataFrame(fold_metrics)
    result = {
        "accuracy_mean": metrics_df["accuracy"].mean(),
        "accuracy_std": metrics_df["accuracy"].std(),
        "precision_mean": metrics_df["precision"].mean(),
        "recall_mean": metrics_df["recall"].mean(),
        "f1_mean": metrics_df["f1"].mean(),
        "f1_std": metrics_df["f1"].std(),
        "confusion_matrix": all_cm.tolist(),
        "fold_details": fold_metrics,
    }
    return result


def run_phase4():
    print("=" * 70)
    print("PHASE 4: Classification")
    print("=" * 70)

    classifier_names = ["RF", "SVM", "MLP"]
    all_results = {}
    summary_rows = []

    for iso_method in cfg.ISOLATION_METHODS:
        # Load temporal summary vectors
        df = utils.load_parquet(cfg.RUN_VECTORS_DIR / f"{iso_method}_temporal_summary.parquet")
        meta_cols = ["toolpath", "run_id", "prefix"]
        feature_cols = [c for c in df.columns if c not in meta_cols]

        X = df[feature_cols].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        le = LabelEncoder()
        y = le.fit_transform(df["toolpath"].values)
        class_names = le.classes_.tolist()

        for clf_name in classifier_names:
            for use_pca in [False, True]:
                pca_label = "pca" if use_pca else "nopca"
                config_key = f"{iso_method}_{clf_name}_{pca_label}"
                print(f"  Running {config_key}...", end=" ", flush=True)

                result = run_classification(X, y, clf_name, use_pca=use_pca)
                result["isolation"] = iso_method
                result["classifier"] = clf_name
                result["pca"] = use_pca
                result["class_names"] = class_names
                all_results[config_key] = result

                print(f"Acc={result['accuracy_mean']:.3f}±{result['accuracy_std']:.3f}  "
                      f"F1={result['f1_mean']:.3f}")

                summary_rows.append({
                    "isolation": iso_method,
                    "classifier": clf_name,
                    "pca": use_pca,
                    "accuracy_mean": result["accuracy_mean"],
                    "accuracy_std": result["accuracy_std"],
                    "precision_mean": result["precision_mean"],
                    "recall_mean": result["recall_mean"],
                    "f1_mean": result["f1_mean"],
                    "f1_std": result["f1_std"],
                })

    # Save all results
    results_path = cfg.CLASSIFICATION_DIR / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(cfg.CLASSIFICATION_DIR / "summary_table.csv", index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("CLASSIFICATION SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<12} {'Clf':<5} {'PCA':<5} {'Accuracy':>12} {'F1':>12}")
    print("-" * 50)
    for _, row in summary_df.sort_values("accuracy_mean", ascending=False).iterrows():
        print(f"{row['isolation']:<12} {row['classifier']:<5} "
              f"{'Yes' if row['pca'] else 'No':<5} "
              f"{row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}  "
              f"{row['f1_mean']:.3f}±{row['f1_std']:.3f}")

    # Best result
    best_idx = summary_df["accuracy_mean"].idxmax()
    best = summary_df.iloc[best_idx]
    print(f"\nBest: {best['isolation']} + {best['classifier']} "
          f"(PCA={'Yes' if best['pca'] else 'No'}) → "
          f"{best['accuracy_mean']:.3f}±{best['accuracy_std']:.3f}")

    print("\nPhase 4 complete.")


if __name__ == "__main__":
    run_phase4()
