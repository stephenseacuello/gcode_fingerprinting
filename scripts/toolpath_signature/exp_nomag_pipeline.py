"""Experiment A: Full pipeline re-run without magnetometers.
All 4 isolation methods × 3 classifiers × repeated CV × noise robustness.
Uses only non-confounded channels: accel + gyro + RMS + temp + electrical.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolpath_signature import config as cfg, utils
warnings.filterwarnings("ignore")

CONFOUND_FREE = [".Ax",".Ay",".Az",".Gx",".Gy",".Gz",".RMS",".Temperature"]

def select_clean_features(feature_cols):
    """Select only non-confounded features (no mag, color, pressure, proximity)."""
    clean = []
    for f in feature_cols:
        # Keep electrical features
        if any(e in f for e in cfg.ELECTRICAL_COLS):
            clean.append(f); continue
        # Keep accel, gyro, RMS, temp from Arduino modules
        if any(ch in f for ch in CONFOUND_FREE):
            clean.append(f)
    return clean

def cv_accuracy(X, y, clf, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        s = StandardScaler()
        Xtr = np.nan_to_num(s.fit_transform(X[tr]))
        Xte = np.nan_to_num(s.transform(X[te]))
        c = type(clf)(**clf.get_params())
        c.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], c.predict(Xte)))
    return accs

def run():
    print("="*70)
    print("EXPERIMENT A: Full Pipeline Without Magnetometers")
    print("="*70)

    classifiers = {
        "RF": RandomForestClassifier(**cfg.RF_PARAMS),
        "SVM": SVC(**cfg.SVM_PARAMS),
        "MLP": MLPClassifier(**cfg.MLP_PARAMS),
    }
    results = {}

    for iso in cfg.ISOLATION_METHODS:
        df = utils.load_parquet(cfg.RUN_VECTORS_DIR / f"{iso}_temporal_summary.parquet")
        meta_cols = ["toolpath","run_id","prefix"]
        feature_cols = [c for c in df.columns if c not in meta_cols]
        clean_cols = select_clean_features(feature_cols)
        X = np.nan_to_num(df[clean_cols].values.astype(np.float64))
        le = LabelEncoder()
        y = le.fit_transform(df["toolpath"].values)

        for clf_name, clf in classifiers.items():
            # 20×5 repeated CV
            all_accs = []
            for rep in range(20):
                all_accs.extend(cv_accuracy(X, y, clf, seed=cfg.RANDOM_STATE+rep))

            key = f"{iso}_{clf_name}"
            results[key] = {
                "mean": float(np.mean(all_accs)),
                "std": float(np.std(all_accs)),
                "min": float(np.min(all_accs)),
                "max": float(np.max(all_accs)),
                "n": len(all_accs),
                "n_features": len(clean_cols),
            }
            print(f"  {key:25s}: {np.mean(all_accs):.3f}±{np.std(all_accs):.3f} "
                  f"(min={np.min(all_accs):.3f}) [{len(clean_cols)} features]")

    # Noise robustness (raw vs subtracted, clean features only)
    print("\n  Noise robustness (clean features):")
    noise_results = {}
    for iso in ["raw", "subtracted"]:
        df = utils.load_parquet(cfg.RUN_VECTORS_DIR / f"{iso}_temporal_summary.parquet")
        meta_cols = ["toolpath","run_id","prefix"]
        feature_cols = [c for c in df.columns if c not in meta_cols]
        clean_cols = select_clean_features(feature_cols)
        X = np.nan_to_num(df[clean_cols].values.astype(np.float64))
        le = LabelEncoder()
        y = le.fit_transform(df["toolpath"].values)
        feat_stds = np.std(X, axis=0)
        feat_stds = np.where(feat_stds < 1e-10, 1.0, feat_stds)

        iso_noise = {}
        for noise_level in [0.0, 0.1, 0.5, 1.0, 2.0]:
            accs = []
            for rep in range(10):
                rng = np.random.RandomState(cfg.RANDOM_STATE + rep)
                noise = rng.randn(*X.shape) * feat_stds * noise_level
                a = cv_accuracy(X + noise, y, RandomForestClassifier(**cfg.RF_PARAMS))
                accs.extend(a)
            iso_noise[str(noise_level)] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
            print(f"    {iso} noise={noise_level}: {np.mean(accs):.3f}±{np.std(accs):.3f}")
        noise_results[iso] = iso_noise

    results["noise_robustness_clean"] = noise_results

    out = cfg.RESULTS_DIR / "experiment_a_nomag.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    run()
