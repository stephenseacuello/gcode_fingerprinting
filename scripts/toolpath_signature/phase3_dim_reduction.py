"""Phase 3: PCA and Random Forest feature importance."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolpath_signature import config as cfg
from toolpath_signature import utils


def run_phase3():
    print("=" * 70)
    print("PHASE 3: Dimensionality Reduction & Feature Selection")
    print("=" * 70)

    for iso_method in cfg.ISOLATION_METHODS:
        print(f"\n── {iso_method} ──")

        # Load run vectors
        df = utils.load_parquet(cfg.RUN_VECTORS_DIR / f"{iso_method}_temporal_summary.parquet")
        meta_cols = ["toolpath", "run_id", "prefix"]
        feature_cols = [c for c in df.columns if c not in meta_cols]

        X = df[feature_cols].values.astype(np.float64)
        y = df["toolpath"].values

        # Handle any remaining NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # ── StandardScaler ──
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── PCA ──
        pca = PCA(random_state=cfg.RANDOM_STATE)
        X_pca_full = pca.fit_transform(X_scaled)

        # Find n_components for 95% variance
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n95 = np.searchsorted(cumvar, 0.95) + 1
        n90 = np.searchsorted(cumvar, 0.90) + 1

        print(f"  PCA: {n90} components for 90% var, {n95} components for 95% var")
        print(f"  First 5 components explain: {cumvar[4]*100:.1f}%")
        print(f"  First 10 components explain: {cumvar[min(9, len(cumvar)-1)]*100:.1f}%")

        # Save PCA results
        np.savez_compressed(
            cfg.PCA_DIR / f"{iso_method}_pca.npz",
            X_pca=X_pca_full[:, :n95],
            explained_variance_ratio=pca.explained_variance_ratio_,
            components=pca.components_[:n95],
            cumulative_variance=cumvar,
            n_components_95=n95,
            labels=y,
        )

        # Save explained variance as CSV
        var_df = pd.DataFrame({
            "component": range(1, len(cumvar) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": cumvar,
        })
        var_df.to_csv(cfg.PCA_DIR / f"{iso_method}_explained_variance.csv", index=False)

        # ── Random Forest Feature Importance ──
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        rf = RandomForestClassifier(**cfg.RF_PARAMS)
        rf.fit(X_scaled, y_enc)

        importances = rf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        # Build importance table
        imp_records = []
        for rank, idx in enumerate(sorted_idx):
            col_name = feature_cols[idx]
            imp_records.append({
                "rank": rank + 1,
                "feature": col_name,
                "importance": importances[idx],
            })

        imp_df = pd.DataFrame(imp_records)
        imp_df.to_csv(cfg.IMPORTANCE_DIR / f"{iso_method}_rf_importance.csv", index=False)

        print(f"  RF importance sum: {importances.sum():.4f}")
        print(f"  Top 10 features:")
        for _, row in imp_df.head(10).iterrows():
            print(f"    {row['rank']:3d}. {row['feature']:45s} {row['importance']:.4f}")

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    run_phase3()
