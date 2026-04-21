"""Experiments B, C, D, E — lighter compute, run together.
B: Permutation test (1000 label shuffles)
C: Error analysis on pocket150025_001
D: Run-order temporal drift
E: Feature redundancy / effective dimensionality
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolpath_signature import config as cfg, utils
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
PALETTE = sns.color_palette("colorblind")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

def save_fig(fig, name):
    for ext in ["png","pdf"]:
        fig.savefig(cfg.FIGURES_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ═══════════════════════════════════════════════════════════════════
# B: Permutation test
# ═══════════════════════════════════════════════════════════════════
def experiment_b():
    print("="*70)
    print("B: PERMUTATION TEST (1000 shuffles)")
    print("="*70)

    df = utils.load_parquet(cfg.RUN_VECTORS_DIR / "raw_temporal_summary.parquet")
    meta = ["toolpath","run_id","prefix"]
    fcols = [c for c in df.columns if c not in meta]
    X = np.nan_to_num(df[fcols].values.astype(np.float64))
    le = LabelEncoder()
    y = le.fit_transform(df["toolpath"].values)

    # Observed accuracy
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    obs_accs = []
    for tr, te in skf.split(X, y):
        s = StandardScaler()
        Xtr = np.nan_to_num(s.fit_transform(X[tr]))
        Xte = np.nan_to_num(s.transform(X[te]))
        clf = RandomForestClassifier(**cfg.RF_PARAMS)
        clf.fit(Xtr, y[tr])
        obs_accs.append(accuracy_score(y[te], clf.predict(Xte)))
    obs_mean = np.mean(obs_accs)

    # Null distribution
    n_perm = 1000
    null_accs = []
    rng = np.random.RandomState(42)
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        accs = []
        for tr, te in skf.split(X, y_shuf):
            s = StandardScaler()
            Xtr = np.nan_to_num(s.fit_transform(X[tr]))
            Xte = np.nan_to_num(s.transform(X[te]))
            clf = RandomForestClassifier(**cfg.RF_PARAMS)
            clf.fit(Xtr, y_shuf[tr])
            accs.append(accuracy_score(y_shuf[te], clf.predict(Xte)))
        null_accs.append(np.mean(accs))
        if (i+1) % 100 == 0:
            print(f"  {i+1}/1000 permutations done...")

    null_accs = np.array(null_accs)
    p_value = np.mean(null_accs >= obs_mean)

    print(f"  Observed accuracy: {obs_mean:.3f}")
    print(f"  Null distribution: {np.mean(null_accs):.3f}±{np.std(null_accs):.3f}")
    print(f"  Null max: {np.max(null_accs):.3f}")
    print(f"  p-value: {p_value:.4f}")

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_accs, bins=30, alpha=0.7, color=PALETTE[0], edgecolor="black", label="Null distribution")
    ax.axvline(obs_mean, color="red", linewidth=2.5, linestyle="--", label=f"Observed: {obs_mean:.3f}")
    ax.set_xlabel("Mean 5-fold CV Accuracy")
    ax.set_ylabel("Count (of 1000 permutations)")
    ax.set_title(f"Permutation Test: p < {max(p_value, 0.001):.3f}")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig16_permutation_test")

    return {"observed": float(obs_mean), "null_mean": float(np.mean(null_accs)),
            "null_std": float(np.std(null_accs)), "null_max": float(np.max(null_accs)),
            "p_value": float(p_value), "n_permutations": n_perm}


# ═══════════════════════════════════════════════════════════════════
# C: Error analysis on pocket150025_001
# ═══════════════════════════════════════════════════════════════════
def experiment_c():
    print("\n"+"="*70)
    print("C: ERROR ANALYSIS — pocket150025_001")
    print("="*70)

    df = utils.load_parquet(cfg.RUN_VECTORS_DIR / "raw_temporal_summary.parquet")
    meta = ["toolpath","run_id","prefix"]
    fcols = [c for c in df.columns if c not in meta]

    # Find the borderline sample
    target_idx = df[(df["prefix"]=="pocket150025") & (df["run_id"]=="001")].index[0]
    target_row = df.iloc[target_idx]
    target_type = target_row["toolpath"]

    # Compute class centroids
    centroids = {}
    for tp in cfg.TOOLPATH_TYPES:
        mask = df["toolpath"] == tp
        centroids[tp] = df[mask][fcols].values.astype(np.float64).mean(axis=0)

    target_vec = df.iloc[target_idx][fcols].values.astype(np.float64)

    # Distance to each centroid
    distances = {}
    for tp, centroid in centroids.items():
        dist = np.sqrt(np.sum((target_vec - centroid)**2))
        distances[tp] = float(dist)
        print(f"  Distance to {tp} centroid: {dist:.2f}")

    # Find features where this sample deviates most from its class
    pocket_centroid = centroids["pocket"]
    pocket_std = df[df["toolpath"]=="pocket"][fcols].values.astype(np.float64).std(axis=0)
    pocket_std = np.where(pocket_std < 1e-10, 1.0, pocket_std)

    z_scores = np.abs((target_vec - pocket_centroid) / pocket_std)
    top_deviating = np.argsort(z_scores)[::-1][:15]

    print(f"\n  Top 15 features where pocket150025_001 deviates from pocket class:")
    deviation_records = []
    for rank, idx in enumerate(top_deviating):
        fname = fcols[idx]
        z = z_scores[idx]
        val = target_vec[idx]
        pocket_mean = pocket_centroid[idx]
        face_mean = centroids["face"][idx]
        print(f"    {rank+1:2d}. {fname:45s} z={z:.2f}  val={val:.4f}  pocket_mean={pocket_mean:.4f}  face_mean={face_mean:.4f}")
        deviation_records.append({
            "rank": rank+1, "feature": fname, "z_score": float(z),
            "sample_value": float(val), "pocket_mean": float(pocket_mean),
            "face_mean": float(face_mean),
        })

    # Is the sample closer to face on these deviating features?
    closer_to_face = sum(1 for rec in deviation_records
                         if abs(rec["sample_value"]-rec["face_mean"]) < abs(rec["sample_value"]-rec["pocket_mean"]))
    print(f"\n  Of top 15 deviating features: {closer_to_face}/15 are closer to face centroid")

    return {"target_index": int(target_idx), "distances": distances,
            "top_deviations": deviation_records, "closer_to_face": closer_to_face}


# ═══════════════════════════════════════════════════════════════════
# D: Temporal drift analysis
# ═══════════════════════════════════════════════════════════════════
def experiment_d():
    print("\n"+"="*70)
    print("D: RUN-ORDER TEMPORAL DRIFT ANALYSIS")
    print("="*70)

    df = utils.load_parquet(cfg.RUN_VECTORS_DIR / "raw_temporal_summary.parquet")
    meta = ["toolpath","run_id","prefix"]
    fcols = [c for c in df.columns if c not in meta]

    results = {}
    for tp in cfg.TOOLPATH_TYPES:
        sub = df[df["toolpath"]==tp].copy()
        sub["run_num"] = sub["run_id"].astype(int)
        sub = sub.sort_values("run_num")

        # Check if key features correlate with run order
        key_features = ["binmean_frame_r2.Mx_mean", "binmean_y_bed__3.Az_mean",
                        "binmean_spindle_mean", "binmean_frame_l2.Temperature_mean"]

        tp_result = {}
        for feat in key_features:
            if feat not in sub.columns:
                continue
            vals = sub[feat].values.astype(float)
            run_nums = sub["run_num"].values.astype(float)
            if len(vals) < 3:
                continue
            from scipy.stats import pearsonr, spearmanr
            r_pearson, p_pearson = pearsonr(run_nums, vals)
            r_spearman, p_spearman = spearmanr(run_nums, vals)
            tp_result[feat] = {
                "pearson_r": float(r_pearson), "pearson_p": float(p_pearson),
                "spearman_r": float(r_spearman), "spearman_p": float(p_spearman),
            }
            sig = "*" if p_spearman < 0.05 else ""
            print(f"  {tp}/{feat}: Spearman r={r_spearman:.3f} p={p_spearman:.3f} {sig}")

        results[tp] = tp_result

    # Plot key feature vs run order
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    key_features = ["binmean_frame_r2.Mx_mean", "binmean_y_bed__3.Az_mean",
                    "binmean_spindle_mean", "binmean_frame_l2.Temperature_mean"]

    for idx, feat in enumerate(key_features):
        ax = axes[idx//2, idx%2]
        for tp in cfg.TOOLPATH_TYPES:
            sub = df[df["toolpath"]==tp].copy()
            sub["run_num"] = sub["run_id"].astype(int)
            sub = sub.sort_values("run_num")
            if feat in sub.columns:
                ax.scatter(sub["run_num"], sub[feat], label=tp.capitalize(),
                          color={"adaptive":PALETTE[0],"face":PALETTE[1],"pocket":PALETTE[2]}[tp],
                          alpha=0.7, s=40)
        ax.set_xlabel("Run Number (chronological)")
        ax.set_ylabel(feat.replace("binmean_",""), fontsize=9)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Feature Values vs Run Order: Checking for Temporal Drift", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig17_temporal_drift")

    return results


# ═══════════════════════════════════════════════════════════════════
# E: Effective dimensionality
# ═══════════════════════════════════════════════════════════════════
def experiment_e():
    print("\n"+"="*70)
    print("E: EFFECTIVE DIMENSIONALITY & CORRELATION STRUCTURE")
    print("="*70)

    df = utils.load_parquet(cfg.RUN_VECTORS_DIR / "raw_temporal_summary.parquet")
    meta = ["toolpath","run_id","prefix"]
    fcols = [c for c in df.columns if c not in meta]
    X = np.nan_to_num(df[fcols].values.astype(np.float64))

    scaler = StandardScaler()
    X_s = np.nan_to_num(scaler.fit_transform(X))

    pca = PCA(random_state=42)
    pca.fit(X_s)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    # Effective dimensionality at various thresholds
    results = {}
    for thresh in [0.50, 0.80, 0.90, 0.95, 0.99]:
        n = np.searchsorted(cumvar, thresh) + 1
        results[f"n_components_{int(thresh*100)}pct"] = int(n)
        print(f"  {thresh*100:.0f}% variance: {n} components (of {len(cumvar)} total)")

    # Effective rank (number of non-negligible eigenvalues)
    eigenvalues = pca.explained_variance_
    significant = np.sum(eigenvalues > eigenvalues[0] * 1e-3)
    results["effective_rank"] = int(significant)
    results["total_features"] = len(fcols)
    results["n_samples"] = X.shape[0]
    results["feature_to_sample_ratio"] = len(fcols) / X.shape[0]
    results["effective_ratio"] = int(significant) / X.shape[0]
    print(f"  Effective rank (eigenvalue > 0.1% of max): {significant}")
    print(f"  Feature-to-sample ratio: {len(fcols)}/{X.shape[0]} = {len(fcols)/X.shape[0]:.1f}")
    print(f"  Effective ratio: {significant}/{X.shape[0]} = {significant/X.shape[0]:.1f}")

    # Correlation analysis
    corr_matrix = np.corrcoef(X_s.T)
    upper = np.triu(corr_matrix, k=1)
    n_high_corr = np.sum(np.abs(upper) > 0.95)
    n_total_pairs = upper.size - np.sum(upper == 0)
    pct_high_corr = n_high_corr / max(n_total_pairs, 1) * 100
    results["pct_pairs_above_095"] = float(pct_high_corr)
    results["n_pairs_above_095"] = int(n_high_corr)
    print(f"  Feature pairs with |r| > 0.95: {n_high_corr} ({pct_high_corr:.1f}%)")

    # Scree plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(range(1, min(51, len(cumvar)+1)), cumvar[:50], "o-", markersize=4, color=PALETTE[0])
    ax1.axhline(0.95, color="gray", linestyle="--", label="95% threshold")
    ax1.axhline(0.90, color="gray", linestyle=":", label="90% threshold")
    n95 = results["n_components_95pct"]
    ax1.axvline(n95, color="red", linestyle="--", alpha=0.5)
    ax1.annotate(f"{n95} components\nfor 95%", xy=(n95, 0.95), fontsize=9,
                ha="left", va="bottom", color="red")
    ax1.set_xlabel("Number of Principal Components")
    ax1.set_ylabel("Cumulative Variance Explained")
    ax1.set_title("Scree Plot: Effective Dimensionality")
    ax1.legend(fontsize=9)

    # Eigenvalue spectrum
    ax2.semilogy(range(1, min(51, len(eigenvalues)+1)), eigenvalues[:50], "o-",
                markersize=4, color=PALETTE[1])
    ax2.axhline(eigenvalues[0] * 1e-3, color="gray", linestyle="--",
               label=f"0.1% of max ({eigenvalues[0]*1e-3:.2f})")
    ax2.set_xlabel("Component Index")
    ax2.set_ylabel("Eigenvalue (log scale)")
    ax2.set_title(f"Eigenvalue Spectrum (effective rank = {significant})")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    save_fig(fig, "fig18_dimensionality")

    return results


# ═══════════════════════════════════════════════════════════════════
def run_all():
    results = {}
    results["permutation_test"] = experiment_b()
    results["error_analysis"] = experiment_c()
    results["temporal_drift"] = experiment_d()
    results["dimensionality"] = experiment_e()

    out = cfg.RESULTS_DIR / "experiments_bcde.json"
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nAll B-E results saved to {out}")

if __name__ == "__main__":
    run_all()
