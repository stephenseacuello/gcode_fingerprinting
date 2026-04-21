"""Path 3 experiments: theoretical/analytical extensions.

9.  Mutual information analysis (feature vs class, run order, position proxy)
10. Bayesian Beta-Binomial posterior on accuracy differences
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

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
    for ext in ["png", "pdf"]:
        fig.savefig(cfg.FIGURES_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def load_data(iso="raw"):
    df = utils.load_parquet(cfg.RUN_VECTORS_DIR / f"{iso}_temporal_summary.parquet")
    meta = ["toolpath", "run_id", "prefix"]
    fcols = [c for c in df.columns if c not in meta]
    X = np.nan_to_num(df[fcols].values.astype(np.float64))
    le = LabelEncoder()
    y = le.fit_transform(df["toolpath"].values)
    return X, y, fcols, df, le


def get_channel_type(fname):
    if any(m in fname for m in [".Mx", ".My", ".Mz"]): return "Magnetometer"
    if any(m in fname for m in [".Ax", ".Ay", ".Az"]): return "Accelerometer"
    if any(m in fname for m in [".Gx", ".Gy", ".Gz"]): return "Gyroscope"
    if ".RMS" in fname: return "Vibration RMS"
    if ".Pressure" in fname: return "Pressure"
    if ".Temperature" in fname: return "Temperature"
    if ".Proximity" in fname: return "Proximity"
    if ".Color" in fname: return "Color"
    if any(e in fname for e in cfg.ELECTRICAL_COLS): return "Electrical"
    return "Other"


# ═══════════════════════════════════════════════════════════════════
# Item 9: Mutual Information analysis
# ═══════════════════════════════════════════════════════════════════
def experiment_mi():
    print("=" * 70)
    print("PATH 3 / ITEM 9: MUTUAL INFORMATION ANALYSIS")
    print("=" * 70)

    X, y, fcols, df, le = load_data("raw")

    # Run order as a continuous variable
    run_order = df["run_id"].astype(int).values.astype(float)

    # Position proxy: use first principal component of magnetometer features
    mag_cols = [i for i, f in enumerate(fcols) if any(m in f for m in [".Mx", ".My", ".Mz"])]
    X_mag = X[:, mag_cols]
    X_mag_s = StandardScaler().fit_transform(X_mag)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1, random_state=42)
    position_proxy = pca.fit_transform(X_mag_s).ravel()

    print(f"  Position proxy: PC1 of {len(mag_cols)} magnetometer features (var explained: {pca.explained_variance_ratio_[0]*100:.1f}%)")

    # Compute MI between each feature and (class, run_order, position_proxy)
    print("  Computing MI(feature; class)...")
    mi_class = mutual_info_classif(X, y, random_state=42)
    print("  Computing MI(feature; run_order)...")
    mi_order = mutual_info_regression(X, run_order, random_state=42)
    print("  Computing MI(feature; position_proxy)...")
    mi_pos = mutual_info_regression(X, position_proxy, random_state=42)

    # Convert nats to bits (sklearn returns nats)
    mi_class_bits = mi_class / np.log(2)
    mi_order_bits = mi_order / np.log(2)
    mi_pos_bits = mi_pos / np.log(2)

    # Aggregate by channel type
    records = []
    for i, fname in enumerate(fcols):
        records.append({
            "feature": fname,
            "channel_type": get_channel_type(fname),
            "mi_class_bits": float(mi_class_bits[i]),
            "mi_order_bits": float(mi_order_bits[i]),
            "mi_position_bits": float(mi_pos_bits[i]),
        })
    mi_df = pd.DataFrame(records)

    # Per-channel-type aggregation
    agg = mi_df.groupby("channel_type").agg({
        "mi_class_bits": ["mean", "median", "max"],
        "mi_order_bits": ["mean", "max"],
        "mi_position_bits": ["mean", "max"],
        "feature": "count",
    }).round(3)
    print("\n  Mean MI by channel type (bits):")
    print(f"  {'Type':<18} {'MI(class)':>11} {'MI(order)':>11} {'MI(pos)':>11} {'%pos/cls':>11} {'N':>5}")

    type_results = {}
    for ctype in ["Magnetometer", "Accelerometer", "Gyroscope", "Vibration RMS",
                  "Temperature", "Pressure", "Proximity", "Color", "Electrical"]:
        sub = mi_df[mi_df["channel_type"] == ctype]
        if len(sub) == 0: continue
        mc = sub["mi_class_bits"].mean()
        mo = sub["mi_order_bits"].mean()
        mp = sub["mi_position_bits"].mean()
        ratio = mp / max(mc, 1e-6) * 100
        n = len(sub)
        print(f"  {ctype:<18} {mc:>11.3f} {mo:>11.3f} {mp:>11.3f} {ratio:>10.1f}% {n:>5}")
        type_results[ctype] = {
            "mean_mi_class_bits": float(mc),
            "mean_mi_order_bits": float(mo),
            "mean_mi_position_bits": float(mp),
            "position_to_class_ratio": float(ratio),
            "n_features": int(n),
        }

    # Total information about the class label
    # Maximum possible MI between feature and 3-class label = log2(3) = 1.585 bits
    print(f"\n  Maximum possible MI for 3-class label: {np.log2(3):.3f} bits")
    top_class = mi_df.nlargest(10, "mi_class_bits")
    print(f"\n  Top 10 features by MI(feature; class):")
    for _, row in top_class.iterrows():
        print(f"    {row['feature']:50s} MI(class)={row['mi_class_bits']:.3f}  MI(pos)={row['mi_position_bits']:.3f}")

    # Compute decomposition: how much of MI(feature; class) is "explained" by position?
    # Approximation: features where MI(pos) is large likely encode position
    high_pos_features = (mi_df["mi_position_bits"] > 0.5).sum()
    high_class_features = (mi_df["mi_class_bits"] > 0.5).sum()
    both = ((mi_df["mi_position_bits"] > 0.5) & (mi_df["mi_class_bits"] > 0.5)).sum()
    print(f"\n  Features with MI(class) > 0.5 bits: {high_class_features}")
    print(f"  Features with MI(position) > 0.5 bits: {high_pos_features}")
    print(f"  Features with both: {both}")

    return {
        "by_type": type_results,
        "total_features": len(fcols),
        "max_mi_class_bits": float(np.log2(3)),
        "n_high_class": int(high_class_features),
        "n_high_position": int(high_pos_features),
        "n_both": int(both),
        "top10_by_class": top_class[["feature", "mi_class_bits", "mi_position_bits"]].to_dict(orient="records"),
    }, mi_df


# ═══════════════════════════════════════════════════════════════════
# Item 10: Bayesian Beta-Binomial posterior on accuracy
# ═══════════════════════════════════════════════════════════════════
def experiment_bayesian():
    print("\n" + "=" * 70)
    print("PATH 3 / ITEM 10: BAYESIAN POSTERIOR ON ACCURACY")
    print("=" * 70)

    # For each isolation method, get the per-fold (n_correct, n_total) from
    # repeated CV. Treat as Binomial. Use Beta(1,1) prior → Beta(1+sum(correct), 1+sum(incorrect)) posterior.
    rf_params = cfg.RF_PARAMS

    method_data = {}
    for iso in cfg.ISOLATION_METHODS:
        X, y, fcols, df, le = load_data(iso)

        total_correct = 0
        total_n = 0
        fold_accs = []
        for rep in range(20):
            seed = cfg.RANDOM_STATE + rep
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for tr, te in skf.split(X, y):
                s = StandardScaler()
                Xtr = np.nan_to_num(s.fit_transform(X[tr]))
                Xte = np.nan_to_num(s.transform(X[te]))
                clf = RandomForestClassifier(**rf_params)
                clf.fit(Xtr, y[tr])
                preds = clf.predict(Xte)
                n_correct = int((preds == y[te]).sum())
                n_total = len(te)
                total_correct += n_correct
                total_n += n_total
                fold_accs.append(n_correct / n_total)

        # Posterior: Beta(1 + total_correct, 1 + total_incorrect)
        alpha_post = 1 + total_correct
        beta_post = 1 + (total_n - total_correct)
        method_data[iso] = {
            "total_correct": total_correct,
            "total_n": total_n,
            "alpha": alpha_post,
            "beta": beta_post,
            "fold_accs": fold_accs,
        }
        print(f"  {iso}: {total_correct}/{total_n} = {total_correct/total_n*100:.2f}% "
              f"(Beta posterior: α={alpha_post}, β={beta_post})")

    # Sample from posteriors
    from scipy.stats import beta as beta_dist
    n_samples = 100000
    rng = np.random.RandomState(42)
    samples = {}
    for iso, data in method_data.items():
        s = beta_dist.rvs(data["alpha"], data["beta"], size=n_samples, random_state=rng)
        samples[iso] = s

    # Posterior probability that referenced > raw, for each referenced method
    print("\n  Posterior P(referenced > raw):")
    p_better = {}
    for iso in ["subtracted", "ratio", "zscore"]:
        p = float((samples[iso] > samples["raw"]).mean())
        p_better[iso] = p
        diff = samples[iso] - samples["raw"]
        print(f"    P({iso} > raw) = {p:.4f}")
        print(f"      Posterior accuracy difference: median={np.median(diff)*100:.3f}%, "
              f"95% CI=[{np.percentile(diff, 2.5)*100:.3f}%, {np.percentile(diff, 97.5)*100:.3f}%]")

    # Posterior accuracy 95% CI per method
    print("\n  Posterior 95% credible intervals on accuracy:")
    cis = {}
    for iso, s in samples.items():
        lo, med, hi = np.percentile(s, [2.5, 50, 97.5])
        cis[iso] = {"median": float(med), "ci_low": float(lo), "ci_high": float(hi)}
        print(f"    {iso}: median={med*100:.3f}%, 95% CI=[{lo*100:.3f}%, {hi*100:.3f}%]")

    # Plot posterior densities
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, iso in enumerate(cfg.ISOLATION_METHODS):
        data = method_data[iso]
        x_grid = np.linspace(0.95, 1.0, 1000)
        pdf = beta_dist.pdf(x_grid, data["alpha"], data["beta"])
        ax.plot(x_grid * 100, pdf, linewidth=2, color=PALETTE[i],
                label=f"{iso.capitalize()} ({data['total_correct']}/{data['total_n']})")
        ax.fill_between(x_grid * 100, 0, pdf, alpha=0.15, color=PALETTE[i])

    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Posterior density")
    ax.set_title("Bayesian Beta-Binomial Posterior on RF Classification Accuracy\n(20×5-fold = 100 evaluations, ~10 test samples per fold = 1000 total predictions)")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "fig21_bayesian_posterior")

    return {
        "method_posteriors": {
            iso: {
                "alpha": data["alpha"],
                "beta": data["beta"],
                "n_correct": data["total_correct"],
                "n_total": data["total_n"],
            }
            for iso, data in method_data.items()
        },
        "credible_intervals": cis,
        "p_better_than_raw": p_better,
    }


# ═══════════════════════════════════════════════════════════════════
# MI figure
# ═══════════════════════════════════════════════════════════════════
def fig_mi_decomposition(mi_df):
    print("\n  Generating MI decomposition figure...")

    # Aggregate per channel type
    types = ["Magnetometer", "Accelerometer", "Gyroscope", "Vibration RMS",
             "Temperature", "Pressure", "Proximity", "Color", "Electrical"]
    means_class = []
    means_pos = []
    means_order = []
    labels = []

    for ctype in types:
        sub = mi_df[mi_df["channel_type"] == ctype]
        if len(sub) == 0:
            continue
        means_class.append(sub["mi_class_bits"].mean())
        means_pos.append(sub["mi_position_bits"].mean())
        means_order.append(sub["mi_order_bits"].mean())
        labels.append(ctype)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    width = 0.27

    ax.bar(x - width, means_class, width, label="MI(feature; class)", color=PALETTE[0], edgecolor="black", linewidth=0.5)
    ax.bar(x, means_pos, width, label="MI(feature; position proxy)", color=PALETTE[1], edgecolor="black", linewidth=0.5)
    ax.bar(x + width, means_order, width, label="MI(feature; run order)", color=PALETTE[2], edgecolor="black", linewidth=0.5)

    ax.axhline(np.log2(3), color="gray", linestyle="--", linewidth=1, label=f"Max possible MI(class)={np.log2(3):.2f} bits")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Mutual Information (bits)")
    ax.set_title("Mutual Information Decomposition by Sensor Channel Type")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "fig22_mi_decomposition")


# ═══════════════════════════════════════════════════════════════════
def run_all():
    results = {}
    mi_results, mi_df = experiment_mi()
    results["mutual_information"] = mi_results

    fig_mi_decomposition(mi_df)

    bayes_results = experiment_bayesian()
    results["bayesian"] = bayes_results

    out = cfg.RESULTS_DIR / "experiments_path3.json"
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=conv)
    print(f"\nAll Path 3 results saved to {out}")


if __name__ == "__main__":
    run_all()
