"""Final round experiments (items 9-15):
9. True held-out test set
10. Per-channel SNR budget
11. Causal/confound analysis (partial correlations)
12. Finer learning curves
13. Calibration metrics (ECE, reliability)
14. MiniRocket-style comparison (variant of existing ROCKET)
15. Confound-free pipeline visualization
"""
import json, sys, warnings
import builtins
_orig_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return _orig_print(*args, **kwargs)
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import pearsonr, spearmanr

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

CONFOUND_FREE = [".Ax", ".Ay", ".Az", ".Gx", ".Gy", ".Gz", ".RMS", ".Temperature"]


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(cfg.FIGURES_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def select_clean_features(feature_cols):
    clean = []
    for f in feature_cols:
        # electrical: exact electrical-channel match, excluding Arduino IMU modules
        # ('spindle' must not substring-match the IMU module 'spindle2').
        if any(e in f for e in cfg.ELECTRICAL_COLS) and not any(m in f for m in cfg.ARDUINO_MODULES):
            clean.append(f); continue
        if any(ch in f for ch in CONFOUND_FREE):
            clean.append(f)
    return clean


def load_data(iso="raw"):
    df = utils.load_parquet(cfg.RUN_VECTORS_DIR / f"{iso}_temporal_summary.parquet")
    meta = ["toolpath", "run_id", "prefix"]
    fcols = [c for c in df.columns if c not in meta]
    X = np.nan_to_num(df[fcols].values.astype(np.float64))
    le = LabelEncoder()
    y = le.fit_transform(df["toolpath"].values)
    return X, y, fcols, df, le


# ═══════════════════════════════════════════════════════════════════
# 9. True held-out test set
# ═══════════════════════════════════════════════════════════════════
def experiment_9():
    print("=" * 70)
    print("9. TRUE HELD-OUT TEST SET")
    print("=" * 70)

    results = {}
    for iso in cfg.ISOLATION_METHODS:
        X, y, fcols, df, le = load_data(iso)

        # Stratified split: hold out 12 runs (~24%) — at least 3 per class
        X_dev, X_test, y_dev, y_test = train_test_split(
            X, y, test_size=12, stratify=y, random_state=42
        )

        # Train on dev set with full pipeline (just RF, no further splits)
        scaler = StandardScaler()
        X_dev_s = np.nan_to_num(scaler.fit_transform(X_dev))
        X_test_s = np.nan_to_num(scaler.transform(X_test))

        clf = RandomForestClassifier(**cfg.RF_PARAMS)
        clf.fit(X_dev_s, y_dev)
        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        results[iso] = {
            "test_accuracy": float(acc),
            "n_train": int(len(y_dev)),
            "n_test": int(len(y_test)),
            "test_class_counts": np.bincount(y_test).tolist(),
        }
        print(f"  {iso} RF: test_acc={acc:.3f} (n_train={len(y_dev)}, n_test={len(y_test)})")

    return results


# ═══════════════════════════════════════════════════════════════════
# 10. Per-channel SNR budget
# ═══════════════════════════════════════════════════════════════════
def experiment_10():
    print("\n" + "=" * 70)
    print("10. PER-CHANNEL SNR BUDGET")
    print("=" * 70)

    X, y, fcols, df, le = load_data("raw")

    # SNR per feature: signal = between-class variance, noise = within-class variance
    # SNR_dB = 10*log10(between/within)
    snr_records = []
    for i, fname in enumerate(fcols):
        feat_vals = X[:, i]
        # Between-class variance (signal)
        class_means = []
        for cls in range(3):
            class_means.append(np.mean(feat_vals[y == cls]))
        between_var = np.var(class_means, ddof=1)

        # Within-class variance (noise)
        within_vars = []
        for cls in range(3):
            cls_vals = feat_vals[y == cls]
            if len(cls_vals) > 1:
                within_vars.append(np.var(cls_vals, ddof=1))
        within_var = np.mean(within_vars) if within_vars else 1e-12

        if within_var < 1e-12:
            snr = 0.0
            snr_db = -np.inf
        else:
            snr = between_var / within_var
            snr_db = 10 * np.log10(snr + 1e-12)

        snr_records.append({"feature": fname, "snr": float(snr),
                            "snr_db": float(snr_db),
                            "between_var": float(between_var),
                            "within_var": float(within_var)})

    snr_df = pd.DataFrame(snr_records)
    snr_df = snr_df.sort_values("snr", ascending=False)

    # Aggregate by sensor channel type
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

    snr_df["type"] = snr_df["feature"].apply(get_channel_type)
    agg = snr_df.groupby("type").agg({
        "snr_db": ["mean", "median", "max"],
        "feature": "count"
    }).round(2)
    print("\n  SNR by channel type (median across features):")
    print(f"  {'Type':<18} {'Median SNR (dB)':>16} {'Max SNR (dB)':>14} {'N features':>11}")
    type_results = {}
    for ctype in ["Magnetometer", "Accelerometer", "Gyroscope", "Vibration RMS",
                  "Temperature", "Pressure", "Proximity", "Color", "Electrical"]:
        sub = snr_df[snr_df["type"] == ctype]
        if len(sub) == 0: continue
        med = sub["snr_db"].median()
        mx = sub["snr_db"].max()
        n = len(sub)
        print(f"  {ctype:<18} {med:>16.2f} {mx:>14.2f} {n:>11}")
        type_results[ctype] = {"median_snr_db": float(med),
                                "max_snr_db": float(mx),
                                "n_features": int(n)}

    # Top 10 features by SNR
    print("\n  Top 10 features by SNR:")
    top10 = []
    for _, row in snr_df.head(10).iterrows():
        print(f"    {row['feature']:50s} SNR={row['snr_db']:.2f} dB")
        top10.append({"feature": row["feature"], "snr_db": float(row["snr_db"])})

    return {"by_type": type_results, "top10_features": top10}


# ═══════════════════════════════════════════════════════════════════
# 11. Partial correlation: feature vs class controlling for run order
# ═══════════════════════════════════════════════════════════════════
def experiment_11():
    print("\n" + "=" * 70)
    print("11. PARTIAL CORRELATION ANALYSIS (controlling for run order)")
    print("=" * 70)

    X, y, fcols, df, le = load_data("raw")

    # Add run order
    df_meta = df[["toolpath", "run_id", "prefix"]].copy()
    df_meta["run_num"] = df_meta["run_id"].astype(int)

    # For each feature, compute:
    # (a) raw correlation with class
    # (b) partial correlation with class controlling for run order
    from scipy.stats import pearsonr

    def partial_corr(x, y, z):
        """Partial correlation between x and y controlling for z."""
        # Residualize x against z
        coef_xz = np.polyfit(z, x, 1)
        x_res = x - np.polyval(coef_xz, z)
        # Residualize y against z
        coef_yz = np.polyfit(z, y, 1)
        y_res = y - np.polyval(coef_yz, z)
        # Correlate residuals
        if np.std(x_res) < 1e-10 or np.std(y_res) < 1e-10:
            return 0.0, 1.0
        return pearsonr(x_res, y_res)

    # Compute for top-importance features
    imp_df = pd.read_csv(cfg.IMPORTANCE_DIR / "raw_rf_importance.csv")
    top_features = imp_df.head(20)["feature"].tolist()

    run_order = df_meta["run_num"].values.astype(float)

    results = []
    for fname in top_features:
        if fname not in fcols: continue
        idx = fcols.index(fname)
        feat = X[:, idx]

        # Raw correlation with class
        r_class, p_class = pearsonr(feat, y.astype(float))
        # Partial correlation controlling for run order
        r_partial, p_partial = partial_corr(feat, y.astype(float), run_order)
        # Correlation with run order alone
        r_order, p_order = pearsonr(feat, run_order)

        results.append({
            "feature": fname,
            "r_class": float(r_class),
            "r_class_partial": float(r_partial),
            "r_run_order": float(r_order),
            "reduction_pct": float((abs(r_class) - abs(r_partial)) / max(abs(r_class), 1e-10) * 100),
        })

    results_df = pd.DataFrame(results)
    print(f"\n  {'Feature':<48} {'r_class':>9} {'r_partial':>11} {'r_order':>9} {'%reduce':>9}")
    for _, row in results_df.iterrows():
        print(f"  {row['feature']:<48} {row['r_class']:>9.3f} "
              f"{row['r_class_partial']:>11.3f} {row['r_run_order']:>9.3f} "
              f"{row['reduction_pct']:>9.1f}")

    avg_reduction = results_df["reduction_pct"].mean()
    print(f"\n  Average importance reduction after controlling for run order: {avg_reduction:.1f}%")

    return {"per_feature": results, "avg_reduction_pct": float(avg_reduction)}


# ═══════════════════════════════════════════════════════════════════
# 12. Finer learning curve
# ═══════════════════════════════════════════════════════════════════
def experiment_12():
    print("\n" + "=" * 70)
    print("12. FINER LEARNING CURVE")
    print("=" * 70)

    results = {}
    for iso in cfg.ISOLATION_METHODS:
        X, y, fcols, df, le = load_data(iso)
        max_per_class = min(np.bincount(y))

        train_sizes = [s for s in [2, 3, 5, 7, 9, 11, 13] if s < max_per_class]
        iso_results = {}

        for n in train_sizes:
            accs = []
            for rep in range(10):
                rng = np.random.RandomState(cfg.RANDOM_STATE + rep)
                train_mask = np.zeros(len(y), dtype=bool)
                test_mask = np.zeros(len(y), dtype=bool)
                for cls in range(3):
                    cls_idx = np.where(y == cls)[0]
                    rng.shuffle(cls_idx)
                    train_mask[cls_idx[:n]] = True
                    test_mask[cls_idx[n:]] = True

                if test_mask.sum() == 0: continue
                scaler = StandardScaler()
                Xtr = np.nan_to_num(scaler.fit_transform(X[train_mask]))
                Xte = np.nan_to_num(scaler.transform(X[test_mask]))
                clf = RandomForestClassifier(**cfg.RF_PARAMS)
                clf.fit(Xtr, y[train_mask])
                accs.append(accuracy_score(y[test_mask], clf.predict(Xte)))

            iso_results[n] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
        results[iso] = iso_results
        print(f"  {iso}: {len(iso_results)} sizes, accuracy at smallest "
              f"({train_sizes[0]} runs): {iso_results[train_sizes[0]]['mean']:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, iso in enumerate(cfg.ISOLATION_METHODS):
        sizes = sorted(results[iso].keys())
        means = [results[iso][s]["mean"] for s in sizes]
        stds = [results[iso][s]["std"] for s in sizes]
        ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=4,
                   linewidth=2, markersize=6, label=iso.capitalize(), color=PALETTE[i])

    ax.axhline(1/3, color="gray", linestyle="--", linewidth=1, label="Random baseline")
    ax.set_xlabel("Training Runs per Class")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Fine-Grained Learning Curve (RF, 20 random subsamples per point)")
    ax.set_ylim(0.30, 1.05)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig19_learning_curve_fine")

    return results


# ═══════════════════════════════════════════════════════════════════
# 13. Calibration metrics (ECE, reliability)
# ═══════════════════════════════════════════════════════════════════
def experiment_13():
    print("\n" + "=" * 70)
    print("13. CALIBRATION METRICS (ECE, reliability)")
    print("=" * 70)

    results = {}
    for iso in cfg.ISOLATION_METHODS:
        X, y, fcols, df, le = load_data(iso)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        all_probs = []
        all_true = []
        all_pred = []

        for tr, te in skf.split(X, y):
            scaler = StandardScaler()
            Xtr = np.nan_to_num(scaler.fit_transform(X[tr]))
            Xte = np.nan_to_num(scaler.transform(X[te]))
            clf = RandomForestClassifier(**cfg.RF_PARAMS)
            clf.fit(Xtr, y[tr])
            probs = clf.predict_proba(Xte)
            preds = clf.predict(Xte)
            all_probs.append(probs)
            all_true.append(y[te])
            all_pred.append(preds)

        all_probs = np.vstack(all_probs)
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)

        # Compute ECE (Expected Calibration Error)
        max_probs = np.max(all_probs, axis=1)
        correct = (all_pred == all_true).astype(float)

        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i+1])
            if i == n_bins - 1:
                in_bin = (max_probs >= bin_edges[i]) & (max_probs <= bin_edges[i+1])
            if in_bin.sum() > 0:
                bin_acc = correct[in_bin].mean()
                bin_conf = max_probs[in_bin].mean()
                ece += (in_bin.sum() / len(max_probs)) * abs(bin_acc - bin_conf)

        # Log loss
        probs_clipped = np.clip(all_probs, 1e-15, 1 - 1e-15)
        logloss = log_loss(all_true, probs_clipped, labels=[0, 1, 2])

        results[iso] = {
            "ece": float(ece),
            "log_loss": float(logloss),
            "mean_max_prob": float(np.mean(max_probs)),
            "accuracy": float(np.mean(correct)),
        }
        print(f"  {iso}: ECE={ece:.4f}, LogLoss={logloss:.4f}, "
              f"mean_conf={np.mean(max_probs):.3f}, acc={np.mean(correct):.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# 14. MiniRocket variant comparison
# ═══════════════════════════════════════════════════════════════════
def experiment_14():
    """MiniRocket via fully-vectorized sliding-window convolution."""
    print("\n" + "=" * 70)
    print("14. MINIROCKET-STYLE COMPARISON (vectorized)")
    print("=" * 70)

    le = LabelEncoder()

    # Load and downsample raw time series
    all_series = []
    all_labels = []
    for toolpath in cfg.TOOLPATH_TYPES:
        prefix = cfg.ACTIVE_PREFIX[toolpath]
        files = utils.discover_runs(prefix)
        for fpath in files:
            df_run = utils.load_run(fpath)
            sensor_data = df_run[[c for c in cfg.SENSOR_COLS if c in df_run.columns]].values
            all_series.append(sensor_data)
            all_labels.append(toolpath)

    y = le.fit_transform(all_labels)
    n_channels = all_series[0].shape[1]
    n_runs = len(all_series)

    # Downsample to 80 timesteps (smaller for speed)
    target_len = 80
    X_ds = np.zeros((n_runs, target_len, n_channels), dtype=np.float32)
    for i, s in enumerate(all_series):
        old_idx = np.linspace(0, len(s) - 1, len(s))
        new_idx = np.linspace(0, len(s) - 1, target_len)
        for ch in range(n_channels):
            X_ds[i, :, ch] = np.interp(new_idx, old_idx, s[:, ch])

    # Use a small set of kernels for tractability: 12 kernels (subset of MiniRocket's 84)
    from itertools import combinations
    kernel_weights = []
    combos = list(combinations(range(9), 3))
    # Take every 7th to get diverse subset of 12
    for combo in combos[::7][:12]:
        w = np.full(9, -1.0, dtype=np.float32)
        for c in combo:
            w[c] = 2.0
        kernel_weights.append(w)
    kernel_weights = np.stack(kernel_weights)  # (12, 9)
    n_kernels = len(kernel_weights)

    print(f"  Computing {n_kernels} kernels × {n_channels} channels on length-{target_len}...")

    # Fully vectorized: build sliding windows once
    # X_ds: (n_runs, target_len, n_channels) -> (n_runs, n_windows, 9, n_channels)
    n_windows = target_len - 9 + 1
    # sliding_window_view returns (n_runs, n_windows, n_channels, 9)
    windows = np.lib.stride_tricks.sliding_window_view(X_ds, 9, axis=1)
    print(f"  Windows shape: {windows.shape}")

    # Convolve: (n_kernels, 9) @ (..., 9) -> (..., n_kernels)
    # windows: (n_runs, n_windows, n_channels, 9)
    # einsum is fast: 'rwcl,kl->rwck'
    conv = np.einsum('rwcl,kl->rwck', windows, kernel_weights)  # (n_runs, n_windows, n_channels, n_kernels)

    # PPV across windows
    ppv = (conv > 0).mean(axis=1)  # (n_runs, n_channels, n_kernels)

    # Flatten to (n_runs, n_channels * n_kernels)
    X_mr = ppv.reshape(n_runs, -1).astype(np.float64)
    X_mr = np.nan_to_num(X_mr, nan=0., posinf=0., neginf=0.)
    print(f"  Total MiniRocket features: {X_mr.shape[1]}")

    # Classify
    from sklearn.linear_model import RidgeClassifierCV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs_rf, accs_ridge = [], []
    for tr, te in skf.split(X_mr, y):
        s = StandardScaler()
        Xtr = np.nan_to_num(s.fit_transform(X_mr[tr]))
        Xte = np.nan_to_num(s.transform(X_mr[te]))

        rf = RandomForestClassifier(**cfg.RF_PARAMS)
        rf.fit(Xtr, y[tr])
        accs_rf.append(accuracy_score(y[te], rf.predict(Xte)))

        ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        ridge.fit(Xtr, y[tr])
        accs_ridge.append(accuracy_score(y[te], ridge.predict(Xte)))

    print(f"  MiniRocket + RF: {np.mean(accs_rf):.3f}±{np.std(accs_rf):.3f}")
    print(f"  MiniRocket + Ridge: {np.mean(accs_ridge):.3f}±{np.std(accs_ridge):.3f}")

    return {
        "minirocket_rf": {"mean": float(np.mean(accs_rf)), "std": float(np.std(accs_rf)),
                          "n_features": X_mr.shape[1]},
        "minirocket_ridge": {"mean": float(np.mean(accs_ridge)), "std": float(np.std(accs_ridge)),
                             "n_features": X_mr.shape[1]},
    }


# ═══════════════════════════════════════════════════════════════════
# 15. Confound-free pipeline visualization
# ═══════════════════════════════════════════════════════════════════
def experiment_15():
    print("\n" + "=" * 70)
    print("15. CONFOUND-FREE PIPELINE VISUALIZATION")
    print("=" * 70)

    # Load Exp A results
    with open(cfg.RESULTS_DIR / "experiment_a_nomag.json") as f:
        nomag = json.load(f)

    # Bar chart: raw vs subtracted/ratio/zscore for full vs confound-free features
    classifiers = ["RF", "SVM", "MLP"]

    # Get full-feature results from earlier round
    with open(cfg.RESULTS_DIR / "additional_experiments.json") as f:
        full = json.load(f)
    full_rcv = full["repeated_cv"]

    methods = ["raw", "subtracted", "ratio", "zscore"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: full features
    ax = axes[0]
    x = np.arange(len(methods))
    width = 0.25
    for i, clf in enumerate(classifiers):
        means = []
        stds = []
        for m in methods:
            key = f"{m}_{clf}"
            if key in full_rcv:
                means.append(full_rcv[key]["accuracy_mean"])
                stds.append(full_rcv[key]["accuracy_std"])
            else:
                means.append(0); stds.append(0)
        ax.bar(x + i * width, means, width, yerr=stds, label=clf,
              color=PALETTE[i], capsize=3, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in methods])
    ax.set_ylabel("Accuracy (20×5-fold CV)")
    ax.set_title("(a) Full feature set (1,540 features)")
    ax.set_ylim(0.7, 1.05)
    ax.legend(fontsize=9, loc="lower right")

    # Right: confound-free features
    ax = axes[1]
    for i, clf in enumerate(classifiers):
        means = []
        stds = []
        for m in methods:
            key = f"{m}_{clf}"
            if key in nomag:
                means.append(nomag[key]["mean"])
                stds.append(nomag[key]["std"])
            else:
                means.append(0); stds.append(0)
        ax.bar(x + i * width, means, width, yerr=stds, label=clf,
              color=PALETTE[i], capsize=3, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in methods])
    ax.set_ylabel("Accuracy (20×5-fold CV)")
    ax.set_title("(b) Confound-free features (784 features, no mag/color/pressure/proximity)")
    ax.set_ylim(0.7, 1.05)
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("Air-Cut Referencing Performance: Full vs Confound-Free Feature Sets",
                fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig20_confound_free_comparison")

    return {"figure": "fig20_confound_free_comparison"}


# ═══════════════════════════════════════════════════════════════════
def run_all():
    results = {}
    results["heldout"] = experiment_9()
    results["snr_budget"] = experiment_10()
    results["partial_correlation"] = experiment_11()
    results["learning_curve_fine"] = experiment_12()
    results["calibration"] = experiment_13()
    results["minirocket"] = experiment_14()
    results["confound_free_viz"] = experiment_15()

    out = cfg.RESULTS_DIR / "experiments_final.json"
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=conv)
    print(f"\nAll final-round results saved to {out}")


if __name__ == "__main__":
    run_all()
