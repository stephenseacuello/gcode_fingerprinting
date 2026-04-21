"""Phase 7: Generate all figures (PNG + PDF)."""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolpath_signature import config as cfg
from toolpath_signature import utils

warnings.filterwarnings("ignore")

# Style
sns.set_style("whitegrid")
PALETTE = sns.color_palette("colorblind")
CLASS_COLORS = {"adaptive": PALETTE[0], "face": PALETTE[1], "pocket": PALETTE[2]}
FONTSIZE = 11
plt.rcParams.update({"font.size": FONTSIZE, "figure.dpi": 150})


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(cfg.FIGURES_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png/pdf")


# ── Fig 1: PSD Comparison ──────────────────────────────────────────────────

def fig1_psd():
    print("\n── Fig 1: PSD Comparison ──")
    sensors_to_plot = ["frame_l2.Ax", "frame_l2.Gx", "spindle"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))

    for row, toolpath in enumerate(cfg.TOOLPATH_TYPES):
        # Load one representative air and active run
        air_files = utils.discover_runs(cfg.AIR_PREFIX[toolpath])
        act_files = utils.discover_runs(cfg.ACTIVE_PREFIX[toolpath])
        air_df = utils.load_run(air_files[0])
        act_df = utils.load_run(act_files[0])

        for col_idx, sensor in enumerate(sensors_to_plot):
            ax = axes[row, col_idx]
            fs = 4.0  # ~0.25s sampling

            for df, label, color in [(air_df, "Air", PALETTE[0]),
                                      (act_df, "Active", PALETTE[1])]:
                if sensor in df.columns:
                    sig = df[sensor].dropna().values
                    nperseg = min(64, len(sig) // 2)
                    if nperseg < 8:
                        continue
                    freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg)
                    psd_db = 10 * np.log10(psd + 1e-20)
                    ax.plot(freqs, psd_db, label=label, color=color, linewidth=1.2)

            ax.set_title(f"{toolpath} / {sensor}", fontsize=9)
            if row == 2:
                ax.set_xlabel("Frequency (Hz)")
            if col_idx == 0:
                ax.set_ylabel("PSD (dB)")
            if row == 0 and col_idx == 2:
                ax.legend(fontsize=8)

    fig.suptitle("Power Spectral Density: Air vs Active Cutting", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig1_psd")


# ── Fig 2: PCA Scatter ────────────────────────────────────────────────────

def fig2_pca_scatter():
    print("\n── Fig 2: PCA Scatter ──")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for idx, iso_method in enumerate(cfg.ISOLATION_METHODS):
        ax = axes[idx // 2, idx % 2]
        data = dict(np.load(cfg.PCA_DIR / f"{iso_method}_pca.npz", allow_pickle=True))
        X_pca = data["X_pca"]
        labels = data["labels"]
        evr = data["explained_variance_ratio"]

        for cls_name, color in CLASS_COLORS.items():
            mask = labels == cls_name
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[color], label=cls_name, s=80, alpha=0.7, edgecolors="black", linewidth=0.5)

        ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
        ax.set_title(iso_method.capitalize())
        if idx == 0:
            ax.legend()

    fig.suptitle("PCA Projections by Isolation Method", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig2_pca_scatter")


# ── Fig 3: Classification Accuracy Bars ───────────────────────────────────

def fig3_accuracy_bars():
    print("\n── Fig 3: Classification Accuracy Bars ──")
    summary = pd.read_csv(cfg.CLASSIFICATION_DIR / "summary_table.csv")
    # Filter to no-PCA only for cleaner plot
    nopca = summary[~summary["pca"]].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    clf_names = ["RF", "SVM", "MLP"]
    x = np.arange(len(cfg.ISOLATION_METHODS))
    width = 0.25

    for i, clf in enumerate(clf_names):
        sub = nopca[nopca["classifier"] == clf].set_index("isolation")
        vals = [sub.loc[m, "accuracy_mean"] if m in sub.index else 0 for m in cfg.ISOLATION_METHODS]
        errs = [sub.loc[m, "accuracy_std"] if m in sub.index else 0 for m in cfg.ISOLATION_METHODS]
        ax.bar(x + i * width, vals, width, yerr=errs, label=clf, color=PALETTE[i],
               capsize=3, edgecolor="black", linewidth=0.5)

    ax.axhline(1/3, color="gray", linestyle="--", linewidth=1, label="Random baseline")
    ax.set_xlabel("Isolation Method")
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification Accuracy by Isolation Method and Classifier (No PCA)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in cfg.ISOLATION_METHODS])
    ax.set_ylim(0, 1.08)
    ax.legend()

    fig.tight_layout()
    save_fig(fig, "fig3_accuracy_bars")


# ── Fig 4: Confusion Matrices ─────────────────────────────────────────────

def fig4_confusion():
    print("\n── Fig 4: Confusion Matrices ──")
    with open(cfg.CLASSIFICATION_DIR / "all_results.json") as f:
        results = json.load(f)

    # Show no-PCA results
    clf_names = ["RF", "SVM", "MLP"]
    fig, axes = plt.subplots(len(cfg.ISOLATION_METHODS), len(clf_names), figsize=(14, 16))

    class_labels = ["Adaptive", "Facing", "Pocketing"]

    for row, iso in enumerate(cfg.ISOLATION_METHODS):
        for col, clf in enumerate(clf_names):
            key = f"{iso}_{clf}_nopca"
            ax = axes[row, col]
            cm = np.array(results[key]["confusion_matrix"])
            # Normalize
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

            sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                       xticklabels=class_labels, yticklabels=class_labels,
                       ax=ax, vmin=0, vmax=100, cbar=False)
            ax.set_title(f"{iso.capitalize()} + {clf}", fontsize=10)
            if col == 0:
                ax.set_ylabel("True")
            if row == len(cfg.ISOLATION_METHODS) - 1:
                ax.set_xlabel("Predicted")

    fig.suptitle("Normalized Confusion Matrices (%)", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig4_confusion")


# ── Fig 5: Feature Importance ─────────────────────────────────────────────

def fig5_importance():
    print("\n── Fig 5: Feature Importance ──")

    # Use raw method (best performing)
    imp_df = pd.read_csv(cfg.IMPORTANCE_DIR / "raw_rf_importance.csv")
    top20 = imp_df.head(20).copy()

    # Color by sensor group
    def get_group(feat_name):
        for mod in cfg.ARDUINO_MODULES:
            if mod in feat_name:
                return mod
        for ec in cfg.ELECTRICAL_COLS:
            if ec in feat_name:
                return "electrical"
        return "other"

    top20["group"] = top20["feature"].apply(get_group)
    group_colors = {mod: PALETTE[i] for i, mod in enumerate(cfg.ARDUINO_MODULES)}
    group_colors["electrical"] = PALETTE[6]
    group_colors["other"] = "gray"

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [group_colors.get(g, "gray") for g in top20["group"]]
    ax.barh(range(len(top20)), top20["importance"].values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("RF Importance")
    ax.set_title("Top 20 Feature Importances (Raw, RF)")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=group_colors[g], edgecolor="black", label=g)
                      for g in sorted(set(top20["group"]))]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()
    save_fig(fig, "fig5_importance")


# ── Fig 6: Temporal Fingerprint (KEY FIGURE) ──────────────────────────────

def fig6_temporal_fingerprint():
    print("\n── Fig 6: Temporal Fingerprint (KEY FIGURE) ──")

    # Find the top feature from importance analysis
    imp_df = pd.read_csv(cfg.IMPORTANCE_DIR / "raw_rf_importance.csv")
    # Use a sensor-based feature for visualization
    top_sensor_features = []
    for _, row in imp_df.iterrows():
        feat = row["feature"]
        # Extract the base sensor and stat
        for stat in cfg.BLOCK_STATS:
            if f"_{stat}" in feat:
                base = feat.replace(f"binmean_", "").replace(f"binstd_", "").replace(f"_{stat}", "")
                top_sensor_features.append((base, stat, feat))
                break
        if len(top_sensor_features) >= 3:
            break

    # Use the first top feature's sensor with mean stat for clear visualization
    viz_features = ["frame_r2.Mx_mean", "y_bed__3.Az_mean", "y_bed__3.Mx_mean"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for row, toolpath in enumerate(cfg.TOOLPATH_TYPES):
        prefix = cfg.ACTIVE_PREFIX[toolpath]
        files = utils.discover_runs(prefix)

        for col, (viz_feat, title_suffix) in enumerate([
            (viz_features[0], "Raw Active"),
            (viz_features[0], "Subtracted"),
        ]):
            ax = axes[row, col]
            iso_method = "raw" if col == 0 else "subtracted"

            # Plot individual runs
            for fpath in files:
                run_id = utils.extract_run_id(fpath)
                bin_path = cfg.ISOLATED_DIR / iso_method / f"{prefix}_{run_id}_bins.parquet"
                if not bin_path.exists():
                    continue
                iso_df = utils.load_parquet(bin_path)
                if viz_feat in iso_df.columns:
                    ax.plot(range(1, cfg.N_BINS + 1), iso_df[viz_feat].values,
                           color=CLASS_COLORS[toolpath], alpha=0.2, linewidth=0.8)

            # Plot mean
            all_vals = []
            for fpath in files:
                run_id = utils.extract_run_id(fpath)
                bin_path = cfg.ISOLATED_DIR / iso_method / f"{prefix}_{run_id}_bins.parquet"
                if not bin_path.exists():
                    continue
                iso_df = utils.load_parquet(bin_path)
                if viz_feat in iso_df.columns:
                    all_vals.append(iso_df[viz_feat].values)

            if all_vals:
                all_vals = np.stack(all_vals)
                mean_vals = np.mean(all_vals, axis=0)
                std_vals = np.std(all_vals, axis=0)
                x = range(1, cfg.N_BINS + 1)
                ax.plot(x, mean_vals, color=CLASS_COLORS[toolpath], linewidth=2.5, label=f"Mean")
                ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals,
                              color=CLASS_COLORS[toolpath], alpha=0.15)

            ax.set_title(f"{toolpath.capitalize()} — {title_suffix}", fontsize=10)
            if row == 2:
                ax.set_xlabel("Temporal Bin")
            if col == 0:
                ax.set_ylabel(viz_feat)

    fig.suptitle(f"Temporal Fingerprints: {viz_features[0]}", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig6_temporal_fingerprint")

    # Also create an overlay version (all 3 types on same plot)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for col, (iso_method, title) in enumerate([("raw", "Raw Active"), ("subtracted", "Subtracted")]):
        ax = axes2[col]
        for toolpath in cfg.TOOLPATH_TYPES:
            prefix = cfg.ACTIVE_PREFIX[toolpath]
            files = utils.discover_runs(prefix)
            all_vals = []
            for fpath in files:
                run_id = utils.extract_run_id(fpath)
                bin_path = cfg.ISOLATED_DIR / iso_method / f"{prefix}_{run_id}_bins.parquet"
                if not bin_path.exists():
                    continue
                iso_df = utils.load_parquet(bin_path)
                if viz_features[0] in iso_df.columns:
                    all_vals.append(iso_df[viz_features[0]].values)

            if all_vals:
                all_vals = np.stack(all_vals)
                mean_vals = np.mean(all_vals, axis=0)
                std_vals = np.std(all_vals, axis=0)
                x = range(1, cfg.N_BINS + 1)
                ax.plot(x, mean_vals, color=CLASS_COLORS[toolpath], linewidth=2, label=toolpath.capitalize())
                ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals,
                              color=CLASS_COLORS[toolpath], alpha=0.15)

        ax.set_xlabel("Temporal Bin")
        ax.set_ylabel(viz_features[0])
        ax.set_title(title)
        ax.legend()

    fig2.suptitle("Temporal Fingerprint Overlay: All Toolpath Types", fontsize=13, y=1.02)
    fig2.tight_layout()
    save_fig(fig2, "fig6b_temporal_overlay")


# ── Fig 7: Ablation Bar Charts ───────────────────────────────────────────

def fig7_ablation():
    print("\n── Fig 7: Ablation Bar Charts ──")
    with open(cfg.ABLATION_DIR / "all_ablation_results.json") as f:
        abl = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 7a: Sensor group ablation
    ax = axes[0, 0]
    sensor_data = abl["sensor_ablation"]
    names = list(sensor_data.keys())
    accs = [sensor_data[n]["accuracy"] for n in names]
    stds = [sensor_data[n]["std"] for n in names]
    ax.bar(range(len(names)), accs, yerr=stds, capsize=3, color=PALETTE[0], edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("(a) Sensor Group Ablation")
    ax.set_ylim(0, 1.08)

    # 7b: Feature stat ablation
    ax = axes[0, 1]
    stat_data = abl["stat_ablation"]
    names = list(stat_data.keys())
    accs = [stat_data[n]["accuracy"] for n in names]
    stds = [stat_data[n]["std"] for n in names]
    ax.bar(range(len(names)), accs, yerr=stds, capsize=3, color=PALETTE[1], edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("(b) Feature Statistic Ablation")
    ax.set_ylim(0, 1.08)

    # 7c: Alignment ablation
    ax = axes[0, 2]
    align_data = abl["alignment_ablation"]
    names = list(align_data.keys())
    accs = [align_data[n]["accuracy"] for n in names]
    stds = [align_data[n]["std"] for n in names]
    ax.bar(range(len(names)), accs, yerr=stds, capsize=3, color=PALETTE[2], edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("(c) Alignment Ablation")
    ax.set_ylim(0, 1.08)

    # 7d: PCA ablation
    ax = axes[1, 0]
    pca_data = abl["pca_ablation"]
    names = list(pca_data.keys())
    accs = [pca_data[n]["accuracy"] for n in names]
    stds = [pca_data[n].get("std", 0) for n in names]
    ax.bar(range(len(names)), accs, yerr=stds, capsize=3, color=PALETTE[3], edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("(d) PCA Ablation")
    ax.set_ylim(0, 1.08)

    # 7e: Learning curve (replot here too)
    ax = axes[1, 1]
    lc_data = abl["learning_curve"]
    sizes = sorted([int(k) for k in lc_data.keys()])
    accs = [lc_data[str(s)]["accuracy_mean"] for s in sizes]
    stds = [lc_data[str(s)]["accuracy_std"] for s in sizes]
    ax.errorbar(sizes, accs, yerr=stds, marker="o", capsize=4, color=PALETTE[4], linewidth=2)
    ax.set_xlabel("Training Runs per Class")
    ax.set_ylabel("Accuracy")
    ax.set_title("(e) Learning Curve")
    ax.set_ylim(0, 1.08)

    # Empty subplot
    axes[1, 2].axis("off")

    fig.suptitle("Ablation Study Results", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig7_ablation")


# ── Fig 8: Learning Curve ─────────────────────────────────────────────────

def fig8_learning_curve():
    print("\n── Fig 8: Learning Curve ──")
    with open(cfg.ABLATION_DIR / "all_ablation_results.json") as f:
        abl = json.load(f)

    lc_data = abl["learning_curve"]
    sizes = sorted([int(k) for k in lc_data.keys()])

    fig, ax = plt.subplots(figsize=(8, 6))
    accs = [lc_data[str(s)]["accuracy_mean"] for s in sizes]
    stds = [lc_data[str(s)]["accuracy_std"] for s in sizes]

    ax.errorbar(sizes, accs, yerr=stds, marker="o", capsize=5, color=PALETTE[0],
               linewidth=2, markersize=8)
    ax.fill_between(sizes, np.array(accs) - np.array(stds), np.array(accs) + np.array(stds),
                   color=PALETTE[0], alpha=0.15)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Full-data accuracy")
    ax.set_xlabel("Training Runs per Class")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve (RF, Raw Features)")
    ax.set_ylim(0.7, 1.05)
    ax.legend()

    fig.tight_layout()
    save_fig(fig, "fig8_learning_curve")


# ── Fig 9: Sensor Contribution Heatmap ────────────────────────────────────

def fig9_sensor_heatmap():
    print("\n── Fig 9: Sensor Contribution Heatmap ──")

    sensor_groups = list(cfg.SENSOR_GROUPS.keys())
    sensor_groups.append("all")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    results = np.zeros((len(sensor_groups), len(cfg.ISOLATION_METHODS)))

    for j, iso_method in enumerate(cfg.ISOLATION_METHODS):
        df = utils.load_parquet(cfg.RUN_VECTORS_DIR / f"{iso_method}_temporal_summary.parquet")
        meta_cols = ["toolpath", "run_id", "prefix"]
        feature_cols = [c for c in df.columns if c not in meta_cols]

        le = LabelEncoder()
        y = le.fit_transform(df["toolpath"].values)

        for i, group in enumerate(sensor_groups):
            if group == "all":
                selected = feature_cols
            elif group in cfg.SENSOR_GROUPS:
                group_cols = cfg.SENSOR_GROUPS[group]
                selected = [f for f in feature_cols if any(sc in f for sc in group_cols)]
            else:
                selected = feature_cols

            if not selected:
                continue

            X = df[selected].values.astype(np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            skf = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
            accs = []
            for train_idx, test_idx in skf.split(X, y):
                scaler = StandardScaler()
                X_train = np.nan_to_num(scaler.fit_transform(X[train_idx]), nan=0.0, posinf=0.0, neginf=0.0)
                X_test = np.nan_to_num(scaler.transform(X[test_idx]), nan=0.0, posinf=0.0, neginf=0.0)
                clf = RandomForestClassifier(**cfg.RF_PARAMS)
                clf.fit(X_train, y[train_idx])
                accs.append(accuracy_score(y[test_idx], clf.predict(X_test)))
            results[i, j] = np.mean(accs)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(results, annot=True, fmt=".3f", cmap="YlOrRd",
               xticklabels=[m.capitalize() for m in cfg.ISOLATION_METHODS],
               yticklabels=sensor_groups,
               ax=ax, vmin=0.3, vmax=1.0)
    ax.set_title("Sensor Group × Isolation Method Classification Accuracy")
    ax.set_xlabel("Isolation Method")
    ax.set_ylabel("Sensor Group")

    fig.tight_layout()
    save_fig(fig, "fig9_sensor_heatmap")


def run_phase7():
    print("=" * 70)
    print("PHASE 7: Generating Figures")
    print("=" * 70)

    fig1_psd()
    fig2_pca_scatter()
    fig3_accuracy_bars()
    fig4_confusion()
    fig5_importance()
    fig6_temporal_fingerprint()
    fig7_ablation()
    fig8_learning_curve()
    fig9_sensor_heatmap()

    print(f"\nAll figures saved to {cfg.FIGURES_DIR}")
    print("Phase 7 complete.")


if __name__ == "__main__":
    run_phase7()
