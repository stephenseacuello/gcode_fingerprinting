"""Fix figure issues flagged by reviewer: heatmap colorbar, learning curve y-axis, fig6b size."""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolpath_signature import config as cfg
from toolpath_signature import utils

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
PALETTE = sns.color_palette("colorblind")
CLASS_COLORS = {"adaptive": PALETTE[0], "face": PALETTE[1], "pocket": PALETTE[2]}
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(cfg.FIGURES_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png/pdf")


def fix_learning_curve():
    """Fix y-axis to cap at 1.0."""
    print("Fixing fig8_learning_curve...")
    with open(cfg.ABLATION_DIR / "all_ablation_results.json") as f:
        abl = json.load(f)

    lc_data = abl["learning_curve"]
    sizes = sorted([int(k) for k in lc_data.keys()])

    fig, ax = plt.subplots(figsize=(8, 6))
    accs = [lc_data[str(s)]["accuracy_mean"] for s in sizes]
    stds = [lc_data[str(s)]["accuracy_std"] for s in sizes]

    ax.errorbar(sizes, accs, yerr=stds, marker="o", capsize=5, color=PALETTE[0],
               linewidth=2, markersize=8)
    # Clip fill_between to [0, 1]
    lower = np.clip(np.array(accs) - np.array(stds), 0, 1)
    upper = np.clip(np.array(accs) + np.array(stds), 0, 1)
    ax.fill_between(sizes, lower, upper, color=PALETTE[0], alpha=0.15)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Full-data accuracy")
    ax.set_xlabel("Training Runs per Class")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve (RF, Raw Features)")
    ax.set_ylim(0.75, 1.02)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig8_learning_curve")


def fix_heatmap():
    """Fix colorbar range to start at 0.90."""
    print("Fixing fig9_sensor_heatmap...")

    sensor_groups = list(cfg.SENSOR_GROUPS.keys()) + ["all"]

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
            X = np.nan_to_num(df[selected].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
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
               ax=ax, vmin=0.90, vmax=1.0)
    ax.set_title("Sensor Group x Isolation Method Classification Accuracy")
    ax.set_xlabel("Isolation Method")
    ax.set_ylabel("Sensor Group")
    fig.tight_layout()
    save_fig(fig, "fig9_sensor_heatmap")


def fix_temporal_overlay():
    """Make fig6b larger and clearer."""
    print("Fixing fig6b_temporal_overlay...")

    imp_df = pd.read_csv(cfg.IMPORTANCE_DIR / "raw_rf_importance.csv")
    viz_feat = "frame_r2.Mx_mean"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for col, (iso_method, title) in enumerate([("raw", "Raw Active"), ("subtracted", "Subtracted (Active - Air)")]):
        ax = axes[col]
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
                if viz_feat in iso_df.columns:
                    all_vals.append(iso_df[viz_feat].values)

            if all_vals:
                all_vals = np.stack(all_vals)
                mean_vals = np.mean(all_vals, axis=0)
                std_vals = np.std(all_vals, axis=0)
                x = range(1, cfg.N_BINS + 1)
                ax.plot(x, mean_vals, color=CLASS_COLORS[toolpath], linewidth=2.5,
                       label=toolpath.capitalize())
                ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals,
                              color=CLASS_COLORS[toolpath], alpha=0.15)

        ax.set_xlabel("Temporal Bin", fontsize=12)
        ax.set_ylabel(viz_feat, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11)
        ax.tick_params(labelsize=10)

    fig.suptitle("Temporal Fingerprint Overlay: All Toolpath Types", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig6b_temporal_overlay")


def fix_accuracy_bars():
    """Start y-axis at 0.80 to show meaningful differences."""
    print("Fixing fig3_accuracy_bars...")
    summary = pd.read_csv(cfg.CLASSIFICATION_DIR / "summary_table.csv")
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
    ax.set_ylim(0.30, 1.05)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig3_accuracy_bars")


if __name__ == "__main__":
    fix_learning_curve()
    fix_heatmap()
    fix_temporal_overlay()
    fix_accuracy_bars()
    print("\nAll figure fixes complete.")
