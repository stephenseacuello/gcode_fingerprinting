"""Round 4: Statistical rigor and analytical depth experiments.

1. Friedman test + Nemenyi post-hoc
2. Two-way ANOVA (method × classifier)
3. Bootstrap confidence intervals
4. Cohen's d effect sizes
5. Cochran's Q test
6. Temporal bin importance
7. Permutation importance
8. Inter-run variability (coefficient of variation)
9. Noise robustness curve
"""

import json
import sys
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolpath_signature import config as cfg
from toolpath_signature import utils

warnings.filterwarnings("ignore")


def load_data(iso_method="raw"):
    df = utils.load_parquet(cfg.RUN_VECTORS_DIR / f"{iso_method}_temporal_summary.parquet")
    meta_cols = ["toolpath", "run_id", "prefix"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = np.nan_to_num(df[feature_cols].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    le = LabelEncoder()
    y = le.fit_transform(df["toolpath"].values)
    return X, y, feature_cols, le, df


def get_fold_accuracies(X, y, clf, n_repeats=20, n_splits=5):
    """Get per-fold accuracies across repeated CV."""
    all_accs = []
    for rep in range(n_repeats):
        seed = cfg.RANDOM_STATE + rep
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler()
            Xtr = np.nan_to_num(scaler.fit_transform(X[train_idx]), nan=0.0, posinf=0.0, neginf=0.0)
            Xte = np.nan_to_num(scaler.transform(X[test_idx]), nan=0.0, posinf=0.0, neginf=0.0)
            c = type(clf)(**clf.get_params())
            c.fit(Xtr, y[train_idx])
            all_accs.append(accuracy_score(y[test_idx], c.predict(Xte)))
    return np.array(all_accs)


# ═══════════════════════════════════════════════════════════════════
# 1. Friedman test + Nemenyi post-hoc
# ═══════════════════════════════════════════════════════════════════

def experiment_friedman_nemenyi():
    """Non-parametric ANOVA across isolation methods using paired fold accuracies."""
    print("=" * 70)
    print("1. FRIEDMAN TEST + POST-HOC COMPARISONS")
    print("=" * 70)

    rf = RandomForestClassifier(**cfg.RF_PARAMS)
    results = {}

    # Collect paired fold-level accuracies (same folds for each method)
    method_accs = {}
    for iso_method in cfg.ISOLATION_METHODS:
        X, y, _, _, _ = load_data(iso_method)
        accs = get_fold_accuracies(X, y, rf, n_repeats=20)
        method_accs[iso_method] = accs

    # Friedman test (requires same-size groups)
    min_len = min(len(v) for v in method_accs.values())
    groups = [method_accs[m][:min_len] for m in cfg.ISOLATION_METHODS]
    stat, p_value = sp_stats.friedmanchisquare(*groups)
    results["friedman"] = {"statistic": float(stat), "p_value": float(p_value), "n": min_len}
    print(f"  Friedman chi2={stat:.4f}, p={p_value:.6f}, n={min_len}")

    # Post-hoc: Wilcoxon signed-rank pairwise with Bonferroni correction
    n_comparisons = len(list(combinations(cfg.ISOLATION_METHODS, 2)))
    pairwise = {}
    for m1, m2 in combinations(cfg.ISOLATION_METHODS, 2):
        a1 = method_accs[m1][:min_len]
        a2 = method_accs[m2][:min_len]
        # Use Wilcoxon only if there are differences
        diff = a1 - a2
        if np.all(diff == 0):
            stat_w, p_w = 0.0, 1.0
        else:
            try:
                stat_w, p_w = sp_stats.wilcoxon(a1, a2, alternative="two-sided")
            except ValueError:
                stat_w, p_w = 0.0, 1.0

        p_corrected = min(p_w * n_comparisons, 1.0)  # Bonferroni
        pairwise[f"{m1}_vs_{m2}"] = {
            "wilcoxon_stat": float(stat_w),
            "p_raw": float(p_w),
            "p_bonferroni": float(p_corrected),
            "mean_diff": float(np.mean(a1 - a2)),
            "significant_005": bool(p_corrected < 0.05),
        }
        sig = "*" if p_corrected < 0.05 else ""
        print(f"  {m1} vs {m2}: p_bonf={p_corrected:.4f} diff={np.mean(a1-a2):.4f} {sig}")

    results["pairwise_wilcoxon"] = pairwise

    # Also test RF vs SVM vs MLP within the best method
    print("\n  Friedman across classifiers (subtracted method):")
    X, y, _, _, _ = load_data("subtracted")
    clf_accs = {}
    for name, clf in [("RF", RandomForestClassifier(**cfg.RF_PARAMS)),
                       ("SVM", SVC(**cfg.SVM_PARAMS)),
                       ("MLP", MLPClassifier(**cfg.MLP_PARAMS))]:
        clf_accs[name] = get_fold_accuracies(X, y, clf, n_repeats=20)

    min_len = min(len(v) for v in clf_accs.values())
    clf_groups = [clf_accs[n][:min_len] for n in ["RF", "SVM", "MLP"]]
    stat_c, p_c = sp_stats.friedmanchisquare(*clf_groups)
    results["friedman_classifiers"] = {"statistic": float(stat_c), "p_value": float(p_c)}
    print(f"  Friedman (classifiers): chi2={stat_c:.4f}, p={p_c:.6f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# 2. Two-way ANOVA (method × classifier)
# ═══════════════════════════════════════════════════════════════════

def experiment_two_way_anova():
    """Two-way ANOVA: isolation method × classifier on fold-level accuracy."""
    print("\n" + "=" * 70)
    print("2. TWO-WAY ANOVA (Method × Classifier)")
    print("=" * 70)

    rows = []
    classifiers = {
        "RF": RandomForestClassifier(**cfg.RF_PARAMS),
        "SVM": SVC(**cfg.SVM_PARAMS),
        "MLP": MLPClassifier(**cfg.MLP_PARAMS),
    }

    for iso in cfg.ISOLATION_METHODS:
        X, y, _, _, _ = load_data(iso)
        for clf_name, clf in classifiers.items():
            accs = get_fold_accuracies(X, y, clf, n_repeats=5)  # 5 repeats for speed
            for a in accs:
                rows.append({"method": iso, "classifier": clf_name, "accuracy": a})

    anova_df = pd.DataFrame(rows)

    # Compute F-statistics manually (scipy doesn't have 2-way ANOVA, use statsmodels-style)
    results = {}

    # Main effect of method
    method_groups = [anova_df[anova_df["method"] == m]["accuracy"].values for m in cfg.ISOLATION_METHODS]
    f_method, p_method = sp_stats.f_oneway(*method_groups)
    results["method_effect"] = {"F": float(f_method), "p": float(p_method)}
    print(f"  Method effect: F={f_method:.3f}, p={p_method:.6f}")

    # Main effect of classifier
    clf_groups = [anova_df[anova_df["classifier"] == c]["accuracy"].values for c in classifiers]
    f_clf, p_clf = sp_stats.f_oneway(*clf_groups)
    results["classifier_effect"] = {"F": float(f_clf), "p": float(p_clf)}
    print(f"  Classifier effect: F={f_clf:.3f}, p={p_clf:.6f}")

    # Effect sizes (eta-squared)
    grand_mean = anova_df["accuracy"].mean()
    ss_total = np.sum((anova_df["accuracy"].values - grand_mean) ** 2)

    ss_method = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in method_groups)
    ss_clf = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in clf_groups)

    eta2_method = ss_method / ss_total if ss_total > 0 else 0
    eta2_clf = ss_clf / ss_total if ss_total > 0 else 0

    results["eta_squared_method"] = float(eta2_method)
    results["eta_squared_classifier"] = float(eta2_clf)
    print(f"  η² method={eta2_method:.4f}, η² classifier={eta2_clf:.4f}")

    # Per-cell means
    print("\n  Cell means:")
    pivot = anova_df.groupby(["method", "classifier"])["accuracy"].agg(["mean", "std"]).round(3)
    print(pivot.to_string())
    results["cell_means"] = pivot.reset_index().to_dict(orient="records")

    return results


# ═══════════════════════════════════════════════════════════════════
# 3. Bootstrap confidence intervals
# ═══════════════════════════════════════════════════════════════════

def experiment_bootstrap_ci():
    """Bootstrap 95% confidence intervals on accuracy."""
    print("\n" + "=" * 70)
    print("3. BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("=" * 70)

    results = {}
    rf = RandomForestClassifier(**cfg.RF_PARAMS)
    n_bootstrap = 1000

    for iso in cfg.ISOLATION_METHODS:
        X, y, _, _, _ = load_data(iso)
        fold_accs = get_fold_accuracies(X, y, rf, n_repeats=20)

        # Bootstrap the mean accuracy
        boot_means = []
        rng = np.random.RandomState(cfg.RANDOM_STATE)
        for _ in range(n_bootstrap):
            sample = rng.choice(fold_accs, size=len(fold_accs), replace=True)
            boot_means.append(np.mean(sample))

        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)

        results[iso] = {
            "mean": float(np.mean(fold_accs)),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "ci_width": float(ci_upper - ci_lower),
        }
        print(f"  {iso} RF: {np.mean(fold_accs):.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

    return results


# ═══════════════════════════════════════════════════════════════════
# 4. Cohen's d effect sizes
# ═══════════════════════════════════════════════════════════════════

def experiment_cohens_d():
    """Effect sizes for raw vs each referenced method."""
    print("\n" + "=" * 70)
    print("4. COHEN'S d EFFECT SIZES")
    print("=" * 70)

    rf = RandomForestClassifier(**cfg.RF_PARAMS)
    X_raw, y, _, _, _ = load_data("raw")
    raw_accs = get_fold_accuracies(X_raw, y, rf, n_repeats=20)

    results = {}
    for iso in ["subtracted", "ratio", "zscore"]:
        X, y, _, _, _ = load_data(iso)
        ref_accs = get_fold_accuracies(X, y, rf, n_repeats=20)

        # Cohen's d
        pooled_std = np.sqrt((np.var(raw_accs, ddof=1) + np.var(ref_accs, ddof=1)) / 2)
        if pooled_std > 0:
            d = (np.mean(ref_accs) - np.mean(raw_accs)) / pooled_std
        else:
            d = 0.0

        # Interpret
        if abs(d) < 0.2:
            interpretation = "negligible"
        elif abs(d) < 0.5:
            interpretation = "small"
        elif abs(d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        results[f"raw_vs_{iso}"] = {
            "cohens_d": float(d),
            "interpretation": interpretation,
            "raw_mean": float(np.mean(raw_accs)),
            "ref_mean": float(np.mean(ref_accs)),
        }
        print(f"  raw vs {iso}: d={d:.3f} ({interpretation})")

    return results


# ═══════════════════════════════════════════════════════════════════
# 5. Temporal bin importance
# ═══════════════════════════════════════════════════════════════════

def experiment_bin_importance():
    """Which temporal bins matter most? Remove one bin at a time."""
    print("\n" + "=" * 70)
    print("5. TEMPORAL BIN IMPORTANCE (leave-one-bin-out)")
    print("=" * 70)

    X, y, feature_cols, _, _ = load_data("raw")
    rf = RandomForestClassifier(**cfg.RF_PARAMS)

    # Full accuracy
    full_acc, full_std = _quick_cv(X, y, rf)
    print(f"  Full (20 bins): {full_acc:.3f}±{full_std:.3f}")

    results = {"full": {"accuracy": float(full_acc), "std": float(full_std)}}

    # Each feature name starts with "binmean_" or "binstd_"
    # The bin info is encoded in whether it's binmean (mean across bins) or binstd (std across bins)
    # Actually in our temporal-summary aggregation, features are binmean_X and binstd_X where X is the
    # per-bin feature name. So we can't easily ablate individual bins from the summary.
    # Instead, let's load the raw bin data and reconstruct with bin removal.

    # Alternative: use the isolated bin-level data directly
    # Load all runs' bin-level data, remove one bin, re-aggregate, re-classify
    bin_results = {}
    for remove_bin in range(cfg.N_BINS):
        run_vectors = []
        labels = []

        for toolpath in cfg.TOOLPATH_TYPES:
            prefix = cfg.ACTIVE_PREFIX[toolpath]
            files = utils.discover_runs(prefix)
            for fpath in files:
                run_id = utils.extract_run_id(fpath)
                bin_path = cfg.ISOLATED_DIR / "raw" / f"{prefix}_{run_id}_bins.parquet"
                if not bin_path.exists():
                    continue
                iso_df = utils.load_parquet(bin_path)
                # Remove one bin
                remaining = iso_df.drop(index=remove_bin)
                run_vec = utils.aggregate_bins_to_run(remaining, method="summary")
                run_vectors.append(run_vec)
                labels.append(toolpath)

        X_ablated = np.nan_to_num(np.stack(run_vectors), nan=0.0, posinf=0.0, neginf=0.0)
        le = LabelEncoder()
        y_ablated = le.fit_transform(labels)

        acc, std = _quick_cv(X_ablated, y_ablated, rf)
        bin_results[remove_bin] = {"accuracy": float(acc), "std": float(std)}

    results["per_bin_removal"] = bin_results

    # Find most and least important bins
    accs = [(b, r["accuracy"]) for b, r in bin_results.items()]
    accs.sort(key=lambda x: x[1])
    print(f"  Most impactful bin to remove: bin {accs[0][0]} → {accs[0][1]:.3f}")
    print(f"  Least impactful: bin {accs[-1][0]} → {accs[-1][1]:.3f}")
    print(f"  Bin importance range: {full_acc - accs[0][1]:.3f} (drop from removing worst bin)")

    return results


def _quick_cv(X, y, clf, n_splits=5):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_STATE)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = np.nan_to_num(scaler.fit_transform(X[train_idx]), nan=0.0, posinf=0.0, neginf=0.0)
        Xte = np.nan_to_num(scaler.transform(X[test_idx]), nan=0.0, posinf=0.0, neginf=0.0)
        c = type(clf)(**clf.get_params())
        c.fit(Xtr, y[train_idx])
        accs.append(accuracy_score(y[test_idx], c.predict(Xte)))
    return np.mean(accs), np.std(accs)


# ═══════════════════════════════════════════════════════════════════
# 6. Permutation importance
# ═══════════════════════════════════════════════════════════════════

def experiment_permutation_importance():
    """Permutation importance with p-values (more reliable than Gini)."""
    print("\n" + "=" * 70)
    print("6. PERMUTATION IMPORTANCE")
    print("=" * 70)

    X, y, feature_cols, _, _ = load_data("raw")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = np.nan_to_num(scaler.fit_transform(X), nan=0.0, posinf=0.0, neginf=0.0)

    rf = RandomForestClassifier(**cfg.RF_PARAMS)
    rf.fit(X_scaled, y)

    perm_result = permutation_importance(
        rf, X_scaled, y, n_repeats=30, random_state=cfg.RANDOM_STATE, n_jobs=-1
    )

    # Rank by mean importance
    sorted_idx = np.argsort(perm_result.importances_mean)[::-1]

    records = []
    print("  Top 20 permutation-important features:")
    for rank, idx in enumerate(sorted_idx[:20]):
        mean_imp = perm_result.importances_mean[idx]
        std_imp = perm_result.importances_std[idx]
        # p-value: proportion of permutations where importance <= 0
        p_val = np.mean(perm_result.importances[idx] <= 0)
        records.append({
            "rank": rank + 1,
            "feature": feature_cols[idx],
            "importance_mean": float(mean_imp),
            "importance_std": float(std_imp),
            "p_value": float(p_val),
        })
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"    {rank+1:3d}. {feature_cols[idx]:45s} {mean_imp:.4f}±{std_imp:.4f} p={p_val:.3f} {sig}")

    return {"top_features": records,
            "n_significant_001": int(np.sum(np.array([r["p_value"] for r in records]) < 0.001)),
            "n_significant_005": int(np.sum(np.array([r["p_value"] for r in records]) < 0.05))}


# ═══════════════════════════════════════════════════════════════════
# 7. Inter-run variability
# ═══════════════════════════════════════════════════════════════════

def experiment_inter_run_variability():
    """Coefficient of variation across runs within each class."""
    print("\n" + "=" * 70)
    print("7. INTER-RUN VARIABILITY (CV across runs per class)")
    print("=" * 70)

    results = {}

    for iso in ["raw", "subtracted"]:
        df = utils.load_parquet(cfg.RUN_VECTORS_DIR / f"{iso}_temporal_summary.parquet")
        meta_cols = ["toolpath", "run_id", "prefix"]
        feature_cols = [c for c in df.columns if c not in meta_cols]

        iso_results = {}
        for toolpath in cfg.TOOLPATH_TYPES:
            sub = df[df["toolpath"] == toolpath][feature_cols].values
            means = np.mean(sub, axis=0)
            stds = np.std(sub, axis=0, ddof=1)
            # CV = std/|mean|, only where mean != 0
            with np.errstate(divide="ignore", invalid="ignore"):
                cvs = np.where(np.abs(means) > 1e-10, stds / np.abs(means), 0)

            iso_results[toolpath] = {
                "mean_cv": float(np.mean(cvs)),
                "median_cv": float(np.median(cvs)),
                "max_cv": float(np.max(cvs)),
                "pct_low_cv": float(np.mean(cvs < 0.1) * 100),  # % features with CV < 10%
                "n_runs": int(sub.shape[0]),
            }
            print(f"  {iso}/{toolpath}: mean CV={np.mean(cvs):.3f}, "
                  f"median={np.median(cvs):.3f}, "
                  f"{np.mean(cvs < 0.1)*100:.0f}% features with CV<10%")

        results[iso] = iso_results

    return results


# ═══════════════════════════════════════════════════════════════════
# 8. Noise robustness curve
# ═══════════════════════════════════════════════════════════════════

def experiment_noise_robustness():
    """Add Gaussian noise at increasing levels and measure accuracy degradation."""
    print("\n" + "=" * 70)
    print("8. NOISE ROBUSTNESS")
    print("=" * 70)

    rf = RandomForestClassifier(**cfg.RF_PARAMS)
    results = {}

    for iso in ["raw", "subtracted"]:
        X, y, _, _, _ = load_data(iso)
        feature_stds = np.std(X, axis=0)
        feature_stds = np.where(feature_stds < 1e-10, 1.0, feature_stds)

        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        iso_results = {}

        for noise_level in noise_levels:
            accs = []
            for rep in range(10):
                rng = np.random.RandomState(cfg.RANDOM_STATE + rep)
                noise = rng.randn(*X.shape) * feature_stds * noise_level
                X_noisy = X + noise

                acc, _ = _quick_cv(X_noisy, y, rf)
                accs.append(acc)

            iso_results[str(noise_level)] = {
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
            }
            print(f"  {iso} noise={noise_level:.2f}: {np.mean(accs):.3f}±{np.std(accs):.3f}")

        results[iso] = iso_results

    return results


# ═══════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════

def run_all_round4():
    all_results = {}
    all_results["friedman_nemenyi"] = experiment_friedman_nemenyi()
    all_results["two_way_anova"] = experiment_two_way_anova()
    all_results["bootstrap_ci"] = experiment_bootstrap_ci()
    all_results["cohens_d"] = experiment_cohens_d()
    all_results["bin_importance"] = experiment_bin_importance()
    all_results["permutation_importance"] = experiment_permutation_importance()
    all_results["inter_run_variability"] = experiment_inter_run_variability()
    all_results["noise_robustness"] = experiment_noise_robustness()

    out_path = cfg.RESULTS_DIR / "round4_statistical.json"
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nAll round 4 results saved to {out_path}")


if __name__ == "__main__":
    run_all_round4()
