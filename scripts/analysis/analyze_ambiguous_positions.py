#!/usr/bin/env python3
"""
Analyze which sensor features discriminate between G-code sequences
that share the same (window_index, operation_type) — the ambiguous positions.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data ───────────────────────────────────────────────────────────
BASE = "outputs/decoder20260304/preprocessed_v8_f98/fold_1"

print("Loading data...")
d = np.load(f"{BASE}/train_sequences.npz", allow_pickle=True)
meta = json.load(open(f"{BASE}/metadata.json"))

continuous   = d["continuous"]        # (303, 256, 98)
gcode_texts  = d["gcode_texts"]       # (303,) strings
window_index = d["window_index"]      # (303,)
operation_type = d["operation_type"]  # (303,)
source_file  = d["source_file"]       # (303,)
col_names    = meta["continuous_columns"]  # 98 names

N, T, C = continuous.shape
print(f"Dataset: {N} samples, {T} timesteps, {C} channels")
print(f"Unique operation types: {np.unique(operation_type)}")
print(f"Unique window indices: {len(np.unique(window_index))}")

# ── 2. Find ambiguous positions ───────────────────────────────────────────
print("\n" + "="*70)
print("FINDING AMBIGUOUS POSITIONS")
print("="*70)

# Group by (window_index, operation_type)
position_groups = defaultdict(list)
for i in range(N):
    key = (int(window_index[i]), int(operation_type[i]))
    position_groups[key].append(i)

ambiguous_positions = {}
for key, indices in position_groups.items():
    texts = set(str(gcode_texts[i]) for i in indices)
    if len(texts) > 1:
        ambiguous_positions[key] = indices

total_ambig_samples = sum(len(v) for v in ambiguous_positions.values())
print(f"Total (window_index, operation_type) groups: {len(position_groups)}")
print(f"Ambiguous positions (>1 distinct gcode): {len(ambiguous_positions)}")
print(f"Total samples in ambiguous positions: {total_ambig_samples}")
print(f"Non-ambiguous samples: {N - total_ambig_samples}")

# Show some examples
print("\nExample ambiguous positions:")
for i, (key, indices) in enumerate(sorted(ambiguous_positions.items())[:5]):
    texts = [str(gcode_texts[j]) for j in indices]
    unique_texts = set(texts)
    print(f"  Position {key}: {len(indices)} samples, {len(unique_texts)} distinct G-codes")
    for t in list(unique_texts)[:3]:
        count = texts.count(t)
        preview = t[:80] + "..." if len(t) > 80 else t
        print(f"    [{count}x] {preview}")

# ── 3. Compute per-window summary statistics ──────────────────────────────
print("\n" + "="*70)
print("COMPUTING SUMMARY STATISTICS")
print("="*70)

stat_names = ["mean", "std", "max", "min", "rms"]

def compute_global_stats(windows):
    """Compute 5 summary stats per channel for each window. Shape: (n, 98*5)"""
    n = windows.shape[0]
    feats = np.zeros((n, C * 5), dtype=np.float32)
    feats[:, 0*C:1*C] = np.mean(windows, axis=1)
    feats[:, 1*C:2*C] = np.std(windows, axis=1)
    feats[:, 2*C:3*C] = np.max(windows, axis=1)
    feats[:, 3*C:4*C] = np.min(windows, axis=1)
    feats[:, 4*C:5*C] = np.sqrt(np.mean(windows**2, axis=1))
    return feats

def compute_temporal_stats(windows):
    """Split into 4 quarters, compute mean per quarter per channel. Shape: (n, 4*98)"""
    n = windows.shape[0]
    quarter_len = T // 4  # 64
    feats = np.zeros((n, 4 * C), dtype=np.float32)
    for q in range(4):
        start = q * quarter_len
        end = (q + 1) * quarter_len
        feats[:, q*C:(q+1)*C] = np.mean(windows[:, start:end, :], axis=1)
    return feats

global_feat_names = [f"{s}_{col}" for s in stat_names for col in col_names]
temporal_feat_names = [f"q{q}mean_{col}" for q in range(4) for col in col_names]

all_global_feats = compute_global_stats(continuous)
all_temporal_feats = compute_temporal_stats(continuous)

print(f"Global features shape: {all_global_feats.shape}  ({len(global_feat_names)} names)")
print(f"Temporal features shape: {all_temporal_feats.shape}  ({len(temporal_feat_names)} names)")

# ── 4. Per-position discrimination analysis ──────────────────────────────
print("\n" + "="*70)
print("PER-POSITION DISCRIMINATION ANALYSIS")
print("="*70)

# Track which channels/stats are discriminative across positions
channel_importance_accum = defaultdict(list)   # channel_name -> list of importances
stat_importance_accum = defaultdict(list)       # stat_name -> list of total importance
temporal_vs_global = []  # (pos_key, global_acc, temporal_acc, combined_acc)

per_position_results = []

for pos_key, indices in sorted(ambiguous_positions.items()):
    indices = np.array(indices)
    texts = [str(gcode_texts[j]) for j in indices]
    le = LabelEncoder()
    labels = le.fit_transform(texts)
    n_classes = len(le.classes_)

    if len(indices) < 6 or n_classes < 2:
        continue  # need enough samples for CV

    # Check class balance — need at least 2 per class for stratified CV
    class_counts = Counter(labels)
    if min(class_counts.values()) < 2:
        continue

    X_global = all_global_feats[indices]
    X_temporal = all_temporal_feats[indices]
    X_combined = np.hstack([X_global, X_temporal])
    y = labels

    # Remove constant features for this subset
    global_var = np.var(X_global, axis=0)
    temporal_var = np.var(X_temporal, axis=0)
    combined_var = np.var(X_combined, axis=0)

    # ── Mutual information on global features ──
    valid_global = global_var > 1e-10
    if valid_global.sum() > 0:
        mi = np.zeros(X_global.shape[1])
        mi[valid_global] = mutual_info_classif(
            X_global[:, valid_global], y, discrete_features=False, random_state=42
        )

        # Accumulate per-channel importance (sum MI across stats for each channel)
        for stat_idx, stat_name in enumerate(stat_names):
            for ch_idx, ch_name in enumerate(col_names):
                feat_idx = stat_idx * C + ch_idx
                if mi[feat_idx] > 0:
                    channel_importance_accum[ch_name].append(mi[feat_idx])
                    stat_importance_accum[stat_name].append(mi[feat_idx])

    # ── Quick RF accuracy ──
    n_cv = min(3, min(class_counts.values()))
    if n_cv < 2:
        continue

    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)

    try:
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        global_scores = cross_val_score(rf, X_global, y, cv=cv, scoring="accuracy")
        global_acc = global_scores.mean()
    except:
        global_acc = np.nan

    try:
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        temporal_scores = cross_val_score(rf, X_temporal, y, cv=cv, scoring="accuracy")
        temporal_acc = temporal_scores.mean()
    except:
        temporal_acc = np.nan

    try:
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        combined_scores = cross_val_score(rf, X_combined, y, cv=cv, scoring="accuracy")
        combined_acc = combined_scores.mean()
    except:
        combined_acc = np.nan

    # Majority vote baseline (always predict the most common class)
    majority_acc = max(class_counts.values()) / len(y)

    per_position_results.append({
        "pos": pos_key,
        "n_samples": len(indices),
        "n_classes": n_classes,
        "majority_acc": majority_acc,
        "global_acc": global_acc,
        "temporal_acc": temporal_acc,
        "combined_acc": combined_acc,
    })

    temporal_vs_global.append((pos_key, global_acc, temporal_acc, combined_acc))

print(f"\nAnalyzed {len(per_position_results)} ambiguous positions (with enough samples for CV)")

# ── 5. Feature importance via full RF on all ambiguous samples ────────────
print("\n" + "="*70)
print("AGGREGATE FEATURE IMPORTANCE (RF on all ambiguous samples pooled)")
print("="*70)

# Pool all ambiguous samples, label = gcode_text
all_ambig_indices = []
all_ambig_labels = []
for pos_key, indices in ambiguous_positions.items():
    for idx in indices:
        all_ambig_indices.append(idx)
        # Use (pos_key, gcode_text) as label so we don't mix across positions
        all_ambig_labels.append(f"{pos_key}|{gcode_texts[idx]}")

all_ambig_indices = np.array(all_ambig_indices)
le_all = LabelEncoder()
y_all = le_all.fit_transform(all_ambig_labels)

X_global_all = all_global_feats[all_ambig_indices]
X_temporal_all = all_temporal_feats[all_ambig_indices]
X_combined_all = np.hstack([X_global_all, X_temporal_all])
combined_feat_names = global_feat_names + temporal_feat_names

print(f"Pooled ambiguous samples: {len(all_ambig_indices)}")
print(f"Unique (position, gcode) labels: {len(le_all.classes_)}")

# Train RF on global features
rf_global = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_global.fit(X_global_all, y_all)
global_importances = rf_global.feature_importances_

# Train RF on combined features
rf_combined = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_combined.fit(X_combined_all, y_all)
combined_importances = rf_combined.feature_importances_

# ── 6. Aggregate: Top channels ───────────────────────────────────────────
print("\n" + "="*70)
print("TOP 20 MOST DISCRIMINATIVE SENSOR CHANNELS")
print("="*70)

# Method 1: Sum RF importance across stats for each channel
channel_rf_importance = defaultdict(float)
for stat_idx, stat_name in enumerate(stat_names):
    for ch_idx, ch_name in enumerate(col_names):
        feat_idx = stat_idx * C + ch_idx
        channel_rf_importance[ch_name] += global_importances[feat_idx]

sorted_channels = sorted(channel_rf_importance.items(), key=lambda x: -x[1])

print(f"\n{'Rank':<5} {'Channel':<30} {'RF Importance (sum)':<20} {'MI count':<10} {'MI mean':<10}")
print("-" * 75)
for rank, (ch, imp) in enumerate(sorted_channels[:20], 1):
    mi_vals = channel_importance_accum.get(ch, [])
    mi_count = len(mi_vals)
    mi_mean = np.mean(mi_vals) if mi_vals else 0.0
    print(f"{rank:<5} {ch:<30} {imp:<20.6f} {mi_count:<10} {mi_mean:<10.4f}")

# Method 2: MI-based ranking (mean MI across positions where channel was discriminative)
print("\n\nChannel ranking by mean mutual information (across positions):")
channel_mi_mean = {ch: np.mean(vals) for ch, vals in channel_importance_accum.items() if vals}
sorted_mi_channels = sorted(channel_mi_mean.items(), key=lambda x: -x[1])
print(f"\n{'Rank':<5} {'Channel':<30} {'Mean MI':<12} {'# positions':<12}")
print("-" * 60)
for rank, (ch, mi) in enumerate(sorted_mi_channels[:20], 1):
    n_pos = len(channel_importance_accum[ch])
    print(f"{rank:<5} {ch:<30} {mi:<12.4f} {n_pos:<12}")

# ── 7. Aggregate: Top statistics ─────────────────────────────────────────
print("\n" + "="*70)
print("TOP 5 MOST DISCRIMINATIVE STATISTICS")
print("="*70)

stat_rf_importance = defaultdict(float)
for stat_idx, stat_name in enumerate(stat_names):
    for ch_idx in range(C):
        feat_idx = stat_idx * C + ch_idx
        stat_rf_importance[stat_name] += global_importances[feat_idx]

sorted_stats = sorted(stat_rf_importance.items(), key=lambda x: -x[1])
print(f"\n{'Rank':<5} {'Statistic':<15} {'RF Importance (sum)':<20} {'MI mean':<12} {'MI count':<10}")
print("-" * 62)
for rank, (stat, imp) in enumerate(sorted_stats, 1):
    mi_vals = stat_importance_accum.get(stat, [])
    mi_mean = np.mean(mi_vals) if mi_vals else 0.0
    print(f"{rank:<5} {stat:<15} {imp:<20.6f} {mi_mean:<12.4f} {len(mi_vals):<10}")

# ── 8. Accuracy analysis ─────────────────────────────────────────────────
print("\n" + "="*70)
print("ACCURACY ANALYSIS ON AMBIGUOUS POSITIONS")
print("="*70)

if per_position_results:
    majority_accs = [r["majority_acc"] for r in per_position_results]
    global_accs = [r["global_acc"] for r in per_position_results if not np.isnan(r["global_acc"])]
    temporal_accs = [r["temporal_acc"] for r in per_position_results if not np.isnan(r["temporal_acc"])]
    combined_accs = [r["combined_acc"] for r in per_position_results if not np.isnan(r["combined_acc"])]

    print(f"\nPer-position cross-validated accuracy (mean over {len(per_position_results)} positions):")
    print(f"  Majority vote baseline:  {np.mean(majority_accs):.3f} +/- {np.std(majority_accs):.3f}")
    if global_accs:
        print(f"  Global stats RF:         {np.mean(global_accs):.3f} +/- {np.std(global_accs):.3f}")
    if temporal_accs:
        print(f"  Temporal stats RF:       {np.mean(temporal_accs):.3f} +/- {np.std(temporal_accs):.3f}")
    if combined_accs:
        print(f"  Combined RF:             {np.mean(combined_accs):.3f} +/- {np.std(combined_accs):.3f}")

    # Theoretical ceiling: if we could perfectly classify each position
    # The ceiling is 1.0 (since each position has distinct g-code texts)
    print(f"\n  Theoretical accuracy ceiling (perfect features): 1.000")

    # Do temporal features help?
    if global_accs and temporal_accs:
        temp_better = sum(1 for r in per_position_results
                         if not np.isnan(r["temporal_acc"]) and not np.isnan(r["global_acc"])
                         and r["temporal_acc"] > r["global_acc"])
        glob_better = sum(1 for r in per_position_results
                         if not np.isnan(r["temporal_acc"]) and not np.isnan(r["global_acc"])
                         and r["global_acc"] > r["temporal_acc"])
        tied = sum(1 for r in per_position_results
                   if not np.isnan(r["temporal_acc"]) and not np.isnan(r["global_acc"])
                   and abs(r["temporal_acc"] - r["global_acc"]) < 0.001)

        print(f"\n  Temporal vs Global comparison:")
        print(f"    Temporal better: {temp_better} positions")
        print(f"    Global better:   {glob_better} positions")
        print(f"    Tied:            {tied} positions")

    if combined_accs and global_accs:
        comb_better = sum(1 for r in per_position_results
                         if not np.isnan(r["combined_acc"]) and not np.isnan(r["global_acc"])
                         and r["combined_acc"] > r["global_acc"] + 0.01)
        print(f"    Combined beats Global by >1%: {comb_better} positions")

# ── 9. Quick pooled RF accuracy with best features ───────────────────────
print("\n" + "="*70)
print("POOLED RF ACCURACY (best features, all ambiguous samples)")
print("="*70)

# Use top-K features by importance
for top_k in [10, 20, 50, 100]:
    top_feat_idx = np.argsort(combined_importances)[-top_k:]
    X_top = X_combined_all[:, top_feat_idx]

    # Need at least 2 per class for stratified CV
    class_counts_all = Counter(y_all)
    valid_classes = {c for c, cnt in class_counts_all.items() if cnt >= 2}
    mask = np.array([y in valid_classes for y in y_all])

    if mask.sum() < 6:
        print(f"  Top-{top_k}: Not enough samples with >=2 per class")
        continue

    X_cv = X_top[mask]
    y_cv = y_all[mask]
    # Re-encode labels to be contiguous
    le_cv = LabelEncoder()
    y_cv = le_cv.fit_transform(y_cv)

    n_cv = min(3, min(Counter(y_cv).values()))
    if n_cv < 2:
        print(f"  Top-{top_k}: Min class count too small for CV")
        continue

    try:
        cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        scores = cross_val_score(rf, X_cv, y_cv, cv=cv, scoring="accuracy")
        print(f"  Top-{top_k} features: CV accuracy = {scores.mean():.3f} +/- {scores.std():.3f}  (on {mask.sum()} samples, {len(le_cv.classes_)} classes)")
    except Exception as e:
        print(f"  Top-{top_k}: Error: {e}")

# ── 10. Temporal vs Global: detailed feature type breakdown ──────────────
print("\n" + "="*70)
print("TEMPORAL vs GLOBAL FEATURE IMPORTANCE BREAKDOWN")
print("="*70)

n_global = len(global_feat_names)
n_temporal = len(temporal_feat_names)
global_imp_sum = combined_importances[:n_global].sum()
temporal_imp_sum = combined_importances[n_global:].sum()
total_imp = global_imp_sum + temporal_imp_sum

print(f"  Global features ({n_global}):   total importance = {global_imp_sum:.4f} ({100*global_imp_sum/total_imp:.1f}%)")
print(f"  Temporal features ({n_temporal}): total importance = {temporal_imp_sum:.4f} ({100*temporal_imp_sum/total_imp:.1f}%)")

# Top 10 combined features by importance
top_combined_idx = np.argsort(combined_importances)[-10:][::-1]
print(f"\nTop 10 features overall (combined set):")
print(f"{'Rank':<5} {'Feature':<45} {'Importance':<12} {'Type':<10}")
print("-" * 72)
for rank, idx in enumerate(top_combined_idx, 1):
    fname = combined_feat_names[idx]
    imp = combined_importances[idx]
    ftype = "global" if idx < n_global else "temporal"
    print(f"{rank:<5} {fname:<45} {imp:<12.6f} {ftype:<10}")

# ── 11. Sensor group analysis ────────────────────────────────────────────
print("\n" + "="*70)
print("SENSOR GROUP ANALYSIS")
print("="*70)

# Group channels by sensor location
sensor_groups = defaultdict(list)
for ch_name in col_names:
    if "." in ch_name:
        sensor = ch_name.split(".")[0]
    else:
        sensor = ch_name
    sensor_groups[sensor].append(ch_name)

print(f"\nSensor groups and their total RF importance:")
group_imp = {}
for sensor, channels in sorted(sensor_groups.items()):
    total = sum(channel_rf_importance[ch] for ch in channels)
    group_imp[sensor] = total

sorted_groups = sorted(group_imp.items(), key=lambda x: -x[1])
print(f"\n{'Rank':<5} {'Sensor Group':<20} {'# Channels':<12} {'Total RF Imp':<15} {'% of total':<10}")
print("-" * 62)
total_all = sum(group_imp.values())
for rank, (sensor, imp) in enumerate(sorted_groups, 1):
    n_ch = len(sensor_groups[sensor])
    pct = 100 * imp / total_all if total_all > 0 else 0
    print(f"{rank:<5} {sensor:<20} {n_ch:<12} {imp:<15.6f} {pct:<10.1f}%")

# ── 12. Per-position detail table ────────────────────────────────────────
print("\n" + "="*70)
print("PER-POSITION DETAIL (sorted by combined accuracy)")
print("="*70)

if per_position_results:
    sorted_results = sorted(per_position_results, key=lambda r: r.get("combined_acc", 0) or 0, reverse=True)
    print(f"\n{'Pos (win,op)':<18} {'N':<5} {'K':<4} {'Majority':<10} {'Global':<10} {'Temporal':<10} {'Combined':<10}")
    print("-" * 67)
    for r in sorted_results:
        print(f"{str(r['pos']):<18} {r['n_samples']:<5} {r['n_classes']:<4} "
              f"{r['majority_acc']:<10.3f} "
              f"{r['global_acc']:<10.3f} "
              f"{r['temporal_acc']:<10.3f} "
              f"{r['combined_acc']:<10.3f}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  Ambiguous positions: {len(ambiguous_positions)}")
print(f"  Total ambiguous samples: {total_ambig_samples} / {N} ({100*total_ambig_samples/N:.1f}%)")
print(f"  Positions analyzed (with CV): {len(per_position_results)}")
if per_position_results:
    if global_accs:
        print(f"  Mean global-stats RF accuracy: {np.mean(global_accs):.3f}")
    if temporal_accs:
        print(f"  Mean temporal-stats RF accuracy: {np.mean(temporal_accs):.3f}")
    if combined_accs:
        print(f"  Mean combined RF accuracy: {np.mean(combined_accs):.3f}")
    if majority_accs:
        print(f"  Mean majority baseline: {np.mean(majority_accs):.3f}")
    print(f"  Temporal features help? {'Yes' if temporal_imp_sum > global_imp_sum * 0.5 else 'Marginal'} "
          f"(temporal/global importance ratio: {temporal_imp_sum/global_imp_sum:.2f})")
print(f"\n  Top 5 channels: {', '.join(ch for ch, _ in sorted_channels[:5])}")
print(f"  Top stat: {sorted_stats[0][0]}")
print()
