#!/usr/bin/env python3
"""
Sensor subset analysis for the 6-class (non-damage) problem.

Loads NPZ data, filters out the 3 damage classes, determines which sensors
are active per window (variance > 1e-6), and finds optimal sensor subsets
at each cardinality N=4..16 that cover all 6 remaining classes with maximum
window retention.
"""

import json
import numpy as np
from itertools import combinations
from collections import defaultdict
from pathlib import Path

# ─────────────────────── Configuration ───────────────────────
DATA_DIR = Path("/home/seacuello/Documents/gcode_fingerprinting/"
                "outputs/7class_cascade_to_9class/9class_moddropout_final/data")
VARIANCE_THRESHOLD = 1e-6

CLASS_NAMES = [
    "adaptive", "adaptive150025", "damageadaptive", "damageface",
    "damagepocket", "face", "face150025", "pocket", "pocket150025"
]
DAMAGE_CLASSES = {"damageadaptive", "damageface", "damagepocket"}
KEEP_CLASSES = [c for c in CLASS_NAMES if c not in DAMAGE_CLASSES]
KEEP_INDICES = [i for i, c in enumerate(CLASS_NAMES) if c not in DAMAGE_CLASSES]
IDX_TO_NAME = {i: CLASS_NAMES[i] for i in KEEP_INDICES}

# ─────────────────────── Load metadata ───────────────────────
with open(DATA_DIR / "metadata.json") as f:
    meta = json.load(f)
columns = meta["continuous_columns"]

sensor_names = sorted({c.split(".")[0] for c in columns if "." in c})
sensor_col_indices = {}
for s in sensor_names:
    sensor_col_indices[s] = [i for i, c in enumerate(columns) if c.startswith(s + ".")]

print(f"Total features: {len(columns)}")
print(f"Sensors ({len(sensor_names)}): {sensor_names}")
print(f"Channels per sensor: {len(sensor_col_indices[sensor_names[0]])}")
print(f"\nKept classes (6): {KEEP_CLASSES}")
print(f"Kept class indices: {KEEP_INDICES}")
print()

# ─────────────────────── Load all splits ───────────────────────
all_X = []
all_y = []
all_split = []

for split_name in ["train", "val", "test"]:
    d = np.load(DATA_DIR / f"{split_name}_sequences.npz")
    X = d["continuous"]
    y = d["operation_type"]
    all_X.append(X)
    all_y.append(y)
    all_split.append(np.full(len(y), split_name, dtype=object))

X_all = np.concatenate(all_X, axis=0)
y_all = np.concatenate(all_y, axis=0)
split_all = np.concatenate(all_split, axis=0)

print(f"Total windows loaded: {len(y_all)}")

# ─────────────────────── Filter to 6-class subset ───────────────────────
keep_mask = np.isin(y_all, KEEP_INDICES)
X_6 = X_all[keep_mask]
y_6 = y_all[keep_mask]
split_6 = split_all[keep_mask]

print(f"Windows after removing damage classes: {len(y_6)}")
print(f"  Per class:")
for idx in KEEP_INDICES:
    cnt = np.sum(y_6 == idx)
    print(f"    {IDX_TO_NAME[idx]:20s} (idx {idx}): {cnt:5d}")
print()

# ─────────────────────── Compute per-window sensor presence ───────────────────────
n_windows = len(y_6)
n_sensors = len(sensor_names)

sensor_present = np.zeros((n_windows, n_sensors), dtype=bool)

for j, sname in enumerate(sensor_names):
    col_idx = sensor_col_indices[sname]
    chunk = X_6[:, :, col_idx]
    var = np.var(chunk, axis=1)
    sensor_present[:, j] = np.any(var > VARIANCE_THRESHOLD, axis=1)

sensors_per_window = sensor_present.sum(axis=1)
print("Sensors-per-window distribution (6-class subset):")
for n in range(0, n_sensors + 1):
    cnt = np.sum(sensors_per_window == n)
    if cnt > 0:
        print(f"  {n:2d} sensors: {cnt:5d} windows")

print()

print("Per-sensor presence across all 6-class windows:")
for j, sname in enumerate(sensor_names):
    cnt = sensor_present[:, j].sum()
    print(f"  {sname:15s}: {cnt:5d} / {n_windows} ({100*cnt/n_windows:5.1f}%)")
print()

print("Per-class sensor presence rate:")
header = f"{'sensor':>15s} | " + " | ".join(f"{IDX_TO_NAME[idx]:>14s}" for idx in KEEP_INDICES) + " |"
print(header)
print("-" * len(header))
for j, sname in enumerate(sensor_names):
    row = f"{sname:>15s} | "
    for idx in KEEP_INDICES:
        class_mask = y_6 == idx
        cnt = sensor_present[class_mask, j].sum()
        tot = class_mask.sum()
        row += f"{100*cnt/tot:13.1f}% | "
    print(row)
print()

# ─────────────────────── Find optimal sensor subsets ───────────────────────
print("=" * 100)
print("OPTIMAL SENSOR SUBSET SEARCH")
print("=" * 100)
print()
print("For each N, finding sensor subset of size N that maximizes surviving")
print("windows while covering all 6 classes.")
print()

def evaluate_subset(sensor_indices):
    mask = np.all(sensor_present[:, list(sensor_indices)], axis=1)
    surviving_labels = y_6[mask]
    surviving_splits = split_6[mask]
    n_surviving = mask.sum()
    classes_covered = set(np.unique(surviving_labels))
    all_6_covered = classes_covered == set(KEEP_INDICES)

    per_class = {}
    for idx in KEEP_INDICES:
        per_class[IDX_TO_NAME[idx]] = int(np.sum(surviving_labels == idx))

    per_split = {}
    for sp in ["train", "val", "test"]:
        per_split[sp] = int(np.sum(surviving_splits == sp))

    per_split_class = {}
    for sp in ["train", "val", "test"]:
        sp_mask = surviving_splits == sp
        sp_labels = surviving_labels[sp_mask]
        per_split_class[sp] = {}
        for idx in KEEP_INDICES:
            per_split_class[sp][IDX_TO_NAME[idx]] = int(np.sum(sp_labels == idx))

    return {
        "n_surviving": n_surviving,
        "pct": 100 * n_surviving / n_windows,
        "all_6_covered": all_6_covered,
        "classes_covered": len(classes_covered),
        "per_class": per_class,
        "per_split": per_split,
        "per_split_class": per_split_class,
    }


results = {}

for N in range(4, n_sensors + 1):
    n_combos = 1
    for i in range(N):
        n_combos = n_combos * (n_sensors - i) // (i + 1)

    best = None
    best_info = None

    for combo in combinations(range(n_sensors), N):
        info = evaluate_subset(combo)
        if best is None:
            best = combo
            best_info = info
        else:
            if info["all_6_covered"] and not best_info["all_6_covered"]:
                best = combo
                best_info = info
            elif info["all_6_covered"] == best_info["all_6_covered"]:
                if info["classes_covered"] > best_info["classes_covered"]:
                    best = combo
                    best_info = info
                elif info["classes_covered"] == best_info["classes_covered"]:
                    if info["n_surviving"] > best_info["n_surviving"]:
                        best = combo
                        best_info = info

    sensor_set = [sensor_names[i] for i in best]
    results[N] = {"sensors": sensor_set, "info": best_info, "n_combos": n_combos}

    covered_str = "YES" if best_info["all_6_covered"] else f"NO ({best_info['classes_covered']}/6)"
    print(f"N={N:2d} | combos={n_combos:6d} | best: {best_info['n_surviving']:5d} windows "
          f"({best_info['pct']:5.1f}%) | all 6 covered: {covered_str}")

# ─────────────────────── Clean summary table ───────────────────────
print()
print("=" * 150)
print("SUMMARY TABLE")
print("=" * 150)
fmt = "{:>3s} | {:>65s} | {:>7s} | {:>6s} | {:>10s} | {:>7s} {:>7s} {:>7s}"
print(fmt.format("N", "Best Sensor Set", "Windows", "% Cov", "All 6 cls?",
                  "Train", "Val", "Test"))
print("-" * 150)
for N in range(4, n_sensors + 1):
    r = results[N]
    info = r["info"]
    sset = ", ".join(r["sensors"])
    if len(sset) > 65:
        sset = sset[:62] + "..."
    covered = "YES" if info["all_6_covered"] else f"NO({info['classes_covered']}/6)"
    print(fmt.format(
        str(N), sset,
        str(info["n_surviving"]),
        f"{info['pct']:.1f}%",
        covered,
        str(info["per_split"]["train"]),
        str(info["per_split"]["val"]),
        str(info["per_split"]["test"]),
    ))

# ─────────────────────── Detailed breakdown ───────────────────────
print()
print("=" * 140)
print("DETAILED PER-CLASS/PER-SPLIT BREAKDOWN FOR KEY N VALUES")
print("=" * 140)

first_all6 = None
for N in range(4, n_sensors + 1):
    if results[N]["info"]["all_6_covered"]:
        first_all6 = N
        break

detail_Ns = sorted(set([4, 5, 6, 7, 8, 9, 10] + ([first_all6] if first_all6 else []) + [n_sensors]))
detail_Ns = [n for n in detail_Ns if 4 <= n <= n_sensors]

for N in detail_Ns:
    r = results[N]
    info = r["info"]
    print(f"\n--- N = {N} sensors ---")
    print(f"Sensor set: {r['sensors']}")
    print(f"Total surviving windows: {info['n_surviving']} / {n_windows} ({info['pct']:.1f}%)")
    print(f"All 6 classes covered: {'YES' if info['all_6_covered'] else 'NO'}")
    print()

    hdr = f"  {'Class':>20s} | {'Train':>7s} | {'Val':>7s} | {'Test':>7s} | {'Total':>7s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for idx in KEEP_INDICES:
        cname = IDX_TO_NAME[idx]
        tr = info["per_split_class"]["train"][cname]
        va = info["per_split_class"]["val"][cname]
        te = info["per_split_class"]["test"][cname]
        tot = info["per_class"][cname]
        print(f"  {cname:>20s} | {tr:7d} | {va:7d} | {te:7d} | {tot:7d}")
    tr_tot = info["per_split"]["train"]
    va_tot = info["per_split"]["val"]
    te_tot = info["per_split"]["test"]
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL':>20s} | {tr_tot:7d} | {va_tot:7d} | {te_tot:7d} | {info['n_surviving']:7d}")

# ─────────────────────── Key question ───────────────────────
print()
print("=" * 100)
print("KEY QUESTION: Can we get 7+ sensors while keeping all 6 classes?")
print("=" * 100)
for N in range(7, n_sensors + 1):
    r = results[N]
    info = r["info"]
    if info["all_6_covered"]:
        print(f"  N={N:2d}: YES -- {info['n_surviving']} windows ({info['pct']:.1f}%), sensors: {r['sensors']}")
    else:
        print(f"  N={N:2d}: NO  -- only {info['classes_covered']}/6 classes covered")
