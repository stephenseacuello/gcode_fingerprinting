#!/usr/bin/env python3
"""
Sensor Presence Analysis
========================
Determine which of the 16 sensors were physically present in each window
of the preprocessed NPZ data. A sensor is "absent" (zero-padded) if ALL 17
of its channels have exactly zero variance across the 64 timesteps in a window.
"""

import json
import numpy as np
from pathlib import Path
from collections import OrderedDict

DATA_DIR = Path("/home/seacuello/Documents/gcode_fingerprinting/outputs/"
                "7class_cascade_to_9class/9class_moddropout_final/data")

# ── 1. Load metadata ────────────────────────────────────────────────────
with open(DATA_DIR / "metadata.json") as f:
    meta = json.load(f)

columns = meta["continuous_columns"]
n_cols = len(columns)

# Identify sensor columns (contain a dot)
sensor_col_indices = {i: c for i, c in enumerate(columns) if "." in c}
sensors_ordered = list(OrderedDict.fromkeys(
    c.split(".")[0] for c in columns if "." in c
))
channels = sorted(set(c.split(".")[1] for c in columns if "." in c))

# Build mapping: sensor_name -> list of column indices
sensor_to_col_idx = {}
for sensor in sensors_ordered:
    indices = [i for i, c in enumerate(columns) if c.startswith(sensor + ".")]
    sensor_to_col_idx[sensor] = indices

print("=" * 72)
print("SENSOR PRESENCE ANALYSIS")
print("=" * 72)
print(f"\nData directory: {DATA_DIR}")
print(f"Window size: {meta['window_size']}, Stride: {meta['stride']}")
print(f"Total continuous features: {n_cols}")
print(f"  Non-sensor features: {n_cols - len(sensor_col_indices)}")
print(f"  Sensor features:     {len(sensor_col_indices)}  "
      f"(16 sensors x 17 channels = 272)")

print(f"\n16 Sensor locations:")
for i, s in enumerate(sensors_ordered, 1):
    print(f"  {i:2d}. {s}")

print(f"\n17 Channels per sensor:")
print(f"  {', '.join(channels)}")

# ── 2. Load NPZ files and compute per-window sensor presence ────────────
splits = ["train", "val", "test"]
split_files = {s: meta[f"{s}_files"] for s in splits}

# Results storage
# For each split: boolean array of shape (n_windows, 16)
# True = sensor present (non-zero variance in at least one channel)
split_presence = {}
split_n_windows = {}

for split in splits:
    npz_path = DATA_DIR / f"{split}_sequences.npz"
    npz = np.load(npz_path)
    X = npz["continuous"]  # (n_windows, 64, 296)
    n_windows = X.shape[0]
    split_n_windows[split] = n_windows

    presence = np.zeros((n_windows, len(sensors_ordered)), dtype=bool)

    for s_idx, sensor in enumerate(sensors_ordered):
        col_indices = sensor_to_col_idx[sensor]
        # Extract all channels for this sensor: (n_windows, 64, 17)
        sensor_data = X[:, :, col_indices]
        # Variance across timestep axis (axis=1): (n_windows, 17)
        var_per_channel = np.var(sensor_data, axis=1)
        # Sensor is present if ANY channel has non-zero variance
        # Use a small epsilon to handle floating-point noise
        presence[:, s_idx] = np.any(var_per_channel > 1e-12, axis=1)

    split_presence[split] = presence
    print(f"\nLoaded {split}: {n_windows} windows from {len(split_files[split])} files")

total_windows = sum(split_n_windows.values())
print(f"\nTotal windows across all splits: {total_windows}")

# ── 3. Per-sensor presence counts by split ──────────────────────────────
print("\n" + "=" * 72)
print("PER-SENSOR PRESENCE (windows with non-zero variance)")
print("=" * 72)

header = f"{'Sensor':<12} | {'Train':>14} | {'Val':>14} | {'Test':>14} | {'Total':>14}"
print(header)
print("-" * len(header))

common_sensors = []
mixed_sensors = []

for s_idx, sensor in enumerate(sensors_ordered):
    counts = {}
    for split in splits:
        present = split_presence[split][:, s_idx].sum()
        total = split_n_windows[split]
        counts[split] = (present, total)

    total_present = sum(c[0] for c in counts.values())
    total_all = sum(c[1] for c in counts.values())

    parts = []
    for split in splits:
        p, t = counts[split]
        pct = 100.0 * p / t if t > 0 else 0
        parts.append(f"{p:>5}/{t:<5} {pct:>4.0f}%")

    total_pct = 100.0 * total_present / total_all
    total_part = f"{total_present:>5}/{total_all:<5} {total_pct:>4.0f}%"

    print(f"{sensor:<12} | {parts[0]} | {parts[1]} | {parts[2]} | {total_part}")

    if total_present == total_all:
        common_sensors.append(sensor)
    else:
        mixed_sensors.append((sensor, total_present, total_all))

# ── 4. Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

print(f"\nSensors present in ALL {total_windows} windows ({len(common_sensors)}):")
for s in common_sensors:
    print(f"  - {s}")

n_common_sensor_features = len(common_sensors) * 17
n_non_sensor = n_cols - len(sensor_col_indices)
n_common_total = n_non_sensor + n_common_sensor_features

print(f"\nCommon-set feature count:")
print(f"  Non-sensor features:    {n_non_sensor:>4}")
print(f"  Common sensor features: {n_common_sensor_features:>4}  "
      f"({len(common_sensors)} sensors x 17 channels)")
print(f"  Total common features:  {n_common_total:>4}")

print(f"\nSensors with MIXED presence ({len(mixed_sensors)}):")
for sensor, present, total in mixed_sensors:
    absent = total - present
    print(f"  - {sensor}: present in {present}/{total} windows "
          f"({100*present/total:.1f}%), absent in {absent}")

# ── 5. Unique sensor-presence patterns ──────────────────────────────────
print("\n" + "=" * 72)
print("UNIQUE SENSOR-PRESENCE PATTERNS")
print("=" * 72)

# Concatenate all splits
all_presence = np.vstack([split_presence[s] for s in splits])
# Convert each row to a tuple for hashing
patterns = {}
for i in range(all_presence.shape[0]):
    key = tuple(all_presence[i])
    if key not in patterns:
        patterns[key] = 0
    patterns[key] += 1

print(f"\nFound {len(patterns)} unique sensor-deployment patterns:\n")

# Sort by frequency descending
for rank, (pattern, count) in enumerate(
        sorted(patterns.items(), key=lambda x: -x[1]), 1):
    present_sensors = [sensors_ordered[i] for i, v in enumerate(pattern) if v]
    absent_sensors = [sensors_ordered[i] for i, v in enumerate(pattern) if not v]
    n_present = sum(pattern)
    pct = 100.0 * count / total_windows
    print(f"  Pattern {rank}: {count:>5} windows ({pct:>5.1f}%) — "
          f"{n_present}/16 sensors present")
    if absent_sensors:
        print(f"    Absent: {', '.join(absent_sensors)}")
    else:
        print(f"    All 16 sensors present")

# ── 6. Absent-sensor breakdown per split ────────────────────────────────
print("\n" + "=" * 72)
print("ABSENT-SENSOR BREAKDOWN BY SPLIT")
print("=" * 72)
for split in splits:
    pres = split_presence[split]
    n = split_n_windows[split]
    n_absent_per_window = (~pres).sum(axis=1)
    print(f"\n{split.upper()} ({n} windows):")
    print(f"  Windows with all 16 sensors: {(n_absent_per_window == 0).sum()}")
    print(f"  Windows missing any sensor:  {(n_absent_per_window > 0).sum()}")
    # How many distinct sensors are ever absent
    ever_absent = [sensors_ordered[i] for i in range(len(sensors_ordered))
                   if (~pres[:, i]).any()]
    if ever_absent:
        print(f"  Sensors ever absent: {', '.join(ever_absent)}")

print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
