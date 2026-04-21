"""Physical interpretation analysis.

A. Map top features to sensor physics and cutting mechanics
B. Map temporal bins to G-code phases for each toolpath type
C. Compare temporal fingerprints against expected cutting force profiles
D. Analyze WHY specific sensors/channels dominate
"""

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


# ═══════════════════════════════════════════════════════════════════
# A1: Map temporal bins to G-code phases
# ═══════════════════════════════════════════════════════════════════

def analyze_gcode_phases():
    """Map temporal bins to actual G-code command phases for each toolpath type."""
    print("=" * 70)
    print("A1: TEMPORAL BIN → G-CODE PHASE MAPPING")
    print("=" * 70)

    results = {}

    for toolpath in cfg.TOOLPATH_TYPES:
        prefix = cfg.ACTIVE_PREFIX[toolpath]
        files = utils.discover_runs(prefix)
        # Use first run as representative
        fpath = files[0]
        df = pd.read_csv(fpath)

        n_rows = len(df)
        t_norm = np.arange(n_rows) / max(n_rows - 1, 1)
        bin_ids = np.clip((t_norm * cfg.N_BINS).astype(int), 0, cfg.N_BINS - 1)

        # Extract G-code command type per row
        def extract_cmd(s):
            if pd.isna(s): return "NONE"
            s = str(s).strip()
            for cmd in ["G53", "G3", "G2", "G1", "G0", "M30", "M5", "M9", "M3"]:
                if s.startswith(cmd) or f" {cmd}" in s:
                    return cmd
            if any(c in s for c in ["X", "Y", "Z"]):
                return "MOVE"
            return "OTHER"

        df["cmd_type"] = df["gcode_string"].apply(extract_cmd)
        df["bin_id"] = bin_ids

        # Classify phases
        def classify_phase(cmd, gcode_line, max_line):
            if cmd == "G0":
                return "rapid"
            elif cmd in ("G1", "MOVE"):
                return "cutting"
            elif cmd in ("G2", "G3"):
                return "arc_cutting"
            elif cmd in ("G53", "M30", "M5"):
                return "retract/end"
            else:
                return "other"

        max_line = df["gcode_line"].max()
        df["phase"] = df.apply(lambda r: classify_phase(r["cmd_type"], r["gcode_line"], max_line), axis=1)

        toolpath_result = {"bins": {}}
        for b in range(cfg.N_BINS):
            bin_data = df[df["bin_id"] == b]
            if len(bin_data) == 0:
                continue

            phase_counts = bin_data["phase"].value_counts().to_dict()
            cmd_counts = bin_data["cmd_type"].value_counts().to_dict()
            gcode_range = (int(bin_data["gcode_line"].min()), int(bin_data["gcode_line"].max()))

            # Dominant phase
            dominant_phase = max(phase_counts, key=phase_counts.get)

            # Representative gcode strings
            sample_gcodes = bin_data["gcode_string"].head(3).tolist()

            toolpath_result["bins"][b] = {
                "dominant_phase": dominant_phase,
                "phase_distribution": phase_counts,
                "cmd_distribution": cmd_counts,
                "gcode_line_range": gcode_range,
                "n_rows": len(bin_data),
                "sample_gcodes": sample_gcodes,
            }

        results[toolpath] = toolpath_result

        # Print summary
        print(f"\n  {toolpath} ({n_rows} rows, {len(files)} runs):")
        print(f"  {'Bin':>3} {'Phase':<15} {'G-code lines':>15} {'Rows':>5} {'Sample G-code'}")
        for b in range(cfg.N_BINS):
            if b not in toolpath_result["bins"]:
                continue
            info = toolpath_result["bins"][b]
            sample = info["sample_gcodes"][0][:40] if info["sample_gcodes"] else ""
            print(f"  {b+1:3d} {info['dominant_phase']:<15} "
                  f"{info['gcode_line_range'][0]:>5}-{info['gcode_line_range'][1]:<5} "
                  f"{info['n_rows']:5d}  {sample}")

    return results


# ═══════════════════════════════════════════════════════════════════
# A2: Analyze WHY specific sensors/channels dominate
# ═══════════════════════════════════════════════════════════════════

def analyze_sensor_physics():
    """Analyze what the top-ranked features are physically measuring."""
    print("\n" + "=" * 70)
    print("A2: SENSOR PHYSICS ANALYSIS")
    print("=" * 70)

    imp_df = pd.read_csv(cfg.IMPORTANCE_DIR / "raw_rf_importance.csv")
    top50 = imp_df.head(50)

    # Parse feature names into components
    def parse_feature(name):
        # e.g., binmean_frame_r2.Mx_min
        parts = name.split("_", 1)  # binmean, rest
        agg_type = parts[0]  # binmean or binstd
        rest = parts[1]

        # Find the sensor and channel
        for mod in cfg.ARDUINO_MODULES:
            if rest.startswith(mod + "."):
                after_mod = rest[len(mod) + 1:]
                for ch in cfg.ARDUINO_CHANNELS:
                    if after_mod.startswith(ch + "_"):
                        stat = after_mod[len(ch) + 1:]
                        return agg_type, mod, ch, stat
                    elif after_mod == ch:
                        return agg_type, mod, ch, "direct"
                break

        for ec in cfg.ELECTRICAL_COLS:
            if rest.startswith(ec + "_"):
                stat = rest[len(ec) + 1:]
                return agg_type, "electrical", ec, stat
            elif rest == ec:
                return agg_type, "electrical", ec, "direct"

        return agg_type, "unknown", rest, "unknown"

    records = []
    for _, row in top50.iterrows():
        agg, sensor, channel, stat = parse_feature(row["feature"])
        records.append({
            "rank": row["rank"],
            "feature": row["feature"],
            "importance": row["importance"],
            "aggregation": agg,
            "sensor_module": sensor,
            "channel": channel,
            "statistic": stat,
        })

    parsed_df = pd.DataFrame(records)

    # Channel type distribution in top 50
    channel_types = {
        "Magnetometer": ["Mx", "My", "Mz"],
        "Accelerometer": ["Ax", "Ay", "Az"],
        "Gyroscope": ["Gx", "Gy", "Gz"],
        "Color": ["ColorR", "ColorG", "ColorB", "ColorA"],
        "Environmental": ["Pressure", "Temperature", "Proximity"],
        "Vibration RMS": ["RMS"],
        "Motor Current": cfg.ELECTRICAL_COLS,
    }

    print("\n  Channel type distribution in top 50 features:")
    type_importance = {}
    for ctype, channels in channel_types.items():
        mask = parsed_df["channel"].isin(channels)
        count = mask.sum()
        total_imp = parsed_df[mask]["importance"].sum()
        type_importance[ctype] = {"count": int(count), "total_importance": float(total_imp)}
        if count > 0:
            print(f"    {ctype:<20}: {count:2d} features, total importance = {total_imp:.4f}")

    # Sensor module distribution in top 50
    print("\n  Sensor module distribution in top 50:")
    mod_importance = {}
    for mod in cfg.ARDUINO_MODULES + ["electrical"]:
        mask = parsed_df["sensor_module"] == mod
        count = mask.sum()
        total_imp = parsed_df[mask]["importance"].sum()
        mod_importance[mod] = {"count": int(count), "total_importance": float(total_imp)}
        if count > 0:
            print(f"    {mod:<15}: {count:2d} features, total importance = {total_imp:.4f}")

    # Statistic distribution
    print("\n  Statistic type distribution in top 50:")
    for stat in cfg.BLOCK_STATS:
        mask = parsed_df["statistic"] == stat
        count = mask.sum()
        if count > 0:
            total_imp = parsed_df[mask]["importance"].sum()
            print(f"    {stat:<12}: {count:2d} features, total importance = {total_imp:.4f}")

    return {
        "channel_types": type_importance,
        "sensor_modules": mod_importance,
        "top50_parsed": records,
    }


# ═══════════════════════════════════════════════════════════════════
# B: Cross-reference with cutting mechanics
# ═══════════════════════════════════════════════════════════════════

def analyze_cutting_mechanics():
    """Compare sensor signatures against expected cutting force profiles."""
    print("\n" + "=" * 70)
    print("B: CUTTING MECHANICS CROSS-REFERENCE")
    print("=" * 70)

    # For each toolpath type, analyze key sensor channels across temporal bins
    # and compare against expected mechanical behavior

    key_channels = [
        ("frame_r2.Mx", "Magnetometer X (frame right) — sensitive to motor current fields"),
        ("y_bed__3.Az", "Accelerometer Z (bed) — vertical force/vibration"),
        ("y_bed__3.Mx", "Magnetometer X (bed) — Y-axis motor proximity"),
        ("spindle2.RMS", "Vibration RMS (spindle) — overall vibration energy"),
        ("spindle", "Spindle current draw — proportional to cutting torque"),
        ("frame_l2.Gy", "Gyroscope Y (frame left) — rotational vibration"),
    ]

    results = {}

    for ch_name, ch_description in key_channels:
        print(f"\n  {ch_name}: {ch_description}")

        # Collect per-toolpath temporal profiles (mean across runs)
        for toolpath in cfg.TOOLPATH_TYPES:
            prefix = cfg.ACTIVE_PREFIX[toolpath]
            files = utils.discover_runs(prefix)

            all_bin_means = []
            for fpath in files:
                bin_path = cfg.ISOLATED_DIR / "raw" / f"{prefix}_{utils.extract_run_id(fpath)}_bins.parquet"
                if not bin_path.exists():
                    continue
                iso_df = utils.load_parquet(bin_path)
                col = f"{ch_name}_mean"
                if col in iso_df.columns:
                    all_bin_means.append(iso_df[col].values)

            if all_bin_means:
                all_bin_means = np.stack(all_bin_means)
                mean_profile = np.mean(all_bin_means, axis=0)
                std_profile = np.std(all_bin_means, axis=0)

                # Compute variability metrics
                cv = np.std(mean_profile) / (np.abs(np.mean(mean_profile)) + 1e-10)
                range_val = np.max(mean_profile) - np.min(mean_profile)

                results.setdefault(ch_name, {})[toolpath] = {
                    "mean_profile": mean_profile.tolist(),
                    "std_profile": std_profile.tolist(),
                    "temporal_cv": float(cv),
                    "range": float(range_val),
                    "overall_mean": float(np.mean(mean_profile)),
                }
                print(f"    {toolpath}: mean={np.mean(mean_profile):.4f}, "
                      f"temporal_CV={cv:.3f}, range={range_val:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# Generate physical interpretation figure
# ═══════════════════════════════════════════════════════════════════

def fig_physical_interpretation():
    """Multi-panel figure showing sensor physics across toolpath types."""
    print("\n" + "=" * 70)
    print("GENERATING PHYSICAL INTERPRETATION FIGURE")
    print("=" * 70)

    channels = [
        ("frame_r2.Mx_mean", "Magnetometer Mx\n(frame right)", "Magnetic field disturbance\nfrom motor currents"),
        ("y_bed__3.Az_mean", "Accelerometer Az\n(bed sensor)", "Vertical vibration\nat workpiece"),
        ("y_bed__3.Mx_mean", "Magnetometer Mx\n(bed sensor)", "Y-axis motor\nmagnetic field"),
        ("spindle_mean", "Spindle Current\n(electrical)", "Cutting torque\nproxy"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (col_name, sensor_label, physics_label) in enumerate(channels):
        ax = axes[idx // 2, idx % 2]

        for toolpath in cfg.TOOLPATH_TYPES:
            prefix = cfg.ACTIVE_PREFIX[toolpath]
            files = utils.discover_runs(prefix)

            all_vals = []
            for fpath in files:
                run_id = utils.extract_run_id(fpath)
                bin_path = cfg.ISOLATED_DIR / "raw" / f"{prefix}_{run_id}_bins.parquet"
                if not bin_path.exists():
                    continue
                iso_df = utils.load_parquet(bin_path)
                if col_name in iso_df.columns:
                    all_vals.append(iso_df[col_name].values)

            if all_vals:
                all_vals = np.stack(all_vals)
                mean_v = np.mean(all_vals, axis=0)
                std_v = np.std(all_vals, axis=0)
                x = range(1, cfg.N_BINS + 1)
                ax.plot(x, mean_v, color=CLASS_COLORS[toolpath], linewidth=2,
                       label=toolpath.capitalize())
                ax.fill_between(x, mean_v - std_v, mean_v + std_v,
                              color=CLASS_COLORS[toolpath], alpha=0.12)

        ax.set_xlabel("Temporal Bin")
        ax.set_ylabel(sensor_label, fontsize=9)
        ax.set_title(physics_label, fontsize=10, style="italic")
        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle("Physical Interpretation: Key Sensor Channels Across Toolpath Types",
                fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig13_physical_interpretation")

    # Also make a subtracted version
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (col_name, sensor_label, physics_label) in enumerate(channels):
        ax = axes2[idx // 2, idx % 2]

        for toolpath in cfg.TOOLPATH_TYPES:
            prefix = cfg.ACTIVE_PREFIX[toolpath]
            files = utils.discover_runs(prefix)

            all_vals = []
            for fpath in files:
                run_id = utils.extract_run_id(fpath)
                bin_path = cfg.ISOLATED_DIR / "subtracted" / f"{prefix}_{run_id}_bins.parquet"
                if not bin_path.exists():
                    continue
                iso_df = utils.load_parquet(bin_path)
                if col_name in iso_df.columns:
                    all_vals.append(iso_df[col_name].values)

            if all_vals:
                all_vals = np.stack(all_vals)
                mean_v = np.mean(all_vals, axis=0)
                std_v = np.std(all_vals, axis=0)
                x = range(1, cfg.N_BINS + 1)
                ax.plot(x, mean_v, color=CLASS_COLORS[toolpath], linewidth=2,
                       label=toolpath.capitalize())
                ax.fill_between(x, mean_v - std_v, mean_v + std_v,
                              color=CLASS_COLORS[toolpath], alpha=0.12)

        ax.set_xlabel("Temporal Bin")
        ax.set_ylabel(f"{sensor_label}\n(subtracted)", fontsize=9)
        ax.set_title(f"{physics_label}\n(air-cut referenced)", fontsize=10, style="italic")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        if idx == 0:
            ax.legend(fontsize=9)

    fig2.suptitle("Air-Cut Referenced Fingerprints: Isolated Process Signatures",
                 fontsize=13, y=1.02)
    fig2.tight_layout()
    save_fig(fig2, "fig14_physics_subtracted")


def fig_gcode_phase_annotation():
    """Annotated temporal fingerprint with G-code phase labels."""
    print("\nGenerating annotated G-code phase figure...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    col_name = "frame_r2.Mx_mean"  # Top importance feature

    phase_colors = {
        "rapid": "#ff7f0e",
        "cutting": "#2ca02c",
        "arc_cutting": "#1f77b4",
        "retract/end": "#d62728",
        "other": "#7f7f7f",
    }

    for row, toolpath in enumerate(cfg.TOOLPATH_TYPES):
        ax = axes[row]
        prefix = cfg.ACTIVE_PREFIX[toolpath]
        files = utils.discover_runs(prefix)

        # Plot sensor profiles
        all_vals = []
        for fpath in files:
            run_id = utils.extract_run_id(fpath)
            bin_path = cfg.ISOLATED_DIR / "raw" / f"{prefix}_{run_id}_bins.parquet"
            if not bin_path.exists():
                continue
            iso_df = utils.load_parquet(bin_path)
            if col_name in iso_df.columns:
                all_vals.append(iso_df[col_name].values)

        if all_vals:
            all_vals = np.stack(all_vals)
            mean_v = np.mean(all_vals, axis=0)
            std_v = np.std(all_vals, axis=0)
            x = np.arange(1, cfg.N_BINS + 1)
            ax.plot(x, mean_v, color=CLASS_COLORS[toolpath], linewidth=2.5)
            ax.fill_between(x, mean_v - std_v, mean_v + std_v,
                          color=CLASS_COLORS[toolpath], alpha=0.15)

        # Annotate with G-code phases from first run
        fpath = files[0]
        df_raw = pd.read_csv(fpath)
        n_rows = len(df_raw)
        t_norm = np.arange(n_rows) / max(n_rows - 1, 1)
        bin_ids = np.clip((t_norm * cfg.N_BINS).astype(int), 0, cfg.N_BINS - 1)

        def extract_cmd(s):
            if pd.isna(s): return "MOVE"
            s = str(s).strip()
            for cmd in ["G53", "G3", "G2", "G0", "M30"]:
                if s.startswith(cmd): return cmd
            return "MOVE"

        df_raw["cmd"] = df_raw["gcode_string"].apply(extract_cmd)
        df_raw["bin_id"] = bin_ids

        # Color background by dominant phase per bin
        for b in range(cfg.N_BINS):
            bin_data = df_raw[df_raw["bin_id"] == b]
            if len(bin_data) == 0:
                continue
            cmds = bin_data["cmd"].values
            n_rapid = np.sum(cmds == "G0") + np.sum(cmds == "G53")
            n_cut = np.sum(cmds == "MOVE") + np.sum(cmds == "G1")
            n_arc = np.sum(cmds == "G2") + np.sum(cmds == "G3")
            n_end = np.sum(cmds == "M30")

            if n_end > 0 and b >= cfg.N_BINS - 2:
                phase = "retract/end"
            elif n_rapid > n_cut + n_arc:
                phase = "rapid"
            elif n_arc > n_cut:
                phase = "arc_cutting"
            else:
                phase = "cutting"

            ax.axvspan(b + 0.5, b + 1.5, alpha=0.08, color=phase_colors[phase])

        ax.set_ylabel(f"{toolpath.capitalize()}\n{col_name}", fontsize=10)
        ax.set_xlim(0.5, cfg.N_BINS + 0.5)

        # Add phase annotations for first and last bins
        if row == 0:
            ax.annotate("Approach\n& ramp-in", xy=(2, ax.get_ylim()[1]),
                        fontsize=8, ha="center", va="top", color="gray")
            ax.annotate("Retract\n& end", xy=(19, ax.get_ylim()[1]),
                        fontsize=8, ha="center", va="top", color="gray")

    axes[-1].set_xlabel("Temporal Bin (normalized program time)")

    # Add legend for phases
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.3, label=p.replace("_", " ").capitalize())
                      for p, c in phase_colors.items() if p != "other"]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8, title="G-code Phase")

    fig.suptitle("Temporal Fingerprints with G-Code Phase Annotations\n"
                "(Background shading indicates dominant G-code command type per bin)",
                fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig15_gcode_phases")


# ═══════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════

def run_all():
    results = {}
    results["gcode_phases"] = analyze_gcode_phases()
    results["sensor_physics"] = analyze_sensor_physics()
    results["cutting_mechanics"] = analyze_cutting_mechanics()

    fig_physical_interpretation()
    fig_gcode_phase_annotation()

    # Save
    out_path = cfg.RESULTS_DIR / "physical_interpretation.json"
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"\nAll physical interpretation results saved to {out_path}")


if __name__ == "__main__":
    run_all()
