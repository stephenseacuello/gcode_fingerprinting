#!/usr/bin/env python3
"""Aggregate the full_window LOMO encoder study -> fold-blocked attribution.

Reads each cell's decoder/results/metrics.json across the lomo arm:
  <root>/lomo/<modality>/fold_<1..5>/decoder/results/metrics.json

The leave-one-modality-out comparison is FOLD-BLOCKED. Fold-to-fold variance
(~9 pp: baseline ranges 0.839-0.932 across folds) dwarfs the modality effect
(~1 pp), so an unpaired test would drown the signal in fold noise. The
modality effect is therefore measured PER FOLD as

    Delta(m, f) = command(modality m, fold f) - command(baseline, fold f)

and aggregated over folds. Reported:
  - per modality: 5-fold command / token / numeric mean +/- sd (raw operating point)
  - per modality: paired Delta mean +/- sd, and a one-sample t-test of
    {Delta(m,f)} vs 0, Holm-corrected across the 7 modalities
  - one-way ANOVA ACROSS the 7 modalities on the per-fold Deltas (fold effect
    removed) -- "do the modalities differ from each other in removal cost?"
  - the unpaired Welch test (modality 5 accs vs baseline 5 accs) is also
    reported but flagged as the weaker, fold-confounded comparison; it is kept
    only to mirror the paper's inference-time-zeroing ablation stats.

Runs on partial sweeps. Non-destructive by default:
  (no flags)      -> writes audit/lomo_attribution.json + prints a summary
  --write-table   -> regenerates decoder_paper_v2/tables/lomo_modality.tex
  --write-figure  -> renders decoder_paper_v2/figures/lomo_modality_bars.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT_DEFAULT = REPO / "outputs/decoder20260511/lomo_fw"
PAPER_DEFAULT = REPO / "outputs/decoder20260511/decoder_paper_v2"
AUDIT_DEFAULT = REPO / "outputs/decoder20260511/audit"

MODALITIES = [
    ("baseline", "Full stack (baseline)"),
    ("accelerometer", "Accelerometer"),
    ("gyroscope", "Gyroscope"),
    ("magnetometer", "Magnetometer"),
    ("color", "Color"),
    ("temperature", "Temperature"),
    ("audio", "Audio-RMS"),
    ("electrical", "Electrical"),
]
# Companion group for the sensor-level LOMO -- one entry per physical sensor
# unit (each carrying 15 channels in the f98 base). Rebound onto MODALITIES in
# main() when --group-kind sensor.
SENSORS = [
    ("baseline", "Full stack (baseline)"),
    ("frame_l2", r"\texttt{frame\_l2}"),
    ("frame_l3", r"\texttt{frame\_l3}"),
    ("frame_r2", r"\texttt{frame\_r2}"),
    ("spindle2", r"\texttt{spindle2}"),
    ("y_bed__3", r"\texttt{y\_bed\_\_3}"),
    ("y_bed__4", r"\texttt{y\_bed\_\_4}"),
]
# Nested grid axes (modality x sensor; electrical excluded because it is a
# global modality not tied to a single physical sensor unit).
NESTED_SENSORS = ["frame_l2", "frame_l3", "frame_r2", "spindle2", "y_bed__3", "y_bed__4"]
NESTED_MODALITIES = ["accelerometer", "gyroscope", "magnetometer", "color", "temperature", "audio"]
# Cell key format: f"{sensor}__{modality}".  Underscore-underscore avoids
# collision with the y_bed__N sensor naming (kept as underscore-underscore
# itself; the split is always the LAST `__<modality>` suffix).
def _nested_cells() -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = [("baseline", "Full stack (baseline)")]
    for s in NESTED_SENSORS:
        s_disp = s.replace("_", r"\_")
        for m in NESTED_MODALITIES:
            key = f"{s}__{m}"
            cells.append((key, rf"\texttt{{{s_disp}}}\,/\,{m}"))
    return cells
NESTED = _nested_cells()
LOMO_MODS = [m for m, _ in MODALITIES if m != "baseline"]
FOLDS = [1, 2, 3, 4, 5]
METRICS = ["command", "token", "numeric", "type", "param_type"]

# Per-group display config -- main() swaps this for --group-kind sensor.
GROUP_CFG = {
    "kind": "modality",
    "header": "Modality removed",
    "table_basename": "lomo_modality",
    "table_label": "tab:lomo_modality",
    "caption_lead": "Leave-one-modality-out (LOMO)",
    "caption_row_phrase": "with that modality's channels removed",
    "figure_title": "Leave-one-modality-out encoder retraining (full-window, 5-fold)",
}
_SENSOR_CFG = {
    "kind": "sensor",
    "header": "Sensor unit removed",
    "table_basename": "lomo_sensor",
    "table_label": "tab:lomo_sensor",
    "caption_lead": "Leave-one-sensor-out (LOSO)",
    "caption_row_phrase": "with that sensor unit's channels removed (15 channels per unit)",
    "figure_title": "Leave-one-sensor-out encoder retraining (full-window, 5-fold)",
}
_NESTED_CFG = {
    "kind": "nested",
    "header": "Sensor / Modality removed",
    "table_basename": "lomo_nested",
    "table_label": "tab:lomo_nested",
    "caption_lead": "Nested leave-one-(sensor,modality)-out",
    "caption_row_phrase": "with only that sensor unit's channels for that modality removed (1--4 channels per cell)",
    "figure_title": "Nested leave-one-(sensor,modality)-out encoder retraining (full-window, 5-fold)",
}


def load_test_metrics(metrics_json: Path) -> dict | None:
    if not metrics_json.is_file():
        return None
    try:
        d = json.loads(metrics_json.read_text())
        tm = d.get("test_metrics", {})
        return {m: float(tm[f"{m}_accuracy"]) for m in METRICS if f"{m}_accuracy" in tm}
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        print(f"  WARN: unreadable {metrics_json}: {e}", file=sys.stderr)
        return None


def gate_ok(cell: Path) -> bool | None:
    g = cell / "channel_identity.json"
    if not g.is_file():
        return None
    try:
        return bool(json.loads(g.read_text()).get("all_pass"))
    except Exception:
        return None


def mean_sd(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return st.mean(xs), st.stdev(xs)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, p * (m - i)))
        out[k] = running
    return out


def collect(root: Path) -> dict:
    """root/lomo/<modality>/fold_<F> -> {modality: {fold: {metrics, gate}}}."""
    data: dict[str, dict] = {}
    for mod, _ in MODALITIES:
        per_fold = {}
        for f in FOLDS:
            cell = root / "lomo" / mod / f"fold_{f}"
            tm = load_test_metrics(cell / "decoder/results/metrics.json")
            if tm is not None:
                per_fold[f] = {"metrics": tm, "gate": gate_ok(cell)}
        data[mod] = per_fold
    return data


def summarize(data: dict) -> dict:
    # by_fold[modality][metric] = {fold: value}
    by_fold = {mod: {m: {} for m in METRICS} for mod in data}
    for mod, folds in data.items():
        for f, rec in folds.items():
            for m in METRICS:
                if m in rec["metrics"]:
                    by_fold[mod][m][f] = rec["metrics"][m]

    out: dict = {"modalities": {}, "stats": {}, "coverage": {}}
    base_cmd = by_fold.get("baseline", {}).get("command", {})

    delta_lists: dict[str, list[float]] = {}   # modality -> per-fold Deltas
    for mod, _disp in MODALITIES:
        v = by_fold.get(mod, {m: {} for m in METRICS})
        rec: dict = {"n_folds": len(v["command"])}
        for m in METRICS:
            vals = [v[m][f] for f in sorted(v[m])]
            mu, sd = mean_sd(vals)
            rec[m] = {"mean": mu, "sd": sd,
                      "by_fold": {str(f): v[m][f] for f in sorted(v[m])}}
        if mod != "baseline":
            common = sorted(set(v["command"]) & set(base_cmd))
            deltas = [v["command"][f] - base_cmd[f] for f in common]
            delta_lists[mod] = deltas
            dm, dsd = mean_sd(deltas)
            rec["delta_command"] = {
                "mean": dm, "sd": dsd, "n_paired": len(deltas),
                "per_fold": {str(f): v["command"][f] - base_cmd[f] for f in common},
            }
        out["modalities"][mod] = rec

    try:
        from scipy import stats as sps

        # (a) paired one-sample t-test per modality: per-fold Deltas vs 0
        paired = {}
        for m in LOMO_MODS:
            d = delta_lists.get(m, [])
            if len(d) >= 2:
                t, p = sps.ttest_1samp(d, 0.0)
                paired[m] = {"t": float(t), "p": float(p), "n": len(d)}
        if paired:
            hp = holm({m: paired[m]["p"] for m in paired})
            for m in paired:
                paired[m]["p_holm"] = hp[m]
        out["stats"]["paired_ttest_vs_baseline"] = paired

        # (b) fold-blocked ANOVA: across modalities, on the per-fold Deltas
        groups = [delta_lists[m] for m in LOMO_MODS if len(delta_lists.get(m, [])) >= 2]
        if len(groups) >= 2:
            F, p = sps.f_oneway(*groups)
            out["stats"]["anova_on_deltas"] = {"F": float(F), "p": float(p),
                                               "groups": len(groups)}

        # (c) unpaired Welch (weaker, fold-confounded; mirrors inference-time ablation)
        welch = {}
        bc = [base_cmd[f] for f in sorted(base_cmd)]
        if len(bc) >= 2:
            for m in LOMO_MODS:
                cm = [by_fold[m]["command"][f] for f in sorted(by_fold.get(m, {}).get("command", {}))]
                if len(cm) >= 2:
                    t, p = sps.ttest_ind(cm, bc, equal_var=False)
                    welch[m] = {"t": float(t), "p": float(p)}
        out["stats"]["welch_unpaired_vs_baseline"] = welch
    except ImportError:
        out["stats"]["error"] = "scipy not available -- stats skipped"

    done = sum(len(f) for f in data.values())
    exp = len(MODALITIES) * len(FOLDS)
    out["coverage"] = {"cells_done": done, "cells_expected": exp, "complete": done >= exp}
    return out


def _acc(rec_metric: dict, partial: bool) -> str:
    mu, sd = rec_metric["mean"], rec_metric["sd"]
    if mu is None:
        return "[pending]"
    s = f"${mu:.3f} \\pm {sd:.3f}$" if sd else f"${mu:.3f}$"
    return s + (r"$^{\dagger}$" if partial else "")


def build_table_tex(summary: dict) -> str:
    cov = summary["coverage"]
    complete = cov["complete"]
    st_ = summary.get("stats", {})
    _lead = GROUP_CFG["caption_lead"]
    _row = GROUP_CFG["caption_row_phrase"]
    _lbl = GROUP_CFG["table_label"]
    _hdr = GROUP_CFG["header"]
    L = [
        "% Regenerated by scripts/analysis/aggregate_lomo_results.py",
        f"% coverage: {cov['cells_done']}/{cov['cells_expected']} cells"
        + ("" if complete else "  -- PARTIAL"),
        r"\begin{table}[ht]", r"\centering",
        rf"\caption{{{_lead} encoder retraining, full-window, "
        rf"five-fold. Each row: the encoder retrained from scratch "
        rf"{_row}, then a fresh decoder trained on it "
        r"(headline recipe). \emph{Command} is the five-fold mean $\pm$ s.d. "
        r"$\Delta$\,Command is the \emph{fold-paired} effect --- the mean over "
        r"folds of command$(m,f)-$command$(\text{baseline},f)$ --- which "
        r"removes the large fold-to-fold variance. $p_{\text{Holm}}$ is the "
        r"Holm-corrected one-sample $t$-test of those per-fold deltas against "
        r"zero. Contrast with the inference-time zeroing of "
        r"Table~\ref{tab:sensor_ablation}, which holds a frozen encoder fixed.}",
        rf"\label{{{_lbl}}}", r"\small",
        r"\begin{tabular}{l c c c c c}", r"\toprule",
        rf"\textbf{{{_hdr}}} & "
        r"\textbf{Command} & \textbf{Token} & "
        r"\textbf{Numeric} & \textbf{$\Delta$ Command (paired)} & "
        r"\textbf{$p_{\text{Holm}}$} \\", r"\midrule",
    ]
    paired = st_.get("paired_ttest_vs_baseline", {})
    for mod, disp in MODALITIES:
        r = summary["modalities"][mod]
        partial = 0 < r["n_folds"] < len(FOLDS)
        cmd, tok, num = (_acc(r[k], partial) for k in ("command", "token", "numeric"))
        if mod == "baseline":
            L.append(f"\\textit{{{disp}}} & {cmd} & {tok} & {num} & --- & --- \\\\")
        else:
            dc = r.get("delta_command", {})
            if dc.get("mean") is None:
                dtxt, ptxt = "[pending]", "---"
            else:
                dtxt = f"${dc['mean']:+.3f} \\pm {dc['sd']:.3f}$"
                ph = paired.get(mod, {}).get("p_holm")
                ptxt = "---" if ph is None else f"${ph:.3f}$"
            L.append(f"{disp} & {cmd} & {tok} & {num} & {dtxt} & {ptxt} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip"]
    foot = []
    an = st_.get("anova_on_deltas")
    if an:
        foot.append(f"Fold-blocked one-way ANOVA across the {an['groups']} LOMO "
                    f"conditions (on the per-fold deltas): $F={an['F']:.2f}$, "
                    f"$p={an['p']:.3g}$.")
    if paired:
        sig = sum(1 for v in paired.values() if v.get("p_holm", 1) < 0.05)
        foot.append(f"Per-modality paired $t$-test vs.\\ baseline "
                    f"(Holm-corrected): {sig}/{len(paired)} significant at "
                    f"$\\alpha=0.05$.")
    foot.append(f"Coverage: {cov['cells_done']}/{cov['cells_expected']} cells.")
    if not complete:
        foot.append(r"$^{\dagger}$ partial ($<$5 folds).")
    L += [r"\footnotesize\textit{" + " ".join(foot) + "}", r"\end{table}", ""]
    return "\n".join(L)


def render_figure(summary: dict, out_pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mods = [(m, d) for m, d in MODALITIES if m != "baseline"]
    deltas = [(summary["modalities"][m].get("delta_command") or {}).get("mean") or 0
              for m, _ in mods]
    sds = [(summary["modalities"][m].get("delta_command") or {}).get("sd") or 0
           for m, _ in mods]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = range(len(mods))
    ax.bar(x, deltas, yerr=sds, capsize=4, color="#4878a8")
    ax.axhline(0, ls="--", color="#a83232")
    ax.set_xticks(list(x))
    # plain-text x labels (strip \texttt{} etc.); only for the figure
    def _plain(s): return s.replace(r"\texttt{", "").replace(r"\_", "_").rstrip("}")
    ax.set_xticklabels([_plain(d) for _, d in mods], rotation=30, ha="right")
    ax.set_ylabel(r"$\Delta$ command vs baseline (fold-paired)")
    ax.set_title(GROUP_CFG["figure_title"])
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150)
    plt.close(fig)


def render_nested_heatmap(summary: dict, out_pdf: Path) -> None:
    """6 sensors x 6 modalities heatmap of paired Delta command."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    paired = summary.get("stats", {}).get("paired_ttest_vs_baseline", {})
    sensors = NESTED_SENSORS
    modalities = NESTED_MODALITIES
    grid = np.full((len(sensors), len(modalities)), np.nan)
    p_grid = np.full((len(sensors), len(modalities)), np.nan)
    n_grid = np.zeros((len(sensors), len(modalities)), dtype=int)
    for i, s in enumerate(sensors):
        for j, m in enumerate(modalities):
            key = f"{s}__{m}"
            rec = summary["modalities"].get(key, {})
            dc = rec.get("delta_command") or {}
            if dc.get("mean") is not None:
                grid[i, j] = dc["mean"]
            n_grid[i, j] = rec.get("n_folds", 0)
            ph = paired.get(key, {}).get("p_holm")
            if ph is not None:
                p_grid[i, j] = ph

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    vmax = max(0.03, np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 0.03)
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(modalities)))
    ax.set_xticklabels(modalities, rotation=30, ha="right")
    ax.set_yticks(range(len(sensors)))
    ax.set_yticklabels(sensors)
    for i in range(len(sensors)):
        for j in range(len(modalities)):
            v = grid[i, j]
            if np.isnan(v):
                txt = "·" if n_grid[i, j] == 0 else "?"
                ax.text(j, i, txt, ha="center", va="center", color="#666", fontsize=8)
                continue
            star = "*" if (np.isfinite(p_grid[i, j]) and p_grid[i, j] < 0.05) else ""
            ax.text(j, i, f"{v:+.3f}{star}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 * vmax else "black", fontsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\Delta$ command (paired)")
    ax.set_title(GROUP_CFG["figure_title"])
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150)
    plt.close(fig)


def print_summary(summary: dict) -> None:
    cov = summary["coverage"]
    print(f"\n=== LOMO full_window aggregation -- {cov['cells_done']}/{cov['cells_expected']} cells "
          f"({'COMPLETE' if cov['complete'] else 'PARTIAL'}) ===")
    print(f"{'modality':<22}{'n':>3}  {'command':>16}  {'paired Δcmd':>18}  {'p_holm':>8}")
    paired = summary.get("stats", {}).get("paired_ttest_vs_baseline", {})
    for mod, disp in MODALITIES:
        r = summary["modalities"][mod]
        mu, sd = r["command"]["mean"], r["command"]["sd"]
        c = "  pending" if mu is None else f"{mu:.3f}±{sd:.3f}"
        if mod == "baseline":
            print(f"{disp:<22}{r['n_folds']:>3}  {c:>16}")
            continue
        dc = r.get("delta_command", {})
        dt = "" if dc.get("mean") is None else f"{dc['mean']:+.4f}±{dc['sd']:.4f}"
        ph = paired.get(mod, {}).get("p_holm")
        pt = "" if ph is None else f"{ph:.3f}"
        print(f"{disp:<22}{r['n_folds']:>3}  {c:>16}  {dt:>18}  {pt:>8}")
    an = summary.get("stats", {}).get("anova_on_deltas")
    if an:
        print(f"\nfold-blocked ANOVA on per-fold deltas ({an['groups']} modalities): "
              f"F={an['F']:.3f}  p={an['p']:.4g}")
    elif summary["coverage"]["cells_done"] < summary["coverage"]["cells_expected"]:
        print("\n(ANOVA pending -- needs >=2 folds per modality)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--paper-dir", type=Path, default=PAPER_DEFAULT)
    ap.add_argument("--audit-dir", type=Path, default=AUDIT_DEFAULT)
    ap.add_argument("--group-kind", choices=["modality", "sensor", "nested"], default="modality",
                    help="modality (channel-type LOMO), sensor (physical-unit LOMO), "
                         "or nested (6 sensors x 6 modalities grid)")
    ap.add_argument("--write-table", action="store_true")
    ap.add_argument("--write-figure", action="store_true")
    a = ap.parse_args()

    # Switch the active group + display config based on --group-kind.
    global MODALITIES, LOMO_MODS, GROUP_CFG
    if a.group_kind == "sensor":
        MODALITIES = SENSORS
        LOMO_MODS = [m for m, _ in SENSORS if m != "baseline"]
        GROUP_CFG = _SENSOR_CFG
    elif a.group_kind == "nested":
        MODALITIES = NESTED
        LOMO_MODS = [m for m, _ in NESTED if m != "baseline"]
        GROUP_CFG = _NESTED_CFG

    if not a.root.is_dir():
        print(f"LOMO root not found: {a.root}", file=sys.stderr)
        return 1

    summary = summarize(collect(a.root))
    print_summary(summary)

    a.audit_dir.mkdir(parents=True, exist_ok=True)
    js = a.audit_dir / f"{GROUP_CFG['table_basename']}_attribution.json"
    js.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {js}")

    if a.write_table:
        tex = a.paper_dir / f"tables/{GROUP_CFG['table_basename']}.tex"
        tex.write_text(build_table_tex(summary))
        print(f"wrote {tex}")
    if a.write_figure:
        try:
            if a.group_kind == "nested":
                pdf = a.paper_dir / f"figures/{GROUP_CFG['table_basename']}_heatmap.pdf"
                render_nested_heatmap(summary, pdf)
            else:
                pdf = a.paper_dir / f"figures/{GROUP_CFG['table_basename']}_bars.pdf"
                render_figure(summary, pdf)
            print(f"wrote {pdf} (+ .png)")
        except Exception as e:
            print(f"figure render failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
