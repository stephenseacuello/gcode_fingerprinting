# Summer Tormach DOE — Specification

Phase-7 deliverable of the `decoder20260511` remediation. This document
spec'd alongside Tim's hire (Stephen, 2026-04-28 meeting) and the
forthcoming Tormach arrival.

## Why a new dataset

The 2026 audit (`AUDIT_REPORT.md`) showed:

1. The existing 120-file Bantam dataset has only **22 distinct G-code
   lines per fold**. A metadata-only baseline reaches 75–95% on this
   small label set, leaving little room for the sensor decoder to
   demonstrate actual signal recovery.
2. Feed rate `F` is largely absent from the V7 labels (only 2 NUM_F
   tokens in the V8 vocab over 225K source tokens). Any "recover feed
   rate" claim has to be made on data that actually varies feed rate.
3. The same tool path repeats across most files within each operation
   class, which is what enables the positional shortcut.

The summer DOE is the cure: vary the factors that matter, generate
**hundreds of distinct single-line G-code experiments**, and run them
on the Tormach so the decoder has a label set large and diverse
enough that positional shortcut becomes useless.

## Factor design

| Factor          | Levels                                                  | Reason                                             |
|-----------------|---------------------------------------------------------|----------------------------------------------------|
| motion_type     | straight / arc_cw / arc_ccw                             | Recover G0/G1 vs G2/G3 classification              |
| direction       | x_pos / x_neg / y_pos / y_neg / xy_diag / n/a           | Test sensor sign discrimination                    |
| feed_rate       | 50 / 100 / 200 / 400 mm·min⁻¹                           | Phase-5 audit blocked on F absence — this fixes it |
| spindle_speed   | 3000 / 6000 / 9000 rpm                                  | Strong audio + current effect                      |
| depth_of_cut    | 0.10 / 0.25 / 0.50 / 1.00 mm  (plus 0.0 for air cuts)   | Stephen's specific recoverability target           |
| material        | aluminum_6061 / steel_1018 / delrin                     | Different chip-formation regimes; stresses encoder |
| tool_diameter   | 3.0 / 6.0 / 9.5 mm                                      | Modulates force / vibration signature              |
| air_cut         | False / True                                            | Negative control: spindle on but no material       |

Full factorial: 3×6×4×3×5×3×3×2 = 19,440 runs. **Way too many.**

Default DOE: sampled 200 runs using a randomized-sweep fractional
design (`scripts/doe/build_doe_table.py --target-n 200`). Output
shows reasonable factor coverage across all levels for ≤200 runs.

Recommended growth path:

- **Pilot (50 runs)** — week 1 after Tormach lands. Validate the
  recording pipeline, alignment timing, tool wear under steel cuts.
- **Main (200–300 runs)** — once pilot looks clean. This is the dataset
  the manuscript is rewritten against.
- **Extension (500+ runs)** — only if Phase 5–6 numbers on the main
  set still show residual positional shortcut.

## Per-run G-code structure

Each run produces a minimal G-code program:

```
G21 G90 G17 G94          (setup)
M3 S<spindle> if !air_cut (spindle on)
G0 X0 Y0 Z2.5            (safe Z rapid)
G1 Z<-depth_of_cut> F<feed>   (plunge)
<motion line>            (the characterising move — THIS is the line of interest)
G0 Z2.5                  (retract)
M5 M30                   (stop)
```

Generated automatically by `scripts/doe/generate_single_line_gcode.py
--doe doe_vN.json --output-dir programs/`. Verified on sample
DOE_0000–DOE_0187. Distance of the characterising move defaults to
25 mm (overridable).

## Per-run data capture

For each run we need (in addition to the standard sensor CSV):

1. **CNC log with per-line timestamps.** Required for the
   timestamp-based alignment in `scripts/doe/auto_label_alignment.py`.
   Without timestamps we fall back to interpolation, which is OK for
   the manuscript but adds noise.
2. **Tool image before AND after the run.** Stephen's R2-camera idea
   from the 2026-04-28 meeting. Even a single before/after PNG per run
   is enough for the manuscript's tool-wear analysis.
3. **Material origin tracking.** Stock thickness, fixture ID, anything
   that lets us split train/test by stock and check for cross-stock
   generalization.

## Train/test split policy

The audit (`audit/shortcut_leakage.json`) showed `source_file` is a
near-perfect shortcut on the V7 data. For the DOE dataset:

- Split by **(material, tool_diameter)** pair so the test split contains
  a (material, tool) combination unseen in training. This is the
  hardest split and the one whose results we should headline.
- Reject splits where a G-code line appears in test but not train.

`scripts/preprocessing/run_preprocessing_v8_cv_fold.py` already
implements coverage repair via file swaps; reuse it.

## Pipeline path summary

1. **DOE table**:    `scripts/doe/build_doe_table.py` → JSON + CSV.
2. **G-code emit**:  `scripts/doe/generate_single_line_gcode.py` → 200 .gcode files + manifest.
3. **Run experiments**: Tormach + sensor stack record one CSV per run, named `{run_id}.csv`.
4. **Align**:         `scripts/doe/auto_label_alignment.py` → `{run_id}_aligned.csv` with `gcode_string` column.
5. **Preprocess**:    `scripts/preprocessing/run_preprocessing_v8_cv_fold.py` consumes the aligned CSVs.
6. **Train**:         existing `scripts/experiments/train_v8_smoke.sh` / `train_v8_full_window.sh` with `--data_dir` pointing at the new preprocessed root.
7. **Evaluate**:      `scripts/analysis/aggregate_v8_results.py` produces the per-field recoverability table for the manuscript.

## Open items for the Monday working session

- Tormach PO status (Stephen → Lois / Woody).
- Material order: aluminum_6061 standard stock, steel_1018 stock,
  delrin block — confirm enough for ~200 runs across 3 materials.
- Tool inventory: 3.0 / 6.0 / 9.5 mm endmills (Tim's responsibility).
- R2 camera placement and image-capture trigger.
- Per-line timestamp logging from the Tormach controller (have to
  check whether the LinuxCNC backend exposes this in the standard CSV
  output).

---

Files in `outputs/decoder20260511/DOE/`:

```
DOE/
├── doe_v1.json            # 188-run factorial (post-validity filter)
├── doe_v1.csv             # same, flat CSV for sharing
└── programs/
    ├── manifest.json      # run_id → factors + file path
    ├── DOE_0000.gcode
    ├── DOE_0001.gcode
    └── ... (188 .gcode files)
```
