# Tables Regeneration Guide — `decoder_paper_v2`

**Last updated: 2026-05-12**

This document explains how to fill the `\TBD{...}` placeholders in
`latex/decoder_paper_v2.tex` and the placeholder cells in `tables/*.tex` once
the V8 full_window 5-fold sweep completes.

## Why placeholders exist

The previous manuscript shipped with numbers from a buggy run where
`--encoder_config` silently overrode `--data_dir`, causing the trainer to
load V7 dedup'd data (303 samples/fold) instead of V8 per_row data
(19,584 samples/fold). The bug is fixed at
[scripts/evaluation/run_decoder_quick_test.py:1488-1503](../../scripts/evaluation/run_decoder_quick_test.py#L1488-L1503).

The V7-data tables are preserved under `tables/v7_legacy/` for diffing
purposes. **Do not cite the legacy numbers in the rewritten manuscript.**

## Regeneration trigger

The full_window 5-fold sweep is queued to auto-launch the moment the HP
sweep (Stage 2 + Stage 3) finishes. The watcher is at
`/tmp/launch_full_window_when_sweep_done.sh` (pid recorded in the project
journal `notes.md`).

After the watcher fires:

1. Verify all five folds wrote `metrics.json`:
   ```
   ls outputs/decoder20260511/checkpoints/full_window_5fold/fold_*/results/metrics.json
   ```
2. Run the aggregator, telling it to use full\_window as the headline sweep:
   ```
   python3 scripts/analysis/aggregate_v8_results.py --sweep-name full_window_5fold
   ```
   This populates `outputs/decoder20260511/RESULTS_TABLE.json` with both the
   `full_window_5fold_aggregate` (the new headline) and the
   `per_row_5fold_aggregate` (kept for diffing). The legacy keys
   `phase5_5fold_sweep` and `per_class_5fold` mirror the headline sweep for
   backwards compatibility with downstream readers.
3. Re-run the per-field evaluator:
   ```
   python3 scripts/analysis/v8_per_field_eval.py \
     --checkpoints-root outputs/decoder20260511/checkpoints/full_window_5fold \
     --out outputs/decoder20260511/audit/v8_per_field_fullwindow.json
   ```
4. Re-run the ANOVA + bootstrap:
   ```
   python3 scripts/analysis/anova_and_bootstrap.py \
     --baseline-root outputs/decoder20260511/checkpoints/full_window_5fold \
     --out-dir       outputs/decoder20260511/decoder_paper_v2/tables/
   ```
5. Apply the placeholder replacements below.
6. Recompile:
   ```
   cd outputs/decoder20260511/decoder_paper_v2/latex
   pdflatex decoder_paper_v2.tex
   bibtex   decoder_paper_v2
   pdflatex decoder_paper_v2.tex
   pdflatex decoder_paper_v2.tex
   ```

## Placeholder → data-source map

The `\TBD{key}` macro produces a red `[TBD: key]` in the PDF that's easy to
grep for. To fill a placeholder, locate it in the `.tex` source and replace
with the value from the indicated JSON path.

### Headline 5-fold metrics

Source JSON: `outputs/decoder20260511/checkpoints/full_window_5fold/aggregate_metrics.json`
(or call `aggregate_v8_results.py` to emit it).

| Key | Path | Format |
|---|---|---|
| `token_acc_5fold` | `mean_test.token_accuracy ± std` | `$0.XXX \pm 0.XXX$` |
| `seq_acc_5fold` | `mean_test.sequence_accuracy ± std` | same |
| `type_acc_5fold` | `mean_test.type_accuracy ± std` | same |
| `cmd_acc_5fold` | `mean_test.command_accuracy ± std` | same |
| `paramtype_acc_5fold` | `mean_test.param_type_accuracy ± std` | same |
| `sign_acc_5fold` | `mean_test.sign_accuracy ± std` | same |
| `digit_acc_pooled` | `mean_test.numeric_accuracy ± std` | same |

The macro-{p,r,f1} keys (`token_macro_p` etc.) come from
`mean_test.per_head.<head>.macro_precision` etc. in the same JSON.

### Per-class command (Table~\ref{tab:per_class_command})

Source: `outputs/decoder20260511/audit/v8_per_class_command_fullwindow.json`.

| Key | Path |
|---|---|
| `G0_p`, `G0_r`, `G0_f1`, `n_G0_test` | `per_class.G0.{precision, recall, f1, support}` |
| `G1_p`, etc. | analogous |
| `G2_p`, etc. | analogous |
| `G3_p`, etc. | analogous |
| `M30_p`, etc. | analogous |

If a class doesn't appear in V8 (e.g. no `M30` in full_window labels),
remove that row from the table rather than leaving `0±0`.

### Per-class param-type (Table~\ref{tab:per_class_param_type})

Same scheme. Keys: `paramX_p`, `paramY_p`, `paramZ_p`, `paramR_p` and
corresponding `_r`, `_f1`, `n_<axis>_test`.

### Per-axis recoverability (Table~\ref{tab:per_axis})

Source: `outputs/decoder20260511/audit/v8_per_field_fullwindow.json`.

| Key | Path |
|---|---|
| `has_X_acc`, `has_X_f1`, `sign_X_acc`, `val_X_mae`, `presence_X_recall` | `per_axis.X.{has_accuracy, has_f1, sign_accuracy, value_mae, presence_recall}` |
| `has_Y_acc`, etc. | analogous |
| `has_Z_acc`, etc. | analogous |
| `has_R_acc`, etc. | analogous |

**Do NOT fill F/S/I/J cells** — leave the "insufficient support" text;
those fields have <5% positive support across all splits (Table~\ref{tab:data_coverage}).

### Per-digit-position (Table~\ref{tab:per_digit_position})

Source: `outputs/decoder20260511/audit/v8_per_digit_position_fullwindow.json`
(extend `aggregate_v8_results.py` to emit this).

Keys: `digit0_acc`, `digit0_f1`, `digit0_p`, `digit0_r` ... through digit 5.

For the inline numeric-decomposition sentence (Section~\ref{sec:numeric-decomposition}):
- `digit_mid_min`, `digit_mid_max`: scan digits 1–4 for min/max accuracy and quote them.
- `digit_min_f1`, `digit_min_label`, `digit_max_f1`, `digit_max_label`: scan per-digit-value F1 across all positions.

### Ablations summary (Table~\ref{tab:ablations})

Source: per-experiment `metrics.json` files:
- `abl_base_*`:  `outputs/decoder20260511/checkpoints/full_window_5fold/`
- `abl_sc_*`:    `outputs/decoder20260511/checkpoints/per_row_5fold_with_shortcuts/` (or full_window equivalent)
- `abl_pat_*`:   `outputs/decoder20260511/checkpoints/per_row_pattern_aware/` (or full_window equivalent)
- `abl_nogyro_*`, `abl_nocol_*`, etc.: `outputs/decoder20260511/ablations/sensor/zero_{group}/`

### ANOVA + bootstrap (Tables \ref{tab:anova}, \ref{tab:bootstrap_ci})

Run `scripts/analysis/anova_and_bootstrap.py` — it writes
`tables/anova.tex` and `tables/bootstrap_ci.tex` directly (replacing the
placeholders). You then re-add the placeholder header comment manually if
desired.

### Nested ablations (in-body table at Section~\ref{sec:abl-nested})

Source: per-experiment fold-1 `metrics.json` files. Cells are:
- `nest_base_full`, `nest_base_nogyro`, `nest_base_nocolor` — no shortcuts baseline
- `nest_sc_full`, `nest_sc_nogyro`, `nest_sc_nocolor` — with shortcuts
- `nest_pat_full`, `nest_pat_nocolor` — pattern-aware
- `nest_scpat_full` — shortcuts + pattern

Interaction effect numbers (`interaction_sc_gyro`, `interaction_sc_color`)
are computed as: `(nest_sc_nogyro - nest_sc_full) - (nest_base_nogyro - nest_base_full)`.

### LOCO (Section~\ref{sec:loco})

Source: `outputs/decoder20260511/loco/aggregate_metrics.json`.

| Key | Path |
|---|---|
| `loco_tok`, `loco_seq`, `loco_cmd`, `loco_num` | `mean ± std` across 9 holdouts |
| `loco_tok_drop` | `(baseline_tok - loco_tok) * 100` rounded |
| etc. for seq/cmd/num |
| `baseline_5fold_summary` | one-liner like `$0.XXX / 0.XXX / 0.XXX / 0.XXX$` |
| `loco_interpretation_phrase` | free-text 1-2 sentences interpreting the gap |

### Noise augmentation (Section~\ref{sec:noise-aug})

Source: `outputs/decoder20260511/checkpoints/noise_aug_5fold/aggregate_metrics.json`.

| Key | Path |
|---|---|
| `noise_tok`, `noise_seq`, `noise_cmd`, `noise_num` | 5-fold means ± std |
| `noise_effect_qualifier` | `improves` / `degrades` / `does not improve` |
| `noise_delta_phrase` | `within ~Xpp of` / `Xpp better than` / etc. |
| `noise_interpretation` | free-text 1-2 sentences |

### Per-axis qualitative (Section~\ref{sec:per-axis})

| Key | Source |
|---|---|
| `X_sign_qualifier` | `lower` if X-sign acc < Y-sign acc, else `comparable` |
| `X_sign_quality_qualifier` | free-text like `meaningful but imperfect` or `near-perfect` |

### Modality importance (Sections~\ref{sec:results-ablations} and ~\ref{sec:discussion})

| Key | Source |
|---|---|
| `loadbearing_modalities` | top-2 modalities by absolute Δ in sensor_ablation table |
| `loadbearing_delta_pp` | typical absolute Δ in pp (range like `1--2`) |
| `nonloadbearing_modalities` | modalities with Δ < 0.5pp |
| `loadbearing_modality_interpretation` | free-text 1-2 sentences |

### Pattern-aware (sequence-classifier) ablation

| Key | Source |
|---|---|
| `pattern_cmd_delta_qualifier` | `upward` / `downward` / `not significantly` |
| `pattern_cmd_delta_pp` | absolute Δ in pp on command |
| `pattern_num_delta_qualifier` | same for numeric |
| `pattern_num_delta_pp` | absolute Δ in pp on numeric |
| `pattern_paramtype_caveat` | optional caveat, e.g. `at the cost of a Xpp regression in parameter-type accuracy`, or empty |
| `pattern_net_effect_qualifier` | `small but positive` / `null` / `negative` |

### Vocabulary precision (2-digit) ablation

| Key | Source |
|---|---|
| `vocab2digit_tok`, `vocab2digit_seq`, `vocab2digit_cmd`, `vocab2digit_num` | fold-1 metrics from outputs/decoder20260511/checkpoints/vocab_2digit/fold_1/results/metrics.json |
| `vocab2digit_delta_phrase` | free-text describing the difference |
| `vocab2digit_categorical_qualifier` | `essentially unchanged` / `regressed` / etc. |

### Free-text qualifier keys (judgment calls)

These should be filled by hand based on the numbers, not by a script:

- `factorization_strength_qualifier`: `clean` / `qualified` / `partial` depending on how cleanly categorical heads exceed numeric heads.
- `categorical_recovery_qualifier`: e.g. `essentially solved` / `partially solved` / `not solved`.
- `categorical_significance_qualifier`, `categorical_significance_phrase`: based on ANOVA results.
- `categorical_claim_qualifier`, `numeric_claim_qualifier`: `are` / `appear` / etc.
- `pattern_color_interaction_summary`, `pattern_color_interaction_explanation`: free text describing the interaction (if any).
- `interaction_sc_gyro_phrase`: e.g. `the two main effects compose nearly additively` / `the interaction is significant`.

## Sanity checks before publishing

1. **No `\TBD{}` placeholders remain** in `decoder_paper_v2.tex` or in any
   `tables/*.tex` (except the v7_legacy backups). Grep:
   ```
   grep -rn '\\TBD{' outputs/decoder20260511/decoder_paper_v2/ \
     --exclude-dir=v7_legacy
   ```
   Expected: empty.
2. **Data coverage table** (`tables/data_coverage.tex`) is included and
   referenced near the per-axis recoverability table. Numbers should be
   re-verified from the V8 NPZ files (the current values are from
   2026-05-12 alignment check).
3. **Sensor-modality ablation directional claims** (gyroscope + color
   load-bearing) should be re-validated against the new full_window
   ablation numbers; remove or qualify if the direction flips.
4. **Frozen-encoder leakage footnote** (Section~\ref{sec:future}) — keep
   it. If Phase F (encoder retrain) runs, replace with the result.
5. **Compile the PDF and visually diff against `decoder_paper_v2.pdf`**
   to confirm structure is unchanged.

## Removed claims (do NOT bring back)

The following statements from the V7-data version are NOT to be restored
in the V8 version unless re-validated:

- "the decoder reaches 0.836 ± 0.025 token accuracy" — must come from V8 5-fold.
- "the dominant class G1 (490 test instances)" — V8 has different supports.
- "X-positive vs. X-negative motion sign is recovered with substantially lower fidelity (0.70)" — pp number from V7.
- "without metadata, a 27.1 pp ceiling exists on numeric recovery" — pp gap from V7-data ablations.
- "the gyroscope and color modalities are the two most-load-bearing modalities; their removal moves command accuracy 1--2 pp downward" — direction may hold, magnitude must be re-verified.
- "5-fold averages: token 0.821 ± 0.027, sequence 0.399 ± 0.038" for noise aug — V7 numbers.

## Open questions for the author after regeneration

After filling in the numbers, decide:

1. **Does the V8 5-fold baseline beat the V7 paper's 0.976 command ceiling?**
   If no (likely given the HP sweep plateau at ~0.50 cmd in per_row), the
   manuscript's narrative needs to pivot from "we beat the prior work" to
   "we honestly characterize the recoverability ceiling on this dataset
   and identify the data conditions needed to break it."
2. **Is the metadata-vs-no-metadata gap on numeric still ~27pp?** If the
   V8 corrected numbers show a different gap, the "information-theoretic
   ceiling" claim may need rewording.
3. **Is sequence accuracy >= 0.20?** If sequence-level reconstruction is
   essentially 0 on V8 (as the HP sweep suggests), the paper should
   downgrade sequence-level claims to per-field claims throughout.

These decisions block the rewrite of Sections \ref{sec:discussion} and
\ref{sec:conclusion}.
