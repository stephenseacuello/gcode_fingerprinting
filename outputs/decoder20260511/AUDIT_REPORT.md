# Phase 1 Audit Report — Decoder Remediation

Generated 2026-05-11 as the **Phase 1 gate G1 deliverable** for the
`decoder20260511` work. See master plan at
`/home/seacuello/.claude/plans/so-i-want-to-warm-wave.md`.

This report verifies the 11 concerns raised in the 2026-04-28 weekly
meeting against the actual V7 code, data, and checkpoints in
`outputs/decoder20260304/`. **No code has been modified** as of this
report — everything below is read-only evidence, JSON artifacts, and
file/line citations. Phase 2 begins only after user sign-off on these
findings.

Artifacts:

- `audit/diagnostics_v7.json` — NPZ structural probe (15 files, 5 folds × 3 splits)
- `audit/diagnostics_v7.md` — per-file table
- `audit/truncation_impact.json` — direct + hypothetical multi-line truncation
- `audit/shortcut_leakage.json` — metadata-only label predictability
- `audit/recoverability_baseline.json` — metadata-only per-field accuracy / MAE
- `audit/v7_per_field.json` — actual V7 decoder per-field accuracy / MAE (5-fold, best seeds)

---

## TL;DR

**The truncation bug is real but DORMANT in V7 data.** Preprocessing collapses
each 256-sample window to a single G-code line, so `tokens.shape[1]==6` for
every NPZ — well below the 16-token cap. The cap would silently lose
~17–19% of source files and ~4–15% of tokens *if* multi-line targets were
generated (Phase 2 will do this), and **it must be fixed before Phase 2 lands.**

**The deeper architectural problem dominates:** with one G-code line per
64-second window, the decoder is effectively a 22-way classifier with full
positional metadata in scope. A metadata-only baseline (XGBoost on
`window_index, total_windows, operation_type, source_file_hash`) recovers
the G-code label at **75–95% test accuracy** — within a few points of the
published V7 token accuracy of 97.9%. The sensor pathway contributes
marginal lift over positional shortcut.

**Per-field recoverability from metadata alone** (no sensors):
command 98–100%, has_x 87–98%, has_y 95–98%, has_z 100% (nearly all
windows lack Z anyway), x_val MAE 0.13–0.32, y_val MAE 0.13–0.64. Feed
rate `F` is absent from the V7 labels entirely — it cannot be evaluated
as a recoverability target on this dataset.

**Actual V7 decoder per-field (the ceiling — see `audit/v7_per_field.json`):**
command 97.5±1.1%, has_x 100.0%, has_y 99.2±0.8%, x_sign 96.2±1.6%,
y_sign 100.0%, x_val MAE 0.20±0.09, y_val MAE 0.24±0.14. **V7 vs floor
gap**: command is *matched* by the metadata floor (0pp gap); axis
presence is improved by V7 (+2–13pp on has_x, +1–4pp on has_y); value
regression MAE is in the same range as the metadata baseline. The
sensor pathway contributes meaningful lift on axis-presence detection
but is matched by positional shortcut on command and value regression.

**Recommended go-aheads for Phase 2+:** all 11 priorities remain in
scope; the order is unchanged. Phase 2 must (a) write `token_length`
not `window_length`, (b) raise the per-line cap, and (c) preserve
multi-line targets per window. Phase 4 must (a) remove
`use_window_position` from the default config and (b) hide
`window_index / total_windows / source_file` from the dataset
`__getitem__` unless explicitly requested for ablation.

---

## Priority-by-priority verification

### Priority 1 — Token / window truncation bug

**Claim (meeting):** `max_token_length=16` is applied to the entire sensor
window's target instead of each G-code line, silently dropping tokens.

**Evidence:**

1. **`src/miracle/dataset/preprocessing.py:387`** writes
   `lengths = np.array([w['length'] for w in windows], dtype=np.int64)`
   where `w['length']` is set to `self.window_size` (256) at line 299,
   not `len(token_ids)`. So the NPZ `lengths` field stores the **sensor**
   window length, not the **token** sequence length.

2. Across all 15 V7 NPZ files, `lengths` is constant at `256` and
   content-token length (counting non-PAD/BOS/EOS) ranges 1–6. See
   `audit/diagnostics_v7.json` — every report carries the issue
   `lengths_field_is_sensor_not_token_length`.

3. **`src/miracle/dataset/decoder_dataset.py:91`** uses the buggy field
   directly: `seq_len = min(self.lengths[i].item(), max_len - 2)`. With
   `max_token_len=16` (the V7 default) and `lengths=256`, this resolves
   to `seq_len = 14`. Truncation reapplied at lines 110–112.

4. **The cap is dormant in V7** because the preprocessing already
   collapsed each window to a single G-code line. Static analysis (re-
   tokenising each window's `gcode_texts` entry) shows 0 samples exceed
   16 content tokens — see `audit/truncation_impact.json` direct
   analysis. Maximum observed content length is 6.

5. **The cap is NOT dormant once preprocessing emits multi-line
   targets.** Hypothetical analysis (concatenating all unique G-code
   lines per `source_file`) shows that at `cap=16`, 17–19% of source
   files would lose tokens and 4–15% of all tokens would be discarded.
   At `cap=14` (the effective content limit after BOS/EOS reservation),
   the losses are 19–20% of files and 9–15% of tokens. At `cap=32` the
   loss drops to ≤4% of files / ≤2% of tokens.

6. **Hardcoded `max_token_len=16` defaults at five eval entry points** —
   they all instantiate `DecoderQuickTestDataset(..., max_token_len=16)`
   regardless of NPZ contents:
   - [scripts/evaluation/run_decoder_baselines.py:522](scripts/evaluation/run_decoder_baselines.py#L522), 566
   - [scripts/evaluation/eval_full_per_class_v7.py:69](scripts/evaluation/eval_full_per_class_v7.py#L69)
   - [scripts/evaluation/eval_constrained_decoding.py:213](scripts/evaluation/eval_constrained_decoding.py#L213)
   - [scripts/evaluation/run_hybrid_position_decoding.py:359](scripts/evaluation/run_hybrid_position_decoding.py#L359), 418, 420
   - [scripts/evaluation/run_decoder_quick_test.py:236, 1192](scripts/evaluation/run_decoder_quick_test.py)

7. **`DecoderQuickTestDataset` re-tokenises from `gcode_texts`** at
   runtime ([scripts/evaluation/run_decoder_quick_test.py:284–300](scripts/evaluation/run_decoder_quick_test.py#L284-L300))
   and explicitly truncates at line 300: `tok_ids = tok_ids[:max_token_len - 1]`.
   The class supports multi-line input via `text_str.split('\n')`, so
   once preprocessing emits multi-line targets the cap will start biting
   unless raised.

**Severity:** HIGH — the latent bug will silently corrupt training the moment
Phase 2's multi-line preprocessing lands.

**Phase 2+ fix:** (a) `preprocessing.py:299, 387` writes `token_length =
len(token_ids)` and separately `window_length = self.window_size`;
(b) `decoder_dataset.py` accepts `length_field={"token_length"}` and
asserts `max(token_length) <= max_len - 2`; (c) eval scripts default
`max_token_len=None`, resolved from NPZ metadata at load.

### Priority 2 — Compare per-row vs full-window decoder strategies

**Claim (meeting):** Currently each 256-sample window holds one G-code
label, so the model never sees the rest. Two alternatives were proposed.

**Evidence:**

- `audit/diagnostics_v7.json` confirms `gcode_texts_line_count_distribution`
  is `{"1": <n>}` across all 15 NPZ splits. There are no multi-line
  windows in V7.
- 18–26 distinct G-code texts per split (very small label set).
- Preprocessing `create_windows` ([src/miracle/dataset/preprocessing.py:273–289](src/miracle/dataset/preprocessing.py#L273-L289)) DOES extract `ordered_unique` G-code lines and tokenize them
  jointly, but the saved `gcode_text` is the newline-joined string of
  uniques (line 289). Empirically, the join only has ONE entry per
  window because at 4 Hz a 256-sample window covers 64 seconds and
  the test programs spend most windows inside a single G-code instruction.

**Severity:** STRUCTURAL. Cannot be remediated by truncation fix alone.

**Phase 2+ fix:** generate two preprocessed datasets fresh:
`outputs/decoder20260511/preprocessed/full_window/` (multi-line targets
preserved) and `outputs/decoder20260511/preprocessed/per_row/` (one
sample per G-code line, full 256-sample sensor context). The team has
already approved Phase 4 compute budget of full_window × 5 folds +
per_row fold-1 pilot.

### Priority 3 — Remove shortcut-learning features

**Claim (meeting):** The decoder may be memorizing window position and
file identity rather than learning sensor → G-code mappings.

**Evidence (`audit/shortcut_leakage.json`):**

| Fold | Majority | OpType only | (Op, WindowIdx) lookup | XGBoost (all metadata) |
|------|----------|-------------|------------------------|------------------------|
| 1    | 0.205    | 0.432       | **0.803**              | **0.754**              |
| 2    | 0.204    | 0.487       | **0.885**              | **0.901**              |
| 3    | 0.217    | 0.443       | **0.943**              | **0.934**              |
| 4    | 0.204    | 0.491       | **0.944**              | **0.915**              |
| 5    | 0.194    | 0.462       | **0.946**              | **0.946**              |

Inputs to the metadata classifiers: `window_index, total_windows,
norm_position (=window_index/total_windows), operation_type,
source_file_hash`. NO sensor data.

The published V7 multi-seed token accuracy was 97.9%. The sensor pathway
buys only ~3–5pp over the pure positional+operation lookup baseline.

**V7 checkpoints actively consume `window_pos_embed`:**

```
src/miracle/model/sensor_multihead_decoder.py:783–786, 1143–1151
v7 ckpt has: window_pos_embed.weight [13, 96],
              window_frac_proj.weight [48, 1],
              window_pos_proj.weight [384, 144]
```

So V7 best is trained with `use_window_position=True`. Disabling it
should reduce decoder accuracy materially and is the right ablation to
isolate sensor contribution.

**Severity:** STRUCTURAL. Most of the V7 paper's reconstruction number
is attributable to shortcuts.

**Phase 4 fix:** `decoder_v8_no_shortcuts.json` config with
`use_window_position=false`, plus `expose_position_metadata=False`
default in `DecoderDataset.__getitem__` so `window_index / total_windows
/ source_file` never reach the model unless an ablation explicitly opts
in.

### Priority 4 — Noise-based augmentation

**Claim (meeting):** Add noise to features / labels to reduce
memorization and force sensor-driven learning.

**Evidence (audit only — no run yet):**

- `src/miracle/dataset/data_augmentation.py` already exists in the repo
  with sensor noise / jitter knobs at conservative defaults. Token
  masking is not wired up.
- With shortcut leakage at 80–95%, light sensor noise alone is unlikely
  to break the shortcut path — the *source* of the shortcut is the
  metadata fields, not the sensors. Priority 4 should be combined with
  Priority 3 (remove the metadata path entirely) and noise should be
  used to test sensor robustness, not as the primary cure.

**Phase 4 fix:** extend augmentation to `(sensor_noise_std,
feature_dropout_prob, token_mask_prob)`; expose via
`decoder_v8_noise_aug.json`.

### Priority 5 — Reframe evaluation around recoverable G-code components

**Claim (meeting):** Move from full-string reconstruction to per-field
recoverability — command, direction, feed rate, depth of cut.

**Evidence (`audit/recoverability_baseline.json`):**

Metadata-only XGBoost test accuracy per field, averaged across 5 folds:

| Field      | Maj test | XGB test | n_classes | Notes |
|------------|----------|----------|-----------|-------|
| command    | 0.72     | **0.99** | 3         | G0/G1/G2 (G3 absent from V7 train splits) |
| has_x      | 0.70     | **0.93** | 2         |       |
| has_y      | 0.60     | **0.96** | 2         |       |
| has_z      | 0.98     | **1.00** | 2         | Z almost never present |
| has_r      | 0.96     | **1.00** | 2         | R almost never present |
| has_f      | n/a      | n/a      | n/a       | **Feed rate absent from V7 labels** |

Regression fields, MAE (mean-fill vs XGBoost):

| Field | Mean-fill MAE | XGB MAE | n_non_nan_train | Notes |
|-------|---------------|---------|-----------------|-------|
| x_val | 1.41–1.46     | 0.13–0.32 | high           |       |
| y_val | 0.85–1.05     | 0.13–0.64 | high           |       |
| z_val | ~0            | ~0      | few–zero        | Mostly absent |
| f_val | n/a           | n/a     | zero            | **F absent in V7 labels** |
| r_val | 0–0.02        | 0.00    | sparse          | Constant per file |

**Key takeaway:** The V7 dataset does not contain feed rate. Any
recoverability claim about `F` requires the new summer DOE dataset
(Priority 10). For the existing data, `command`, `has_x/y`, `x_val`,
`y_val` are the meaningful recoverability targets, and the metadata-only
baseline already saturates them.

**Per-field ceiling: V7 actual decoder vs metadata floor (5-fold means):**

| Field    | Floor (XGB, no sensors) | V7 actual (with sensors+pos) | Gap     |
|----------|-------------------------|------------------------------|---------|
| command  | 0.98–1.00              | **0.975 ± 0.011**             | ≈0      |
| has_x    | 0.87–0.98              | **1.000 ± 0.000**             | +2–13pp |
| has_y    | 0.95–0.98              | **0.992 ± 0.008**             | +1–4pp  |
| x_sign   | n/a (XGB skipped)      | **0.962 ± 0.016**             | n/a     |
| y_sign   | n/a                    | **1.000 ± 0.000**             | n/a     |
| has_z    | ~1.00 (all 0)          | **1.000**                     | 0       |
| has_r    | ~1.00 (all 0)          | **1.000**                     | 0       |
| x_val MAE| 0.13–0.32              | **0.199 ± 0.091**             | similar |
| y_val MAE| 0.13–0.64              | **0.241 ± 0.142**             | similar |
| z_val MAE| ~0 (sparse target)     | **0.000** (presence recall 0.6) | ≈0    |
| r_val MAE| ~0 (constant per file) | **0.000**                     | 0       |

**Interpretation:** The V7 published 97.9% token accuracy and the
shortcut audit's 75–95% metadata baseline are not in direct
contradiction. The sensor pathway adds real lift on **axis-presence
detection** (whether an X/Y appears in the line at all). On **command
identity** and **numeric value regression**, V7's accuracy and MAE
match the metadata floor — i.e., these fields are recoverable from
position + operation type alone. This is what the manuscript reframe
needs to acknowledge.

**Phase 5+ fix:** add per-field metrics to
`src/miracle/training/metrics.py`. Report v8 decoder accuracy on
command, axis-presence, and X/Y value recovery against the metadata
baselines above as the floor.

### Priority 6 — Command/parameter-level structured heads

**Claim (meeting):** Build a parser that converts each G-code line into
structured fields, then add prediction heads for command type / feed /
depth / direction.

**Evidence:**

- `src/miracle/model/sensor_multihead_decoder.py` already has
  `type_head`, `command_head`, `param_type_head`, `sign_head`,
  `digit_value_head` (multi-head decoder), and the trainer logs all of
  them. This is the existing structured-head infrastructure to extend.
- A clean fielded parser is *not* in the repo; this audit's
  `score_recoverability.py` includes a minimal regex parser that should
  be moved into `src/miracle/utilities/` in Phase 6 and reused by both
  the trainer's metric module and the dataset's structured-target
  builder.

**Phase 6 fix:** add `src/miracle/model/structured_field_heads.py` for
the new heads (command + axis + numeric value) sharing the sensor
memory, and surface the regex parser to a stable utility module.

### Priority 7 — Pattern-aware decoder

**Claim (meeting, Dr. Sodhi):** Long G-code programs reuse motion
patterns with small parameter changes; the decoder should be informed
of those patterns.

**Evidence:**

- 18–26 distinct `gcode_texts` per split. The pattern set is small
  enough to enumerate.
- `src/miracle/model/sensor_multihead_decoder.py` includes a
  `sequence_classifier` head and operation-sequence prior masking
  (lines 1157–1162). That's the seed of a pattern-aware decoder.

**Phase 6 fix:** add `src/miracle/model/pattern_aware_decoder.py` that
biases the token-level logits with a sensor-conditioned prior over the
observed pattern set, treating the 22–34 distinct lines as the initial
template inventory.

### Priority 8 — Diagnostics and failure visibility

**Claim (meeting):** The bug was silent. Add hard checks.

**Evidence:**

- `decoder_dataset.py:91` is the canonical silent-truncation site
  (`min(...)` rather than `assert`).
- `DecoderQuickTestDataset` records `stats = {'mean_token_len':...,
  'max_token_len':...}` at line 328 but the value is only logged in
  console output, not asserted against a target.

**Phase 3 fix:** replace `min(...)` with `assert max(token_length) <=
max_len - 2`, add pytest fixtures in
`tests/unit/test_preprocessing_invariants.py` and
`tests/unit/test_decoder_dataset_no_truncation.py`, and write
`src/miracle/dataset/preprocessing_diagnostics.py` that emits a JSON
report after each preprocessing run.

### Priority 9 — Sensor importance for decoder tasks

**Claim (meeting):** Revisit which sensors are useful for reconstruction
(gyroscope was important in prior ablations, per Stephen).

**Evidence:**

- V7 used 110 continuous sensor channels across 6 boards (`frame_l2,
  frame_l3, frame_r2, spindle2, y_bed__3, y_bed__4`) — see
  `outputs/decoder20260304/preprocessed_v7/fold_1/metadata.json:consistent_sensors`.
- `src/miracle/training/modal_groups.py` exists for grouping channels by
  modality.

**Phase 6 fix:** new driver `scripts/experiments/run_sensor_ablation_v8.sh`
running leave-one-modality-out under the v8 (shortcut-free) config.
Report per-field recoverability (Priority 5) per ablation group.

### Priority 10 — DOE-driven dataset for summer

**Claim (meeting):** The current dataset's repetition enables shortcuts.
A new DOE varying direction × speed × depth × material is needed.

**Evidence (audit):**

- The whole `shortcut_leakage.json` table is the audit-side argument for
  this. Without dataset diversity, even a perfect remediation of the
  truncation and shortcut paths will be measured against a 22-class
  positional baseline.
- Feed rate F is not represented in V7 labels — recoverability claims
  about F require new data.

**Phase 7 fix:** add `scripts/doe/` infrastructure to generate
single-line G-code experiments, build factorial tables, and produce
auto-aligned per-row labels. Persist a DOE specification at
`outputs/decoder20260511/DESIGN_OF_EXPERIMENTS.md`.

### Priority 11 — Manuscript-support outputs

**Claim (meeting):** The paper needs to be trimmed and reframed around
recoverability.

**Evidence:**

- `outputs/decoder20260304/paper/decoder_paper_mdpi.tex` is the current
  manuscript; figures live in `paper/figures/`.
- The paper currently reports 97.9% token accuracy as the headline.
  Given the shortcut analysis above, that number needs context: the
  metadata-only baseline reaches ~94%, so the paper's framing must shift
  to per-field recoverability deltas against the metadata baseline.

**Phase 8 fix:** `scripts/analysis/aggregate_v8_results.py` produces a
single `RESULTS_TABLE.json` plus a `MANUSCRIPT_TABLES/` directory with
the decoder-strategy comparison, shortcut-baseline floor, per-field
recoverability table, and sensor-ablation table.

---

## Recommended changes to the Phase-2+ plan

Based on the audit, **no priorities are dropped** but the framing
shifts:

1. **Phase 2 must raise `max_token_len` substantially.** Hypothetical
   multi-line targets per source file reach 35 tokens. Use `64` as the
   safe default; resolve from NPZ metadata going forward.

2. **Phase 4 should treat shortcut removal as a load-bearing fix, not
   an ablation.** The V7 numbers are largely positional. The "v8 with
   shortcuts" config is for the paper's ablation table; the default
   config should remove the shortcut path entirely.

3. **Phase 5 (retraining) must include the metadata baseline as a
   floor** in every results table. Token accuracy alone is misleading
   on this dataset.

4. **Priority 5 (feed rate, depth) is partially blocked by data.** The
   V7 labels lack F and have very sparse Z. The new DOE in Phase 7 is a
   *prerequisite* for full per-field recoverability claims, not a
   follow-up — though command, axis-presence, and X/Y can still be
   evaluated on V7 data.

5. **Romesh's `romesh_changes/extract_xgboost_importance.py` pattern
   was reused** for the shortcut audit. The same pattern should drive
   the new feature-importance reporting in Phase 8.

---

## Gate G1

This report is the Phase 1 deliverable. **No code outside
`scripts/analysis/` and `outputs/decoder20260511/` has been modified.**
Phase 2 will begin only after user review and sign-off.

Resolved during this session (user sign-off):

1. **V7 per-field ceiling computed.** See `audit/v7_per_field.json` and
   the ceiling table above. The V7 sensor pathway adds real lift on
   axis-presence (+2–13pp) but is matched by metadata-only XGBoost on
   command and numeric values.
2. **Per-row mode dedup policy:** per `(line, window)` pair — one
   training sample for each window that contains the line, full
   256-sample sensor context. ~5× more samples than per `(file, line)`.
   Shortcut removal in Phase 4 handles the within-line position memorization
   risk.
3. **`max_token_len=64` confirmed** as the Phase 2 preprocessing default.
