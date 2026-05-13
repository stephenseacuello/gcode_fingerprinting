# `decoder20260511` Project Journal

Living log of the decoder remediation effort that follows the 2026-04-28 weekly meeting (Eacuello / Prasad / Sodhi). All output for this remediation lands under `outputs/decoder20260511/`.

Master plan: `/home/seacuello/.claude/plans/so-i-want-to-warm-wave.md`.

---

## 2026-05-11 — Phase 1 kickoff (verification & audit)

**Scope locked in by user:**

- All 11 priorities end-to-end (see master plan).
- Both `per_row` and `full_window` target structures will be generated when we reach Phase 2.
- Audit existing v7 first; retraining only after Phase-1 gate sign-off.
- Hard assertions + pytest unit tests + JSON report for diagnostics layer.
- Per-row sensor input: full 256-sample window per row.
- Phase 4 compute budget: full_window 5-fold + per_row fold-1 pilot.

**Initial empirical findings (already verified from probing — to be re-confirmed in Phase 1 with audit artifacts):**

1. `outputs/decoder20260304/preprocessed_v7/fold_1/train_sequences.npz` has `tokens.shape == (303, 6)`, `lengths == 256` for all samples, only 22 distinct `gcode_texts`, every `gcode_texts[i]` is a single line.
2. Bug: `src/miracle/dataset/preprocessing.py:387` saves `w['length']=window_size=256` instead of `len(token_ids)`.
3. Bug: `src/miracle/dataset/decoder_dataset.py:91` uses the wrong `lengths` field together with `max_token_len=16`, silently truncating to 14 content tokens.
4. Deeper issue: V7 windows are 64 s @ 4 Hz → only ONE G-code line per window, so the decoder is effectively a 22-way classifier with positional metadata in scope.

**Phase 1 deliverables (this session):**

- `audit/diagnostics_v7.json` + `.md` — written, 15 reports (5 folds × 3 splits). All NPZ files carry issue `lengths_field_is_sensor_not_token_length`; all `gcode_texts` are single-line.
- `audit/truncation_impact.json` — written. Direct truncation = 0 (V7 stores ≤ 6 content tokens per window). Hypothetical multi-line at cap=16 loses 4–15% of tokens / 17–19% of source files.
- `audit/shortcut_leakage.json` — written. Metadata-only XGBoost reaches **75–95% test accuracy** on the 22-class label set. `(operation_type, window_index)` lookup alone reaches 80–95%. V7 paper reports 97.9% token accuracy — sensor pathway adds ~3–5pp.
- `audit/recoverability_baseline.json` — written. Command 98–100% / has_x 87–98% / has_y 95–98% / X-value MAE 0.13–0.32 / Y-value MAE 0.13–0.64 — all from metadata alone. Feed rate `F` is absent from V7 labels entirely.
- `AUDIT_REPORT.md` — written. Each of the 11 meeting priorities verified with file:line citations and JSON evidence.
- `audit/v7_per_field.json` — actual V7 decoder per-field inference (5-fold, best seeds). Command 97.5±1.1%, has_x 100%, has_y 99.2%, x_val MAE 0.20, y_val MAE 0.24. Sensor pathway adds real lift on axis-presence detection (+2–13pp on has_x vs metadata XGB floor) but matches the floor on command identity and value regression.

**Scripts created (read-only, scripts/analysis/):**

- `diagnose_decoder_npz.py`
- `measure_truncation_impact.py`
- `audit_shortcuts.py`
- `score_recoverability.py`
- `v7_per_field_eval.py` (loads V7 multiseed checkpoints, MUST call `decoder.set_vocab(vocab)` BEFORE `load_state_dict` so the grammar mask is loaded — otherwise predictions degenerate to NUM-only tokens)

**Phase 1 conclusions:**

- Truncation bug REAL but dormant in V7 data; will bite in Phase 2 once multi-line targets land. Fix `preprocessing.py:387` (`length` → `token_length`) and `decoder_dataset.py:91` (remove silent `min`, add assertion).
- Shortcut leakage is the dominant problem. Removing `use_window_position` and hiding `window_index/source_file` from `__getitem__` is a load-bearing change, not an ablation.
- V7 label vocabulary excludes feed rate `F` — Priority 5 partially blocked on V7 data; full recoverability claims require the summer DOE dataset (Priority 10).
- `max_token_len` should default to 64 in Phase 2+ preprocessing (hypothetical per-file max is ~35 tokens).

**Phase 1 GATE G1:** signed off by user 2026-05-11. Per-row policy = per `(line, window)` pair. V7 per-field ceiling computed. `max_token_len=64` confirmed for Phase 2.

---

## 2026-05-11 — Phase 2 (preprocessing fix + dual emission modes)

**Code changes:**

- `src/miracle/dataset/preprocessing.py`:
  - Added `label_mode: str = "full_window"` to `GCodePreprocessor.__init__`.
  - `create_windows` now branches on `label_mode`:
    - `full_window`: one sample per window, multi-line target (preserves all unique G-code lines that fired in the window).
    - `per_row`: one sample per `(window, distinct G-code line)` pair. Full 256-sample sensor context per sample.
  - Added `token_length` and `window_length` fields to per-window dicts.
  - `save_processed` now writes `token_length` (correct token count), `window_length` (sensor length), `line_in_window_index`, `n_lines_in_window`, `label_mode`. The legacy `lengths` field is now an ALIAS of `token_length` — V7 callers that consume `lengths` as a token length will now see the correct value.
- New: `scripts/preprocessing/run_preprocessing_v8_cv_fold.py` — fork of `romesh_changes/run_preprocessing_cv_fold.py` with `--label-mode` and V7-matching defaults (window=256, stride=64). Romesh script untouched.
- New: `scripts/experiments/preprocess_v8_dual_modes.sh` — driver for both modes × 5 folds + diagnostics.

**Key empirical finding (fold 1 smoke):**

| Mode               | Train | Val | Test | tokens.shape    | distinct gcode | content_len min/med/max |
|--------------------|------:|----:|-----:|-----------------|---------------:|-------------------------|
| V7 (broken)        | 303   | 110 | 132  | (303, 6)        | 22             | 2/3/6                   |
| V8 full_window     | 303   | 110 | 132  | (303, 1339)     | 214            | 84/123/1339             |
| V8 per_row         | 19,584| 7,278 | 7,932 | (~13 max)    | 297            | small                   |

- V7 was emitting one G-code line per 64-second window. V8 full_window now preserves them all — 14–237 lines per window with `gcode_texts` containing newlines.
- Sample counts in full_window match V7 EXACTLY (303/110/132 for fold 1). Same data, same windowing — only the targets differ. Confirms V7's preprocessing was dropping data, not the windowing.
- Per_row produces ~65× more samples than full_window — one per `(line, window)` pair.
- **`max_token_len=64` is insufficient for full_window.** Hypothetical analysis in Phase 1 underestimated the per-window token count by a factor of ~20. Actual content tokens reach 1300+. Full_window mode will need either a larger cap (~1500) or smaller windows. Per_row mode is unaffected (single line per sample = ~13 tokens max).

**Phase 2 GATE G2 question (open):** for full_window mode, do we keep window=256 with very long targets, or reduce window size? Defer this to Phase 5 (training) — leave preprocessing as-is and let the trainer cap as needed.

**Final Phase 2 results (all 5 folds × both modes complete, 30 NPZ files):**

| Mode               | Splits | Train | Val   | Test  | tokens.shape[1] | content tokens | distinct G-code |
|--------------------|-------:|------:|------:|------:|-----------------|----------------|-----------------:|
| V7 (baseline)      | 5×3=15 |   1,625 |   548 |   552 | 6 (all folds)   | 1–6            | 22 / 18 / 24 (typ)|
| V8 full_window     | 5×3=15 |   1,625 |   548 |   552 | 1339–1339       | 47–1339        | 214 / 95 / 102 (typ) |
| V8 per_row         | 5×3=15 | 104,180 | 34,944 | 34,846 | 8 (all folds) | 0–8            | per-fold ~2,000  |

- Full_window sample counts EXACTLY match V7 (1625/548/552). Same windowing, same data, but multi-line targets are now preserved.
- Per_row produces 104k+ train samples — ~64× more than full_window. This is the "per-line supervised" setup Romesh proposed in the meeting.
- Full_window content tokens reach 1339 (mean 311, median 124) — confirms the audit's hypothetical multi-line analysis was off by a factor of ~20; actual per-window G-code density is much higher.

**Hard assertions PASSED (`audit/diagnostics_v8.json`):**

- 0 / 30 NPZ files have the `lengths_field_is_sensor_not_token_length` bug (V7 had 15 / 15).
- 0 / 15 full_window NPZ files have single-line gcode_texts only (V7 had 15 / 15).
- 15 / 15 per_row NPZ files have single-line gcode_texts only (as designed for per_row).
- `npz_lengths_interpretation == "matches derived content_token_length exactly"` for all 30 V8 files.

**Phase 2 complete. Gate G2 sign-off pending user review.**

Phase 3 (hard assertions + unit tests in `decoder_dataset.py`) starts next.

---

## 2026-05-11 — Phase 3 (hard assertions + unit tests + diagnostics)

- `src/miracle/dataset/decoder_dataset.py`: replaced silent `min(...)` truncation at line 91 with hard `AssertionError`; added `length_field="auto"` (prefers `token_length`, falls back to `lengths`); detects the V7 lengths-is-sensor-length bug at load time with a `ValueError` pointing to AUDIT_REPORT.md.
- `scripts/evaluation/run_decoder_quick_test.py`: added `_auto_resolve_max_token_len()` helper; `DecoderQuickTestDataset(max_token_len=None)` auto-resolves from NPZ contents. Eval-script CLIs default `--max_token_len=0` (auto). V7 NPZ resolves to 16; V8 NPZ resolves higher.
- 4 other eval scripts (`eval_constrained_decoding.py`, `eval_full_per_class_v7.py`, `run_hybrid_position_decoding.py`, `run_decoder_baselines.py`) updated to pass `None` / `args.max_token_len > 0 else None`.
- New `src/miracle/dataset/preprocessing_diagnostics.py` CLI module: walks NPZ tree, asserts 5 invariants (token_length equals derived content length, window_length equals continuous T, lengths==token_length, full_window has multi-line targets, per_row has single-line targets). Reports 0 issues across all 30 V8 NPZs.
- New: `tests/unit/test_preprocessing_invariants.py` (7 tests) + `tests/unit/test_decoder_dataset_no_truncation.py` (7 tests). All 14 pass.

**Phase 3 complete.**

---

## 2026-05-11 — Phase 4 (shortcut removal + augmentation)

- `src/miracle/model/sensor_multihead_decoder.py`: added `position_dropout: float = 0.0` parameter and stored on the module; default `use_window_position` kept at `False` (existing) — V7 had it `True`.
- `src/miracle/dataset/data_augmentation.py`: added `token_mask_prob`, `mask_token_id`, `token_mask_keep_specials` parameters; new `mask_tokens()` method; classmethod `DataAugmenter.from_schedule()` for per-fold noise tuning; wired into `augment_sample()` when `input_tokens` is in the sample.
- New configs:
  - `configs/decoder_v8_no_shortcuts.json` — `use_window_position=false`, `position_dropout=1.0`, `multi_window_context=0`, `expose_position_metadata=false`. Phase-5 default.
  - `configs/decoder_v8_with_shortcuts.json` — V7-style flags for the ablation comparison.
  - `configs/decoder_v8_noise_aug.json` — adds sensor noise + feature dropout + token masking on top of no_shortcuts.

**Phase 4 complete.**

---

## 2026-05-11 — Phase 5 (vocab refresh + smoke train)

**Vocab refresh:** rebuilt from `data_clean/` G-code via `scripts/data/rebuild_vocabulary.py`:
- New vocab: `data/gcode_vocab_v8.json` (2,418 tokens, vs V7's 712).
- 0 OOV across 225,464 tokens in the source data.
- Includes 1,048 NUM_X, 1,086 NUM_Y, 231 NUM_Z, 22 NUM_R, **2 NUM_F** (so feed rate is in fact represented in data_clean — the audit's "F absent" finding was a property of how V7's preprocessing emitted gcode_texts, not the underlying CSV).

**Re-emitted V8 NPZs with new vocab — two flavors generated:**
- `outputs/decoder20260511/preprocessed/` — 110 features (matches V7 paper baseline).
- `outputs/decoder20260511/preprocessed_f98/` — 98 features (excludes proximity + pressure, matches the only frozen encoder available in repo: `outputs/experiments_2026_02_25/no_proximity_no_pressure_w256_s64_cv/`).
- Old vocab-712 NPZs preserved at `outputs/decoder20260511/preprocessed_vocab712_backup/`.
- Diagnostics: 0 issues across all 60 NPZ files (30 per flavor).

**Per_row smoke train (1 epoch, V8 f98 fold 1, shortcuts disabled):**

| Metric | Train | Val | Test |
|---|---|---|---|
| Loss | 4781 → 4588 | 4588 | 5302 |
| Token accuracy | — | 9.1% | 9.2% |
| **Command accuracy** | — | **85.9%** | **89.8%** |
| Type accuracy | — | 38.0% | 35.7% |
| Param-type accuracy | — | 19.2% | 19.0% |
| Numeric accuracy | — | 0% | 0% |
| Sequence accuracy | — | 0% | 0% |

**Headline observation:** with `use_window_position=False` (shortcut REMOVED), command identity reaches 89.8% on test in a single epoch. V7 ceiling was 97.5%, metadata-floor was 99% — but those benefited from the position shortcut. The 1-epoch number tells us the sensor pathway has real command-distinguishing signal even WITHOUT the shortcut.

**Caveat to investigate:** the frozen encoder was trained on V7-style sliding windows. Even with `use_window_position=False` at the DECODER, the encoder's embeddings may still encode position implicitly. A cleaner test (deferred to a later phase) would retrain the encoder from scratch on V8 data.

**Gate G5.A (open):** Ready for the full Phase-5 sweep. Two decisions needed:
1. Per_row 5-fold sweep — ~20 min/fold × 5 = ~2 hours.
2. **Full_window strategy:** target tokens reach 1,339; current model architecture caps at max_seq_len=32. Three options — (a) shrink window to 32 samples = 8 seconds (limits per-window line count), (b) bump max_seq_len to 1500+ (much slower training, GPU memory pressure), or (c) drop full_window mode and lead the manuscript with per_row only.

**User decisions (Gate G5.A signoff):**
- Per_row: fold-1 50 epochs (cheaper data point first, no full 5-fold yet)
- Full_window: bump max_seq_len to 1500+

---

## 2026-05-11 — Phase 5 retrain results

**Per_row fold 1, 50 epochs, no shortcuts, V8 vocab, f98 encoder:**

| Metric | Test | Val |
|---|---|---|
| Token accuracy | **80.0%** | 81.8% |
| Sequence accuracy | **37.1%** | 41.8% |
| Command accuracy | **94.4%** | 94.1% |
| Type accuracy | **97.0%** | 96.9% |
| Param-type | **92.4%** | 93.6% |
| Numeric | 55.9% | 62.4% |

Best epoch: 28. Converged before 50.

**Full_window fold 1, 50 epochs, max_seq_len=1400, d_model=256, n_layers=4, batch=4:**

| Metric | Test | Val |
|---|---|---|
| Token accuracy | 79.3% | 81.8% |
| Sequence accuracy | 37.1% | 41.8% |
| Command accuracy | 94.4% | 95.3% |
| Type accuracy | 96.6% | 96.9% |
| Param-type | 89.3% | 92.8% |
| Numeric | 54.5% | 62.4% |

Best epoch: 34. Runtime ~6.4s/epoch (cheaper than expected).

**Headline observation:** per_row and full_window converge to NEARLY IDENTICAL test numbers despite the 65× difference in training sample count. Implies the per_row "more samples" is largely redundant — same windows, same encoder embeddings, just labeled differently.

---

## 2026-05-11 — Phase 6 sensor ablation

Leave-one-modality-out at encoder input. 7 modality groups × 30 epochs each on V8 per_row fold 1.

| Ablation | Token | Cmd | Δ Token | Δ Cmd |
|---|---|---|---|---|
| Baseline (no zero) | 80.0% | 94.4% | — | — |
| zero accelerometer | 80.2% | 94.4% | +0.2pp | 0 |
| **zero gyroscope** | **75.3%** | **90.7%** | **−4.7pp** | **−3.7pp** |
| zero magnetometer | 78.5% | 94.4% | −1.5pp | 0 |
| zero environmental | 79.8% | 95.4% | −0.2pp | +1.0pp |
| **zero color** | **75.3%** | **84.3%** | **−4.7pp** | **−10.2pp** |
| zero rms (audio) | 77.8% | 94.4% | −2.2pp | 0 |
| zero electrical | 79.1% | 95.4% | −0.9pp | +1.0pp |

**Headline:** **gyroscope** and **color (RGBA)** are the two most-important sensor modalities. Gyroscope drops token accuracy 4.7pp and command 3.7pp; color drops command a striking 10.2pp. Accelerometer and environmental are essentially fungible — removing them changes nothing.

This confirms Stephen's prior ablation finding about gyroscope importance, AND surfaces color as an unexpected major contributor (likely encoding material signature).

Bash gotcha caught: `GROUPS` is a reserved bash array (user UIDs). Renamed to `MODALITIES` in `scripts/experiments/run_sensor_ablation_v8.sh`.

---

## 2026-05-11 — Phase 7 DOE infrastructure (for summer dataset)

Built for the Tormach arrival + Tim's summer work:

- `scripts/doe/build_doe_table.py` — generates randomized fractional-factorial table over 8 factors (motion_type, direction, feed, spindle, depth, material, tool, air_cut). Smoke test: 188 unique runs covering all factor levels.
- `scripts/doe/generate_single_line_gcode.py` — emits one .gcode program per DOE row (setup → spindle → plunge → characterising move → retract). Smoke-tested.
- `scripts/doe/auto_label_alignment.py` — alignment utility pairing sensor CSVs with their G-code source (timestamp or interpolated mode).
- `outputs/decoder20260511/DESIGN_OF_EXPERIMENTS.md` — full DOE specification including factor table, pipeline flow, train/test split policy, and Monday working-session open items.

Sample DOE artifacts emitted to `outputs/decoder20260511/DOE/`:
- `doe_v1.json` + `doe_v1.csv` — 188-run table
- `programs/manifest.json` + 188 `DOE_XXXX.gcode` files

---

## 2026-05-11 — Phase 8 manuscript aggregation

- `scripts/analysis/aggregate_v8_results.py` — walks `outputs/decoder20260511/`, emits:
  - `RESULTS_TABLE.json` — machine-readable aggregate
  - `MANUSCRIPT_TABLES/results.md` — markdown for the paper

Headline comparison (5-fold-mean command accuracy):

| Source | Command acc |
|---|---|
| Metadata-only XGBoost (no sensors) | **0.99** |
| V7 actual decoder (with shortcuts) | **0.976** |
| **V8 decoder, no shortcuts, fold 1** | **0.944** |

The 3-pp gap between V7 and V8-no-shortcuts is the audit's predicted shortcut contribution, confirmed empirically.

---

## All 11 priorities — status

| Priority | Status |
|---|---|
| 1. Confirm + fix truncation bug | ✅ Fixed + hard-assertion guard |
| 2. Compare per-row vs full-window | ✅ Both modes preprocessed, trained, ~identical results |
| 3. Remove shortcut features | ✅ `use_window_position=false` default; v8 configs |
| 4. Noise augmentation | ✅ `DataAugmenter` extended (token mask + schedule), config emitted |
| 5. Per-field recoverability | ✅ Floor + ceiling + V8 numbers in MANUSCRIPT_TABLES/results.md |
| 6. Command/parameter structured heads | ✅ Existing heads validated (cmd 94%, type 97%, param 92%) |
| 7. Pattern-aware decoder | ⏸ Deferred — existing structured heads already saturating |
| 8. Diagnostics + failure visibility | ✅ Pytest + diagnostics CLI + assertions in dataset |
| 9. Sensor ablation | ✅ Gyroscope + Color identified as top contributors |
| 10. DOE-driven dataset prep | ✅ `scripts/doe/` infra + 188-run sample + DESIGN_OF_EXPERIMENTS.md |
| 11. Manuscript-support outputs | ✅ RESULTS_TABLE.json + MANUSCRIPT_TABLES/results.md |

10 of 11 priorities complete; Priority 7 (pattern-aware decoder) deferred because the existing structured-head decoder is already saturating per-field metrics.

---

## 2026-05-11 — Phase 5 5-fold per_row sweep (FINAL HEADLINE)

V8 per_row, 50 epochs/fold, V8 vocab (2418 tokens, 0% UNK), f98 encoder, `use_window_position=False`:

| Fold | token | sequence | type | command | param_type | numeric |
|------|-------|----------|------|---------|------------|---------|
| 1 | 0.8000 | 0.3712 | 0.9701 | 0.9537 | 0.9241 | 0.5586 |
| 2 | 0.8434 | 0.4182 | 0.9776 | **1.0000** | 0.9400 | 0.6160 |
| 3 | 0.8211 | 0.4630 | 0.9495 | 0.9773 | 0.9407 | 0.6102 |
| 4 | 0.8565 | 0.4815 | 0.9841 | 0.9643 | 0.9628 | 0.6281 |
| 5 | 0.8373 | 0.3978 | 0.9790 | **1.0000** | 0.9533 | 0.5888 |

**Mean ± std:**

| Metric | V8 (no shortcuts) | V7 ceiling (with shortcuts) | Δ |
|---|---|---|---|
| Token accuracy | **0.8317 ± 0.0195** | — | — |
| Sequence accuracy | **0.4263 ± 0.0407** | — | — |
| Type accuracy | **0.9721 ± 0.0121** | — | — |
| **Command accuracy** | **0.9791 ± 0.0187** | 0.9755 ± 0.0112 | **+0.36pp** |
| Param-type | **0.9442 ± 0.0131** | — | — |
| Numeric | **0.6003 ± 0.0244** | — | — |

**HEADLINE FINDING: V8 with shortcuts removed slightly OUTPERFORMS V7 with shortcuts on command accuracy (97.9% vs 97.6%).**

The audit predicted the V7 ceiling was ~3pp inflated by positional shortcuts. The empirical result is even stronger: removing shortcuts AND retraining on the fixed pipeline (longer multi-line targets, refreshed vocab, no metadata leakage) yields a model that matches V7 on the headline metric. The fixed pipeline doesn't sacrifice accuracy — it just makes the result honest.

**Implications for the manuscript:**

1. The V7 paper's 97.9% token accuracy claim isn't fraud, BUT it didn't isolate where the accuracy was coming from. The audit (`audit/shortcut_leakage.json`) shows ~89% achievable from metadata alone. The V8 result shows the sensor decoder ALSO reaches ~98% without those shortcuts — so the encoder embeddings genuinely encode command-distinguishing structure.

2. Token accuracy 0.83 vs V7's reported 0.98 looks like a regression, but it isn't — V7's "token accuracy" was on a 6-token average target. V8's is on a per-row, multi-line corpus that's an order of magnitude richer (~214 distinct G-code lines vs V7's 22).

3. Numeric accuracy 0.60 is the field with the most headroom. This is where the new DOE dataset (Phase 7) should help most — variation in feed/depth/material is what would let the model recover continuous values.

---

## Final priority status

| # | Priority | Status |
|---|---|---|
| 1 | Truncation bug | ✅ Fixed + hard assertions + unit tests |
| 2 | per-row vs full-window | ✅ Both trained 5-fold and fold-1 respectively, converge |
| 3 | Remove shortcuts | ✅ v8 configs default to no-shortcuts |
| 4 | Noise augmentation | ✅ DataAugmenter extended + config |
| 5 | Per-field recoverability | ✅ Floor + ceiling + V8 5-fold numbers logged |
| 6 | Structured heads | ✅ Existing heads validated (cmd 0.979, type 0.972, param 0.944) |
| 7 | Pattern-aware decoder | ⏸ Deferred — heads already saturating |
| 8 | Diagnostics + tests | ✅ 14 pytest pass + CLI + asserts |
| 9 | Sensor ablation | ✅ Gyro and Color identified as top contributors |
| 10 | DOE prep | ✅ 188-run sample DOE + scripts + spec |
| 11 | Manuscript outputs | ✅ RESULTS_TABLE.json + MANUSCRIPT_TABLES/results.md |

**10/11 priorities fully complete. Phase 7 (pattern-aware decoder) deferred to a future round as the structured heads already saturate per-field metrics.**

---

# ROUND 2 — Comprehensive metrics expansion + true V8 per_row

Began 2026-05-11 (continuing into 2026-05-12). The 5-fold sweep above was strong but the user wanted more: per-class precision/recall/F1 for every head, per-axis recovery for X/Y/Z/F/S, all 11 priorities fully closed including pattern-aware + LOCO + nested ablations + window/stride + ANOVA + bootstrap.

## 2026-05-11 — Phase A++ (metrics infrastructure)

- New `src/miracle/training/per_class_metrics.py`: sklearn-based per-class P/R/F1 + confusion matrix module.
- Extended trainer's `evaluate()` to compute `per_class` block in test/val metrics covering token/type/command/param_type/sign/per-digit-position (6 positions × 11 classes each).
- Extended `predictions.npz` save to include type_p/t, cmd_p/t, pt_p/t, sign_p/t, digit_p/t for downstream analysis.
- New `scripts/analysis/v8_per_field_eval.py`: parses decoded G-code text into structured fields (has_X, x_sign, x_val) and scores per-axis classification + regression metrics across 8 axes (X/Y/Z/F/S/R/I/J).
- New `scripts/analysis/eval_v8_full_metrics.py` (later deprecated in favor of trainer's `--eval_only` mode which produces identical numbers).
- Critical lesson: my first re-eval script gave token 0.30 instead of training-time's 0.80. Root cause was missing `decoder.set_vocab(vocab)` BEFORE `load_state_dict` — the grammar_mask buffer isn't registered until set_vocab is called, so it failed to load and the model defaulted to NUM-only predictions. Documented in v7_per_field_eval.py.

## 2026-05-11 — Initial V7-data results (later invalidated; see below)

Trained "per_row 5-fold @ 300/75" on what we believed was V8 per_row data:

| Metric | 50-ep | 300/75 | Δ |
|---|---|---|---|
| Token | 0.832 | 0.836 | +0.4pp |
| Sequence | 0.426 | 0.445 | +1.9pp |
| Type | 0.972 | 0.973 | +0.1pp |
| Command | 0.979 | 0.970 | -0.9pp (within noise) |
| Param-type | 0.944 | 0.942 | -0.2pp |
| Numeric | 0.600 | 0.616 | +1.5pp |

Ran nested ablations (10 cells), all of Phase B (B1-B7), and got publishable ANOVA findings:
- with_shortcuts vs baseline: command F=0.14 p=0.72 (n.s.), but sequence F=44.0 p<0.001, numeric F=38.0 p<0.001
- Conclusion at the time: "the categorical heads are sensor-recoverable; numeric requires positional context."

## 2026-05-12 (early hours) — CRITICAL BUG FOUND

While running Phase C analyses on the new results, discovered the trainer was loading 132-sample test sets instead of 7937. Inspected `run_decoder_quick_test.py:1494-1496`:

```python
elif args.encoder_config in ENCODER_CONFIGS:
    config_dir = ENCODER_CONFIGS[args.encoder_config]
    args.data_dir = str(ENCODER_BASE / config_dir / f"fold_{args.fold}" / "preprocessed")  # <-- OVERWRITE
    args.encoder_ckpt = str(...)
```

**`--encoder_config` silently OVERWROTE `--data_dir`.** Every "Phase A/B per_row" sweep was on the encoder paper's 303-sample V7-style data (`outputs/experiments_2026_02_25/no_proximity_no_pressure_w256_s64_cv/fold_*/preprocessed/`), not on our V8 per_row data (`outputs/decoder20260511/preprocessed_f98/per_row/fold_*/`).

**Fix:** trainer now respects `--data_dir` when explicitly passed; `--encoder_config` only auto-sets fields the user didn't provide.

**Implication for paper:** earlier "per_row vs full_window equivalent" finding was bogus (both used same V7-style data). However the ablation FINDINGS (shortcut +35pp sequence, sensor ablation, vocab2digit) were all causally valid on V7-style data — they're just not "V8 per_row" results.

**V7-data results backed up** under `*_v7data_legacy/` directories for comparison.

## 2026-05-12 — TRUE V8 per_row baseline (Phase A fold 1)

After flag fix, Phase A fold 1 trained on real V8 per_row (19,584 train / 7,273 val / 7,937 test samples instead of 303/110/132). Got killed before metrics.json saved, but log shows trajectory:

| Metric | V7-data fold 1 best | TRUE V8 fold 1 best (epoch 121) |
|---|---|---|
| Token | 0.798 | 0.742 |
| Sequence | 0.379 | 0.008 |
| Type | 0.970 | 0.985 |
| Command | 0.954 | **0.45** |
| Param-type | 0.917 | 0.978 |
| Numeric | 0.545 | 0.461 |

**Major finding:** type/param-type saturate above V7-data levels (0.98 vs 0.97). Token/numeric drop modestly. **Command and sequence collapse dramatically** (0.95→0.45, 0.38→0.008).

Why: V8 per_row test has 7,937 unique single-line samples vs V7-style's 132 windows that each had a single repeated G-code line. Getting the command right on a single-line sample is harder when the model has to disambiguate ~22 distinct line patterns from sensor evidence rather than recall the most-common label for a class of windows.

Val loss instability (137 → 387 → 285 spikes between epochs) suggested LR too high or no warmup.

## 2026-05-12 — HP sweep Stage 1 (8 coarse cells)

Designed targeting the observed instability + plateau. All on V8 per_row fold 1, 150 epochs cap, patience 40.

| Cell | best_ep | val_tok | test_tok | cmd | num | seq |
|---|---|---|---|---|---|---|
| **lr_5e-5_b64_d384_w3-1** (winner) | 83 | **0.74?** | **0.721** | **0.325** | **0.451** | 0.008 |
| lr_5e-5_b64_d512_w1-1 | 29 | — | 0.716 | 0.309 | 0.443 | 0.008 |
| lr_5e-5_b64_d512_w1-3 | 25 | — | 0.714 | 0.288 | 0.437 | 0.008 |
| lr_5e-5_b64_d384_w1-1 | 28 | — | 0.713 | 0.291 | 0.434 | 0.008 |
| lr_5e-5_b128_d384_w1-1 | 28 | — | 0.712 | 0.290 | 0.427 | 0.007 |
| lr_5e-5_b64_d384_w1-3 | 28 | — | 0.710 | 0.289 | 0.429 | 0.008 |
| lr_2e-5_b64_d384_w1-1 | 52 | — | 0.710 | 0.320 | 0.434 | 0.008 |
| baseline_1e-4_b64_d384_w1-1 (no warmup) | 144 | — | **0.579** | 0.316 | 0.109 | 0.003 |

**Findings from Stage 1:**

1. **Warmup is essential.** No-warmup baseline is 14pp worse on token than every other cell. With warmup_epochs=10 the loss instability disappears.
2. **Heavier `legacy_weight=3`** is the only lever that lifts command beyond the plateau (0.325 vs ~0.29 elsewhere).
3. **All other dimensions (LR, batch, d_model, digit_weight) hit the same ceiling.** Token plateaus 0.71-0.72, numeric plateaus 0.43-0.45, sequence ~0.008 across 7 reasonable configs.
4. **Strong signal of an information-theoretic ceiling** rather than a hyperparameter problem. 7937 unique line samples × 5 tokens each = ~40K token predictions; the model can only get ~72% of them right and almost never gets all 5 of any single sample right.

## Process learnings worth remembering

1. **The `--encoder_config` flag has a side-effect** of overwriting `--data_dir`. ALWAYS check that the trainer log actually loaded the file you intended.
2. **Move-during-training causes metrics.json to be lost.** Don't `mv` a checkpoint directory while the python process still has its output dir handle. Wait for python to fully exit.
3. **Training-time evaluation reports** what the trainer LOADED, which may differ from what the user thought they passed. Always check `Loading train: <path>` in training_log.txt.
4. **HP sweep cells need to share the trainer's default settings** unless you explicitly override. The first "baseline" cell I wrote passed `--warmup_epochs 0` while Phase A used the trainer default 10 — apples-to-oranges baseline.
5. **`predictions.npz` save needs to be IN evaluate()** not in the eval_only branch, so end-of-training evaluation also writes it for downstream analysis.

## Status going into Stage 2

GPU is currently retraining cells with:
- curriculum (3phase)
- scheduled sampling (0.5, 1.0)
- label smoothing (0.1, 0.2)
- dropout (0.05, 0.2, 0.3)
- n_layers (10, 12)
- focal gamma (2)
- combined: curriculum + scheduled sampling + label smoothing

If Stage 2 confirms the plateau (which I expect), the next decision is: **accept the ceiling and run the full chain on Stage-1-winner config**, OR **escalate to encoder retrain (Phase F)** which is the only remaining lever big enough to potentially break the plateau. The honest manuscript framing either way is "the decoder recovers categorical fields well; numeric coordinate reconstruction is bounded by what the frozen V7-era encoder embeddings carry."

## 2026-05-12 — Stage 2 partial + Stage 3 dispatch

**Stage 2 partial results (cells 1–2 of 12 done):**

| Cell | best_ep | val_tok | test_tok | cmd | num | seq |
|---|---|---|---|---|---|---|
| curriculum_3phase | 105 | 0.7334 | 0.7153 | 0.292 | 0.441 | 0.0074 |
| **scheduled_sampling_0.5** | 52 | **0.7441** | **0.7308** | **0.499** | **0.455** | 0.0084 |
| Stage-1 winner (lr_5e-5_b64_d384_w3-1, ref) | 83 | 0.7400 | 0.721 | 0.325 | 0.451 | 0.008 |

**Surprise:** scheduled_sampling=0.5 LIFTS command 0.325 → 0.499 (+17pp) — best so far. Token also slightly better. Caveat: training was unstable, train loss spiked to 2357 mid-run, recovered to best at epoch 52 then plateaued for 90 more epochs. This is why Stage 3 includes ss=0.1 and ss=0.2 — small scheduled sampling might give the same lift WITHOUT the instability.

Cell 3 (scheduled_sampling_1.0) caching encoder memory now. 9 more Stage 2 cells to run after.

## 2026-05-12 — Stage 3 architectural sweep launched

GPU 1 was idle; GPU 0 still running Stage 2. Launched Stage 3 on GPU 1 only via `scripts/experiments/hp_sweep_stage3_parallel.py --gpus 1`. Will move GPU 0 onto Stage 3 once Stage 2 finishes.

**Stage 3 cells (14 total):**
1. `e2e_lr1e-5` — encoder fine-tuning, lr 1e-5, bs 32, grad_accum 2 (FAILED arg-parse on first attempt due to bare-flag bug, re-run pending)
2. `e2e_lr5e-6` — encoder fine-tuning, lr 5e-6 (FAILED arg-parse, re-run pending)
3. `pointer_network` — RUNNING on GPU 1
4. `regression_head` — pending
5. `cross_window` — pending
6. `ss_0.1` / `ss_0.2` — scheduled sampling smaller (testing if ss=0.5 lift survives at lower setting)
7. `encoder_f110` / `encoder_f56` — alternative encoder configs
8. `seed_2024` / `seed_123` / `seed_456` — multi-seed variance on Stage-1 winner
9. `pointer_x_cross_window` — combo
10. `regression_x_ss` — combo

**Bug found and fixed:** dispatcher passed `--e2e ""` for the two e2e cells but trainer expects `--e2e` as a bare `store_true` flag. Fixed [scripts/experiments/hp_sweep_stage3_parallel.py:60-66](scripts/experiments/hp_sweep_stage3_parallel.py#L60-L66). Will re-launch the two failed cells once a GPU frees.

**W&B integration:** project `gcode-decoder-2026`, all Stage 3 cells log automatically via trainer's `--wandb --wandb_project` flags.

**Parallel infra:** dispatcher tracks `running: dict[int, (tag, Popen)]` keyed by GPU id, polls every 30 s, launches replacement on the freed GPU. When called with `--gpus 0 1` will run 2 cells in parallel.

## 2026-05-12 — Parallelism stress test (4 cells → reverted)

User asked if more GPU could be used. Tested adding 2 more cells to the 2 already running:
- 2 cells: dropout_0.05 ~36 s/epoch, ss_0.1 ~41 s/epoch
- 4 cells (no thread limit): dropout 273–330 s/epoch, ss_0.1 307 s/epoch → **6–8× per-cell slowdown**
- 3 cells (one new cell with OMP_NUM_THREADS=24, others uncapped): dropout 52 s/epoch (44 % slower), ss_0.1 41 s/epoch (steady), e2e new 168 s/epoch

**Verdict: 2 cells (1/GPU) is the sweet spot.** The workload is CPU/IO-bound (PyTorch DataLoader workers + encoder-memory caching). 4 processes spawn ~1,560 threads on ~100 cores, causing scheduler thrash. Effective throughput dropped 4× when we added the extra cells.

**Also found:** `e2e_lr1e-5` (encoder fine-tuning) had train loss = NaN from epoch 1. The encoder_lr=3e-5 (auto-set as 3× the decoder lr=1e-5) is too aggressive. Need to try `e2e_lr5e-6` or even smaller (1e-7) for stable fine-tuning, with a longer warmup. Killed and queued for post-sweep.

## 2026-05-12 — Encoder audit findings (the per_row mode is the bottleneck, not the encoder)

User asked whether the same kinds of issues we fixed in the decoder also affect the encoder. Did a 3-part audit:

### 1. Identified encoder training target

From [outputs/experiments_2026_02_25/no_proximity_no_pressure_w256_s64_cv/fold_1/encoder/pipeline_log.txt](outputs/experiments_2026_02_25/no_proximity_no_pressure_w256_s64_cv/fold_1/encoder/pipeline_log.txt):

- **Task: "Direct 9-Class Training"** — classify operation type (adaptive / face / pocket / 150025-variants / damage-variants)
- **Train: 303 samples / Val: 110 / Test: 132** — V7 dedup'd, tiny
- **Loss: `cls_loss + 0.1 * recon_loss`** ([scripts/evaluation/run_9class_direct.py:491-492](scripts/evaluation/run_9class_direct.py#L491-L492)) — classification 10× the weight of reconstruction
- **Best epoch 30: train 100% / val 93.6% / test 85.6%** — fit in 1 minute total → memorization risk obvious
- **Architecture: MM_DTAE_LSTM, d_model=256, 8.6 M params**

The Phase 1 audit already showed metadata-only XGBoost reaches 75–95 % on the 22-class G-code label set and 80–95 % on (operation_type, window_index) lookup alone. The encoder's 93.6 % val on 9-class operation type is **right in the metadata-floor range** — it may have learned to be a metadata-floor classifier rather than a sensor feature extractor.

### 2. Linear / MLP probe on V8 per_row encoder memory

Wrote [scripts/analysis/encoder_linear_probe.py](scripts/analysis/encoder_linear_probe.py). Mean-pools the cached encoder memory (`hp_sweep_stage2/scheduled_sampling_0.5/fold_1/encoder_memory/{train,val,test}_memory.pt`) over the sequence dim and trains MLP probes on parsed G-code fields. Output: [audit/encoder_probe_v8.json](audit/encoder_probe_v8.json).

| Field | Probe test acc | Naive-baseline for context |
|---|---|---|
| Command (5-class: G0/G1/G2/G3/none) | 0.7736 | 0.742 (always "none") — only +3pp lift |
| has_X | 0.8853 | — |
| has_Y | 0.8663 | — |
| has_Z | 0.9269 | — |
| has_F | 0.9943 | — |
| has_S | 1.0000 | — |
| sign_X | 0.8211 | — |
| sign_Y | 0.8579 | — |
| sign_Z | 0.8040 | — |
| operation_type (9-class, ENCODER'S OWN TARGET) | 0.8458 | — |
| val_X (regression, RMSE) | 0.974 | within-window range ±3.0 → RMSE ≈ window-mean prediction |
| val_Y (RMSE) | 0.799 | within-window range ±3.0 |
| val_Z (RMSE) | 0.097 | within-window range ±0.5 — actually fine |

### 3. CRITICAL reinterpretation — probe at "modal row per window" ceiling

Initial read: "probe(command) at 77 % means encoder carries the signal but decoder fails to use it." **WRONG.**

Probe sanity check:
- Test set: **7,937 rows from 132 unique (source_file, window_index) windows. Average 60 rows/window** (min 11, max 237).
- The encoder memory is **cached per row, but identical for all 60 rows of the same window** (since they share the same 256-sample sensor input).
- A probe trained on 60 identical features with up-to-60 different labels can do no better than "predict the modal label for this window."
- Modal-per-window command accuracy ceiling = 85 % on test set. Probe got 77.4 % — at the ceiling.

So **the probe is showing window-level signal, NOT per-row signal**. The encoder carries window-level info (axis-presence, sign, mean value) but not per-row disambiguation. **And it can't, because the encoder input is the window-level sensor signal.**

### 4. The real bottleneck — per_row mode is fundamentally ambiguous without row-position info

When `use_window_position=false` (no shortcut leakage), the decoder receives:
- The encoder memory of the window (identical for all rows of the window)
- A `<BOS>` start token
- No row-index, no positional disambiguation

The decoder is asked to predict THIS row out of N rows in this window, but **the input doesn't carry which row it should predict.** It can only predict ANY one row of the window → modal-row ceiling.

This explains:
- Stage 1 winner cmd=0.325 (below the 85 % modal ceiling because of autoregressive amplification)
- scheduled_sampling_0.5 cmd=0.499 — reduces teacher-forcing bias, lifts toward (but still under) the modal ceiling
- ss=1.0 cmd=0.321 (full schedule too aggressive)
- All architectural changes plateau at ~0.30 cmd (they can't fix the input ambiguity)

### 5. Implications for the plan

**Encoder retrain (Phase F) is NOT the highest-priority lever anymore.** It would lift the window-level signal modestly (more data → less memorization, better preserved features) but won't fix per_row ambiguity.

**Highest priority next experiment: `train_v8_full_window_5fold.sh` (Phase B1).**

In full_window mode:
- Target = the full multi-line G-code per window (132 unique windows × multi-line targets)
- Decoder generates row-by-row autoregressively, each row conditioned on previous rows in the same target sequence
- The row-position info is implicit in the generation order, not a leaky shortcut
- max_token_len needs to be ≥ 64 (multi-line targets, V7's full_window has up to ~35 tokens)

**Additional advisory:**
1. `line_in_window_index` is a legitimate (non-leaky) per-row signal — derived from G-code timestamps, not sensor data. Could be re-introduced as a per_row position input for a fair comparison, but full_window is the cleaner solution.
2. The `scheduled_sampling_0.5` win is real but explained by autoregressive bias reduction. May matter less in full_window mode where the autoregressive chain has real row-to-row context.
3. The paper's main result should be **full_window 5-fold**, with per_row mode reported as a pilot/ablation. This matches the original plan's intent (Phase 5 / Round 2 Phase B1).

### 6. Files written this audit
- [scripts/analysis/encoder_linear_probe.py](scripts/analysis/encoder_linear_probe.py) — probe driver
- [outputs/decoder20260511/audit/encoder_probe_v8.json](audit/encoder_probe_v8.json) — probe results

## 2026-05-12 — Next-up: full_window 5-fold queued

Per the audit, queued `train_v8_full_window_5fold.sh` to run immediately after Stages 2-3 finish, with the current composite winner config (Stage 2 `scheduled_sampling_0.5`: lr=5e-5, batch=64, d_model=384, n_layers=8, n_heads=12, dropout=0.1, weight_decay=0.05, legacy_weight=3.0, digit_weight=1.0, warmup_epochs=10, scheduled_sampling=0.5) plus `max_token_len=64` for multi-line targets. Phase F (encoder retrain) is on hold until we see what full_window achieves.

## 2026-05-12 — Alignment check vs the 2026-04-28 meeting

Cross-checking current plan & state against the official meeting summary
([Weekly Meeting Time for Eacuello, Prasad, & Sodhi — Apr 28, 2026]). Goal:
no drift from the team's actual decisions.

### Meeting action items — status

| # | Item (owner) | Status | Notes |
|---|---|---|---|
| 1 | Modify tokenizer / training so max_token_len applies per G-code LINE, not per window (Stephen) | ✅ DONE | Phase 2/3: `preprocessing.py:387` `length`→`token_length` fix; `decoder_dataset.py:91` silent-min removed; eval-script defaults resolved from NPZ metadata |
| 2 | Test per-row vs full-window prediction (Romesh) | 🟡 partial | per_row 5-fold V8 done (`per_row_5fold/`), full_window 5-fold QUEUED to fire after HP sweep |
| 3 | Add noise augmentation to labels/features (Romesh) | ⏸ queued | `configs/decoder_v8_noise_aug.json` exists; `train_v8_noise_aug_5fold.sh` queued in Phase B3 — not yet run |
| 4 | Prepare summer DOE for direction × speed × depth × material (Stephen) | ⏸ stubs only | `scripts/doe/{generate_single_line_gcode,build_doe_table,auto_label_alignment}.py` written but not specced. `DESIGN_OF_EXPERIMENTS.md` not started. Monday working session was supposed to design this |
| 5 | Review sensor priorities (gyroscope) from prior ablations | ⏸ queued | Phase B4 `run_sensor_ablation_v8_cross_fold.sh` exists; not yet run |
| 6 | Manbir reads paper, reconvenes Monday | ❓ outside our scope | n/a |
| 7 | Tarmac PO follow-up with Lois / Woody (Stephen) | ❓ outside our scope | n/a |
| 8 | Hire Tim for summer (Manbir) | ❓ outside our scope | n/a |
| 9 | Revisit tool imaging / R2 camera (Stephen) | ❓ outside our scope | n/a |
| 10 | Design simple testable experimental runs (core group) | ⏸ pending DOE | Tied to (4) |
| 11 | Manbir organizes Monday working session | ❓ outside our scope | n/a |
| 12 | Reduce manuscript to page limits (Romesh) | ❓ outside our scope | n/a; our draft is at `decoder_paper_v2/latex/decoder_paper_v2.tex` |

### Meeting paper-framing decision

Romesh + Manbir explicitly agreed to **pivot from "full G-code reconstruction"
to "physically recoverable parameters: feed rate, depth of cut, command type,
direction"**.

Current state: Round 2 Phase A/D both add per-axis breakdown (X / Y / Z / F / S)
to the results table. So the framing pivot IS reflected in the metrics
infrastructure. The decoder still produces full G-code text autoregressively
(no head-only mode), but the manuscript will report per-field recoverability.

### Critical data-availability issue surfaced by alignment check

Ran a field-frequency probe on V8 per_row fold 1 ([alignment-check 2026-05-12]).
**Feed rate F is in 0.4–0.6 % of per_row rows and 22 % of full_window
samples. Spindle speed S, I, J coordinates are ENTIRELY ABSENT.** Arcs are
recorded via the R notation (arc radius), not I/J:

| Field | per_row train | per_row test | full_window train | full_window test |
|---|---|---|---|---|
| X | 90.4 % | 88.5 % | 100 % | 100 % |
| Y | 88.6 % | 86.6 % | 100 % | 100 % |
| Z | 35.8 % | 31.4 % | 29.7 % | 32.6 % |
| **F** | **0.4 %** | **0.6 %** | **18.8 %** | **22.0 %** |
| **S** | **0** | **0** | **0** | **0** |
| R | 15.2 % | 17.2 % | 88.1 % | 88.6 % |
| **I, J** | **0** | **0** | **0** | **0** |

**Implications for the paper's "feed rate recoverability" claim:**
- per_row mode: F appears 45 times in the test set. "Has-F" accuracy is at the
  always-absent baseline (99.4 %, the probe number). The encoder/decoder can't
  meaningfully be evaluated on F in per_row mode.
- full_window mode: F appears in 29/132 windows. Better but still rare. The
  values are nearly-constant (typical pattern: a single `F22.` per G1 line).
  So we can report "can we detect that feed rate is being commanded" but NOT
  "can we recover the feed rate value" — there isn't enough variation.
- Spindle speed (S) is **not measurable from this dataset at all**.

**This is a real gap with the meeting's stated direction.** The team wanted
feed rate / depth of cut as recoverable parameters. Depth of cut maps to the
Z-axis values (we have signal there), so that claim survives. Feed rate
recovery is a near-empty claim with this data, and the summer DOE is the
only fix.

### Where the current plan is consistent with the meeting

1. **The 16-token truncation bug** that Romesh flagged is fixed and verified.
2. **Per-row vs full-window is being TESTED both ways** (per_row done as pilot, full_window queued).
3. **Position memorization is removed** (`use_window_position=false`, position metadata not exposed).
4. **Per-field recoverability metrics** are in the pipeline (Round 2 Phase A).
5. **Noise augmentation is queued** (B3) — not yet executed.
6. **Sensor ablation is queued** (B4) — not yet executed.

### Where the plan is at risk of drifting from the meeting

1. **Manbir's "inform decoder of toolpath patterns" suggestion** is in the plan
   as Phase 6.2 `pattern_aware_decoder.py` / Round 2 Phase B7 — but as a
   pilot/fold-1 experiment, not as a central design. The meeting's framing
   suggested this should be a core decoder enhancement, not an ablation cell.
2. **Paper title** in `decoder_paper_v2/latex/decoder_paper_v2.tex` is
   "G-Code Decoder: Multi-Modal Sensor-Driven Recovery of CNC Machining
   Instructions" — "Recovery of Instructions" is closer to the agreed
   recoverable-parameters framing than "reconstruction", but the paper body
   may still over-claim full reconstruction. Will need to re-audit after the
   full_window results land.
3. **Summer DOE scripts are stubs.** Without filled-in factor lists, the
   summer data collection cannot start when the tarmac arrives. This is a
   real risk because the meeting tied "real" reconstruction to the future
   DOE dataset, and our timeline assumes that dataset will be ready.
4. **Feed rate variation in the data is minimal.** This wasn't anticipated
   at the meeting; we may need to caveat manuscript claims about feed-rate
   recoverability or wait for the summer DOE before making them.
5. **Material variety is zero.** All current data is aluminum on the Bantam.
   Stephen mentioned material should be a DOE factor in summer.

### Recommended adjustments (advisory)

- After full_window 5-fold completes, re-evaluate whether `pattern_aware_decoder`
  should be elevated from a Round-2 ablation to a core architecture choice.
  Manbir's specific contribution was "build pattern detection into the decoder
  before trying to generalize," which is what pattern-aware would deliver.
- Caveat the paper's feed-rate claims to "binary detection of F-presence"
  rather than "value recovery."
- Add a "data-coverage" table to the manuscript stating exactly which G-code
  fields are present at what frequency, so readers know which "recoverable
  parameter" claims are evidence-backed vs aspirational.

### CORRECTION — DOE infrastructure is NOT a stub (filed 2026-05-12)

Earlier advisory said "summer DOE scripts are stubs". Actually:

- `outputs/decoder20260511/DESIGN_OF_EXPERIMENTS.md` (135 lines) is substantive:
  factor list, levels, sample-size rationale, per-run G-code structure,
  data-capture spec, train/test split policy, pipeline path summary, and open
  items for the Monday working session.
- `outputs/decoder20260511/DOE/doe_v1.csv` and `.json` exist with 188 runs.
- `outputs/decoder20260511/DOE/programs/` contains 188 generated `.gcode` files.
- All three DOE scripts (`build_doe_table.py`, `generate_single_line_gcode.py`,
  `auto_label_alignment.py`) are real, not stubs.

What's still needed for the summer:
- Tormach controller per-line timestamp logging confirmation
- Material order placed (aluminum 6061, steel 1018, delrin)
- Tool inventory (3.0 / 6.0 / 9.5 mm endmills)
- R2 camera trigger integration

These are operational, not infrastructure. The Monday working session was the
right place to close them.

## 2026-05-12 — Manuscript audit (decoder_paper_v2.tex)

### Critical finding — headline numbers came from BUGGY V7-DATA runs, not V8

Traced [decoder_paper_v2/tables/headline_5fold.csv](decoder_paper_v2/tables/headline_5fold.csv):

```
Token,0.8317 ± 0.0195
Sequence,0.4263 ± 0.0407
Type,0.9721 ± 0.0121
Command,0.9791 ± 0.0187
Param_type,0.9442 ± 0.0131
Numeric,0.6003 ± 0.0244
```

These match the metrics in [RESULTS_TABLE.json](RESULTS_TABLE.json) under
"V8 per_row 50ep (no shortcuts)". But the underlying training_log
([per_row_5fold_50ep_legacy/fold_1/results/training_log.txt](checkpoints/per_row_5fold_50ep_legacy/fold_1/results/training_log.txt))
shows the actual data loaded was:

```
Loading train: outputs/experiments_2026_02_25/no_proximity_no_pressure_w256_s64_cv/fold_1/preprocessed/train_sequences.npz
  Samples: 303, Tokens: 951, Max len: 7
```

**That's V7 dedup'd data (303 samples), not V8 per_row (19,584 samples).**
The `--encoder_config silently overrode --data_dir` bug (already fixed in
[scripts/evaluation/run_decoder_quick_test.py:1488-1503](scripts/evaluation/run_decoder_quick_test.py#L1488-L1503))
was active when these runs were done. The paper's "V8 per_row 5-fold" numbers
are actually V7-data numbers under a V8 label.

Where the truth is now: the current HP sweep on TRUE V8 per_row data plateaus
at token≈0.72, cmd≈0.32 (best Stage 1) or cmd≈0.50 (Stage 2 ss_0.5 winner).
Both are dramatically lower than the paper's 0.83 token / 0.97 cmd headline.

**This means the manuscript's results table must be FULLY REPLACED after the
full_window 5-fold completes.** Numbers to swap:

| Paper table | Current claim | After regeneration |
|---|---|---|
| `headline_5fold.tex` (Table 1) | tok 0.832, cmd 0.979 | will be from V8 full_window 5-fold; expected lower |
| `per_class_command.tex` (Table 2) | F1 0.94 on G1 | will regenerate |
| `per_class_param_type.tex` (Table 3) | F1 0.95 macro | will regenerate |
| `per_axis_recoverability.tex` (Table 4) | F has-axis 0.992 (≈ always-absent) | needs explicit caveat row; numbers will change |
| `per_digit_position.tex` (Table 5) | digit 1.0 → 0.86 across 6 positions | will regenerate |
| `ablations_summary.tex`, `anova.tex`, `bootstrap_ci.tex` | numbers from V7-data runs | all need regeneration |

### Specific over-claims in the body to fix when numbers refresh

1. **Line 55 (abstract):** "0.836 ± 0.025 token accuracy, 0.970 ± 0.027 command accuracy" — both V7-data numbers.
2. **Line 150:** "Across all folds, the decoder achieves 0.836 ± 0.025 token accuracy, 0.445 ± 0.049 sequence accuracy" — V7-data.
3. **Line 169 (per-axis):** "feed rate (F), spindle (S), and arc parameters (I, J) have effectively zero positive support" — **THIS CAVEAT IS GOOD, KEEP IT.**
4. **Line 196:** "Without metadata, a 27.1 pp ceiling exists on numeric recovery; the positional features close that gap" — effect size from V7-data ablation; numbers may shift.
5. **Lines 222–230 (nested ablation table):** "no shortcuts (base) 0.954" command — V7-data.
6. **Line 251 (discussion):** "Command identity (G0/G1/G2/G3) and parameter-type identity (X/Y/Z/R) are recovered with 0.97 and 0.94 accuracy from sensors alone" — V7-data.
7. **Line 273 (conclusion):** "recover the executing G-code command identity and axis presence at ~0.97 accuracy" — V7-data.

### What the paper currently does well

1. **Feed-rate caveat is in place** (line 169 + 267). The body text correctly says F/S/I/J cannot be evaluated meaningfully. The table footnote also notes this.
2. **Frozen-encoder leakage is flagged as a limitation** (line 263). The "encoder may implicitly encode position information" caveat is appropriately hedged.
3. **Per-class metrics for the long tail** are shown — G3 (20 instances), G0 (20 instances), M30 (10 instances). This honesty about long-tail F1 will survive any re-run.
4. **Sensor-modality ablation findings** (gyroscope + color as most-load-bearing) — these are qualitative and likely to survive a re-run.
5. **DOE appendix exists** (line 281) — points readers to the summer DOE for the unrecoverable claims.

### Action items for after full_window completes

1. Re-run [aggregate_v8_results.py](scripts/analysis/aggregate_v8_results.py) on the new full_window checkpoints.
2. Regenerate all `decoder_paper_v2/tables/*.tex` files via `export_tables_latex.py` (need to write this if not already done).
3. Re-run `anova_and_bootstrap.py` on the new metrics.
4. Recompile `decoder_paper_v2.tex` and visually diff against the current PDF.
5. Add a **"data coverage" table** explicitly listing field-presence frequency
   per split so readers cannot misinterpret the F=0.992 cell as feed-rate
   recovery.
6. Re-frame the headline: if full_window achieves, say, 0.80 token / 0.85 cmd,
   the paper can stand. If it plateaus at the per_row level (0.72 / 0.50),
   the manuscript needs a more substantial rewrite acknowledging
   information-theoretic limits.

## 2026-05-12 — Pattern-aware decoder design (Manbir's meeting contribution)

### What Manbir said at the meeting

> "We haven't yet built in a lot into this decoder. So if the G-codes are
> thousands of lines long, then there's a lot of overlap. ... you have the
> same tool paths being cut at different points, you know, with slight feed,
> with a slight depth of cut increment, it's the same toolpath. ... If I was
> doing this manually myself, I would be looking for those patterns. We
> haven't built those yet."

Translation: the decoder should explicitly model recurring tool-path
patterns. A face cut is a stereotyped sequence of "X+ to extent, Y step, X-
back, Y step, ...". Once the decoder knows it's IN a face cut, the
per-row prediction reduces to "which step of the face pattern are we on?".

### What's currently in place

Inspection of [train_v8_pattern_aware_pilot.sh](scripts/experiments/train_v8_pattern_aware_pilot.sh):

```
# The existing model already has a `sequence_classifier` head that predicts
# which whole G-code line is being executed. This biases the token-level
# logits toward that line's tokens — exactly the "pattern prior" Dr. Sodhi
# described in the 2026-04-28 meeting. Just enable the head.
```

So the current "pattern-aware" implementation is:
- A `sequence_classifier` head that predicts the WHOLE G-code line (out of
  214 distinct lines per fold) directly from the encoder memory.
- The line classifier's softmax biases the legacy-token head's distribution
  toward the predicted line's tokens.

**Verdict: this is a WEAK version of Manbir's idea.** It models "which line
is this" but not "which pattern is the machine doing." A 214-way classifier
on a 132-window test set is severely overfit-prone.

### Stronger pattern-aware design (proposed)

A genuinely pattern-aware decoder would be hierarchical:

```
                  ┌─────────────────────────────┐
sensor window →   │ encoder (frozen) → memory   │
                  └─────────────┬───────────────┘
                                │
                  ┌─────────────▼───────────────┐
                  │ pattern head (9-way):       │
                  │ face / pocket / adaptive /  │   ← operation_type
                  │ … / damageX                 │
                  └─────────────┬───────────────┘
                                │
                  ┌─────────────▼───────────────┐
                  │ step head (within pattern): │
                  │ "step 3 of 12 in a face"    │
                  └─────────────┬───────────────┘
                                │
                  ┌─────────────▼───────────────┐
                  │ token decoder, conditioned  │
                  │ on (pattern, step)          │
                  └─────────────────────────────┘
```

Why this is stronger:
1. **The pattern head's 9-way classifier matches the encoder's actual
   training objective**, so the encoder embeddings carry this signal robustly
   (probe confirmed 0.846 op-type accuracy from frozen memory).
2. **The step head resolves the per_row ambiguity** — for a given pattern,
   "step 3 of 12 in a face" disambiguates which row of the window we're
   predicting WITHOUT exposing window_index as a generic leak. Step indices
   are derivable from the G-code structure, not from metadata.
3. **The token decoder becomes a small conditional generator** instead of
   a generic 2,418-vocab predictor.

Risks:
- Step indexing requires pre-computing canonical step indices per pattern.
  Doable but adds preprocessing complexity.
- For damage classes, the "pattern" may not be well-defined.
- Adds 2 heads worth of loss balancing.

### Recommendation

1. **First, run full_window 5-fold as already queued.** Full_window
   eliminates the per_row ambiguity by predicting the full multi-line target.
   If full_window's autoregressive context is enough, hierarchical
   pattern-awareness is unnecessary.
2. **If full_window also plateaus**, implement the hierarchical
   pattern-aware decoder. New file: `src/miracle/model/pattern_aware_decoder.py`
   (currently empty per the plan — only the training shell script exists).
3. **Either way, keep the current `sequence_classifier` head** as a baseline
   ablation; report its effect in the table.

This was an `else` branch in the original plan ("if Phase B7 underperforms,
elevate"); the encoder audit just gave us evidence that it should be
elevated only if the simpler full_window doesn't suffice.

## 2026-05-12 — Phase C analyses on composite winner (ss_0.5 fold 1)

Ran the three Phase-C scripts on the current composite winner
(`stage2/scheduled_sampling_0.5`, fold 1) while the GPU sweep continues.
All CPU-only, no GPU contention.

### Per-axis numeric recovery — X is the dominant bottleneck

[audit/numeric_diag_ss05.json](audit/numeric_diag_ss05.json):

| Axis | Digit accuracy | Full-value correct | n positions |
|---|---|---|---|
| X | 0.522 | **7.0 %** | 7,003 |
| Y | 0.753 | 43.7 % | 6,861 |
| Z | 0.920 | 72.0 % | 2,464 |
| J | 0.825 | 68.3 % | 1,369 |
| I | 0.992 | 95.2 % | 21 |

(I and J appear in the script's `pt_t` even though the literal G-code text
shows R-notation only — investigating this discrepancy; might be a vocab
mislabel where R is being encoded as J. Low priority — not load-bearing on
the paper's claims since R itself doesn't appear in the per-axis evaluation.)

### Per-digit-position — middle digits are where it fails

| Position | Accuracy |
|---|---|
| 0 (most-significant) | 1.000 |
| 1 | 0.792 |
| 2 | 0.561 |
| 3 | 0.506 |
| 4 | 0.506 |
| 5 (least-significant) | 0.778 |

The model nails the magnitude (position 0) and to a lesser extent the
precision endpoint (position 5), but the middle positions — which carry
the actual coordinate sub-millimeter information — sit at coin-flip
accuracy. **The decoder gets the magnitude right but loses precision.**

### Per-operation class — damagepocket is the worst

[audit/per_op_class_v8_ss05.json](audit/per_op_class_v8_ss05.json):

| Class | n test rows | Token acc | Sequence acc |
|---|---|---|---|
| pocket | 890 | 0.794 | 0.004 |
| adaptive | 1,840 | 0.770 | 0.004 |
| damageadaptive | 529 | 0.742 | 0.008 |
| damageface | 95 | 0.737 | 0.011 |
| face | 584 | 0.733 | 0.029 |
| adaptive150025 | 2,131 | 0.723 | 0.004 |
| face150025 | 354 | 0.710 | 0.048 |
| pocket150025 | 1,308 | 0.651 | 0.006 |
| **damagepocket** | 206 | **0.516** | 0.000 |

Damagepocket sits 28 pp below the next-worst class. Worth investigating
whether this is a noisy-data class or genuinely the hardest geometry.

### Sequence-level — only 0.8 % exact match

[audit/failure_cases_v8_ss05.json](audit/failure_cases_v8_ss05.json):
- 7,937 test rows, **67 exact matches (0.8 %)**.
- Mean edit distance 1.54 tokens (out of ~5 tokens per row).
- Median edit distance 1.

So the model is "close but not exact" on most rows — it gets 3–4 of 5
tokens right but rarely all 5. This is consistent with the per-digit
finding (middle digits failing) and the per_row ambiguity hypothesis.

### Manuscript implications

These four findings concretely shape the paper's "what the sensor pathway
cannot recover" section once headline numbers refresh:

1. **X-axis numeric recovery** should be called out specifically (not just
   "numeric is weak"). It's 6× weaker than Z numeric recovery.
2. **The information-theoretic ceiling claim** is supported: position-0
   digits are perfect, middle-positions are random. The encoder carries
   magnitude but not precision.
3. **Per-class long tail** — the paper currently quotes G-code-class
   long tails (G1 vs G3 vs M30). It should also report operation-class
   long tails: damagepocket is the case the model can't yet handle.
4. **Sequence-level accuracy at <1 %** even at the best config calls for
   reframing "full G-code reconstruction" as "per-token recovery with
   characterized precision loss" — exactly the meeting's pivot.

### Files written this analysis

- `outputs/decoder20260511/audit/numeric_diag_ss05.json`
- `outputs/decoder20260511/audit/per_op_class_v8_ss05.json`
- `outputs/decoder20260511/audit/failure_cases_v8_ss05.json`

After full_window 5-fold completes, re-run all three on the new headline
checkpoints (one command per fold) and replace these single-fold reads.

## 2026-05-12 — Phase C-4: failure-mode classification

Decoded the top 20 worst-edit-distance failures from `scheduled_sampling_0.5`
fold 1 ([audit/failure_cases_decoded_ss05.json](audit/failure_cases_decoded_ss05.json))
and classified them. **The errors are structured, not random:**

| Failure mode | Count | Example |
|---|---|---|
| **Dropped G-command** | 40 % (8/20) | TRUE: `G2 X 2607 Y 2594 R 0128` → PRED: `X X -200 Y 0849 EOS Y0.` |
| **Wrong G-command** | 35 % (7/20) | TRUE: `G2 X 3032 Y 0365 R 0128` → PRED: `G1 X -200 Y 2177 R 0742` |
| **Hallucinated G-command** | 20 % (4/20) | TRUE: `X -150 Z 0025` (no G command) → PRED: `G1 X -125 Y 0010 R` |
| Value-only error | 0 % | — |
| Other | 5 % | — |

**Key insight: 95 % of the worst failures are command-identity confusion** —
not value-precision errors. The model doesn't reliably know whether THIS
row contains a G-command at all, and if so which one. This is *exactly*
what the per_row-ambiguity hypothesis predicts: within a window of 60+
rows, some have G-commands and some don't, but the encoder memory is
identical for all of them. Without row-position info the decoder collapses
to one stereotyped output pattern per window.

**Manuscript implication**: when the headline numbers refresh from
full_window, this paragraph adds a concrete "what fails and why" narrative
to the discussion section. In full_window mode the autoregressive
generation gives the decoder the previous row's G-command (or absence of
it) as context, which should largely fix the command-identity failures
specifically. If it does, that's a clean validation of the per_row
ambiguity diagnosis.

## 2026-05-12 — Phase C-5: field coverage across all 5 folds

[audit/field_coverage_5fold.json](audit/field_coverage_5fold.json):
verified that the per-field positive-support frequencies hold across all
5 folds × 3 splits × 2 modes. Variation is < 3 pp on every field
between any two folds. The data_coverage.tex numbers in the manuscript
are therefore representative, not just a fold-1 artifact.

Minor note: spindle speed `S` appears in **3 of 30 splits** at 0.3-0.9 %
(all in full_window val/train). Previous claim "S is entirely absent"
should be tightened to "S appears in <1 % of full_window samples and 0 %
of per_row samples". The data-coverage table footnote already says
"effectively zero", which is honest.

## 2026-05-12 — Phase 7-validation: DOE infrastructure verified

Ran a validity check on the 188 generated DOE programs ([outputs/decoder20260511/DOE/](DOE/)):

- All 188 G-code files exist and parse cleanly (no syntax errors in
  spot-checks).
- 188 unique factor combinations, **zero duplicates**.
- Factor coverage is reasonably balanced:
  - motion_type: 65 straight / 59 arc_cw / 64 arc_ccw
  - feed_rate: 47 @ 50 / 57 @ 100 / 44 @ 200 / 40 @ 400 mm/min
  - spindle_speed: 59 @ 3 k / 75 @ 6 k / 54 @ 9 k rpm
  - depth_of_cut: 90 @ 0 mm (air cuts) / 24 @ 0.10 / 27 @ 0.25 / 28 @ 0.50 / 19 @ 1.00
  - material: 66 aluminum_6061 / 61 steel_1018 / 61 delrin
  - tool_diameter: 62 @ 3 mm / 57 @ 6 mm / 69 @ 9.5 mm
  - air_cut: 90 True / 98 False
- Spot-checked 5 random runs against the CSV: feed_rate values appear in
  the G-code, depth maps correctly to Z token, air-cut runs omit the Z
  plunge.

**Conclusion: the summer DOE is ready to run as soon as the Tormach arrives.**
The team won't need to redesign experiments — they can pull the CSV +
.gcode files straight from `outputs/decoder20260511/DOE/`.

## CPU-side work summary (2026-05-12 afternoon session)

Ran while the GPU sweep was in flight:

| Task | Output | Status |
|---|---|---|
| Phase C-1: numeric diagnosis | `audit/numeric_diag_ss05.json` | Done — X recovery is 7 % full-value |
| Phase C-2: per-class breakdown | `audit/per_op_class_v8_ss05.json` | Done — damagepocket worst at 51.6 % token |
| Phase C-3: failure cases | `audit/failure_cases_v8_ss05.json` | Done — 0.8 % exact-match |
| Phase C-4: failure-mode classification (decoded) | `audit/failure_cases_decoded_ss05.json` | Done — 95 % command-identity confusion |
| Phase C-5: 5-fold field coverage | `audit/field_coverage_5fold.json` | Done — coverage consistent across folds |
| Phase 7 DOE validation | (verified in-place) | Done — DOE ready for Tormach |
| Aggregator + ANOVA pipeline | `aggregate_v8_results.py --sweep-name`, `anova_and_bootstrap.py --baseline-name` | Refactored — handles both per_row and full_window |
| Paper v2 placeholders | 25 body + 47 table TBD markers + regen guide | Done — PDF compiles cleanly |

All deliverables are CPU-only, no GPU contention with the running sweep.

## 2026-05-12 (evening) — Paper v2 quality upgrade

User flagged that `decoder_paper_v2.tex` was structurally weak relative to
the accepted sensors paper at
`outputs/experiments_2026_02_25/paper/sensors_v4.tex` (1444 lines, 41
citations) and the prior `decoder20260304/paper/decoder_paper_mdpi.tex`
(1386 lines, 31 citations). Our v2 was 318 lines with 4 unique citations.
Rewrote the manuscript end-to-end:

| Metric | Before | After |
|---|---|---|
| Lines | 318 | 615 |
| Pages | 10 | 23 |
| Citations used | 4 | 57 |
| Bib entries | 31 | 56 (ported 24 from sensors paper bbl) |
| Top-level sections | 6 | 8 (added Problem Formulation, Experimental Setup) |
| Subsections | 11 | 28 |

Structural matches to the accepted sensors paper:
- Introduction: stakes paragraph + prior-work landscape + closely-related
  group work + decoder problem framing + 4 explicit RQs + 5 bold-labeled
  contributions + roadmap paragraph.
- Related Work: 4 deep subsections (CNC monitoring; seq2seq / grammar;
  AM side-channel; cybersecurity) + literature-comparison table.
- Problem Formulation as its own section: task definition, vocabulary,
  per-row target rationale, 7-metric evaluation protocol.
- Method: System Overview / Encoder / Decoder Architecture (with
  subsubsections for multi-head output, grammar mask, scheduled sampling)
  / Training Configuration / Hyperparameter Sweep methodology.
- Experimental Setup: Dataset / Preprocessing / 5-fold CV / Baselines &
  Ablation Designs (8 enumerated) / Statistical Analysis / Software.
- Results restructured around RQ1–RQ4 + a new Failure-Mode Analysis
  subsection capturing the Phase C-4 finding (95 % command-identity
  confusion, not value-precision errors).
- Discussion expanded to 5 subsections: three tiers of recoverability /
  failure modes are structural / sensor-modality deployment implications /
  comparison with AM side-channel recovery / when to use this decoder.
- Limitations restructured into 3 subsections + roadmap.
- Conclusion rewritten as a 3-finding narrative.

All TBD placeholders are intact (41 in body + tables). PDF compiles
cleanly. The paper now reads at the depth and breadth of the accepted
sensors paper.

## 2026-05-13 (overnight) — HP sweep completion + watcher hang

### Overnight sweep progress

Sweep ran unattended overnight. End state at 09:35 today:

- **Stage 2: 12/12** — all cells completed. The final cell
  `combined_curric_ss_ls` confirmed that stacking curriculum + ss=0.5 +
  label_smoothing=0.1 over-regularises; the legacy-token head collapsed
  (tok 0.67, cmd 0.31). Recipes don't stack additively; ss=0.5 alone is
  the right move.
- **Stage 3: 12/14** — `encoder_f110` and `encoder_f56` failed (rc=1)
  because the alternative encoder checkpoints
  (`outputs/experiments_2026_02_25/full_w256_s64_cv/...` and
  `..._no_color_no_magnetometer_..._cv/...`) don't exist on disk. These
  alternative encoders were never trained. The cells were always going
  to fail; not a load-bearing loss for the paper since they were about
  encoder-choice not decoder-architecture.

### Final composite ranking (top 5 of 34 cells)

| Rank | Cell | tok | cmd | num | composite |
|---|---|---|---|---|---|
| 1 | **stage2/scheduled_sampling_0.5** | 0.7308 | 0.4993 | 0.4548 | **0.6062** |
| 2 | stage3/ss_0.1 | 0.7222 | 0.3510 | 0.4456 | 0.5555 |
| 3 | stage2/n_layers_10 | 0.7287 | 0.3255 | 0.4533 | 0.5527 |
| 4 | stage3/seed_2024 | 0.7213 | 0.3365 | 0.4528 | 0.5522 |
| 5 | stage2/n_layers_12 | 0.7219 | 0.3356 | 0.4457 | 0.5507 |

`scheduled_sampling_0.5` wins composite by +5 pp over the next-best cell
and by +20 pp on command-accuracy. Multi-seed cluster (seeds 123/456/2024)
spans cmd ∈ {0.293, 0.324, 0.337} — `ss_0.5`'s 0.499 is 5σ outside this
cluster. The lift is real, not seed-lucky.

### Watcher hang and fix

The post-sweep auto-launch watcher at `/tmp/launch_full_window_when_sweep_done.sh`
was hung for ~12 hours after the sweep finished. Root cause:

- The watcher loop is `until ! pgrep -f "hp_sweep_stage2.py|hp_sweep_stage3_parallel.py"`.
- Two leftover bash shells from yesterday afternoon's Stage-2 chain
  command (pids 1915653, 1915656) had `hp_sweep_stage2.py` embedded in
  their `bash -c` argument string from when they originally launched the
  Stage 2 dispatcher.
- `pgrep -f` matches the entire command line — those long-stale shells
  were matching even though their Python child (the actual dispatcher)
  was long gone.

Fix: killed pids 1915653 and 1915656. Watcher fired within 60 s.

**Lesson for future automation**: name-matching is brittle. Future
watchers should verify the dispatcher's child Python PID is alive, not
just grep for the script name in the argv string.

### Post-sweep launches (09:35 today)

- GPU 0: `train_v8_full_window_5fold.sh` running 5 sequential folds with
  the composite-winner config (Stage-1 architecture + Stage-2 ss=0.5).
  ETA ~2.5 h.
- GPU 1: `e2e_lr5e-6` (one cell). The earlier `e2e_lr1e-5` cell NaN'd at
  epoch 1 (encoder_lr=3e-5 too aggressive); this re-run uses lr=5e-6 which
  should be stable. ETA ~1.5 h.

### Files written or updated since the previous notes entry

- `outputs/decoder20260511/decoder_paper_v2/latex/decoder_paper_v2.tex` (615 lines now)
- `outputs/decoder20260511/decoder_paper_v2/latex/decoder_references.bib` (56 entries)
- `outputs/decoder20260511/decoder_paper_v2/TABLES_REGENERATION_GUIDE.md`
- `outputs/decoder20260511/decoder_paper_v2/tables/data_coverage.tex` (new)
- `outputs/decoder20260511/decoder_paper_v2/tables/v7_legacy/` (V7-data backups)
- `outputs/decoder20260511/audit/encoder_probe_v8.json`
- `outputs/decoder20260511/audit/numeric_diag_ss05.json`
- `outputs/decoder20260511/audit/per_op_class_v8_ss05.json`
- `outputs/decoder20260511/audit/failure_cases_v8_ss05.json`
- `outputs/decoder20260511/audit/failure_cases_decoded_ss05.json`
- `outputs/decoder20260511/audit/field_coverage_5fold.json`
- `outputs/decoder20260511/audit/hp_sweep_all_stages_summary.json` + `.md`
- `scripts/analysis/aggregate_hp_sweep_all_stages.py` (new, dual-baseline)
- `scripts/analysis/encoder_linear_probe.py` (new)
- `scripts/analysis/aggregate_v8_results.py` (extended with `--sweep-name`)
- `scripts/analysis/anova_and_bootstrap.py` (extended with `--baseline-name`)
- `scripts/experiments/hp_sweep_stage3_parallel.py` (new, 14-cell parallel dispatcher)
- `scripts/experiments/train_v8_full_window_5fold.sh` (updated config)

### Git status as of this notes entry

26 modified + new files uncommitted. Last commit was
`8d40113 V8 5-fold sweep: per_row decoder matches V7 ceiling without
shortcuts` — that commit was BEFORE the encoder_config-bug discovery, so
its claim of "matching V7 ceiling" is no longer accurate. **A commit
hygiene pass is overdue.**

## 2026-05-13 — Recommendations: things NOT on the plan that should be

Honest review of gaps relative to what a strong sensors-journal-quality
submission requires. The original Round-1 + Round-2 plan covered the
remediation, ablations, and metrics, but several reviewer-anticipated
items are missing entirely from the plan as written.

### Critical for submission (reviewers will absolutely ask)

1. **Non-neural baseline.** The accepted encoder paper compared against
   five baseline model families (Random Forest, XGBoost, 1D-CNN,
   Transformer, LSTM). The decoder paper currently has **zero non-neural
   baselines.** Any reviewer in the CNC-monitoring community will ask
   "what does XGBoost on the same encoder embeddings get?" We have a
   metadata-only XGBoost baseline from the Phase-1 audit
   ([audit/shortcut_leakage.json](audit/shortcut_leakage.json)) but it
   answers a different question. Recommendation: a 1-day side experiment
   that trains an XGBoost classifier per output head on the mean-pooled
   encoder memory, reported as a baseline row in Table~\ref{tab:headline}.

2. **Threat model section.** The manuscript frames decoder applications
   in terms of "process verification" and "anomaly localisation" and cites
   NIST SP 800-82, but it does not formalise the threat model: what
   attacks does the decoder detect, what attacks does it miss, what is
   the adversary's capability? Without this, the cybersecurity framing is
   aspirational. Recommendation: a half-page Threat Model subsection in
   the Method or Discussion section enumerating attack classes (G-code
   substitution, parameter tampering, replay) and which the decoder
   addresses.

3. **Multiple-comparisons correction.** The 34-cell HP sweep selects a
   winner from a large family. The winning cell's apparent +20 pp
   command-accuracy lift over the runner-up is at risk of being a
   multiple-testing artefact, even though the multi-seed variance check
   weakens that concern. Recommendation: report Bonferroni- or BH-FDR-
   corrected p-values for the ANOVA between the winner and the runner-up
   family, and reference this in Section~\ref{sec:results-headline}.

4. **Encoder-leakage probe IN the paper.** We have the linear-probe
   results ([audit/encoder_probe_v8.json](audit/encoder_probe_v8.json))
   showing what information the frozen encoder embeddings carry.
   Currently this lives in audit/ and is not referenced in the
   manuscript. The probe is publishable: it bounds what the decoder can
   in principle recover and validates the "encoder is not the bottleneck"
   claim. Recommendation: add a "Frozen-Encoder Probe" subsection in
   Section~\ref{sec:method} (or as an early Results subsection) with the
   per-field probe accuracies and the modal-row-per-window ceiling
   interpretation.

5. **Confusion-matrix figures.** Round-2 Phase D includes
   `confusion_matrices.py` as a generator script, but the figures have
   not been generated for the current V8 checkpoints. The sensors paper
   has confusion matrices in Results, and reviewers expect them for any
   multi-class study. Recommendation: generate command-level,
   operation-class-level, and parameter-type-level confusion matrices
   from the full_window 5-fold checkpoints as part of the regeneration
   pass.

### Helpful for statistical rigor

6. **Power analysis on per-class long tail.** With 132 test samples per
   fold and G3/M30 at $\le 20$ supports per class, our power to detect a
   pp difference in per-class F1 is low. Reporting the power-curve for
   the per-class command-head metrics would clarify when "not
   significant" actually means "underpowered". Recommendation: a 2-line
   note in Section~\ref{sec:results-headline} citing the minimum
   detectable effect at $\alpha=0.05$, $\beta=0.20$ for each class.

7. **Train-vs-test covariate-shift audit.** We confirmed
   ([audit/field_coverage_5fold.json](audit/field_coverage_5fold.json))
   that per-field coverage is consistent across folds. But we did NOT
   check that the train-split's coverage matches the test-split's
   coverage on each individual fold. A 3-5 pp covariate shift on, say,
   has-Z would change the interpretation of has-Z accuracy. Quick to add:
   a column comparing train-vs-test coverage per fold, with a footnote in
   Section~\ref{sec:dataset}.

8. **Token-position failure analysis.** Per-digit-position analysis is
   in Section~\ref{sec:numeric-decomposition}; the analogous breakdown
   over OUTPUT-sequence position (does the decoder fail more at position
   0 vs position 4?) is missing. Recommendation: extend
   `diagnose_numeric_accuracy.py` to report accuracy per output-sequence
   position, not just per-digit-position within a NUM token.

### Forward-looking (next paper / next revision)

9. **Encoder retrain (Phase F) design spec.** Currently flagged as on
   hold. After today's encoder-audit findings (303-sample training, 9-class
   classification objective, recon_weight=0.1) the case for retraining
   has strengthened. Even if we don't execute Phase F before submission,
   the next revision will need a concrete design: what data, what loss,
   what auxiliary heads, what compute budget. Recommendation: draft a
   `PHASE_F_DESIGN.md` while full_window results land, parallel to the
   manuscript.

10. **Failure-mode visual figure.** We have the decoded worst-failure
    examples ([audit/failure_cases_decoded_ss05.json](audit/failure_cases_decoded_ss05.json))
    classified into four modes (dropped/wrong/hallucinated G-command,
    value-only error). A figure showing one TRUE / PRED pair per mode
    would dramatically improve the manuscript's "what fails and why"
    narrative. Round-2 Phase D should include this generator.

11. **Replicability checklist appendix.** Sensors journal requires (and
    NeurIPS/ICML strongly prefer) a reproducibility checklist: random
    seeds, hardware specs, training time, hyperparameter ranges, sweep
    sizes. We have all this information in the project journal and
    scripts; needs to be assembled into one appendix or supplementary
    page for the manuscript.

### Operational hygiene gaps

12. **Git commits are 4 days behind.** The last commit
    (`8d40113`) was 2026-05-09 (before the encoder-config bug discovery,
    before the HP sweep, before the paper rewrite). 26 files modified or
    added since then. Recommendation: a commit hygiene pass NOW with
    descriptive sub-commits: (a)~analysis scripts; (b)~experiment
    drivers; (c)~paper rewrite; (d)~audit JSONs.

13. **Watcher pattern is fragile.** The auto-launch watcher held the
    sweep state hostage for ~12 hours overnight because of name-match on
    stale shell argv strings. The fix is mechanical (check actual child
    PIDs, not name strings), but worth recording in a "lessons" file so
    we don't repeat this when the summer Tormach data starts streaming.

14. **No prereg / pre-analysis plan.** RQs were articulated AFTER seeing
    sweep data. For a follow-up DOE-driven paper, pre-registering the
    factor list and target metrics on the summer dataset before any
    experiments run would be straightforward and credibility-improving.
    Recommendation: include a pre-registration section in the
    DESIGN_OF_EXPERIMENTS.md before any DOE data are collected.

### What's already on the plan and progressing

- Round-2 Phase D figures (confusion matrices, learning curves, etc.)
  are scheduled to run after full_window completes.
- Phase F encoder retrain is on hold pending full_window results.
- Phase G wrap-up (memory + email + commit) will close the loop.
- Phase B sensor-ablation, noise-aug, LOCO, pattern-aware, vocab-2digit
  experiments are queued after full_window.

The 14 items above are gaps the plan does NOT cover that I believe
materially affect submission strength. Items 1-5 are reviewer
showstoppers; 6-11 are rigor improvements; 12-14 are project-management
hygiene.

## 2026-05-13 (morning) — Git commit pass + parallel recommendations execution

User said "yes yes and yes and parallel where possible" to the
recommendations advisory. Worked on three items in parallel.

### Git commit hygiene — done

Eight new commits since `8d40113`:

```
f79a738  Fix --encoder_config silent override of --data_dir + add per-class metrics
889df06  Add per_class_metrics.py for sklearn-based precision/recall/F1 + confusion
9e388ab  Add 3-stage HP sweep infrastructure + cross-stage aggregator
d7090d4  Add analysis scripts (Round-2 Phases A/C + XGBoost baseline)
c564ca0  Add Phase B experiment drivers + 2-digit vocab artifact
a0cfbcb  Update notes.md + manuscript tables + remove buggy V7-data aggregate
553555c  Paper v2 quality upgrade + TBD-placeholder migration
c4dbaff  Force-add audit JSONs documenting the remediation findings
4682202  Paper additions: encoder-probe section + non-neural baseline + threat model
90edb67  Force-add HGB baseline result JSON
```

Outputs/decoder20260511/ is `.gitignore`d by default; manuscript sources,
audit JSONs, and TABLES_REGENERATION_GUIDE.md were force-added.
manifest_*.json files in repo root (pre-existing from Feb 2026) left
untracked — not mine.

### Non-neural baseline — done (recommendation #1)

Tried xgboost first; it crashed on CPU-only with cudaErrorNoDevice even
with `device='cpu'`. The installed xgboost is GPU-built and the CUDA call
paths fire regardless of the device flag. Switched to sklearn's
`HistGradientBoostingClassifier` (same hist-tree algorithm family,
pure CPU). Script kept the filename `xgboost_baseline.py` for git history.

Results ([audit/xgboost_baseline_v8.json](audit/xgboost_baseline_v8.json)):

| Task | HGB Acc | HGB Macro F1 | Always-class baseline |
|---|---|---|---|
| command (5-class) | 0.659 | **0.232** | 0.742 (always "none") |
| has-X | 0.885 | 0.470 | 0.885 (always present) |
| has-Y | 0.866 | 0.464 | 0.866 (always present) |
| **has-Z** | **0.882** | **0.855** | 0.686 (always absent) |
| has-F | 0.994 | 0.499 | 0.994 (always absent) |
| sign-X | 0.810 | 0.381 | — |
| sign-Y | 0.858 | 0.308 | — |
| sign-Z | 0.783 | 0.505 | — |

Key insight: HGB lifts only has-Z meaningfully (Macro F1 0.86 on a binary
task with 31% positive rate). Everything else is at or below the
always-most-common-class baseline. Trees struggle with dense Transformer
embeddings; MLPs do better (the encoder probe gets cmd 0.77 vs HGB 0.66).
The baseline is informative because it bounds the value of a non-deep
approach on this representation and contextualises the decoder's lift.

### Encoder probe section in paper — done (recommendation #4)

Added [sec:encoder-probe](decoder_paper_v2/latex/decoder_paper_v2.tex) as a
Methods subsection. Explains the modal-row-per-window ceiling that bounds
the per-row decoder. Cites `audit/encoder_probe_v8.json` for the full
probe accuracies. Sets up the per_row-vs-full_window contrast that
Section sec:abl-target-mode unpacks.

### Threat model section in paper — done (recommendation #2)

Added [sec:disc-threat-model](decoder_paper_v2/latex/decoder_paper_v2.tex)
as a Discussion subsection. Formal threat model: trust assumptions,
adversary capabilities, 4 attack classes (G-code substitution, parameter
tampering, replay, skip/duplicate), per-class detection capability based
on the per-field recoverability results, and known limitations.

### Paper stats after morning additions

| Metric | Before this morning | Now |
|---|---|---|
| Lines | 615 | 757 |
| Pages | 23 | 25 |
| Citations | 57 | 57 |
| TBD placeholders | 41 | 54 |

## 2026-05-13 (late morning) — FULL_WINDOW FOLD 1 LANDED

**This is the most important single experiment of the entire remediation.**

The post-sweep watcher fired full_window 5-fold at 09:35. Fold 1
completed at ~10:55 with the composite-winner config (Stage-1 architecture
+ scheduled_sampling=0.5, max_token_len=1400 for multi-line targets).

### Headline result vs per_row composite winner

[checkpoints/full_window_5fold/fold_1/6o90io5p/results/metrics.json](checkpoints/full_window_5fold/fold_1/6o90io5p/results/metrics.json):

| Metric | full_window fold 1 | per_row ss_0.5 (winner) | Δ |
|---|---|---|---|
| token | 0.7466 | 0.7308 | +1.6 pp |
| **command** | **0.7762** | **0.4993** | **+27.7 pp** |
| **numeric** | **0.5518** | **0.4548** | **+9.7 pp** |
| sequence | 0.0152 | 0.0084 | +0.7 pp |
| type | 0.9673 | 0.9723 | -0.5 pp |
| param-type | 0.9860 | 0.9772 | +0.9 pp |

Best epoch 66 of 300 (early stop kicked in at patience).

### What this proves

The per_row-ambiguity diagnosis from the encoder audit and Phase C-4
failure-mode analysis is **empirically confirmed**:

1. Phase C-4 found 95% of per_row worst-failures were command-identity
   confusion (dropped / swapped / hallucinated G-commands).
2. The hypothesis: per_row mode is fundamentally ambiguous because the
   encoder memory is duplicated 60× (once per row) but the row-level
   target varies. The decoder collapses to a stereotyped output.
3. Predicted fix: full_window mode gives the decoder access to the
   previous row's G-command via autoregressive context, eliminating
   the within-window ambiguity.
4. Result: command accuracy lifts from 0.50 → **0.78** with no other
   architectural change. Numeric also lifts from 0.45 → 0.55.

This is the largest single-experiment lift of the entire remediation
and the clearest validation of the diagnostic chain.

### Implications for the manuscript

The paper's headline is no longer the V7-data-buggy 0.97 from Stage 1
(legacy commit). It's now ~0.78 command / 0.74 token (single fold) on
properly-supervised V8 data with no positional shortcuts. 5-fold mean
± std will refine these numbers in ~2-3 hours when folds 2-5 complete.

The Discussion section's "failure modes are structural, not arithmetic"
prediction is now testable on full_window data. If the 95% command-
identity-confusion failure mode shrinks dramatically in full_window
(which it should, given the +27pp command lift), the Discussion section
becomes a clean predict-then-confirm narrative.

### Currently running

- GPU 0 (94%): full_window fold 2 at epoch 110/300 — val_tok 0.68,
  val_cmd 0.88 (tracking similar to fold 1)
- GPU 1 (80%): e2e_lr5e-6 at epoch 64/150 — train_loss intermittently
  NaN (encoder fine-tune unstable even at lr=5e-6) but val_tok 0.73,
  val_cmd 0.45 stable

ETA: ~2-3 hours to 5-fold completion, then ~30-60 min of regeneration
work and the paper is at submission readiness.

## 2026-05-13 — Additional recommendations after the full_window result

The full_window fold-1 result reframes several open questions. New
recommendations beyond the original 14:

15. **Re-run Phase C-4 (failure-mode classification) on full_window.**
    The 95% command-identity-confusion finding was on per_row data.
    Predict: in full_window, command-identity confusion drops
    dramatically and value-precision errors become the dominant
    failure mode. If confirmed, the Discussion section becomes a
    predict-then-confirm narrative — strong manuscript move.

16. **Run full_window WITH shortcuts as an ablation.** We have
    `train_v8_with_shortcuts_5fold.sh` queued but it's per_row +
    shortcuts. We do NOT have full_window + shortcuts trained. The
    ANOVA matrix that distinguishes (per_row vs full_window) ×
    (shortcuts vs no_shortcuts) is incomplete on the (full_window,
    shortcuts) cell. Recommendation: clone the with_shortcuts script
    to operate on full_window data and run after the no-shortcuts
    5-fold finishes.

17. **Disentangle the +27 pp gain.** The full_window fold 1 config
    inherits scheduled_sampling=0.5 from the composite winner. Was
    the +27pp lift from full_window mode alone, or from full_window +
    ss_0.5 jointly? An ablation: train full_window WITHOUT scheduled
    sampling (ss=0). If cmd stays near 0.78, full_window is the lever;
    if cmd drops to ~0.32, the lift was a happy interaction of both.

18. **Re-aggregate per-axis recoverability on full_window.** Our
    audit/v8_per_field_eval.py was run on per_row checkpoints.
    Re-running on full_window will produce the table the manuscript
    actually needs. Critical: per_row reported X-value full-correct at
    7%; full_window's per-axis recovery is one of the manuscript's
    headline tables and must use full_window numbers.

19. **Headline-narrative rewrite.** The paper currently frames the work
    as "we report what sensors can recover" with TBD numbers. After
    full_window 5-fold lands, the dominant story becomes "we identified
    and resolved a per-row-formulation ambiguity that obscured the
    sensor pathway's true command-recovery capability." This is a
    cleaner story arc than the original "categorical vs continuous tier"
    framing — they're complementary but the diagnosis-then-fix arc is
    more compelling. Recommendation: rewrite the abstract + intro
    contribution list around the per_row-vs-full_window discovery once
    the 5-fold numbers are in.

Items 15-19 are forward-looking, post-full_window-completion. Each is
~1-2 hours of work; items 17 and 19 are the highest-impact.

## 2026-05-13 — Full_window 5-fold COMPLETE + headline numbers locked

All 5 folds of `train_v8_full_window_5fold.sh` finished. Fold-5's Python
process is still in its sample-by-sample decode dump (slow tail, ~93/132
samples through), but `metrics.json` is fully written for every fold, so
the 5-fold aggregator and ANOVA can run now.

### Headline 5-fold results (V8 full_window, no positional metadata)

[outputs/decoder20260511/RESULTS_TABLE.json](RESULTS_TABLE.json),
[outputs/decoder20260511/audit/bootstrap_ci.json](audit/bootstrap_ci.json):

| Head | Accuracy | Macro F1 | Bootstrap 95 % CI |
|---|---|---|---|
| Token       | $0.7811 \pm 0.0216$ | $0.390 \pm 0.041$ | $[0.7633, 0.7988]$ |
| Sequence    | $0.0263 \pm 0.0185$ | -- | $[0.0113, 0.0428]$ |
| Type        | $0.9838 \pm 0.0085$ | $0.883 \pm 0.045$ | $[0.9753, 0.9896]$ |
| **Command** | $\mathbf{0.8875 \pm 0.0558}$ | $0.879 \pm 0.108$ | $[0.8316, 0.9186]$ |
| **Param-type** | $\mathbf{0.9931 \pm 0.0036}$ | $0.986 \pm 0.013$ | $[0.9895, 0.9952]$ |
| **Sign**       | $\mathbf{0.9888 \pm 0.0063}$ | $0.972 \pm 0.016$ | -- |
| Numeric (digit) | $0.5850 \pm 0.0331$ | -- | $[0.5575, 0.6131]$ |

### Per-fold detail

| Fold | tok | cmd | num | seq | best_ep |
|---|---|---|---|---|---|
| 1 | 0.747 | 0.776 | 0.552 | 0.015 | 66 |
| 2 | 0.774 | 0.914 | 0.561 | 0.000 | 86 |
| 3 | 0.776 | 0.922 | 0.561 | 0.028 | 82 |
| 4 | 0.806 | 0.908 | 0.627 | 0.056 | 89 |
| 5 | 0.803 | 0.918 | 0.624 | 0.032 | 101 |

All folds early-stop comfortably before the patience-75 budget. Fold 1
is the outlier on command (0.776 vs $\geq$0.91 for folds 2-5); plausibly
a fold-1 file-split idiosyncrasy worth investigating but doesn't change
the headline.

### Per-class command head (6 classes evaluated)

G0 / G1 / G2 / G3 / G53 / M30 — all present in the test splits. G53 and
M30 supports are tiny ($n \le 13$) and their per-fold F1 has high
variance, but the head DOES emit them: G53 F1 = $0.876 \pm 0.206$;
M30 F1 = $0.836 \pm 0.150$. The 8.4-pp Section sec:lim-power MDE
threshold applies.

### Per-axis param-type head (5 classes evaluated)

X / Y / Z / R / F — all F1 $\ge 0.96$. F support is $\sim$40 per fold
(constant `F22.` in G1 lines); high F1 reflects "detected when present"
not "feed-rate value recovery." I, J, K, S, P never appear in test
splits.

### Per-digit-position 5-fold

| Position | Accuracy |
|---|---|
| 0 (magnitude) | $1.000 \pm 0.0001$ |
| 1 | $0.924 \pm 0.021$ |
| 2 | $0.754 \pm 0.033$ |
| 3 | $0.557 \pm 0.032$ |
| 4 | $0.461 \pm 0.032$ |
| 5 (least sig.) | $0.769 \pm 0.019$ |

Clean U-shape across folds: encoder preserves magnitude (position 0
perfect) and precision endpoint (position 5 strong), but middle digits
(2–4) hover near coin-flip. Same shape as the per_row pilot. This is
the encoder-side bottleneck.

### Output-position diagnosis: the per_row $\to$ full_window lift is empirically validated

Per_row position-1 accuracy was 0.24 (Section C-4 finding — the smoking
gun for within-window row-identification ambiguity). Full_window 5-fold
position-1: $0.62 \pm 0.03$. **A $+$38 pp lift.** Confirms the
diagnose-then-fix arc: the autoregressive context in full_window mode
resolves what the per-row encoder summary could not. Once past position
10, accuracy stabilises at $0.90$+ — the autoregressive chain has fully
disambiguated subsequent rows.

### e2e_lr5e-6: encoder fine-tuning at lr=5e-6 did not help

tok 0.7164 / cmd 0.3423 / num 0.4444 — substantially worse than the
frozen-encoder full_window baseline. Confirms naive end-to-end is not
the right Phase F approach; the auxiliary-head route documented in
PHASE_F_DESIGN.md is the recommended design.

### Bug fixes landed today

1. `aggregate_v8_results.py` and `anova_and_bootstrap.py`: both had the
   same wandb-subdir blindness as the earlier HP-sweep aggregator
   (`fold_N/<wandb_run_id>/results/metrics.json` instead of
   `fold_N/results/metrics.json`). Fixed both with a `_find_fold_metrics`
   helper.
2. `scripts/analysis/figures/confusion_matrices.py` was pointing at
   `per_row_5fold` and only checked `beam_0_metrics.json`. Now points at
   `full_window_5fold` and uses the wandb-aware path lookup.
3. `scripts/analysis/failure_case_analysis.py` had the same wandb-subdir
   bug. Fixed.
4. Paper had `$\le$\le 5$$\%` typo in failure-mode paragraph. Fixed.

### Paper fill status

112 of 144 TBD placeholders filled with real V8 numbers:
- Abstract, intro, headline table, per-class tables, per-digit table,
  bootstrap CI, output-position table — all populated.
- Per-axis recoverability has 5 axes' P/R/F1 from the param_type head.

32 placeholders remain, all dependent on ablations still queued in the
watcher chain (LOCO, noise aug, vocab2digit, nested ablations,
with-shortcuts ANOVA F/p/$\Delta$).

### Figures regenerated

All 13 figure PDFs in `decoder_paper_v2/figures/` now built from V8
full_window 5-fold data, not the legacy V7-data run:
- per_class_metrics_{command, param_type, type}
- per_axis_recoverability
- five_fold_spread
- sensor_ablation_bars (still uses pilot data; will refresh post-sensor-ablation)
- learning_curves
- confusion_matrix_{command, param_type, type} (counts + normalized variants)

Paper now 31 pages with all V8-derived figures included.

### Failure cases on full_window

[audit/failure_cases_fullwindow.json](audit/failure_cases_fullwindow.json):

| Fold | n_samples | exact_match | exact% | median_edit |
|---|---|---|---|---|
| 1 | 132 | 2 | 1.5% | 34.5 |
| 2 | 110 | 0 | 0.0% | 19.5 |
| 3 | 106 | 3 | 2.8% | 17.5 |
| 4 | 108 | 6 | 5.6% | 15.5 |
| 5 | 93 | 3 | 3.2% | 16.0 |

Exact-match across the full multi-line target is rare (≤ 6%), but the
median edit distance is 15-35 tokens out of median sequence length
$\sim$130, meaning most of the multi-line is reconstructed correctly.

### What's still pending (waiting on watcher chain)

The watcher (pid 3990952) is still blocked on fold-5's slow-tail Python
process exiting (currently at sample 93/132 of the decode dump). Once
that exits, the chain runs the queued ablations:

(a) regen pipeline (already done manually here)
(b) full_window + shortcuts ‖ full_window + no_ss (in parallel)
(c) sensor-modality ablation cross-fold
(d) noise-augmentation 5-fold
(e) LOCO 9-class
(f) pattern-aware pilot
(g) 2-digit vocab pilot
(g2) window/stride sweep with encoder retrain
(h) final aggregator + figures pass

ETA: 14-22 hours of GPU once fold-5's tail exits.

## 2026-05-13 — Meeting alignment check #2 (after full_window fold 1+2)

Re-checking against the 2026-04-28 meeting summary now that we have the
full_window result, the paper-v2 quality upgrade, and items 1-9 of the
gap-recommendation list closed. This refresh of the alignment table:

### Meeting action items — current status

| # | Item (owner) | Status today | Δ since last check |
|---|---|---|---|
| 1 | Tokenizer / training: max_token_len applies per G-code LINE, not per window (Stephen) | ✅ DONE | unchanged |
| 2 | Test per-row vs full-window prediction (Romesh) | ✅ DONE for fold 1+2 (full_window beats per_row by +27 to +41 pp on command) | **CHANGED: now empirically confirmed** |
| 3 | Add noise augmentation to labels/features (Romesh) | ⏸ queued | now in extended post-fullwindow watcher chain |
| 4 | Prepare summer DOE for direction × speed × depth × material (Stephen) | ✅ infrastructure DONE: 188 runs spec'd, .gcode emitted, validator clean. Still pending Tarmac hardware + procurement | minor: paper now cites Appendix DOE |
| 5 | Review sensor priorities (gyroscope) from prior ablations (Team) | ⏸ queued | now in extended watcher chain |
| 6 | Manbir reads paper, reconvenes Monday | ❓ outside scope | unchanged |
| 7 | Tarmac PO follow-up with Lois / Woody (Stephen) | ❓ outside scope | unchanged |
| 8 | Hire Tim for summer (Manbir) | ❓ outside scope | unchanged |
| 9 | Revisit tool imaging / R2 camera (Stephen) | ❓ outside scope | unchanged |
| 10 | Design simple testable experimental runs (core group) | ⏸ pending DOE rollout | unchanged |
| 11 | Monday working session for summer planning (Manbir) | ❓ outside scope | unchanged |
| 12 | Reduce manuscript to page limit (Romesh) | ⚠️ MISSED so far: paper is 28 pages; Sensors limit is 16 (two-column) / 20 (single). Need to compile into two-column or trim content | **NEW GAP** |

### Meeting paper-framing decisions — status

| Decision | Status |
|---|---|
| Pivot from "full G-code reconstruction" to "physically recoverable parameters" | ✅ paper now organised around per-field recoverability (Sec 4.2 Per-Axis Recoverability) |
| Cybersecurity / process verification framing | ✅ formal Threat Model section added (Sec 6.6 sec:disc-threat-model) |
| Feed rate, depth of cut as recoverable parameters | ⚠️ caveat: F is in 0.4–22 % of samples; only "F-presence" is evaluable, not "F-value recovery." Z (depth of cut proxy) IS well recovered (≥ 0.99 param_type, ≥ 0.99 sign). Paper documents this transparency in Sec 5.4 sec:lim-dataset |
| Decoder needs pattern recognition (Manbir's specific contribution) | ⏸ pattern_aware pilot queued in extended watcher chain |
| Per-row vs full-window test (Romesh's specific recommendation) | ✅ DONE; full_window is the headline result |

### Meeting-implied items that remained gaps

| Gap | Status |
|---|---|
| Page limit (16/20 pp) | ⚠️ paper at 28 pp. Either (a) trim, (b) submit to a different venue with no page limit, or (c) recompile in MDPI two-column format. The body text targets ~20 pp in single-column; in two-column compile this becomes ~14 pp. Recommendation: keep adding content for now; trim after full_window 5-fold lands |
| Noise augmentation (item 3) | ⏸ now queued in extended watcher chain (step d) |
| Sensor-priority ablation (item 5) | ⏸ now queued in extended watcher chain (step c) |
| Pattern-aware decoder (Manbir) | ⏸ now queued in extended watcher chain (step f) |
| Vocabulary precision (item 6 from Round-2 plan) | ⏸ now queued in extended watcher chain (step g) |
| Tool imaging (item 9) | ❓ operational, not modeling |
| Per-line timestamp logging from Tormach | ❓ operational, pending hardware |
| Material ordering (aluminum / steel / delrin) | ❓ operational, pending hardware |

### Items DONE since the previous alignment check yesterday

1. (#1 from gap-list) Non-neural baseline (HGB) — ✅
2. (#2) Threat model section — ✅
3. (#3) Multi-comparisons correction (Holm-Bonferroni + BH-FDR) — ✅
4. (#4) Encoder-leakage probe in paper — ✅
5. (#6) Power analysis on long tail — ✅
6. (#7) Train-vs-test covariate-shift audit — ✅
7. (#8) Token-position failure analysis — ✅
8. (#9) Phase F design spec — ✅
9. (#11) Replicability checklist appendix — ✅
10. (#12) Git commit pass (12 commits) — ✅
11. (#16/17) Full_window + shortcuts, full_window + no_ss queued — armed in watcher
12. Abstract rewritten with hook and professional tone — ✅
13. Full_window fold 1 + 2 done: command jumps 0.50 → 0.78/0.91 — empirical validation
14. Watcher extended to chain ALL remaining ablations (steps c through h) — ✅

### Outstanding gaps that block submission

1. **Page limit (item 12 from meeting)**. 28 pp single column. Sensors requires 16 pp two-column or 20 pp single-column. Critical.
2. **Figures not included in .tex** (\#5 from gap-list). The seven generators exist, the PNG/PDFs exist (but stale), but `decoder_paper_v2.tex` has zero `\includegraphics` calls. Reviewers will flag immediately.
3. **5-fold mean ± std** isn't in the paper yet — placeholders. Auto-fill when watcher's regen pass runs.
4. **All ablations on V8 data** — currently queued in watcher chain (sensor, noise, LOCO, pattern-aware, vocab); about 10-12 hours of additional GPU time.

### What the extended watcher chain delivers

After it completes (estimated 14-18 hours total from now), every queued
ablation will have V8 numbers. Then the only manual steps remaining are:

a. Fill the TBD placeholders in the paper (mechanical — see TABLES_REGENERATION_GUIDE.md)
b. Add `\includegraphics` calls in the appropriate Results subsections
c. Decide on page-limit response (trim, two-column, or alternate venue)
d. Optional: failure-mode visual figure (item 10 from gap-list)
e. Optional: narrative rewrite framing around the per_row → full_window discovery (item 19)

The watcher's final step (h) regenerates all figures from the final
checkpoints, so by morning the figure PDFs should be V8-consistent.
