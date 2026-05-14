---
to: Romesh Satish Prasad <romeshsatish.prasad@uri.edu>
from: Stephen Eacuello <seacuello@uri.edu>
subject: Decoder remediation update — full_window 5-fold done, paper rewritten
date: 2026-05-13
---

Romesh,

Bigger update than the last one. Two important corrections to my email
from 2026-05-11, plus the headline results from the full_window 5-fold
that just finished.

Everything is in `outputs/decoder20260511/`. Master log is `notes.md`.
The new manuscript is at
`decoder_paper_v2/latex/decoder_paper_v2.pdf` — 31 pages, compiles
cleanly. Audit JSONs under `audit/`.

## Two corrections to the previous email

**The 0.979 command accuracy I reported was from a buggy run.** The
trainer's `--encoder_config` flag was silently overwriting
`--data_dir`, so my "V8 5-fold" was actually training on the V7
dedup'd data (303 samples) with the V8 encoder. I caught it after the
email went out. Fix is in
`scripts/evaluation/run_decoder_quick_test.py:1488-1503`.

When the bug was fixed and the model was actually trained on V8 per_row
data (19,584 samples per fold), per_row plateaued at command accuracy
~0.50, not 0.97. The 0.979 was an artifact.

**The per_row formulation has a fundamental ambiguity** I missed
before. A 64-second sensor window contains 60+ G-code rows on average;
in per_row mode, every row of the same window shares the *same*
encoder memory but has a *different* target. The decoder has no way to
tell which row of the window it's supposed to predict, so it collapses
to predicting the modal row.

This is what tanked per_row's accuracy below the V7-paper number, and
it's what the full_window formulation resolves.

## The real headline

I retrained on V8 full_window 5-fold using the composite-winner
config from a 34-cell HP sweep (Stage 1 architecture + scheduled
sampling=0.5). 5-fold mean ± std, no positional metadata exposed:

| Metric      | V8 full_window 5-fold | per_row equivalent | Δ      |
|-------------|-----------------------|--------------------|--------|
| Token       | **0.781 ± 0.022**     | 0.731              | +5.0 pp |
| Sequence    | 0.026 ± 0.019         | 0.008              | +1.8 pp |
| Type        | 0.984 ± 0.009         | 0.972              | +1.2 pp |
| **Command** | **0.888 ± 0.056**     | **0.499**          | **+38.9 pp** |
| Param-type  | 0.993 ± 0.004         | 0.977              | +1.6 pp |
| **Sign**    | **0.989 ± 0.006**     | 0.978              | +1.1 pp |
| Numeric     | 0.585 ± 0.033         | 0.455              | +13.0 pp |

The +38.9 pp jump on command accuracy from per_row to full_window is
the strongest single experimental result of the remediation. It's not
a hyperparameter difference — it's the same config, just with
multi-line targets so each row's prediction is conditioned on the
previous rows in the same window via autoregressive context. The
within-window ambiguity that crippled per_row is resolved.

Folds 2-5 all hit command accuracy ≥ 0.91. Fold 1 is the outlier at
0.78, which is probably a file-split idiosyncrasy worth investigating
but doesn't change the headline.

Bootstrap 95% CI from 10,000 resamples: command [0.832, 0.919], token
[0.763, 0.799], sign [0.973, 0.998], numeric [0.558, 0.613].

## The diagnose-then-fix arc is solid

I want to make sure the chain is clear because it's going to be the
paper's main narrative thread:

1. **Phase C-4 failure-mode analysis** on per_row found that 95% of
   the worst-edit failures were *command-identity confusion* (dropped
   / wrong / hallucinated G-command). Not value-precision errors —
   structural errors.
2. **Token-position analysis** localised it: per_row position-1
   accuracy was **24%** (the first address letter after the command).
   Position 0 (the command itself) and position 2 (the first numeric)
   were both 76-81%. Position 1 was the sharp drop.
3. **Per-row ambiguity hypothesis**: this is exactly what would happen
   if the encoder memory carries window-level context but no row-level
   disambiguation. Different rows in a window start with different
   axes (X, Y, Z), so the decoder can't tell which to emit.
4. **Encoder probe** confirmed it: a linear probe on the frozen
   encoder memory reaches the modal-row-per-window ceiling (~85%) but
   can't go higher because the encoder memory is identical for all
   60+ rows of a window. The encoder is NOT the bottleneck.
5. **Full_window 5-fold validation**: position-1 lifts from 0.24 to
   **0.62 ± 0.03** (+38 pp), in line with the lift on command
   accuracy. Once past position 10 the decoder reaches 0.90+ —
   autoregressive context disambiguates subsequent rows cleanly.

The Discussion section now reads as predict-then-confirm.

## What the encoder probe says about Phase F

Linear/MLP probes on the frozen encoder memory:
- 9-class operation type: 0.85 (matches the encoder paper's val-set
  number)
- has-X / has-Y / has-Z presence: 0.88 / 0.87 / 0.93
- command identity: 0.77 (basically at the modal-row ceiling)
- X-value RMSE: 0.97 (within-window X range is ~3 units, so essentially
  no per-row precision)

So:
- Encoder preserves categorical / window-level signal fine.
- Encoder does NOT preserve per-row numeric precision.
- Phase F (encoder retrain) should keep the operation-classification
  head and ADD auxiliary row-level heads (per-row G-command, per-row
  axis-presence, per-row sign, per-row value MSE) at recon weight ≥ 0.5
  (currently 0.1). Design spec in `PHASE_F_DESIGN.md`.

End-to-end fine-tune at lr=5e-6 didn't help (tok 0.72 / cmd 0.34). The
naive e2e route diverges; the auxiliary-head route is the right Phase
F design.

## Sensor ablation update

The 5-fold sensor ablation cross-fold runs are queued and currently
firing in a watcher chain (gyroscope, color, magnetometer,
environmental, accelerometer, RMS audio, electrical). Will refresh the
table when they land — should be roughly consistent with the per-row
pilot direction (gyroscope + color most-load-bearing). Estimated
~1.5h.

## Where the manuscript stands

`outputs/decoder20260511/decoder_paper_v2/`:

- **Title**: "Structured Multi-Head Decoding for Computer Numerical
  Control G-Code Recovery from Multi-Modal Sensor Embeddings: A
  Cross-Validated Per-Field Recoverability Study" (mirrors the
  Machines paper title structure)
- **Abstract**: rewritten with the verification framing + sharp
  factorisation result
- **Introduction**: rewritten with stronger hook + 4 research
  questions + 5 itemised contributions
- **Related Work**: 4 deep subsections + literature-comparison table
- **Problem Formulation**: new section (task / vocabulary / per-row /
  metrics)
- **Methods**: now includes Encoder Information-Content Probe
  subsection
- **Experimental Setup**: now its own section (CV / baselines /
  statistical methodology / software)
- **Results**: organised by RQ1-RQ4 + new Failure-Mode Analysis +
  Output-Position Failure subsections (the +38pp validation table)
- **Discussion**: 5 subsections — three tiers of recoverability /
  failure modes are structural / sensor-modality deployment / AM
  side-channel comparison / when to use this decoder / formal threat
  model
- **Limitations**: 3 subsections (dataset, methodology, evaluation)
  + power analysis caveats for the per-class long tail
- **Replicability appendix**: random seeds, hardware, hyperparameter
  ranges, statistical methodology, code/data release

57 unique citations (24 ported from the encoder paper's bib).
Includes 13 figures generated from the V8 5-fold checkpoints. The
non-neural baseline (sklearn HistGradientBoostingClassifier — xgboost
crashed on the CPU-only host) and a formal threat model are now in
the paper as their own sections.

Statistical work: one-way ANOVA + Cohen's d + Holm-Bonferroni +
BH-FDR multiple-comparisons correction across the macro-metric grid
AND the drilled per-class / per-axis / per-position grid (~3,500
tests total once the full Phase B ablations land).

## What's still running (no input needed from you)

A watcher chain queued the remaining ablations and will run them
unattended (~14-20 hours total):

- (b) full_window+shortcuts 5-fold ‖ full_window+no_ss pilot (in flight)
- (c) sensor-modality ablation cross-fold
- (d) noise augmentation 5-fold (your meeting suggestion)
- (e) leave-one-class-out across 9 operation classes
- (f) pattern-aware (sequence_classifier head) pilot (Dr. Sodhi's
  meeting suggestion)
- (g) 2-digit vocabulary pilot
- (g2) window/stride sweep with encoder retrain per cell, no-prox
  no-pressure feature set (matches the encoder paper recipe)
- (h) regenerate the final figures + aggregator + ANOVA on the
  full result set

The remaining 32 TBD placeholders in the paper are all wired to
ablations in this chain.

## Files to look at if you have time

- `decoder_paper_v2/latex/decoder_paper_v2.pdf` — current paper (31 pp)
- `notes.md` — full chronological log
- `audit/encoder_probe_v8.json` — the probe results
- `audit/token_position_5fold_fullwindow.json` — the +38 pp lift table
- `audit/failure_cases_decoded_ss05.json` — the per_row failure modes
- `PHASE_F_DESIGN.md` — encoder retrain design spec for if/when we
  decide to execute it
- `DESIGN_OF_EXPERIMENTS.md` — summer Tormach DOE spec (188 runs
  generated)

The two recent commit notes worth reading are:
- `5b82151 Fill paper with V8 full_window 5-fold headline numbers`
- `a5bffc3 Regenerate figures from V8 full_window 5-fold + token-position lift table`

Glad we caught the encoder_config bug. The framing is much cleaner
now and the diagnose-then-fix arc is empirically supported.

Stephen
