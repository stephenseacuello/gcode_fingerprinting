# Phase F — Encoder Retrain Design Specification

**Status: 2026-05-13. Drafted alongside the post-full_window regeneration. The
encoder retrain itself is on hold until the full_window 5-fold is in. This
document is the blueprint for that future experiment.**

## Why Phase F is no longer a stretch goal

The 2026-05-12 encoder audit established three findings that locate the
remaining performance gap at the encoder:

1. **The encoder was trained on only 303 samples per fold** (the V7
   dedup'd dataset). Reached 100% training accuracy in 1 minute on a
   9-class operation-type task. Severe memorization risk.
2. **The encoder's training objective is dominated by classification**
   (`loss = cls_loss + 0.1 * recon_loss`; see
   `scripts/evaluation/run_9class_direct.py:491-492`). The reconstruction
   weight is 10× smaller than the classification weight; the encoder
   learns to encode operation type, not to preserve fine-grained per-row
   sensor structure.
3. **The probe ceiling matches the decoder's per_row ceiling.** Linear/MLP
   probes on the frozen encoder memory plateau at ~0.77 on the 5-class
   command task; the per-row decoder plateau is ~0.50. The 27-pp gap that
   full_window mode recovers is the within-window row-disambiguation
   gap. The remaining gap between full_window's ~0.78 cmd and the
   theoretical 1.0 ceiling is plausibly bounded by the encoder's loss
   of per-row sensor structure during pre-training.

If full_window 5-fold confirms the headline result (cmd ≈ 0.78, num ≈
0.55), the next natural lever is to **retrain the encoder with
auxiliary row-level supervision** — replacing the 9-class operation-type
objective with one that explicitly preserves the structure the decoder
needs.

## Design

### Encoder architecture

Same `MM_DTAE_LSTM` as the published encoder (`encoder_paper`):
- Per-modality projection (7 modality groups → d=256)
- Learned modality-weighting gates
- Multi-head self-attention stack
- Two-layer LSTM for temporal modeling
- Reconstruction head (denoising autoencoder)

**No architectural changes** in the encoder itself. We retain comparability
with the published encoder paper.

### Training data

V8 per_row preprocessed data (the corrected pipeline):
- 19,584 train / 7,273 val / 7,937 test samples per fold (65× more than
  the original encoder)
- File-level 5-fold stratified split (matched to decoder folds)
- Already on disk at `outputs/decoder20260511/preprocessed_f98/per_row/`

### Loss function (the key change)

```
loss = λ_cls * loss_cls_9class
     + λ_recon * loss_recon_denoising
     + λ_row_cmd * loss_row_command           # NEW
     + λ_row_param * loss_row_param_presence  # NEW
     + λ_row_sign * loss_row_motion_sign      # NEW
     + λ_row_value * loss_row_numeric_mse     # NEW
```

Weighting (initial proposal, sweep around these in Stage-F sweep):
- λ_cls = 1.0 (matches published encoder)
- λ_recon = 1.0 (boost from 0.1 — reconstruction should be a co-equal objective)
- λ_row_cmd = 0.5 (row-level G-command from G-code label)
- λ_row_param = 0.3
- λ_row_sign = 0.3
- λ_row_value = 0.5

The four `loss_row_*` terms add lightweight prediction heads to the
encoder, each consuming the encoder's pooled or sequence output and
predicting a row-level field. The heads do NOT replace the decoder; they
exist only during encoder pre-training to shape the embedding space. At
deployment time, the encoder is frozen and the decoder consumes the
encoder memory as before.

This is exactly the structured-supervision approach the encoder paper
left out (its task was operation classification only).

### Architecture of the new auxiliary heads

Each auxiliary row-level head is a small MLP (one hidden layer, 128 dim):
- `head_row_cmd`: mean-pooled encoder memory → 6-way softmax (G0/G1/G2/G3/M30/none)
- `head_row_param`: mean-pooled → 8-way binary multi-task (has-X, has-Y, has-Z, has-F, has-S, has-R, has-I, has-J)
- `head_row_sign`: mean-pooled → 9-way (3 axes × 3 sign classes, multi-label)
- `head_row_value`: mean-pooled → 4-way MSE (X, Y, Z, F values; NaN-masked loss)

Pooling: the auxiliary heads operate on the same mean-pooled
representation the decoder's cross-attention consumes after sequence-dim
reduction. We could alternatively use a [CLS] token; mean-pooling is
simpler and matches the encoder paper's classification head.

### Training schedule

- Optimizer: AdamW, lr=1e-3 (matches published encoder)
- Schedule: linear warmup 10 epochs + cosine annealing 30 epochs
- Batch size: 32 (down from 64 due to memory)
- Modality dropout: 0.1 (matches published encoder)
- Label smoothing: 0.1
- Early stopping on val on a composite of (cls_acc + row_cmd_acc) /2
- Patience 15

Estimated training time: ~1 minute/fold × 65× more samples × 4 auxiliary
heads ≈ 5-10 minutes/fold. Total Phase F encoder retrain (5 folds): ~30-60
minutes.

### Evaluation protocol

For each fold:
1. Train the new encoder under the loss above.
2. Compare encoder-only metrics against the published encoder:
   - 9-class operation accuracy on per_row test (should match or exceed
     93.6% — adding row-level supervision should not hurt operation
     classification)
   - Probe ceiling on per-row command (Holm-Bonferroni-corrected ANOVA
     against the published encoder's probe baseline)
3. Freeze the new encoder. Retrain the decoder with the composite-winner
   config (Stage-1 architecture + Stage-2 scheduled sampling) on V8
   full_window 5-fold using the new encoder.
4. Compare decoder metrics against the current full_window 5-fold:
   - Token, command, numeric accuracy mean ± std across folds
   - Per-axis recoverability (the central manuscript table)
   - Failure-mode classification — does the residual error redistribute
     across the 4 modes (dropped/wrong/hallucinated command / value-only)?

### Decision rule

Three possible outcomes inform whether Phase F changes the manuscript
narrative:

1. **Phase F lifts headline command by > 5 pp**: include the retrained-
   encoder result as the headline; the "encoder retraining with row-
   level supervision is the right path forward" claim is empirically
   supported.
2. **Phase F lifts headline by 1-5 pp**: include as a confirmatory
   ablation, keep the current full_window result as the headline, frame
   Phase F as evidence that further encoder work is warranted but the
   present results are not bottlenecked at the encoder.
3. **Phase F lifts headline by < 1 pp or harms it**: report as a clean
   negative result. The encoder is not the load-bearing component for
   the current performance ceiling; the next leverage point is data
   diversity (Appendix~\ref{app:doe}).

In all three outcomes the manuscript gains a publishable answer to the
"is the encoder the bottleneck?" question that has hung over this work
since the 2026-04-28 meeting.

## Files to create when Phase F runs

- `scripts/experiments/train_v8_encoder_phaseF.py` — encoder retrain
  driver with the new auxiliary heads
- `src/miracle/model/encoder_phaseF_heads.py` — the four small MLP heads
  appended to the encoder during pre-training
- `outputs/decoder20260511/encoder_v8/fold_{1..5}/` — checkpoint dir
- `outputs/decoder20260511/checkpoints/per_row_5fold_v8encoder/` and
  `full_window_5fold_v8encoder/` — decoder retrain on the new encoder
- `outputs/decoder20260511/audit/phase_f_comparison.json` — encoder-vs-
  decoder metric comparison vs. the v0 encoder

## Compute estimate

- Encoder retrain: ~30-60 min × 5 folds = 2-5 hours
- Decoder retrain on new encoder: ~3 hours (5 folds × 30 min)
- Phase C analyses on new checkpoints: ~30 min
- Aggregator + paper recompile: ~30 min

**Total: 6-9 hours of GPU + 30 min of CPU. Achievable in one overnight
run.**

## Pre-registration

Before running Phase F, the following predictions are written down:

1. The retrained encoder will reach ≥ 93% on 9-class operation type (no
   regression vs. published).
2. The per-row command probe ceiling will lift from ~0.77 to ≥ 0.85.
3. The decoder retrained on the new encoder will reach ≥ 0.80 command
   accuracy on the full_window 5-fold.
4. Per-axis numeric MAE will drop by ≥ 10% relative on X and Y.

If any of (2-4) fail to materialise, the design has a problem with its
auxiliary supervision and the result becomes a negative finding rather
than a positive one. Either result is informative.
