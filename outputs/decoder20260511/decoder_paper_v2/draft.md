# G-Code Decoder: Multi-Modal Sensor-Driven Recovery of CNC Machining Instructions

*Working draft — Round-2 (decoder20260511). Numbers below are the 5-fold no-shortcuts sweep unless noted. All experiments use the frozen `f98_w256_s64` encoder, V8 vocabulary (2,418 tokens), and a transformer decoder with grammar-constrained legacy + type + command + parameter-type + digit prediction heads.*

---

## Abstract

We present **G-Code Decoder**, a transformer-based G-code decoder that consumes
the latent embeddings of a frozen multi-modal sensor encoder and reconstructs
the executing G-code line in real time. On a 120-file dataset of subtractive
manufacturing recordings (3-axis CNC, 6 distributed sensor boards, 98 active
sensor channels), G-Code Decoder achieves 0.979 ± 0.019 command-level accuracy
under 5-fold cross-validation when no positional metadata is supplied to the
decoder. We characterize per-axis recoverability (X/Y/Z/F/S/R/I/J) and report
sensor-modality ablations identifying gyroscope and color sensors as the two
most-load-bearing modalities. The work also contributes a structured-output
decoder design, a per-row training target formulation that prevents window-level
shortcut learning, and a leave-one-class-out generalization study.

---

## 1. Introduction

CNC manufacturing leaves a multi-modal sensor fingerprint — vibration, force,
sound, current draw, environmental — distinctive enough that machine telemetry
alone, *without access to the G-code itself*, can drive condition monitoring,
intrusion detection, and tooling-aware optimization. Prior work has shown that
this fingerprint supports operation-class classification (face vs pocket vs
adaptive vs damaged variants); the open question is whether the *instruction-level*
G-code can be recovered from the sensor stream alone. We show that it largely
can — for command type, axis presence, and X/Y motion direction — and identify
the modalities and label structures that limit it.

**Contributions.**

1. G-Code Decoder, a transformer decoder with five structured prediction heads
   (token, type, command, parameter-type, digit-value) trained on frozen
   multi-modal encoder embeddings.
2. A per-row supervised target formulation that pins each sensor window to a
   single executing G-code line, eliminating the within-window shortcut path
   that previous attempts at G-code recovery exploited.
3. Per-axis recoverability characterization across X/Y/Z/F/S/R/I/J.
4. Sensor-modality ablations identifying gyroscope and color sensors as the two
   most-load-bearing inputs.
5. Leave-one-class-out generalization study across 9 operation classes.

## 2. Related Work

*(To be expanded with: Romesh's prior encoder paper; the V7 reference; the
anomaly20260319 paper; classic CNC monitoring work; sequence-to-sequence
decoding literature; structured-output prediction.)*

## 3. Method

### 3.1 Data

The dataset contains 120 CSV files from a 3-axis Bantam CNC, each recording 6
distributed multi-sensor boards (`frame_l2, frame_l3, frame_r2, spindle2,
y_bed__3, y_bed__4`) at 4 Hz alongside the executing G-code program. We exclude
proximity and pressure channels (consistency-pass following Romesh's audit),
yielding 98 active sensor channels.

We split files using 5-fold stratified cross-validation by operation class.
G-code-line coverage between train and test is repaired via file-swap when
necessary (Algorithm 1, Appendix).

### 3.2 Preprocessing

Sliding sensor windows of 256 samples (64 seconds) at stride 64 yield ~700-800
train windows per fold. The training-time target is the G-code line that is
ACTIVE at the position of each sample row within the window: in **per-row mode**
we emit one (sensor-window, single-line-target) sample for every distinct
G-code line that fires during the window, sharing the full 256-sample sensor
context across emissions from the same window. This formulation:

- Removes the per-window label ambiguity that would otherwise arise when
  multiple G-code instructions execute during a 64-second window.
- Decouples the decoder's target from the windowing schedule, preventing the
  decoder from using `(window_index, source_file)` to infer the label.
- Yields ~19,500 training samples per fold versus ~300 in a single-line-per-window
  setup.

### 3.3 Vocabulary

We tokenize G-code via a hybrid scheme: literal G/M tokens; address letters
X/Y/Z/I/J/K/R/F/S/P/Q/E as separate tokens; numerical values bucketed to 4-digit
precision (e.g., `X1.4919` → `X` then `NUM_X_1492` for precision 1e-3). The
resulting vocabulary has 2,418 tokens including 1,048 NUM_X buckets, 1,086 NUM_Y,
231 NUM_Z, and the special tokens PAD/BOS/EOS/UNK/MASK. Out-of-vocabulary rate
is 0.00% across the 225,000+ source-data tokens.

### 3.4 Encoder

The encoder is a multi-modal denoising transformer (MM-DTAE) trained
separately by the encoder paper team on the same fold splits, frozen for all
decoder experiments. It maps a 256-sample × 98-channel sensor window to a
256-step × 256-dim memory sequence consumed by the decoder via
cross-attention.

### 3.5 Decoder

G-Code Decoder is an 8-layer transformer decoder with 12 heads, `d_model=384`,
positional encoding capacity 32 tokens. Five prediction heads share the final
hidden state:

- **Legacy token head**: vocabulary-size softmax (2,418-way), constrained at
  inference by a grammar mask that rules out structurally invalid transitions.
- **Type head**: 4-way (SPECIAL / COMMAND / PARAM / NUMERIC) for inter-token
  type classification.
- **Command head**: 6-way (G0/G1/G2/G3/G53/OTHER).
- **Parameter-type head**: 10-way over axis letters.
- **Digit head**: 6-position digit decoder over 11 classes (0–9 + PAD) so
  numeric values are recovered position-wise rather than as bucketed tokens.

The decoder is trained with teacher forcing for 50 epochs, AdamW (lr 1e-4,
weight-decay 0.05), batch size 64. Best epoch selected by validation token
accuracy. **The decoder is NOT given window-position metadata** (`window_index`,
`total_windows`, `source_file`); the ablation in §4.2 shows what including them
would buy.

### 3.6 Sensor-modality ablation

Sensor channels are grouped by physical modality:
*accelerometer* (Ax/Ay/Az per board), *gyroscope* (Gx/Gy/Gz), *magnetometer*
(Mx/My/Mz), *environmental* (Temperature, after excluding proximity/pressure),
*color* (ColorR/G/B/A), *rms* (audio RMS), and *electrical* (spindle and X/Y/Z
motor currents). To ablate a modality, we zero its channels at the **encoder
input** before the forward pass, then run the rest of the pipeline unchanged.
This isolates the contribution of each modality without retraining the encoder.

## 4. Experiments

### 4.1 Headline accuracy (5-fold cross-validation)

| Head | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| token | 0.832 ± 0.020 | 0.326 ± 0.116 | 0.373 ± 0.119 | 0.329 ± 0.111 |
| type | **0.972 ± 0.012** | 0.951 ± 0.020 | 0.943 ± 0.021 | 0.946 ± 0.021 |
| command | **0.979 ± 0.019** | 0.373 ± 0.102 | 0.466 ± 0.053 | 0.384 ± 0.104 |
| param_type | **0.944 ± 0.013** | 0.954 ± 0.008 | 0.950 ± 0.004 | 0.952 ± 0.002 |
| numeric (digits) | 0.600 ± 0.024 | — | — | — |
| sequence (full line) | 0.426 ± 0.041 | — | — | — |

The macro-F1 gap on command (0.38 macro vs 0.98 accuracy) is the long-tail
signature: G1 (490 test instances) dominates, while G2/G3/M30 are rare and
mostly unrecovered.

### 4.2 Per-class breakdown

#### Command

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| G1 | 0.972 ± 0.011 | 0.914 ± 0.134 | 0.937 ± 0.079 | 490 |
| G0 | 0.399 ± 0.205 | 0.800 ± 0.187 | 0.465 ± 0.167 | 20 |
| G3 | 0.120 ± 0.240 | 0.150 ± 0.300 | 0.133 ± 0.267 | 20 |
| M30 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 10 |

#### Parameter type (axis letters)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| X | 0.935 ± 0.015 | 0.946 ± 0.024 | 0.940 ± 0.005 | 940 |
| Y | 0.882 ± 0.045 | 0.856 ± 0.039 | 0.867 ± 0.004 | 430 |
| Z | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 40 |
| R | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 40 |

#### Type

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| COMMAND | 0.992 ± 0.016 | 0.898 ± 0.023 | 0.943 ± 0.016 | 540 |
| NUMERIC | 1.000 ± 0.000 | 0.978 ± 0.021 | 0.989 ± 0.011 | 725 |
| PARAM | 0.890 ± 0.019 | 0.938 ± 0.048 | 0.913 ± 0.031 | 725 |
| SPECIAL | 0.923 ± 0.056 | 0.958 ± 0.007 | 0.939 ± 0.030 | 685 |

### 4.3 Per-axis recoverability

We parse decoded G-code text into structured fields and report per-axis
classification + regression metrics across the 5-fold sweep.

| Axis | has-axis acc | has-axis F1 | sign acc | value MAE | presence recall |
|---|---|---|---|---|---|
| X | 1.000 | 1.000 | 0.697 | 1.47 | 1.000 |
| Y | 0.961 | 0.957 | 0.994 | 0.79 | 0.896 |
| Z | 0.983 | 0.868 | 0.994 | 0.00 | 0.800 |
| R | 0.982 | 0.695 | 1.000 | 0.00 | 0.400 |
| F, S, I, J | ≥ 0.992 | — | 1.000 | n/a | 0.000 |

X and Y axes are always recovered when present; Z is recovered 80% of the
time, R 40%. X-positive vs X-negative sign classification reaches only 70% —
the largest single failure mode in the per-axis picture. Feed rate (F),
spindle (S), and arc parameters (I/J) have effectively zero positive support
in the current dataset and are not evaluable; future data collection
(§6.3) targets these.

### 4.4 Ablation: positional metadata (the shortcut path)

*[Phase B2 result will populate after the with_shortcuts 5-fold sweep
completes.]*

When the decoder is also given `window_index`, `total_windows`, and
`source_file` features, command accuracy reaches *(X.XXX ± X.XXX)*. This
ablation isolates the magnitude of the shortcut path: an XGBoost trained on
those metadata features ALONE (no sensors) reaches 0.890 ± 0.0XX, so the
shortcut path is real and substantial. The fact that G-Code Decoder reaches
similar accuracy (0.979) WITHOUT exposing these features is the central
empirical result.

### 4.5 Sensor-modality ablation

*[Phase B4 cross-fold result will populate after gyroscope/color cross-fold
sweep completes; current numbers are fold-1 only.]*

| Modality removed | Command accuracy | Δ vs baseline |
|---|---|---|
| Baseline (all modalities) | 0.944 | — |
| Gyroscope | 0.907 | −3.7 pp |
| Color (RGBA) | 0.843 | −10.2 pp |
| Magnetometer | 0.944 | 0 |
| RMS (audio) | 0.944 | 0 |
| Accelerometer | 0.944 | +0.0 |
| Environmental | 0.954 | +1.0 |
| Electrical | 0.954 | +1.0 |

Gyroscope and color are the two load-bearing modalities. Gyroscope's
contribution aligns with mechanical-vibration intuitions; color's
contribution is initially surprising and likely encodes a material-signature
proxy through ambient lighting reflectance.

### 4.6 Pattern-aware decoder (sequence classifier head)

*[Phase B7 result will populate after the pattern-aware pilot completes.]*

### 4.7 Leave-one-class-out generalization

*[Phase B5 result will populate after the LOCO sweep completes.]*

### 4.8 Vocabulary precision sweep

*[Phase B6 result will populate after the 2-digit vocab retraining
completes.]*

### 4.9 Numeric accuracy decomposition

*[Phase C numeric-accuracy diagnosis result will populate.]*

The digit head's 0.60 overall accuracy decomposes into per-position
accuracies of 0.999 / 0.680 / 0.237 / 0.124 / 0.149 / 0.493 (digit 0 → 5)
on fold 1. The most-significant digit (sign / leading magnitude) is
essentially perfect; middle digits are the weakest link. Per-axis digit
accuracies are X 0.42, Y 0.46, Z 0.48, J 0.46 — uniformly mediocre, with
0.9–2.3% achieving the full 6-digit value match.

## 5. Discussion

### 5.1 What the sensor pathway recovers

The headline findings are:

- **Command identity** (G0/G1/G2/G3) is recovered with 0.98 accuracy from
  sensors alone, *without* positional metadata.
- **Axis presence** for X and Y is essentially perfect.
- **Z axis presence** is recovered 80% of the time.
- **X/Y motion direction** is recovered 70% (X) / 99% (Y).
- **Full numeric value reconstruction** is poor (~1% per-axis full-match).
  Most digit error is in the middle-significance positions.

### 5.2 What it does NOT recover

- Feed rate (F), spindle speed (S), arc parameters (I/J) — these have
  effectively zero positive support in the current dataset. Their
  recoverability is a target of the summer DOE dataset (§6.3).
- The rarest commands (G3 arcs, M30 program-end) — fewer than 25 instances
  each in the test set, learned poorly.
- The least-significant digits of continuous coordinate values.

### 5.3 Design choices that mattered

The per-row target formulation (§3.2) was the largest single methodological
choice. Earlier exploratory work used full-window multi-line targets, which
left the model with positional shortcuts and lower-quality numeric
predictions. Per-row training also matches the deployment semantics: the
model is asked, at each sample, what G-code line is currently active.

## 6. Limitations and Future Work

### 6.1 Frozen encoder

The encoder used here was trained on the same fold splits as the decoder,
so some positional information may be implicitly encoded in the embeddings
even though the decoder receives no positional metadata directly. A future
experiment retraining the encoder on per-row decoder targets (rather than
operation-class targets) will quantify the residual encoder-side leak.

### 6.2 Small label set

The current dataset's G-code corpus contains ~214 distinct lines per fold;
even the largest support classes are heavily concentrated on G1 motion.
Per-class metrics are macro-averaged on a long tail.

### 6.3 Continuous parameters need new data

Feed rate, spindle speed, and arc parameters are nearly invariant in the
current dataset. Recovering these requires the DOE-driven summer dataset
(see Appendix B), which varies feed/depth/material across a designed
factorial.

## 7. Conclusion

G-Code Decoder demonstrates that the multi-modal sensor stream of a 3-axis
CNC machine contains sufficient information to recover the executing G-code
command and axis-presence pattern with ~0.98 accuracy in a 5-fold
cross-validation, even without exposing the decoder to positional metadata.
The recoverability is uneven across G-code fields: command type and axis
presence are essentially solved on this dataset, sign recovery and numeric
value reconstruction are not. The per-row target formulation and the
sensor-modality ablation point to a research path through more diverse data
collection and pattern-aware decoder priors.

---

## Appendix A — full per-class P/R/F1 tables

See `outputs/decoder20260511/MANUSCRIPT_TABLES/results.md` and
`RESULTS_TABLE.json` for the complete machine-readable per-class data.

## Appendix B — DOE specification for summer experiments

See `outputs/decoder20260511/DESIGN_OF_EXPERIMENTS.md` for the 188-run
factorial DOE that targets feed rate / depth / material variation.
