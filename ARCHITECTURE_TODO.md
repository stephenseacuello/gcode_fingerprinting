# Architecture Improvement Ideas - G-code Fingerprinting

## Context

Current best: 84.3% token accuracy (all 12 sensors), but single-sensor xa_motor achieves 94.6%. Adding sensors hurts. PCA confirms heavy redundancy (47 components capture 95% of 296 features). Frame sensors share identical Pressure/Temperature/Ax signals (r=1.00). The architecture needs to handle multi-sensor fusion without degradation.

---

## 1. Per-Sensor Encoding

**Problem:** All 296 features are concatenated into a flat vector. The encoder has no notion of sensor identity -- it must learn from data that features 0-16 come from xa_motor and features 17-33 come from frame_r1.

**Solution:** Give each sensor its own encoder path (17 features -> compact embedding) before fusion.

**Variants:**
- **Shared weights:** All 12 sensors use the same MLP (17 -> 64 dims). Treats sensors as interchangeable. Fewer parameters, but cannot specialize per sensor.
- **Per-sensor weights:** Each sensor gets its own MLP. Can learn that xa_motor's Ax differs from y_bed__1's Ax. 12x the parameters but data is large enough.
- **Hybrid:** Shared base layer + sensor-specific residual connection. Shared structure captures common patterns, residual captures sensor-specific quirks.

**Expected impact:** High -- directly addresses the proven redundancy/degradation problem.

---

## 2. Cross-Sensor Attention

**Problem:** After encoding sensors, naive concatenation or averaging doesn't capture inter-sensor relationships.

**Solution:** After per-sensor encoding, apply multi-head attention where each sensor embedding can attend to all others. The model learns relationships like "high vibration on frame_r1 AND high current on xa_motor = informative for this operation."

**Details:**
- Add learnable sensor position embeddings so the model knows which sensor is which
- Could use a small number of attention layers (1-2) since there are only 12 sensors
- Machine features (motor electrical, positions, controller) get their own projection and participate in attention as additional "sensor" tokens

**Expected impact:** High when combined with per-sensor encoding.

---

## 3. Gating / Sensor Selection

**Problem:** Different G-code operations may benefit from different sensor subsets. A rapid traverse (G00) might depend on position sensors, while a cutting operation (G01) might depend on vibration/RMS.

**Solution:** A lightweight gating network that learns per-sample soft weights over sensors.

**Variants:**
- **Simple sigmoid gate:** Per-sensor scalar weight, conditioned on a global feature summary. Cheapest option.
- **Operation-conditioned gating:** Gate weights depend on the operation type embedding. Different G-code commands select different sensor subsets.
- **Mixture-of-experts:** Multiple sensor fusion strategies, with a router selecting which to use per sample.

**Expected impact:** Medium -- essentially a learned version of our manual ablation experiments.

---

## 4. Temporal Architecture Changes

**Current:** Multiscale 1D convolutions (4 scales, different kernel sizes) over [T=64, D=296], followed by attention-based pooling.

### 4a. Transformer Encoder Over Time
Replace conv backbone with self-attention over the 64 timesteps. Each timestep's feature vector becomes a token. Self-attention can directly relate timestep 1 to timestep 64 in a single layer, unlike convolutions which are local.

Note: 64 tokens is small -- convolutions may be fine at this scale. This matters more if we increase sequence length.

### 4b. Temporal-Sensor Factorized Attention
Reshape input from [T=64, D=296] to [T=64, S=12, M=17]. Then:
- **Step 1 (temporal):** For each sensor independently, self-attention across 64 timesteps
- **Step 2 (cross-sensor):** At each timestep (or after pooling), attention across 12 sensors

This is the ViViT/TimeSformer factorization from video transformers. Each attention step is smaller and more interpretable -- can inspect which timesteps and which sensors the model attends to.

### 4c. Patch-Based Temporal Encoding (PatchTST)
Group 64 timesteps into patches (e.g., 8 patches of 8 timesteps). Each patch embedded as a single token. Benefits:
- Reduces sequence length (8 tokens instead of 64), cheaper attention
- Forces local temporal summarization
- Strong results on time-series benchmarks

**Expected impact:** Medium -- unclear if 64 timesteps is long enough to need this.

---

## 5. Decoder Architecture Changes

### 5a. Multi-Source Cross-Attention
Currently the decoder cross-attends to a single encoder memory. With per-sensor encoding, could have separate cross-attention heads per sensor, letting the decoder dynamically focus on whichever sensor is most informative for the current token.

### 5b. Operation-Conditioned Decoding
The model already receives operation_type, but could make it a stronger prior -- different decoder heads or weight matrices per operation type. G00 (rapid) vs G01 (linear cut) have fundamentally different sensor signatures and parameter structures.

### 5c. Two-Stage Decoding
Separate the "what operation" question from the "what values" question:
- **Stage 1:** Predict G-code command structure (G01 X_ Y_ Z_ F_) -- command and parameter names only
- **Stage 2:** Given predicted structure + sensor data, fill in numeric values with a regression head or specialized numeric decoder

Motivation: Our worst performance is numeric tokens (54.9% F1 vs 84.7% for commands). Classification and regression are different tasks sharing one decoder.

**Expected impact:** High for two-stage, medium for others.

---

## 6. Multi-Task Learning

### 6a. Operation Classification Head
Auxiliary task: predict which G-code command (G00, G01, G02, etc.) from sensor data alone. Easier than full sequence prediction, forces the encoder to learn operationally meaningful features.

### 6b. Sensor Reconstruction (Autoencoder)
Encode sensor data, then decode back to raw sensor values. Regularizes the latent space and ensures no information is discarded prematurely.

### 6c. Parameter Regression Head
Alongside token prediction, directly regress numeric X/Y/Z/F values from sensor data. Numeric tokens are our weakest category -- a continuous regression head may help.

**Expected impact:** Medium -- better encoder representations benefit all downstream tasks.

---

## 7. Loss Function Changes

### 7a. Focal Loss
Downweights easy examples (command tokens at 84.7% F1), focuses training on hard examples (numeric tokens at 54.9% F1). Drop-in replacement for cross-entropy.

### 7b. Per-Token-Type Loss Weighting
Different loss weights for command vs parameter vs numeric tokens. Upweight numeric tokens since they're hardest and most important for accurate G-code reconstruction.

### 7c. Sequence-Level Losses
Add BLEU or edit distance as a reward signal via REINFORCE or minimum risk training. Optimizes what actually matters (full-sequence correctness) rather than per-token accuracy.

**Expected impact:** Medium for focal/weighting (easy wins), medium for sequence-level (complex to implement).

---

## 8. Input Feature Engineering

### 8a. Drop Non-Functional Sensors
Remove 68 features from 4 non-functional sensors. PCA shows this saves only 3 components (47 -> 44), so impact may be small, but it's a clean experiment that reduces input from 296 to 228 dimensions.

### 8b. PCA-Reduced Input
Feed 47 PCA components instead of 296 raw features. Pre-compressed, decorrelated input. Downside: loses interpretability, requires fixed PCA transform.

### 8c. Per-Modality Aggregation
Instead of 12 sensors x 17 modalities, compute summary statistics across sensors for each modality (mean, std, min, max of Ax across all 12 sensors). Reduces 204 sensor features to 17 x 4 = 68. Aggressive but directly addresses redundancy.

**Expected impact:** Low-Medium -- preprocessing changes, not architectural.

---

## 9. Data Augmentation

### 9a. Sensor Dropout
Randomly zero out entire sensors during training. Forces the model not to over-rely on any single sensor. Essentially dropout at the sensor level rather than the neuron level.

### 9b. Temporal Jitter
Slightly shift or stretch the 64-timestep window. Makes the model robust to temporal alignment variation.

### 9c. Noise Injection
Add Gaussian noise per sensor, scaled to each sensor's observed variance. Regularizes and simulates real-world sensor noise variation.

**Expected impact:** Medium -- free regularization, especially sensor dropout.

---

## 10. Training Changes

### 10a. Longer Training (500+ epochs)
Current models train ~100 epochs with early stopping (patience=20). Single-sensor models converging higher than multi-sensor suggests the multi-sensor model may need more time to learn fusion. Cheapest experiment -- just time on GPU.

### 10b. Curriculum Learning
Train on easy operations first (G00/G01 with simple parameters), then introduce harder ones (arcs G02/G03, complex numeric values). Gradual difficulty increase.

### 10c. Contrastive Pre-Training
Before seq2seq training, pre-train encoder with contrastive loss: same-operation sensor windows pulled together, different-operation windows pushed apart. Gives encoder better representations before decoding begins.

**Expected impact:** Medium-High for longer training, medium for curriculum/contrastive.

---

## Priority Matrix

| # | Idea | Effort | Expected Impact | Dependencies |
|---|------|--------|----------------|--------------|
| 10a | Longer training (500+ epochs) | Trivial | Medium-High | None (just GPU time) |
| 8a | Drop non-functional sensors | Trivial | Low | None |
| 9a | Sensor dropout augmentation | Low | Medium | None |
| 7a | Focal loss / per-type weighting | Low | Medium | None |
| 1 | Per-sensor encoding | Medium | High | None |
| 2 | Cross-sensor attention | Medium | High | Builds on #1 |
| 3 | Gating / sensor selection | Medium | Medium | Builds on #1 |
| 6a | Operation classification head | Medium | Medium | None |
| 5c | Two-stage decoding | Medium-High | High | None |
| 4b | Temporal-sensor factorization | High | Medium | Builds on #1 |
| 7c | Sequence-level losses | High | Medium | None |
| 10c | Contrastive pre-training | High | Medium | None |

## Recommended Implementation Order

**Phase A (quick wins, no architecture changes):**
1. Longer training (500+ epochs)
2. Drop non-functional sensors
3. Sensor dropout augmentation
4. Focal loss / per-token-type weighting

**Phase B (encoder overhaul):**
5. Per-sensor encoding (hybrid variant)
6. Cross-sensor attention
7. Gating mechanism

**Phase C (decoder improvements):**
8. Two-stage decoding (command structure then numeric values)
9. Operation-conditioned decoding

**Phase D (advanced, if needed):**
10. Temporal-sensor factorization
11. Contrastive pre-training
12. Sequence-level losses
