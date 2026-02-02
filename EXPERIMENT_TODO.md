# Experiment TODO - G-code Fingerprinting

> **Encoder:** All experiments use the MM_DTAE_LSTM encoder from the verified 100% pipeline.
> Checkpoint: `outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt`
> See `docs/post_encoder_fix.md` for migration details.

## 🔴 COMPREHENSIVE NESTED ABLATION (Primary Experiment)

### Conference Mode (501 runs, ~3.5 days on 2x RTX A6000)
```bash
# Step 1: Generate manifest (no GPU)
python scripts/experiments/run_comprehensive_ablation.py \
    --phase manifest --mode conference \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --output-dir outputs/comprehensive_ablation

# Step 2: Validate masks (no GPU)
python scripts/experiments/run_comprehensive_ablation.py \
    --phase validate \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --output-dir outputs/comprehensive_ablation

# Step 3: Train on Bizon with Ray (2x RTX A6000, ~3.5 days)
python scripts/experiments/run_comprehensive_ablation.py \
    --phase train \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/comprehensive_ablation \
    --num-workers 32 --no-save-weights \
    --use-ray --num-gpus 2

# Step 4: Analyze (no GPU)
python scripts/experiments/run_comprehensive_ablation.py \
    --phase analyze \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --output-dir outputs/comprehensive_ablation
```

### Full Mode (2,403 runs, ~16.5 days on 2x RTX A6000)
```bash
python scripts/experiments/run_comprehensive_ablation.py \
    --phase all --mode full \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/comprehensive_ablation_full \
    --num-workers 32 --no-save-weights \
    --use-ray --num-gpus 2
```

### Components Breakdown

| Component | Description | Conference | Full |
|-----------|-------------|-----------|------|
| A | Per-sensor nested modality ablation (6 or 12 sensors) | 342 | 2,124 |
| B | Global baselines (all-features, machine-only, sensor-only) | 42 | 42 |
| C | Sensor-level leave-one-out (all 12 sensors) | 36 | 36 |
| D | Cross-sensor grouped modality ablation | 54 | 54 |
| E | Cross-sensor individual modality ablation | -- | 120 |
| F | Leakage isolation (top 3 sensors) | 27 | 27 |
| **Total** | | **501** | **2,403** |

### Research Questions Answered
- Q1: Which sensor is most informative? (C)
- Q2: Which modality matters most per sensor? (A)
- Q3: Which modality is most important globally? (D, E)
- Q4: How much do machine features contribute? (B)
- Q5: How much accuracy comes from leaked features? (F)
- Q6: Are differences statistically significant? (3 seeds → ANOVA)

---

## 🔴 LEGACY EXPERIMENTS (GPU Required, may be superseded by comprehensive study)

### Phase 0: Architecture Ablations (TBD -- DEFERRED for conference)
```bash
python scripts/experiments/run_architecture_ablations.py \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/jan30/architecture_ablations
```

### Phase 1: Complete v22 Sensor Ablations (8 remaining sensors)
```bash
# Already have: xa_motor, frame_r1, frame_b2, frame_r2
# Missing: frame_l3, frame_l2, spindle1, spindle2, y_bed__1, y_bed__2, y_bed__3, y_bed__4

python scripts/experiments/run_sensor_ablations.py \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/jan30/sensor_ablations \
    --ablation-type leave_one_in
```

### Phase 2: Modality Ablations

**2a. Grouped Modality Ablations (19 runs: 1 baseline + 9 leave-one-out + 9 leave-one-in)**

Groups: accelerometer, gyroscope, magnetometer, environmental, color, rms, motor_electrical, positions, controller

```bash
python scripts/experiments/run_grouped_modality_ablations.py \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/jan30/grouped_modality_ablations
```

**2b. Individual Modality Ablations (41 runs: 1 baseline + 20 leave-one-out + 20 leave-one-in)**

Modalities: Ax, Ay, Az, Gx, Gy, Gz, Mx, My, Mz, Pressure, Temperature, Proximity, ColorR, ColorG, ColorB, ColorA, RMS, motor_electrical, positions, controller

```bash
python scripts/experiments/run_individual_modality_ablations.py \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/jan30/individual_modality_ablations
```

### Phase 3: Baseline Comparisons
```bash
python scripts/experiments/run_baseline_comparisons.py \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/jan30/baseline_comparisons
```

### Phase 4: ANOVA Statistical Validation

**4a. Sensor ANOVA (13 groups x 3 seeds = 39 runs)**
```bash
python scripts/experiments/run_sensor_anova.py \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/anova/sensors \
    --seeds 42,123,456
```

**4b. Grouped Modality ANOVA (10 groups x 3 seeds = 30 runs)**
```bash
python scripts/experiments/run_grouped_modality_anova.py \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/anova/grouped_modalities \
    --seeds 42,123,456
```

**4c. Individual Modality ANOVA (21 groups x 3 seeds = 63 runs) -- Optional**
```bash
python scripts/experiments/run_individual_modality_anova.py \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/anova/individual_modalities \
    --seeds 42,123,456
```

---

## 🟡 NEEDS SCRIPT (Then GPU)

### Phase 5: Top 2-3 Sensor Combination Experiments
- Test if pairs/triples of best sensors outperform singles
- Example: frame_r2 + xa_motor (58 features), frame_r2 + xa_motor + frame_r1 (75 features)
- **Script needed:** `run_sensor_combinations.py`

### Phase 6: Embedding-Level Sensor Fusion / Per-Sensor Normalization
- Architectural changes to address multi-sensor degradation
- Requires modifying encoder architecture
- **Script needed:** TBD

---

## 📊 POST-GPU ANALYSIS (No GPU needed)

### 7. Update Paper with New Sensor Rankings
```bash
# After sensor ablations complete, update Table 3 in paper with v22 results
# Edit: outputs/archive/2024-12_best_runs/sensor_multihead_v3/visualizations/paper/corrected_main_v22.tex
```

### 8. Add ANOVA Results to Paper
```bash
# After ANOVA runs complete, add statistical significance section
# Report F-statistic, p-value, and key post-hoc comparisons
```

### 9. Recompile Paper
```bash
cd /Users/stepheneacuello/Projects/gcode_fingerprinting/outputs/archive/2024-12_best_runs/sensor_multihead_v3/visualizations/paper && \
pdflatex -interaction=nonstopmode corrected_main_v22.tex && \
bibtex corrected_main_v22 && \
pdflatex -interaction=nonstopmode corrected_main_v22.tex && \
pdflatex -interaction=nonstopmode corrected_main_v22.tex && \
open corrected_main_v22.pdf
```

---

## Summary Table

| Phase | Experiment | Runs | Script | Status |
|-------|-----------|------|--------|--------|
| 0 | Architecture Ablations | TBD | `run_architecture_ablations.py` | TBD |
| 1 | Sensor Ablations (remaining) | 8 | `run_sensor_ablations.py` | Pending |
| 2a | Grouped Modality Ablations | 19 | `run_grouped_modality_ablations.py` | Pending |
| 2b | Individual Modality Ablations | 41 | `run_individual_modality_ablations.py` | Pending |
| 3 | Baseline Comparisons | ~5 | `run_baseline_comparisons.py` | Pending |
| 4a | Sensor ANOVA | 39 | `run_sensor_anova.py` | Pending |
| 4b | Grouped Modality ANOVA | 30 | `run_grouped_modality_anova.py` | Pending |
| 4c | Individual Modality ANOVA | 63 | `run_individual_modality_anova.py` | Optional |
| 5 | Top Sensor Combinations | TBD | -- | Needs Script |
| 6 | Embedding-Level Fusion | TBD | -- | Needs Design |
| 7 | Update Paper (Sensors) | -- | -- | Post-GPU |
| 8 | Update Paper (ANOVA) | -- | -- | Post-GPU |
| 9 | Recompile Paper | -- | -- | Post-GPU |

**Total GPU runs: ~200+** (excluding TBD items)
