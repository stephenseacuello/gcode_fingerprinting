# Experiment TODO - G-code Fingerprinting

> **Encoder:** All experiments use the frozen MM_DTAE_LSTM encoder from the verified 100% hybrid pipeline.
> Checkpoint: `outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt`
> See `docs/encoder_architecture_decision.md` and `docs/post_encoder_fix.md` for details.
>
> **All outputs go to `outputs/jan30/`.**
> **Run GPU phases on Bizon (2x RTX A6000) inside `tmux`.**

---

## Phase 1: Ray Hyperparameter Sweep (find best decoder config)

**Purpose:** Search decoder architecture/optimizer/scheduler space to find the best config before running ablations.

```bash
python scripts/training/ray_tune_final.py \
    --config configs/lambda_sweeps/final_comprehensive_sweep.yaml \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/jan30/ray_sweep \
    --num-samples 300 --max-concurrent 2
```

**After sweep completes:** Update `configs/decoder_config.json` with the best hyperparameters found.

---

## Phase 2: Comprehensive Nested Ablation — Conference Mode (501 runs)

**Purpose:** Primary conference experiment. Answers all 6 research questions with statistical significance.

### Research Questions
- Q1: Which sensor is most informative? (Component C)
- Q2: Which modality matters most per sensor? (Component A)
- Q3: Which modality is most important globally? (Components D, E)
- Q4: How much do machine features contribute? (Component B)
- Q5: How much accuracy comes from leaked features? (Component F)
- Q6: Are differences statistically significant? (3 seeds per config)

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

### Commands

```bash
# Step 1: Generate manifest (no GPU)
python scripts/experiments/run_comprehensive_ablation.py \
    --phase manifest --mode conference \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --output-dir outputs/jan30/comprehensive_ablation

# Step 2: Validate masks (no GPU)
python scripts/experiments/run_comprehensive_ablation.py \
    --phase validate \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --output-dir outputs/jan30/comprehensive_ablation

# Step 3: Train on Bizon with Ray (2x RTX A6000, ~3.5 days)
python scripts/experiments/run_comprehensive_ablation.py \
    --phase train \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/jan30/comprehensive_ablation \
    --num-workers 32 --no-save-weights \
    --use-ray --num-gpus 2

# Step 4: Analyze (no GPU)
python scripts/experiments/run_comprehensive_ablation.py \
    --phase analyze \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --output-dir outputs/jan30/comprehensive_ablation
```

---

## Phase 2.5 (Optional): Comprehensive Nested Ablation — Full Mode (2,403 runs)

**Purpose:** Complete version with all 12 sensors and individual cross-sensor modality ablation (Component E). Run this if conference mode results warrant deeper investigation or for journal submission.

```bash
# Step 1: Generate manifest (no GPU)
python scripts/experiments/run_comprehensive_ablation.py \
    --phase manifest --mode full \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --output-dir outputs/jan30/comprehensive_ablation_full

# Step 2: Validate masks (no GPU)
python scripts/experiments/run_comprehensive_ablation.py \
    --phase validate \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --output-dir outputs/jan30/comprehensive_ablation_full

# Step 3: Train on Bizon with Ray (2x RTX A6000, ~16.5 days)
python scripts/experiments/run_comprehensive_ablation.py \
    --phase train \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/jan30/comprehensive_ablation_full \
    --num-workers 32 --no-save-weights \
    --use-ray --num-gpus 2

# Step 4: Analyze (no GPU)
python scripts/experiments/run_comprehensive_ablation.py \
    --phase analyze \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --output-dir outputs/jan30/comprehensive_ablation_full
```

---

## Phase 3: ANOVA Statistical Validation

**Purpose:** Multi-seed runs for statistical significance testing (F-statistic, p-values, post-hoc comparisons).

```bash
# 3a. Sensor ANOVA (13 groups x 3 seeds = 39 runs)
python scripts/experiments/run_sensor_anova.py \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/jan30/anova/sensors \
    --seeds 42,123,456

# 3b. Grouped Modality ANOVA (10 groups x 3 seeds = 30 runs)
python scripts/experiments/run_grouped_modality_anova.py \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/jan30/anova/grouped_modalities \
    --seeds 42,123,456

# 3c. Individual Modality ANOVA (21 groups x 3 seeds = 63 runs) -- Optional
python scripts/experiments/run_individual_modality_anova.py \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/jan30/anova/individual_modalities \
    --seeds 42,123,456
```

---

## Phase 4: Legacy Standalone Experiments (if needed)

> These are largely superseded by the comprehensive ablation (Phase 2), which covers sensor leave-one-out, grouped/individual modality ablations, baselines, and leakage isolation in one unified framework. Run only if specific gaps need filling.

```bash
# 4a. Sensor leave-one-in ablations (8 remaining sensors)
python scripts/experiments/run_sensor_ablations.py \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/jan30/sensor_ablations \
    --ablation-type leave_one_in

# 4b. Grouped modality ablations (19 runs)
python scripts/experiments/run_grouped_modality_ablations.py \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/jan30/grouped_modality_ablations

# 4c. Individual modality ablations (41 runs)
python scripts/experiments/run_individual_modality_ablations.py \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/jan30/individual_modality_ablations

# 4d. Baseline comparisons
python scripts/experiments/run_baseline_comparisons.py \
    --config configs/decoder_config.json \
    --data-dir outputs/jan30/encoder_pipeline/data \
    --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/jan30/baseline_comparisons
```

---

## Phase 5: Future Work (Needs Scripts)

- **Top 2-3 sensor combinations** — test if pairs/triples of best sensors outperform singles. Needs `run_sensor_combinations.py`.
- **Embedding-level sensor fusion / per-sensor normalization** — architectural changes to address multi-sensor degradation. Needs design + script.

---

## Phase 6: Paper Updates (no GPU, after experiments complete)

```bash
# 6a. Update paper with new sensor rankings (after Phase 2 results)
# Edit: outputs/archive/2024-12_best_runs/sensor_multihead_v3/visualizations/paper/corrected_main_v22.tex

# 6b. Add ANOVA results to paper (after Phase 3 results)
# Report F-statistic, p-value, and key post-hoc comparisons

# 6c. Recompile paper
cd outputs/archive/2024-12_best_runs/sensor_multihead_v3/visualizations/paper && \
pdflatex -interaction=nonstopmode corrected_main_v22.tex && \
bibtex corrected_main_v22 && \
pdflatex -interaction=nonstopmode corrected_main_v22.tex && \
pdflatex -interaction=nonstopmode corrected_main_v22.tex && \
open corrected_main_v22.pdf
```

---

## Summary Table

| Phase | Experiment | Runs | Status |
|-------|-----------|------|--------|
| 1 | Ray Hyperparameter Sweep | 300 | Pending |
| 2 | Comprehensive Ablation (conference) | 501 | Pending |
| 2.5 | Comprehensive Ablation (full) | 2,403 | Optional |
| 3a | Sensor ANOVA | 39 | Pending |
| 3b | Grouped Modality ANOVA | 30 | Pending |
| 3c | Individual Modality ANOVA | 63 | Optional |
| 4 | Legacy standalone experiments | ~70 | If needed |
| 5 | Sensor combinations / fusion | TBD | Needs scripts |
| 6 | Paper updates | -- | Post-GPU |

**Execution order:** 1 → update decoder config → 2 → 3 → 6. Run 2.5 if conference results warrant it.
