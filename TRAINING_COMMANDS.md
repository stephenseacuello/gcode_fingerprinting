# G-Code Fingerprinting Training Commands

## Current State
- **Best Model**: 81.76% val / 81.39% test token accuracy
- **Model Path**: `outputs/breakthrough_full_vocab/best_model.pt`
- **Vocabulary**: `data/vocabulary_4digit_full.json` (2498 tokens, 0% UNK)
- **Data Splits**: `outputs/stratified_splits_full_vocab`

---

## 1. Local Training (Mac M1/M2)

### Single Training Run (Full 100 epochs, ~6 hours)
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --split-dir outputs/stratified_splits_full_vocab \
    --encoder-path outputs/mm_dtae_lstm_v2/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/training_v2 \
    --d-model 256 --n-layers 5 --n-heads 8 \
    --dropout 0.35 --embed-dropout 0.15 \
    --batch-size 24 --max-epochs 100 --patience 30 \
    --learning-rate 1.5e-4 --weight-decay 0.05 \
    --lr-scheduler cosine --warmup-epochs 10 \
    --use-type-constraint --use-sensor-prior --sensor-prior-weight 0.5 \
    --use-position-weights --position-weight-scale 3.0 \
    --use-error-sampling --error-boost 2.0 \
    --label-smoothing 0.1 --grad-clip 1.0 \
    --use-wandb --wandb-project gcode-training --run-name training-v2
```

---

## 2. Enhancement Pipeline (Post-Training)

### Phase 1: SCST Fine-Tuning (~2 hours)
Directly optimize sequence-level accuracy using REINFORCE with self-critical baseline.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/train_scst.py \
    --checkpoint outputs/breakthrough_full_vocab/best_model.pt \
    --split-dir outputs/stratified_splits_full_vocab \
    --vocab-path data/vocabulary_4digit_full.json \
    --encoder-path outputs/mm_dtae_lstm_v2/best_model.pt \
    --output-dir outputs/scst_finetuned \
    --scst-epochs 20 \
    --learning-rate 5e-6 \
    --sample-temperature 0.7 \
    --reward-scale 1.0 \
    --batch-size 16 \
    --grad-clip 0.5 \
    --mixed-training \
    --ce-weight 0.3 \
    --scst-weight 0.7
```

### Phase 2: Train Ensemble (3 seeds, ~18 hours total)
Train models with different seeds for ensemble diversity.

```bash
# Seed 123
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --split-dir outputs/stratified_splits_full_vocab \
    --encoder-path outputs/mm_dtae_lstm_v2/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/ensemble/model_seed_123 \
    --seed 123 \
    --d-model 256 --n-layers 5 --n-heads 8 \
    --dropout 0.35 --embed-dropout 0.15 \
    --batch-size 24 --max-epochs 100 --patience 30 \
    --learning-rate 1.5e-4 --weight-decay 0.05 \
    --lr-scheduler cosine --warmup-epochs 10 \
    --use-type-constraint --use-sensor-prior --sensor-prior-weight 0.5 \
    --use-position-weights --position-weight-scale 3.0 \
    --use-error-sampling --error-boost 2.0 \
    --label-smoothing 0.1 --grad-clip 1.0 \
    --use-wandb --wandb-project gcode-ensemble --run-name seed-123

# Seed 456
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --split-dir outputs/stratified_splits_full_vocab \
    --encoder-path outputs/mm_dtae_lstm_v2/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/ensemble/model_seed_456 \
    --seed 456 \
    --d-model 256 --n-layers 5 --n-heads 8 \
    --dropout 0.35 --embed-dropout 0.15 \
    --batch-size 24 --max-epochs 100 --patience 30 \
    --learning-rate 1.5e-4 --weight-decay 0.05 \
    --lr-scheduler cosine --warmup-epochs 10 \
    --use-type-constraint --use-sensor-prior --sensor-prior-weight 0.5 \
    --use-position-weights --position-weight-scale 3.0 \
    --use-error-sampling --error-boost 2.0 \
    --label-smoothing 0.1 --grad-clip 1.0 \
    --use-wandb --wandb-project gcode-ensemble --run-name seed-456
```

### Phase 3: Evaluate Ensemble + TTA
```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_ensemble.py \
    --checkpoints outputs/breakthrough_full_vocab/best_model.pt \
                  outputs/ensemble/model_seed_123/best_model.pt \
                  outputs/ensemble/model_seed_456/best_model.pt \
    --data-dir outputs/stratified_splits_full_vocab \
    --vocab-path data/vocabulary_4digit_full.json \
    --encoder-path outputs/mm_dtae_lstm_v2/best_model.pt \
    --output reports/ensemble_evaluation.json \
    --use-tta --tta-augmentations 5
```

### Phase 4: Advanced Decoding Evaluation
```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_advanced_decoding.py \
    --model-dir outputs/breakthrough_full_vocab \
    --encoder-path outputs/mm_dtae_lstm_v2/best_model.pt \
    --split-dir outputs/stratified_splits_full_vocab \
    --vocab-path data/vocabulary_4digit_full.json \
    --samples 500
```

---

## 3. Lambda GPU Cluster Training

### Setup (One-time)
```bash
# Clone repo and install dependencies
git clone <repo_url> gcode_fingerprinting
cd gcode_fingerprinting
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio
pip install wandb numpy tqdm matplotlib seaborn pandas scikit-learn

# Login to wandb
wandb login
```

### Option A: Focused Sweep (Recommended - ~1 day)
```bash
# Create the sweep
wandb sweep configs/sweep_lambda_focused.yaml

# Launch 8 agents (one per GPU)
# Replace SWEEP_ID with the ID from previous command
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i wandb agent YOUR_ENTITY/gcode-lambda-focused/SWEEP_ID --count 25 &
done
```

### Option B: Comprehensive Sweep (~3 days)
```bash
wandb sweep configs/sweep_lambda_comprehensive.yaml

for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i wandb agent YOUR_ENTITY/gcode-lambda-sweep/SWEEP_ID --count 50 &
done
```

### Single High-Performance Run on Lambda
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_sensor_multihead.py \
    --split-dir outputs/stratified_splits_full_vocab \
    --encoder-path outputs/mm_dtae_lstm_v2/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/lambda_training \
    --d-model 384 --n-layers 6 --n-heads 16 \
    --dropout 0.35 --embed-dropout 0.15 \
    --batch-size 64 --max-epochs 150 --patience 40 \
    --learning-rate 1.5e-4 --weight-decay 0.05 \
    --lr-scheduler cosine --warmup-epochs 15 \
    --use-type-constraint --use-sensor-prior --sensor-prior-weight 0.5 \
    --use-position-weights --position-weight-scale 3.0 \
    --use-error-sampling --error-boost 2.0 \
    --label-smoothing 0.1 --grad-clip 1.0 \
    --curriculum --curriculum-phases 3 --curriculum-epochs-per-phase 40 \
    --use-wandb --wandb-project gcode-lambda --run-name lambda-v1
```

---

## 4. Full Enhancement Pipeline Script

Save and run this for the complete pipeline:

```bash
#!/bin/bash
# scripts/run_full_pipeline.sh

set -e

BASE_MODEL="outputs/breakthrough_full_vocab/best_model.pt"
ENCODER="outputs/mm_dtae_lstm_v2/best_model.pt"
DATA_DIR="outputs/stratified_splits_full_vocab"
VOCAB="data/vocabulary_4digit_full.json"

echo "=============================================="
echo "G-CODE FINGERPRINTING ENHANCEMENT PIPELINE"
echo "=============================================="

# Phase 1: SCST Fine-tuning
echo "Phase 1: SCST Fine-tuning (20 epochs)..."
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/train_scst.py \
    --checkpoint "$BASE_MODEL" \
    --split-dir "$DATA_DIR" \
    --vocab-path "$VOCAB" \
    --encoder-path "$ENCODER" \
    --output-dir outputs/enhancement_pipeline/scst \
    --scst-epochs 20 \
    --learning-rate 5e-6 \
    --batch-size 16

# Phase 2: Ensemble Training
echo "Phase 2: Training ensemble seeds..."
for seed in 123 456; do
    echo "  Training seed $seed..."
    PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
        --split-dir "$DATA_DIR" \
        --encoder-path "$ENCODER" \
        --vocab-path "$VOCAB" \
        --output-dir outputs/enhancement_pipeline/ensemble/model_seed_$seed \
        --seed $seed \
        --d-model 256 --n-layers 5 --n-heads 8 \
        --dropout 0.35 --embed-dropout 0.15 \
        --batch-size 24 --max-epochs 100 --patience 30 \
        --learning-rate 1.5e-4 --weight-decay 0.05 \
        --lr-scheduler cosine --warmup-epochs 10 \
        --use-type-constraint --use-sensor-prior --sensor-prior-weight 0.5 \
        --use-position-weights --position-weight-scale 3.0 \
        --use-error-sampling --error-boost 2.0 \
        --label-smoothing 0.1 --grad-clip 1.0
done

# Phase 3: Ensemble Evaluation
echo "Phase 3: Evaluating ensemble..."
PYTHONPATH=src .venv/bin/python scripts/evaluate_ensemble.py \
    --checkpoints "$BASE_MODEL" \
                  outputs/enhancement_pipeline/ensemble/model_seed_123/best_model.pt \
                  outputs/enhancement_pipeline/ensemble/model_seed_456/best_model.pt \
    --data-dir "$DATA_DIR" \
    --vocab-path "$VOCAB" \
    --encoder-path "$ENCODER" \
    --output reports/pipeline_ensemble.json \
    --use-tta --tta-augmentations 5

# Phase 4: Advanced Decoding
echo "Phase 4: Evaluating decoding strategies..."
PYTHONPATH=src .venv/bin/python scripts/evaluate_advanced_decoding.py \
    --model-dir outputs/breakthrough_full_vocab \
    --encoder-path "$ENCODER" \
    --split-dir "$DATA_DIR" \
    --vocab-path "$VOCAB" \
    --samples 500

echo "=============================================="
echo "PIPELINE COMPLETE!"
echo "=============================================="
echo "Results saved to: reports/"
```

---

## 5. ENHANCED Training Commands (All Improvements Enabled)

These commands incorporate ALL the additional improvement ideas discovered in the codebase.

### Quick Wins Training (Local Mac)
Enable the highest-impact improvements for fastest gains:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --split-dir outputs/stratified_splits_full_vocab \
    --encoder-path outputs/mm_dtae_lstm_v2/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/enhanced_training \
    --seed 42 \
    # Architecture (proven best)
    --d-model 256 --n-layers 5 --n-heads 8 \
    --dropout 0.35 --embed-dropout 0.15 \
    # Encoder enhancements (NEW +5-7%)
    --use-enhanced-encoder \
    --use-multihead-pooling \
    --use-auxiliary-heads --auxiliary-loss-weight 0.3 \
    # Training basics
    --batch-size 24 --max-epochs 100 --patience 30 \
    # Optimizer
    --learning-rate 1.5e-4 --weight-decay 0.05 \
    # LR Schedule (NEW: cosine warmup)
    --lr-scheduler cosine --warmup-epochs 15 --warmup-type cosine \
    # Proven beneficial
    --use-type-constraint --use-sensor-prior --sensor-prior-weight 0.5 \
    --use-position-weights --position-weight-scale 3.0 \
    --use-error-sampling --error-boost 2.0 \
    # Adaptive weighting + EOS calibration (NEW +2-3%)
    --adaptive-weighting \
    --use-eos-calibration --eos-calibration-weight 0.2 \
    # Loss tuning
    --label-smoothing 0.1 --grad-clip 1.0 \
    --digit-focal-gamma 7.0 \
    # Progressive augmentation (NEW +1-2%)
    --augment --progressive-augmentation \
    --augment-start 0.1 --augment-end 0.5 \
    # Regularization
    --drop-path-rate 0.1 \
    # Logging
    --use-wandb --wandb-project gcode-enhanced --run-name enhanced-quickwins
```

### Full Enhancement Training (Lambda GPU)
Maximum improvements for highest accuracy:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_sensor_multihead.py \
    --split-dir outputs/stratified_splits_full_vocab \
    --encoder-path outputs/mm_dtae_lstm_v2/best_model.pt \
    --vocab-path data/vocabulary_4digit_full.json \
    --output-dir outputs/lambda_enhanced \
    --seed 42 \
    # === ARCHITECTURE (larger for GPU) ===
    --d-model 384 --n-layers 6 --n-heads 16 \
    --dropout 0.35 --embed-dropout 0.15 \
    # === ENCODER ENHANCEMENTS ===
    --use-enhanced-encoder \
    --use-multihead-pooling --pooling-n-heads 4 --pooling-n-queries 8 \
    --use-auxiliary-heads --auxiliary-loss-weight 0.3 \
    # === TRAINING SCHEDULE ===
    --batch-size 64 --max-epochs 150 --patience 40 \
    --accumulation-steps 2 \
    # === OPTIMIZER ===
    --optimizer adamw \
    --learning-rate 1.5e-4 --weight-decay 0.05 \
    # === LR SCHEDULER (with warm restarts) ===
    --lr-scheduler cosine_restarts --restart-period 30 \
    --warmup-epochs 15 --warmup-type cosine \
    --min-lr 1e-6 \
    # === PROVEN BENEFICIAL ===
    --use-type-constraint \
    --use-sensor-prior --sensor-prior-weight 0.5 \
    --use-position-weights --position-weight-scale 3.0 \
    --use-error-sampling --error-boost 2.0 \
    # === ADVANCED LOSS IMPROVEMENTS ===
    --adaptive-weighting \
    --use-eos-calibration --eos-calibration-weight 0.2 \
    --use-dual-head-value --dual-head-weight 0.3 \
    --use-consistency-loss --consistency-weight 0.1 \
    --label-smoothing 0.1 \
    --digit-focal-gamma 7.0 \
    # === DATA AUGMENTATION ===
    --augment --progressive-augmentation \
    --augment-start 0.1 --augment-end 0.5 \
    --use-mixup --mixup-prob 0.4 --mixup-alpha 0.15 \
    # === REGULARIZATION ===
    --drop-path-rate 0.1 \
    --grad-clip 1.0 \
    # === CURRICULUM LEARNING ===
    --curriculum --curriculum-phases 3 --curriculum-epochs-per-phase 40 \
    --scheduled-sampling --teacher-forcing-start 1.0 --teacher-forcing-end 0.3 \
    # === LOGGING ===
    --use-wandb --wandb-project gcode-lambda-enhanced --run-name full-enhanced
```

### Expected Improvement Breakdown

| Improvement | Flag | Expected Gain |
|-------------|------|---------------|
| Enhanced encoder | `--use-enhanced-encoder` | +3-5% |
| Multihead pooling | `--use-multihead-pooling` | +1-2% |
| Auxiliary heads | `--use-auxiliary-heads` | +2-4% |
| Cosine warmup | `--warmup-type cosine` | +1% |
| LR restarts | `--lr-scheduler cosine_restarts` | +1-2% |
| EOS calibration | `--use-eos-calibration` | +1-2% |
| Dual-head value | `--use-dual-head-value` | +2-3% |
| Progressive aug | `--progressive-augmentation` | +1-2% |
| Mixup | `--use-mixup` | +1-2% |
| Drop path | `--drop-path-rate 0.1` | +1-2% |
| **Combined Total** | All above | **+15-25%** |

---

## 6. Expected Results Summary

| Configuration | Expected Token Accuracy |
|---------------|-------------------------|
| Baseline (current) | 81.76% |
| + SCST fine-tuning | 84-87% |
| + 3-model ensemble | 86-89% |
| + TTA (5 augmentations) | 87-91% |
| + Grammar-constrained decoding | 89-94% |
| **Lambda sweep (larger models)** | **90-95%+** |

---

## Key Configuration Files

| File | Purpose |
|------|---------|
| `configs/sweep_lambda_comprehensive.yaml` | Full hyperparameter sweep (~400 runs) |
| `configs/sweep_lambda_focused.yaml` | Focused sweep on key params (~200 runs) |
| `scripts/train_sensor_multihead.py` | Main training script |
| `scripts/train_scst.py` | SCST fine-tuning script |
| `scripts/evaluate_ensemble.py` | Ensemble + TTA evaluation |
| `scripts/evaluate_advanced_decoding.py` | Grammar/MBR decoding evaluation |

---

## Troubleshooting

### Scheduler Error (T_max=0)
If training with 1 epoch for testing, use at least 2:
```bash
--max-epochs 2
```

### State Dict Mismatch
Always use `strict=False` when loading checkpoints:
```python
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

### Memory Issues on Mac
Reduce batch size:
```bash
--batch-size 16
```
