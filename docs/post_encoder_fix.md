# Post-Encoder Fix: Change Log and Next Steps

**Date:** January 31, 2025

---

## What Changed

### 1. Data Leakage Fix (Jan 23)

**Problem:** The StandardScaler was fit on the entire dataset (train+val+test) before splitting, leaking test statistics into training.

**Fix:** Scaler now fits on training data only, then transforms val/test. Corrected splits saved to `outputs/jan23_followup/no_leakage/`.

**Impact:** Multi-sensor token accuracy 80.7% → 84.3%, sequence accuracy 35.4% → 50.6%.

### 2. MM_DTAE_LSTM Encoder (Jan 23–30)

Built a new encoder architecture (`MM_DTAE_LSTM`) with per-modality encoding:
- 7 modality groups encoded independently (accelerometer, gyroscope, magnetometer, environmental, color, RMS, machine)
- Cross-modal fusion with learned gates
- Denoising Transformer autoencoder → bidirectional LSTM
- Per-modality embeddings preserved before fusion

Trained on corrected no-leakage data. Achieves 90.2% on 9-class operation classification (100% on normal classes, 0% on 2/3 damage classes due to zero-sum CE loss).

### 3. Hybrid Pipeline → 100% (Jan 30–31)

Froze the MM_DTAE_LSTM encoder and added post-hoc LogReg classifiers on per-modality embeddings:
- Damage router on machine modality (4-class: normal/c6/c7/c8)
- Gyroscope specialist for face damage, magnetometer specialist for pocket damage
- Mean pooling over time (key improvement)
- Result: **100.00% test accuracy (615/615)**, robust across thresholds 0.30–0.70

**Reproduction:** `PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/evaluation/run_encoder_pipeline.py`

---

## Correct Encoder Going Forward

**Use this checkpoint for all downstream tasks:**
```
outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt
```

**Architecture:** `MM_DTAE_LSTM` (from `src/miracle/model/model.py`)
- Input: list of per-modality tensors `List[[B, T, Cm]]`
- Output memory: `[B, T, 256]`
- Requires `lengths` tensor
- Config stored in checkpoint under `'config'` key

**Self-contained pipeline directory:**
```
outputs/jan30/encoder_pipeline/
├── data/                          # No-leakage splits
├── encoder_checkpoint/best_model.pt
├── router.joblib, specialist_c7.joblib, specialist_c8.joblib
├── results.json
└── (plots and logs)
```

---

## What's Obsolete

| Item | Why |
|------|-----|
| `outputs/jan26/ensemble/seed_42/best_model.pt` | Old `EnhancedEncoder` architecture, trained before MM_DTAE_LSTM |
| `EnhancedEncoder` class references | Replaced by `MM_DTAE_LSTM` |
| Any encoder path pointing to jan26 | Pre-dates the corrected encoder |

---

## Scripts Needing Migration

These scripts still reference `EnhancedEncoder` and the jan26 checkpoint:

| Script | Change Required |
|--------|----------------|
| `scripts/training/ray_tune_final.py` | Replace EnhancedEncoder → MM_DTAE_LSTM, add modality splitting, sensor_dim 128→256 |
| `scripts/experiments/run_comprehensive_ablation.py` | Same as above |
| `EXPERIMENT_TODO.md` | Update all encoder paths |

### Migration checklist per script:
1. Import `MM_DTAE_LSTM` and `ModelConfig` instead of `EnhancedEncoder`
2. Add `build_modality_indices()` and `split_modalities()` functions
3. Load encoder from jan30 checkpoint using saved `ModelConfig`
4. Split flat `[B, T, 296]` sensor tensor into per-modality list before encoder forward
5. Pass `lengths` to encoder (required by MM_DTAE_LSTM)
6. Update decoder `sensor_dim` from 128 to 256
7. Sanity check: verify loss decreases over a few epochs before running full sweep
