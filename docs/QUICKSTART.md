# Quick Start Guide

**Project**: G-code Fingerprinting with Machine Learning
**Version**: 2.0.0 (v16 Final)
**Last Updated**: 2025-12-14

---

## Quick Evaluation (No Training Required)

The repository includes pretrained models and test data. You can run evaluation immediately:

```bash
# Clone and setup
git clone https://github.com/stephenseacuello/gcode_fingerprinting.git
cd gcode_fingerprinting
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run evaluation on test set
PYTHONPATH=src .venv/bin/python scripts/evaluate_checkpoint.py \
    --encoder-path outputs/mm_dtae_lstm_v2/best_model.pt \
    --decoder-path outputs/sensor_multihead_v3/best_model.pt \
    --split-dir outputs/stratified_splits_v2 \
    --vocab-path data/vocabulary_4digit_hybrid.json
```

**Expected Results:**
- Operation Classification: **100%**
- Token Accuracy: **90.23%**
- Exact Sequence Match: **60.6%**

---

## Full Training Pipeline (Requires Training Data)

> **Note**: Training requires the full dataset (~117MB training split) which is not included in the repository due to GitHub size limits. Contact the author for access, or use your own sensor data.

### 1. Create Stratified Splits

```bash
PYTHONPATH=src .venv/bin/python scripts/create_multilabel_stratified_splits.py \
    --input-dir outputs/processed_v2 \
    --output-dir outputs/stratified_splits_v2
```

### 2. Train Encoder (MM-DTAE-LSTM)

```bash
PYTHONPATH=src .venv/bin/python scripts/train_mm_dtae_lstm.py \
    --data-dir outputs/stratified_splits_v2 \
    --output-dir outputs/mm_dtae_lstm_v2 \
    --epochs 100 \
    --patience 15
```

### 3. Train Decoder (SensorMultiHeadDecoder)

```bash
PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --split-dir outputs/stratified_splits_v2 \
    --vocab-path data/vocabulary_4digit_hybrid.json \
    --encoder-path outputs/mm_dtae_lstm_v2/best_model.pt \
    --output-dir outputs/sensor_multihead_v3 \
    --max-epochs 150 \
    --patience 30
```

---

## Architecture Overview

### Two-Stage Pipeline

```
Stage 1: MM-DTAE-LSTM Encoder (FROZEN after training)
┌─────────────────────────────────────────────────────────┐
│  Sensor Data [B, T, 155] → BiLSTM → Latent [B, T, 128] │
│                    ↓                                    │
│         Operation Classification → 100% accuracy        │
└─────────────────────────────────────────────────────────┘

Stage 2: SensorMultiHeadDecoder (TRAINABLE)
┌─────────────────────────────────────────────────────────┐
│  Latent + Tokens → Transformer Decoder (4 layers)       │
│                    ↓                                    │
│  Multi-Head Output:                                     │
│    ├─ Type Head [4 classes]                            │
│    ├─ Command Head [6 classes]                         │
│    ├─ Param Type Head [10 classes]                     │
│    └─ Digit Heads [4 positions × 10 digits]            │
│                    ↓                                    │
│         Token Accuracy → 90.23%                         │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Frozen Encoder**: Pre-trained encoder achieves 100% operation classification
2. **4-Digit Hybrid Vocabulary**: 668 tokens with precise numeric encoding
3. **Multi-Head Decomposition**: Separate heads for type, command, param, digits
4. **Focal Loss**: γ=3.0 handles class imbalance
5. **Curriculum Learning**: 3-phase training (structure → digits → full)

---

## Model Files

| File | Size | Description |
|------|------|-------------|
| `outputs/mm_dtae_lstm_v2/best_model.pt` | 45 MB | Encoder checkpoint |
| `outputs/sensor_multihead_v3/best_model.pt` | 32 MB | Decoder checkpoint |
| `data/vocabulary_4digit_hybrid.json` | 15 KB | 668-token vocabulary |

---

## Run Inference

```python
import torch
import json
from pathlib import Path

# Load models
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load decoder
checkpoint = torch.load('outputs/sensor_multihead_v3/best_model.pt',
                        map_location=device, weights_only=False)
decoder = SensorMultiHeadDecoder(
    vocab_size=668, d_model=192, n_heads=8, n_layers=4,
    sensor_dim=128, n_operations=9, n_types=4,
    n_commands=6, n_param_types=10
).to(device)
decoder.load_state_dict(checkpoint['model_state_dict'])
decoder.eval()

# Load vocabulary
with open('data/vocabulary_4digit_hybrid.json') as f:
    vocab = json.load(f)
id_to_token = {v: k for k, v in vocab['vocab'].items()}

print(f"Model loaded: {sum(p.numel() for p in decoder.parameters()):,} parameters")
```

---

## Notebooks

| Notebook | Description | Runnable? |
|----------|-------------|-----------|
| [01_getting_started](../notebooks/01_getting_started.ipynb) | Project overview | Yes |
| [04_inference_prediction](../notebooks/04_inference_prediction.ipynb) | Run inference | Yes |
| [08_model_evaluation](../notebooks/08_model_evaluation.ipynb) | Evaluate model | Yes |
| [03_training_models](../notebooks/03_training_models.ipynb) | Training guide | Needs training data |

---

## Common Errors

### ModuleNotFoundError: No module named 'miracle'

```bash
# Solution: Add PYTHONPATH
PYTHONPATH=src .venv/bin/python scripts/...
```

### MPS backend issues (Apple Silicon)

```bash
# Solution: Enable fallback
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/...
```

### Missing training data

The training split (`train_sequences.npz`, 117MB) is not included in the repository. You can:
1. Run evaluation with included test/val splits
2. Contact author for full training data
3. Use your own sensor data with the preprocessing pipeline

---

## Next Steps

- **Evaluation**: See [TRAINING_RESULTS_SUMMARY.md](TRAINING_RESULTS_SUMMARY.md)
- **Architecture Details**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Full Training Guide**: See [TRAINING.md](TRAINING.md)
- **Paper**: See `outputs/sensor_multihead_v3/visualizations/paper/corrected_main_v16.pdf`

---

**Questions?** Open an issue at https://github.com/stephenseacuello/gcode_fingerprinting/issues
