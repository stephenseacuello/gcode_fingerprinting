# G-code Fingerprinting with Machine Learning

**Two-stage deep learning system for predicting G-code commands from 3D printer sensor data using a frozen encoder and hierarchical multi-head decoder.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project implements a **two-stage architecture** for G-code prediction from multi-modal sensor data:

1. **Stage 1: MM-DTAE-LSTM Encoder** (frozen) - Encodes sensor sequences with 100% operation classification
2. **Stage 2: SensorMultiHeadDecoder** (trainable) - Generates G-code tokens with hierarchical multi-head prediction

### Performance (v16 Final Model)

| Metric | Accuracy |
|--------|----------|
| **Operation Classification** | **100.0%** |
| **Token Accuracy** | **90.23%** |
| Type Classification | 99.8% |
| Command Prediction | 99.9% |
| Parameter Type | 96.2% |

**600x improvement** over random baseline (0.15%)

### Key Features

- **Two-Stage Architecture**: Frozen encoder + trainable decoder for optimal performance
- **4-Digit Hybrid Tokenization**: 668-token vocabulary with precise numeric encoding
- **Multi-Head Decoder**: Separate heads for type, command, param type, and digit values
- **Focal Loss**: Handles class imbalance with γ=3.0
- **Comprehensive Ablation Studies**: Sensor modality importance analysis
- **Production-Ready**: FastAPI server, ONNX export, <10ms inference

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/gcode_fingerprinting.git
cd gcode_fingerprinting

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# 1. Create stratified splits
PYTHONPATH=src .venv/bin/python scripts/create_multilabel_stratified_splits.py \
    --input-dir outputs/processed_v2 \
    --output-dir outputs/stratified_splits_v2

# 2. Train encoder (MM-DTAE-LSTM)
PYTHONPATH=src .venv/bin/python scripts/train_mm_dtae_lstm.py \
    --data-dir outputs/stratified_splits_v2 \
    --output-dir outputs/mm_dtae_lstm_v2 \
    --epochs 50

# 3. Train decoder (with frozen encoder)
PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --data-dir outputs/stratified_splits_v2 \
    --encoder-path outputs/mm_dtae_lstm_v2/best_model.pt \
    --output-dir outputs/sensor_multihead_v3 \
    --epochs 50
```

### Inference

```python
import torch
import numpy as np
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.model.mm_dtae_lstm import MMDTAELSTM

# Load models
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

encoder = MMDTAELSTM(continuous_dim=155, categorical_dims=[3,3,3,3], hidden_dim=128, n_operations=9)
encoder.load_state_dict(torch.load('outputs/mm_dtae_lstm_v2/best_model.pt', weights_only=False)['model_state_dict'])
encoder.eval().to(device)

decoder = SensorMultiHeadDecoder(sensor_dim=128, d_model=192, n_heads=8, n_layers=4,
                                  n_operations=9, n_types=4, n_commands=6, n_param_types=10)
decoder.load_state_dict(torch.load('outputs/sensor_multihead_v3/best_model.pt', weights_only=False)['model_state_dict'])
decoder.eval().to(device)

# Run inference
data = np.load('outputs/stratified_splits_v2/test_sequences.npz', allow_pickle=True)
continuous = torch.tensor(data['continuous'][0:1], dtype=torch.float32).to(device)
categorical = torch.tensor(data['categorical'][0:1], dtype=torch.long).to(device)

with torch.no_grad():
    memory, op_logits = encoder(continuous, categorical)
    operation = op_logits.argmax(dim=-1)
    # Decode tokens (see notebooks/04_inference_prediction.ipynb for full example)
```

---

## Architecture

### Two-Stage Pipeline

```
Stage 1: MM-DTAE-LSTM Encoder (FROZEN)
┌─────────────────────────────────────────────────────────┐
│  Continuous [B, 64, 155] ──→ Linear ──→ LayerNorm      │
│  Categorical [B, 64, 4]  ──→ Embed ──→ Linear          │
│                              ↓                          │
│                         Fusion (add)                    │
│                              ↓                          │
│                    BiLSTM (2 layers)                    │
│                              ↓                          │
│              Memory [B, 64, 128] + Operation [B, 9]     │
└─────────────────────────────────────────────────────────┘
                    Operation: 100% accuracy

Stage 2: SensorMultiHeadDecoder (TRAINABLE)
┌─────────────────────────────────────────────────────────┐
│  Memory [B, 64, 128] + Tokens [B, 7]                   │
│                              ↓                          │
│         Token Embedding (668) + Positional Encoding     │
│                              ↓                          │
│         Transformer Decoder (4 layers, 8 heads)         │
│                              ↓                          │
│  ┌──────────┬──────────┬────────────┬────────────────┐ │
│  │ Type [4] │ Cmd [6]  │ PType [10] │ Digits [4×10]  │ │
│  └──────────┴──────────┴────────────┴────────────────┘ │
└─────────────────────────────────────────────────────────┘
                    Token: 90.23% accuracy
```

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Encoder hidden_dim | 128 |
| Decoder d_model | 192 |
| Decoder n_heads | 8 |
| Decoder n_layers | 4 |
| Vocabulary size | 668 |
| Max sequence length | 32 |
| Dropout | 0.3 |

---

## Project Structure

```
gcode_fingerprinting/
├── src/miracle/              # Source code
│   ├── dataset/             # Data loading, preprocessing
│   ├── model/               # Model architectures
│   │   ├── mm_dtae_lstm.py        # Encoder
│   │   └── sensor_multihead_decoder.py  # Decoder
│   └── training/            # Losses, metrics
├── scripts/                  # Essential scripts (9 files)
│   ├── train_multihead.py
│   ├── train_sensor_multihead.py
│   ├── train_mm_dtae_lstm.py
│   ├── create_multilabel_stratified_splits.py
│   ├── evaluate_checkpoint.py
│   └── archived/            # Old scripts (142 files)
├── notebooks/               # Jupyter tutorials (01-19)
├── configs/                 # Configuration files
├── docs/                    # Documentation
│   └── archived/            # Old troubleshooting docs
├── data/                    # Vocabulary files
│   └── vocabulary_4digit_hybrid.json
├── outputs/                 # Training outputs
│   ├── mm_dtae_lstm_v2/     # Encoder checkpoint
│   ├── sensor_multihead_v3/ # Decoder checkpoint + ablations
│   └── stratified_splits_v2/  # Data splits (NPZ)
└── tests/                   # Unit tests
```

---

## Ablation Studies

### Sensor Modality Importance

| Modality Removed | Token Accuracy | Impact |
|-----------------|----------------|--------|
| Full Model | 90.23% | Baseline |
| − Proximity | 83.53% | **-6.70%** |
| − Pressure | 84.98% | **-5.25%** |
| − Accelerometer X | 87.54% | -2.69% |
| − Motor Current | 90.23% | 0.00% (Redundant) |

**Key Finding**: Proximity and pressure sensors are most critical for token prediction.

### Training Technique Ablation

| Configuration | Token Accuracy |
|--------------|----------------|
| Cross-Entropy Only | 90.49% |
| + Label Smoothing | 90.45% |
| + Focal (γ=2) | **90.68%** |
| + Focal (γ=3) | 90.26% |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [01_getting_started](notebooks/01_getting_started.ipynb) | Project overview and setup |
| [02_data_preprocessing](notebooks/02_data_preprocessing.ipynb) | Data pipeline and splits |
| [03_training_models](notebooks/03_training_models.ipynb) | Training encoder and decoder |
| [04_inference_prediction](notebooks/04_inference_prediction.ipynb) | Running inference |
| [08_model_evaluation](notebooks/08_model_evaluation.ipynb) | Comprehensive evaluation |
| [09_ablation_studies](notebooks/09_ablation_studies.ipynb) | Ablation analysis |

---

## Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](docs/QUICKSTART.md) | Get started quickly |
| [TRAINING.md](docs/TRAINING.md) | Training guide |
| [TRAINING_RESULTS_SUMMARY.md](docs/TRAINING_RESULTS_SUMMARY.md) | Final results (v16) |
| [PAPER_OUTLINE.md](docs/PAPER_OUTLINE.md) | Academic paper outline |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Architecture details |
| [API.md](docs/API.md) | REST API reference |

---

## Data Format

### Input (NPZ Files)

```python
data = np.load('train_sequences.npz', allow_pickle=True)
continuous = data['continuous']      # [N, 64, 155] float32
categorical = data['categorical']    # [N, 64, 4] int64
tokens = data['tokens']              # [N, 7] int64
operation_type = data['operation_type']  # [N] int64
```

### Vocabulary

- **File**: `data/vocabulary_4digit_hybrid.json`
- **Size**: 668 tokens
- **Commands**: G0, G1, G3, G53, M30, NONE
- **Parameters**: F, R, X, Y, Z with 4-digit precision (0000-9999)
- **Special**: PAD, UNK, SOS, EOS

---

## Citation

```bibtex
@misc{gcode_fingerprinting_2025,
  title={G-code Fingerprinting: Inferring 3D Printer Commands from Multi-Modal Sensor Data},
  author={ELE 588 Team},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/YOUR_USERNAME/gcode_fingerprinting}}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Course**: ELE 588 Applied Machine Learning
- **Frameworks**: PyTorch, Weights & Biases, FastAPI
- **Dataset**: Custom 3D printer sensor recordings

---

**Last Updated**: December 14, 2025 | **Version**: 2.0.0 (v16) | **Status**: Production-ready
