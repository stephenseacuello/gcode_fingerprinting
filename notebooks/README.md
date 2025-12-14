# G-code Fingerprinting Notebooks

This directory contains interactive Jupyter notebooks for exploring, training, and evaluating the G-code fingerprinting model.

## Model Performance

| Metric | Value |
|--------|-------|
| **Operation Classification** | 100% (frozen encoder) |
| **Token Accuracy** | 90.23% (decoder) |

## Quick Start

For a complete walkthrough, follow the notebooks in order:

```
00 → 01 → 02 → 03 → 04 → 08
```

For quick experimentation with a trained model:

```
01 → 04 → 08
```

---

## Notebook Overview

### Data Exploration & Preprocessing

| Notebook | Description | Prerequisites |
|----------|-------------|---------------|
| [00_raw_data_analysis.ipynb](00_raw_data_analysis.ipynb) | Explore raw sensor data and G-code files | Raw data in `data/` |
| [01_getting_started.ipynb](01_getting_started.ipynb) | Project intro, environment setup, quick demo | None |
| [02_data_preprocessing.ipynb](02_data_preprocessing.ipynb) | Full preprocessing pipeline walkthrough | Raw data in `data/` |

### Training & Inference

| Notebook | Description | Prerequisites |
|----------|-------------|---------------|
| [03_training_models.ipynb](03_training_models.ipynb) | Train models from scratch with live metrics | Preprocessed data |
| [04_inference_prediction.ipynb](04_inference_prediction.ipynb) | Run predictions, batch inference, decode tokens | Trained checkpoint |

### Deployment & APIs

| Notebook | Description | Prerequisites |
|----------|-------------|---------------|
| [05_api_usage.ipynb](05_api_usage.ipynb) | REST API usage with requests/aiohttp | Running API server |
| [06_dashboard_usage.ipynb](06_dashboard_usage.ipynb) | Interactive Flask dashboard guide | Dashboard running |

### Evaluation & Analysis

| Notebook | Description | Prerequisites |
|----------|-------------|---------------|
| [07_hyperparameter_sweeps.ipynb](07_hyperparameter_sweeps.ipynb) | W&B sweep configuration and analysis | W&B account |
| [08_model_evaluation.ipynb](08_model_evaluation.ipynb) | Comprehensive model evaluation | Trained checkpoint |
| [09_ablation_studies.ipynb](09_ablation_studies.ipynb) | Component contribution analysis | Multiple checkpoints |
| [10_visualization_experiments.ipynb](10_visualization_experiments.ipynb) | Publication-quality figures | Evaluation results |

### Advanced Analysis

| Notebook | Description | Prerequisites |
|----------|-------------|---------------|
| [11_model_interpretability.ipynb](11_model_interpretability.ipynb) | Attention visualization, saliency maps, embeddings | Trained checkpoint |
| [12_error_analysis.ipynb](12_error_analysis.ipynb) | Error patterns, hard examples, failure taxonomy | Trained checkpoint, test data |
| [14_robustness_testing.ipynb](14_robustness_testing.ipynb) | Noise, dropout, adversarial robustness | Trained checkpoint |
| [17_uncertainty_quantification.ipynb](17_uncertainty_quantification.ipynb) | MC Dropout, calibration, selective prediction | Trained checkpoint |

### Architecture & Training

| Notebook | Description | Prerequisites |
|----------|-------------|---------------|
| [15_data_augmentation.ipynb](15_data_augmentation.ipynb) | Time warping, mixup, sensor augmentations | Sample data |
| [16_architecture_comparison.ipynb](16_architecture_comparison.ipynb) | LSTM vs Transformer vs CNN comparison | None |
| [18_transfer_learning.ipynb](18_transfer_learning.ipynb) | Fine-tuning, few-shot, domain adaptation | Trained checkpoint |

### Production & Deployment

| Notebook | Description | Prerequisites |
|----------|-------------|---------------|
| [13_deployment_guide.ipynb](13_deployment_guide.ipynb) | ONNX export, quantization, Docker | Trained checkpoint |
| [19_streaming_inference.ipynb](19_streaming_inference.ipynb) | Sliding window, real-time processing | Trained checkpoint |

---

## Prerequisites

### Required Files

```
gcode_fingerprinting/
├── data/
│   └── vocabulary_4digit_hybrid.json    # 4-digit hybrid vocabulary
├── outputs/
│   ├── stratified_splits_v2/            # OR multilabel_stratified_splits/
│   │   ├── train_sequences.npz          # Training data
│   │   ├── val_sequences.npz            # Validation data
│   │   ├── test_sequences.npz           # Test data
│   │   └── split_info.json              # Split statistics
│   ├── sensor_multihead_v3/
│   │   ├── best_model.pt                # Best decoder checkpoint
│   │   └── results.json                 # Training results
│   └── mm_dtae_lstm_v2/
│       └── best_model.pt                # Encoder checkpoint (frozen)
```

### Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Ensure PYTHONPATH includes src/
export PYTHONPATH=src:$PYTHONPATH

# Start Jupyter
jupyter notebook notebooks/
```

### Package Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- Weights & Biases (for sweeps)
- Flask (for dashboard)

---

## Recommended Learning Paths

### Path 1: Complete Beginner
For users new to the project:

1. **[01_getting_started](01_getting_started.ipynb)** - Understand the project
2. **[00_raw_data_analysis](00_raw_data_analysis.ipynb)** - Explore the data
3. **[02_data_preprocessing](02_data_preprocessing.ipynb)** - Learn preprocessing
4. **[03_training_models](03_training_models.ipynb)** - Train your first model
5. **[08_model_evaluation](08_model_evaluation.ipynb)** - Evaluate results

### Path 2: Quick Experimentation
For users with a trained model:

1. **[01_getting_started](01_getting_started.ipynb)** - Quick setup check
2. **[04_inference_prediction](04_inference_prediction.ipynb)** - Run predictions
3. **[08_model_evaluation](08_model_evaluation.ipynb)** - Analyze performance

### Path 3: Research & Publication
For preparing papers/reports:

1. **[08_model_evaluation](08_model_evaluation.ipynb)** - Get metrics
2. **[09_ablation_studies](09_ablation_studies.ipynb)** - Ablation analysis
3. **[10_visualization_experiments](10_visualization_experiments.ipynb)** - Create figures

### Path 4: Deployment
For production deployment:

1. **[05_api_usage](05_api_usage.ipynb)** - Test API endpoints
2. **[06_dashboard_usage](06_dashboard_usage.ipynb)** - Interactive dashboard
3. **[13_deployment_guide](13_deployment_guide.ipynb)** - ONNX export, Docker
4. **[19_streaming_inference](19_streaming_inference.ipynb)** - Real-time inference

### Path 5: Deep Dive (Advanced)
For thorough model analysis:

1. **[11_model_interpretability](11_model_interpretability.ipynb)** - Understand what model learns
2. **[12_error_analysis](12_error_analysis.ipynb)** - Diagnose failures
3. **[14_robustness_testing](14_robustness_testing.ipynb)** - Test reliability
4. **[17_uncertainty_quantification](17_uncertainty_quantification.ipynb)** - Calibrate confidence

### Path 6: Architecture Research
For exploring model design:

1. **[16_architecture_comparison](16_architecture_comparison.ipynb)** - Compare architectures
2. **[15_data_augmentation](15_data_augmentation.ipynb)** - Improve training
3. **[18_transfer_learning](18_transfer_learning.ipynb)** - Adapt to new domains

---

## Key Concepts

### Two-Stage Architecture

```
Sensor Data → MM-DTAE-LSTM (frozen) → Embeddings → SensorMultiHeadDecoder → G-code
(155 + 4)        (Encoder)              128-dim         (Decoder)           (Tokens)
```

**Stage 1: Encoder (MM-DTAE-LSTM v2)**
- Processes sensor windows [64, 155+4]
- Outputs 128-dim embeddings
- **100% operation classification accuracy**
- Frozen during decoder training

**Stage 2: Decoder (SensorMultiHeadDecoder v3)**
- d_model: 192, n_heads: 8, n_layers: 4
- Multi-head output structure
- **90.23% token accuracy**

### Token Structure (4-Digit Hybrid)

G-code tokens are encoded into 7 positions:

```
[Type, Command, Param, Sign, Digit1, Digit2, Digit3]
```

- **Type**: 4 classes (SPECIAL, COMMAND, PARAM, NUMERIC)
- **Command**: 6 classes (G0, G1, G3, G53, M30, NONE)
- **Param Type**: 10 classes (X, Y, Z, F, R, NONE, etc.)
- **Digits**: 10 classes each (0-9)

### Model Output Heads

| Head | Classes | Description |
|------|---------|-------------|
| Operation | 9 | Machining operation type (from encoder) |
| Type | 4 | Token type classification |
| Command | 6 | G-code command (G0, G1, etc.) |
| Param Type | 10 | Parameter axis (X, Y, Z, etc.) |
| Digit 1-3 | 10 each | Numeric value digits |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: miracle` | Set `PYTHONPATH=src` or run from project root |
| No checkpoints found | Check `outputs/sensor_multihead_v3/best_model.pt` |
| Vocabulary missing | Check `data/vocabulary_4digit_hybrid.json` |
| CUDA out of memory | Reduce batch size or use CPU |
| MPS errors | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Import errors in notebook | Restart kernel after path changes |

---

## Quick Commands

```bash
# Create stratified splits
PYTHONPATH=src .venv/bin/python scripts/create_multilabel_stratified_splits.py \
    --data-dir outputs/processed_v2 \
    --output-dir outputs/multilabel_stratified_splits \
    --val-size 0.15 --test-size 0.15

# Train decoder model
PYTHONPATH=src .venv/bin/python scripts/train_sensor_multihead.py \
    --split-dir outputs/stratified_splits_v2 \
    --vocab-path data/vocabulary_4digit_hybrid.json \
    --encoder-path outputs/mm_dtae_lstm_v2/best_model.pt \
    --output-dir outputs/sensor_multihead_v3 \
    --max-epochs 150

# Start dashboard
PYTHONPATH=src .venv/bin/python flask_dashboard.py

# Run W&B sweep
wandb sweep configs/sweep_config.yaml
```

---

## Contributing

When adding new notebooks:

1. Follow the naming convention: `XX_descriptive_name.ipynb`
2. Include a Table of Contents in the first cell
3. Add navigation links in the final cell
4. Update this README with the new notebook

### Notebook Structure

Each notebook should include:

1. **Title & TOC** - Clear title with table of contents
2. **Environment Setup** - Import statements and path configuration
3. **Main Content** - Numbered sections with explanations
4. **Summary** - Key takeaways
5. **Navigation** - Links to previous/next notebooks

---

## File Structure

```
notebooks/
├── README.md                           # This file
│
│   # Core Pipeline (00-10)
├── 00_raw_data_analysis.ipynb          # Data exploration
├── 01_getting_started.ipynb            # Project introduction
├── 02_data_preprocessing.ipynb         # Preprocessing pipeline
├── 03_training_models.ipynb            # Model training
├── 04_inference_prediction.ipynb       # Running predictions
├── 05_api_usage.ipynb                  # REST API usage
├── 06_dashboard_usage.ipynb            # Interactive dashboard
├── 07_hyperparameter_sweeps.ipynb      # W&B sweeps
├── 08_model_evaluation.ipynb           # Evaluation & metrics
├── 09_ablation_studies.ipynb           # Ablation experiments
├── 10_visualization_experiments.ipynb  # Publication figures
│
│   # Advanced Topics (11-19)
├── 11_model_interpretability.ipynb     # Attention & saliency analysis
├── 12_error_analysis.ipynb             # Error patterns & failure modes
├── 13_deployment_guide.ipynb           # ONNX, quantization, Docker
├── 14_robustness_testing.ipynb         # Noise & adversarial testing
├── 15_data_augmentation.ipynb          # Time series augmentation
├── 16_architecture_comparison.ipynb    # LSTM vs Transformer vs CNN
├── 17_uncertainty_quantification.ipynb # MC Dropout & calibration
├── 18_transfer_learning.ipynb          # Fine-tuning & domain adaptation
└── 19_streaming_inference.ipynb        # Real-time inference
```

---

**Project Documentation:** [../docs/](../docs/)

**Main Repository:** [../](../)
