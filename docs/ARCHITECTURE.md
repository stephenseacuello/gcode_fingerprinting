# G-code Fingerprinting - System Architecture

This document provides comprehensive architectural visualizations of the G-code fingerprinting system.

## Table of Contents
- [System Overview](#system-overview)
- [Data Pipeline](#data-pipeline)
- [Model Architecture](#model-architecture)
- [Hierarchical Token Decomposition](#hierarchical-token-decomposition)
- [Training Workflow](#training-workflow)
- [Component Interactions](#component-interactions)
- [Deployment Architecture](#deployment-architecture)

---

## System Overview

```mermaid
graph TB
    subgraph "Data Layer"
        A[Raw CSV Files<br/>100 files, 8 sensors] --> B[Vocabulary Builder<br/>170 tokens]
        B --> C[Preprocessing<br/>Windowing + Normalization]
        C --> D[Processed Sequences<br/>3160 .npz files]
    end

    subgraph "Model Layer"
        D --> E[Dataset Loader<br/>PyTorch DataLoader]
        E --> F[Data Augmentation<br/>7 techniques]
        F --> G[MM_DTAE_LSTM<br/>Backbone]
        G --> H[MultiHeadGCodeLM<br/>5 Prediction Heads]
    end

    subgraph "Training Layer"
        H --> I[Loss Computation<br/>Focal + Huber]
        I --> J[Optimizer<br/>AdamW + Cosine LR]
        J --> K[Checkpointing<br/>Best + Latest]
        K --> L[W&B Tracking<br/>Metrics + Artifacts]
    end

    subgraph "Evaluation Layer"
        K --> M[Inference Engine]
        M --> N[Metrics Calculator<br/>Per-head Accuracy]
        N --> O[Visualization Generator<br/>14 Figures]
    end

    subgraph "Deployment Layer"
        K --> P[FastAPI Server<br/>REST Endpoints]
        P --> Q[Production Inference<br/>15ms latency]
    end

    style G fill:#4A90E2
    style H fill:#E24A4A
    style I fill:#F5A623
    style P fill:#7ED321
```

---

## Data Pipeline

```mermaid
flowchart TD
    Start([Raw Data<br/>100 CSV files]) --> A{Preprocessing Mode}

    A -->|Phase 1| B1[Build Vocabulary<br/>Scan all G-code]
    B1 --> B2[Apply Bucketing<br/>2-digit precision]
    B2 --> B3[Save gcode_vocab_v2.json<br/>170 tokens]

    A -->|Phase 2| C1[Load Master Column List<br/>8 sensor features]
    C1 --> C2[Sliding Window<br/>size=64, stride=16]
    C2 --> C3[Handle Missing Data<br/>Forward fill + Imputation]
    C3 --> C4[Normalize Features<br/>RobustScaler]
    C4 --> C5[Tokenize G-code<br/>Hierarchical decomposition]
    C5 --> C6[Extract Raw Values<br/>For regression head]
    C6 --> C7[Train/Val/Test Split<br/>70/15/15]
    C7 --> C8[Save .npz files<br/>outputs/processed_v2/]

    C8 --> D{Training Phase}
    D -->|Enabled| E1[Data Augmentation<br/>On-the-fly]
    E1 --> E2[Gaussian Noise<br/>σ=0.01]
    E1 --> E3[Time Warping<br/>±5%]
    E1 --> E4[Magnitude Scaling<br/>0.98-1.02x]
    E1 --> E5[Class Oversampling<br/>3x for rare tokens]
    E2 & E3 & E4 & E5 --> F[Augmented Batches]

    D -->|Disabled| G[Original Batches]

    F & G --> H[DataLoader<br/>Batch size 32]
    H --> I([Training Ready])

    style B3 fill:#7ED321
    style C8 fill:#7ED321
    style F fill:#F5A623
```

---

## Model Architecture

### Complete Neural Network Stack

```mermaid
graph TB
    subgraph "Input"
        I1[Sensor Data<br/>B×T×8]
        I2[Categorical Context<br/>Tool, State, etc.]
    end

    I1 --> M1
    I2 --> M2

    subgraph "MM_DTAE_LSTM Backbone"
        M1[Linear Modality Encoder<br/>Per-modality MLP]
        M1 --> M3[Positional Encoding<br/>Sinusoidal]
        M3 --> M4[Cross-Modal Fusion<br/>Attention + Gates]

        M2[Context Embeddings<br/>Learned vectors]
        M2 --> M4

        M4 --> M5[Add Noise + Mask<br/>Denoising strategy]
        M5 --> M6[DTAE Encoder<br/>2-layer Transformer]
        M6 --> M7[DTAE Decoder<br/>Reconstruction]
        M7 --> M8[LSTM Layers<br/>2-6 layers, 256-512 hidden]
        M8 --> M9[Contextualized Memory<br/>B×T×d_model]
    end

    M9 --> L1

    subgraph "MultiHeadGCodeLM"
        L1[Token Embedding<br/>Learned lookup]
        L1 --> L2[Positional Encoding]
        L2 --> L3[Causal Transformer Decoder<br/>2-5 layers, Multi-head Attention]
        L3 --> L4{5 Prediction Heads}

        L4 -->|Head 1| H1[Token Type<br/>Linear → Softmax<br/>4 classes]
        L4 -->|Head 2| H2[Command ID<br/>Linear → Softmax<br/>15 classes]
        L4 -->|Head 3| H3[Parameter Type<br/>Linear → Softmax<br/>10 classes]
        L4 -->|Head 4| H4[Parameter Value<br/>Linear → Regression<br/>Continuous output]
        L4 -->|Head 5| H5[Operation Type<br/>Attention Pool → Linear<br/>10 classes]
    end

    H1 & H2 & H3 & H4 & H5 --> O1

    subgraph "Loss & Optimization"
        O1[Multi-head Loss<br/>Weighted combination]
        O1 --> O2[Type: CrossEntropy<br/>w=1.0]
        O1 --> O3[Command: FocalLoss<br/>w=5.0, γ=2.5]
        O1 --> O4[ParamType: CrossEntropy<br/>w=3.0]
        O1 --> O5[ParamValue: HuberLoss<br/>w=1.0, δ=1.0]
        O1 --> O6[Operation: CrossEntropy<br/>w=2.0]

        O2 & O3 & O4 & O5 & O6 --> O7[Total Loss]
        O7 --> O8[AdamW Optimizer<br/>lr=5e-5, wd=0.05]
        O8 --> O9[Cosine LR Schedule<br/>+ Warmup]
        O9 --> O10[Gradient Clipping<br/>max_norm=1.0]
    end

    style M4 fill:#4A90E2
    style M6 fill:#4A90E2
    style M8 fill:#4A90E2
    style L3 fill:#E24A4A
    style H1 fill:#7ED321
    style H2 fill:#7ED321
    style H3 fill:#7ED321
    style H4 fill:#F5A623
    style H5 fill:#7ED321
    style O3 fill:#E24A4A
```

### Tensor Shapes Through the Network

```mermaid
flowchart LR
    A["Input Sensor Data<br/>[B, T, 8]"] --> B["Modality Encoding<br/>[B, T, d_model]"]
    B --> C["Cross-Modal Fusion<br/>[B, T, d_model]"]
    C --> D["DTAE Processing<br/>[B, T, d_model]"]
    D --> E["LSTM Output<br/>[B, T, d_model]"]
    E --> F["Decoder Hidden<br/>[B, T, d_model]"]
    F --> G1["Type Logits<br/>[B, T, 4]"]
    F --> G2["Cmd Logits<br/>[B, T, 15]"]
    F --> G3["ParamType Logits<br/>[B, T, 10]"]
    F --> G4["ParamValue<br/>[B, T, 1]"]
    F --> G5["Operation<br/>[B, 10]"]

    style E fill:#4A90E2
    style F fill:#E24A4A
    style G4 fill:#F5A623
```

---

## Hierarchical Token Decomposition

This is the key innovation that solves the 130:1 class imbalance problem.

```mermaid
graph TD
    subgraph "Traditional Approach (170-token vocab)"
        T1[Single Softmax<br/>170 classes]
        T1 -.->|Problem| T2[Severe Class Imbalance<br/>G1: 45%, X50: 0.3%]
        T2 -.->|Result| T3[Poor Rare Token Learning<br/>Mode Collapse]
    end

    subgraph "Hierarchical Approach (5 heads)"
        H1[Head 1: Token Type<br/>4 classes]
        H2[Head 2: Command<br/>15 classes]
        H3[Head 3: Param Type<br/>10 classes]
        H4[Head 4: Param Value<br/>Regression]
        H5[Head 5: Operation<br/>10 classes]

        H1 -->|99.8% acc| R1
        H2 -->|100% acc| R1
        H3 -->|84.3% acc| R1
        H4 -->|56.2% acc| R1
        H5 -->|92% acc| R1
        R1[Token Reconstruction<br/>Combine predictions]
    end

    style T3 fill:#E24A4A
    style R1 fill:#7ED321
```

### Example Token Decomposition

```mermaid
flowchart TB
    subgraph "Example 1: Parameter Token 'X120.5'"
        E1A[Full Token: X120.5] --> E1B{Decompose}
        E1B --> E1C[Type: PARAM<br/>class 2/4]
        E1B --> E1D[Command: PAD<br/>N/A]
        E1B --> E1E[Param Type: X<br/>class 0/10]
        E1B --> E1F[Param Value: 120.5<br/>regression]
        E1B --> E1G[Operation: adaptive<br/>sequence-level]
    end

    subgraph "Example 2: Command Token 'G1'"
        E2A[Full Token: G1] --> E2B{Decompose}
        E2B --> E2C[Type: CMD<br/>class 1/4]
        E2B --> E2D[Command: G1<br/>class 1/15]
        E2B --> E2E[Param Type: PAD<br/>N/A]
        E2B --> E2F[Param Value: PAD<br/>N/A]
        E2B --> E2G[Operation: face<br/>sequence-level]
    end

    subgraph "Example 3: Numeric Token '50' (from F50)"
        E3A[Full Token: NUM_F_50] --> E3B{Decompose}
        E3B --> E3C[Type: NUMERIC<br/>class 3/4]
        E3B --> E3D[Command: PAD<br/>N/A]
        E3B --> E3E[Param Type: F<br/>class 3/10]
        E3B --> E3F[Param Value: 50.0<br/>regression]
        E3B --> E3G[Operation: pocket<br/>sequence-level]
    end

    style E1C fill:#7ED321
    style E2D fill:#4A90E2
    style E3F fill:#F5A623
```

---

## Training Workflow

```mermaid
stateDiagram-v2
    [*] --> LoadConfig
    LoadConfig --> BuildVocab: First run
    LoadConfig --> LoadVocab: Vocab exists

    BuildVocab --> LoadVocab
    LoadVocab --> LoadData

    LoadData --> InitModel
    InitModel --> LoadCheckpoint: Resume training
    InitModel --> RandomInit: Fresh start

    LoadCheckpoint --> TrainingLoop
    RandomInit --> TrainingLoop

    state TrainingLoop {
        [*] --> Epoch
        Epoch --> BatchLoop

        state BatchLoop {
            [*] --> LoadBatch
            LoadBatch --> Augment: Training
            LoadBatch --> NoAugment: Validation

            Augment --> Forward
            NoAugment --> Forward

            Forward --> ComputeLoss
            ComputeLoss --> Backward: Training
            ComputeLoss --> Metrics: Validation

            Backward --> ClipGrad
            ClipGrad --> UpdateWeights
            UpdateWeights --> Metrics

            Metrics --> NextBatch: More batches
            Metrics --> [*]: Epoch done
        }

        BatchLoop --> Validate
        Validate --> EarlyStop: Check patience

        EarlyStop --> SaveCheckpoint: Best model
        EarlyStop --> Epoch: Continue
        EarlyStop --> [*]: Max epochs / Early stop
    }

    TrainingLoop --> FinalEval
    FinalEval --> SWA: If enabled
    FinalEval --> SaveFinal: Standard

    SWA --> SaveFinal
    SaveFinal --> [*]
```

### Hyperparameter Sweep Workflow

```mermaid
graph TB
    A[Create Sweep Config<br/>sweep_config.yaml] --> B[wandb sweep]
    B --> C[Generate Sweep ID]
    C --> D{Launch Agents}

    D --> E1[Agent 1<br/>GPU 0]
    D --> E2[Agent 2<br/>GPU 1]
    D --> E3[Agent N<br/>Cloud]

    E1 & E2 & E3 --> F[Bayesian Optimization<br/>Sample hyperparameters]

    F --> G[Train Model<br/>Full training loop]
    G --> H[Report Metrics<br/>Val loss, accuracies]
    H --> I{Sweep Complete?}

    I -->|No| F
    I -->|Yes| J[Select Best Config]
    J --> K[Train Final Model<br/>Best hyperparameters]
    K --> L[Save Ultimate Model]

    style F fill:#4A90E2
    style K fill:#7ED321
```

---

## Component Interactions

```mermaid
graph TB
    subgraph "Core Modules"
        A[dataset/<br/>Data loading]
        B[model/<br/>Neural networks]
        C[training/<br/>Training loops]
        D[inference/<br/>Evaluation]
        E[visualization/<br/>Plotting]
        F[utilities/<br/>Tokenizer, helpers]
        G[config/<br/>Configuration]
        H[api/<br/>REST server]
    end

    subgraph "Data Flow"
        A -->|Batches| B
        B -->|Predictions| C
        C -->|Checkpoints| D
        D -->|Results| E
        F -->|Tokenization| A
        G -->|Parameters| B
        G -->|Parameters| C
        B -->|Loaded model| H
    end

    subgraph "External Dependencies"
        I[PyTorch]
        J[NumPy]
        K[Weights & Biases]
        L[FastAPI]
        M[Matplotlib]
    end

    B -.->|Uses| I
    A -.->|Uses| J
    C -.->|Logs to| K
    H -.->|Built on| L
    E -.->|Uses| M

    style A fill:#4A90E2
    style B fill:#E24A4A
    style C fill:#F5A623
    style D fill:#7ED321
    style H fill:#BD10E0
```

### File Organization

```mermaid
graph TD
    subgraph "Source Code (src/miracle/)"
        S1[dataset/]
        S2[model/]
        S3[training/]
        S4[inference/]
        S5[visualization/]
        S6[utilities/]
        S7[config/]
        S8[api/]
    end

    subgraph "Scripts (scripts/)"
        SC1[train_multihead.py]
        SC2[train_sweep.py]
        SC3[evaluate_*.py]
        SC4[generate_*.py]
        SC5[api_server.py]
    end

    subgraph "Data (data/)"
        D1[*.csv - Raw data]
        D2[gcode_vocab_v2.json]
        D3[class_weights.pt]
    end

    subgraph "Outputs (outputs/)"
        O1[processed_v2/]
        O2[training/]
        O3[figures/]
        O4[wandb_sweeps/]
    end

    subgraph "Configuration (configs/)"
        C1[ultimate_model.json]
        C2[phase1_*.json]
        C3[sweep_config.yaml]
    end

    D1 --> SC1
    D2 --> SC1
    C1 --> SC1
    SC1 --> O2
    O2 --> SC3
    SC3 --> O3

    SC2 --> O4
    C3 --> SC2

    O2 --> SC5

    style SC1 fill:#E24A4A
    style O2 fill:#7ED321
```

---

## Deployment Architecture

```mermaid
graph TB
    subgraph "Production Environment"
        A[FastAPI Server<br/>Uvicorn]
        B[Model Checkpoint<br/>checkpoint_best.pt]
        C[Vocabulary<br/>gcode_vocab_v2.json]
        D[Device Manager<br/>CUDA/MPS/CPU]
    end

    B --> A
    C --> A
    D --> A

    subgraph "API Endpoints"
        E[POST /predict<br/>Single prediction]
        F[POST /batch_predict<br/>Batch processing]
        G[GET /health<br/>Health check]
        H[GET /model_info<br/>Metadata]
    end

    A --> E
    A --> F
    A --> G
    A --> H

    subgraph "Client Applications"
        I[Web Dashboard]
        J[Python SDK]
        K[Manufacturing System]
    end

    E --> I
    F --> J
    E & F --> K

    subgraph "Performance"
        L[Mac M1: 67 req/s<br/>15ms latency]
        M[RTX 3090: 125 req/s<br/>8ms latency]
    end

    A -.-> L
    A -.-> M

    style A fill:#7ED321
    style B fill:#4A90E2
    style L fill:#F5A623
    style M fill:#F5A623
```

### API Request Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API Server
    participant P as Preprocessor
    participant M as Model
    participant D as Decoder

    C->>A: POST /predict<br/>{sensor_data}
    A->>P: Normalize & tokenize
    P->>P: Apply RobustScaler
    P->>P: Create windows
    P-->>A: Preprocessed tensor

    A->>M: Forward pass
    M->>M: MM_DTAE_LSTM
    M->>M: MultiHeadGCodeLM
    M-->>A: 5 head predictions

    A->>D: Reconstruct tokens
    D->>D: Combine head outputs
    D->>D: Validate grammar
    D-->>A: G-code string

    A-->>C: {predicted_gcode,<br/>confidence,<br/>per_head_probs}

    Note over C,D: Total latency: 8-15ms
```

---

## Key Innovations

### 1. Hierarchical Decomposition
- Eliminates 130:1 class imbalance
- Shared learning across token types
- Smooth numeric prediction via regression

### 2. Multi-Modal Fusion
- Cross-attention between sensor modalities
- Learned modality importance gates
- Robust to missing modalities

### 3. Denoising Pretraining
- DTAE learns robust representations
- Improves generalization to noisy sensors
- Regularization effect

### 4. Advanced Loss Design
- Focal loss for command prediction (most imbalanced)
- Huber loss for numeric regression (outlier-robust)
- Weighted combination tuned via hyperparameter search

### 5. Production-Ready Infrastructure
- FastAPI with automatic docs
- Device-agnostic deployment
- Sub-15ms inference latency

---

## Performance Summary

| Component | Metric | Value |
|-----------|--------|-------|
| **Token Type Head** | Accuracy | 99.8% ± 0.1% |
| **Command Head** | Accuracy | 100.0% ± 0.0% |
| **Param Type Head** | Accuracy | 84.3% ± 1.2% |
| **Param Value Head** | MAE | 8.2 ± 1.5 |
| **Param Value Head** | Accuracy (bucket) | 56.2% ± 2.8% |
| **Operation Type Head** | Accuracy | 92.0% ± 2.1% |
| **Overall** | String Match | 45-50% |
| **Overall** | Grammar Valid | 95%+ |
| **Inference** | Latency (M1) | 15ms |
| **Inference** | Throughput (M1) | 67 req/s |
| **Model Size** | Parameters | 2.5M (default)<br/>12M (ultimate) |
| **Training** | Time per epoch | 2-5 min |
| **Training** | Total time | 1-2 hours |

---

## References

- **Code**: [src/miracle/](../src/miracle/)
- **Training Scripts**: [scripts/](../scripts/)
- **Configuration**: [configs/](../configs/)
- **Documentation**: [docs/](../docs/)

---

*Generated: 2025-11-30*
*Project: G-code Fingerprinting with Multi-Head Transformers*
