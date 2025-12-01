# System Architecture Overview

This document provides visual architecture diagrams for the G-code fingerprinting system.

---

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph "Data Collection"
        A[3D Printer] -->|Sensor Data| B[Data Acquisition]
        B -->|Raw Signals| C[Preprocessing]
    end

    subgraph "Training Pipeline"
        C -->|Processed Data| D[Data Augmentation]
        D -->|Augmented Batches| E[Multi-Head Transformer]
        E -->|Predictions| F[Multi-Task Loss]
        F -->|Gradients| E
        E -->|Checkpoint| G[(Model Registry)]
    end

    subgraph "Evaluation & Optimization"
        G -->|Best Model| H[Hyperparameter Sweeps]
        H -->|Optimal Config| I[Production Training]
        I -->|Final Model| J[Model Export ONNX]
        J -->|Quantized Models| K[Deployment Package]
    end

    subgraph "Production Deployment"
        K -->|Deploy| L[FastAPI Server]
        L -->|REST API| M[Client Applications]
        L -.->|Metrics| N[Prometheus]
        N -.->|Visualize| O[Grafana]
    end

    style E fill:#90EE90
    style L fill:#87CEEB
    style K fill:#FFD700
```

---

## 2. Data Processing Pipeline

```mermaid
flowchart LR
    subgraph "Input"
        A[Raw Sensor Data<br/>135 channels<br/>@1kHz]
    end

    subgraph "Preprocessing"
        B[Normalization<br/>Z-score]
        C[Resampling<br/>Fixed length]
        D[Feature Engineering<br/>Categorical]
    end

    subgraph "Augmentation"
        E[Sensor Noise<br/>σ=0.02]
        F[Temporal Shift<br/>±3 steps]
        G[Magnitude Scale<br/>0.95-1.05]
        H[Mixup<br/>α=0.3]
    end

    subgraph "Output"
        I[Training Batch<br/>Continuous: B×T×135<br/>Categorical: B×T×4]
    end

    A --> B --> C --> D --> E --> F --> G --> H --> I

    style I fill:#90EE90
```

---

## 3. Model Architecture

```mermaid
graph TB
    subgraph "Input Layer"
        A[Continuous Sensor<br/>B×64×135]
        B[Categorical Features<br/>B×64×4]
    end

    subgraph "Embedding Layer"
        C[Linear Projection<br/>135→d_model]
        D[Category Embedding<br/>4→d_model]
        E[Positional Encoding]
    end

    subgraph "Transformer Encoder"
        F[Multi-Head Attention<br/>h=8 heads]
        G[Feed-Forward<br/>d_ff=512]
        H[Layer Norm + Residual]
    end

    subgraph "Multi-Head Decoder"
        I[Type Head<br/>2 classes]
        J[Command Head<br/>~170 tokens]
        K[Param Type Head<br/>~170 tokens]
        L[Param Value Head<br/>~170 tokens]
    end

    subgraph "Auxiliary Tasks"
        M[Reconstruction<br/>MSE Loss]
        N[Fingerprint<br/>Contrastive]
    end

    A --> C --> E
    B --> D --> E
    E --> F --> G --> H
    H -.->|Encoder Output| M
    H -.->|Global Pool| N
    H --> I & J & K & L

    style F fill:#87CEEB
    style I fill:#90EE90
    style J fill:#90EE90
    style K fill:#90EE90
    style L fill:#90EE90
```

---

## 4. Multi-Head Token Decomposition

```mermaid
flowchart TB
    subgraph "Token Decomposition"
        A[G-code Token<br/>e.g., 'G0_X_1575']
        B[Type<br/>0=Command<br/>1=Parameter]
        C[Command<br/>G0, M104, etc.]
        D[Param Type<br/>X, Y, Z, E, F, etc.]
        E[Param Value<br/>Bucketed: 15]
    end

    subgraph "Model Predictions"
        F[Type Logits<br/>B×64×2]
        G[Command Logits<br/>B×64×170]
        H[Param Type Logits<br/>B×64×170]
        I[Param Value Logits<br/>B×64×170]
    end

    subgraph "Reconstruction"
        J[Argmax Each Head]
        K[Compose Token<br/>type+cmd+ptype+pval]
        L[Final Token ID]
    end

    A -.->|Training| B & C & D & E
    B -.->|Supervise| F
    C -.->|Supervise| G
    D -.->|Supervise| H
    E -.->|Supervise| I

    F --> J
    G --> J
    H --> J
    I --> J
    J --> K --> L

    style A fill:#FFD700
    style L fill:#90EE90
```

---

## 5. Training Loop

```mermaid
sequenceDiagram
    participant D as DataLoader
    participant A as Augmenter
    participant M as Model
    participant L as Loss Function
    participant O as Optimizer
    participant S as Scheduler

    loop Each Epoch
        D->>A: Get batch
        A->>M: Augmented batch
        M->>L: Forward pass (logits)
        L->>L: Compute multi-head loss
        L->>O: Backward pass
        O->>M: Update weights
        O->>S: Step optimizer
        S->>S: Adjust learning rate

        alt Validation
            M->>M: Eval mode
            M->>L: Validation metrics
            L->>L: Track best model
        end
    end

    Note over M,L: Early stopping if<br/>no improvement
```

---

## 6. Hyperparameter Optimization

```mermaid
graph LR
    subgraph "Sweep Configurations"
        A[Vocabulary<br/>Bucketing]
        B[Augmentation<br/>Optimization]
        C[Warmup<br/>Scheduler]
        D[Architecture<br/>Search]
        E[Loss<br/>Weighting]
    end

    subgraph "W&B Sweeps"
        F[Bayesian<br/>Optimization]
        G[Grid<br/>Search]
        H[Hyperband<br/>Early Stop]
    end

    subgraph "Results"
        I[Best Config<br/>JSON]
        J[Performance<br/>Metrics]
        K[Parameter<br/>Importance]
    end

    A & B & C & D & E --> F & G
    F & G --> H
    H --> I & J & K

    I -->|Train| L[Production<br/>Model]

    style I fill:#FFD700
    style L fill:#90EE90
```

---

## 7. Inference Pipeline

```mermaid
flowchart TB
    subgraph "Input"
        A[Sensor Data<br/>Continuous + Categorical]
    end

    subgraph "Preprocessing"
        B[Normalize]
        C[Tensorize]
    end

    subgraph "Model Inference"
        D[Forward Pass<br/>Encoder + Decoder]
        E[Multi-Head Logits]
    end

    subgraph "Decoding Strategy"
        F{Method?}
        G[Greedy<br/>Argmax]
        H[Beam Search<br/>Width=5]
        I[Temperature<br/>Sampling]
    end

    subgraph "Post-Processing"
        J[Compose Tokens]
        K[Detokenize]
        L[G-code Sequence]
    end

    A --> B --> C --> D --> E --> F
    F -->|greedy| G
    F -->|beam| H
    F -->|sample| I
    G & H & I --> J --> K --> L

    style L fill:#90EE90
```

---

## 8. Production Deployment

```mermaid
graph TB
    subgraph "Model Artifacts"
        A[PyTorch<br/>Checkpoint]
        B[ONNX<br/>FP32]
        C[ONNX<br/>FP16]
        D[ONNX<br/>INT8]
    end

    subgraph "Docker Container"
        E[FastAPI<br/>Server]
        F[Model Manager<br/>Singleton]
        G[Inference<br/>Engine]
    end

    subgraph "API Endpoints"
        H[/predict]
        I[/batch_predict]
        J[/fingerprint]
        K[/health]
    end

    subgraph "Clients"
        L[Python<br/>Client]
        M[Web<br/>Dashboard]
        N[Edge<br/>Device]
    end

    subgraph "Monitoring"
        O[Prometheus]
        P[Grafana]
    end

    A -->|Export| B
    B -->|Quantize| C & D

    B & C & D --> F
    F --> G
    G --> H & I & J & K

    H & I & J --> L & M & N

    E -.->|Metrics| O
    O -.->|Visualize| P

    style E fill:#87CEEB
    style G fill:#90EE90
```

---

## 9. Quantization Workflow

```mermaid
flowchart LR
    subgraph "Input"
        A[ONNX Model<br/>FP32]
    end

    subgraph "FP16 Quantization"
        B[Convert Weights<br/>FP32→FP16]
        C[FP16 Model<br/>50% size]
    end

    subgraph "INT8 Dynamic"
        D[Quantize Weights<br/>FP32→INT8]
        E[INT8 Dynamic<br/>75% reduction]
    end

    subgraph "INT8 Static"
        F[Calibration Data<br/>100+ samples]
        G[Quantize Weights<br/>+ Activations]
        H[INT8 Static<br/>75% reduction<br/>Best speed]
    end

    subgraph "Validation"
        I[Accuracy Check<br/>vs FP32]
        J[Speed Benchmark<br/>Latency/Throughput]
    end

    A --> B --> C
    A --> D --> E
    A & F --> G --> H

    C & E & H --> I & J

    style H fill:#90EE90
```

---

## 10. End-to-End Workflow

```mermaid
graph TB
    A[Raw 3D Printer<br/>Sensor Data] --> B[Preprocessing<br/>Pipeline]
    B --> C{Training or<br/>Inference?}

    C -->|Training| D[Data Augmentation]
    D --> E[Train Multi-Head<br/>Transformer]
    E --> F[Hyperparameter<br/>Sweeps]
    F --> G[Best Model<br/>Checkpoint]

    C -->|Inference| H[Normalize<br/>Input]

    G --> I[Export to<br/>ONNX]
    I --> J[Quantization<br/>FP16/INT8]
    J --> K[Deploy<br/>FastAPI]

    H --> K
    K --> L[REST API<br/>Predictions]

    L --> M[G-code<br/>Sequence]
    L --> N[Machine<br/>Fingerprint]

    style M fill:#90EE90
    style N fill:#FFD700
```

---

## Diagram Legend

- **Green boxes**: Final outputs/results
- **Blue boxes**: Core components/services
- **Yellow boxes**: Important artifacts
- **Dotted lines**: Monitoring/auxiliary flows
- **Solid lines**: Main data/control flow

---

## Component Descriptions

### Data Collection
- **3D Printer**: RepRap/Prusa machines with sensor arrays
- **Data Acquisition**: Real-time sensor reading (1kHz sampling)
- **Preprocessing**: Normalization, resampling, feature engineering

### Model Training
- **Multi-Head Transformer**: Encoder-decoder architecture
- **Data Augmentation**: Noise, shifts, scaling, mixup
- **Multi-Task Loss**: Command + parameter + auxiliary losses

### Hyperparameter Optimization
- **W&B Sweeps**: Bayesian optimization + Hyperband
- **5 Major Sweeps**: Vocabulary, augmentation, warmup, architecture, loss
- **300+ Experiments**: Systematic search for >70% accuracy

### Production Deployment
- **ONNX Export**: Cross-platform model format
- **Quantization**: FP16 (2x speedup), INT8 (3-4x speedup)
- **FastAPI**: REST API with auto-documentation
- **Docker**: Containerized deployment with monitoring

---

**Last Updated:** November 19, 2025
