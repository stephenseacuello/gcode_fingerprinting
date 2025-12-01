# G-Code Fingerprinting Presentation Outline

**ELE 588 Applied Machine Learning - Final Project**
**Duration:** 15-20 minutes
**Date:** December 2025

---

## Slide 1: Title Slide
**G-Code Fingerprinting: Inferring 3D Printer Commands from Sensor Data**

- Student Name(s)
- Course: ELE 588 Applied Machine Learning
- Date
- University of Rhode Island

---

## Slide 2: Problem Statement & Motivation

**The Challenge:**
- 3D printers execute G-code commands to control motion
- Can we infer the executed commands from sensor data alone?

**Why It Matters:**
- **Security**: Detect malicious command injection
- **Quality Control**: Verify correct execution without G-code access
- **Monitoring**: Real-time command tracking in industrial settings
- **Forensics**: Reconstruct printing history from sensor logs

**Key Insight:** Sensor data (position, temperature, speed) encodes G-code semantics

---

## Slide 3: Dataset Overview

**Data Collection:**
- Source: Multi-sensor 3D printer dataset
- Sensors: Position (X, Y, Z), temperature, speed, acceleration
- **2,368 sequences** across train/val/test splits

**G-Code Vocabulary:**
- **170 unique tokens** (v2 vocabulary with 2-digit bucketing)
- Commands: G0, G1, G2, G3, G53
- Parameters: F (feedrate), X, Y, Z (positions), R (radius)
- Values: Bucketed to 2 digits (e.g., 1575 → 15)

**Preprocessing:**
- 8 continuous features (positions, speeds)
- 18 categorical features (movement types, states)
- Variable-length sequences (padded to max length)

---

## Slide 4: Technical Approach - Architecture

**Two-Stage Pipeline:**

1. **Sensor Encoder** (MM-DTAE-LSTM)
   - Multi-modal processing of continuous + categorical features
   - LSTM backbone for temporal modeling
   - Outputs: Memory representation of sensor sequence

2. **Multi-Head Language Model**
   - Transformer decoder with 4 prediction heads:
     - **Type Head**: Token type (command/parameter/special)
     - **Command Head**: G-code command (G0, G1, etc.)
     - **Param Type Head**: Parameter type (X, Y, Z, F, R)
     - **Param Value Head**: Parameter value (00-99)

**Hierarchical Decomposition:** Each G-code token split into 4 components

---

## Slide 5: Model Architecture Diagram

[Visual diagram showing:]
```
Sensor Data → MM-DTAE-LSTM → Memory → Transformer Decoder → 4 Heads
                                                              ↓
                                            [Type, Command, Param Type, Param Value]
                                                              ↓
                                                        G-Code Token
```

**Key Design Choices:**
- Multi-task learning with weighted losses
- Teacher forcing during training
- Autoregressive generation at inference

---

## Slide 6: Training Strategy

**Data Augmentation (6 techniques):**
- Sensor noise injection
- Temporal shifting
- Magnitude scaling
- Mixup augmentation
- Class-aware oversampling (3x for rare tokens)

**Training Configuration:**
- Optimizer: AdamW with weight decay
- Learning rate: 0.001 with warmup
- Batch size: 8
- Early stopping on validation accuracy
- Multi-head loss with command emphasis

**Challenge:** Imbalanced token distribution (some commands rare)

---

## Slide 7: Current Results - Phase 1

**Performance Metrics (Baseline Model):**

| Metric | Accuracy |
|--------|----------|
| **Command Accuracy** | **100.0%** ✓ |
| Type Gate | 99.8% |
| Parameter Type | 84.3% |
| Parameter Value | 56.2% |
| **Overall Accuracy** | **58.5%** |

**Key Insights:**
- ✅ Perfect command classification (most important)
- ✅ Strong type detection
- ⚠️ Parameter values challenging (high cardinality)
- All heads must be correct for overall accuracy

---

## Slide 8: Error Analysis

**Common Error Patterns:**

1. **Parameter Value Confusion:**
   - Similar bucketed values (14 vs 15)
   - Rare parameter values
   - Fine-grained motion differences

2. **Parameter Type Errors:**
   - X/Y confusion in similar motions
   - Context-dependent parameters

**Error Distribution:**
- 41.5% of predictions have ≥1 head wrong
- Most errors: Single head incorrect (not catastrophic)
- Command head nearly perfect (critical for security)

---

## Slide 9: Hyperparameter Optimization

**Systematic Sweep Strategy:**
- Method: Bayesian optimization with Hyperband early stopping
- Platform: Weights & Biases
- Target: >70% overall accuracy

**Optimizing:**
- Architecture: hidden_dim (96-256), layers (2-4), heads (4-8)
- Training: batch_size, learning_rate, weight_decay
- Loss weighting: command_weight (1.0-5.0)

**Current Status:**
- 🔄 7 runs in progress
- Expected: 50-100 runs over 2-3 days
- Goal: Find optimal configuration

---

## Slide 10: Production Deployment

**Complete MLOps Pipeline:**

1. **Model Export**
   - ONNX format for cross-platform deployment
   - Dynamic batch size support

2. **Quantization**
   - FP16: 50% size reduction, minimal accuracy loss
   - INT8: 75% size reduction, 3-4x speedup

3. **REST API**
   - FastAPI server with 5 endpoints
   - Docker containerized
   - <10ms inference latency

4. **Inference Optimization**
   - ONNX Runtime for efficient serving
   - Batch processing support

---

## Slide 11: Software Engineering

**Production-Grade Implementation:**

- **14,000+ lines of code**
- **88 unit tests** (85% coverage)
- **Automated CI/CD** with pre-commit hooks
- **Complete documentation** (MkDocs site)
- **Docker deployment** with monitoring

**Infrastructure:**
- PyTorch for training
- ONNX for deployment
- FastAPI for serving
- W&B for experiment tracking

---

## Slide 12: Comparison to Related Work

**Novel Contributions:**

1. **Hierarchical Token Decomposition**
   - First to decompose G-code into 4 semantic components
   - Enables interpretable multi-task learning

2. **Production-Ready System**
   - Complete deployment pipeline
   - Real-time inference (<10ms)
   - Quantized models for edge deployment

3. **Systematic Evaluation**
   - Comprehensive error analysis
   - Per-head accuracy metrics
   - Hyperparameter optimization study

**Related Approaches:**
- Traditional: Sequence-to-sequence models (treat as translation)
- Our approach: Hierarchical multi-task with domain structure

---

## Slide 13: Challenges & Lessons Learned

**Technical Challenges:**
1. **Class Imbalance**
   - Solution: Class-aware oversampling + augmentation

2. **Long Sequences**
   - Solution: LSTM encoder + Transformer decoder

3. **High Cardinality Output**
   - Solution: Hierarchical decomposition reduces complexity

**Engineering Challenges:**
1. Parameter naming inconsistencies (W&B vs argparse)
2. Data format evolution (preprocessing versions)
3. Model checkpoint compatibility

**Key Learnings:**
- Multi-task learning crucial for structured outputs
- Production deployment requires significant engineering
- Systematic hyperparameter optimization essential

---

## Slide 14: Future Work

**Immediate Next Steps:**
- Complete hyperparameter sweeps (in progress)
- Train ensemble of best models
- Publish results and deploy production system

**Research Extensions:**
1. **Self-Supervised Pre-training**
   - Learn sensor representations without labels
   - Transfer to other CNC machines

2. **Fingerprinting Validation**
   - Identify specific printer from sensor patterns
   - Forensic applications

3. **Real-Time Anomaly Detection**
   - Detect command injection attacks
   - Quality control applications

4. **Mobile Deployment**
   - TensorFlow Lite for on-device inference
   - Edge computing scenarios

---

## Slide 15: Demo (Optional)

**Live Demo or Video:**
- Show REST API inference
- Input: Sensor data visualization
- Output: Predicted G-code sequence
- Highlight: Real-time performance

**Alternative:** Screenshots of:
- W&B dashboard showing training curves
- API client making predictions
- Docker deployment architecture

---

## Slide 16: Key Takeaways

**Summary:**
1. ✅ Achieved 100% command accuracy (critical metric)
2. ✅ Built production-ready system with complete MLOps
3. ✅ Systematic optimization in progress (>70% target)
4. ✅ Novel hierarchical approach for structured outputs

**Impact:**
- Security monitoring for 3D printing
- Quality control without G-code access
- Framework applicable to other CNC/robotic systems

**Deliverables:**
- Open-source implementation (14K+ lines)
- Complete documentation
- Deployment-ready models

---

## Slide 17: Q&A

**Questions?**

**Key Papers & Resources:**
- Project Repository: [GitHub link]
- Documentation: [MkDocs site]
- W&B Dashboard: [Sweep results]

**Contact:**
- Email: [your email]
- GitHub: [your handle]

---

## Backup Slides (If Time Permits)

### Backup 1: Detailed Architecture
- Model parameter counts
- Layer-by-layer breakdown
- Memory and compute requirements

### Backup 2: Augmentation Ablation
- Performance with/without augmentation
- Individual augmentation technique impact

### Backup 3: Deployment Benchmarks
- Latency vs batch size
- Model size comparisons
- Quantization accuracy/speed tradeoff

### Backup 4: Data Statistics
- Token frequency distribution
- Sequence length histogram
- Train/val/test split details

---

## Presentation Tips

**Timing (20 minutes):**
- Intro & Problem (3 min)
- Technical Approach (5 min)
- Results & Analysis (5 min)
- Deployment & Engineering (3 min)
- Future Work & Conclusions (3 min)
- Q&A (remaining time)

**Key Messages:**
1. Novel problem with real-world applications
2. Strong results (100% command accuracy)
3. Production-ready implementation
4. Systematic optimization approach

**Visual Elements:**
- Architecture diagrams (from docs)
- Training curves (from W&B)
- Error analysis plots
- Demo or screenshots
