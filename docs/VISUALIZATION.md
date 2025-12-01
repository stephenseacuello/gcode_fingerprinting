# Visualization Guide

**Project**: G-code Fingerprinting with Machine Learning
**Last Updated**: 2025-11-20

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Figure Gallery](#figure-gallery)
4. [Advanced Visualizations](#advanced-visualizations)
5. [Using Real Data](#using-real-data)
6. [Customization](#customization)

---

## Overview

The visualization system generates 14 publication-quality figures at 300 DPI, covering:
- Model performance metrics
- Confusion matrices
- Training dynamics
- Statistical confidence intervals
- Token embeddings
- Attention patterns

**Script**: `scripts/generate_visualizations.py`

---

## Quick Start

### Generate All Figures (Mock Data)

```bash
.venv/bin/python scripts/generate_visualizations.py --all --output outputs/figures/
```

**Output**: 14 PNG files in `outputs/figures/`

### Generate Specific Figures

```bash
# Confusion matrices only
.venv/bin/python scripts/generate_visualizations.py --confusion-matrices

# Performance metrics
.venv/bin/python scripts/generate_visualizations.py --performance-metrics

# Training curves
.venv/bin/python scripts/generate_visualizations.py --training-curves

# Bootstrap confidence intervals
.venv/bin/python scripts/generate_visualizations.py --confidence-intervals

# Accuracy distributions
.venv/bin/python scripts/generate_visualizations.py --accuracy-distribution

# Token embedding space (t-SNE)
.venv/bin/python scripts/generate_visualizations.py --embedding-space

# Attention heatmap
.venv/bin/python scripts/generate_visualizations.py --attention-heatmap
```

---

## Figure Gallery

### 1. **confusion_matrix_type.png** - Token Type Classification
**Purpose**: Shows classification performance for token types (Command/Parameter/Special)

**Interpretation**:
- Diagonal = correct predictions
- Off-diagonal = misclassifications
- Target: >99% accuracy (strong diagonal)

**Typical Performance**: 99.8% accuracy

---

### 2. **confusion_matrix_command.png** - G-code Command Classification
**Purpose**: Shows which G-code commands are confused (G0, G1, M104, etc.)

**Interpretation**:
- Perfect diagonal expected for common commands
- Look for systematic confusions (e.g., G0 vs G1)

**Typical Performance**: 100% accuracy

---

### 3. **confusion_matrix_param_type.png** - Parameter Type Classification
**Purpose**: Shows confusion between parameter types (X, Y, Z, E, F, etc.)

**Interpretation**:
- Harder than command classification
- Watch for X/Y confusion, E/F confusion

**Typical Performance**: 84-86% accuracy

---

### 4. **confusion_matrix_param_val.png** - Parameter Value Classification
**Purpose**: Shows confusion between numeric parameter values

**Interpretation**:
- Most challenging head (170 classes)
- Look for systematic numeric confusions
- Adjacent buckets more likely to be confused

**Typical Performance**: 56-60% accuracy

---

### 5. **performance_metrics.png** - Per-Head Accuracy Comparison
**Purpose**: Bar chart comparing accuracies across 4 heads

**Interpretation**:
- Type > Command > Param Type > Param Value (expected hierarchy)
- Shows model is learning hierarchical structure

**Visualization**: Horizontal bars with accuracy labels

---

### 6. **training_curves.png** - Loss Over Time
**Purpose**: Training and validation loss convergence

**Interpretation**:
- Both curves should decrease
- Val loss should track train loss closely
- Divergence = overfitting

**Expected Pattern**: Smooth convergence, val loss 10-20% higher

---

### 7. **unique_tokens.png** - Unique Token Coverage
**Purpose**: Tracks how many unique tokens (out of 170) are predicted

**Interpretation**:
- Starts low (~50-80 tokens at epoch 10)
- Increases over training
- Target: >100 unique tokens by epoch 50

**Indicates**: Model learning rare tokens, not just memorizing common ones

---

### 8. **per_head_accuracy.png** - Accuracy vs Epoch (All Heads)
**Purpose**: Shows training dynamics for each prediction head

**Interpretation**:
- Type & Command: Rapid convergence (epoch 5-10)
- Param Type: Moderate convergence (epoch 20-30)
- Param Value: Slow convergence (continues improving)

**4 Lines**: One per head, color-coded

---

### 9. **token_frequency.png** - Vocabulary Distribution
**Purpose**: Shows class imbalance in G-code tokens

**Interpretation**:
- Long-tail distribution (few tokens very common)
- Rare tokens (<1% frequency) need special handling
- Informs augmentation/oversampling strategy

**Visualization**: Log-scale histogram

---

### 10. **error_heatmap.png** - Error Analysis by Token
**Purpose**: Identifies which tokens are hardest to predict

**Interpretation**:
- Brighter = higher error rate
- Compare across heads (rows)
- Rare tokens typically harder

**Use Case**: Targeted error analysis

---

### 11. **confidence_intervals.png** - Bootstrap Confidence Intervals ⭐ NEW
**Purpose**: Statistical confidence intervals using bootstrap resampling

**Method**:
- 1000 bootstrap iterations
- 95% confidence intervals
- Per-head accuracy estimates

**Interpretation**:
- Error bars show uncertainty
- Narrow bars = high confidence
- Use for reporting statistical significance

**Example**:
```
Token Type:    99.8% ± 0.1%  [99.7%, 99.9%]
Command:      100.0% ± 0.0%  [100.0%, 100.0%]
Param Type:    84.3% ± 1.2%  [82.1%, 86.5%]
Param Value:   56.2% ± 2.8%  [50.6%, 61.8%]
```

---

### 12. **accuracy_distribution.png** - Per-Sample Accuracy Variance ⭐ NEW
**Purpose**: Violin plots showing accuracy distribution across test samples

**Interpretation**:
- Width = density (more samples at that accuracy)
- Median line = typical performance
- Outliers = problematic samples
- Symmetry = consistency

**Use Case**:
- Identify if model is consistent or erratic
- Detect bimodal distributions (some samples easy, some hard)

**Example Pattern**:
- Type: Narrow spike at 99-100% (very consistent)
- Command: Spike at 100% (perfect consistency)
- Param Type: Wider distribution 75-90% (moderate variance)
- Param Value: Very wide distribution 30-80% (high variance)

---

### 13. **embedding_space.png** - Token Embedding t-SNE ⭐ NEW
**Purpose**: 2D visualization of learned 128-dim token embeddings

**Method**:
- t-SNE dimensionality reduction
- Color-coded by token type
- Shows semantic clustering

**Interpretation**:
- Clusters = semantically related tokens
- Commands should cluster separately from parameters
- Similar commands (G0/G1) should be nearby
- Numeric parameters should form gradients

**Colors**:
- Blue: Commands (G0, G1, M104, etc.)
- Orange: Parameter Types (X, Y, Z, E, F)
- Green: Parameter Values (numeric buckets)
- Red: Special tokens (<PAD>, <UNK>, etc.)

**Indicates**: Model has learned meaningful semantic structure

---

### 14. **attention_heatmap.png** - Attention Weights ⭐ NEW
**Purpose**: Visualizes model attention patterns

**Two Subplots**:
1. **Cross-Attention** (Sensor → G-code)
   - Which sensor timesteps attend to which G-code positions
   - Shows how model associates sensor patterns with commands

2. **Self-Attention** (G-code → G-code)
   - Which G-code positions attend to each other
   - Shows sequential dependencies

**Interpretation**:
- Bright diagonal = strong local attention
- Off-diagonal patterns = long-range dependencies
- Uniform = weak/confused attention

**Use Case**: Model interpretability, debugging attention issues

---

## Advanced Visualizations

### Bootstrap Confidence Intervals

**Theory**: Bootstrap resampling estimates uncertainty by:
1. Randomly sampling test set with replacement (n=1000 iterations)
2. Computing accuracy for each sample
3. Computing 95% percentile intervals

**Why Use It**:
- Reports statistically rigorous results
- Quantifies uncertainty
- Required for academic papers

**Example Usage**:
```bash
.venv/bin/python scripts/generate_visualizations.py \
    --confidence-intervals \
    --use-real-data \
    --checkpoint-path outputs/training/checkpoint_best.pt \
    --test-data outputs/processed_v2/test_sequences.npz
```

---

### Accuracy Distribution (Violin Plots)

**What It Shows**: Full distribution of per-sample accuracies

**Components**:
- **Violin Shape**: Kernel density estimate
- **Box Inside**: Quartiles (25th, 50th, 75th percentile)
- **Whiskers**: Min/max values
- **Median Line**: Typical performance

**Interpretation Guide**:
- **Narrow violin**: Consistent performance across samples
- **Wide violin**: High variance, some samples easy/hard
- **Bimodal**: Two distinct difficulty levels
- **Skewed**: Most samples easy OR most samples hard

---

### Token Embedding Space (t-SNE)

**Parameters**:
- `n_components=2`: Reduce to 2D
- `perplexity=30`: Local neighborhood size
- `max_iter=1000`: Optimization iterations
- `random_state=42`: Reproducibility

**Customization**:
```python
# Adjust perplexity for different cluster granularity
tsne = TSNE(n_components=2, perplexity=50, max_iter=1000)

# 3D visualization (requires plotly)
tsne = TSNE(n_components=3, perplexity=30)
```

**Expected Patterns**:
- Commands cluster by function (motion vs temperature control)
- Parameters form semantic gradients
- Special tokens isolated from content tokens

---

### Attention Heatmap

**Architecture**: Multi-head attention (4 heads default)

**Visualization**:
- Aggregates across attention heads (mean)
- Shows first 32 timesteps (for readability)
- Sensor timesteps on Y-axis, G-code positions on X-axis

**Debugging**:
- **Diagonal pattern**: Local attention (expected for sequential data)
- **Vertical stripes**: Specific G-code position attended globally (sentinel tokens)
- **Uniform**: Attention collapse (increase learning rate)

---

## Using Real Data

### Basic Usage

```bash
.venv/bin/python scripts/generate_visualizations.py \
    --all \
    --use-real-data \
    --checkpoint-path outputs/training/checkpoint_best.pt \
    --test-data outputs/processed_v2/test_sequences.npz \
    --vocab-path data/gcode_vocab_v2.json \
    --output outputs/figures/real_results/
```

### Specific Figures with Real Data

```bash
# Only confidence intervals and distributions
.venv/bin/python scripts/generate_visualizations.py \
    --confidence-intervals \
    --accuracy-distribution \
    --use-real-data \
    --checkpoint-path outputs/wandb_sweeps/RUN_ID/checkpoint_best.pt \
    --test-data outputs/processed_v2/test_sequences.npz \
    --output outputs/figures/sweep_results/
```

### Requirements

**Real data visualizations require**:
1. Trained checkpoint (`checkpoint_best.pt`)
2. Test data (`test_sequences.npz`)
3. Vocabulary file (`gcode_vocab_v2.json`)

**Backward Compatibility**:
- Mock data still works by default (no checkpoint needed)
- Real data is opt-in via `--use-real-data` flag

---

## Customization

### Change Output Format

Edit `scripts/generate_visualizations.py`:

```python
# Change DPI
plt.savefig(output_path, dpi=600, bbox_inches='tight')  # Higher quality

# Change format
plt.savefig(output_path.replace('.png', '.pdf'), format='pdf')  # Vector graphics
```

### Modify Color Schemes

```python
# Use different colormap for heatmaps
sns.heatmap(cm, cmap='viridis', annot=True)  # Instead of 'Blues'

# Custom palette for accuracy bars
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
```

### Adjust Figure Sizes

```python
# Larger figures for presentations
fig, ax = plt.subplots(figsize=(14, 10))  # Instead of (10, 8)

# Smaller figures for papers
fig, ax = plt.subplots(figsize=(6, 4))
```

### Font Sizes

```python
# Increase font size for readability
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 12
```

---

## Batch Generation

### Generate Figures for Multiple Checkpoints

```bash
#!/bin/bash
# Script: generate_all_figures.sh

CHECKPOINTS=(
    "outputs/wandb_sweeps/run1/checkpoint_best.pt"
    "outputs/wandb_sweeps/run2/checkpoint_best.pt"
    "outputs/wandb_sweeps/run3/checkpoint_best.pt"
)

for ckpt in "${CHECKPOINTS[@]}"; do
    run_id=$(basename $(dirname $ckpt))
    echo "Generating figures for $run_id..."

    .venv/bin/python scripts/generate_visualizations.py \
        --all \
        --use-real-data \
        --checkpoint-path "$ckpt" \
        --test-data outputs/processed_v2/test_sequences.npz \
        --output "outputs/figures/$run_id/"
done

echo "Done! Figures saved to outputs/figures/"
```

---

## Integration with Papers

### LaTeX Inclusion

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/confidence_intervals.png}
    \caption{Bootstrap confidence intervals (n=1000) for per-head accuracies on test set (474 sequences). Error bars show 95\% confidence intervals.}
    \label{fig:confidence_intervals}
\end{figure}
```

### Reporting Statistics

From `confidence_intervals.png`:
```
Token Type accuracy: 99.8% (95% CI: [99.7%, 99.9%])
G-code Command accuracy: 100.0% (95% CI: [100.0%, 100.0%])
Parameter Type accuracy: 84.3% (95% CI: [82.1%, 86.5%])
Parameter Value accuracy: 56.2% (95% CI: [50.6%, 61.8%])
```

---

## Troubleshooting

### Figure Not Generated

**Check**:
1. Output directory exists: `mkdir -p outputs/figures/`
2. Script has write permissions
3. No errors in terminal output

### Poor Quality / Blurry

**Solution**: Increase DPI
```python
plt.savefig(output_path, dpi=600)  # Publication quality
```

### Memory Error (Large Datasets)

**Solution**: Batch evaluation
```python
# In load_results.py
eval_data = evaluate_model_on_test(model_dict, test_npz_path, max_batches=10)
```

### t-SNE Takes Too Long

**Solution**: Reduce iterations or subsample tokens
```python
tsne = TSNE(n_components=2, max_iter=500)  # Faster
# Or subsample vocabulary:
embeddings_subset = embeddings[:100]  # Only first 100 tokens
```

---

## Performance Benchmarks

### Generation Time

| Figure Type | Time (Mock Data) | Time (Real Data) |
|-------------|------------------|------------------|
| Confusion Matrices | 2 sec | 30 sec (includes evaluation) |
| Performance Metrics | 1 sec | - |
| Training Curves | 1 sec | - |
| Confidence Intervals | 1 sec | 60 sec (bootstrap n=1000) |
| Accuracy Distribution | 1 sec | 5 sec |
| Embedding Space | 2 sec | 10 sec (t-SNE) |
| Attention Heatmap | 1 sec | 5 sec |
| **Total (All 14)** | **15 sec** | **~2 minutes** |

### Disk Space

| Figure Type | File Size (PNG, 300 DPI) |
|-------------|--------------------------|
| Confusion Matrices (4) | 200-400 KB each |
| Line Plots | 100-200 KB each |
| Bar Charts | 100 KB each |
| Heatmaps | 300-500 KB each |
| **Total** | **~4-5 MB** |

---

## Next Steps

- **Training Guide**: [TRAINING.md](TRAINING.md)
- **Pipeline Overview**: [PIPELINE.md](PIPELINE.md)
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **API Reference**: [API.md](API.md)

---

**Questions?** Check the documentation or open an issue.
