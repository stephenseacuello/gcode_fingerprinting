# Fingerprint & Reconstruction Score Visualization

**Date:** 2025-11-30
**Feature:** Added API endpoints for visualizing fingerprint data and reconstruction quality metrics
**Status:** ✅ IMPLEMENTED

---

## Overview

Added two new API endpoints to expose fingerprint vectors and reconstruction quality metrics for visualization and monitoring.

### What is a Fingerprint?

The **fingerprint** is the model's compressed representation of the 232 sensor readings:
- **Sensor Input**: 232 sensor channels (vibrations, force, sound, temperature, etc.)
- **LSTM Encoder**: Processes time-series sensor data
- **Fingerprint Output**: 128-dimensional vector (last hidden state of LSTM)
- **Purpose**: Unique signature that represents the G-code command's physical characteristics

### What is the Reconstruction Score?

The **reconstruction score** measures how well the model captures information from the sensors:
- **Metric**: Standard deviation of memory vectors
- **Higher = Better**: More variance indicates richer representation
- **Lower = Worse**: Less variance suggests information loss or mode collapse
- **Purpose**: Monitor model's ability to encode sensor information effectively

---

## API Endpoints

### 1. `/api/fingerprint/current`

Returns the most recent fingerprint data for visualization.

**Method:** GET

**Response:**
```json
{
  "success": true,
  "fingerprint": [0.42, -0.15, 0.83, ...],  // 128-dimensional vector
  "fingerprint_dimension": 128,
  "recent_scores": [12.5, 12.7, 12.6, ...],  // Last 50 fingerprint scores
  "current_score": 12.6,  // L2 norm of current fingerprint
  "timestamp": "2025-11-30T14:51:30.123456"
}
```

**Usage:**
```bash
curl http://localhost:5001/api/fingerprint/current
```

**Visualization Ideas:**
- **Heatmap**: Display 128-dimensional fingerprint as a color-coded grid
- **Line plot**: Show fingerprint vector values across dimensions
- **Trend plot**: Plot fingerprint_score over time to monitor consistency

---

### 2. `/api/reconstruction_metrics`

Returns comprehensive reconstruction quality metrics and trends.

**Method:** GET

**Response:**
```json
{
  "success": true,
  "reconstruction_score": {
    "current": 2.34,
    "mean": 2.31,
    "std": 0.12,
    "min": 2.10,
    "max": 2.50,
    "history": [2.30, 2.32, 2.34, ...]  // Last 100 values
  },
  "fingerprint_score": {
    "current": 12.6,
    "mean": 12.5,
    "std": 0.3,
    "min": 12.0,
    "max": 13.0,
    "history": [12.4, 12.5, 12.6, ...]  // Last 100 values
  },
  "confidence": {
    "current": 0.85,
    "mean": 0.82,
    "std": 0.05,
    "history": [0.81, 0.83, 0.85, ...]  // Last 100 values
  },
  "anomaly": {
    "current": 0.02,
    "mean": 0.03,
    "std": 0.01,
    "history": [0.03, 0.02, 0.02, ...]  // Last 100 values
  },
  "timestamp": "2025-11-30T14:51:30.123456"
}
```

**Usage:**
```bash
curl http://localhost:5001/api/reconstruction_metrics
```

**Visualization Ideas:**
- **Multi-line plot**: Show reconstruction_score, fingerprint_score, confidence, and anomaly trends together
- **Correlation plot**: Compare reconstruction_score vs. confidence to see if better encoding leads to better predictions
- **Statistics panel**: Display mean, std, min, max for each metric
- **Health indicator**: Use reconstruction_score as a model health metric

---

## Metrics Explained

### Fingerprint Score (L2 Norm)

**Formula:** `sqrt(sum(fingerprint[i]^2 for i in range(128)))`

**What it measures:**
- Magnitude of the fingerprint vector
- How "distinctive" the sensor signature is
- Higher values → more distinctive fingerprint
- Lower values → weaker/less informative sensor pattern

**Typical range:** 10-15 (depends on model architecture)

**Use cases:**
- Detect mode collapse (all fingerprints become similar)
- Monitor sensor input quality
- Identify anomalous machining conditions

---

### Reconstruction Score (Memory Std Dev)

**Formula:** `std(memory_vectors)`

**What it measures:**
- Variance in the LSTM's internal representations
- How much information the model captures from sensors
- Higher values → richer internal representation
- Lower values → information bottleneck or collapse

**Typical range:** 2-3 (depends on model architecture)

**Use cases:**
- Monitor model health
- Detect training degradation
- Identify when model isn't learning from sensors

---

### Confidence

**Formula:** Average probability of predicted G-code tokens

**What it measures:**
- Model's certainty in its predictions
- Higher values → more confident predictions
- Lower values → uncertain/ambiguous predictions

**Typical range:** 0.6-0.9

**Correlation with quality:**
- High confidence + High reconstruction → Good predictions
- High confidence + Low reconstruction → Overconfident (potential issues)
- Low confidence + High reconstruction → Model uncertain but has good info
- Low confidence + Low reconstruction → Poor sensor data or model issues

---

### Anomaly Score

**Formula:** Model's anomaly detection output (placeholder in current implementation)

**What it measures:**
- How unusual the current sensor pattern is
- Higher values → anomalous/unexpected machining conditions
- Lower values → normal operation

**Typical range:** 0-1 (normalized)

---

## Implementation Details

### Tracking

**Location:** [flask_dashboard.py:1664-1679](../flask_dashboard.py#L1664-L1679)

Added tracking for fingerprint_score and reconstruction_score in `state['statistics']`:

```python
# Update running statistics
state['statistics']['anomaly'].append(predictions['anomaly_score'])
state['statistics']['confidence'].append(predictions['gcode_confidence'])
state['statistics']['fingerprint_score'].append(predictions['fingerprint_score'])
state['statistics']['reconstruction_score'].append(predictions['reconstruction_score'])
```

### Calculation

**Location:** [flask_dashboard.py:917-930](../flask_dashboard.py#L917-L930)

Scores are calculated during model inference:

```python
# Extract predictions
fingerprint_vec = outputs['fingerprint'][0].cpu().numpy()  # [128]
memory_vec = outputs['memory'][0].cpu().numpy()  # [64, 128]

# Calculate fingerprint score (L2 norm - higher means more distinctive)
fingerprint_score = float(np.linalg.norm(fingerprint_vec))

# Calculate sensor reconstruction quality score
# Use memory variance as a proxy for information capture
reconstruction_score = float(np.std(memory_vec))

predictions = {
    'fingerprint': fingerprint_vec.tolist(),
    'fingerprint_score': fingerprint_score,
    'reconstruction_score': reconstruction_score,
    ...
}
```

---

## Example Usage

### Test the Endpoints

```bash
# Get current fingerprint
curl http://localhost:5001/api/fingerprint/current | jq '.fingerprint_dimension'

# Get reconstruction metrics
curl http://localhost:5001/api/reconstruction_metrics | jq '.reconstruction_score.current'

# Get full metrics with history
curl http://localhost:5001/api/reconstruction_metrics | jq
```

### Monitor in Real-time

```bash
# Watch reconstruction score trend
watch -n 1 'curl -s http://localhost:5001/api/reconstruction_metrics | jq ".reconstruction_score.current"'

# Watch fingerprint score
watch -n 1 'curl -s http://localhost:5001/api/fingerprint/current | jq ".current_score"'
```

### Python Example

```python
import requests
import matplotlib.pyplot as plt

# Get metrics
response = requests.get('http://localhost:5001/api/reconstruction_metrics')
data = response.json()

# Plot trends
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(data['reconstruction_score']['history'])
plt.title('Reconstruction Score')
plt.ylabel('Std Dev')

plt.subplot(2, 2, 2)
plt.plot(data['fingerprint_score']['history'])
plt.title('Fingerprint Score')
plt.ylabel('L2 Norm')

plt.subplot(2, 2, 3)
plt.plot(data['confidence']['history'])
plt.title('Prediction Confidence')
plt.ylabel('Probability')

plt.subplot(2, 2, 4)
plt.plot(data['anomaly']['history'])
plt.title('Anomaly Score')
plt.ylabel('Score')

plt.tight_layout()
plt.show()
```

---

## Visualization Recommendations

### Dashboard Panels

1. **Fingerprint Heatmap**
   - 128-dimensional vector as 8x16 grid
   - Color scale: blue (negative) → white (zero) → red (positive)
   - Updates in real-time with each prediction

2. **Metrics Trend Plot**
   - 4 subplots: reconstruction_score, fingerprint_score, confidence, anomaly
   - Show last 100 predictions
   - Highlight current value

3. **Quality Dashboard**
   - Current metrics with color-coded health indicators
   - Green: reconstruction_score > 2.0, confidence > 0.7
   - Yellow: moderate values
   - Red: low values (potential issues)

4. **Correlation Scatter Plot**
   - X-axis: reconstruction_score
   - Y-axis: confidence
   - Color: fingerprint_score
   - Shows relationship between sensor encoding quality and prediction confidence

---

## Troubleshooting

### "No fingerprint data available"

**Cause:** No predictions have been made yet
**Solution:** Run at least one prediction first

### "No reconstruction metrics available"

**Cause:** Statistics not populated yet
**Solution:** Run several predictions to build history

### All scores are zero

**Cause:** Model not loaded properly
**Solution:**
1. Check model loading in dashboard
2. Verify LSTM encoder is working
3. Check logs for errors during inference

### Reconstruction score constant/unchanging

**Cause:** Mode collapse or sensor data issues
**Solution:**
1. Check if sensor input is varying
2. Verify different G-code commands give different fingerprints
3. May indicate training issue if always the same

---

## Summary

**Added:**
- ✅ Fingerprint score tracking (L2 norm of 128D vector)
- ✅ Reconstruction score tracking (memory std dev)
- ✅ API endpoint `/api/fingerprint/current` for current fingerprint data
- ✅ API endpoint `/api/reconstruction_metrics` for comprehensive metrics
- ✅ Statistics tracking with rolling window (last 100 values)
- ✅ Running stats (mean, std) for all metrics

**Benefits:**
- Monitor model health in real-time
- Detect mode collapse or degradation
- Visualize sensor encoding quality
- Correlate sensor quality with prediction accuracy
- Debug training and inference issues

**Next Steps:**
- Add frontend visualization components
- Implement health alerts when metrics drop below thresholds
- Add export functionality for metrics history
- Create correlation analysis tools

---

**Author:** Claude Code
**Date:** 2025-11-30
**Version:** 1.0

**Related:**
- [PARAMETER_REPETITION_FIX.md](PARAMETER_REPETITION_FIX.md) - Parameter validation fixes
- [GRAMMAR_CONSTRAINT_FIX.md](GRAMMAR_CONSTRAINT_FIX.md) - Grammar constraint improvements
