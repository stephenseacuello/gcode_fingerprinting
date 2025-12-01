# Production Deployment Guide

This guide covers training, exporting, and deploying production-ready models for the G-code fingerprinting system.

---

## Overview

The production pipeline includes:

1. **Training** - Train final models with optimal hyperparameters
2. **Export** - Convert to ONNX format for cross-platform deployment
3. **Quantization** - Reduce model size with FP16/INT8 quantization
4. **Benchmarking** - Measure inference performance
5. **Deployment** - Package and deploy with Docker

---

## 1. Training Production Models

### Single Model Training

Train a single production model with the best configuration:

```bash
# From hyperparameter sweep results
python scripts/train_production.py \
  --sweep-id your-entity/gcode-fingerprinting/sweep_id \
  --output models/production

# From config file
python scripts/train_production.py \
  --config configs/production_best.json \
  --output models/production \
  --seed 42
```

### Ensemble Training

Train multiple models with different random seeds for ensemble predictions:

```bash
# Train 3-model ensemble
python scripts/train_production.py \
  --config configs/production_best.json \
  --output models/production \
  --ensemble 3 \
  --seed 42
```

This creates:
- `models/production/ensemble_model_1_seed42/`
- `models/production/ensemble_model_2_seed43/`
- `models/production/ensemble_model_3_seed44/`
- `models/production/ensemble_config.json`

### Full Production Pipeline

Train, export, quantize, and package in one command:

```bash
python scripts/train_production.py \
  --config configs/production_best.json \
  --output models/production \
  --export-onnx \
  --quantize \
  --calibration-data data/preprocessed/test \
  --create-package
```

**Outputs:**
- PyTorch checkpoint: `models/production/model_seed42/checkpoint_best.pt`
- ONNX model: `models/production/model_seed42/model.onnx`
- Quantized models: `models/production/model_seed42/quantized/`
- Deployment package: `models/production/deployment_package/`

---

## 2. ONNX Export

Convert PyTorch checkpoints to ONNX format for deployment.

### Basic Export

```bash
# Export full model
python scripts/export_onnx.py \
  --checkpoint outputs/checkpoint_best.pt \
  --output models/production/model.onnx

# Export with dynamic batch size
python scripts/export_onnx.py \
  --checkpoint outputs/checkpoint_best.pt \
  --output models/production/model.onnx \
  --dynamic-batch
```

### Encoder-Only Export

For fingerprinting tasks:

```bash
python scripts/export_onnx.py \
  --checkpoint outputs/checkpoint_best.pt \
  --output models/production/encoder.onnx \
  --encoder-only \
  --dynamic-batch
```

### Export Parameters

- `--opset`: ONNX opset version (default: 13 for wide compatibility)
- `--dynamic-batch`: Enable variable batch size
- `--batch-size`: Fixed batch size (if not dynamic)
- `--seq-len`: Sequence length (default: 64)

### Verification

The export script automatically:
- Validates ONNX model structure
- Compares PyTorch vs ONNX outputs
- Saves metadata (`model.json`)
- Reports model size and parameter count

---

## 3. Model Quantization

Reduce model size and improve inference speed with quantization.

### FP16 Quantization

**Best for:** GPU deployment, minimal accuracy loss (~0.1%)

```bash
python scripts/quantize_model.py \
  --model models/production/model.onnx \
  --output models/production/model_fp16.onnx \
  --method fp16
```

**Benefits:**
- 50% size reduction
- 1.5-2x faster on GPU
- Negligible accuracy loss

### INT8 Dynamic Quantization

**Best for:** CPU deployment, weights-only quantization

```bash
python scripts/quantize_model.py \
  --model models/production/model.onnx \
  --output models/production/model_int8.onnx \
  --method int8
```

**Benefits:**
- 75% size reduction
- 2-3x faster on CPU
- Small accuracy loss (<1%)

### INT8 Static Quantization

**Best for:** Maximum compression, requires calibration data

```bash
python scripts/quantize_model.py \
  --model models/production/model.onnx \
  --output models/production/model_int8_static.onnx \
  --method int8-static \
  --calibration-data data/preprocessed/test
```

**Benefits:**
- 75% size reduction
- 3-4x faster on CPU
- Quantizes weights + activations
- Requires calibration (100+ samples)

### Compare All Methods

```bash
python scripts/quantize_model.py \
  --model models/production/model.onnx \
  --output models/production/model_quantized.onnx \
  --compare \
  --calibration-data data/preprocessed/test
```

**Outputs:**
- `models/production/quantized/model_fp16.onnx`
- `models/production/quantized/model_int8_dynamic.onnx`
- `models/production/quantized/model_int8_static.onnx`
- `models/production/quantized/quantization_results.json`

---

## 4. Inference Benchmarking

Measure latency, throughput, and memory usage.

### Benchmark Single Model

```bash
# PyTorch
python scripts/benchmark_inference.py \
  --checkpoint outputs/checkpoint_best.pt \
  --device cpu \
  --batch-sizes 1 2 4 8 16

# ONNX
python scripts/benchmark_inference.py \
  --onnx models/production/model.onnx \
  --batch-sizes 1 2 4 8 16
```

### Compare All Formats

```bash
python scripts/benchmark_inference.py \
  --checkpoint outputs/checkpoint_best.pt \
  --compare-all \
  --device cpu \
  --output results/benchmark.json
```

**Metrics Reported:**
- **Latency**: mean, median, std, min, max, p95, p99
- **Throughput**: samples per second
- **Memory**: GPU/CPU memory usage
- **Speedup**: relative to baseline

### Example Results

```
Performance Comparison (Batch Size = 1)
================================================================================
Model                | Latency (ms) |  Throughput | Speedup
--------------------------------------------------------------------------------
int8_static          |         3.45 |    289.9/s |   3.62x
int8_dynamic         |         4.12 |    242.7/s |   3.03x
onnx_fp16            |         5.89 |    169.8/s |   2.12x
onnx_fp32            |         8.76 |    114.2/s |   1.43x
pytorch              |        12.50 |     80.0/s |   1.00x
================================================================================
```

---

## 5. Docker Deployment

Deploy the model as a REST API using Docker.

### Build Image

```bash
# Build inference image
docker build -f Dockerfile.inference -t gcode-fingerprinting:latest .
```

### Run Container

```bash
# Run API server
docker run -p 8000:8000 \
  -v $(pwd)/outputs:/app/models:ro \
  gcode-fingerprinting:latest
```

### Docker Compose (Full Stack)

```bash
# Start API + Prometheus + Grafana
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

**Services:**
- **API**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### API Endpoints

- `GET /`: Welcome message
- `GET /health`: Health check
- `GET /info`: Model metadata
- `POST /predict`: Single prediction
- `POST /batch_predict`: Batch prediction
- `POST /fingerprint`: Extract fingerprint

### Example Request

```python
import requests
import numpy as np

# Create sample data
continuous = np.random.randn(64, 135).astype(np.float32)
categorical = np.random.randint(0, 5, size=(64, 4)).astype(np.int64)

# Make prediction
response = requests.post('http://localhost:8000/predict', json={
    'sensor_data': {
        'continuous': continuous.tolist(),
        'categorical': categorical.tolist(),
    },
    'return_fingerprint': True,
    'inference_config': {
        'method': 'greedy',
        'temperature': 1.0,
    }
})

result = response.json()
print(f"G-code: {result['gcode_sequence']}")
print(f"Inference time: {result['inference_time_ms']:.2f} ms")
```

---

## 6. Client Libraries

### Python Client

```python
from examples.api_client import GCodeAPIClient

# Initialize client
client = GCodeAPIClient("http://localhost:8000")

# Check health
health = client.health_check()
print(health)

# Predict
result = client.predict(continuous, categorical, return_fingerprint=True)
print(result['gcode_sequence'])

# Batch prediction
batch_results = client.batch_predict(
    [continuous, continuous],
    [categorical, categorical],
)

# Get fingerprint
fingerprint = client.get_fingerprint(continuous, categorical)
print(f"Fingerprint dim: {fingerprint['embedding_dim']}")
```

See [examples/api_client.py](../examples/api_client.py) for full documentation.

---

## 7. Production Best Practices

### Model Selection

**For GPU deployment:**
- Use FP16 quantized model
- Expect 1.5-2x speedup
- ~0.1% accuracy loss

**For CPU deployment:**
- Use INT8 dynamic quantization
- Expect 2-3x speedup
- <1% accuracy loss

**For edge devices:**
- Use INT8 static quantization
- Calibrate with representative data
- Test accuracy on validation set

### Performance Optimization

1. **Batch Processing**
   - Use batch_predict for multiple samples
   - Optimal batch size: 4-8 on CPU, 16-32 on GPU

2. **Caching**
   - Model loaded once at startup (singleton pattern)
   - Keep session alive for repeated inference

3. **Concurrent Requests**
   - FastAPI handles concurrency with async
   - Use multiple uvicorn workers for CPU-bound tasks

4. **Monitoring**
   - Enable Prometheus metrics
   - Set up Grafana dashboards
   - Monitor latency, throughput, errors

### Deployment Checklist

- [ ] Train final model with optimal hyperparameters
- [ ] Export to ONNX and verify outputs
- [ ] Quantize and benchmark all formats
- [ ] Test on target hardware (CPU/GPU)
- [ ] Validate accuracy on held-out test set
- [ ] Build Docker image
- [ ] Test API endpoints
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Load test with expected traffic
- [ ] Create deployment package
- [ ] Document API usage

---

## 8. Troubleshooting

### ONNX Export Fails

```bash
# Check ONNX version
pip install --upgrade onnx onnxruntime

# Try different opset
python scripts/export_onnx.py --checkpoint model.pt --output model.onnx --opset 11
```

### Quantization Accuracy Drop

- Use more calibration data (500-1000 samples)
- Try FP16 instead of INT8
- Compare per-layer activations
- Check for outlier values in activations

### Slow Inference

- Use quantized model (FP16 or INT8)
- Increase batch size
- Use ONNX Runtime instead of PyTorch
- Enable GPU if available
- Profile with `torch.profiler` or ONNX Runtime profiler

### Docker Build Issues

```bash
# Clear build cache
docker system prune -a

# Build with no cache
docker build --no-cache -f Dockerfile.inference -t gcode-fingerprinting .

# Check logs
docker logs <container_id>
```

### API Errors

```bash
# Check model path
ls -l outputs/checkpoint_best.pt

# Test locally first
python -c "import torch; torch.load('outputs/checkpoint_best.pt')"

# Check API logs
docker-compose logs -f api
```

---

## 9. Performance Targets

Based on benchmarks:

| Metric | CPU | GPU (CUDA) | Apple M1 (MPS) |
|--------|-----|------------|----------------|
| **Latency (ms)** | <10 | <5 | <8 |
| **Throughput** | >100/s | >200/s | >125/s |
| **Memory (MB)** | <500 | <1000 | <800 |
| **Model Size** | <50 MB | <100 MB | <100 MB |

**Production SLA:**
- P95 latency: <20 ms
- P99 latency: <50 ms
- Availability: 99.9%
- Error rate: <0.1%

---

## 10. Next Steps

After deploying to production:

1. **Monitor performance**
   - Track latency, throughput, errors
   - Set up alerts for anomalies
   - A/B test model versions

2. **Continuous improvement**
   - Collect production data for retraining
   - Run periodic hyperparameter sweeps
   - Update model with new data

3. **Scale deployment**
   - Load balancing with multiple replicas
   - Auto-scaling based on traffic
   - Multi-region deployment

4. **Advanced features**
   - Model versioning (MLflow)
   - Feature store integration
   - Online learning updates

---

## Resources

- [ONNX Documentation](https://onnx.ai/onnx/)
- [ONNX Runtime Optimization](https://onnxruntime.ai/docs/performance/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

**Last Updated:** November 19, 2025
**Status:** Production-ready
