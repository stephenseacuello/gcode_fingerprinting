# API Reference

**Project**: G-code Fingerprinting with Machine Learning
**Last Updated**: 2025-11-20

---

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Endpoints](#endpoints)
4. [Python Client](#python-client)
5. [cURL Examples](#curl-examples)
6. [Error Handling](#error-handling)
7. [Deployment](#deployment)

---

## Overview

The G-code Fingerprinting API provides RESTful endpoints for:
- **Inference**: Predict G-code sequences from sensor data
- **Batch Processing**: Process multiple samples efficiently
- **Fingerprinting**: Extract machine-specific embeddings
- **Model Metadata**: Query model configuration and status

**Technology Stack**:
- FastAPI (async Python web framework)
- PyTorch (model inference)
- Uvicorn (ASGI server)

**Base URL**: `http://localhost:8000` (default)

---

## Getting Started

### Start the Server

```bash
# Basic startup
PYTHONPATH=src .venv/bin/python scripts/api_server.py \
    --checkpoint outputs/training/checkpoint_best.pt \
    --vocab-path data/gcode_vocab_v2.json

# Custom host/port
PYTHONPATH=src .venv/bin/python scripts/api_server.py \
    --checkpoint outputs/training/checkpoint_best.pt \
    --vocab-path data/gcode_vocab_v2.json \
    --host 0.0.0.0 \
    --port 8080

# With GPU/MPS
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/api_server.py \
    --checkpoint outputs/training/checkpoint_best.pt \
    --vocab-path data/gcode_vocab_v2.json \
    --device mps
```

### Check Server Status

```bash
curl http://localhost:8000/health
# Output: {"status": "healthy", "model_loaded": true}
```

### Interactive API Docs

Once server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Description**: Check if API is running and model is loaded

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Example**:
```bash
curl http://localhost:8000/health
```

---

### 2. Model Information

**Endpoint**: `GET /info`

**Description**: Get model metadata and configuration

**Response**:
```json
{
  "model_type": "MultiHeadGCodeLM",
  "vocab_size": 170,
  "d_model": 128,
  "nhead": 4,
  "num_layers": 4,
  "embedding_dim": 128,
  "checkpoint_path": "outputs/training/checkpoint_best.pt",
  "device": "mps"
}
```

**Example**:
```bash
curl http://localhost:8000/info
```

---

### 3. Single Prediction

**Endpoint**: `POST /predict`

**Description**: Predict G-code sequence from sensor data

**Request Body**:
```json
{
  "sensor_data": {
    "continuous": [[1.2, 0.8, ...], ...],  // Shape: [T, 135]
    "categorical": [[0, 1, 2, 0], ...]     // Shape: [T, 4]
  },
  "return_fingerprint": false,              // Optional, default: false
  "inference_config": {                     // Optional
    "method": "greedy",                     // "greedy" or "beam_search"
    "temperature": 1.0,                     // Sampling temperature
    "beam_width": 5                         // For beam_search only
  }
}
```

**Response**:
```json
{
  "gcode_sequence": ["G0", "X120", "Y85", "G1", "E5"],
  "predictions": {
    "type": [1, 2, 2, 1, 2],
    "command": [45, 0, 0, 46, 0],
    "param_type": [0, 12, 13, 0, 8],
    "param_val": [0, 120, 85, 0, 5]
  },
  "confidences": {
    "type": [0.998, 0.995, 0.997, 0.999, 0.996],
    "command": [1.0, 0.0, 0.0, 1.0, 0.0],
    "param_type": [0.0, 0.89, 0.91, 0.0, 0.88],
    "param_val": [0.0, 0.65, 0.71, 0.0, 0.58]
  },
  "fingerprint": [0.12, -0.34, ...],        // If return_fingerprint=true
  "inference_time_ms": 15.3
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_data": {
      "continuous": [[1.2, 0.8, 1.1, 0.9, 25.3, ...], ...],
      "categorical": [[0, 1, 2, 0], ...]
    },
    "return_fingerprint": true
  }'
```

---

### 4. Batch Prediction

**Endpoint**: `POST /batch_predict`

**Description**: Process multiple sensor sequences in one request

**Request Body**:
```json
{
  "sensor_data_batch": [
    {
      "continuous": [[...], ...],
      "categorical": [[...], ...]
    },
    {
      "continuous": [[...], ...],
      "categorical": [[...], ...]
    }
  ],
  "return_fingerprint": false
}
```

**Response**:
```json
{
  "predictions": [
    {
      "gcode_sequence": ["G0", "X120", ...],
      "predictions": {...},
      "confidences": {...},
      "inference_time_ms": 15.3
    },
    {
      "gcode_sequence": ["G1", "Y85", ...],
      "predictions": {...},
      "confidences": {...},
      "inference_time_ms": 14.8
    }
  ],
  "total_inference_time_ms": 30.1,
  "batch_size": 2
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d @batch_payload.json
```

---

### 5. Fingerprint Extraction

**Endpoint**: `POST /fingerprint`

**Description**: Extract machine-specific embedding from sensor data

**Request Body**:
```json
{
  "sensor_data": {
    "continuous": [[...], ...],
    "categorical": [[...], ...]
  }
}
```

**Response**:
```json
{
  "fingerprint": [0.12, -0.34, 0.56, ...],  // Length: 128 (d_model)
  "embedding_dim": 128,
  "norm": 12.45,
  "extraction_time_ms": 8.2
}
```

**Use Cases**:
- Machine authentication
- Manufacturer identification
- Anomaly detection
- Clustering similar printers

**Example**:
```bash
curl -X POST http://localhost:8000/fingerprint \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_data": {
      "continuous": [[1.2, 0.8, ...], ...],
      "categorical": [[0, 1, 2, 0], ...]
    }
  }'
```

---

## Python Client

### Installation

The Python client is located at `examples/api_client.py`.

```bash
# No installation needed, uses requests library
pip install requests numpy
```

### Basic Usage

```python
from examples.api_client import GCodeAPIClient
import numpy as np

# Initialize client
client = GCodeAPIClient("http://localhost:8000")

# Check health
health = client.health_check()
print(health)  # {'status': 'healthy', 'model_loaded': True}

# Get model info
info = client.get_info()
print(f"Model: {info['model_type']}, Vocab: {info['vocab_size']}")

# Load sensor data
continuous = np.random.randn(64, 135).astype(np.float32)
categorical = np.random.randint(0, 5, size=(64, 4)).astype(np.int64)

# Single prediction
result = client.predict(continuous, categorical, return_fingerprint=True)
print("G-code:", result['gcode_sequence'])
print("Inference time:", f"{result['inference_time_ms']:.2f}ms")
```

### Batch Processing

```python
# Prepare batch
continuous_batch = [np.random.randn(64, 135).astype(np.float32) for _ in range(10)]
categorical_batch = [np.random.randint(0, 5, size=(64, 4)).astype(np.int64) for _ in range(10)]

# Batch prediction
results = client.batch_predict(continuous_batch, categorical_batch)

print(f"Processed {len(results['predictions'])} samples")
print(f"Total time: {results['total_inference_time_ms']:.2f}ms")
print(f"Avg time per sample: {results['total_inference_time_ms'] / len(results['predictions']):.2f}ms")

for i, pred in enumerate(results['predictions']):
    print(f"Sample {i}: {' '.join(pred['gcode_sequence'][:5])}")
```

### Fingerprint Extraction

```python
# Extract fingerprint
fp_result = client.get_fingerprint(continuous, categorical)

print(f"Fingerprint dimension: {fp_result['embedding_dim']}")
print(f"Norm: {fp_result['norm']:.4f}")

# Use for clustering
fingerprint = np.array(fp_result['fingerprint'])
# Apply cosine similarity, UMAP, etc.
```

### Advanced Configuration

```python
# Beam search decoding
result = client.predict(
    continuous,
    categorical,
    method="beam_search",
    temperature=0.8
)

# Custom temperature for diversity
result = client.predict(
    continuous,
    categorical,
    method="greedy",
    temperature=1.2  # Higher = more diverse
)
```

---

## cURL Examples

### Single Prediction (From File)

**payload.json**:
```json
{
  "sensor_data": {
    "continuous": [[1.2, 0.8, 1.1, 0.9, 25.3, 26.1, 24.8, 25.5, ...]],
    "categorical": [[0, 1, 2, 0]]
  }
}
```

**Request**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @payload.json | jq
```

### Extract Pretty JSON

```bash
curl -s http://localhost:8000/info | jq '.'
```

### Measure Latency

```bash
time curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @payload.json > /dev/null
```

### Health Check Loop

```bash
# Monitor server health every 5 seconds
watch -n 5 "curl -s http://localhost:8000/health | jq"
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid input shape or format |
| 422 | Validation Error | Missing required fields |
| 500 | Internal Server Error | Model inference failed |
| 503 | Service Unavailable | Model not loaded |

### Error Response Format

```json
{
  "detail": "Invalid input shape: expected (T, 135), got (T, 8)",
  "error_type": "ValidationError",
  "timestamp": "2025-11-20T10:30:45Z"
}
```

### Common Errors

#### 1. Invalid Input Shape

**Error**:
```json
{
  "detail": "continuous sensor data must have shape [T, 135], got [T, 8]"
}
```

**Solution**: Check input dimensions match expected format

#### 2. Model Not Loaded

**Error**:
```json
{
  "detail": "Model not loaded. Check server logs.",
  "error_type": "ModelNotLoadedError"
}
```

**Solution**: Restart server with valid checkpoint path

#### 3. CUDA/MPS Out of Memory

**Error**:
```json
{
  "detail": "MPS backend out of memory during inference"
}
```

**Solution**: Reduce batch size or use CPU inference

---

## Deployment

### Production Deployment

#### Using Gunicorn (Multiple Workers)

```bash
# Install gunicorn
pip install gunicorn

# Run with 4 workers
PYTHONPATH=src gunicorn scripts.api_server:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120
```

#### Using Docker

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY data/gcode_vocab_v2.json data/
COPY outputs/training/checkpoint_best.pt models/

# Expose port
EXPOSE 8000

# Set environment
ENV PYTHONPATH=/app/src

# Run server
CMD ["uvicorn", "scripts.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build & Run**:
```bash
# Build image
docker build -t gcode-api:latest .

# Run container
docker run -p 8000:8000 gcode-api:latest

# With GPU support
docker run --gpus all -p 8000:8000 gcode-api:latest
```

#### Using Kubernetes

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gcode-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gcode-api
  template:
    metadata:
      labels:
        app: gcode-api
    spec:
      containers:
      - name: api
        image: gcode-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: gcode-api-service
spec:
  selector:
    app: gcode-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Performance Tuning

#### Enable Caching

```python
# In api_server.py
from functools import lru_cache

@lru_cache(maxsize=128)
def get_prediction(sensor_hash):
    # Cache predictions for repeated inputs
    pass
```

#### Async Batch Processing

```python
# Process batches asynchronously
@app.post("/batch_predict")
async def batch_predict(batch: BatchRequest):
    tasks = [process_sample(sample) for sample in batch.sensor_data_batch]
    results = await asyncio.gather(*tasks)
    return {"predictions": results}
```

#### Model Quantization

```bash
# Quantize to INT8 for faster inference
PYTHONPATH=src .venv/bin/python scripts/quantize_model.py \
    --checkpoint outputs/training/checkpoint_best.pt \
    --output models/checkpoint_int8.pt

# Start server with quantized model
PYTHONPATH=src .venv/bin/python scripts/api_server.py \
    --checkpoint models/checkpoint_int8.pt \
    --vocab-path data/gcode_vocab_v2.json
```

### Monitoring

#### Prometheus Metrics

```python
# Add prometheus_client
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter('api_requests_total', 'Total API requests')
inference_time = Histogram('inference_time_seconds', 'Inference latency')

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

#### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("gcode-api")
logger.info(f"Prediction completed in {elapsed_ms:.2f}ms")
```

---

## Security

### API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != os.environ.get("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/predict", dependencies=[Security(verify_api_key)])
def predict(request: PredictRequest):
    # Protected endpoint
    pass
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
def predict(request: Request, data: PredictRequest):
    # Max 10 requests per minute per IP
    pass
```

---

## Performance Benchmarks

### Latency (Single Prediction)

| Device | Batch Size | Latency (ms) | Throughput (req/s) |
|--------|------------|--------------|---------------------|
| Mac M1 | 1 | 15 | 67 |
| Mac M2 | 1 | 12 | 83 |
| RTX 3090 | 1 | 8 | 125 |
| CPU (8 cores) | 1 | 45 | 22 |

### Batch Processing

| Batch Size | Latency (ms) | Throughput (samples/s) |
|------------|--------------|------------------------|
| 1 | 15 | 67 |
| 8 | 60 | 133 |
| 16 | 100 | 160 |
| 32 | 180 | 178 |

### Memory Usage

| Configuration | Memory (MB) |
|---------------|-------------|
| Model only | 150 |
| + 1 sample | 180 |
| + Batch of 16 | 350 |
| + Batch of 64 | 800 |

---

## Next Steps

- **Training Guide**: [TRAINING.md](TRAINING.md)
- **Pipeline Overview**: [PIPELINE.md](PIPELINE.md)
- **Visualization**: [VISUALIZATION.md](VISUALIZATION.md)
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)

---

**Questions?** Check the documentation or open an issue.
