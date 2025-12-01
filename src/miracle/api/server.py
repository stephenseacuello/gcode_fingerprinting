"""
FastAPI server for G-code fingerprinting model inference.

Provides REST API endpoints for model predictions and fingerprint extraction.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import numpy as np
from typing import List
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from miracle.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    FingerprintRequest,
    FingerprintResponse,
    HealthResponse,
    InfoResponse,
    ErrorResponse,
    GenerationMethod,
    LoadCheckpointRequest,
    LoadCheckpointResponse,
)
from miracle.api.model_manager import ModelManager

# Create FastAPI app
app = FastAPI(
    title="G-Code Fingerprinting API",
    description="CNC Machine Fingerprinting & G-code Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model_manager = ModelManager()
start_time = time.time()


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    print("Starting G-Code Fingerprinting API...")

    # Try to load default model
    default_checkpoint = "outputs/training_50epoch/checkpoint_best.pt"
    default_vocab = "data/vocabulary.json"

    if Path(default_checkpoint).exists():
        try:
            # Force CPU device for stable inference (MPS has compatibility issues)
            model_manager.load_model(default_checkpoint, vocab_path=default_vocab, device='cpu')
            print(f"✓ Model loaded from {default_checkpoint}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print("  API will run without model (health check only)")
    else:
        print(f"✗ Checkpoint not found: {default_checkpoint}")
        print("  API will run without model (health check only)")


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "G-Code Fingerprinting API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "info": "/info",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "fingerprint": "/fingerprint",
        }
    }


# Health check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns API status and model loading state.
    """
    uptime = time.time() - start_time

    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        model_version=model_manager._model_version or "none",
        uptime_seconds=uptime
    )


# Model info
@app.get("/info", response_model=InfoResponse)
async def get_info():
    """
    Get model information and capabilities.

    Returns model metadata, configuration, and supported operations.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model_info = model_manager.get_model_info()

    return InfoResponse(
        model_name="MM-DTAE-LSTM-MultiHead",
        model_version=model_info.get('model_version', 'unknown'),
        vocab_size=model_info.get('config', {}).get('vocab_size', 0),
        d_model=model_info.get('config', {}).get('d_model', 0),
        num_parameters=model_info.get('num_parameters', 0),
        supported_endpoints=["/predict", "/batch_predict", "/fingerprint", "/health", "/info"],
        supported_generation_methods=[m.value for m in GenerationMethod]
    )


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict G-code sequence from sensor data.

    Runs model inference on provided sensor readings and returns predicted G-code.
    Optionally includes machine fingerprint and confidence scores.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to numpy arrays
        sensor_data = {
            'continuous': np.array(request.sensor_data.continuous, dtype=np.float32),
            'categorical': np.array(request.sensor_data.categorical, dtype=np.int64),
        }

        # Validate shapes
        if sensor_data['continuous'].ndim != 2:
            raise ValueError(f"Expected continuous data shape [T, 135], got {sensor_data['continuous'].shape}")
        if sensor_data['categorical'].ndim != 2:
            raise ValueError(f"Expected categorical data shape [T, 4], got {sensor_data['categorical'].shape}")

        # Get inference config
        inference_config = request.inference_config or {}
        config_dict = inference_config.dict() if inference_config else {}

        # Run prediction
        result = model_manager.predict(
            sensor_data,
            method=config_dict.get('method', 'greedy'),
            temperature=config_dict.get('temperature', 1.0),
            max_length=config_dict.get('max_length', 64),
        )

        # Optionally add fingerprint
        fingerprint = None
        if request.return_fingerprint:
            fp = model_manager.get_fingerprint(sensor_data)
            fingerprint = fp.flatten().tolist()

        return PredictionResponse(
            gcode_sequence=result['gcode_sequence'],
            fingerprint=fingerprint,
            inference_time_ms=result['inference_time_ms'],
            model_version=result['model_version'],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Batch prediction endpoint
@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict G-code sequences for multiple sensor data samples.

    Processes multiple samples in batch for efficiency.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.sensor_data_batch) == 0:
        raise HTTPException(status_code=400, detail="Empty batch")

    if len(request.sensor_data_batch) > 32:
        raise HTTPException(status_code=400, detail="Batch size too large (max 32)")

    start_time_batch = time.time()
    predictions = []

    try:
        for sensor_data_item in request.sensor_data_batch:
            sensor_data = {
                'continuous': np.array(sensor_data_item.continuous, dtype=np.float32),
                'categorical': np.array(sensor_data_item.categorical, dtype=np.int64),
            }

            # Get inference config
            inference_config = request.inference_config or {}
            config_dict = inference_config.dict() if inference_config else {}

            # Run prediction
            result = model_manager.predict(
                sensor_data,
                method=config_dict.get('method', 'greedy'),
                temperature=config_dict.get('temperature', 1.0),
                max_length=config_dict.get('max_length', 64),
            )

            # Add fingerprint if requested
            fingerprint = None
            if request.return_fingerprint:
                fp = model_manager.get_fingerprint(sensor_data)
                fingerprint = fp.flatten().tolist()

            predictions.append(PredictionResponse(
                gcode_sequence=result['gcode_sequence'],
                fingerprint=fingerprint,
                inference_time_ms=result['inference_time_ms'],
                model_version=result['model_version'],
            ))

        total_time = (time.time() - start_time_batch) * 1000

        return BatchPredictionResponse(
            predictions=predictions,
            total_inference_time_ms=total_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# Fingerprint endpoint
@app.post("/fingerprint", response_model=FingerprintResponse)
async def get_fingerprint(request: FingerprintRequest):
    """
    Extract machine fingerprint embedding from sensor data.

    Returns a normalized embedding vector that represents unique machine characteristics.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        sensor_data = {
            'continuous': np.array(request.sensor_data.continuous, dtype=np.float32),
            'categorical': np.array(request.sensor_data.categorical, dtype=np.int64),
        }

        # Extract fingerprint
        fp = model_manager.get_fingerprint(sensor_data)
        fp_flat = fp.flatten()

        # Compute norm
        norm = float(np.linalg.norm(fp_flat))

        return FingerprintResponse(
            fingerprint=fp_flat.tolist(),
            embedding_dim=len(fp_flat),
            norm=norm
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fingerprint extraction failed: {str(e)}")


# Load checkpoint endpoint
@app.post("/load_checkpoint", response_model=LoadCheckpointResponse)
async def load_checkpoint(request: LoadCheckpointRequest):
    """
    Load a new checkpoint without restarting the server.

    Dynamically loads a different model checkpoint and vocabulary.
    Useful for testing different models or deploying new checkpoints.
    """
    try:
        checkpoint_path = Path(request.checkpoint_path)

        # Validate checkpoint exists
        if not checkpoint_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint not found: {request.checkpoint_path}"
            )

        # Use default vocab if not specified
        vocab_path = request.vocab_path or "data/vocabulary.json"
        if not Path(vocab_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Vocabulary file not found: {vocab_path}"
            )

        # Load model
        start_load = time.time()
        model_manager.load_model(
            str(checkpoint_path),
            vocab_path=vocab_path,
            device=request.device
        )
        load_time = (time.time() - start_load) * 1000

        # Get model info
        model_info = model_manager.get_model_info()

        return LoadCheckpointResponse(
            status="success",
            checkpoint_path=str(checkpoint_path),
            model_version=model_info.get('model_version', 'unknown'),
            vocab_size=model_info.get('config', {}).get('vocab_size', 0),
            d_model=model_info.get('config', {}).get('d_model', 0),
            load_time_ms=load_time
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load checkpoint: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
