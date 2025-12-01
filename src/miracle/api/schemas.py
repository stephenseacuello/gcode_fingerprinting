"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class GenerationMethod(str, Enum):
    """Generation methods for inference."""
    greedy = "greedy"
    beam_search = "beam_search"
    temperature = "temperature"
    top_k = "top_k"
    nucleus = "nucleus"


class InferenceConfig(BaseModel):
    """Configuration for inference-time parameters."""

    method: GenerationMethod = Field(
        default=GenerationMethod.greedy,
        description="Generation method"
    )

    # Beam search
    beam_width: int = Field(default=5, ge=1, le=20, description="Beam width for beam search")
    length_penalty: float = Field(default=0.6, ge=0.0, le=2.0, description="Length normalization penalty")

    # Temperature sampling
    temperature: float = Field(default=1.0, ge=0.1, le=3.0, description="Temperature for sampling")

    # Top-k sampling
    top_k: int = Field(default=50, ge=1, le=200, description="Top-k for sampling")

    # Nucleus sampling
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus (top-p) probability")

    # Repetition penalty
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0, description="Repetition penalty")

    # Multi-head temperatures (optional)
    type_temp: Optional[float] = Field(default=None, ge=0.1, le=3.0, description="Temperature for type gate")
    command_temp: Optional[float] = Field(default=None, ge=0.1, le=3.0, description="Temperature for commands")
    param_type_temp: Optional[float] = Field(default=None, ge=0.1, le=3.0, description="Temperature for param types")
    param_value_temp: Optional[float] = Field(default=None, ge=0.1, le=3.0, description="Temperature for param values")

    # Max length
    max_length: int = Field(default=64, ge=1, le=256, description="Maximum sequence length")


class SensorData(BaseModel):
    """Sensor data input."""

    continuous: List[List[float]] = Field(
        ...,
        description="Continuous sensor data [T, 135]",
        min_length=1
    )
    categorical: List[List[int]] = Field(
        ...,
        description="Categorical features [T, 4]",
        min_length=1
    )

    class Config:
        json_schema_extra = {
            "example": {
                "continuous": [[0.1] * 135] * 64,
                "categorical": [[0, 1, 2, 3]] * 64
            }
        }


class PredictionRequest(BaseModel):
    """Request for G-code prediction."""

    sensor_data: SensorData
    inference_config: Optional[InferenceConfig] = None
    return_fingerprint: bool = Field(default=False, description="Include machine fingerprint in response")
    return_confidence: bool = Field(default=False, description="Include confidence scores")
    return_alternatives: bool = Field(default=False, description="Include top-k alternatives")
    num_alternatives: int = Field(default=3, ge=1, le=10, description="Number of alternatives to return")


class TokenPrediction(BaseModel):
    """Single token prediction."""

    token: str
    confidence: float = Field(ge=0.0, le=1.0)
    token_type: str  # COMMAND, PARAMETER, NUMERIC, SPECIAL


class PredictionResponse(BaseModel):
    """Response from G-code prediction."""

    gcode_sequence: List[str]
    tokens: Optional[List[TokenPrediction]] = None
    fingerprint: Optional[List[float]] = None
    inference_time_ms: float
    model_version: str

    class Config:
        json_schema_extra = {
            "example": {
                "gcode_sequence": ["G0", "X10", "Y20", "Z05"],
                "inference_time_ms": 45.2,
                "model_version": "multihead_aug_v2"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch prediction."""

    sensor_data_batch: List[SensorData]
    inference_config: Optional[InferenceConfig] = None
    return_fingerprint: bool = False


class BatchPredictionResponse(BaseModel):
    """Response from batch prediction."""

    predictions: List[PredictionResponse]
    total_inference_time_ms: float


class FingerprintRequest(BaseModel):
    """Request for fingerprint extraction."""

    sensor_data: SensorData


class FingerprintResponse(BaseModel):
    """Response from fingerprint extraction."""

    fingerprint: List[float]
    embedding_dim: int
    norm: float  # Should be ~1.0 for normalized embeddings

    class Config:
        json_schema_extra = {
            "example": {
                "fingerprint": [0.1] * 128,
                "embedding_dim": 128,
                "norm": 1.0
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class InfoResponse(BaseModel):
    """Model information response."""

    model_name: str
    model_version: str
    vocab_size: int
    d_model: int
    num_parameters: int
    supported_endpoints: List[str]
    supported_generation_methods: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "MM-DTAE-LSTM-MultiHead",
                "model_version": "multihead_aug_v2",
                "vocab_size": 170,
                "d_model": 128,
                "num_parameters": 2100000,
                "supported_endpoints": ["/predict", "/batch_predict", "/fingerprint", "/health", "/info"],
                "supported_generation_methods": ["greedy", "beam_search", "temperature", "top_k", "nucleus"]
            }
        }


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


class LoadCheckpointRequest(BaseModel):
    """Request to load a new checkpoint."""

    checkpoint_path: str = Field(..., description="Path to checkpoint file")
    vocab_path: Optional[str] = Field(default=None, description="Path to vocabulary file (optional)")
    device: Optional[str] = Field(default=None, description="Device to load model on (cpu, cuda, mps)")

    class Config:
        json_schema_extra = {
            "example": {
                "checkpoint_path": "outputs/best_from_sweep/checkpoint_best.pt",
                "vocab_path": "data/vocabulary.json",
                "device": "cpu"
            }
        }


class LoadCheckpointResponse(BaseModel):
    """Response from loading checkpoint."""

    status: str
    checkpoint_path: str
    model_version: str
    vocab_size: int
    d_model: int
    load_time_ms: float

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "checkpoint_path": "outputs/best_from_sweep/checkpoint_best.pt",
                "model_version": "multihead_sweep_v1",
                "vocab_size": 170,
                "d_model": 128,
                "load_time_ms": 542.3
            }
        }
