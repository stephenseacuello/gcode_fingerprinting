"""
Shared fixtures for pytest test suite.

This module provides reusable fixtures for testing the G-code fingerprinting project.
"""

import pytest
import numpy as np
import torch
import json
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def sample_continuous_data():
    """Generate sample continuous sensor data.

    Returns:
        np.ndarray: Shape (5, 64, 135) - 5 sequences, 64 timesteps, 135 continuous features
    """
    np.random.seed(42)
    return np.random.randn(5, 64, 135).astype(np.float32)


@pytest.fixture
def sample_categorical_data():
    """Generate sample categorical sensor data.

    Returns:
        np.ndarray: Shape (5, 64, 4) - 5 sequences, 64 timesteps, 4 categorical features
    """
    np.random.seed(42)
    return np.random.randint(0, 5, size=(5, 64, 4)).astype(np.int64)


@pytest.fixture
def sample_gcode_sequences():
    """Generate sample G-code sequences.

    Returns:
        list: List of 5 G-code sequences
    """
    sequences = [
        ["G00", "X10", "Y20", "Z05"],
        ["G01", "X15", "Y25", "F10"],
        ["G02", "X20", "Y30", "R05"],
        ["G00", "Z10", "M03", "S10"],
        ["G01", "X05", "Y05", "F20"],
    ]
    return sequences


@pytest.fixture
def sample_gcode_tokens():
    """Generate sample tokenized G-code sequences.

    Returns:
        np.ndarray: Shape (5, 64) - 5 sequences, 64 max tokens
    """
    np.random.seed(42)
    # Assuming vocab size of 170 (v2), PAD token is 0
    tokens = np.random.randint(1, 170, size=(5, 64)).astype(np.int64)
    # Add some padding
    tokens[:, 50:] = 0
    return tokens


@pytest.fixture
def sample_vocabulary():
    """Generate sample vocabulary dictionary.

    Returns:
        dict: Vocabulary with token_to_id and id_to_token mappings
    """
    tokens = [
        "PAD", "UNK", "BOS", "EOS",  # Special tokens
        "G00", "G01", "G02", "G03",  # G-commands
        "M03", "M05",  # M-commands
        "X", "Y", "Z", "F", "R", "S",  # Parameter types
        "NUM_X_00", "NUM_X_10", "NUM_Y_00", "NUM_Y_10",  # Numeric values
    ]

    token_to_id = {token: i for i, token in enumerate(tokens)}
    id_to_token = {i: token for i, token in enumerate(tokens)}

    return {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "vocab_size": len(tokens),
        "special_tokens": ["PAD", "UNK", "BOS", "EOS"],
    }


@pytest.fixture
def small_model_config():
    """Generate configuration for a small test model.

    Returns:
        dict: Model configuration with small parameters for fast testing
    """
    return {
        "d_model": 32,
        "n_heads": 2,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "dropout": 0.1,
        "gcode_vocab": 20,  # Small vocab for testing
        "n_continuous_features": 135,
        "n_categorical_features": 4,
        "categorical_vocab_sizes": [5, 5, 5, 5],
        "max_seq_len": 64,
    }


@pytest.fixture
def training_config():
    """Generate training configuration for tests.

    Returns:
        dict: Training configuration
    """
    return {
        "batch_size": 2,
        "lr": 0.001,
        "epochs": 2,
        "optimizer": "adam",
        "scheduler": "none",
        "gradient_clip": 1.0,
        "device": "cpu",
    }


@pytest.fixture
def augmentation_config():
    """Generate data augmentation configuration.

    Returns:
        dict: Augmentation configuration
    """
    return {
        "oversampling_factor": 3,
        "noise_std": 0.02,
        "noise_prob": 0.5,
        "temporal_shift_max": 2,
        "shift_prob": 0.5,
        "magnitude_scale_range": [0.95, 1.05],
        "scale_prob": 0.5,
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts.

    Yields:
        Path: Path to temporary directory
    """
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_npz_file(temp_dir, sample_continuous_data, sample_categorical_data, sample_gcode_tokens):
    """Create sample .npz file for testing data loading.

    Args:
        temp_dir: Temporary directory fixture
        sample_continuous_data: Continuous sensor data fixture
        sample_categorical_data: Categorical sensor data fixture
        sample_gcode_tokens: G-code tokens fixture

    Returns:
        Path: Path to .npz file
    """
    file_path = temp_dir / "test_data.npz"
    np.savez(
        file_path,
        continuous=sample_continuous_data,
        categorical=sample_categorical_data,
        tokens=sample_gcode_tokens,
        n_sequences=5,
    )
    return file_path


@pytest.fixture
def sample_config_file(temp_dir, small_model_config, training_config):
    """Create sample configuration JSON file.

    Args:
        temp_dir: Temporary directory fixture
        small_model_config: Model configuration fixture
        training_config: Training configuration fixture

    Returns:
        Path: Path to config JSON file
    """
    config = {
        "model_config": small_model_config,
        "training_args": training_config,
    }

    file_path = temp_dir / "test_config.json"
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)

    return file_path


@pytest.fixture
def mock_model(small_model_config):
    """Create a mock model for testing (without loading actual weights).

    Args:
        small_model_config: Model configuration fixture

    Returns:
        torch.nn.Module: Small model instance
    """
    # We'll create this after implementing the actual model tests
    # For now, return None as placeholder
    return None


@pytest.fixture
def device():
    """Get device for testing (CPU to ensure tests run everywhere).

    Returns:
        torch.device: CPU device
    """
    return torch.device("cpu")


@pytest.fixture
def set_random_seeds():
    """Set random seeds for reproducibility in tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture(autouse=True)
def reset_random_seeds(set_random_seeds):
    """Automatically reset random seeds before each test.

    Args:
        set_random_seeds: Random seed setting fixture
    """
    pass


@pytest.fixture
def sample_loss_weights():
    """Generate sample loss weights for multi-head model.

    Returns:
        dict: Loss weights for each head
    """
    return {
        "type_gate_weight": 1.0,
        "command_weight": 3.0,
        "param_type_weight": 1.0,
        "param_value_weight": 1.0,
        "reconstruction_weight": 0.5,
        "fingerprint_weight": 0.5,
    }


@pytest.fixture
def sample_batch(sample_continuous_data, sample_categorical_data, sample_gcode_tokens):
    """Create a sample batch for testing.

    Returns:
        dict: Dictionary containing batch data
    """
    return {
        "continuous": torch.from_numpy(sample_continuous_data),
        "categorical": torch.from_numpy(sample_categorical_data),
        "tokens": torch.from_numpy(sample_gcode_tokens),
        "batch_size": 5,
        "seq_len": 64,
    }


# Performance measurement fixtures
@pytest.fixture
def benchmark_timer():
    """Simple timer for benchmarking test performance.

    Yields:
        callable: Function that returns elapsed time in seconds
    """
    import time
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    yield elapsed
