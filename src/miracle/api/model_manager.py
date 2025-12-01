"""
Model manager for loading and running inference.

Implements singleton pattern to load model once and cache in memory.
"""

import torch
import numpy as np
import time
from pathlib import Path
from typing import Optional, Dict, List
import json

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.multihead_lm import MultiHeadGCodeLM
from miracle.dataset.target_utils import TokenDecomposer


class ModelManager:
    """Singleton model manager for inference."""

    _instance = None
    _backbone = None
    _multihead_lm = None
    _decomposer = None
    _config = None
    _vocab_path = None
    _device = None
    _model_version = None
    _load_time = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, checkpoint_path: str, vocab_path: str = 'data/vocabulary.json', device: Optional[str] = None):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            vocab_path: Path to vocabulary JSON file
            device: Device to load model on ('cpu', 'cuda', 'mps', or None for auto)
        """
        if self._backbone is not None:
            print("Model already loaded, skipping...")
            return

        print(f"Loading model from {checkpoint_path}...")
        start_time = time.time()

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self._device = torch.device('mps')
            else:
                self._device = torch.device('cpu')
        else:
            self._device = torch.device(device)

        print(f"Using device: {self._device}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        self._config = checkpoint.get('config', {})
        self._model_version = Path(checkpoint_path).parent.name
        self._vocab_path = vocab_path

        # Create decomposer
        self._decomposer = TokenDecomposer(vocab_path)

        # Infer sensor dimensions from checkpoint
        sensor_dims = None
        if 'backbone_state_dict' in checkpoint:
            backbone_state = checkpoint['backbone_state_dict']

            # Try to infer dimensions from encoder weights
            if 'encoders.0.proj.0.weight' in backbone_state:
                n_continuous = backbone_state['encoders.0.proj.0.weight'].shape[1]
            else:
                n_continuous = 155  # Default fallback

            if 'encoders.1.proj.0.weight' in backbone_state:
                n_categorical = backbone_state['encoders.1.proj.0.weight'].shape[1]
            else:
                n_categorical = 4  # Default fallback

            sensor_dims = [n_continuous, n_categorical]
        else:
            sensor_dims = [155, 4]  # Default dimensions

        # Get vocabulary size from decomposer or default
        vocab_size = len(self._decomposer.vocab) if hasattr(self._decomposer, 'vocab') else 170

        # Create backbone (MM_DTAE_LSTM)
        backbone_config = ModelConfig(
            sensor_dims=sensor_dims,
            d_model=self._config.get('hidden_dim', 128),
            lstm_layers=self._config.get('num_layers', 2),
            gcode_vocab=vocab_size,
            n_heads=self._config.get('num_heads', 4),
        )
        self._backbone = MM_DTAE_LSTM(backbone_config).to(self._device)

        # Load backbone state
        if 'backbone_state_dict' in checkpoint:
            self._backbone.load_state_dict(checkpoint['backbone_state_dict'])
        else:
            raise KeyError("Checkpoint missing backbone_state_dict")

        # Create multihead LM
        self._multihead_lm = MultiHeadGCodeLM(
            d_model=self._config.get('hidden_dim', 128),
            n_commands=self._decomposer.n_commands,
            n_param_types=self._decomposer.n_param_types,
            n_param_values=self._decomposer.n_param_values,
            nhead=self._config.get('num_heads', 4),
            num_layers=self._config.get('num_layers', 2),
            vocab_size=vocab_size,
        ).to(self._device)

        # Load multihead state
        if 'multihead_state_dict' in checkpoint:
            self._multihead_lm.load_state_dict(checkpoint['multihead_state_dict'])
        else:
            raise KeyError("Checkpoint missing multihead_state_dict")

        # Set to eval mode
        self._backbone.eval()
        self._multihead_lm.eval()

        self._load_time = time.time() - start_time
        print(f"✓ Model loaded in {self._load_time:.2f}s")

    def predict(
        self,
        sensor_data: Dict[str, np.ndarray],
        method: str = "greedy",
        temperature: float = 1.0,
        max_length: int = 64,
        **kwargs
    ) -> Dict:
        """
        Run inference on sensor data.

        Args:
            sensor_data: Dictionary with 'continuous' and 'categorical'
            method: Generation method
            temperature: Sampling temperature
            max_length: Maximum sequence length
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with predictions
        """
        if self._backbone is None or self._multihead_lm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # Convert to tensors
        continuous = torch.from_numpy(sensor_data['continuous']).float()
        categorical = torch.from_numpy(sensor_data['categorical']).long()

        # Add batch dimension if needed
        if continuous.dim() == 2:
            continuous = continuous.unsqueeze(0)
        if categorical.dim() == 2:
            categorical = categorical.unsqueeze(0)

        # Move to device and convert categorical to float for Linear encoder
        continuous = continuous.to(self._device)
        categorical = categorical.to(self._device).float()  # Convert int to float

        # Compute sequence lengths (non-padded timesteps)
        # Assume all timesteps are valid (no padding in API inference)
        batch_size = continuous.size(0)
        seq_len = continuous.size(1)
        lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=self._device)

        # Run inference through backbone first
        with torch.no_grad():
            # Backbone: sensor data -> contextualized embeddings
            mods = [continuous, categorical]
            backbone_out = self._backbone(mods=mods, lengths=lengths)
            memory = backbone_out['memory']  # [B, T, d_model]

            # Multihead LM: autoregressive generation
            token_ids, predictions = self._multihead_lm.generate(
                memory=memory,
                max_len=max_length,
                bos_id=1,  # Assuming BOS token ID is 1
                decomposer=self._decomposer
            )

        # Decode token IDs to strings
        gcode_tokens = self._decode_token_ids(token_ids)

        inference_time = (time.time() - start_time) * 1000  # ms

        return {
            'gcode_sequence': gcode_tokens,
            'inference_time_ms': inference_time,
            'model_version': self._model_version,
        }

    def get_fingerprint(self, sensor_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract machine fingerprint embedding.

        Args:
            sensor_data: Dictionary with 'continuous' and 'categorical'

        Returns:
            Fingerprint embedding as numpy array
        """
        if self._backbone is None:
            raise RuntimeError("Model not loaded")

        # Convert to tensors
        continuous = torch.from_numpy(sensor_data['continuous']).float()
        categorical = torch.from_numpy(sensor_data['categorical']).long()

        if continuous.dim() == 2:
            continuous = continuous.unsqueeze(0)
        if categorical.dim() == 2:
            categorical = categorical.unsqueeze(0)

        continuous = continuous.to(self._device)
        categorical = categorical.to(self._device).float()  # Convert int to float

        # Compute sequence lengths
        batch_size = continuous.size(0)
        seq_len = continuous.size(1)
        lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=self._device)

        # Extract fingerprint using backbone
        with torch.no_grad():
            mods = [continuous, categorical]
            backbone_out = self._backbone(mods=mods, lengths=lengths)
            fingerprint = backbone_out['fingerprint']  # Already pooled by FingerprintHead

        return fingerprint.cpu().numpy()

    def get_model_info(self) -> Dict:
        """Get model metadata and configuration."""
        if self._backbone is None or self._multihead_lm is None:
            return {
                'model_loaded': False,
                'error': 'Model not loaded'
            }

        # Count parameters
        backbone_params = sum(p.numel() for p in self._backbone.parameters())
        multihead_params = sum(p.numel() for p in self._multihead_lm.parameters())
        num_params = backbone_params + multihead_params

        return {
            'model_loaded': True,
            'model_version': self._model_version,
            'num_parameters': num_params,
            'device': str(self._device),
            'config': self._config,
        }

    def _decode_token_ids(self, token_ids: torch.Tensor) -> List[str]:
        """
        Decode token IDs to G-code string tokens.

        Args:
            token_ids: [B, T] tensor of token IDs

        Returns:
            List of G-code token strings for first batch
        """
        # Convert first batch to list of token strings
        tokens = []
        for token_id in token_ids[0].cpu().tolist():
            token_str = self._decomposer.id2token.get(token_id, '<UNK>')
            # Stop at EOS or PAD tokens
            if token_str in ['<EOS>', '<PAD>']:
                break
            tokens.append(token_str)

        return tokens

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._backbone is not None and self._multihead_lm is not None
