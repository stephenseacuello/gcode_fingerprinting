"""
PyTorch Dataset for G-code sensor data.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

__all__ = ["GCodeDataset", "collate_fn"]


class GCodeDataset(Dataset):
    """Dataset for G-code sensor sequences."""

    def __init__(self, npz_path: Path):
        """
        Load processed sequences from .npz file.

        Args:
            npz_path: Path to .npz file from preprocessing
        """
        self.data = np.load(npz_path, allow_pickle=True)

        self.continuous = torch.from_numpy(self.data['continuous']).float()  # [N, T, D_cont]
        self.categorical = torch.from_numpy(self.data['categorical']).long()  # [N, T, D_cat]
        self.tokens = torch.from_numpy(self.data['tokens']).long()  # [N, max_token_len]
        self.lengths = torch.from_numpy(self.data['lengths']).long()  # [N]
        self.gcode_texts = self.data['gcode_texts']  # [N]

        # Load operation types if available
        if 'operation_type' in self.data:
            self.operation_type = torch.from_numpy(self.data['operation_type']).long()  # [N]
        else:
            # Default to unknown (9) if not available for backwards compatibility
            self.operation_type = torch.full((len(self.continuous),), 9, dtype=torch.long)

        # Load residuals if available
        if 'residuals' in self.data:
            self.residuals = torch.from_numpy(self.data['residuals']).float()  # [N, max_token_len]
        else:
            # Default to zeros if not available for backwards compatibility
            self.residuals = torch.zeros_like(self.tokens).float()

        # Load param_value_raw for regression training
        if 'param_value_raw' in self.data:
            self.param_value_raw = torch.from_numpy(self.data['param_value_raw']).float()  # [N, max_token_len]
        else:
            # Default to zeros if not available for backwards compatibility
            self.param_value_raw = torch.zeros_like(self.tokens).float()

        self.n_samples = len(self.continuous)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - continuous: [T, D_cont]
                - categorical: [T, D_cat]
                - tokens: [max_token_len]
                - residuals: [max_token_len]
                - param_value_raw: [max_token_len] - raw numeric values for regression
                - length: scalar
                - gcode_text: string
                - operation_type: scalar
        """
        return {
            'continuous': self.continuous[idx],
            'categorical': self.categorical[idx],
            'tokens': self.tokens[idx],
            'residuals': self.residuals[idx],
            'param_value_raw': self.param_value_raw[idx],
            'length': self.lengths[idx],
            'gcode_text': str(self.gcode_texts[idx]),
            'operation_type': self.operation_type[idx],
        }

    def get_feature_dims(self) -> Tuple[int, int]:
        """Get feature dimensions."""
        return self.continuous.size(-1), self.categorical.size(-1)


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched tensors
    """
    # Stack continuous and categorical features
    continuous = torch.stack([item['continuous'] for item in batch])  # [B, T, D_cont]
    categorical = torch.stack([item['categorical'] for item in batch])  # [B, T, D_cat]
    tokens = torch.stack([item['tokens'] for item in batch])  # [B, max_token_len]
    residuals = torch.stack([item['residuals'] for item in batch])  # [B, max_token_len]
    param_value_raw = torch.stack([item['param_value_raw'] for item in batch])  # [B, max_token_len]
    lengths = torch.stack([item['length'] for item in batch])  # [B]
    gcode_texts = [item['gcode_text'] for item in batch]  # List[str]
    operation_types = torch.stack([item['operation_type'] for item in batch])  # [B]

    return {
        'continuous': continuous,
        'categorical': categorical,
        'tokens': tokens,
        'residuals': residuals,
        'param_value_raw': param_value_raw,
        'lengths': lengths,
        'gcode_texts': gcode_texts,
        'operation_type': operation_types,
    }
