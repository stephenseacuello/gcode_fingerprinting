"""
PyTorch Dataset for G-code sensor data.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

from .target_utils import (
    decompose_value_to_digits,
    SIGN_POSITIVE, SIGN_NEGATIVE, SIGN_PAD, DIGIT_PAD
)

__all__ = ["GCodeDataset", "collate_fn"]


# Special token IDs (from vocabulary)
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


class GCodeDataset(Dataset):
    """Dataset for G-code sensor sequences."""

    def __init__(
        self,
        npz_path: Path,
        use_digit_targets: bool = False,
        max_int_digits: int = 2,
        n_decimal_digits: int = 4,
        augment_values: bool = False,
        augment_noise_std: float = 0.01,
        augment_scale_range: Tuple[float, float] = (0.95, 1.05),
        prepend_bos: bool = False,
        bos_token_id: int = BOS_TOKEN_ID,
    ):
        """
        Load processed sequences from .npz file.

        Args:
            npz_path: Path to .npz file from preprocessing
            use_digit_targets: Whether to compute digit-by-digit targets for numeric values
            max_int_digits: Number of integer digit positions for digit targets
            n_decimal_digits: Number of decimal digit positions for digit targets
            augment_values: Whether to apply random augmentation to numeric values (for training)
            augment_noise_std: Standard deviation of Gaussian noise to add to values
            augment_scale_range: Range for random scaling (min, max)
            prepend_bos: Whether to prepend BOS token to sequences (required for proper
                teacher forcing where input=[BOS,t1,...,tn] and target=[t1,...,tn,EOS])
            bos_token_id: ID of BOS token in vocabulary (default: 1)
        """
        self.data = np.load(npz_path, allow_pickle=True)
        self.prepend_bos = prepend_bos
        self.bos_token_id = bos_token_id
        self.use_digit_targets = use_digit_targets
        self.max_int_digits = max_int_digits
        self.n_decimal_digits = n_decimal_digits
        self.n_digits = max_int_digits + n_decimal_digits
        self.augment_values = augment_values
        self.augment_noise_std = augment_noise_std
        self.augment_scale_range = augment_scale_range

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

        # Prepend BOS token if requested
        # This is required for proper teacher forcing: input=[BOS,t1,...,tn], target=[t1,...,tn,...]
        # Without BOS, the first token (often a command) would never be a prediction target
        if self.prepend_bos:
            N, T = self.tokens.shape
            # Create new tensors with room for BOS at position 0
            new_tokens = torch.full((N, T + 1), PAD_TOKEN_ID, dtype=torch.long)
            new_tokens[:, 0] = self.bos_token_id
            new_tokens[:, 1:] = self.tokens
            self.tokens = new_tokens

            # Shift residuals (BOS has no residual, set to 0)
            new_residuals = torch.zeros((N, T + 1), dtype=torch.float)
            new_residuals[:, 1:] = self.residuals
            self.residuals = new_residuals

            # Shift param_value_raw (BOS has no param value, set to 0)
            new_param_value_raw = torch.zeros((N, T + 1), dtype=torch.float)
            new_param_value_raw[:, 1:] = self.param_value_raw
            self.param_value_raw = new_param_value_raw

            # Increment lengths by 1 to account for BOS
            self.lengths = self.lengths + 1

        self.n_samples = len(self.continuous)

        # Precompute digit targets if enabled
        if self.use_digit_targets:
            self._precompute_digit_targets()

    def _precompute_digit_targets(self):
        """Precompute digit-by-digit targets from raw numeric values."""
        N, T = self.param_value_raw.shape

        # Initialize with PAD values
        self.sign_targets = torch.full((N, T), SIGN_PAD, dtype=torch.long)
        self.digit_targets = torch.full((N, T, self.n_digits), DIGIT_PAD, dtype=torch.long)

        # We need to identify which positions are numeric values
        # For now, decompose all non-zero param_value_raw values
        # A more accurate approach would use the TokenDecomposer, but this is faster

        for i in range(N):
            for t in range(T):
                raw_val = self.param_value_raw[i, t].item()
                # Only decompose if there's a meaningful value
                # Check if it's a numeric position (non-zero or explicitly zero coordinate)
                if raw_val != 0.0 or t < self.lengths[i].item():
                    sign, digits = decompose_value_to_digits(
                        raw_val, self.max_int_digits, self.n_decimal_digits
                    )
                    self.sign_targets[i, t] = sign
                    for d, digit in enumerate(digits):
                        self.digit_targets[i, t, d] = digit

    def __len__(self) -> int:
        return self.n_samples

    def _augment_value(self, value: float) -> float:
        """Apply random augmentation to a numeric value."""
        if value == 0.0:
            return value
        # Add Gaussian noise
        noise = np.random.normal(0, self.augment_noise_std)
        # Random scaling
        scale = np.random.uniform(*self.augment_scale_range)
        augmented = value * scale + noise
        return augmented

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
                - sign_targets: [max_token_len] (if use_digit_targets)
                - digit_targets: [max_token_len, n_digits] (if use_digit_targets)
        """
        param_value_raw = self.param_value_raw[idx].clone()

        # Apply augmentation if enabled (recompute digit targets on the fly)
        if self.augment_values:
            T = param_value_raw.shape[0]
            for t in range(T):
                raw_val = param_value_raw[t].item()
                if raw_val != 0.0:
                    param_value_raw[t] = self._augment_value(raw_val)

        sample = {
            'continuous': self.continuous[idx],
            'categorical': self.categorical[idx],
            'tokens': self.tokens[idx],
            'residuals': self.residuals[idx],
            'param_value_raw': param_value_raw,
            'length': self.lengths[idx],
            'gcode_text': str(self.gcode_texts[idx]),
            'operation_type': self.operation_type[idx],
        }

        if self.use_digit_targets:
            if self.augment_values:
                # Recompute digit targets from augmented values
                T = param_value_raw.shape[0]
                sign_targets = torch.full((T,), SIGN_PAD, dtype=torch.long)
                digit_targets = torch.full((T, self.n_digits), DIGIT_PAD, dtype=torch.long)

                for t in range(T):
                    raw_val = param_value_raw[t].item()
                    if raw_val != 0.0 or t < self.lengths[idx].item():
                        sign, digits = decompose_value_to_digits(
                            raw_val, self.max_int_digits, self.n_decimal_digits
                        )
                        sign_targets[t] = sign
                        for d, digit in enumerate(digits):
                            digit_targets[t, d] = digit

                sample['sign_targets'] = sign_targets
                sample['digit_targets'] = digit_targets
            else:
                sample['sign_targets'] = self.sign_targets[idx]
                sample['digit_targets'] = self.digit_targets[idx]

        return sample

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

    result = {
        'continuous': continuous,
        'categorical': categorical,
        'tokens': tokens,
        'residuals': residuals,
        'param_value_raw': param_value_raw,
        'lengths': lengths,
        'gcode_texts': gcode_texts,
        'operation_type': operation_types,
    }

    # Add digit targets if present
    if 'sign_targets' in batch[0]:
        result['sign_targets'] = torch.stack([item['sign_targets'] for item in batch])  # [B, max_token_len]
        result['digit_targets'] = torch.stack([item['digit_targets'] for item in batch])  # [B, max_token_len, n_digits]

    return result
