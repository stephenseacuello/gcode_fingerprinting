"""
Decoder Dataset for Sensor-Conditioned Token Generation.

This dataset prepares sensor-token pairs for training the SensorConditionedTokenDecoder.
For each sample, it returns:
- sensor_features: The continuous sensor data [T_s, D_cont]
- input_tokens: [BOS, t1, t2, ..., tn] for teacher forcing
- target_tokens: [t1, t2, ..., tn, EOS] for loss computation
- length: Actual token sequence length (excluding BOS/EOS padding)
- operation_type: Operation class label for evaluation
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Special token IDs (from vocabulary)
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3


class DecoderDataset(Dataset):
    """
    Dataset for training sensor-conditioned token decoder.

    Prepares input/target pairs for autoregressive training:
    - input_tokens: [BOS, t1, t2, ..., tn] (shifted right)
    - target_tokens: [t1, t2, ..., tn, EOS] (what model should predict)
    """

    def __init__(
        self,
        npz_path: Path,
        max_token_len: int = 32,
        pad_token_id: int = PAD_TOKEN_ID,
        bos_token_id: int = BOS_TOKEN_ID,
        eos_token_id: int = EOS_TOKEN_ID,
    ):
        """
        Load processed sequences from .npz file.

        Args:
            npz_path: Path to .npz file from preprocessing
            max_token_len: Maximum token sequence length (including BOS/EOS)
            pad_token_id: Padding token ID (default: 0)
            bos_token_id: Beginning of sequence token ID (default: 1)
            eos_token_id: End of sequence token ID (default: 2)
        """
        self.data = np.load(npz_path, allow_pickle=True)
        self.max_token_len = max_token_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Load sensor features (continuous only for now)
        self.continuous = torch.from_numpy(self.data['continuous']).float()  # [N, T_s, D_cont]

        # Load tokens and lengths
        self.tokens = torch.from_numpy(self.data['tokens']).long()  # [N, max_len]
        self.lengths = torch.from_numpy(self.data['lengths']).long()  # [N]

        # Load operation types
        if 'operation_type' in self.data:
            self.operation_type = torch.from_numpy(self.data['operation_type']).long()
        else:
            self.operation_type = torch.zeros(len(self.continuous), dtype=torch.long)

        # Store G-code texts for debugging
        self.gcode_texts = self.data.get('gcode_texts', None)

        self.n_samples = len(self.continuous)

        # Precompute input/target pairs for efficiency
        self._prepare_decoder_pairs()

    def _prepare_decoder_pairs(self):
        """Prepare input and target token sequences for decoder training."""
        N = self.n_samples
        max_len = self.max_token_len

        # Initialize with padding
        self.input_tokens = torch.full((N, max_len), self.pad_token_id, dtype=torch.long)
        self.target_tokens = torch.full((N, max_len), self.pad_token_id, dtype=torch.long)
        self.token_lengths = torch.zeros(N, dtype=torch.long)

        for i in range(N):
            seq_len = min(self.lengths[i].item(), max_len - 2)  # Reserve space for BOS/EOS

            # Get original tokens (excluding any existing BOS/EOS)
            orig_tokens = self.tokens[i, :seq_len]

            # Filter out any existing special tokens
            mask = (orig_tokens != self.pad_token_id) & \
                   (orig_tokens != self.bos_token_id) & \
                   (orig_tokens != self.eos_token_id)
            clean_tokens = orig_tokens[mask]
            clean_len = len(clean_tokens)

            if clean_len == 0:
                # Edge case: no valid tokens, just BOS -> EOS
                self.input_tokens[i, 0] = self.bos_token_id
                self.target_tokens[i, 0] = self.eos_token_id
                self.token_lengths[i] = 1
            else:
                # Truncate if needed
                if clean_len > max_len - 2:
                    clean_tokens = clean_tokens[:max_len - 2]
                    clean_len = max_len - 2

                # Input: [BOS, t1, t2, ..., tn]
                self.input_tokens[i, 0] = self.bos_token_id
                self.input_tokens[i, 1:1+clean_len] = clean_tokens

                # Target: [t1, t2, ..., tn, EOS]
                self.target_tokens[i, :clean_len] = clean_tokens
                self.target_tokens[i, clean_len] = self.eos_token_id

                # Length includes the EOS token
                self.token_lengths[i] = clean_len + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample for decoder training.

        Returns:
            Dictionary with:
                - sensor_features: [T_s, D_cont] - sensor readings
                - input_tokens: [max_len] - input for decoder (with BOS prefix)
                - target_tokens: [max_len] - targets (shifted, with EOS suffix)
                - length: scalar - actual sequence length
                - operation_type: scalar - operation class label
                - padding_mask: [max_len] - True where padded
        """
        length = self.token_lengths[idx]

        # Create padding mask (True for positions that should be ignored)
        padding_mask = torch.zeros(self.max_token_len, dtype=torch.bool)
        padding_mask[length:] = True

        return {
            'sensor_features': self.continuous[idx],  # [T_s, D_cont]
            'input_tokens': self.input_tokens[idx],   # [max_len]
            'target_tokens': self.target_tokens[idx], # [max_len]
            'length': length,
            'operation_type': self.operation_type[idx],
            'padding_mask': padding_mask,
        }

    def get_sensor_dim(self) -> int:
        """Get sensor feature dimension."""
        return self.continuous.size(-1)

    def get_sensor_seq_len(self) -> int:
        """Get sensor sequence length."""
        return self.continuous.size(1)


def decoder_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for decoder DataLoader.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched tensors
    """
    sensor_features = torch.stack([item['sensor_features'] for item in batch])
    input_tokens = torch.stack([item['input_tokens'] for item in batch])
    target_tokens = torch.stack([item['target_tokens'] for item in batch])
    lengths = torch.stack([item['length'] for item in batch])
    operation_types = torch.stack([item['operation_type'] for item in batch])
    padding_masks = torch.stack([item['padding_mask'] for item in batch])

    return {
        'sensor_features': sensor_features,      # [B, T_s, D_cont]
        'input_tokens': input_tokens,            # [B, max_len]
        'target_tokens': target_tokens,          # [B, max_len]
        'lengths': lengths,                      # [B]
        'operation_type': operation_types,       # [B]
        'padding_mask': padding_masks,           # [B, max_len]
    }


class DecoderDatasetFromSplits(DecoderDataset):
    """
    Decoder dataset that loads from grouped splits directory.

    This is used when train/val/test splits are stored in separate files
    to prevent data leakage.
    """

    def __init__(
        self,
        split_dir: Path,
        split: str,  # 'train', 'val', or 'test'
        max_token_len: int = 32,
        pad_token_id: int = PAD_TOKEN_ID,
        bos_token_id: int = BOS_TOKEN_ID,
        eos_token_id: int = EOS_TOKEN_ID,
    ):
        """
        Load from split directory.

        Args:
            split_dir: Directory containing train.npz, val.npz, test.npz
            split: Which split to load ('train', 'val', or 'test')
            max_token_len: Maximum token sequence length
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
        """
        split_dir = Path(split_dir)

        # Try different naming conventions
        npz_path = split_dir / f"{split}.npz"
        if not npz_path.exists():
            # Try the _sequences naming convention
            npz_path = split_dir / f"{split}_sequences.npz"

        if not npz_path.exists():
            raise FileNotFoundError(
                f"Split file not found: tried {split_dir / f'{split}.npz'} "
                f"and {split_dir / f'{split}_sequences.npz'}"
            )

        super().__init__(
            npz_path=npz_path,
            max_token_len=max_token_len,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
