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
        allow_truncation: bool = False,
        length_field: str = "auto",
    ):
        """
        Load processed sequences from .npz file.

        Args:
            npz_path: Path to .npz file from preprocessing
            max_token_len: Maximum token sequence length (including BOS/EOS)
            pad_token_id: Padding token ID (default: 0)
            bos_token_id: Beginning of sequence token ID (default: 1)
            eos_token_id: End of sequence token ID (default: 2)
            allow_truncation: If False (default, V8 behavior), raise an
                AssertionError when any sample's true content length would
                require truncation under `max_token_len`. If True, fall back
                to the V7 silent-truncation behavior. **Always leave False
                for new training.** Phase-2 audit traced a silent
                target-truncation bug to this path (see
                outputs/decoder20260511/AUDIT_REPORT.md, Priority 1).
            length_field: Which NPZ field to consume as the per-sample token
                length. "auto" prefers `token_length` (V8 schema), falls back
                to `lengths` (legacy). "token_length" / "lengths" force a
                specific field. Raises a clear error if the chosen field is
                actually the sensor window length (the V7 bug).
        """
        self.data = np.load(npz_path, allow_pickle=True)
        self.max_token_len = max_token_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.allow_truncation = allow_truncation
        self.npz_path = Path(npz_path)

        # Load sensor features (continuous only for now)
        self.continuous = torch.from_numpy(self.data['continuous']).float()  # [N, T_s, D_cont]

        # Load tokens and lengths.
        self.tokens = torch.from_numpy(self.data['tokens']).long()  # [N, max_len]

        # Pick the length field. V8 NPZ writes `token_length` explicitly;
        # legacy V7 NPZ has `lengths` populated with the WRONG value (sensor
        # window length, not token length). Auto-detection prefers
        # `token_length` and validates whatever it picks.
        if length_field == "auto":
            chosen = "token_length" if "token_length" in self.data.files else "lengths"
        else:
            chosen = length_field
        if chosen not in self.data.files:
            raise KeyError(
                f"NPZ {self.npz_path} has no field '{chosen}'. "
                f"Available: {list(self.data.files)}"
            )
        self.lengths = torch.from_numpy(self.data[chosen]).long()  # [N]
        self.length_field = chosen

        # Derive ground-truth content length from `tokens` directly and verify
        # that the chosen `lengths` field matches. This catches the V7
        # lengths-is-sensor-length bug at load time rather than silently
        # truncating downstream.
        content_mask = (
            (self.tokens != self.pad_token_id)
            & (self.tokens != self.bos_token_id)
            & (self.tokens != self.eos_token_id)
        )
        self._content_lengths = content_mask.sum(dim=1).long()  # [N]

        if not torch.equal(self.lengths, self._content_lengths):
            # If the mismatch is small (off by 1 for EOS / BOS accounting),
            # tolerate it; otherwise hard-fail with a diagnostic message.
            max_diff = int((self.lengths - self._content_lengths).abs().max().item()) if self.lengths.numel() else 0
            if max_diff > 1:
                window_size = int(self.data["continuous"].shape[1]) if "continuous" in self.data.files else -1
                if int(self.lengths.max().item()) == window_size and int(self.lengths.min().item()) == window_size:
                    raise ValueError(
                        f"[DecoderDataset] {self.npz_path} field '{chosen}' looks like the "
                        f"SENSOR window length ({window_size}), not the token length. "
                        f"This is the V7 preprocessing bug — see "
                        f"outputs/decoder20260511/AUDIT_REPORT.md, Priority 1. "
                        f"Pass length_field='token_length' to opt into the V8 NPZ schema "
                        f"or regenerate the NPZ with the fixed preprocessing."
                    )
                # Otherwise just warn but proceed
                import warnings
                warnings.warn(
                    f"[DecoderDataset] {self.npz_path} field '{chosen}' disagrees with "
                    f"derived content length by up to {max_diff}. Proceeding with the NPZ "
                    f"value, but check the preprocessing pipeline.",
                    stacklevel=2,
                )

        # Hard assertion: refuse to silently truncate decoder targets.
        max_true_len = int(self._content_lengths.max().item()) if self._content_lengths.numel() else 0
        if max_true_len > max_token_len - 2 and not self.allow_truncation:
            n_over = int((self._content_lengths > max_token_len - 2).sum().item())
            raise AssertionError(
                f"[DecoderDataset] {self.npz_path}: {n_over}/{self.n_samples_eager()} "
                f"samples have content length > {max_token_len - 2} (max needed = {max_true_len}). "
                f"Silent truncation prohibited. Pass max_token_len >= {max_true_len + 2} "
                f"or set allow_truncation=True if intentional (audit logged)."
            )

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

    def n_samples_eager(self) -> int:
        """Sample count usable before __init__ finishes (for error messages)."""
        return int(self.continuous.shape[0])

    def _prepare_decoder_pairs(self):
        """Prepare input and target token sequences for decoder training.

        Phase-3 (decoder20260511) rewrite: removed the silent `min(...)`
        truncation. By the time we reach this method, __init__ has already
        verified that `max_token_len` is large enough; any truncation here
        is opt-in via `allow_truncation`.
        """
        N = self.n_samples
        max_len = self.max_token_len

        # Initialize with padding
        self.input_tokens = torch.full((N, max_len), self.pad_token_id, dtype=torch.long)
        self.target_tokens = torch.full((N, max_len), self.pad_token_id, dtype=torch.long)
        self.token_lengths = torch.zeros(N, dtype=torch.long)

        # Stats for the optional diagnostics report
        self._truncation_count = 0

        for i in range(N):
            # Use the derived content length (already validated in __init__).
            content_len = int(self._content_lengths[i].item())
            # Pull the actual content tokens from row i, in original order,
            # skipping any PAD/BOS/EOS the preprocessor may have left behind.
            row = self.tokens[i]
            mask = (
                (row != self.pad_token_id)
                & (row != self.bos_token_id)
                & (row != self.eos_token_id)
            )
            clean_tokens = row[mask]
            clean_len = int(clean_tokens.numel())

            if clean_len == 0:
                # Edge case: empty target (e.g., per_row placeholder for a
                # window with no G-code firing). Emit BOS->EOS.
                self.input_tokens[i, 0] = self.bos_token_id
                self.target_tokens[i, 0] = self.eos_token_id
                self.token_lengths[i] = 1
                continue

            # Capacity check. The eager assertion in __init__ should have
            # caught violations, but allow_truncation users land here.
            cap = max_len - 2  # reserve BOS + EOS
            if clean_len > cap:
                if not self.allow_truncation:
                    # Defensive: should not be reachable if __init__ assertion fired.
                    raise AssertionError(
                        f"[DecoderDataset] sample {i} content_len={clean_len} exceeds "
                        f"max_token_len-2={cap}. Set allow_truncation=True or raise max_token_len."
                    )
                clean_tokens = clean_tokens[:cap]
                clean_len = cap
                self._truncation_count += 1

            # Input: [BOS, t1, t2, ..., tn]
            self.input_tokens[i, 0] = self.bos_token_id
            self.input_tokens[i, 1:1 + clean_len] = clean_tokens

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
