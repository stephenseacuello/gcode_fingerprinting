"""
Self-Critical Sequence Training (SCST) for G-code generation.

SCST uses REINFORCE with a self-critical baseline:
- Baseline: Greedy decoded sequence reward
- Sample: Stochastically sampled sequence reward
- Loss: -log_prob * (sample_reward - baseline_reward)

This allows direct optimization of sequence-level metrics (like sequence accuracy)
rather than just token-level cross-entropy loss.

Reference:
Rennie et al. "Self-Critical Sequence Training for Image Captioning" (CVPR 2017)

Author: Claude Code
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class SCSTTrainer:
    """
    Self-Critical Sequence Training (SCST) for sequence-level optimization.

    SCST uses REINFORCE with a greedy baseline to reduce variance:
    - Sample a sequence from the model
    - Compute greedy baseline sequence (no sampling)
    - Reward = sequence_accuracy(sampled) - sequence_accuracy(greedy)
    - Loss = -sum(log_probs) * reward

    This directly optimizes for sequence-level accuracy.
    """

    def __init__(
        self,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 32,
        sample_temperature: float = 0.8,
        reward_scale: float = 1.0,
    ):
        """
        Args:
            pad_token_id: PAD token ID for masking
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            max_length: Maximum generation length
            sample_temperature: Temperature for sampling (lower = more greedy)
            reward_scale: Scale factor for rewards (for gradient stability)
        """
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_length = max_length
        self.sample_temperature = sample_temperature
        self.reward_scale = reward_scale

    def compute_sequence_reward(
        self,
        pred_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sequence-level reward (exact sequence match).

        Args:
            pred_tokens: Predicted token IDs [B, L_pred]
            target_tokens: Target token IDs [B, L_target]

        Returns:
            rewards: [B] reward for each sequence (1.0 if match, 0.0 otherwise)
        """
        B = pred_tokens.size(0)
        device = pred_tokens.device
        rewards = torch.zeros(B, device=device)

        for b in range(B):
            # Get non-padded lengths
            pred_mask = (pred_tokens[b] != self.pad_token_id) & (pred_tokens[b] != self.bos_token_id)
            target_mask = target_tokens[b] != self.pad_token_id

            pred_seq = pred_tokens[b][pred_mask]
            target_seq = target_tokens[b][target_mask]

            # Strip BOS from prediction if present
            if len(pred_seq) > 0 and pred_seq[0] == self.bos_token_id:
                pred_seq = pred_seq[1:]

            # Strip EOS from both
            if len(pred_seq) > 0 and pred_seq[-1] == self.eos_token_id:
                pred_seq = pred_seq[:-1]
            if len(target_seq) > 0 and target_seq[-1] == self.eos_token_id:
                target_seq = target_seq[:-1]

            # Check exact match
            if len(pred_seq) == len(target_seq):
                if (pred_seq == target_seq).all():
                    rewards[b] = 1.0
                else:
                    # Partial credit: token accuracy
                    rewards[b] = (pred_seq == target_seq).float().mean() * 0.5

        return rewards

    def compute_token_reward(
        self,
        pred_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute token-level reward (useful as softer signal).

        Args:
            pred_tokens: Predicted token IDs [B, L_pred]
            target_tokens: Target token IDs [B, L_target]

        Returns:
            rewards: [B] token accuracy for each sequence
        """
        B = pred_tokens.size(0)
        device = pred_tokens.device
        rewards = torch.zeros(B, device=device)

        for b in range(B):
            # Get non-padded targets
            target_mask = target_tokens[b] != self.pad_token_id
            target_seq = target_tokens[b][target_mask]

            # Get comparable length of predictions (strip BOS)
            pred_seq = pred_tokens[b]
            if len(pred_seq) > 0 and pred_seq[0] == self.bos_token_id:
                pred_seq = pred_seq[1:]

            # Compare up to target length
            compare_len = min(len(pred_seq), len(target_seq))
            if compare_len > 0:
                correct = (pred_seq[:compare_len] == target_seq[:compare_len]).float().sum()
                rewards[b] = correct / len(target_seq)

        return rewards

    def compute_scst_loss(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        sensor_features: torch.Tensor,
        target_tokens: torch.Tensor,
        device: torch.device,
        use_mixed_reward: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute SCST loss for a batch.

        Args:
            encoder: Frozen sensor encoder
            decoder: Token decoder (gradients flow through this)
            sensor_features: Raw sensor data [B, T_s, D]
            target_tokens: Ground truth token sequences [B, L]
            device: Compute device
            use_mixed_reward: If True, combine sequence and token rewards

        Returns:
            loss: SCST loss (negative to minimize)
            metrics: Dictionary with training metrics
        """
        B = sensor_features.size(0)

        # Encode sensors
        with torch.no_grad():
            sensor_emb, _ = encoder.encode(sensor_features)
            op_logits, _ = encoder.classify(sensor_emb)
            operation_type = op_logits.argmax(-1)

        # 1. Sample sequence with gradients
        decoder.train()
        sampled_tokens, log_probs, _ = decoder.generate_with_gradients(
            sensor_embeddings=sensor_emb,
            operation_type=operation_type,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            max_length=self.max_length,
            temperature=self.sample_temperature,
        )

        # 2. Greedy baseline (no gradients)
        decoder.eval()
        with torch.no_grad():
            baseline_tokens, _ = decoder.generate(
                sensor_embeddings=sensor_emb,
                operation_type=operation_type,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                max_length=self.max_length,
                temperature=1.0,  # Greedy (will use argmax internally)
                top_k=1,  # Force greedy
            )
        decoder.train()

        # 3. Compute rewards
        if use_mixed_reward:
            # Combine sequence and token rewards
            sample_seq_reward = self.compute_sequence_reward(sampled_tokens, target_tokens)
            sample_tok_reward = self.compute_token_reward(sampled_tokens, target_tokens)
            sample_reward = 0.7 * sample_seq_reward + 0.3 * sample_tok_reward

            baseline_seq_reward = self.compute_sequence_reward(baseline_tokens, target_tokens)
            baseline_tok_reward = self.compute_token_reward(baseline_tokens, target_tokens)
            baseline_reward = 0.7 * baseline_seq_reward + 0.3 * baseline_tok_reward
        else:
            # Pure sequence reward
            sample_reward = self.compute_sequence_reward(sampled_tokens, target_tokens)
            baseline_reward = self.compute_sequence_reward(baseline_tokens, target_tokens)

        # 4. Advantage (REINFORCE with baseline)
        advantage = (sample_reward - baseline_reward) * self.reward_scale  # [B]

        # 5. REINFORCE loss: -log_prob * advantage
        # Sum log_probs over sequence, then apply advantage
        seq_log_probs = log_probs.sum(dim=1)  # [B]
        loss = -seq_log_probs * advantage
        loss = loss.mean()  # Average over batch

        # Metrics
        metrics = {
            'scst_loss': loss.item(),
            'sample_reward': sample_reward.mean().item(),
            'baseline_reward': baseline_reward.mean().item(),
            'advantage': advantage.mean().item(),
            'sample_seq_acc': sample_seq_reward.mean().item() if use_mixed_reward else sample_reward.mean().item(),
            'baseline_seq_acc': baseline_seq_reward.mean().item() if use_mixed_reward else baseline_reward.mean().item(),
        }

        return loss, metrics


class MixedSCSTLoss(nn.Module):
    """
    Combined cross-entropy + SCST loss for gradual transition.

    During fine-tuning, we mix:
    - Cross-entropy loss (stable, token-level)
    - SCST loss (high variance, sequence-level)

    Mixing ratio typically starts high on CE and gradually increases SCST.
    """

    def __init__(
        self,
        scst_trainer: SCSTTrainer,
        ce_weight: float = 0.5,
        scst_weight: float = 0.5,
    ):
        super().__init__()
        self.scst_trainer = scst_trainer
        self.ce_weight = ce_weight
        self.scst_weight = scst_weight

    def forward(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        sensor_features: torch.Tensor,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        device: torch.device,
        ce_loss_fn: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute mixed CE + SCST loss.

        Args:
            encoder: Sensor encoder
            decoder: Token decoder
            sensor_features: [B, T_s, D]
            input_tokens: [B, L] input tokens for CE
            target_tokens: [B, L] target tokens
            device: Compute device
            ce_loss_fn: Cross-entropy loss function

        Returns:
            total_loss: Combined loss
            metrics: Dictionary of metrics
        """
        metrics = {}

        # Cross-entropy loss (standard training)
        with torch.no_grad():
            sensor_emb, _ = encoder.encode(sensor_features)
            op_logits, _ = encoder.classify(sensor_emb)
            operation_type = op_logits.argmax(-1)

        outputs = decoder(
            tokens=input_tokens,
            sensor_embeddings=sensor_emb,
            operation_type=operation_type,
        )

        targets = {'target_tokens': target_tokens}
        ce_loss, ce_metrics = ce_loss_fn(outputs, targets)
        metrics['ce_loss'] = ce_loss.item()
        metrics.update({f'ce_{k}': v for k, v in ce_metrics.items()})

        # SCST loss
        scst_loss, scst_metrics = self.scst_trainer.compute_scst_loss(
            encoder=encoder,
            decoder=decoder,
            sensor_features=sensor_features,
            target_tokens=target_tokens,
            device=device,
        )
        metrics.update(scst_metrics)

        # Combined loss
        total_loss = self.ce_weight * ce_loss + self.scst_weight * scst_loss
        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics
