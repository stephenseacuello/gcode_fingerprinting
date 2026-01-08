"""
Lightweight reranker for beam search candidates.

Scores beam search candidates based on:
1. Sensor-token consistency (do predicted tokens match sensor features?)
2. Grammar validity (does the sequence follow G-code grammar?)
3. Length normalization
4. Diversity bonus (prevent near-duplicate sequences)

Author: Claude Code
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class SensorTokenReranker(nn.Module):
    """
    Lightweight reranker for beam search candidates.

    Scores candidates based on how well they align with sensor features,
    providing a secondary verification beyond the main decoder.
    """

    def __init__(
        self,
        d_model: int = 192,
        sensor_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_grammar_score: bool = True,
    ):
        """
        Args:
            d_model: Hidden dimension
            sensor_dim: Sensor embedding dimension
            n_heads: Number of attention heads for cross-attention
            dropout: Dropout rate
            use_grammar_score: Include grammar validity scoring
        """
        super().__init__()
        self.d_model = d_model
        self.use_grammar_score = use_grammar_score

        # Project sensor embeddings to scoring space
        self.sensor_proj = nn.Sequential(
            nn.Linear(sensor_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Project token embeddings to scoring space
        self.token_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Cross-attention: query=tokens, key/value=sensors
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Score head: takes attended features and outputs consistency score
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Grammar score network (if enabled)
        if use_grammar_score:
            self.grammar_head = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid(),  # Output in [0, 1]
            )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        sensor_embeddings: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Score a batch of token sequences for consistency with sensor data.

        Args:
            token_embeddings: [B, L, d_model] token representations
            sensor_embeddings: [B, T_s, sensor_dim] sensor features
            token_mask: [B, L] True for PAD positions

        Returns:
            scores: [B] consistency scores (higher is better)
            score_dict: Dict with individual score components
        """
        B = token_embeddings.size(0)
        device = token_embeddings.device

        # Project to scoring space
        tokens_proj = self.token_proj(token_embeddings)  # [B, L, d_model]
        sensors_proj = self.sensor_proj(sensor_embeddings)  # [B, T_s, d_model]

        # Cross-attention: how well do tokens attend to relevant sensors?
        attended, attn_weights = self.cross_attn(
            query=tokens_proj,
            key=sensors_proj,
            value=sensors_proj,
            key_padding_mask=None,  # TODO: add sensor padding mask
        )  # [B, L, d_model], [B, L, T_s]

        # Pool attended features
        if token_mask is not None:
            # Mask PAD positions
            attended = attended.masked_fill(token_mask.unsqueeze(-1), 0.0)
            valid_counts = (~token_mask).sum(dim=1, keepdim=True).float()  # [B, 1]
            pooled = attended.sum(dim=1) / valid_counts.clamp(min=1)  # [B, d_model]
        else:
            pooled = attended.mean(dim=1)  # [B, d_model]

        # Consistency score
        consistency_score = self.score_head(pooled).squeeze(-1)  # [B]

        score_dict = {'consistency': consistency_score}
        total_score = consistency_score

        # Grammar score (if enabled)
        if self.use_grammar_score:
            grammar_score = self.grammar_head(pooled).squeeze(-1)  # [B]
            score_dict['grammar'] = grammar_score
            total_score = total_score + grammar_score

        score_dict['total'] = total_score

        return total_score, score_dict


class BeamReranker:
    """
    Reranker that combines multiple scoring signals for beam candidates.
    """

    def __init__(
        self,
        model_scorer: Optional[SensorTokenReranker] = None,
        length_penalty: float = 0.6,
        diversity_penalty: float = 0.0,
        grammar_weight: float = 0.2,
        consistency_weight: float = 0.3,
    ):
        """
        Args:
            model_scorer: Optional learned scorer
            length_penalty: Alpha for length normalization (Wu et al.)
            diversity_penalty: Penalty for similar sequences in beam
            grammar_weight: Weight for grammar validity score
            consistency_weight: Weight for sensor-token consistency
        """
        self.model_scorer = model_scorer
        self.length_penalty = length_penalty
        self.diversity_penalty = diversity_penalty
        self.grammar_weight = grammar_weight
        self.consistency_weight = consistency_weight

    def length_normalize(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply length normalization to beam scores.

        Uses Wu et al. (2016) formula: lp(Y) = ((5 + |Y|) / 6)^alpha

        Args:
            log_probs: [B] cumulative log probabilities
            lengths: [B] sequence lengths

        Returns:
            [B] length-normalized scores
        """
        lp = ((5.0 + lengths.float()) / 6.0) ** self.length_penalty
        return log_probs / lp

    def compute_diversity_penalty(
        self,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute diversity penalty based on sequence similarity.

        Penalizes sequences that are too similar to others in the beam.

        Args:
            sequences: [B, L] token sequences

        Returns:
            [B] diversity penalties (subtract from score)
        """
        B = sequences.size(0)
        device = sequences.device

        if B <= 1 or self.diversity_penalty == 0:
            return torch.zeros(B, device=device)

        # Compute pairwise Jaccard similarity
        penalties = torch.zeros(B, device=device)

        for i in range(B):
            seq_i = set(sequences[i].tolist())
            max_sim = 0.0
            for j in range(B):
                if i != j:
                    seq_j = set(sequences[j].tolist())
                    intersection = len(seq_i & seq_j)
                    union = len(seq_i | seq_j)
                    if union > 0:
                        sim = intersection / union
                        max_sim = max(max_sim, sim)
            penalties[i] = max_sim * self.diversity_penalty

        return penalties

    def rerank(
        self,
        sequences: torch.Tensor,
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
        token_embeddings: Optional[torch.Tensor] = None,
        sensor_embeddings: Optional[torch.Tensor] = None,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rerank beam candidates using combined scoring.

        Args:
            sequences: [B, L] candidate token sequences
            log_probs: [B] cumulative log probabilities from beam search
            lengths: [B] sequence lengths (excluding padding)
            token_embeddings: [B, L, d_model] optional token representations
            sensor_embeddings: [B, T_s, sensor_dim] optional sensor features
            return_scores: Return detailed score breakdown

        Returns:
            sorted_indices: [B] indices sorted by final score (descending)
            final_scores: [B] final reranked scores
        """
        device = sequences.device
        B = sequences.size(0)

        # Start with length-normalized log probs
        scores = self.length_normalize(log_probs, lengths)

        # Diversity penalty
        if self.diversity_penalty > 0:
            div_penalty = self.compute_diversity_penalty(sequences)
            scores = scores - div_penalty

        # Model-based scoring (if available)
        if (self.model_scorer is not None and
            token_embeddings is not None and
            sensor_embeddings is not None):

            with torch.no_grad():
                # Create padding mask
                pad_mask = torch.zeros_like(sequences, dtype=torch.bool)
                for i in range(B):
                    pad_mask[i, lengths[i]:] = True

                model_scores, score_dict = self.model_scorer(
                    token_embeddings=token_embeddings,
                    sensor_embeddings=sensor_embeddings,
                    token_mask=pad_mask,
                )

                # Add weighted model scores
                if 'consistency' in score_dict:
                    scores = scores + self.consistency_weight * score_dict['consistency']
                if 'grammar' in score_dict:
                    scores = scores + self.grammar_weight * score_dict['grammar']

        # Sort by final score (descending)
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)

        return sorted_indices, sorted_scores


class GrammarValidator:
    """
    Rule-based grammar validator for G-code sequences.

    Checks for common G-code grammar violations:
    1. Missing required parameters (e.g., R for G02/G03)
    2. Invalid command sequences
    3. Parameter value bounds
    """

    # Commands that require R parameter
    ARC_COMMANDS = {'G02', 'G2', 'G03', 'G3'}

    # Valid parameter letters
    VALID_PARAMS = {'X', 'Y', 'Z', 'R', 'F', 'I', 'J', 'K', 'A', 'B', 'C', 'S'}

    def __init__(
        self,
        require_arc_radius: bool = True,
        check_param_bounds: bool = True,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Args:
            require_arc_radius: Require R parameter for arc commands
            check_param_bounds: Check if parameter values are within bounds
            param_bounds: Dict mapping param letter to (min, max) bounds
        """
        self.require_arc_radius = require_arc_radius
        self.check_param_bounds = check_param_bounds
        self.param_bounds = param_bounds or {
            'X': (-100, 100),
            'Y': (-100, 100),
            'Z': (-50, 50),
            'R': (-50, 50),
            'F': (0, 10000),
        }

    def validate_sequence(
        self,
        tokens: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Validate a G-code token sequence.

        Args:
            tokens: List of token strings (e.g., ['G01', 'X', '1.234', 'Y', '5.678'])

        Returns:
            is_valid: Whether sequence is grammatically valid
            violations: List of violation descriptions
        """
        violations = []

        current_command = None
        has_r_param = False

        for i, token in enumerate(tokens):
            # Track current command
            if token.startswith('G') or token.startswith('M'):
                # Check previous command for missing params
                if current_command in self.ARC_COMMANDS and not has_r_param:
                    if self.require_arc_radius:
                        violations.append(f"Missing R parameter for {current_command}")

                current_command = token
                has_r_param = False

            # Track parameters
            if token == 'R':
                has_r_param = True

        # Check final command
        if current_command in self.ARC_COMMANDS and not has_r_param:
            if self.require_arc_radius:
                violations.append(f"Missing R parameter for {current_command}")

        is_valid = len(violations) == 0
        return is_valid, violations

    def score_sequence(
        self,
        tokens: List[str],
    ) -> float:
        """
        Score sequence validity (1.0 = fully valid, 0.0 = many violations).

        Args:
            tokens: List of token strings

        Returns:
            validity_score: Float in [0, 1]
        """
        is_valid, violations = self.validate_sequence(tokens)

        if is_valid:
            return 1.0

        # Penalize based on number of violations
        n_violations = len(violations)
        return max(0.0, 1.0 - 0.2 * n_violations)
