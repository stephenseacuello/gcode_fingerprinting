"""
G-code grammar constraint enforcement for training and inference.

This module implements hard constraints and soft penalties to ensure
generated G-code follows proper grammar rules based on RS-274D standard.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Set, List

from .modal_groups import (
    MODAL_GROUPS,
    M_MODAL_GROUPS,
    ALL_MODAL_GROUPS,
    COMMAND_PARAM_RULES,
    SINGLE_LETTER_PARAMS,
    get_modal_group,
    check_modal_conflict,
    is_arc_command,
    is_rapid_command,
    requires_feed_rate,
)


class GCodeGrammarConstraints:
    """
    Enforces G-code grammar rules during training and inference.

    RS-274D Standard Constraints:
    1. Modal group exclusion: Two G-codes from same modal group cannot appear together
    2. Arc parameters: G2/G3 MUST have R or I/J/K parameters
    3. Feed rate rules: G0 never has F; G1/G2/G3 require F
    4. Single letter rule: Only one word per parameter letter per line (except G/M)
    5. Command-parameter association: Each command has required/forbidden params
    6. Modal state: Bare coordinates require prior motion command
    7. Pattern constraints: Z-retract → XY-rapid → Z-plunge
    8. Spindle rules: M3/M4 require S parameter
    """

    def __init__(self, vocab, device='cpu', decomposer=None, allow_modal_commands=False):
        """
        Args:
            vocab: Vocabulary dictionary mapping tokens to IDs
            device: torch device
            decomposer: TokenDecomposer instance for type checking (optional)
            allow_modal_commands: If True, allows sequences to start with parameters
                                 instead of commands (modal G-code behavior)
        """
        self.vocab = vocab
        self.device = device
        self.decomposer = decomposer
        self.allow_modal_commands = allow_modal_commands

        # Extract token IDs for grammar rules
        self.command_ids = self._get_command_ids()
        self.param_ids = self._get_param_ids()
        self.motion_command_ids = {
            vocab.get('G0', -1),
            vocab.get('G1', -1),
            vocab.get('G2', -1),
            vocab.get('G3', -1),
        }
        self.arc_command_ids = {
            vocab.get('G2', -1),
            vocab.get('G3', -1),
        }
        self.rapid_command_id = vocab.get('G0', -1)
        self.linear_command_id = vocab.get('G1', -1)
        self.g2_id = vocab.get('G2', -1)
        self.g3_id = vocab.get('G3', -1)

        # Parameter type IDs
        self.x_id = vocab.get('X', -1)
        self.y_id = vocab.get('Y', -1)
        self.z_id = vocab.get('Z', -1)
        self.r_id = vocab.get('R', -1)
        self.f_id = vocab.get('F', -1)
        self.i_id = vocab.get('I', -1)
        self.j_id = vocab.get('J', -1)
        self.k_id = vocab.get('K', -1)

    def _get_command_ids(self):
        """Extract all command token IDs (G0, G1, M3, etc.)."""
        command_ids = set()
        for token, idx in self.vocab.items():
            if token.startswith('G') or token.startswith('M') or token.startswith('T'):
                if not token.startswith('NUM_'):  # Exclude numeric parameters
                    command_ids.add(idx)
        return command_ids

    def _get_param_ids(self):
        """Extract all parameter type IDs (X, Y, Z, F, R, etc.)."""
        param_ids = set()
        params = ['X', 'Y', 'Z', 'F', 'R', 'S', 'I', 'J', 'K', 'P', 'Q', 'A', 'B', 'C']
        for p in params:
            if p in self.vocab:
                param_ids.add(self.vocab[p])
        return param_ids

    def compute_constraint_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        current_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute soft constraint penalty losses.

        Args:
            predictions: Model predictions (logits)
            targets: Ground truth targets
            current_tokens: Current token sequence [B, T]

        Returns:
            Dictionary of constraint losses
        """
        B, T = current_tokens.shape
        device = current_tokens.device

        losses = {}

        # 1. Arc radius constraint: G2/G3 should be followed by R parameter
        arc_radius_loss = self._arc_radius_constraint(
            predictions, targets, current_tokens
        )
        losses['arc_radius'] = arc_radius_loss

        # 2. Rapid feed constraint: G0 should NOT be followed by F parameter
        rapid_feed_loss = self._rapid_feed_constraint(
            predictions, targets, current_tokens
        )
        losses['rapid_feed'] = rapid_feed_loss

        # 3. Modal state constraint: Bare coordinates need prior motion command
        modal_state_loss = self._modal_state_constraint(
            predictions, targets, current_tokens
        )
        losses['modal_state'] = modal_state_loss

        # 4. Alternating arc constraint: G2/G3 alternate in pocket/face operations
        alternating_arc_loss = self._alternating_arc_constraint(
            predictions, targets, current_tokens
        )
        losses['alternating_arc'] = alternating_arc_loss

        # 5. Linear cutting feed constraint: G1 after G0 should have F
        linear_cutting_feed_loss = self._linear_cutting_feed_constraint(
            predictions, targets, current_tokens
        )
        losses['linear_cutting_feed'] = linear_cutting_feed_loss

        # 6. Z-retract pattern constraint: G0 Z+ → XY move → Z- plunge
        z_retract_pattern_loss = self._z_retract_pattern_constraint(
            predictions, targets, current_tokens
        )
        losses['z_retract_pattern'] = z_retract_pattern_loss

        # 7. Consistent radius constraint (placeholder for now)
        consistent_radius_loss = self._consistent_radius_constraint(
            predictions, targets, current_tokens
        )
        losses['consistent_radius'] = consistent_radius_loss

        # 8. Modal group constraint (RS-274D): No two commands from same modal group
        modal_group_loss = self._modal_group_constraint(
            predictions, targets, current_tokens
        )
        losses['modal_group'] = modal_group_loss

        # 9. Command-parameter association constraint
        cmd_param_loss = self._command_param_association_constraint(
            predictions, targets, current_tokens
        )
        losses['cmd_param_association'] = cmd_param_loss

        # 10. Single letter rule constraint
        single_letter_loss = self._single_letter_constraint(
            predictions, targets, current_tokens
        )
        losses['single_letter'] = single_letter_loss

        # Total constraint loss (weighted sum)
        total = (
            1.0 * losses['arc_radius'] +
            0.5 * losses['rapid_feed'] +
            0.3 * losses['modal_state'] +
            0.4 * losses['alternating_arc'] +
            0.3 * losses['linear_cutting_feed'] +
            0.2 * losses['z_retract_pattern'] +
            0.0 * losses['consistent_radius'] +  # Weight 0 for now (placeholder)
            0.6 * losses['modal_group'] +        # RS-274D modal group rule
            0.4 * losses['cmd_param_association'] +  # Command-param rules
            0.3 * losses['single_letter']        # Single letter rule
        )
        losses['total_constraint'] = total

        return losses

    def _arc_radius_constraint(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        current_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce: G2/G3 commands should be followed by R (or I/J/K) parameter.

        Implementation:
        - Detect positions where current token is G2 or G3
        - Penalize if next predicted param_type is NOT R/I/J/K
        """
        B, T = current_tokens.shape
        device = current_tokens.device

        # Find positions with G2/G3 commands
        is_arc_command = torch.zeros(B, T, dtype=torch.bool, device=device)
        for arc_id in self.arc_command_ids:
            if arc_id >= 0:
                is_arc_command |= (current_tokens == arc_id)

        # Shift to check next position
        is_arc_command_prev = F.pad(is_arc_command[:, :-1], (1, 0), value=False)

        if not is_arc_command_prev.any():
            return torch.tensor(0.0, device=device)

        # Get predicted param_type logits for positions after arc commands
        if 'param_type_logits' not in predictions:
            return torch.tensor(0.0, device=device)

        param_type_logits = predictions['param_type_logits']  # [B, T, n_param_types]

        # Create target distribution: high probability for R/I/J/K
        n_param_types = param_type_logits.size(-1)
        target_dist = torch.zeros(B, T, n_param_types, device=device)

        # Set high probability for R parameter
        if self.r_id >= 0 and self.r_id < n_param_types:
            target_dist[:, :, self.r_id] = 0.7

        # Also allow I/J/K (arc center offsets)
        for param_id in [self.i_id, self.j_id, self.k_id]:
            if param_id >= 0 and param_id < n_param_types:
                target_dist[:, :, param_id] = 0.1

        # Normalize
        target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-8)

        # KL divergence loss on positions after arc commands
        pred_dist = F.softmax(param_type_logits, dim=-1)
        kl_loss = F.kl_div(
            pred_dist[is_arc_command_prev].log(),
            target_dist[is_arc_command_prev],
            reduction='batchmean'
        )

        return kl_loss

    def _rapid_feed_constraint(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        current_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce: G0 (rapid) commands should NOT have F (feed rate) parameter.

        Implementation:
        - Detect positions where current token is G0
        - Penalize if next predicted param_type is F
        """
        B, T = current_tokens.shape
        device = current_tokens.device

        # Find positions with G0 commands
        is_rapid = (current_tokens == self.rapid_command_id)
        is_rapid_prev = F.pad(is_rapid[:, :-1], (1, 0), value=False)

        if not is_rapid_prev.any():
            return torch.tensor(0.0, device=device)

        if 'param_type_logits' not in predictions:
            return torch.tensor(0.0, device=device)

        param_type_logits = predictions['param_type_logits']  # [B, T, n_param_types]
        n_param_types = param_type_logits.size(-1)

        # Penalize high probability for F parameter after G0
        if self.f_id >= 0 and self.f_id < n_param_types:
            f_probs = F.softmax(param_type_logits, dim=-1)[:, :, self.f_id]
            penalty = f_probs[is_rapid_prev].mean()
            return penalty

        return torch.tensor(0.0, device=device)

    def _modal_state_constraint(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        current_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce: Bare coordinate parameters (X/Y/Z without command) require
        a prior motion command in the sequence.

        This is a softer constraint - we just encourage having motion commands
        in the sequence before parameters.
        """
        # For now, implement a simple version:
        # Encourage type_logits to predict COMMAND type at sequence start
        if 'type_logits' not in predictions:
            return torch.tensor(0.0, device=current_tokens.device)

        type_logits = predictions['type_logits']  # [B, T, 4]
        # Type 1 = COMMAND

        # Encourage COMMAND type at early positions (first 25% of sequence)
        B, T, _ = type_logits.shape
        early_pos = T // 4

        early_type_logits = type_logits[:, :early_pos, :]
        command_type_probs = F.softmax(early_type_logits, dim=-1)[:, :, 1]

        # Reward high command probability early in sequence
        # Use negative to turn reward into loss minimization
        reward = command_type_probs.mean()
        loss = -reward

        return loss.clamp(min=0.0)  # Don't penalize if already good

    def _alternating_arc_constraint(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        current_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce: G2/G3 commands often alternate in face/pocket operations.

        Pattern: G2 → ... → G3 → ... → G2 (alternating arcs)
        This is characteristic of face and pocket milling operations.
        """
        B, T = current_tokens.shape
        device = current_tokens.device

        # Find positions with G2 commands
        is_g2 = (current_tokens == self.g2_id)
        # Find positions with G3 commands
        is_g3 = (current_tokens == self.g3_id)

        if not (is_g2.any() or is_g3.any()):
            return torch.tensor(0.0, device=device)

        if 'command_logits' not in predictions:
            return torch.tensor(0.0, device=device)

        command_logits = predictions['command_logits']  # [B, T, n_commands]
        n_commands = command_logits.size(-1)

        # Look for sequences with multiple arcs
        # If we see G2, encourage G3 in next few positions (within 5-10 tokens)
        # If we see G3, encourage G2 in next few positions
        loss = torch.tensor(0.0, device=device)
        count = 0

        for b in range(B):
            for t in range(T - 10):
                if is_g2[b, t]:
                    # After G2, encourage G3 in next 5-10 positions
                    future_logits = command_logits[b, t+5:t+10, :]
                    if self.g3_id >= 0 and self.g3_id < n_commands and future_logits.size(0) > 0:
                        g3_probs = F.softmax(future_logits, dim=-1)[:, self.g3_id]
                        # Reward high G3 probability (negative loss)
                        loss -= g3_probs.mean() * 0.1
                        count += 1
                elif is_g3[b, t]:
                    # After G3, encourage G2 in next 5-10 positions
                    future_logits = command_logits[b, t+5:t+10, :]
                    if self.g2_id >= 0 and self.g2_id < n_commands and future_logits.size(0) > 0:
                        g2_probs = F.softmax(future_logits, dim=-1)[:, self.g2_id]
                        # Reward high G2 probability (negative loss)
                        loss -= g2_probs.mean() * 0.1
                        count += 1

        if count > 0:
            loss = loss / count

        return loss.clamp(min=-0.5, max=0.5)  # Bound the reward/penalty

    def _linear_cutting_feed_constraint(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        current_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce: G1 (linear interpolation) commands should often have F (feed rate).

        When transitioning from G0 (rapid) to G1 (cutting), feed rate should be specified.
        """
        B, T = current_tokens.shape
        device = current_tokens.device

        # Find positions with G1 commands
        is_g1 = (current_tokens == self.linear_command_id)
        # Find positions with G0 commands
        is_g0 = (current_tokens == self.rapid_command_id)

        if not (is_g1.any() and is_g0.any()):
            return torch.tensor(0.0, device=device)

        if 'param_type_logits' not in predictions:
            return torch.tensor(0.0, device=device)

        param_type_logits = predictions['param_type_logits']  # [B, T, n_param_types]
        n_param_types = param_type_logits.size(-1)

        # Find G1 commands that come after G0 (transition to cutting)
        is_g1_after_g0 = torch.zeros(B, T, dtype=torch.bool, device=device)
        for b in range(B):
            for t in range(1, T):
                # Check if current is G1 and any previous position (within window) was G0
                if is_g1[b, t]:
                    # Look back up to 5 positions for G0
                    if is_g0[b, max(0, t-5):t].any():
                        is_g1_after_g0[b, t] = True

        if not is_g1_after_g0.any():
            return torch.tensor(0.0, device=device)

        # Shift to check next position after G1
        is_g1_after_g0_prev = F.pad(is_g1_after_g0[:, :-1], (1, 0), value=False)

        if self.f_id >= 0 and self.f_id < n_param_types:
            # Encourage F parameter after G1 (when transitioning from rapid)
            f_probs = F.softmax(param_type_logits, dim=-1)[:, :, self.f_id]
            reward = f_probs[is_g1_after_g0_prev].mean() if is_g1_after_g0_prev.any() else 0.0
            # Negative loss = reward
            return -reward.clamp(max=0.3) if isinstance(reward, torch.Tensor) else torch.tensor(0.0, device=device)

        return torch.tensor(0.0, device=device)

    def _z_retract_pattern_constraint(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        current_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce: Z-retract pattern → XY rapid → Z-plunge pattern.

        Pattern observed in manufacturing:
        1. G0 Z+ (retract to safe height)
        2. G0 X Y (rapid to new position)
        3. G1 Z- F (plunge to cutting depth with feed rate)
        """
        B, T = current_tokens.shape
        device = current_tokens.device

        # This is a complex multi-token pattern
        # For now, implement a simpler version:
        # After G0 with Z parameter, encourage XY parameters in next few positions

        is_g0 = (current_tokens == self.rapid_command_id)

        if not is_g0.any():
            return torch.tensor(0.0, device=device)

        if 'param_type_logits' not in predictions:
            return torch.tensor(0.0, device=device)

        param_type_logits = predictions['param_type_logits']
        n_param_types = param_type_logits.size(-1)

        # Find G0 commands and encourage XY parameters nearby
        # This encourages the typical retract→move→plunge pattern
        loss = torch.tensor(0.0, device=device)
        count = 0

        for b in range(B):
            for t in range(T - 5):
                if is_g0[b, t]:
                    # Encourage X or Y parameters in next 2-4 positions
                    future_logits = param_type_logits[b, t+2:t+5, :]
                    if future_logits.size(0) > 0:
                        probs = F.softmax(future_logits, dim=-1)
                        # Encourage X or Y
                        x_y_prob = 0.0
                        if self.x_id >= 0 and self.x_id < n_param_types:
                            x_y_prob += probs[:, self.x_id].mean()
                        if self.y_id >= 0 and self.y_id < n_param_types:
                            x_y_prob += probs[:, self.y_id].mean()
                        # Reward (negative loss)
                        loss -= x_y_prob * 0.1
                        count += 1

        if count > 0:
            loss = loss / count

        return loss.clamp(min=-0.3, max=0.3)

    def _consistent_radius_constraint(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        current_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce: R (radius) values should be consistent within a sequence.

        Observation: In milling operations, the same tool is used for multiple
        arcs, so R values (tool radius) tend to repeat.
        """
        # This constraint is harder to implement without access to actual numeric values
        # For now, just encourage R parameter to appear when we have arc commands
        # (this is partially covered by arc_radius_constraint)

        return torch.tensor(0.0, device=current_tokens.device)

    def _modal_group_constraint(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        current_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce RS-274D modal group rule: Two commands from the same modal
        group cannot appear close together in the sequence.

        Modal groups (e.g., G0/G1/G2/G3 are all in motion group):
        - Only one command from each group can be "active"
        - Penalize if we predict a command from the same modal group
          as a recent command in the context window
        """
        B, T = current_tokens.shape
        device = current_tokens.device

        if 'command_logits' not in predictions:
            return torch.tensor(0.0, device=device)

        command_logits = predictions['command_logits']  # [B, T, n_commands]
        n_commands = command_logits.size(-1)

        # Build mapping of token IDs to modal groups
        motion_ids = set()
        for cmd in ['G0', 'G1', 'G2', 'G3']:
            cmd_id = self.vocab.get(cmd, -1)
            if cmd_id >= 0:
                motion_ids.add(cmd_id)

        if not motion_ids:
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=device)
        count = 0

        # For each position, check if a motion command exists in recent context
        # If so, penalize predictions of OTHER motion commands
        context_window = 5

        for b in range(B):
            for t in range(context_window, T):
                # Get recent commands in context window
                recent_tokens = current_tokens[b, t-context_window:t]
                recent_motion = set()
                for token in recent_tokens:
                    if token.item() in motion_ids:
                        recent_motion.add(token.item())

                if recent_motion:
                    # We have a motion command in context
                    # Penalize prediction of different motion commands
                    pred_probs = F.softmax(command_logits[b, t, :], dim=-1)

                    for motion_id in motion_ids:
                        if motion_id not in recent_motion and motion_id < n_commands:
                            # Penalize this motion command
                            loss += pred_probs[motion_id] * 0.5
                            count += 1

        if count > 0:
            loss = loss / count

        return loss.clamp(max=0.5)

    def _command_param_association_constraint(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        current_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce command-parameter association rules from RS-274D.

        Rules:
        - G0 cannot have F parameter (rapid has no feed rate)
        - G2/G3 must have R or I/J/K
        - M3/M4 should have S parameter
        """
        B, T = current_tokens.shape
        device = current_tokens.device

        if 'param_type_logits' not in predictions:
            return torch.tensor(0.0, device=device)

        param_type_logits = predictions['param_type_logits']
        n_param_types = param_type_logits.size(-1)

        loss = torch.tensor(0.0, device=device)
        count = 0

        # Get command IDs
        g0_id = self.vocab.get('G0', -1)
        g2_id = self.vocab.get('G2', -1)
        g3_id = self.vocab.get('G3', -1)
        m3_id = self.vocab.get('M3', -1)
        m4_id = self.vocab.get('M4', -1)

        # Get param IDs
        f_id = self.vocab.get('F', -1)
        r_id = self.vocab.get('R', -1)
        s_id = self.vocab.get('S', -1)
        i_id = self.vocab.get('I', -1)
        j_id = self.vocab.get('J', -1)
        k_id = self.vocab.get('K', -1)

        for b in range(B):
            for t in range(1, T):
                prev_token = current_tokens[b, t-1].item()
                pred_probs = F.softmax(param_type_logits[b, t, :], dim=-1)

                # Rule 1: G0 -> no F
                if prev_token == g0_id and f_id >= 0 and f_id < n_param_types:
                    loss += pred_probs[f_id]
                    count += 1

                # Rule 2: G2/G3 -> encourage R or I/J/K
                if prev_token in [g2_id, g3_id]:
                    arc_param_prob = 0.0
                    if r_id >= 0 and r_id < n_param_types:
                        arc_param_prob += pred_probs[r_id]
                    if i_id >= 0 and i_id < n_param_types:
                        arc_param_prob += pred_probs[i_id]
                    if j_id >= 0 and j_id < n_param_types:
                        arc_param_prob += pred_probs[j_id]
                    if k_id >= 0 and k_id < n_param_types:
                        arc_param_prob += pred_probs[k_id]
                    # Reward having arc params (negative loss)
                    loss -= arc_param_prob * 0.3
                    count += 1

                # Rule 3: M3/M4 -> encourage S
                if prev_token in [m3_id, m4_id] and s_id >= 0 and s_id < n_param_types:
                    # Reward having S parameter
                    loss -= pred_probs[s_id] * 0.2
                    count += 1

        if count > 0:
            loss = loss / count

        return loss.clamp(min=-0.3, max=0.3)

    def _single_letter_constraint(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        current_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce RS-274D single letter rule: Only one word per parameter
        letter can appear on a line (except G and M which can appear multiple times).

        This is a soft constraint that penalizes repeated parameter letters
        within a context window (approximating a "line").
        """
        B, T = current_tokens.shape
        device = current_tokens.device

        if 'param_type_logits' not in predictions:
            return torch.tensor(0.0, device=device)

        param_type_logits = predictions['param_type_logits']
        n_param_types = param_type_logits.size(-1)

        # Build param letter to ID mapping
        param_letter_ids: Dict[str, int] = {}
        for letter in ['X', 'Y', 'Z', 'F', 'R', 'S', 'I', 'J', 'K', 'P', 'Q']:
            param_id = self.vocab.get(letter, -1)
            if param_id >= 0:
                param_letter_ids[letter] = param_id

        if not param_letter_ids:
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=device)
        count = 0

        # Context window approximating a G-code line
        line_window = 8

        for b in range(B):
            for t in range(line_window, T):
                # Check which param letters appeared in recent context
                recent_params = set()
                for i in range(t - line_window, t):
                    token_id = current_tokens[b, i].item()
                    for letter, param_id in param_letter_ids.items():
                        if token_id == param_id:
                            recent_params.add(letter)

                if recent_params:
                    # Penalize predicting already-used parameter letters
                    pred_probs = F.softmax(param_type_logits[b, t, :], dim=-1)

                    for letter in recent_params:
                        param_id = param_letter_ids[letter]
                        if param_id < n_param_types:
                            loss += pred_probs[param_id] * 0.5
                            count += 1

        if count > 0:
            loss = loss / count

        return loss.clamp(max=0.3)

    def apply_inference_constraints(
        self,
        logits: Dict[str, torch.Tensor],
        current_sequence: torch.Tensor,
        step: int,
        modal_command: str = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply hard constraints during inference by masking invalid predictions.

        Args:
            logits: Model output logits
            current_sequence: Current generated sequence [B, step]
            step: Current generation step
            modal_command: Optional modal command context (e.g., 'G1' if previous command was G1)
                          Used when allow_modal_commands=True to allow parameter-only sequences

        Returns:
            Modified logits with invalid options masked out
        """
        # Make a copy to avoid modifying original
        constrained_logits = {k: v.clone() for k, v in logits.items()}

        B = current_sequence.size(0)
        device = current_sequence.device

        # First token handling: Allow COMMAND or PARAMETER (if modal mode enabled)
        if step == 0:
            if 'type_logits' in constrained_logits:
                type_logits = constrained_logits['type_logits']  # [B, 1, 4]

                # Modal mode: Allow PARAMETER tokens if we have a modal command context
                if self.allow_modal_commands and modal_command is not None:
                    # Allow both COMMAND and PARAMETER types
                    type_logits[:, :, 0] -= 100.0  # Suppress TYPE_SPECIAL (PAD, BOS, EOS)
                    type_logits[:, :, 3] -= 100.0  # Suppress TYPE_NUMERIC (must start with param letter)
                    # Boost TYPE_COMMAND and TYPE_PARAMETER equally
                    type_logits[:, :, 1] += 5.0   # TYPE_COMMAND
                    type_logits[:, :, 2] += 5.0   # TYPE_PARAMETER
                else:
                    # Standard mode: First token MUST be a command (type_id=1)
                    # G-code sequences always start with a command (G0, G1, M3, etc.)
                    type_logits[:, :, 0] -= 100.0  # Suppress TYPE_SPECIAL (PAD, BOS, EOS)
                    type_logits[:, :, 2] -= 100.0  # Suppress TYPE_PARAMETER (X, Y, Z)
                    type_logits[:, :, 3] -= 100.0  # Suppress TYPE_NUMERIC (NUM_X_2)
                    # Boost TYPE_COMMAND (G0, G1, M3, etc.)
                    type_logits[:, :, 1] += 10.0

                constrained_logits['type_logits'] = type_logits
            return constrained_logits

        # Get last token
        last_token = current_sequence[:, -1]

        # CRITICAL: Enforce proper type transitions based on G-code grammar
        # After COMMAND → must be PARAMETER (not another COMMAND!)
        # After PARAMETER → must be NUMERIC (value for that parameter)
        if 'type_logits' in constrained_logits and hasattr(self, 'decomposer'):
            type_logits = constrained_logits['type_logits']  # [B, 1, 4]

            for b in range(B):
                last_token_id = last_token[b].item()

                # Get type of last token
                last_type, _, _, _ = self.decomposer.decompose_token(last_token_id)

                # Rule 1: After COMMAND (type=1) → must be PARAMETER (type=2) or SPECIAL (type=0 for EOS)
                if last_type == 1:  # TYPE_COMMAND
                    type_logits[b, :, 1] -= 100.0  # Suppress COMMAND
                    type_logits[b, :, 3] -= 100.0  # Suppress NUMERIC
                    type_logits[b, :, 2] += 10.0   # Boost PARAMETER

                # Rule 2: After PARAMETER (type=2) → must be NUMERIC (type=3)
                # CRITICAL: Parameters MUST have values! Suppress EOS after parameter letter
                elif last_type == 2:  # TYPE_PARAMETER
                    type_logits[b, :, 0] -= 1000.0  # VERY strong suppression of SPECIAL (no stopping mid-parameter!)
                    type_logits[b, :, 1] -= 1000.0  # Suppress COMMAND
                    type_logits[b, :, 2] -= 1000.0  # Suppress PARAMETER
                    type_logits[b, :, 3] += 50.0    # Very strong boost for NUMERIC

                # Rule 3: After NUMERIC (type=3) → can be PARAMETER (more params) or SPECIAL (EOS)
                elif last_type == 3:  # TYPE_NUMERIC
                    type_logits[b, :, 1] -= 100.0  # Suppress COMMAND (no multi-command lines)
                    type_logits[b, :, 3] -= 100.0  # Suppress NUMERIC (need PARAMETER first)
                    type_logits[b, :, 2] += 5.0    # Boost PARAMETER
                    type_logits[b, :, 0] += 2.0    # Allow SPECIAL (EOS)

            constrained_logits['type_logits'] = type_logits

        # 1. If last token was G2/G3, strongly bias toward R parameter
        if 'param_type_logits' in constrained_logits:
            is_arc = torch.zeros(B, dtype=torch.bool, device=device)
            for arc_id in self.arc_command_ids:
                if arc_id >= 0:
                    is_arc |= (last_token == arc_id)

            if is_arc.any():
                # Boost R parameter probability
                param_type_logits = constrained_logits['param_type_logits']  # [B, 1, n_param_types]
                if self.r_id >= 0 and self.r_id < param_type_logits.size(-1):
                    # Add large positive bias to R parameter
                    param_type_logits[is_arc, :, self.r_id] += 10.0

                constrained_logits['param_type_logits'] = param_type_logits

        # 2. If last token was G0, suppress F parameter
        if 'param_type_logits' in constrained_logits:
            is_rapid = (last_token == self.rapid_command_id)

            if is_rapid.any():
                param_type_logits = constrained_logits['param_type_logits']
                if self.f_id >= 0 and self.f_id < param_type_logits.size(-1):
                    # Add large negative bias to F parameter (effectively mask it)
                    param_type_logits[is_rapid, :, self.f_id] -= 100.0

                constrained_logits['param_type_logits'] = param_type_logits

        # 3. CRITICAL: Prevent parameter repetition within the same line
        # Each parameter (X, Y, Z, etc.) should appear at most once per G-code line
        if 'param_type_logits' in constrained_logits and hasattr(self, 'decomposer') and step > 1:
            param_type_logits = constrained_logits['param_type_logits']  # [B, 1, n_param_types]

            for b in range(B):
                # Look back from current position to find the last COMMAND token
                used_params = set()
                for i in range(current_sequence.size(1) - 1, -1, -1):
                    token_id = current_sequence[b, i].item()
                    token_type, _, param_type_id, _ = self.decomposer.decompose_token(token_id)

                    # If we hit a COMMAND, stop looking back (start of current line)
                    if token_type == 1:  # TYPE_COMMAND
                        break

                    # If this is a PARAMETER token, mark it as used
                    if token_type == 2:  # TYPE_PARAMETER
                        used_params.add(param_type_id)

                # Suppress all used parameters
                for param_id in used_params:
                    if param_id < param_type_logits.size(-1):
                        param_type_logits[b, :, param_id] -= 100.0  # Strongly suppress

            constrained_logits['param_type_logits'] = param_type_logits

        return constrained_logits

    def validate_sequence(self, sequence: torch.Tensor, vocab_inv: Dict[int, str]) -> Dict[str, int]:
        """
        Validate a generated sequence and count grammar violations.

        Args:
            sequence: Generated token IDs [T]
            vocab_inv: Inverse vocabulary {id: token}

        Returns:
            Dictionary with violation counts
        """
        violations = {
            'arc_without_radius': 0,
            'rapid_with_feed': 0,
            'invalid_ordering': 0,
            'total': 0,
        }

        T = sequence.shape[0]

        for i in range(T - 1):
            curr_token = sequence[i].item()
            next_token = sequence[i + 1].item()

            # Check arc radius constraint
            if curr_token in self.arc_command_ids:
                # Next few tokens should include R parameter
                found_r = False
                for j in range(i + 1, min(i + 5, T)):
                    if sequence[j].item() == self.r_id:
                        found_r = True
                        break
                    # Also accept I/J/K
                    if sequence[j].item() in [self.i_id, self.j_id, self.k_id]:
                        found_r = True
                        break

                if not found_r:
                    violations['arc_without_radius'] += 1
                    violations['total'] += 1

            # Check rapid feed constraint
            if curr_token == self.rapid_command_id:
                # Check next few tokens for F parameter
                for j in range(i + 1, min(i + 5, T)):
                    if sequence[j].item() == self.f_id:
                        violations['rapid_with_feed'] += 1
                        violations['total'] += 1
                        break

        return violations


def add_grammar_constraints_to_loss(
    loss_fn,
    grammar_constraints: GCodeGrammarConstraints,
    constraint_weight: float = 0.1,
):
    """
    Wrap a loss function to include grammar constraints.

    Args:
        loss_fn: Original MultiHeadLoss instance
        grammar_constraints: GCodeGrammarConstraints instance
        constraint_weight: Weight for constraint loss

    Returns:
        Modified loss function
    """
    original_forward = loss_fn.forward

    def forward_with_constraints(logits, targets, pad_mask=None, current_tokens=None):
        # Compute original loss
        total_loss, loss_dict = original_forward(logits, targets, pad_mask)

        # Compute constraint losses if current_tokens provided
        if current_tokens is not None and grammar_constraints is not None:
            constraint_losses = grammar_constraints.compute_constraint_losses(
                logits, targets, current_tokens
            )

            # Add constraint loss to total
            total_loss = total_loss + constraint_weight * constraint_losses['total_constraint']

            # Add to loss_dict for logging
            for key, value in constraint_losses.items():
                loss_dict[f'constraint_{key}'] = value.item()

        return total_loss, loss_dict

    loss_fn.forward = forward_with_constraints
    return loss_fn
