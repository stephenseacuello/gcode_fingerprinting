"""
G-code Grammar Finite State Machine for constrained decoding.

Enforces valid G-code structure during generation by tracking state
and providing hard masks for invalid token types and parameters.

States:
    START: Beginning of line, expecting COMMAND
    AFTER_COMMAND: After command, expecting PARAM_LETTER or EOS
    AFTER_PARAM: After parameter letter, MUST have NUMERIC value
    AFTER_VALUE: After numeric value, expecting PARAM_LETTER or EOS
    END: Sequence complete
    ERROR: Invalid transition occurred

Transitions:
    START -> AFTER_COMMAND (on COMMAND)
    AFTER_COMMAND -> AFTER_PARAM (on PARAM_LETTER)
    AFTER_COMMAND -> END (on EOS)
    AFTER_PARAM -> AFTER_VALUE (on NUMERIC) [REQUIRED - no other transitions!]
    AFTER_VALUE -> AFTER_PARAM (on PARAM_LETTER)
    AFTER_VALUE -> END (on EOS)

Author: Claude Code
Date: December 2025
"""

from enum import Enum
from typing import Dict, Set, Optional, Tuple, List
import torch


class GCodeState(Enum):
    """FSM states for G-code grammar."""
    START = 0
    AFTER_COMMAND = 1
    AFTER_PARAM = 2
    AFTER_VALUE = 3
    END = 4
    ERROR = 5


# Token type constants (must match TokenDecomposer)
TYPE_SPECIAL = 0
TYPE_COMMAND = 1
TYPE_PARAMETER = 2
TYPE_NUMERIC = 3


# Command-specific parameter rules (RS-274D)
COMMAND_FORBIDDEN_PARAMS = {
    'G0': {'F'},           # Rapid has no feed rate
    'G1': {'R', 'I', 'J', 'K'},  # Linear interpolation has no arc params
}

COMMAND_REQUIRED_PARAMS = {
    'G2': {'R'},  # Arc needs R (or I/J/K, but R is more common)
    'G3': {'R'},
}

COMMAND_ENCOURAGED_PARAMS = {
    'G2': {'R', 'I', 'J', 'K', 'X', 'Y', 'Z'},
    'G3': {'R', 'I', 'J', 'K', 'X', 'Y', 'Z'},
    'G1': {'X', 'Y', 'Z', 'F'},
    'G0': {'X', 'Y', 'Z'},
}


class GCodeGrammarFSM:
    """
    Finite State Machine for G-code grammar enforcement.

    Provides hard masks for invalid predictions based on current state
    and grammar rules. Ensures generated G-code is always syntactically valid.

    Usage:
        fsm = GCodeGrammarFSM(decomposer)

        for step in range(max_len):
            # Get valid masks
            type_mask = fsm.get_valid_type_mask()
            param_mask = fsm.get_valid_param_mask()

            # Apply masks to logits (set invalid to -inf)
            type_logits[~type_mask] = float('-inf')
            param_logits[~param_mask] = float('-inf')

            # Sample/argmax to get prediction
            type_pred = type_logits.argmax()

            # Update FSM state
            fsm.transition(type_pred, token_info)
    """

    def __init__(
        self,
        decomposer,
        allow_modal_commands: bool = False,
    ):
        """
        Args:
            decomposer: TokenDecomposer instance for token info
            allow_modal_commands: If True, allows starting with parameters
                (for modal G-code where command is implicit from previous line)
        """
        self.decomposer = decomposer
        self.allow_modal_commands = allow_modal_commands

        # Current state
        self.state = GCodeState.START

        # Context tracking
        self.current_command: Optional[str] = None
        self.current_param: Optional[str] = None
        self.used_params: Set[str] = set()
        self.sequence_length: int = 0

        # Build param name lookup from decomposer
        self.param_tokens = decomposer.param_tokens
        self.command_tokens = decomposer.command_tokens
        self.n_param_types = decomposer.n_param_types
        self.n_commands = decomposer.n_commands

    def reset(self):
        """Reset FSM to initial state for new sequence."""
        self.state = GCodeState.START
        self.current_command = None
        self.current_param = None
        self.used_params = set()
        self.sequence_length = 0

    def get_valid_type_mask(self, device: str = 'cpu') -> torch.Tensor:
        """
        Get mask of valid token types for current state.

        Returns:
            [4] boolean tensor where True = valid type
            Index: 0=SPECIAL, 1=COMMAND, 2=PARAMETER, 3=NUMERIC
        """
        mask = torch.zeros(4, dtype=torch.bool, device=device)

        if self.state == GCodeState.START:
            # First token must be COMMAND (or PARAMETER if modal mode)
            mask[TYPE_COMMAND] = True
            if self.allow_modal_commands:
                mask[TYPE_PARAMETER] = True

        elif self.state == GCodeState.AFTER_COMMAND:
            # After command: PARAMETER or EOS (SPECIAL)
            mask[TYPE_PARAMETER] = True
            mask[TYPE_SPECIAL] = True  # Allow EOS

        elif self.state == GCodeState.AFTER_PARAM:
            # After parameter letter: MUST be NUMERIC
            # This is the critical constraint - no other options!
            mask[TYPE_NUMERIC] = True

        elif self.state == GCodeState.AFTER_VALUE:
            # After value: PARAMETER (more params) or EOS
            mask[TYPE_PARAMETER] = True
            mask[TYPE_SPECIAL] = True  # Allow EOS

        elif self.state == GCodeState.END:
            # Already ended - only allow PAD/SPECIAL
            mask[TYPE_SPECIAL] = True

        elif self.state == GCodeState.ERROR:
            # Error state - allow anything to recover
            mask[:] = True

        return mask

    def get_valid_param_mask(self, device: str = 'cpu') -> torch.Tensor:
        """
        Get mask of valid parameter types based on:
        1. Single-letter rule: each param used only once per line
        2. Command-specific rules: G0 cannot have F, etc.

        Returns:
            [n_param_types] boolean tensor where True = valid param
        """
        mask = torch.ones(self.n_param_types, dtype=torch.bool, device=device)

        # 1. Exclude already-used parameters (single letter rule)
        for param_name in self.used_params:
            if param_name in self.decomposer.param2id:
                param_id = self.decomposer.param2id[param_name]
                if param_id < self.n_param_types:
                    mask[param_id] = False

        # 2. Apply command-specific forbidden params
        if self.current_command in COMMAND_FORBIDDEN_PARAMS:
            forbidden = COMMAND_FORBIDDEN_PARAMS[self.current_command]
            for param_name in forbidden:
                if param_name in self.decomposer.param2id:
                    param_id = self.decomposer.param2id[param_name]
                    if param_id < self.n_param_types:
                        mask[param_id] = False

        return mask

    def get_encouraged_param_mask(self, device: str = 'cpu') -> torch.Tensor:
        """
        Get mask of encouraged parameters for current command.
        Used for soft boosting, not hard masking.

        Returns:
            [n_param_types] boolean tensor where True = encouraged param
        """
        mask = torch.zeros(self.n_param_types, dtype=torch.bool, device=device)

        if self.current_command in COMMAND_ENCOURAGED_PARAMS:
            encouraged = COMMAND_ENCOURAGED_PARAMS[self.current_command]
            for param_name in encouraged:
                if param_name in self.decomposer.param2id:
                    param_id = self.decomposer.param2id[param_name]
                    if param_id < self.n_param_types:
                        # Only encourage if not already used
                        if param_name not in self.used_params:
                            mask[param_id] = True

        return mask

    def transition(
        self,
        token_type: int,
        command_id: Optional[int] = None,
        param_id: Optional[int] = None,
    ) -> bool:
        """
        Transition to next state based on predicted token type.

        Args:
            token_type: Predicted token type (0-3)
            command_id: Command ID if token_type is COMMAND
            param_id: Parameter ID if token_type is PARAMETER

        Returns:
            True if transition was valid, False if error
        """
        self.sequence_length += 1

        if self.state == GCodeState.START:
            if token_type == TYPE_COMMAND:
                self.state = GCodeState.AFTER_COMMAND
                if command_id is not None and command_id < len(self.command_tokens):
                    self.current_command = self.command_tokens[command_id]
                return True
            elif token_type == TYPE_PARAMETER and self.allow_modal_commands:
                self.state = GCodeState.AFTER_PARAM
                if param_id is not None and param_id < len(self.param_tokens):
                    param_name = self.param_tokens[param_id]
                    self.current_param = param_name
                    self.used_params.add(param_name)
                return True
            else:
                self.state = GCodeState.ERROR
                return False

        elif self.state == GCodeState.AFTER_COMMAND:
            if token_type == TYPE_PARAMETER:
                self.state = GCodeState.AFTER_PARAM
                if param_id is not None and param_id < len(self.param_tokens):
                    param_name = self.param_tokens[param_id]
                    self.current_param = param_name
                    self.used_params.add(param_name)
                return True
            elif token_type == TYPE_SPECIAL:
                self.state = GCodeState.END
                return True
            else:
                self.state = GCodeState.ERROR
                return False

        elif self.state == GCodeState.AFTER_PARAM:
            if token_type == TYPE_NUMERIC:
                self.state = GCodeState.AFTER_VALUE
                return True
            else:
                # Critical: PARAMETER MUST be followed by NUMERIC
                self.state = GCodeState.ERROR
                return False

        elif self.state == GCodeState.AFTER_VALUE:
            if token_type == TYPE_PARAMETER:
                self.state = GCodeState.AFTER_PARAM
                if param_id is not None and param_id < len(self.param_tokens):
                    param_name = self.param_tokens[param_id]
                    self.current_param = param_name
                    self.used_params.add(param_name)
                return True
            elif token_type == TYPE_SPECIAL:
                self.state = GCodeState.END
                return True
            else:
                self.state = GCodeState.ERROR
                return False

        elif self.state == GCodeState.END:
            # Already ended
            return token_type == TYPE_SPECIAL

        else:
            # ERROR state
            return False

    def is_finished(self) -> bool:
        """Check if sequence generation is complete."""
        return self.state == GCodeState.END

    def is_error(self) -> bool:
        """Check if FSM is in error state."""
        return self.state == GCodeState.ERROR

    def get_state_name(self) -> str:
        """Get human-readable state name."""
        return self.state.name

    def __repr__(self) -> str:
        return (
            f"GCodeGrammarFSM(state={self.state.name}, "
            f"cmd={self.current_command}, "
            f"used_params={self.used_params})"
        )


def apply_hard_grammar_masks(
    logits: Dict[str, torch.Tensor],
    fsm: GCodeGrammarFSM,
    boost_encouraged: bool = True,
    encourage_boost: float = 5.0,
) -> Dict[str, torch.Tensor]:
    """
    Apply hard grammar masks to logits (set invalid to -inf).

    Args:
        logits: Dictionary with 'type_logits', 'command_logits', 'param_type_logits'
        fsm: GCodeGrammarFSM instance tracking current state
        boost_encouraged: If True, boost encouraged params (soft, not hard)
        encourage_boost: Amount to boost encouraged params

    Returns:
        Modified logits dictionary with hard masks applied
    """
    device = next(iter(logits.values())).device
    constrained = {}

    # 1. Type logits: hard mask invalid types
    if 'type_logits' in logits:
        type_mask = fsm.get_valid_type_mask(device)  # [4]
        type_logits = logits['type_logits'].clone()

        # Handle different shapes: [B, 1, 4] or [B, 4] or [4]
        if type_logits.dim() == 3:
            type_logits[:, :, ~type_mask] = float('-inf')
        elif type_logits.dim() == 2:
            type_logits[:, ~type_mask] = float('-inf')
        else:
            type_logits[~type_mask] = float('-inf')

        constrained['type_logits'] = type_logits

    # 2. Param type logits: mask used/forbidden params
    if 'param_type_logits' in logits:
        param_mask = fsm.get_valid_param_mask(device)  # [n_params]
        param_logits = logits['param_type_logits'].clone()

        # Handle different shapes
        if param_logits.dim() == 3:
            # Expand mask for batch and sequence dims
            mask_expanded = param_mask.unsqueeze(0).unsqueeze(0)
            param_logits = param_logits.masked_fill(~mask_expanded, float('-inf'))
        elif param_logits.dim() == 2:
            mask_expanded = param_mask.unsqueeze(0)
            param_logits = param_logits.masked_fill(~mask_expanded, float('-inf'))
        else:
            param_logits[~param_mask] = float('-inf')

        # Optionally boost encouraged params (soft boost, not hard)
        if boost_encouraged:
            encouraged_mask = fsm.get_encouraged_param_mask(device)
            if param_logits.dim() == 3:
                mask_expanded = encouraged_mask.unsqueeze(0).unsqueeze(0)
                param_logits = param_logits + mask_expanded.float() * encourage_boost
            elif param_logits.dim() == 2:
                mask_expanded = encouraged_mask.unsqueeze(0)
                param_logits = param_logits + mask_expanded.float() * encourage_boost
            else:
                param_logits = param_logits + encouraged_mask.float() * encourage_boost

        constrained['param_type_logits'] = param_logits

    # 3. Copy other logits unchanged
    for key in ['command_logits', 'param_value_logits', 'digit_logits', 'sign_logits']:
        if key in logits:
            constrained[key] = logits[key]

    return constrained


class BatchGCodeGrammarFSM:
    """
    Batch version of GCodeGrammarFSM for parallel sequence generation.

    Maintains separate FSM state for each item in batch.
    """

    def __init__(
        self,
        batch_size: int,
        decomposer,
        allow_modal_commands: bool = False,
    ):
        """
        Args:
            batch_size: Number of sequences in batch
            decomposer: TokenDecomposer instance
            allow_modal_commands: If True, allows starting with parameters
        """
        self.batch_size = batch_size
        self.decomposer = decomposer

        # Create FSM for each batch item
        self.fsms = [
            GCodeGrammarFSM(decomposer, allow_modal_commands)
            for _ in range(batch_size)
        ]

    def reset(self):
        """Reset all FSMs."""
        for fsm in self.fsms:
            fsm.reset()

    def get_valid_type_masks(self, device: str = 'cpu') -> torch.Tensor:
        """
        Get type masks for all batch items.

        Returns:
            [B, 4] boolean tensor
        """
        masks = torch.stack([
            fsm.get_valid_type_mask(device)
            for fsm in self.fsms
        ])
        return masks

    def get_valid_param_masks(self, device: str = 'cpu') -> torch.Tensor:
        """
        Get param masks for all batch items.

        Returns:
            [B, n_param_types] boolean tensor
        """
        masks = torch.stack([
            fsm.get_valid_param_mask(device)
            for fsm in self.fsms
        ])
        return masks

    def transition_batch(
        self,
        token_types: torch.Tensor,
        command_ids: Optional[torch.Tensor] = None,
        param_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Transition all FSMs based on predictions.

        Args:
            token_types: [B] predicted token types
            command_ids: [B] command IDs (optional)
            param_ids: [B] param IDs (optional)

        Returns:
            [B] boolean tensor indicating valid transitions
        """
        valid = torch.zeros(self.batch_size, dtype=torch.bool)

        for i, fsm in enumerate(self.fsms):
            cmd_id = command_ids[i].item() if command_ids is not None else None
            param_id = param_ids[i].item() if param_ids is not None else None
            valid[i] = fsm.transition(
                token_types[i].item(),
                cmd_id,
                param_id,
            )

        return valid

    def is_finished(self) -> torch.Tensor:
        """Check which sequences are finished."""
        return torch.tensor([fsm.is_finished() for fsm in self.fsms])

    def is_error(self) -> torch.Tensor:
        """Check which sequences are in error state."""
        return torch.tensor([fsm.is_error() for fsm in self.fsms])


def apply_batch_grammar_masks(
    logits: Dict[str, torch.Tensor],
    batch_fsm: BatchGCodeGrammarFSM,
    boost_encouraged: bool = True,
    encourage_boost: float = 5.0,
) -> Dict[str, torch.Tensor]:
    """
    Apply hard grammar masks for batch generation.

    Args:
        logits: Dictionary with 'type_logits' [B, 1, 4], 'param_type_logits' [B, 1, n_params]
        batch_fsm: BatchGCodeGrammarFSM instance
        boost_encouraged: If True, boost encouraged params
        encourage_boost: Amount to boost

    Returns:
        Modified logits with hard masks applied
    """
    device = next(iter(logits.values())).device
    B = batch_fsm.batch_size
    constrained = {}

    # 1. Type logits
    if 'type_logits' in logits:
        type_masks = batch_fsm.get_valid_type_masks(device)  # [B, 4]
        type_logits = logits['type_logits'].clone()

        # Shape: [B, 1, 4] - expand mask
        if type_logits.dim() == 3:
            type_masks = type_masks.unsqueeze(1)  # [B, 1, 4]

        type_logits = type_logits.masked_fill(~type_masks, float('-inf'))
        constrained['type_logits'] = type_logits

    # 2. Param type logits
    if 'param_type_logits' in logits:
        param_masks = batch_fsm.get_valid_param_masks(device)  # [B, n_params]
        param_logits = logits['param_type_logits'].clone()

        # Shape: [B, 1, n_params] - expand mask
        if param_logits.dim() == 3:
            param_masks = param_masks.unsqueeze(1)  # [B, 1, n_params]

        param_logits = param_logits.masked_fill(~param_masks, float('-inf'))

        # Boost encouraged params
        if boost_encouraged:
            encouraged_masks = torch.stack([
                fsm.get_encouraged_param_mask(device)
                for fsm in batch_fsm.fsms
            ])  # [B, n_params]

            if param_logits.dim() == 3:
                encouraged_masks = encouraged_masks.unsqueeze(1)

            param_logits = param_logits + encouraged_masks.float() * encourage_boost

        constrained['param_type_logits'] = param_logits

    # 3. Copy other logits
    for key in ['command_logits', 'param_value_logits', 'digit_logits', 'sign_logits']:
        if key in logits:
            constrained[key] = logits[key]

    return constrained


# Test code
if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from miracle.dataset.target_utils import TokenDecomposer

    # Load decomposer
    vocab_path = Path(__file__).parent.parent.parent.parent / 'data' / 'vocabulary_2digit_hybrid.json'
    if not vocab_path.exists():
        vocab_path = Path(__file__).parent.parent.parent.parent / 'data' / 'gcode_vocab_v2.json'

    print(f"Loading vocab from: {vocab_path}")
    decomposer = TokenDecomposer(str(vocab_path))

    # Test FSM
    fsm = GCodeGrammarFSM(decomposer)
    print(f"\nInitial state: {fsm}")

    # Simulate: G0 X 1.5 Y 2.0 EOS
    print("\nSimulating: G0 X <value> Y <value> EOS")

    # Step 1: G0
    print(f"  Valid types: {fsm.get_valid_type_mask()}")
    fsm.transition(TYPE_COMMAND, command_id=0)  # G0
    print(f"  After G0: {fsm}")

    # Step 2: X
    print(f"  Valid types: {fsm.get_valid_type_mask()}")
    print(f"  Valid params: {fsm.get_valid_param_mask()}")
    x_id = decomposer.param2id.get('X', 0)
    fsm.transition(TYPE_PARAMETER, param_id=x_id)
    print(f"  After X: {fsm}")

    # Step 3: Numeric value
    print(f"  Valid types: {fsm.get_valid_type_mask()}")
    fsm.transition(TYPE_NUMERIC)
    print(f"  After value: {fsm}")

    # Step 4: Y
    print(f"  Valid types: {fsm.get_valid_type_mask()}")
    print(f"  Valid params: {fsm.get_valid_param_mask()}")  # X should be excluded
    y_id = decomposer.param2id.get('Y', 1)
    fsm.transition(TYPE_PARAMETER, param_id=y_id)
    print(f"  After Y: {fsm}")

    # Step 5: Numeric value
    fsm.transition(TYPE_NUMERIC)
    print(f"  After value: {fsm}")

    # Step 6: EOS
    print(f"  Valid types: {fsm.get_valid_type_mask()}")
    fsm.transition(TYPE_SPECIAL)
    print(f"  After EOS: {fsm}")
    print(f"  Finished: {fsm.is_finished()}")
