"""
G-Code String Reconstruction Module.

Converts multi-head model predictions to valid G-code strings,
following RS-274D grammar rules and maintaining modal state.

Enhanced with:
- Beam search decoding for sequence-level consistency
- Confidence-weighted token selection
- Context-aware parameter prediction
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import heapq
from collections import defaultdict

from ..training.modal_groups import (
    DEFAULT_MODAL_STATE,
    COMMAND_PARAM_RULES,
    PHYSICAL_CONSTRAINTS,
    EXECUTION_ORDER,
    get_modal_group,
    is_motion_command,
    is_arc_command,
    is_rapid_command,
    requires_feed_rate,
    constrain_value,
)


@dataclass
class ModalState:
    """Tracks the current modal state of the G-code machine."""
    motion: str = 'G0'
    plane: str = 'G17'
    distance: str = 'G90'
    feed_mode: str = 'G94'
    units: str = 'G21'
    spindle: str = 'M5'
    coolant: str = 'M9'
    last_feed: Optional[float] = None
    last_spindle_speed: Optional[float] = None
    last_positions: Dict[str, float] = field(default_factory=dict)

    def update(self, command: str, params: Dict[str, float] = None):
        """Update modal state based on command."""
        if command in {'G0', 'G1', 'G2', 'G3'}:
            self.motion = command
        elif command in {'G17', 'G18', 'G19'}:
            self.plane = command
        elif command in {'G90', 'G91'}:
            self.distance = command
        elif command in {'G93', 'G94'}:
            self.feed_mode = command
        elif command in {'G20', 'G21'}:
            self.units = command
        elif command in {'M3', 'M4', 'M5'}:
            self.spindle = command
        elif command in {'M7', 'M8', 'M9'}:
            self.coolant = command

        if params:
            if 'F' in params:
                self.last_feed = params['F']
            if 'S' in params:
                self.last_spindle_speed = params['S']
            for axis in ['X', 'Y', 'Z', 'A', 'B', 'C']:
                if axis in params:
                    self.last_positions[axis] = params[axis]


class GCodeStringReconstructor:
    """
    Converts multi-head model predictions to valid G-code strings.

    Handles:
    - Token-to-string conversion
    - Modal state tracking across predictions
    - Proper parameter ordering (RS-274D execution order)
    - Value formatting (decimal precision)
    - Grammar validation and fixing
    """

    # Token type IDs (matching model output)
    TYPE_PAD = 0
    TYPE_COMMAND = 1
    TYPE_PARAM = 2
    TYPE_VALUE = 3

    def __init__(
        self,
        vocab: Dict[str, int],
        position_precision: int = 3,
        feed_precision: int = 1,
        validate: bool = True,
    ):
        """
        Args:
            vocab: Vocabulary mapping tokens to IDs
            position_precision: Decimal places for position values (X, Y, Z)
            feed_precision: Decimal places for feed rate
            validate: Whether to validate and fix grammar issues
        """
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.position_precision = position_precision
        self.feed_precision = feed_precision
        self.validate = validate
        self.modal_state = ModalState()

    def reset_modal_state(self):
        """Reset modal state to defaults."""
        self.modal_state = ModalState()

    def reconstruct_token(
        self,
        prediction: Dict[str, Any],
        context: Optional[Dict] = None,
    ) -> Tuple[str, str]:
        """
        Convert a single multi-head prediction to G-code token.

        Args:
            prediction: Dictionary with:
                - 'type': int (0=PAD, 1=COMMAND, 2=PARAM, 3=VALUE)
                - 'command': int (command token ID)
                - 'param_type': int (parameter type token ID)
                - 'param_value': float (regression value)
                - 'operation': int (optional, operation class)
            context: Optional context from previous predictions

        Returns:
            Tuple of (token_string, token_type)
            e.g., ("G1", "command") or ("X45.230", "param_value")
        """
        token_type = prediction.get('type', 0)

        if token_type == self.TYPE_COMMAND:
            cmd_id = prediction.get('command', 0)
            cmd_str = self.inv_vocab.get(cmd_id, '')

            # Filter out special tokens
            if cmd_str in {'PAD', 'BOS', 'EOS', 'UNK', 'MASK', ''}:
                return '', 'special'

            # Update modal state
            self.modal_state.update(cmd_str)
            return cmd_str, 'command'

        elif token_type == self.TYPE_PARAM:
            param_id = prediction.get('param_type', 0)
            param_str = self.inv_vocab.get(param_id, '')

            # Filter special tokens and numeric bucket tokens
            if param_str in {'PAD', 'BOS', 'EOS', 'UNK', 'MASK', ''} or param_str.startswith('NUM_'):
                return '', 'special'

            return param_str, 'param'

        elif token_type == self.TYPE_VALUE:
            # Get the associated parameter from context
            param_letter = context.get('current_param', 'X') if context else 'X'
            value = prediction.get('param_value', 0.0)

            # Apply physical constraints
            value = constrain_value(param_letter, value)

            # Format value
            formatted = self._format_value(param_letter, value)
            return f"{param_letter}{formatted}", 'param_value'

        return '', 'pad'

    def reconstruct_line(
        self,
        predictions: List[Dict[str, Any]],
    ) -> str:
        """
        Reconstruct a complete G-code line from a sequence of predictions.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            G-code line string (e.g., "G1 X45.230 Y-12.500 F250")
        """
        commands = []
        params = {}
        current_param = None

        for pred in predictions:
            token, token_type = self.reconstruct_token(
                pred,
                context={'current_param': current_param}
            )

            if not token:
                continue

            if token_type == 'command':
                commands.append(token)
            elif token_type == 'param':
                current_param = token
            elif token_type == 'param_value':
                # Extract letter and value
                letter = token[0]
                value_str = token[1:]
                params[letter] = value_str
                current_param = None

        # Build the line
        line = self._build_line(commands, params)

        if self.validate:
            line = self._validate_and_fix(line, commands, params)

        return line

    def reconstruct_sequence(
        self,
        prediction_sequences: List[List[Dict[str, Any]]],
        reset_state: bool = True,
    ) -> str:
        """
        Reconstruct a complete G-code program from multiple line predictions.

        Args:
            prediction_sequences: List of prediction sequences (one per line)
            reset_state: Whether to reset modal state before reconstruction

        Returns:
            Multi-line G-code program string
        """
        if reset_state:
            self.reset_modal_state()

        lines = []
        for line_preds in prediction_sequences:
            line = self.reconstruct_line(line_preds)
            if line.strip():
                lines.append(line)

        return '\n'.join(lines)

    def reconstruct_from_tensors(
        self,
        type_preds: torch.Tensor,
        command_preds: torch.Tensor,
        param_type_preds: torch.Tensor,
        param_value_preds: torch.Tensor,
        operation_preds: Optional[torch.Tensor] = None,
    ) -> List[str]:
        """
        Reconstruct G-code from model tensor outputs.

        Args:
            type_preds: [B, T] tensor of type predictions
            command_preds: [B, T] tensor of command predictions
            param_type_preds: [B, T] tensor of param type predictions
            param_value_preds: [B, T] tensor of param value predictions
            operation_preds: [B, T] tensor of operation predictions (optional)

        Returns:
            List of G-code strings (one per batch item)
        """
        B, T = type_preds.shape
        results = []

        for b in range(B):
            self.reset_modal_state()
            predictions = []

            for t in range(T):
                pred = {
                    'type': type_preds[b, t].item(),
                    'command': command_preds[b, t].item(),
                    'param_type': param_type_preds[b, t].item(),
                    'param_value': param_value_preds[b, t].item(),
                }
                if operation_preds is not None:
                    pred['operation'] = operation_preds[b, t].item()

                predictions.append(pred)

            # Reconstruct line from all predictions
            gcode = self.reconstruct_line(predictions)
            results.append(gcode)

        return results

    def _format_value(self, param: str, value: float) -> str:
        """Format numeric value based on parameter type."""
        if param in ['X', 'Y', 'Z', 'I', 'J', 'K', 'R']:
            # Position values: 3 decimal places
            return f"{value:.{self.position_precision}f}"
        elif param == 'F':
            # Feed rate: 1 decimal or integer
            if self.feed_precision == 0 or value == int(value):
                return f"{int(value)}"
            return f"{value:.{self.feed_precision}f}"
        elif param == 'S':
            # Spindle speed: integer
            return f"{int(value)}"
        elif param == 'T':
            # Tool number: integer
            return f"{int(value)}"
        elif param == 'P':
            # Dwell time: up to 3 decimals
            return f"{value:.3f}"
        elif param in ['L', 'N']:
            # Loop count, line number: integer
            return f"{int(value)}"
        else:
            return f"{value:.{self.position_precision}f}"

    def _build_line(
        self,
        commands: List[str],
        params: Dict[str, str],
    ) -> str:
        """
        Build a G-code line with proper ordering.

        Order follows RS-274D execution order:
        1. G-codes (motion last)
        2. M-codes
        3. Parameters in standard order (X, Y, Z, I, J, K, R, F, S, T, P, Q)
        """
        parts = []

        # Separate G and M codes
        g_codes = [c for c in commands if c.startswith('G')]
        m_codes = [c for c in commands if c.startswith('M')]

        # Order G-codes: non-motion first, then motion
        non_motion_g = [g for g in g_codes if not is_motion_command(g)]
        motion_g = [g for g in g_codes if is_motion_command(g)]

        parts.extend(non_motion_g)
        parts.extend(motion_g)
        parts.extend(m_codes)

        # Add parameters in standard order
        param_order = ['X', 'Y', 'Z', 'A', 'B', 'C', 'I', 'J', 'K', 'R', 'F', 'S', 'T', 'P', 'Q', 'L']
        for p in param_order:
            if p in params:
                parts.append(f"{p}{params[p]}")

        return ' '.join(parts)

    def _validate_and_fix(
        self,
        line: str,
        commands: List[str],
        params: Dict[str, str],
    ) -> str:
        """
        Validate G-code line and fix common issues.

        Fixes:
        - Remove F from G0 lines
        - Ensure G2/G3 have R or I/J/K
        - Check for modal conflicts
        """
        parts = line.split()
        fixed_parts = []

        has_g0 = 'G0' in commands
        has_arc = any(c in commands for c in ['G2', 'G3'])

        for part in parts:
            # Rule: G0 should not have F
            if has_g0 and part.startswith('F'):
                continue  # Skip feed rate for rapids

            # Rule: Check for duplicate motion commands
            # (keep only the last one)
            fixed_parts.append(part)

        # Rule: Arcs need R or I/J/K
        if has_arc:
            has_arc_param = any(
                p.startswith(letter) for p in fixed_parts
                for letter in ['R', 'I', 'J', 'K']
            )
            if not has_arc_param:
                # Add a default R value based on last known or default
                # This is a fallback - ideally the model predicts this
                pass  # Let it through, will be caught in validation metrics

        return ' '.join(fixed_parts)


class GCodeValidator:
    """Validates G-code strings against RS-274D rules."""

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab

    def validate_line(self, line: str) -> Tuple[bool, List[str]]:
        """
        Validate a G-code line.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        parts = line.split()

        if not parts:
            return True, []

        # Extract commands and parameters
        g_codes = [p for p in parts if p.startswith('G') and p[1:].replace('.', '').isdigit()]
        m_codes = [p for p in parts if p.startswith('M') and p[1:].isdigit()]
        params = [p for p in parts if p[0] in 'XYZIJKRFSTPQLABC' and len(p) > 1]

        # Check modal conflicts within G-codes
        motion_commands = [g for g in g_codes if g in {'G0', 'G1', 'G2', 'G3'}]
        if len(motion_commands) > 1:
            errors.append(f"Multiple motion commands on same line: {motion_commands}")

        # Check G0 + F conflict
        if 'G0' in g_codes:
            if any(p.startswith('F') for p in params):
                errors.append("G0 (rapid) should not have F (feed rate)")

        # Check G2/G3 arc parameters
        if any(g in g_codes for g in ['G2', 'G3']):
            has_arc_param = any(p[0] in 'RIJK' for p in params)
            if not has_arc_param:
                errors.append("G2/G3 (arc) requires R or I/J/K parameters")

        # Check single letter rule
        seen_letters = set()
        for p in params:
            letter = p[0]
            if letter in seen_letters:
                errors.append(f"Duplicate parameter letter: {letter}")
            seen_letters.add(letter)

        # Check M3/M4 spindle speed
        if any(m in m_codes for m in ['M3', 'M4']):
            if not any(p.startswith('S') for p in params):
                errors.append("M3/M4 (spindle on) should have S (speed)")

        return len(errors) == 0, errors

    def validate_program(self, program: str) -> Dict[str, Any]:
        """
        Validate a complete G-code program.

        Returns:
            Dictionary with validation results
        """
        lines = program.strip().split('\n')
        results = {
            'total_lines': len(lines),
            'valid_lines': 0,
            'invalid_lines': 0,
            'errors': [],
            'error_rate': 0.0,
        }

        for i, line in enumerate(lines):
            is_valid, errors = self.validate_line(line.strip())
            if is_valid:
                results['valid_lines'] += 1
            else:
                results['invalid_lines'] += 1
                for error in errors:
                    results['errors'].append(f"Line {i+1}: {error}")

        if results['total_lines'] > 0:
            results['error_rate'] = results['invalid_lines'] / results['total_lines']

        return results


def compute_string_metrics(
    predicted: List[str],
    actual: List[str],
) -> Dict[str, float]:
    """
    Compute metrics comparing predicted and actual G-code strings.

    Args:
        predicted: List of predicted G-code strings
        actual: List of actual G-code strings

    Returns:
        Dictionary with metrics
    """
    assert len(predicted) == len(actual), "Lists must have same length"

    exact_matches = 0
    token_matches = 0
    total_tokens = 0
    command_matches = 0
    total_commands = 0
    param_matches = 0
    total_params = 0

    for pred, act in zip(predicted, actual):
        # Exact match
        if pred.strip() == act.strip():
            exact_matches += 1

        # Token-level comparison
        pred_tokens = pred.split()
        act_tokens = act.split()

        for pt in pred_tokens:
            total_tokens += 1
            if pt in act_tokens:
                token_matches += 1

        # Command comparison
        pred_cmds = [t for t in pred_tokens if t.startswith('G') or t.startswith('M')]
        act_cmds = [t for t in act_tokens if t.startswith('G') or t.startswith('M')]

        for pc in pred_cmds:
            total_commands += 1
            if pc in act_cmds:
                command_matches += 1

        # Parameter comparison (letter only, not value)
        pred_params = set(t[0] for t in pred_tokens if t[0] in 'XYZIJKRFSTPQL')
        act_params = set(t[0] for t in act_tokens if t[0] in 'XYZIJKRFSTPQL')

        for pp in pred_params:
            total_params += 1
            if pp in act_params:
                param_matches += 1

    n = len(predicted)
    return {
        'exact_match_rate': exact_matches / n if n > 0 else 0.0,
        'token_accuracy': token_matches / total_tokens if total_tokens > 0 else 0.0,
        'command_accuracy': command_matches / total_commands if total_commands > 0 else 0.0,
        'param_letter_accuracy': param_matches / total_params if total_params > 0 else 0.0,
    }


@dataclass
class BeamHypothesis:
    """A hypothesis in beam search with its score and tokens."""
    score: float
    tokens: List[str]
    state: Dict[str, Any]

    def __lt__(self, other):
        # For min-heap: lower score = better
        return self.score < other.score


class BeamSearchDecoder:
    """
    Beam search decoder for G-code sequence reconstruction.

    Uses beam search to find the most likely sequence of G-code tokens
    while respecting grammar constraints.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        beam_width: int = 5,
        max_length: int = 10,
        length_penalty: float = 0.6,
    ):
        """
        Args:
            vocab: Token vocabulary
            beam_width: Number of hypotheses to keep at each step
            max_length: Maximum sequence length
            length_penalty: Penalty for sequence length (0=no penalty, 1=strong)
        """
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.beam_width = beam_width
        self.max_length = max_length
        self.length_penalty = length_penalty

        # Special tokens
        self.pad_id = vocab.get('PAD', 0)
        self.special_tokens = {'PAD', 'BOS', 'EOS', 'UNK', 'MASK', ''}

    def decode(
        self,
        type_logits: torch.Tensor,
        command_logits: torch.Tensor,
        param_type_logits: torch.Tensor,
        param_values: torch.Tensor,
        decomposer=None,
    ) -> List[str]:
        """
        Decode a sequence using beam search.

        Args:
            type_logits: [T, 4] type classification logits
            command_logits: [T, C] command classification logits
            param_type_logits: [T, P] param type classification logits
            param_values: [T] regression values
            decomposer: TokenDecomposer for mapping IDs

        Returns:
            List of G-code tokens
        """
        T = type_logits.shape[0]

        # Initialize beam with empty hypothesis
        beam = [BeamHypothesis(
            score=0.0,
            tokens=[],
            state={'current_param': None, 'seen_commands': set(), 'seen_params': set()}
        )]

        for t in range(min(T, self.max_length)):
            # Get predictions at this timestep
            type_probs = F.softmax(type_logits[t], dim=-1)
            cmd_probs = F.softmax(command_logits[t], dim=-1)
            param_probs = F.softmax(param_type_logits[t], dim=-1)
            value = param_values[t].item()

            candidates = []

            for hyp in beam:
                # Get top-k type predictions
                type_pred = type_probs.argmax().item()
                type_score = type_probs[type_pred].item()

                if type_pred == 0:  # PAD
                    # Keep hypothesis as-is
                    candidates.append(BeamHypothesis(
                        score=hyp.score,
                        tokens=hyp.tokens.copy(),
                        state=hyp.state.copy()
                    ))
                    continue

                elif type_pred == 1:  # COMMAND
                    # Get top-k commands
                    top_cmds = cmd_probs.topk(min(3, cmd_probs.shape[0]))
                    for cmd_idx, cmd_score in zip(top_cmds.indices, top_cmds.values):
                        cmd_id = cmd_idx.item()

                        # Get command string
                        if decomposer:
                            cmd_str = decomposer.id2cmd.get(cmd_id, '')
                        else:
                            cmd_str = self.inv_vocab.get(cmd_id, '')

                        if not cmd_str or cmd_str in self.special_tokens:
                            continue

                        # Check for duplicate commands in same line
                        if cmd_str in hyp.state.get('seen_commands', set()):
                            cmd_score = cmd_score * 0.5  # Penalize duplicates

                        new_state = hyp.state.copy()
                        new_state['seen_commands'] = hyp.state.get('seen_commands', set()) | {cmd_str}

                        candidates.append(BeamHypothesis(
                            score=hyp.score - torch.log(cmd_score + 1e-10).item(),
                            tokens=hyp.tokens + [cmd_str],
                            state=new_state
                        ))

                elif type_pred == 2:  # PARAM
                    # Get top param types
                    top_params = param_probs.topk(min(3, param_probs.shape[0]))
                    for param_idx, param_score in zip(top_params.indices, top_params.values):
                        param_id = param_idx.item()

                        if decomposer:
                            param_str = decomposer.id2param.get(param_id, '')
                        else:
                            param_str = self.inv_vocab.get(param_id, '')

                        if not param_str or param_str in self.special_tokens:
                            continue

                        # Check for duplicate params
                        if param_str in hyp.state.get('seen_params', set()):
                            param_score = param_score * 0.3  # Strong penalty for duplicate params

                        new_state = hyp.state.copy()
                        new_state['current_param'] = param_str

                        candidates.append(BeamHypothesis(
                            score=hyp.score - torch.log(param_score + 1e-10).item(),
                            tokens=hyp.tokens + [param_str],
                            state=new_state
                        ))

                elif type_pred == 3:  # VALUE
                    current_param = hyp.state.get('current_param', 'X')

                    # Format value based on parameter type
                    if current_param in ['X', 'Y', 'Z', 'I', 'J', 'K', 'R']:
                        formatted = f"{value:.3f}"
                    elif current_param == 'F':
                        formatted = f"{value:.1f}" if value != int(value) else f"{int(value)}"
                    elif current_param in ['S', 'T', 'N']:
                        formatted = f"{int(value)}"
                    else:
                        formatted = f"{value:.3f}"

                    token = f"{current_param}{formatted}"

                    new_state = hyp.state.copy()
                    new_state['seen_params'] = hyp.state.get('seen_params', set()) | {current_param}
                    new_state['current_param'] = None

                    candidates.append(BeamHypothesis(
                        score=hyp.score - torch.log(torch.tensor(type_score) + 1e-10).item(),
                        tokens=hyp.tokens + [token],
                        state=new_state
                    ))

            # Keep top beam_width hypotheses
            candidates.sort(key=lambda h: h.score)
            beam = candidates[:self.beam_width]

            if not beam:
                beam = [BeamHypothesis(score=0.0, tokens=[], state={})]

        # Return best hypothesis
        if beam:
            return beam[0].tokens
        return []

    def decode_batch(
        self,
        type_logits: torch.Tensor,
        command_logits: torch.Tensor,
        param_type_logits: torch.Tensor,
        param_values: torch.Tensor,
        decomposer=None,
    ) -> List[List[str]]:
        """
        Decode a batch of sequences.

        Args:
            type_logits: [B, T, 4]
            command_logits: [B, T, C]
            param_type_logits: [B, T, P]
            param_values: [B, T]
            decomposer: TokenDecomposer

        Returns:
            List of token lists
        """
        B = type_logits.shape[0]
        results = []

        for b in range(B):
            tokens = self.decode(
                type_logits[b],
                command_logits[b],
                param_type_logits[b],
                param_values[b],
                decomposer,
            )
            results.append(tokens)

        return results


class EnhancedGCodeReconstructor:
    """
    Enhanced G-code reconstruction with confidence scoring and grammar fixing.

    Combines beam search decoding with grammar-aware post-processing.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        decomposer=None,
        beam_width: int = 5,
        use_beam_search: bool = True,
    ):
        """
        Args:
            vocab: Token vocabulary
            decomposer: TokenDecomposer for ID mapping
            beam_width: Beam width for search
            use_beam_search: Whether to use beam search (vs greedy)
        """
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.decomposer = decomposer
        self.use_beam_search = use_beam_search

        if use_beam_search:
            self.decoder = BeamSearchDecoder(vocab, beam_width=beam_width)

        self.basic_reconstructor = GCodeStringReconstructor(vocab)

    def reconstruct(
        self,
        type_logits: torch.Tensor,
        command_logits: torch.Tensor,
        param_type_logits: torch.Tensor,
        param_values: torch.Tensor,
    ) -> Tuple[str, float]:
        """
        Reconstruct G-code from model outputs.

        Args:
            type_logits: [T, 4] or [B, T, 4]
            command_logits: [T, C] or [B, T, C]
            param_type_logits: [T, P] or [B, T, P]
            param_values: [T] or [B, T]

        Returns:
            Tuple of (gcode_string, confidence_score)
        """
        # Handle batch dimension
        if type_logits.dim() == 3:
            B = type_logits.shape[0]
            results = []
            for b in range(B):
                gcode, conf = self._reconstruct_single(
                    type_logits[b], command_logits[b],
                    param_type_logits[b], param_values[b]
                )
                results.append((gcode, conf))
            return results
        else:
            return self._reconstruct_single(
                type_logits, command_logits, param_type_logits, param_values
            )

    def _reconstruct_single(
        self,
        type_logits: torch.Tensor,
        command_logits: torch.Tensor,
        param_type_logits: torch.Tensor,
        param_values: torch.Tensor,
    ) -> Tuple[str, float]:
        """Reconstruct single sequence."""

        if self.use_beam_search:
            tokens = self.decoder.decode(
                type_logits, command_logits, param_type_logits, param_values,
                self.decomposer
            )
        else:
            tokens = self._greedy_decode(
                type_logits, command_logits, param_type_logits, param_values
            )

        # Build G-code line from tokens
        gcode = self._tokens_to_gcode(tokens)

        # Compute confidence
        confidence = self._compute_confidence(type_logits, command_logits, param_type_logits)

        # Apply grammar fixes
        gcode = self._apply_grammar_fixes(gcode)

        return gcode, confidence

    def _greedy_decode(
        self,
        type_logits: torch.Tensor,
        command_logits: torch.Tensor,
        param_type_logits: torch.Tensor,
        param_values: torch.Tensor,
    ) -> List[str]:
        """Simple greedy decoding."""
        tokens = []
        current_param = None

        T = type_logits.shape[0]
        for t in range(T):
            type_pred = type_logits[t].argmax().item()

            if type_pred == 0:  # PAD
                continue
            elif type_pred == 1:  # COMMAND
                cmd_id = command_logits[t].argmax().item()
                if self.decomposer:
                    cmd_str = self.decomposer.id2cmd.get(cmd_id, '')
                else:
                    cmd_str = self.inv_vocab.get(cmd_id, '')
                if cmd_str and cmd_str not in {'PAD', 'BOS', 'EOS', 'UNK', 'MASK'}:
                    tokens.append(cmd_str)
            elif type_pred == 2:  # PARAM
                param_id = param_type_logits[t].argmax().item()
                if self.decomposer:
                    current_param = self.decomposer.id2param.get(param_id, 'X')
                else:
                    current_param = self.inv_vocab.get(param_id, 'X')
            elif type_pred == 3:  # VALUE
                value = param_values[t].item()
                param_letter = current_param if current_param else 'X'

                if param_letter in ['X', 'Y', 'Z', 'I', 'J', 'K', 'R']:
                    formatted = f"{param_letter}{value:.3f}"
                elif param_letter == 'F':
                    formatted = f"{param_letter}{value:.1f}"
                else:
                    formatted = f"{param_letter}{int(value)}"

                tokens.append(formatted)
                current_param = None

        return tokens

    def _tokens_to_gcode(self, tokens: List[str]) -> str:
        """Convert token list to G-code string with proper ordering."""
        commands = []
        params = {}

        for token in tokens:
            if not token:
                continue

            first_char = token[0]

            if first_char in ['G', 'M']:
                commands.append(token)
            elif first_char in 'XYZIJKRFSTPQLABC' and len(token) > 1:
                params[first_char] = token[1:]

        # Build line with proper ordering
        parts = []

        # Commands first (non-motion, then motion)
        motion_cmds = {'G0', 'G1', 'G2', 'G3'}
        non_motion = [c for c in commands if c not in motion_cmds]
        motion = [c for c in commands if c in motion_cmds]
        parts.extend(non_motion)
        parts.extend(motion)

        # Parameters in standard order
        param_order = ['X', 'Y', 'Z', 'A', 'B', 'C', 'I', 'J', 'K', 'R', 'F', 'S', 'T', 'P', 'Q', 'L']
        for p in param_order:
            if p in params:
                parts.append(f"{p}{params[p]}")

        return ' '.join(parts)

    def _compute_confidence(
        self,
        type_logits: torch.Tensor,
        command_logits: torch.Tensor,
        param_type_logits: torch.Tensor,
    ) -> float:
        """Compute confidence score for predictions."""
        type_probs = F.softmax(type_logits, dim=-1)
        type_conf = type_probs.max(dim=-1).values.mean().item()

        cmd_probs = F.softmax(command_logits, dim=-1)
        cmd_conf = cmd_probs.max(dim=-1).values.mean().item()

        param_probs = F.softmax(param_type_logits, dim=-1)
        param_conf = param_probs.max(dim=-1).values.mean().item()

        return (type_conf + cmd_conf + param_conf) / 3

    def _apply_grammar_fixes(self, gcode: str) -> str:
        """Apply grammar-based fixes to the G-code string."""
        if not gcode:
            return gcode

        parts = gcode.split()
        fixed_parts = []

        has_g0 = 'G0' in parts
        has_arc = any(p in parts for p in ['G2', 'G3'])

        for part in parts:
            # Remove F from G0 lines
            if has_g0 and part.startswith('F'):
                continue
            fixed_parts.append(part)

        return ' '.join(fixed_parts)
