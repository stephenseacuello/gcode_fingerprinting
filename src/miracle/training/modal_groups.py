"""
RS-274D Modal Groups and G-Code Grammar Rules.

This module defines the modal group structure and command-parameter
associations according to the RS-274D/NGC G-code standard.

Reference: NIST RS274NGC Interpreter, LinuxCNC G-code specification
"""

from typing import Dict, Set, Optional, List, Tuple


# =============================================================================
# MODAL GROUPS (RS-274D Standard)
# =============================================================================
# Only ONE command from each modal group can be active at a time.
# Two G-codes from the same modal group CANNOT appear on the same line.

MODAL_GROUPS: Dict[str, Set[str]] = {
    # Group 0: Non-modal codes (execute immediately, don't persist)
    'G_GROUP_0_NON_MODAL': {'G4', 'G10', 'G28', 'G30', 'G53', 'G92', 'G92.1', 'G92.2', 'G92.3'},

    # Group 1: Motion commands (most important - defines how machine moves)
    'G_GROUP_1_MOTION': {'G0', 'G1', 'G2', 'G3', 'G38.2', 'G80', 'G81', 'G82', 'G83', 'G84', 'G85', 'G86', 'G87', 'G88', 'G89'},

    # Group 2: Plane selection (for arcs and canned cycles)
    'G_GROUP_2_PLANE': {'G17', 'G18', 'G19'},

    # Group 3: Distance mode
    'G_GROUP_3_DISTANCE': {'G90', 'G91'},

    # Group 5: Feed rate mode
    'G_GROUP_5_FEED_MODE': {'G93', 'G94'},

    # Group 6: Units
    'G_GROUP_6_UNITS': {'G20', 'G21'},

    # Group 7: Cutter radius compensation
    'G_GROUP_7_CUTTER_COMP': {'G40', 'G41', 'G42', 'G41.1', 'G42.1'},

    # Group 8: Tool length offset
    'G_GROUP_8_TOOL_LENGTH': {'G43', 'G43.1', 'G49'},

    # Group 10: Return mode in canned cycles
    'G_GROUP_10_RETURN_MODE': {'G98', 'G99'},

    # Group 12: Coordinate system selection
    'G_GROUP_12_COORD_SYSTEM': {'G54', 'G55', 'G56', 'G57', 'G58', 'G59', 'G59.1', 'G59.2', 'G59.3'},

    # Group 13: Path control mode
    'G_GROUP_13_PATH_CONTROL': {'G61', 'G61.1', 'G64'},
}

M_MODAL_GROUPS: Dict[str, Set[str]] = {
    # Group 4: Stopping
    'M_GROUP_4_STOPPING': {'M0', 'M1', 'M2', 'M30', 'M60'},

    # Group 6: Tool change
    'M_GROUP_6_TOOL_CHANGE': {'M6'},

    # Group 7: Spindle control
    'M_GROUP_7_SPINDLE': {'M3', 'M4', 'M5'},

    # Group 8: Coolant control
    'M_GROUP_8_COOLANT': {'M7', 'M8', 'M9'},

    # Group 9: Override switches
    'M_GROUP_9_OVERRIDE': {'M48', 'M49'},
}

# Combined for easy lookup
ALL_MODAL_GROUPS: Dict[str, Set[str]] = {**MODAL_GROUPS, **M_MODAL_GROUPS}


# =============================================================================
# COMMAND-PARAMETER ASSOCIATION RULES
# =============================================================================
# Defines which parameters are required, optional, or forbidden for each command

COMMAND_PARAM_RULES: Dict[str, Dict[str, List[str]]] = {
    # Motion commands
    'G0': {
        'required': [],
        'optional': ['X', 'Y', 'Z', 'A', 'B', 'C'],
        'forbidden': ['F', 'R', 'I', 'J', 'K'],  # Rapid NEVER has feed rate
    },
    'G1': {
        'required': [],
        'optional': ['X', 'Y', 'Z', 'A', 'B', 'C', 'F'],
        'forbidden': ['R', 'I', 'J', 'K'],  # Linear doesn't use arc params
    },
    'G2': {
        'required': ['R_or_IJK'],  # Must have R OR (I and/or J and/or K)
        'optional': ['X', 'Y', 'Z', 'F', 'P'],
        'forbidden': [],
    },
    'G3': {
        'required': ['R_or_IJK'],  # Must have R OR (I and/or J and/or K)
        'optional': ['X', 'Y', 'Z', 'F', 'P'],
        'forbidden': [],
    },

    # Dwell
    'G4': {
        'required': ['P'],  # Dwell time in seconds
        'optional': [],
        'forbidden': ['X', 'Y', 'Z', 'F', 'S'],
    },

    # Reference point commands
    'G28': {
        'required': [],
        'optional': ['X', 'Y', 'Z'],
        'forbidden': ['F', 'S'],
    },
    'G30': {
        'required': [],
        'optional': ['X', 'Y', 'Z'],
        'forbidden': ['F', 'S'],
    },

    # Machine coordinate system
    'G53': {
        'required': [],
        'optional': ['X', 'Y', 'Z'],
        'forbidden': [],
    },

    # Spindle commands
    'M3': {
        'required': ['S'],  # Spindle speed required
        'optional': [],
        'forbidden': [],
    },
    'M4': {
        'required': ['S'],  # Spindle speed required
        'optional': [],
        'forbidden': [],
    },
    'M5': {
        'required': [],
        'optional': [],
        'forbidden': ['S'],  # Spindle stop doesn't need speed
    },

    # Tool change
    'M6': {
        'required': ['T'],  # Tool number required
        'optional': [],
        'forbidden': [],
    },

    # Coolant (no parameters needed)
    'M7': {'required': [], 'optional': [], 'forbidden': []},
    'M8': {'required': [], 'optional': [], 'forbidden': []},
    'M9': {'required': [], 'optional': [], 'forbidden': []},

    # Program control
    'M0': {'required': [], 'optional': [], 'forbidden': []},
    'M1': {'required': [], 'optional': [], 'forbidden': []},
    'M2': {'required': [], 'optional': [], 'forbidden': []},
    'M30': {'required': [], 'optional': [], 'forbidden': []},
}


# =============================================================================
# WORD LETTER DEFINITIONS
# =============================================================================

WORD_LETTERS: Dict[str, str] = {
    # Axis positions
    'X': 'X-axis position',
    'Y': 'Y-axis position',
    'Z': 'Z-axis position',
    'A': 'A-axis rotation',
    'B': 'B-axis rotation',
    'C': 'C-axis rotation',

    # Arc parameters
    'I': 'Arc center X offset (from start)',
    'J': 'Arc center Y offset (from start)',
    'K': 'Arc center Z offset (from start)',
    'R': 'Arc radius',

    # Process parameters
    'F': 'Feed rate (units/min)',
    'S': 'Spindle speed (RPM)',
    'T': 'Tool number',

    # Auxiliary
    'P': 'Dwell time / parameter',
    'Q': 'Feed increment (peck drilling)',
    'L': 'Loop count',
    'N': 'Line number',
    'D': 'Cutter compensation number',
    'H': 'Tool length offset number',
}

# Letters that can only appear once per line (except G and M)
SINGLE_LETTER_PARAMS: Set[str] = {
    'X', 'Y', 'Z', 'A', 'B', 'C',
    'I', 'J', 'K', 'R',
    'F', 'S', 'T',
    'P', 'Q', 'L', 'N', 'D', 'H'
}


# =============================================================================
# PHYSICAL CONSTRAINTS
# =============================================================================
# Typical CNC machine limits (can be adjusted per machine)

PHYSICAL_CONSTRAINTS: Dict[str, Tuple[float, float]] = {
    # Position limits (mm) - typical 3-axis mill
    'X': (-500.0, 500.0),
    'Y': (-500.0, 500.0),
    'Z': (-200.0, 50.0),

    # Rotary axes (degrees)
    'A': (-360.0, 360.0),
    'B': (-360.0, 360.0),
    'C': (-360.0, 360.0),

    # Arc center offsets (mm)
    'I': (-500.0, 500.0),
    'J': (-500.0, 500.0),
    'K': (-200.0, 200.0),
    'R': (0.001, 500.0),  # Radius must be positive

    # Process parameters
    'F': (0.1, 15000.0),      # Feed rate mm/min
    'S': (0.0, 30000.0),      # Spindle speed RPM (0 = off)
    'T': (0, 99),             # Tool number

    # Auxiliary
    'P': (0.0, 3600.0),       # Dwell time (seconds, up to 1 hour)
    'Q': (0.001, 100.0),      # Peck increment (mm)
    'L': (0, 9999),           # Loop count
}


# =============================================================================
# DEFAULT MODAL STATE
# =============================================================================
# Initial state when machine powers on (per RS-274D)

DEFAULT_MODAL_STATE: Dict[str, str] = {
    'motion': 'G0',         # Rapid positioning
    'plane': 'G17',         # XY plane
    'distance': 'G90',      # Absolute positioning
    'feed_mode': 'G94',     # Units per minute
    'units': 'G21',         # Metric (mm) - common default
    'cutter_comp': 'G40',   # Cutter compensation off
    'tool_length': 'G49',   # Tool length compensation off
    'coord_system': 'G54',  # Work coordinate system 1
    'path_control': 'G64', # Best speed path
    'spindle': 'M5',        # Spindle off
    'coolant': 'M9',        # Coolant off
}


# =============================================================================
# EXECUTION ORDER
# =============================================================================
# Commands on a line execute in this fixed order (RS-274D Section 3.5)

EXECUTION_ORDER: List[str] = [
    'comment',              # 1. Comments
    'feed_rate_mode',       # 2. G93, G94
    'feed_rate',            # 3. F
    'spindle_speed',        # 4. S
    'tool_select',          # 5. T
    'tool_change',          # 6. M6
    'spindle_control',      # 7. M3, M4, M5
    'coolant_control',      # 8. M7, M8, M9
    'override_control',     # 9. M48, M49
    'dwell',                # 10. G4
    'plane_select',         # 11. G17, G18, G19
    'units',                # 12. G20, G21
    'cutter_comp',          # 13. G40, G41, G42
    'tool_length_comp',     # 14. G43, G49
    'coord_system',         # 15. G54-G59
    'path_control',         # 16. G61, G64
    'distance_mode',        # 17. G90, G91
    'retract_mode',         # 18. G98, G99
    'reference_changes',    # 19. G28, G30, G92
    'motion',               # 20. G0, G1, G2, G3 and M0, M1, M2, M30
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_modal_group(command: str) -> Optional[str]:
    """
    Get the modal group name for a command.

    Args:
        command: G-code command (e.g., 'G0', 'M3')

    Returns:
        Modal group name or None if non-modal/unknown
    """
    for group_name, commands in ALL_MODAL_GROUPS.items():
        if command in commands:
            return group_name
    return None


def check_modal_conflict(cmd1: str, cmd2: str) -> bool:
    """
    Check if two commands conflict (are in the same modal group).

    Args:
        cmd1: First command
        cmd2: Second command

    Returns:
        True if they conflict (same modal group), False otherwise
    """
    group1 = get_modal_group(cmd1)
    group2 = get_modal_group(cmd2)

    # Non-modal commands never conflict
    if group1 is None or group2 is None:
        return False

    # Same group = conflict
    return group1 == group2


def get_conflicting_commands(command: str) -> Set[str]:
    """
    Get all commands that would conflict with the given command.

    Args:
        command: G-code command

    Returns:
        Set of conflicting commands (same modal group)
    """
    group = get_modal_group(command)
    if group is None:
        return set()

    return ALL_MODAL_GROUPS[group] - {command}


def validate_command_params(command: str, params: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate parameters for a command.

    Args:
        command: G-code command
        params: List of parameter letters used

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    if command not in COMMAND_PARAM_RULES:
        return True, []  # Unknown command, allow anything

    rules = COMMAND_PARAM_RULES[command]

    # Check forbidden parameters
    for param in params:
        if param in rules.get('forbidden', []):
            errors.append(f"{command} cannot have {param} parameter")

    # Check required parameters
    for required in rules.get('required', []):
        if required == 'R_or_IJK':
            # Special case: need R or at least one of I/J/K
            has_r = 'R' in params
            has_ijk = any(p in params for p in ['I', 'J', 'K'])
            if not (has_r or has_ijk):
                errors.append(f"{command} requires R or I/J/K parameters")
        elif required not in params:
            errors.append(f"{command} requires {required} parameter")

    return len(errors) == 0, errors


def check_single_letter_rule(params: List[str]) -> Tuple[bool, List[str]]:
    """
    Check that each parameter letter appears only once.

    Args:
        params: List of parameter letters

    Returns:
        Tuple of (is_valid, list of duplicate letters)
    """
    seen = set()
    duplicates = []

    for param in params:
        letter = param[0] if param else ''
        if letter in SINGLE_LETTER_PARAMS:
            if letter in seen:
                duplicates.append(letter)
            seen.add(letter)

    return len(duplicates) == 0, duplicates


def constrain_value(param: str, value: float) -> float:
    """
    Clamp a parameter value to physical constraints.

    Args:
        param: Parameter letter
        value: Raw value

    Returns:
        Constrained value
    """
    if param not in PHYSICAL_CONSTRAINTS:
        return value

    min_val, max_val = PHYSICAL_CONSTRAINTS[param]
    return max(min_val, min(max_val, value))


def is_motion_command(command: str) -> bool:
    """Check if command is a motion command (G0, G1, G2, G3)."""
    return command in {'G0', 'G1', 'G2', 'G3'}


def is_arc_command(command: str) -> bool:
    """Check if command is an arc command (G2, G3)."""
    return command in {'G2', 'G3'}


def is_rapid_command(command: str) -> bool:
    """Check if command is rapid positioning (G0)."""
    return command == 'G0'


def requires_feed_rate(command: str) -> bool:
    """Check if command typically requires a feed rate."""
    return command in {'G1', 'G2', 'G3'}
