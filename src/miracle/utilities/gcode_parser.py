"""
G-Code to 3D Coordinate Parser for Toolpath Visualization.

Converts G-code strings to 3D coordinate sequences for visualizing
ground truth vs predicted toolpaths on the dashboard.

Handles:
- G0/G1 linear motions
- G2/G3 arc interpolation (R or I/J/K parameters)
- G90/G91 absolute/incremental modes
- Modal state tracking across commands
- Path comparison with deviation metrics
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum


class AccuracyLevel(Enum):
    """Classification of prediction accuracy."""
    CORRECT = "correct"      # Exact match
    PARTIAL = "partial"      # Command matches, values differ slightly
    WRONG = "wrong"          # Command mismatch or large deviation


@dataclass
class MachineState:
    """Tracks CNC machine modal state for toolpath generation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    distance_mode: str = 'G90'  # G90=absolute, G91=incremental
    motion_mode: str = 'G1'     # G0=rapid, G1=linear, G2/G3=arc
    plane: str = 'G17'          # G17=XY, G18=XZ, G19=YZ
    feed_rate: float = 100.0

    def copy(self) -> 'MachineState':
        """Create a copy of the current state."""
        return MachineState(
            x=self.x, y=self.y, z=self.z,
            distance_mode=self.distance_mode,
            motion_mode=self.motion_mode,
            plane=self.plane,
            feed_rate=self.feed_rate
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'x': self.x, 'y': self.y, 'z': self.z,
            'distance_mode': self.distance_mode,
            'motion_mode': self.motion_mode,
            'plane': self.plane,
            'feed_rate': self.feed_rate
        }


@dataclass
class ToolpathPoint:
    """A single point on the toolpath with metadata."""
    x: float
    y: float
    z: float
    motion_type: str      # 'rapid', 'linear', 'arc_cw', 'arc_ccw', 'start'
    feed_rate: float
    line_index: int       # Index in original G-code sequence
    command: str          # Original G-code command string

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'motion_type': self.motion_type,
            'feed_rate': self.feed_rate,
            'line_index': self.line_index,
            'command': self.command
        }

    def distance_to(self, other: 'ToolpathPoint') -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )


@dataclass
class ComparisonPoint:
    """Comparison of GT vs Predicted at same sequence index."""
    index: int
    gt: ToolpathPoint
    pred: ToolpathPoint
    error_distance: float     # Euclidean distance between GT and Pred
    command_match: bool       # Do motion commands match?
    accuracy_level: str       # 'correct', 'partial', 'wrong'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'index': self.index,
            'gt': self.gt.to_dict(),
            'pred': self.pred.to_dict(),
            'error_distance': self.error_distance,
            'command_match': self.command_match,
            'accuracy_level': self.accuracy_level
        }


@dataclass
class PathMetrics:
    """Aggregate metrics for path comparison."""
    total_points: int = 0
    correct_count: int = 0
    partial_count: int = 0
    wrong_count: int = 0
    total_error: float = 0.0
    max_error: float = 0.0
    mean_error: float = 0.0
    accuracy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_points': self.total_points,
            'correct_count': self.correct_count,
            'partial_count': self.partial_count,
            'wrong_count': self.wrong_count,
            'total_error': round(self.total_error, 6),
            'max_error': round(self.max_error, 6),
            'mean_error': round(self.mean_error, 6),
            'accuracy': round(self.accuracy * 100, 2)  # Percentage
        }


class GCodeToCoordinateParser:
    """
    Converts G-code strings to 3D coordinate sequences.

    Handles modal state tracking and supports:
    - G0/G1 linear motions
    - G2/G3 arc interpolation with R or I/J/K parameters
    - G90/G91 absolute/incremental modes
    - G17/G18/G19 plane selection
    """

    # Regex for parsing G-code parameters
    # Matches: X1.234, Y-0.5, G1, M3, etc.
    PARAM_PATTERN = re.compile(r'([A-Z])(-?\d+\.?\d*)')

    # Thresholds for accuracy classification
    CORRECT_THRESHOLD = 0.001    # < 0.001mm = exact match
    PARTIAL_THRESHOLD = 0.1      # < 0.1mm = partial match

    def __init__(self, arc_segments: int = 20):
        """
        Initialize parser.

        Args:
            arc_segments: Number of line segments to interpolate arcs into
        """
        self.arc_segments = arc_segments
        self.reset()

    def reset(self):
        """Reset machine state to defaults."""
        self.state = MachineState()
        self.toolpath: List[ToolpathPoint] = []

    def parse_line(self, gcode: str) -> Dict[str, float]:
        """
        Parse a G-code line into parameter dictionary.

        Args:
            gcode: G-code string like "G1 X1.5 Y2.0 Z-0.1 F100"

        Returns:
            Dictionary of parameters, e.g., {'G': 1, 'X': 1.5, 'Y': 2.0, ...}
        """
        params = {}
        gcode_upper = gcode.upper().strip()

        for match in self.PARAM_PATTERN.finditer(gcode_upper):
            letter, value = match.groups()
            try:
                params[letter] = float(value)
            except ValueError:
                continue

        return params

    def parse_sequence(
        self,
        gcode_lines: List[str],
        initial_state: Optional[MachineState] = None
    ) -> List[ToolpathPoint]:
        """
        Convert a sequence of G-code lines to toolpath points.

        Args:
            gcode_lines: List of G-code strings
            initial_state: Optional starting machine state

        Returns:
            List of ToolpathPoint objects representing the toolpath
        """
        if initial_state:
            self.state = initial_state.copy()
        else:
            self.reset()

        self.toolpath = []

        # Add starting position
        self.toolpath.append(ToolpathPoint(
            x=self.state.x, y=self.state.y, z=self.state.z,
            motion_type='start', feed_rate=0,
            line_index=-1, command='START'
        ))

        for idx, line in enumerate(gcode_lines):
            if not line or not line.strip():
                continue
            self._process_line(line.strip(), idx)

        return self.toolpath

    def _process_line(self, gcode: str, line_idx: int):
        """Process a single G-code line and update state."""
        params = self.parse_line(gcode)

        if not params:
            return

        # Update modal states first
        if 'G' in params:
            g_code = int(params['G'])
            if g_code in [0, 1, 2, 3]:
                self.state.motion_mode = f'G{g_code}'
            elif g_code == 90:
                self.state.distance_mode = 'G90'
            elif g_code == 91:
                self.state.distance_mode = 'G91'
            elif g_code in [17, 18, 19]:
                self.state.plane = f'G{g_code}'

        # Update feed rate
        if 'F' in params:
            self.state.feed_rate = params['F']

        # Check if we have any position parameters
        has_position = any(p in params for p in ['X', 'Y', 'Z', 'A', 'B', 'C'])

        if not has_position:
            return

        # Calculate target position
        target_x, target_y, target_z = self._calculate_target(params)

        # Generate toolpath points based on motion type
        motion = self.state.motion_mode

        if motion in ['G0', 'G1']:
            # Linear motion - single point
            motion_type = 'rapid' if motion == 'G0' else 'linear'
            self._add_linear_point(
                target_x, target_y, target_z,
                motion_type, line_idx, gcode
            )
        elif motion in ['G2', 'G3']:
            # Arc motion - interpolated points
            self._add_arc_points(
                target_x, target_y, target_z,
                params, line_idx, gcode
            )
        else:
            # Default to linear
            self._add_linear_point(
                target_x, target_y, target_z,
                'linear', line_idx, gcode
            )

        # Update current position
        self.state.x = target_x
        self.state.y = target_y
        self.state.z = target_z

    def _calculate_target(self, params: Dict[str, float]) -> Tuple[float, float, float]:
        """Calculate target position based on distance mode."""
        if self.state.distance_mode == 'G91':
            # Incremental mode
            x = self.state.x + params.get('X', 0)
            y = self.state.y + params.get('Y', 0)
            z = self.state.z + params.get('Z', 0)
        else:
            # Absolute mode (default)
            x = params.get('X', self.state.x)
            y = params.get('Y', self.state.y)
            z = params.get('Z', self.state.z)
        return x, y, z

    def _add_linear_point(self, x: float, y: float, z: float,
                          motion_type: str, line_idx: int, cmd: str):
        """Add a single linear motion point."""
        self.toolpath.append(ToolpathPoint(
            x=x, y=y, z=z,
            motion_type=motion_type,
            feed_rate=self.state.feed_rate if motion_type == 'linear' else 0,
            line_index=line_idx,
            command=cmd
        ))

    def _add_arc_points(self, end_x: float, end_y: float, end_z: float,
                        params: Dict[str, float], line_idx: int, cmd: str):
        """Generate interpolated arc points for G2/G3."""
        start_x, start_y, start_z = self.state.x, self.state.y, self.state.z
        clockwise = (self.state.motion_mode == 'G2')
        motion_type = 'arc_cw' if clockwise else 'arc_ccw'

        # Determine arc center from R or I/J/K
        if 'R' in params:
            center_x, center_y = self._arc_center_from_radius(
                start_x, start_y, end_x, end_y,
                params['R'], clockwise
            )
        else:
            # I, J, K are offsets from start point
            i = params.get('I', 0)
            j = params.get('J', 0)
            center_x = start_x + i
            center_y = start_y + j

        # Calculate angles
        start_angle = math.atan2(start_y - center_y, start_x - center_x)
        end_angle = math.atan2(end_y - center_y, end_x - center_x)

        # Adjust angle sweep for direction
        if clockwise:
            if end_angle >= start_angle:
                end_angle -= 2 * math.pi
        else:
            if end_angle <= start_angle:
                end_angle += 2 * math.pi

        # Calculate radius
        radius = math.sqrt((start_x - center_x)**2 + (start_y - center_y)**2)

        # Interpolate arc
        for i in range(1, self.arc_segments + 1):
            t = i / self.arc_segments
            angle = start_angle + t * (end_angle - start_angle)

            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            z = start_z + t * (end_z - start_z)  # Linear Z interpolation

            self.toolpath.append(ToolpathPoint(
                x=x, y=y, z=z,
                motion_type=motion_type,
                feed_rate=self.state.feed_rate,
                line_index=line_idx,
                command=cmd
            ))

    def _arc_center_from_radius(
        self, x1: float, y1: float, x2: float, y2: float,
        r: float, clockwise: bool
    ) -> Tuple[float, float]:
        """Calculate arc center from radius specification."""
        dx, dy = x2 - x1, y2 - y1
        d = math.sqrt(dx*dx + dy*dy)

        if d > 2 * abs(r):
            # Radius too small, use chord midpoint
            return (x1 + x2) / 2, (y1 + y2) / 2

        h = math.sqrt(r*r - (d/2)**2)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        # Perpendicular offset direction
        if (r > 0) != clockwise:
            cx = mx - h * dy / d
            cy = my + h * dx / d
        else:
            cx = mx + h * dy / d
            cy = my - h * dx / d

        return cx, cy

    def compare_paths(
        self,
        gt_gcode: List[str],
        pred_gcode: List[str],
        initial_state: Optional[MachineState] = None
    ) -> Tuple[List[ComparisonPoint], PathMetrics, List[Dict]]:
        """
        Compare ground truth and predicted G-code paths.

        Args:
            gt_gcode: Ground truth G-code lines
            pred_gcode: Predicted G-code lines
            initial_state: Optional initial machine state

        Returns:
            Tuple of:
            - List of ComparisonPoint objects
            - PathMetrics aggregate statistics
            - List of timeline data for visualization
        """
        # Parse both sequences
        if initial_state:
            gt_state = initial_state.copy()
            pred_state = initial_state.copy()
        else:
            gt_state = MachineState()
            pred_state = MachineState()

        self.reset()
        gt_path = self.parse_sequence(gt_gcode, gt_state)

        self.reset()
        pred_path = self.parse_sequence(pred_gcode, pred_state)

        # Compare paths point by point
        comparison = []
        timeline_data = []
        metrics = PathMetrics()

        max_len = max(len(gt_path), len(pred_path))

        for i in range(max_len):
            # Get points (use last point if one sequence is shorter)
            gt_pt = gt_path[min(i, len(gt_path) - 1)]
            pred_pt = pred_path[min(i, len(pred_path) - 1)]

            # Calculate error
            error = gt_pt.distance_to(pred_pt)

            # Check command match
            cmd_match = gt_pt.motion_type == pred_pt.motion_type

            # Classify accuracy
            if error < self.CORRECT_THRESHOLD and cmd_match:
                accuracy_level = AccuracyLevel.CORRECT.value
                metrics.correct_count += 1
            elif error < self.PARTIAL_THRESHOLD:
                accuracy_level = AccuracyLevel.PARTIAL.value
                metrics.partial_count += 1
            else:
                accuracy_level = AccuracyLevel.WRONG.value
                metrics.wrong_count += 1

            # Create comparison point
            cp = ComparisonPoint(
                index=i,
                gt=gt_pt,
                pred=pred_pt,
                error_distance=error,
                command_match=cmd_match,
                accuracy_level=accuracy_level
            )
            comparison.append(cp)

            # Timeline data for 2D chart
            timeline_data.append({
                'index': i,
                'error': error,
                'accuracy_level': accuracy_level,
                'gt_command': gt_pt.command,
                'pred_command': pred_pt.command
            })

            # Update metrics
            metrics.total_error += error
            metrics.max_error = max(metrics.max_error, error)

        # Finalize metrics
        metrics.total_points = max_len
        if metrics.total_points > 0:
            metrics.mean_error = metrics.total_error / metrics.total_points
            metrics.accuracy = metrics.correct_count / metrics.total_points

        return comparison, metrics, timeline_data

    def get_error_indices(
        self,
        comparison: List[ComparisonPoint],
        min_level: str = 'partial'
    ) -> List[int]:
        """
        Get indices of points with errors at or above specified level.

        Args:
            comparison: List of ComparisonPoint objects
            min_level: Minimum error level ('partial' or 'wrong')

        Returns:
            List of indices with errors
        """
        error_levels = {
            'correct': 0,
            'partial': 1,
            'wrong': 2
        }
        min_val = error_levels.get(min_level, 1)

        return [
            cp.index for cp in comparison
            if error_levels.get(cp.accuracy_level, 0) >= min_val
        ]


def parse_gcode_to_coords(
    gcode_lines: List[str],
    initial_x: float = 0,
    initial_y: float = 0,
    initial_z: float = 0
) -> List[Dict[str, Any]]:
    """
    Convenience function to parse G-code to coordinate list.

    Args:
        gcode_lines: List of G-code strings
        initial_x/y/z: Initial position

    Returns:
        List of coordinate dictionaries
    """
    parser = GCodeToCoordinateParser()
    initial_state = MachineState(x=initial_x, y=initial_y, z=initial_z)
    path = parser.parse_sequence(gcode_lines, initial_state)
    return [p.to_dict() for p in path]


def compare_gcode_paths(
    gt_gcode: List[str],
    pred_gcode: List[str],
    initial_x: float = 0,
    initial_y: float = 0,
    initial_z: float = 0
) -> Dict[str, Any]:
    """
    Convenience function to compare GT and predicted paths.

    Args:
        gt_gcode: Ground truth G-code lines
        pred_gcode: Predicted G-code lines
        initial_x/y/z: Initial position

    Returns:
        Dictionary with comparison data, metrics, and timeline
    """
    parser = GCodeToCoordinateParser()
    initial_state = MachineState(x=initial_x, y=initial_y, z=initial_z)

    comparison, metrics, timeline = parser.compare_paths(
        gt_gcode, pred_gcode, initial_state
    )

    return {
        'comparison': [cp.to_dict() for cp in comparison],
        'metrics': metrics.to_dict(),
        'timeline_data': timeline,
        'error_indices': parser.get_error_indices(comparison, 'partial')
    }
