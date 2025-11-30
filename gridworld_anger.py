"""Grid-world with temporary walls for anger/frustration testing."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Set

@dataclass
class EmotionalContext:
    """Context for computing emotional signals."""
    threat_distance: float
    goal_distance: float
    was_blocked: bool
    goal_visible: bool = True
    consecutive_blocks: int = 0
    foregone_value: Optional[float] = None

class BlockedPathGridWorld:
    """Grid-world where direct path to goal is temporarily blocked."""

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right
    ACTION_NAMES = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

    def __init__(self, size: int = 5, wall_positions: List[Tuple[int, int]] = None,
                 wall_duration: int = 10):
        """
        Args:
            size: Grid size
            wall_positions: Positions that block the direct path
            wall_duration: How many steps the wall stays up (0 = permanent)
        """
        self.size = size
        self.goal_pos = np.array([size - 1, size - 1])

        # Default wall blocks the direct diagonal path
        if wall_positions is None:
            # Wall at (2,2), (2,3), (3,2) - blocks center
            wall_positions = [(2, 2), (2, 3), (3, 2)]
        self.wall_positions: Set[Tuple[int, int]] = set(wall_positions)
        self.wall_duration = wall_duration

        self.reset()

    def reset(self) -> int:
        """Reset to start position."""
        self.agent_pos = np.array([0, 0])
        self.steps = 0
        self.consecutive_blocks = 0
        self.walls_active = True
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        """Convert (x, y) to state index."""
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Euclidean distance."""
        return np.linalg.norm(pos1 - pos2)

    def _is_wall(self, pos: np.ndarray) -> bool:
        """Check if position is a wall."""
        if not self.walls_active:
            return False
        return tuple(pos) in self.wall_positions

    def step(self, action: int) -> Tuple[int, float, bool, EmotionalContext]:
        """Take action, return (state, reward, done, emotional_context)."""
        self.steps += 1

        # Walls disappear after duration (if duration > 0)
        if self.wall_duration > 0 and self.steps >= self.wall_duration:
            self.walls_active = False

        # Attempt move
        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        # Check bounds and walls
        was_blocked = False
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            was_blocked = True
            new_pos = self.agent_pos
        elif self._is_wall(new_pos):
            was_blocked = True
            new_pos = self.agent_pos

        # Track consecutive blocks
        if was_blocked:
            self.consecutive_blocks += 1
        else:
            self.consecutive_blocks = 0

        self.agent_pos = new_pos

        # Compute reward
        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 1.0 if done else -0.01  # Small step penalty

        # Emotional context
        context = EmotionalContext(
            threat_distance=float('inf'),  # No threat in this environment
            goal_distance=self._distance(self.agent_pos, self.goal_pos),
            was_blocked=was_blocked,
            goal_visible=True,  # Agent can always see the goal
            consecutive_blocks=self.consecutive_blocks
        )

        return self._pos_to_state(self.agent_pos), reward, done, context

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4

    def render(self) -> str:
        """ASCII render of grid."""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]

        # Show walls if active
        if self.walls_active:
            for wx, wy in self.wall_positions:
                if 0 <= wx < self.size and 0 <= wy < self.size:
                    grid[wx][wy] = '#'

        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        return '\n'.join([' '.join(row) for row in grid])
