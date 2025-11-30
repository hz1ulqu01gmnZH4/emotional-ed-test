"""Minimal grid-world environment for emotional ED testing."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

@dataclass
class EmotionalContext:
    """Context for computing emotional signals."""
    threat_distance: float
    goal_distance: float
    was_blocked: bool
    foregone_value: Optional[float] = None

class GridWorld:
    """Simple grid-world with threat and goal."""

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

    def __init__(self, size: int = 5, threat_pos: Tuple[int, int] = (2, 2)):
        self.size = size
        self.threat_pos = np.array(threat_pos)
        self.goal_pos = np.array([size - 1, size - 1])
        self.reset()

    def reset(self) -> int:
        """Reset to start position, return state index."""
        self.agent_pos = np.array([0, 0])
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        """Convert (x, y) to state index."""
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Euclidean distance."""
        return np.linalg.norm(pos1 - pos2)

    def step(self, action: int, include_threat_penalty: bool = False) -> Tuple[int, float, bool, EmotionalContext]:
        """Take action, return (state, reward, done, emotional_context).

        Args:
            include_threat_penalty: If True, add threat proximity to reward.
                                   If False, threat is only in emotional context.
        """
        # Attempt move
        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        # Check bounds
        was_blocked = False
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            was_blocked = True
            new_pos = self.agent_pos  # Stay in place

        self.agent_pos = new_pos

        # Compute reward
        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 1.0 if done else -0.01  # Small step penalty

        threat_dist = self._distance(self.agent_pos, self.threat_pos)

        # Optional threat penalty in reward (for fair comparison)
        if include_threat_penalty and threat_dist < 1.5:
            reward -= 0.5

        # Emotional context
        context = EmotionalContext(
            threat_distance=threat_dist,
            goal_distance=self._distance(self.agent_pos, self.goal_pos),
            was_blocked=was_blocked
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
        grid[self.threat_pos[0]][self.threat_pos[1]] = 'X'
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        return '\n'.join([' '.join(row) for row in grid])
