"""Grid-world with approach-avoidance conflict for fear vs anger testing."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

@dataclass
class EmotionalContext:
    """Context for computing emotional signals."""
    threat_distance: float
    goal_distance: float
    reward_distance: float
    was_blocked: bool = False
    obtained_reward: float = 0.0
    near_high_value: bool = False  # Near the risky high-value reward

class ApproachAvoidanceGridWorld:
    """Grid-world with valuable reward near threat.

    Classic approach-avoidance conflict (Miller, 1944):
    - Safe reward: Low value, far from threat
    - Risky reward: High value, near threat

    Tests how fear (avoidance) and anger/desire (approach) compete.
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

    def __init__(self, size: int = 7,
                 threat_pos: Tuple[int, int] = (3, 5),
                 safe_reward_pos: Tuple[int, int] = (3, 0),
                 risky_reward_pos: Tuple[int, int] = (3, 4),
                 safe_value: float = 0.3,
                 risky_value: float = 1.0):
        """
        Args:
            size: Grid size
            threat_pos: Location of threat
            safe_reward_pos: Location of safe (low-value) reward
            risky_reward_pos: Location of risky (high-value) reward near threat
        """
        self.size = size
        self.threat_pos = np.array(threat_pos)
        self.safe_reward_pos = np.array(safe_reward_pos)
        self.risky_reward_pos = np.array(risky_reward_pos)
        self.safe_value = safe_value
        self.risky_value = risky_value
        self.reset()

    def reset(self) -> int:
        """Reset to start position."""
        self.agent_pos = np.array([3, 2])  # Middle, equidistant from both rewards
        self.safe_collected = False
        self.risky_collected = False
        self.step_count = 0
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def step(self, action: int) -> Tuple[int, float, bool, EmotionalContext]:
        """Take action."""
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        was_blocked = False
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            was_blocked = True
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        # Check rewards
        reward = -0.01  # Step penalty
        obtained = 0.0

        if np.array_equal(self.agent_pos, self.safe_reward_pos) and not self.safe_collected:
            reward = self.safe_value
            obtained = self.safe_value
            self.safe_collected = True

        if np.array_equal(self.agent_pos, self.risky_reward_pos) and not self.risky_collected:
            reward = self.risky_value
            obtained = self.risky_value
            self.risky_collected = True

        # Threat penalty (in reward, but fear channel also active)
        threat_dist = self._distance(self.agent_pos, self.threat_pos)
        if threat_dist < 1.5:
            reward -= 0.3  # Threat has cost

        # Episode ends when both collected or timeout
        done = (self.safe_collected and self.risky_collected) or self.step_count >= 100

        context = EmotionalContext(
            threat_distance=threat_dist,
            goal_distance=min(
                0 if self.safe_collected else self._distance(self.agent_pos, self.safe_reward_pos),
                0 if self.risky_collected else self._distance(self.agent_pos, self.risky_reward_pos)
            ),
            reward_distance=self._distance(self.agent_pos, self.risky_reward_pos),
            was_blocked=was_blocked,
            obtained_reward=obtained,
            near_high_value=(threat_dist < 2.5 and not self.risky_collected)
        )

        return self._pos_to_state(self.agent_pos), reward, done, context

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4

    def render(self) -> str:
        """ASCII render."""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        grid[self.threat_pos[0]][self.threat_pos[1]] = 'X'

        safe_char = 's' if self.safe_collected else 'S'
        risky_char = 'r' if self.risky_collected else 'R'
        grid[self.safe_reward_pos[0]][self.safe_reward_pos[1]] = safe_char
        grid[self.risky_reward_pos[0]][self.risky_reward_pos[1]] = risky_char

        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        return '\n'.join([' '.join(row) for row in grid])
