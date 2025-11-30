"""Grid-world for emotion regulation testing."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class EmotionalContext:
    """Context for computing emotional signals."""
    threat_distance: float
    goal_distance: float
    was_blocked: bool = False
    # Regulation-specific
    threat_type: str = 'real'  # 'real' or 'fake' (learned to be harmless)
    reappraisal_cue: bool = False  # Environmental cue suggesting threat is fake

class RegulationGridWorld:
    """Grid-world with threats that can be reappraised.

    Some threats are 'real' (always harmful), some are 'fake' (initially
    look threatening but are actually harmless).

    Tests emotion regulation: Can agent learn to reappraise fake threats
    and approach them despite initial fear response?

    Models Ochsner & Gross (2005) cognitive reappraisal:
    - Initial response: fear to all threat-looking stimuli
    - Regulation: learn that some threats are fake → reduce fear response
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 6,
                 real_threat_pos: Tuple[int, int] = (2, 2),
                 fake_threat_pos: Tuple[int, int] = (2, 4),
                 goal_pos: Tuple[int, int] = (5, 5),
                 fake_threat_bonus: float = 0.5):
        """
        Args:
            size: Grid size
            real_threat_pos: Truly dangerous location
            fake_threat_pos: Looks dangerous but gives bonus if approached
            goal_pos: Main goal
            fake_threat_bonus: Reward for overcoming fake threat
        """
        self.size = size
        self.real_threat_pos = np.array(real_threat_pos)
        self.fake_threat_pos = np.array(fake_threat_pos)
        self.goal_pos = np.array(goal_pos)
        self.fake_threat_bonus = fake_threat_bonus
        self.reset()

    def reset(self) -> int:
        self.agent_pos = np.array([0, 0])
        self.fake_bonus_collected = False
        self.step_count = 0
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def step(self, action: int) -> Tuple[int, float, bool, EmotionalContext]:
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        was_blocked = False
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            was_blocked = True
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        # Compute distances
        real_dist = self._distance(self.agent_pos, self.real_threat_pos)
        fake_dist = self._distance(self.agent_pos, self.fake_threat_pos)
        goal_dist = self._distance(self.agent_pos, self.goal_pos)

        # Base reward
        reward = -0.01

        # Real threat penalty
        if real_dist < 1.5:
            reward -= 0.5

        # Fake threat bonus (reward for overcoming fear)
        if np.array_equal(self.agent_pos, self.fake_threat_pos) and not self.fake_bonus_collected:
            reward += self.fake_threat_bonus
            self.fake_bonus_collected = True

        # Goal reached
        done = np.array_equal(self.agent_pos, self.goal_pos) or self.step_count >= 100
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 1.0

        # Determine threat type for context
        min_threat_dist = min(real_dist, fake_dist)
        if real_dist < fake_dist:
            threat_type = 'real'
        else:
            threat_type = 'fake'

        # Reappraisal cue: being near fake threat without harm
        reappraisal_cue = (fake_dist < 2.0 and self.fake_bonus_collected)

        context = EmotionalContext(
            threat_distance=min_threat_dist,
            goal_distance=goal_dist,
            was_blocked=was_blocked,
            threat_type=threat_type,
            reappraisal_cue=reappraisal_cue
        )

        return self._pos_to_state(self.agent_pos), reward, done, context

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4

    def render(self) -> str:
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        grid[self.real_threat_pos[0]][self.real_threat_pos[1]] = 'X'  # Real threat
        marker = 'f' if self.fake_bonus_collected else 'F'
        grid[self.fake_threat_pos[0]][self.fake_threat_pos[1]] = marker  # Fake threat
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        return '\n'.join([' '.join(row) for row in grid])
