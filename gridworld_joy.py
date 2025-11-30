"""Grid-world for joy/curiosity experiments.

Tests positive emotions as approach-motivated parallel channels.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Set


@dataclass
class JoyContext:
    """Context for joy/curiosity signals."""
    novelty: float  # 0-1, how novel is current state
    reward_obtained: float  # Reward just received
    visit_count: int  # How many times visited this state
    hidden_found: bool  # Was hidden reward discovered
    distance_to_hidden: float  # Distance to hidden reward


class JoyCuriosityGridWorld:
    """Grid world with hidden rewards for curiosity/joy experiments.

    Features:
    - Hidden reward (not visible until discovered)
    - Novelty tracking for curiosity
    - Small step penalty to encourage efficiency
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 7):
        self.size = size
        self.hidden_reward_pos = np.array([5, 5])  # Hidden until discovered
        self.goal_pos = np.array([6, 6])  # Main goal
        self.visit_counts = np.zeros((size, size), dtype=int)
        self.hidden_discovered = False
        self.reset()

    def reset(self) -> int:
        self.agent_pos = np.array([0, 0])
        self.step_count = 0
        self.hidden_discovered = False
        # Don't reset visit counts - they persist across episodes
        return self._pos_to_state(self.agent_pos)

    def reset_full(self):
        """Full reset including visit counts."""
        self.visit_counts = np.zeros((self.size, self.size), dtype=int)
        return self.reset()

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def step(self, action: int) -> Tuple[int, float, bool, JoyContext]:
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        # Boundary check
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            new_pos = self.agent_pos

        self.agent_pos = new_pos
        state = self._pos_to_state(self.agent_pos)

        # Track visits
        old_count = self.visit_counts[self.agent_pos[0], self.agent_pos[1]]
        self.visit_counts[self.agent_pos[0], self.agent_pos[1]] += 1
        new_count = self.visit_counts[self.agent_pos[0], self.agent_pos[1]]

        # Compute novelty (inverse of visit count)
        novelty = 1.0 / (1.0 + old_count)

        # Reward
        reward = -0.01  # Small step penalty

        hidden_found = False
        if np.array_equal(self.agent_pos, self.hidden_reward_pos) and not self.hidden_discovered:
            reward += 0.5  # Hidden bonus
            self.hidden_discovered = True
            hidden_found = True

        done = False
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 1.0
            done = True

        if self.step_count >= 100:
            done = True

        context = JoyContext(
            novelty=novelty,
            reward_obtained=reward,
            visit_count=new_count,
            hidden_found=hidden_found,
            distance_to_hidden=self._distance(self.agent_pos, self.hidden_reward_pos)
        )

        return state, reward, done, context

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4

    def get_unvisited_count(self) -> int:
        """Count states never visited."""
        return np.sum(self.visit_counts == 0)

    def render(self) -> str:
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        if self.hidden_discovered:
            grid[self.hidden_reward_pos[0]][self.hidden_reward_pos[1]] = 'H'
        else:
            grid[self.hidden_reward_pos[0]][self.hidden_reward_pos[1]] = '?'
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        return '\n'.join([' '.join(row) for row in grid])
