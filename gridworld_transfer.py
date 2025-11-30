"""Grid-worlds for transfer/generalization testing.

Tests whether learned emotional responses transfer to novel situations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class TransferContext:
    """Context for transfer testing."""
    threat_distance: float
    goal_distance: float
    novel_threat: bool = False  # Is this a new type of threat?
    threat_type: str = 'original'  # 'original' or 'novel'


class TrainingGridWorld:
    """Training environment: Learn to fear one type of threat."""

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 5):
        self.size = size
        self.threat_pos = np.array([2, 2])
        self.goal_pos = np.array([4, 4])
        self.reset()

    def reset(self) -> int:
        self.agent_pos = np.array([0, 0])
        self.step_count = 0
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def step(self, action: int) -> Tuple[int, float, bool, TransferContext]:
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        threat_dist = self._distance(self.agent_pos, self.threat_pos)
        goal_dist = self._distance(self.agent_pos, self.goal_pos)

        reward = -0.01
        if threat_dist < 1.5:
            reward -= 0.5

        done = np.array_equal(self.agent_pos, self.goal_pos) or self.step_count >= 100
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 1.0

        context = TransferContext(
            threat_distance=threat_dist,
            goal_distance=goal_dist,
            novel_threat=False,
            threat_type='original'
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
        grid[self.threat_pos[0]][self.threat_pos[1]] = 'X'
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        return '\n'.join([' '.join(row) for row in grid])


class NovelThreatGridWorld:
    """Test environment: Novel threat in different location."""

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 5):
        self.size = size
        # Novel threat in different position
        self.threat_pos = np.array([1, 3])
        self.goal_pos = np.array([4, 4])
        self.reset()

    def reset(self) -> int:
        self.agent_pos = np.array([0, 0])
        self.step_count = 0
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def step(self, action: int) -> Tuple[int, float, bool, TransferContext]:
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        threat_dist = self._distance(self.agent_pos, self.threat_pos)
        goal_dist = self._distance(self.agent_pos, self.goal_pos)

        reward = -0.01
        if threat_dist < 1.5:
            reward -= 0.5

        done = np.array_equal(self.agent_pos, self.goal_pos) or self.step_count >= 100
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 1.0

        context = TransferContext(
            threat_distance=threat_dist,
            goal_distance=goal_dist,
            novel_threat=True,
            threat_type='novel'
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
        grid[self.threat_pos[0]][self.threat_pos[1]] = 'Y'  # Novel threat
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        return '\n'.join([' '.join(row) for row in grid])


class LargerGridWorld:
    """Test environment: Same threat type, larger space."""

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 7):
        self.size = size
        # Same relative position as training
        self.threat_pos = np.array([3, 3])
        self.goal_pos = np.array([6, 6])
        self.reset()

    def reset(self) -> int:
        self.agent_pos = np.array([0, 0])
        self.step_count = 0
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def step(self, action: int) -> Tuple[int, float, bool, TransferContext]:
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        threat_dist = self._distance(self.agent_pos, self.threat_pos)
        goal_dist = self._distance(self.agent_pos, self.goal_pos)

        reward = -0.01
        if threat_dist < 1.5:
            reward -= 0.5

        done = np.array_equal(self.agent_pos, self.goal_pos) or self.step_count >= 100
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 1.0

        context = TransferContext(
            threat_distance=threat_dist,
            goal_distance=goal_dist,
            novel_threat=False,
            threat_type='original'
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
        grid[self.threat_pos[0]][self.threat_pos[1]] = 'X'
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        return '\n'.join([' '.join(row) for row in grid])


class MultipleThreatGridWorld:
    """Test environment: Multiple threats."""

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 6):
        self.size = size
        self.threat_positions = [
            np.array([2, 2]),  # Same as training
            np.array([2, 4]),  # New
            np.array([4, 2]),  # New
        ]
        self.goal_pos = np.array([5, 5])
        self.reset()

    def reset(self) -> int:
        self.agent_pos = np.array([0, 0])
        self.step_count = 0
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def step(self, action: int) -> Tuple[int, float, bool, TransferContext]:
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        # Distance to nearest threat
        threat_dists = [self._distance(self.agent_pos, t) for t in self.threat_positions]
        threat_dist = min(threat_dists)
        goal_dist = self._distance(self.agent_pos, self.goal_pos)

        reward = -0.01
        if threat_dist < 1.5:
            reward -= 0.5

        done = np.array_equal(self.agent_pos, self.goal_pos) or self.step_count >= 100
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 1.0

        # Check if nearest threat is novel
        nearest_idx = np.argmin(threat_dists)
        novel = nearest_idx > 0  # First threat is original, others are novel

        context = TransferContext(
            threat_distance=threat_dist,
            goal_distance=goal_dist,
            novel_threat=novel,
            threat_type='novel' if novel else 'original'
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
        for i, t in enumerate(self.threat_positions):
            grid[t[0]][t[1]] = 'X' if i == 0 else 'Y'
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        return '\n'.join([' '.join(row) for row in grid])
