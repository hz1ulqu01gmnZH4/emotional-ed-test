"""Grid-world for multi-channel emotion integration testing.

Combines fear, anger, and regret in a single environment to test
how emotional channels interact and compete.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

@dataclass
class IntegratedEmotionalContext:
    """Context for all emotional channels simultaneously."""
    # Fear channel
    threat_distance: float
    threat_type: str = 'real'  # 'real' or 'fake'

    # Anger channel
    was_blocked: bool = False
    consecutive_blocks: int = 0

    # Regret channel
    counterfactual_shown: bool = False
    obtained_reward: float = 0.0
    foregone_reward: float = 0.0

    # Goal tracking
    goal_distance: float = 0.0

    # Additional context
    near_high_value: bool = False  # Near high-value but dangerous target


class IntegrationGridWorld:
    """Grid-world testing interaction of fear, anger, and regret.

    Scenario:
    - Safe path: Low reward (0.3), no threat, no obstacles
    - Risky path: High reward (1.0) but near threat
    - Blocked path: Medium reward (0.6) but initially blocked

    Tests:
    1. Fear vs approach: Does fear prevent reaching high reward?
    2. Anger vs fear: Does frustration from blocked path override fear?
    3. Regret integration: Does seeing foregone rewards affect future choices?

    Optimal strategy depends on emotional balance:
    - Fear-dominant: Takes safe path (0.3)
    - Anger-dominant: Persists at blocked path (0.6)
    - Balanced: May overcome fear for risky path (1.0)
    - Regret-sensitive: Learns from all outcomes
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 7):
        self.size = size

        # Start position - left side center
        self.start_pos = np.array([3, 0])

        # Three paths with clear tradeoffs:
        # SAFE path: Go UP then RIGHT - longest but no obstacles (0.3)
        self.safe_goal = np.array([0, 6])

        # RISKY path: Go RIGHT directly - shortest but threat guards it (1.0)
        self.risky_goal = np.array([3, 6])

        # BLOCKED path: Go DOWN then RIGHT - medium value, wall blocks direct path (0.6)
        self.blocked_goal = np.array([6, 6])

        # Threat guards the middle/risky path
        self.threat_pos = np.array([3, 4])

        # Wall blocks downward path completely until broken
        self.wall_positions = [
            np.array([4, 1]), np.array([4, 2]), np.array([4, 3]),
            np.array([4, 4]), np.array([4, 5])  # Full horizontal wall
        ]
        self.wall_strength = 4  # Hits needed per wall segment

        self.reset()

    def reset(self) -> int:
        self.agent_pos = self.start_pos.copy()
        self.safe_collected = False
        self.risky_collected = False
        self.blocked_collected = False
        self.step_count = 0
        self.wall_hits = 0
        self.consecutive_wall_hits = 0
        self.wall_broken = False

        # Track path for regret
        self.visited_goals = []
        self.last_reward = 0.0

        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def _is_wall(self, pos: np.ndarray) -> bool:
        if self.wall_broken:
            return False
        return any(np.array_equal(pos, w) for w in self.wall_positions)

    def step(self, action: int) -> Tuple[int, float, bool, IntegratedEmotionalContext]:
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        was_blocked = False
        consecutive_blocks = 0

        # Check boundaries
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            was_blocked = True
            new_pos = self.agent_pos

        # Check wall
        if self._is_wall(new_pos):
            was_blocked = True
            self.wall_hits += 1
            self.consecutive_wall_hits += 1
            consecutive_blocks = self.consecutive_wall_hits

            # Break wall after enough hits
            if self.wall_hits >= self.wall_strength:
                self.wall_broken = True
                was_blocked = False
            else:
                new_pos = self.agent_pos
        else:
            self.consecutive_wall_hits = 0

        self.agent_pos = new_pos

        # Compute distances
        threat_dist = self._distance(self.agent_pos, self.threat_pos)
        safe_dist = self._distance(self.agent_pos, self.safe_goal)
        risky_dist = self._distance(self.agent_pos, self.risky_goal)
        blocked_dist = self._distance(self.agent_pos, self.blocked_goal)

        min_goal_dist = min(safe_dist, risky_dist, blocked_dist)

        # Base reward
        reward = -0.01

        # Threat penalty (near risky path) - only for direct collision
        if threat_dist < 0.5:
            reward -= 0.2  # Reduced penalty - risk should be worth taking

        # Goal rewards
        counterfactual_shown = False
        obtained_reward = 0.0
        foregone_reward = 0.0

        if np.array_equal(self.agent_pos, self.safe_goal) and not self.safe_collected:
            reward += 0.3
            self.safe_collected = True
            self.visited_goals.append('safe')
            obtained_reward = 0.3
            # Show what risky path would have given
            foregone_reward = 1.0
            counterfactual_shown = True

        if np.array_equal(self.agent_pos, self.risky_goal) and not self.risky_collected:
            reward += 1.0
            self.risky_collected = True
            self.visited_goals.append('risky')
            obtained_reward = 1.0
            foregone_reward = 0.3
            counterfactual_shown = True

        if np.array_equal(self.agent_pos, self.blocked_goal) and not self.blocked_collected:
            reward += 0.6
            self.blocked_collected = True
            self.visited_goals.append('blocked')
            obtained_reward = 0.6
            foregone_reward = 1.0
            counterfactual_shown = True

        self.last_reward = reward

        # Done when any goal collected or timeout
        done = (self.safe_collected or self.risky_collected or
                self.blocked_collected or self.step_count >= 100)

        # Near high value (risky goal)
        near_high_value = risky_dist < 3.0

        context = IntegratedEmotionalContext(
            threat_distance=threat_dist,
            threat_type='real',
            was_blocked=was_blocked,
            consecutive_blocks=consecutive_blocks,
            counterfactual_shown=counterfactual_shown,
            obtained_reward=obtained_reward,
            foregone_reward=foregone_reward,
            goal_distance=min_goal_dist,
            near_high_value=near_high_value
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

        # Mark goals
        s = 's' if self.safe_collected else 'S'
        grid[self.safe_goal[0]][self.safe_goal[1]] = s

        r = 'r' if self.risky_collected else 'R'
        grid[self.risky_goal[0]][self.risky_goal[1]] = r

        b = 'b' if self.blocked_collected else 'B'
        grid[self.blocked_goal[0]][self.blocked_goal[1]] = b

        # Mark threat
        grid[self.threat_pos[0]][self.threat_pos[1]] = 'X'

        # Mark wall
        for w in self.wall_positions:
            if not self.wall_broken:
                grid[w[0]][w[1]] = '#'

        # Mark agent
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'

        return '\n'.join([' '.join(row) for row in grid])
