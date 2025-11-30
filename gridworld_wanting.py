"""Grid-world for wanting/liking dissociation testing.

Tests Berridge's (2009) distinction between:
- Wanting: Incentive salience, motivation to pursue
- Liking: Hedonic impact, pleasure from consumption
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class WantingLikingContext:
    """Context for wanting/liking information."""
    # Distances to rewards
    high_wanting_distance: float  # Distance to high-wanting reward
    high_liking_distance: float   # Distance to high-liking reward
    regular_distance: float       # Distance to regular reward

    # Consumption history
    just_consumed: str = ''       # Which reward just consumed
    consumption_count: int = 0    # Total consumptions this episode

    # Satiation state
    satiation_level: float = 0.0  # 0 = hungry, 1 = full


class WantingLikingGridWorld:
    """Grid-world testing wanting/liking dissociation.

    Key insight (Berridge, 2009):
    - Wanting (dopamine): Motivational pull toward rewards
    - Liking (opioid): Pleasure experienced from rewards
    - These can dissociate: High wanting ≠ high liking

    Environment:
    - High-wanting reward (W): Draws agent but modest hedonic payoff
    - High-liking reward (L): Less motivating but high pleasure
    - Regular reward (R): Baseline comparison

    Satiation manipulation:
    - Repeated consumption reduces wanting more than liking
    - Models addiction-like wanting without liking
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 6):
        self.size = size

        # Start position
        self.start_pos = np.array([0, 0])

        # Reward positions and properties
        # High wanting: Highly motivating but only moderate pleasure
        self.high_wanting_pos = np.array([2, 5])
        self.wanting_reward = 0.5  # Actual hedonic value
        self.wanting_salience = 1.5  # Motivational pull

        # High liking: Less motivating but high pleasure
        self.high_liking_pos = np.array([5, 2])
        self.liking_reward = 1.0  # Actual hedonic value
        self.liking_salience = 0.7  # Motivational pull

        # Regular reward (baseline)
        self.regular_pos = np.array([3, 3])
        self.regular_reward = 0.6
        self.regular_salience = 1.0

        # Goal (exit)
        self.goal_pos = np.array([5, 5])

        self.reset()

    def reset(self) -> int:
        self.agent_pos = self.start_pos.copy()
        self.step_count = 0
        self.consumption_history = []
        self.satiation = 0.0

        # Rewards respawn each episode
        self.collected = {
            'wanting': False,
            'liking': False,
            'regular': False
        }

        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def step(self, action: int) -> Tuple[int, float, bool, WantingLikingContext]:
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        # Boundary check
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        # Compute distances
        wanting_dist = self._distance(self.agent_pos, self.high_wanting_pos)
        liking_dist = self._distance(self.agent_pos, self.high_liking_pos)
        regular_dist = self._distance(self.agent_pos, self.regular_pos)

        # Base reward
        reward = -0.01
        just_consumed = ''

        # Check reward collection
        if np.array_equal(self.agent_pos, self.high_wanting_pos) and not self.collected['wanting']:
            # High wanting: satiation reduces actual reward
            satiation_factor = 1.0 - self.satiation * 0.5
            reward += self.wanting_reward * satiation_factor
            self.collected['wanting'] = True
            just_consumed = 'wanting'
            self.consumption_history.append('wanting')
            self.satiation = min(1.0, self.satiation + 0.3)

        if np.array_equal(self.agent_pos, self.high_liking_pos) and not self.collected['liking']:
            # High liking: pleasure relatively stable with satiation
            satiation_factor = 1.0 - self.satiation * 0.2  # Less affected
            reward += self.liking_reward * satiation_factor
            self.collected['liking'] = True
            just_consumed = 'liking'
            self.consumption_history.append('liking')
            self.satiation = min(1.0, self.satiation + 0.2)

        if np.array_equal(self.agent_pos, self.regular_pos) and not self.collected['regular']:
            satiation_factor = 1.0 - self.satiation * 0.3
            reward += self.regular_reward * satiation_factor
            self.collected['regular'] = True
            just_consumed = 'regular'
            self.consumption_history.append('regular')
            self.satiation = min(1.0, self.satiation + 0.25)

        # Goal
        done = np.array_equal(self.agent_pos, self.goal_pos) or self.step_count >= 100
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 0.5

        context = WantingLikingContext(
            high_wanting_distance=wanting_dist,
            high_liking_distance=liking_dist,
            regular_distance=regular_dist,
            just_consumed=just_consumed,
            consumption_count=len(self.consumption_history),
            satiation_level=self.satiation
        )

        return self._pos_to_state(self.agent_pos), reward, done, context

    def get_reward_info(self) -> Dict:
        """Return reward properties for agent reference."""
        return {
            'wanting': {'pos': self.high_wanting_pos, 'reward': self.wanting_reward, 'salience': self.wanting_salience},
            'liking': {'pos': self.high_liking_pos, 'reward': self.liking_reward, 'salience': self.liking_salience},
            'regular': {'pos': self.regular_pos, 'reward': self.regular_reward, 'salience': self.regular_salience}
        }

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4

    def render(self) -> str:
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]

        # Mark rewards
        w = 'w' if self.collected['wanting'] else 'W'
        l = 'l' if self.collected['liking'] else 'L'
        r = 'r' if self.collected['regular'] else 'R'

        grid[self.high_wanting_pos[0]][self.high_wanting_pos[1]] = w
        grid[self.high_liking_pos[0]][self.high_liking_pos[1]] = l
        grid[self.regular_pos[0]][self.regular_pos[1]] = r
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'

        return '\n'.join([' '.join(row) for row in grid])
