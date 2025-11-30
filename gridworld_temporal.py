"""Grid-world for temporal emotion dynamics testing.

Tests phasic (acute emotion) vs tonic (mood) responses.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class TemporalEmotionalContext:
    """Context tracking temporal emotional dynamics."""
    # Current state
    threat_distance: float
    goal_distance: float
    reward_obtained: float = 0.0

    # Temporal tracking
    consecutive_negative: int = 0  # Consecutive negative outcomes
    consecutive_positive: int = 0  # Consecutive positive outcomes
    recent_threat_exposures: int = 0  # Threat exposures in last N steps
    time_since_last_reward: int = 0

    # For mood tracking
    cumulative_reward: float = 0.0
    episode_length: int = 0


class TemporalGridWorld:
    """Grid-world with variable threat and reward patterns.

    Tests whether agents develop mood-like tonic states from
    sustained positive/negative experiences.

    Phases:
    1. Neutral: Mixed outcomes
    2. Negative: Sustained threats/penalties
    3. Recovery: Return to neutral
    4. Positive: Sustained rewards

    Key question: Do emotional baselines shift with sustained experience?
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 6, phase_length: int = 50):
        self.size = size
        self.phase_length = phase_length
        self.goal_pos = np.array([size-1, size-1])

        # Threat positions change by phase
        self.threat_positions_negative = [
            np.array([2, 2]), np.array([2, 3]), np.array([3, 2]),
            np.array([3, 3]), np.array([1, 3]), np.array([3, 1])
        ]
        self.threat_positions_neutral = [np.array([2, 2])]
        self.threat_positions_positive = []  # No threats

        # Reward positions
        self.bonus_positions_positive = [
            np.array([1, 1]), np.array([1, 4]), np.array([4, 1])
        ]
        self.bonus_positions_neutral = [np.array([1, 1])]
        self.bonus_positions_negative = []  # No bonuses

        self.reset()

    def reset(self) -> int:
        self.agent_pos = np.array([0, 0])
        self.step_count = 0
        self.episode_step = 0
        self.total_episodes = 0

        # Temporal tracking
        self.consecutive_negative = 0
        self.consecutive_positive = 0
        self.recent_rewards = []  # Last 20 rewards
        self.recent_threats = []  # Last 20 threat exposures

        self.collected_bonuses = set()

        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def _get_current_phase(self) -> str:
        """Determine current phase based on total steps."""
        cycle_position = self.step_count % (4 * self.phase_length)
        if cycle_position < self.phase_length:
            return 'neutral'
        elif cycle_position < 2 * self.phase_length:
            return 'negative'
        elif cycle_position < 3 * self.phase_length:
            return 'recovery'
        else:
            return 'positive'

    def _get_threats(self) -> List[np.ndarray]:
        phase = self._get_current_phase()
        if phase == 'negative':
            return self.threat_positions_negative
        elif phase in ['neutral', 'recovery']:
            return self.threat_positions_neutral
        else:  # positive
            return self.threat_positions_positive

    def _get_bonuses(self) -> List[np.ndarray]:
        phase = self._get_current_phase()
        if phase == 'positive':
            return self.bonus_positions_positive
        elif phase in ['neutral', 'recovery']:
            return self.bonus_positions_neutral
        else:  # negative
            return self.bonus_positions_negative

    def step(self, action: int) -> Tuple[int, float, bool, TemporalEmotionalContext]:
        self.step_count += 1
        self.episode_step += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        # Boundary check
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        # Get current threats and bonuses
        threats = self._get_threats()
        bonuses = self._get_bonuses()

        # Compute threat distance
        if threats:
            threat_dist = min(self._distance(self.agent_pos, t) for t in threats)
        else:
            threat_dist = float('inf')

        goal_dist = self._distance(self.agent_pos, self.goal_pos)

        # Base reward
        reward = -0.01

        # Threat penalty
        threat_exposure = False
        if threat_dist < 1.5:
            reward -= 0.3
            threat_exposure = True

        # Bonus rewards
        for i, bonus in enumerate(bonuses):
            bonus_key = (self._get_current_phase(), i)
            if np.array_equal(self.agent_pos, bonus) and bonus_key not in self.collected_bonuses:
                reward += 0.3
                self.collected_bonuses.add(bonus_key)

        # Goal reached
        done = np.array_equal(self.agent_pos, self.goal_pos) or self.episode_step >= 100
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 1.0
            self.total_episodes += 1

        # Update temporal tracking
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 20:
            self.recent_rewards.pop(0)

        self.recent_threats.append(1 if threat_exposure else 0)
        if len(self.recent_threats) > 20:
            self.recent_threats.pop(0)

        if reward < 0:
            self.consecutive_negative += 1
            self.consecutive_positive = 0
        elif reward > 0.1:
            self.consecutive_positive += 1
            self.consecutive_negative = 0

        context = TemporalEmotionalContext(
            threat_distance=threat_dist,
            goal_distance=goal_dist,
            reward_obtained=reward,
            consecutive_negative=self.consecutive_negative,
            consecutive_positive=self.consecutive_positive,
            recent_threat_exposures=sum(self.recent_threats),
            time_since_last_reward=self.consecutive_negative,
            cumulative_reward=sum(self.recent_rewards),
            episode_length=self.episode_step
        )

        if done:
            self.episode_step = 0
            self.collected_bonuses.clear()

        return self._pos_to_state(self.agent_pos), reward, done, context

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4

    def get_phase(self) -> str:
        return self._get_current_phase()

    def render(self) -> str:
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]

        for t in self._get_threats():
            grid[t[0]][t[1]] = 'X'

        for b in self._get_bonuses():
            grid[b[0]][b[1]] = '+'

        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'

        return '\n'.join([' '.join(row) for row in grid])
