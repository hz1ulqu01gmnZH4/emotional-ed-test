"""Improved Disgust agents with directional repulsion (v2).

Key fix: Disgust should REPEL from contaminants, not boost argmax.
Based on feedback from GPT-5, Gemini, and Grok-4.
"""

import numpy as np
from typing import Dict, Set, Optional
import sys
sys.path.insert(0, '/home/ak/emotional-ed-test')
from gridworld_disgust import DisgustContext


class DisgustOnlyAgentV2:
    """Disgust agent with DIRECTIONAL REPULSION (fixed).

    Key fixes:
    1. Compute action AWAY from contaminant (like Grief's directional yearning)
    2. Penalize actions toward contaminant, not boost argmax
    3. Intrinsic disgust penalty (non-TD signal)

    Disgust properties (Rozin et al., 2008):
    - No habituation
    - One-contact rule
    - Contamination tracking
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 grid_size: int = 6, disgust_weight: float = 0.5):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.grid_size = grid_size
        self.disgust_weight = disgust_weight

        self.disgust_level = 0.0
        self.known_contaminated: Set[int] = set()  # States known contaminated
        self.contaminant_positions: Set[tuple] = set()  # (row, col) positions
        self.contamination_exposures = 0

    def _state_to_pos(self, state: int) -> tuple:
        """Convert state to (row, col)."""
        return (state // self.grid_size, state % self.grid_size)

    def _pos_to_state(self, pos: tuple) -> int:
        """Convert (row, col) to state."""
        return pos[0] * self.grid_size + pos[1]

    def _get_action_away_from_contaminant(self, state: int) -> Optional[int]:
        """Get action that moves AWAY from nearest contaminant.

        This is the key fix: directional repulsion instead of argmax boost.
        """
        if not self.contaminant_positions:
            return None

        current_pos = self._state_to_pos(state)

        # Find nearest contaminant
        min_dist = float('inf')
        nearest_contam = None
        for contam_pos in self.contaminant_positions:
            dist = abs(current_pos[0] - contam_pos[0]) + abs(current_pos[1] - contam_pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest_contam = contam_pos

        if nearest_contam is None:
            return None

        # Compute direction AWAY from contaminant
        row_diff = current_pos[0] - nearest_contam[0]  # Positive = agent above contam
        col_diff = current_pos[1] - nearest_contam[1]  # Positive = agent left of contam

        # Actions: 0=up(-1,0), 1=down(+1,0), 2=left(0,-1), 3=right(0,+1)
        # We want to move AWAY, so increase the difference
        if abs(row_diff) >= abs(col_diff):
            if row_diff < 0:
                return 0  # Go up (more negative row = further up)
            elif row_diff > 0:
                return 1  # Go down

        if col_diff < 0:
            return 2  # Go left
        elif col_diff > 0:
            return 3  # Go right

        return None  # Already at max distance or same position

    def _get_action_toward_contaminant(self, state: int) -> Optional[int]:
        """Get action that moves TOWARD nearest contaminant (to penalize)."""
        if not self.contaminant_positions:
            return None

        current_pos = self._state_to_pos(state)

        # Find nearest contaminant
        min_dist = float('inf')
        nearest_contam = None
        for contam_pos in self.contaminant_positions:
            dist = abs(current_pos[0] - contam_pos[0]) + abs(current_pos[1] - contam_pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest_contam = contam_pos

        if nearest_contam is None:
            return None

        row_diff = nearest_contam[0] - current_pos[0]
        col_diff = nearest_contam[1] - current_pos[1]

        if abs(row_diff) >= abs(col_diff):
            if row_diff < 0:
                return 0  # up
            elif row_diff > 0:
                return 1  # down

        if col_diff < 0:
            return 2  # left
        elif col_diff > 0:
            return 3  # right

        return None

    def _compute_disgust(self, state: int, context: DisgustContext) -> float:
        """Compute disgust level. NO habituation."""
        if context.contaminant_distance >= 3.0:
            base_disgust = 0.0
        else:
            base_disgust = 1.0 - context.contaminant_distance / 3.0

        # Amplify if in known contaminated state
        if state in self.known_contaminated:
            base_disgust = max(base_disgust, 0.8)

        # Amplify if agent is contaminated
        if context.is_contaminated:
            base_disgust = max(base_disgust, 0.5)

        return base_disgust

    def select_action(self, state: int) -> int:
        """Action selection with DIRECTIONAL disgust repulsion."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # KEY FIX: Directional repulsion from contaminant
        if self.disgust_level > 0.1 and self.contaminant_positions:
            # Boost action AWAY from contaminant
            away_action = self._get_action_away_from_contaminant(state)
            if away_action is not None:
                repulsion_boost = self.disgust_level * self.disgust_weight * 0.5
                q_values[away_action] += repulsion_boost

            # Penalize action TOWARD contaminant
            toward_action = self._get_action_toward_contaminant(state)
            if toward_action is not None:
                approach_penalty = self.disgust_level * self.disgust_weight * 0.3
                q_values[toward_action] -= approach_penalty

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: DisgustContext):
        """Update with disgust modulation."""
        self.disgust_level = self._compute_disgust(state, context)

        # Track contaminated states and positions
        if context.touched_contaminant or context.is_contaminated:
            self.known_contaminated.add(state)
            state_pos = self._state_to_pos(state)
            self.contaminant_positions.add(state_pos)
            self.contamination_exposures += 1

        # Intrinsic disgust penalty (non-TD, like curiosity bonus but negative)
        intrinsic_penalty = 0.0
        if context.touched_contaminant:
            intrinsic_penalty = -0.2 * self.disgust_level  # Extra penalty beyond reward

        effective_reward = reward + intrinsic_penalty

        # Strong learning from contamination events
        effective_lr = self.lr
        if context.touched_contaminant:
            effective_lr *= 2.0  # Learn strongly to avoid

        target = effective_reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])

    def reset_episode(self):
        self.disgust_level = 0.0
        # known_contaminated and positions persist

    def get_emotional_state(self) -> Dict:
        return {
            'disgust': self.disgust_level,
            'known_contaminated': len(self.known_contaminated),
            'exposures': self.contamination_exposures
        }


class FearOnlyAgentV2:
    """Fear agent with explicit habituation for comparison."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 grid_size: int = 6, habituation_rate: float = 0.05):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.grid_size = grid_size
        self.habituation_rate = habituation_rate

        self.fear_level = 0.0
        # Initialize all states with 0 exposures - no lazy dict.get() needed
        self.exposure_count: Dict[int, int] = {s: 0 for s in range(n_states)}

    def _state_to_pos(self, state: int) -> tuple:
        return (state // self.grid_size, state % self.grid_size)

    def _compute_fear(self, state: int, context: DisgustContext) -> float:
        """Fear with EXPLICIT habituation."""
        min_danger_dist = min(context.threat_distance, context.contaminant_distance)

        if min_danger_dist >= 3.0:
            base_fear = 0.0
        else:
            base_fear = 1.0 - min_danger_dist / 3.0

        # Explicit habituation based on state exposure
        assert state in self.exposure_count, f"BUG: State {state} not in exposure_count"
        exposures = self.exposure_count[state]
        habituation = np.exp(-self.habituation_rate * exposures)

        return base_fear * habituation

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Fear reduces exploration (lower epsilon effectively)
        if self.fear_level > 0.3:
            # Boost best action
            q_values[np.argmax(q_values)] *= (1 + self.fear_level * 0.5)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: DisgustContext):
        self.fear_level = self._compute_fear(state, context)

        # Record exposure for habituation
        if context.threat_distance < 2.0 or context.contaminant_distance < 2.0:
            self.exposure_count[state] += 1

        effective_lr = self.lr
        if self.fear_level > 0.3 and reward < 0:
            effective_lr *= (1 + self.fear_level)

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])

    def reset_episode(self):
        self.fear_level = 0.0
        # Habituation persists

    def get_emotional_state(self) -> Dict:
        return {
            'fear': self.fear_level,
            'total_exposures': sum(self.exposure_count.values())
        }


class IntegratedFearDisgustAgentV2:
    """Integrated agent with proper directional disgust and habituating fear."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 grid_size: int = 6, habituation_rate: float = 0.05):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.grid_size = grid_size
        self.habituation_rate = habituation_rate

        # Fear with habituation
        self.fear_level = 0.0
        self.threat_exposures: Dict[int, int] = {}

        # Disgust without habituation
        self.disgust_level = 0.0
        self.known_contaminated: Set[int] = set()
        self.contaminant_positions: Set[tuple] = set()

    def _state_to_pos(self, state: int) -> tuple:
        return (state // self.grid_size, state % self.grid_size)

    def _get_action_away_from_position(self, state: int, target_positions: Set[tuple]) -> Optional[int]:
        """Get action away from nearest target position."""
        if not target_positions:
            return None

        current_pos = self._state_to_pos(state)

        min_dist = float('inf')
        nearest = None
        for pos in target_positions:
            dist = abs(current_pos[0] - pos[0]) + abs(current_pos[1] - pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest = pos

        if nearest is None:
            return None

        row_diff = current_pos[0] - nearest[0]
        col_diff = current_pos[1] - nearest[1]

        if abs(row_diff) >= abs(col_diff):
            if row_diff < 0:
                return 0
            elif row_diff > 0:
                return 1
        if col_diff < 0:
            return 2
        elif col_diff > 0:
            return 3

        return None

    def _compute_fear(self, state: int, context: DisgustContext) -> float:
        """Fear with habituation (threat only)."""
        if context.threat_distance >= 3.0:
            base_fear = 0.0
        else:
            base_fear = 1.0 - context.threat_distance / 3.0

        exposures = self.threat_exposures.get(state, 0)
        habituation = np.exp(-self.habituation_rate * exposures)

        return base_fear * habituation

    def _compute_disgust(self, state: int, context: DisgustContext) -> float:
        """Disgust without habituation (contaminant only)."""
        if context.contaminant_distance >= 3.0:
            base_disgust = 0.0
        else:
            base_disgust = 1.0 - context.contaminant_distance / 3.0

        if state in self.known_contaminated:
            base_disgust = max(base_disgust, 0.8)

        if context.is_contaminated:
            base_disgust = max(base_disgust, 0.5)

        return base_disgust

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Disgust: directional repulsion
        if self.disgust_level > 0.1 and self.contaminant_positions:
            away_action = self._get_action_away_from_position(state, self.contaminant_positions)
            if away_action is not None:
                q_values[away_action] += self.disgust_level * 0.3

        # Fear: boost best action (reduces exploration)
        if self.fear_level > 0.2:
            q_values[np.argmax(q_values)] *= (1 + self.fear_level * 0.3)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: DisgustContext):
        self.fear_level = self._compute_fear(state, context)
        self.disgust_level = self._compute_disgust(state, context)

        # Track threat exposures (for fear habituation)
        if context.threat_distance < 2.0:
            self.threat_exposures[state] = self.threat_exposures.get(state, 0) + 1

        # Track contamination (no habituation)
        if context.touched_contaminant:
            self.known_contaminated.add(state)
            self.contaminant_positions.add(self._state_to_pos(state))

        effective_lr = self.lr
        if context.touched_contaminant:
            effective_lr *= 1.5

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])

    def reset_episode(self):
        self.fear_level = 0.0
        self.disgust_level = 0.0

    def get_emotional_state(self) -> Dict:
        return {
            'fear': self.fear_level,
            'disgust': self.disgust_level,
            'threat_habituation': sum(self.threat_exposures.values()),
            'known_contaminated': len(self.known_contaminated)
        }
