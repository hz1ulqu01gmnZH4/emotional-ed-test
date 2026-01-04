"""Feature-based Q-learning agents for Transfer and generalization (v2).

Key insight: Emotions should modulate REPRESENTATIONS, not state-specific Q-values.
This enables generalization to novel threat locations.

Based on feedback from GPT-5, Gemini, and Grok-4.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FeatureContext:
    """Context with feature information for generalization."""
    threat_distance: float
    goal_distance: float
    threat_direction: Tuple[int, int]  # (row_diff, col_diff) to threat
    goal_direction: Tuple[int, int]  # (row_diff, col_diff) to goal
    near_threat: bool
    near_wall: bool


class FeatureBasedFearAgent:
    """Fear agent using feature-based Q-function.

    Q(s, a) → Q(φ(s, context), a)

    Features enable generalization:
    - threat_distance generalizes across locations
    - fear_level provides emotional context
    - action_toward_threat is a learned feature

    This fixes the Transfer problem: fear of threats generalizes
    because it's encoded in features, not state indices.
    """

    def __init__(self, n_actions: int = 4, lr: float = 0.05,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 n_features: int = 8, fear_weight: float = 0.5):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.fear_weight = fear_weight

        # Feature-based weights instead of tabular Q
        # Shape: (n_features, n_actions)
        self.W = np.zeros((n_features, n_actions))

        # Fear module state
        self.fear_level = 0.0

    def _compute_features(self, state_pos: Tuple[int, int],
                         context: FeatureContext, action: int) -> np.ndarray:
        """Compute feature vector φ(s, context, a).

        These features GENERALIZE across states:
        - Threat distance is position-invariant
        - Action-threat alignment enables directional learning
        """
        # Action directions: 0=up(-1,0), 1=down(+1,0), 2=left(0,-1), 3=right(0,+1)
        action_deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        action_delta = action_deltas[action]

        # Feature 1: Bias
        f_bias = 1.0

        # Feature 2: Inverse threat distance (higher = closer to threat)
        f_threat_proximity = 1.0 / (1.0 + context.threat_distance)

        # Feature 3: Inverse goal distance
        f_goal_proximity = 1.0 / (1.0 + context.goal_distance)

        # Feature 4: Action moves TOWARD threat (dot product)
        # Normalize threat direction
        threat_dir = context.threat_direction
        threat_mag = max(1, abs(threat_dir[0]) + abs(threat_dir[1]))
        norm_threat = (threat_dir[0] / threat_mag, threat_dir[1] / threat_mag)
        f_toward_threat = (action_delta[0] * norm_threat[0] +
                          action_delta[1] * norm_threat[1])

        # Feature 5: Action moves TOWARD goal
        goal_dir = context.goal_direction
        goal_mag = max(1, abs(goal_dir[0]) + abs(goal_dir[1]))
        norm_goal = (goal_dir[0] / goal_mag, goal_dir[1] / goal_mag)
        f_toward_goal = (action_delta[0] * norm_goal[0] +
                        action_delta[1] * norm_goal[1])

        # Feature 6: Near threat (binary)
        f_near_threat = 1.0 if context.near_threat else 0.0

        # Feature 7: Fear level (emotional context)
        f_fear = self.fear_level

        # Feature 8: Fear × toward_threat interaction
        # This is KEY: fear should specifically discourage approach
        f_fear_approach = self.fear_level * max(0, f_toward_threat)

        return np.array([
            f_bias,
            f_threat_proximity,
            f_goal_proximity,
            f_toward_threat,
            f_toward_goal,
            f_near_threat,
            f_fear,
            f_fear_approach
        ])

    def _compute_fear(self, context: FeatureContext) -> float:
        """Compute fear level from context."""
        if context.threat_distance >= 3.0:
            return 0.0
        return 1.0 - context.threat_distance / 3.0

    def Q(self, state_pos: Tuple[int, int], context: FeatureContext, action: int) -> float:
        """Compute Q-value using features."""
        φ = self._compute_features(state_pos, context, action)
        return np.dot(self.W[:, action], φ)

    def select_action(self, state_pos: Tuple[int, int], context: FeatureContext) -> int:
        """Select action using feature-based Q-values."""
        self.fear_level = self._compute_fear(context)

        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        q_values = np.array([self.Q(state_pos, context, a) for a in range(self.n_actions)])
        return np.argmax(q_values)

    def update(self, state_pos: Tuple[int, int], action: int, reward: float,
               next_state_pos: Tuple[int, int], done: bool,
               context: FeatureContext, next_context: FeatureContext):
        """Update feature weights with TD learning."""
        self.fear_level = self._compute_fear(context)

        # Current Q
        φ = self._compute_features(state_pos, context, action)
        Q_current = np.dot(self.W[:, action], φ)

        # Next Q (max over actions)
        if done:
            Q_next_max = 0.0
        else:
            Q_next_max = max(self.Q(next_state_pos, next_context, a)
                           for a in range(self.n_actions))

        # TD error
        td_error = reward + self.gamma * Q_next_max - Q_current

        # Fear modulation: increase learning from negative near threat
        effective_lr = self.lr
        if self.fear_level > 0.2 and td_error < 0:
            effective_lr *= (1 + self.fear_level * self.fear_weight)

        # Gradient update: W[:, action] += lr * td_error * φ
        self.W[:, action] += effective_lr * td_error * φ

    def reset_episode(self):
        self.fear_level = 0.0

    def get_emotional_state(self) -> Dict:
        return {'fear': self.fear_level}


class FeatureBasedTransferAgent:
    """Agent specifically designed for Transfer testing.

    Key properties:
    1. Features include threat-relative information
    2. Fear is encoded as a feature, not state-specific modulation
    3. Q-function generalizes via linear function approximation
    """

    def __init__(self, n_actions: int = 4, lr: float = 0.03,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 0.5):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.fear_weight = fear_weight

        # Extended feature set for better generalization
        n_features = 12
        self.W = np.zeros((n_features, n_actions))

        self.fear_level = 0.0
        self.cumulative_fear = 0.0  # Track fear history

    def _compute_features(self, state_pos: Tuple[int, int],
                         context: FeatureContext, action: int) -> np.ndarray:
        """Extended feature set for transfer."""
        action_deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        action_delta = action_deltas[action]

        threat_dir = context.threat_direction
        goal_dir = context.goal_direction

        # Normalize directions
        threat_mag = max(1, abs(threat_dir[0]) + abs(threat_dir[1]))
        goal_mag = max(1, abs(goal_dir[0]) + abs(goal_dir[1]))

        # Action alignments
        toward_threat = (action_delta[0] * threat_dir[0] / threat_mag +
                        action_delta[1] * threat_dir[1] / threat_mag)
        toward_goal = (action_delta[0] * goal_dir[0] / goal_mag +
                      action_delta[1] * goal_dir[1] / goal_mag)

        features = [
            1.0,  # Bias
            1.0 / (1.0 + context.threat_distance),  # Threat proximity
            1.0 / (1.0 + context.goal_distance),  # Goal proximity
            toward_threat,  # Action toward threat
            toward_goal,  # Action toward goal
            1.0 if context.near_threat else 0.0,  # Near threat binary
            1.0 if context.near_wall else 0.0,  # Near wall binary
            self.fear_level,  # Current fear
            self.cumulative_fear / 100.0,  # Fear history (normalized)
            self.fear_level * max(0, toward_threat),  # Fear × approach interaction
            self.fear_level * (1.0 / (1.0 + context.threat_distance)),  # Fear × proximity
            (1 - self.fear_level) * toward_goal  # Safety × goal-seeking
        ]

        return np.array(features)

    def _compute_fear(self, context: FeatureContext) -> float:
        if context.threat_distance >= 3.0:
            return 0.0
        return 1.0 - context.threat_distance / 3.0

    def Q(self, state_pos: Tuple[int, int], context: FeatureContext, action: int) -> float:
        φ = self._compute_features(state_pos, context, action)
        return np.dot(self.W[:, action], φ)

    def select_action(self, state_pos: Tuple[int, int], context: FeatureContext) -> int:
        self.fear_level = self._compute_fear(context)
        self.cumulative_fear += self.fear_level

        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        q_values = np.array([self.Q(state_pos, context, a) for a in range(self.n_actions)])
        return np.argmax(q_values)

    def update(self, state_pos: Tuple[int, int], action: int, reward: float,
               next_state_pos: Tuple[int, int], done: bool,
               context: FeatureContext, next_context: FeatureContext):
        self.fear_level = self._compute_fear(context)

        φ = self._compute_features(state_pos, context, action)
        Q_current = np.dot(self.W[:, action], φ)

        if done:
            Q_next_max = 0.0
        else:
            Q_next_max = max(self.Q(next_state_pos, next_context, a)
                           for a in range(self.n_actions))

        td_error = reward + self.gamma * Q_next_max - Q_current

        # Fear modulation
        effective_lr = self.lr
        if self.fear_level > 0.2 and td_error < 0:
            effective_lr *= (1 + self.fear_level * self.fear_weight)

        self.W[:, action] += effective_lr * td_error * φ

    def reset_episode(self):
        self.fear_level = 0.0
        # cumulative_fear persists

    def get_weights(self) -> np.ndarray:
        """Return learned weights for analysis."""
        return self.W.copy()

    def get_emotional_state(self) -> Dict:
        return {
            'fear': self.fear_level,
            'cumulative_fear': self.cumulative_fear
        }


class TabularBaselineAgent:
    """Tabular Q-learning baseline for comparison."""

    def __init__(self, n_states: int, n_actions: int = 4,
                 lr: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def select_action(self, state: int, context: FeatureContext = None) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context=None, next_context=None):
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += self.lr * (target - self.Q[state, action])

    def reset_episode(self):
        pass

    def get_emotional_state(self) -> Dict:
        return {}
