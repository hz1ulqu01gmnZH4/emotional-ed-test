"""Agents for disgust channel testing.

Disgust differs from fear in key ways:
- Fear habituates; disgust doesn't
- Fear is about threat; disgust is about contamination
- Contamination spreads; threats don't
"""

import numpy as np
from typing import Dict, Set
from gridworld_disgust import DisgustContext


class StandardQLearner:
    """Baseline Q-learning without disgust channel."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: DisgustContext):
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += self.lr * (target - self.Q[state, action])

    def reset_episode(self):
        pass

    def get_emotional_state(self) -> Dict:
        return {'fear': 0.0, 'disgust': 0.0}


class FearOnlyAgent:
    """Agent with fear but no disgust channel.

    Treats all negative stimuli as threats.
    Fear habituates with repeated exposure.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_habituation: float = 0.95):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.fear_habituation = fear_habituation

        self.fear_level = 0.0
        self.threat_exposures = 0

    def _compute_fear(self, context: DisgustContext) -> float:
        """Fear from any proximity to negative stimulus."""
        # Fear both threat and contaminant equally
        min_danger_dist = min(context.threat_distance, context.contaminant_distance)

        if min_danger_dist >= 3.0:
            base_fear = 0.0
        else:
            base_fear = 1.0 - min_danger_dist / 3.0

        # Fear habituates with exposure
        habituation_factor = self.fear_habituation ** self.threat_exposures
        return base_fear * habituation_factor

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        if self.fear_level > 0.3:
            q_values[np.argmax(q_values)] *= (1 + self.fear_level)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: DisgustContext):
        self.fear_level = self._compute_fear(context)

        if context.threat_distance < 2.0 or context.contaminant_distance < 2.0:
            self.threat_exposures += 1

        effective_lr = self.lr
        if self.fear_level > 0.3 and reward < 0:
            effective_lr *= (1 + self.fear_level)

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])

    def reset_episode(self):
        self.fear_level = 0.0
        # Habituation persists across episodes

    def get_emotional_state(self) -> Dict:
        return {'fear': self.fear_level, 'disgust': 0.0, 'habituation': self.threat_exposures}


class DisgustOnlyAgent:
    """Agent with disgust but no fear channel.

    Key disgust properties (Rozin et al., 2008):
    1. No habituation - disgust persists
    2. One-contact rule - any contact is full contamination
    3. Contagion tracking - remembers contaminated locations
    4. Asymmetric valuation - contaminated > neutral >>> clean
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.disgust_level = 0.0
        self.known_contaminated: Set[int] = set()  # States known to be contaminated
        self.contamination_exposures = 0  # Unlike fear, this doesn't reduce disgust

    def _compute_disgust(self, state: int, context: DisgustContext) -> float:
        """Disgust from contamination proximity. NO habituation."""
        if context.contaminant_distance >= 3.0:
            base_disgust = 0.0
        else:
            base_disgust = 1.0 - context.contaminant_distance / 3.0

        # Disgust amplified if state is known contaminated
        if state in self.known_contaminated:
            base_disgust = max(base_disgust, 0.8)

        # Disgust if agent is contaminated
        if context.is_contaminated:
            base_disgust = max(base_disgust, 0.5)

        # NO habituation - this is key difference from fear
        return base_disgust

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Disgust strongly biases away from known contaminated states
        if self.disgust_level > 0.2:
            # Avoid all actions that might lead to contamination
            q_values[np.argmax(q_values)] *= (1 + self.disgust_level * 1.5)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: DisgustContext):
        self.disgust_level = self._compute_disgust(state, context)

        # Track contaminated states
        if context.touched_contaminant or context.is_contaminated:
            self.known_contaminated.add(state)
            self.contamination_exposures += 1

        # Disgust modulates learning differently than fear
        effective_lr = self.lr

        # Strong learning from contamination events (don't repeat)
        if context.touched_contaminant:
            effective_lr *= 2.0  # Learn strongly to avoid

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])

    def reset_episode(self):
        self.disgust_level = 0.0
        # known_contaminated persists - contamination memory doesn't clear

    def get_emotional_state(self) -> Dict:
        return {
            'fear': 0.0,
            'disgust': self.disgust_level,
            'known_contaminated': len(self.known_contaminated),
            'exposures': self.contamination_exposures
        }


class IntegratedFearDisgustAgent:
    """Agent with both fear and disgust channels.

    Tests whether fear and disgust operate differently:
    - Fear: Habituates, immediate avoidance
    - Disgust: Doesn't habituate, contamination tracking
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_habituation: float = 0.95):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.fear_habituation = fear_habituation

        # Fear channel
        self.fear_level = 0.0
        self.threat_exposures = 0

        # Disgust channel
        self.disgust_level = 0.0
        self.known_contaminated: Set[int] = set()

    def _compute_fear(self, context: DisgustContext) -> float:
        """Fear from threat only (not contaminant)."""
        if context.threat_distance >= 3.0:
            base_fear = 0.0
        else:
            base_fear = 1.0 - context.threat_distance / 3.0

        # Fear habituates
        habituation_factor = self.fear_habituation ** self.threat_exposures
        return base_fear * habituation_factor

    def _compute_disgust(self, state: int, context: DisgustContext) -> float:
        """Disgust from contamination only (not threat). No habituation."""
        if context.contaminant_distance >= 3.0:
            base_disgust = 0.0
        else:
            base_disgust = 1.0 - context.contaminant_distance / 3.0

        if state in self.known_contaminated:
            base_disgust = max(base_disgust, 0.8)

        if context.is_contaminated:
            base_disgust = max(base_disgust, 0.5)

        # NO habituation
        return base_disgust

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Both fear and disgust bias action selection
        combined_avoidance = max(self.fear_level, self.disgust_level)
        if combined_avoidance > 0.2:
            q_values[np.argmax(q_values)] *= (1 + combined_avoidance)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: DisgustContext):
        self.fear_level = self._compute_fear(context)
        self.disgust_level = self._compute_disgust(state, context)

        # Track exposures
        if context.threat_distance < 2.0:
            self.threat_exposures += 1

        if context.touched_contaminant:
            self.known_contaminated.add(state)

        # Combined emotional modulation
        effective_lr = self.lr

        if self.fear_level > 0.3 and reward < 0:
            effective_lr *= (1 + self.fear_level * 0.5)

        if context.touched_contaminant:
            effective_lr *= 1.5  # Disgust: strong one-shot learning

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])

    def reset_episode(self):
        self.fear_level = 0.0
        self.disgust_level = 0.0
        # Habituation and contamination memory persist

    def get_emotional_state(self) -> Dict:
        return {
            'fear': self.fear_level,
            'disgust': self.disgust_level,
            'threat_exposures': self.threat_exposures,
            'known_contaminated': len(self.known_contaminated)
        }
