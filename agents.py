"""Q-learning agents: standard and emotional ED variants."""

import numpy as np
from typing import Dict, Optional
from gridworld import EmotionalContext

class StandardQLearner:
    """Vanilla Q-learning with single reward signal."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: Optional[EmotionalContext] = None):
        """Standard TD update. Context ignored."""
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.lr * delta


class FearModule:
    """Computes fear signal based on threat proximity."""

    def __init__(self, safe_distance: float = 3.0, max_fear: float = 1.0):
        self.safe_distance = safe_distance
        self.max_fear = max_fear

    def compute(self, context: EmotionalContext) -> float:
        """Fear increases as threat gets closer."""
        if context.threat_distance >= self.safe_distance:
            return 0.0
        # Linear ramp: fear = max_fear * (1 - dist/safe_dist)
        return self.max_fear * (1 - context.threat_distance / self.safe_distance)


class AngerModule:
    """Computes anger/frustration signal from goal blocking."""

    def __init__(self, frustration_decay: float = 0.9):
        self.frustration = 0.0
        self.decay = frustration_decay

    def compute(self, context: EmotionalContext) -> float:
        """Frustration accumulates when blocked, decays otherwise."""
        if context.was_blocked:
            self.frustration = min(1.0, self.frustration + 0.3)
        else:
            self.frustration *= self.decay
        return self.frustration

    def reset(self):
        self.frustration = 0.0


class EmotionalEDAgent:
    """Q-learning with emotional ED channels modulating learning."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 0.5, anger_weight: float = 0.3):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # Emotional modules
        self.fear_module = FearModule()
        self.anger_module = AngerModule()

        # Channel weights
        self.fear_weight = fear_weight
        self.anger_weight = anger_weight

        # Track emotional state for action selection
        self.current_fear = 0.0

    def select_action(self, state: int) -> int:
        """Action selection modulated by fear (avoidance bias)."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        # Fear biases toward actions with higher Q (safer choices)
        # Anger increases action temperature (more exploitation)
        q_values = self.Q[state].copy()

        # Fear adds penalty to low-Q actions (risk aversion)
        if self.current_fear > 0:
            q_min, q_max = q_values.min(), q_values.max()
            if q_max > q_min:
                normalized = (q_values - q_min) / (q_max - q_min)
                # Fear penalizes risky (low-Q) actions
                q_values += self.current_fear * normalized

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        """ED-style update: multiple channels modulate learning signal."""
        # Compute emotional signals
        fear = self.fear_module.compute(context)
        anger = self.anger_module.compute(context)
        self.current_fear = fear  # Store for action selection

        # Standard TD error
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        # ED broadcast: fear and anger modulate learning rate
        # Fear: increases learning from negative outcomes (danger awareness)
        # Anger: increases learning rate overall (approach motivation)
        fear_modulation = 1.0 + self.fear_weight * fear * (1 if delta < 0 else 0.5)
        anger_modulation = 1.0 + self.anger_weight * anger

        effective_lr = self.lr * fear_modulation * anger_modulation
        self.Q[state, action] += effective_lr * delta

    def reset_episode(self):
        """Reset per-episode emotional state."""
        self.anger_module.reset()
        self.current_fear = 0.0
