"""Agents for approach-avoidance conflict testing."""

import numpy as np
from typing import List, Optional
from gridworld_conflict import EmotionalContext

class StandardQLearner:
    """Standard Q-learning baseline."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.choices: List[str] = []  # 'safe' or 'risky' first

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        pass


class FearModule:
    """Fear signal from threat proximity."""

    def __init__(self, safe_distance: float = 3.0, max_fear: float = 1.0):
        self.safe_distance = safe_distance
        self.max_fear = max_fear

    def compute(self, context: EmotionalContext) -> float:
        if context.threat_distance >= self.safe_distance:
            return 0.0
        return self.max_fear * (1 - context.threat_distance / self.safe_distance)


class ApproachModule:
    """Approach/desire signal from high-value reward proximity.

    Models the "wanting" that competes with fear.
    Increases as agent approaches valuable reward.
    """

    def __init__(self, max_approach: float = 1.0, sensitivity: float = 2.0):
        self.max_approach = max_approach
        self.sensitivity = sensitivity

    def compute(self, context: EmotionalContext) -> float:
        if not context.near_high_value:
            return 0.0
        # Approach increases as reward gets closer
        proximity = max(0, 1 - context.reward_distance / 3.0)
        return self.max_approach * proximity


class FearDominantAgent:
    """Agent where fear dominates approach (risk-averse)."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 1.0, approach_weight: float = 0.3):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.fear_module = FearModule()
        self.approach_module = ApproachModule()
        self.fear_weight = fear_weight
        self.approach_weight = approach_weight

        self.current_fear = 0.0

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Fear biases toward safe (high-Q) actions
        if self.current_fear > 0:
            q_min, q_max = q_values.min(), q_values.max()
            if q_max > q_min:
                normalized = (q_values - q_min) / (q_max - q_min)
                q_values += self.current_fear * self.fear_weight * normalized

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        fear = self.fear_module.compute(context)
        approach = self.approach_module.compute(context)
        self.current_fear = fear

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        # Fear increases learning from negative outcomes
        # Approach has smaller effect
        net_signal = fear * self.fear_weight - approach * self.approach_weight
        if net_signal > 0 and delta < 0:
            # Fear dominant: amplify negative learning
            delta *= (1 + net_signal)

        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        self.current_fear = 0.0


class ApproachDominantAgent:
    """Agent where approach dominates fear (risk-seeking)."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 0.3, approach_weight: float = 1.0):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.fear_module = FearModule()
        self.approach_module = ApproachModule()
        self.fear_weight = fear_weight
        self.approach_weight = approach_weight

        self.current_approach = 0.0

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Approach biases toward actions leading to reward
        # (simplified: boost exploration when approach is high)
        if self.current_approach > 0.3:
            # Increase temperature / reduce exploitation
            q_values += np.random.randn(len(q_values)) * self.current_approach * 0.1

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        fear = self.fear_module.compute(context)
        approach = self.approach_module.compute(context)
        self.current_approach = approach

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        # Approach increases learning from positive outcomes
        # Fear has smaller effect
        net_signal = approach * self.approach_weight - fear * self.fear_weight
        if net_signal > 0 and delta > 0:
            # Approach dominant: amplify positive learning
            delta *= (1 + net_signal)

        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        self.current_approach = 0.0


class BalancedConflictAgent:
    """Agent with balanced fear and approach - shows conflict behavior.

    Models the "STN brake" that slows decisions under conflict.
    When fear ≈ approach, agent hesitates (explores more).
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 0.7, approach_weight: float = 0.7,
                 conflict_threshold: float = 0.3):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.fear_module = FearModule()
        self.approach_module = ApproachModule()
        self.fear_weight = fear_weight
        self.approach_weight = approach_weight
        self.conflict_threshold = conflict_threshold

        self.current_fear = 0.0
        self.current_approach = 0.0
        self.conflict_level = 0.0

    def select_action(self, state: int) -> int:
        # Conflict detection: when fear and approach are both high
        conflict = min(self.current_fear, self.current_approach)
        self.conflict_level = conflict

        # Under conflict, increase exploration (STN brake)
        effective_epsilon = self.epsilon
        if conflict > self.conflict_threshold:
            effective_epsilon = min(0.5, self.epsilon + conflict * 0.3)

        if np.random.random() < effective_epsilon:
            return np.random.randint(self.Q.shape[1])

        return np.argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        fear = self.fear_module.compute(context)
        approach = self.approach_module.compute(context)
        self.current_fear = fear
        self.current_approach = approach

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        # Under conflict, slow down learning (uncertainty)
        conflict = min(fear, approach)
        if conflict > self.conflict_threshold:
            delta *= (1 - conflict * 0.3)

        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        self.current_fear = 0.0
        self.current_approach = 0.0
        self.conflict_level = 0.0
