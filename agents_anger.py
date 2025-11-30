"""Agents for anger/frustration testing - focus on persistence behavior."""

import numpy as np
from typing import Optional, List
from gridworld_anger import EmotionalContext

class StandardQLearner:
    """Vanilla Q-learning - no emotional modulation."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # Track behavior for analysis
        self.action_history: List[int] = []
        self.block_history: List[bool] = []

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.Q.shape[1])
        else:
            action = np.argmax(self.Q[state])
        self.action_history.append(action)
        return action

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        """Standard TD update."""
        self.block_history.append(context.was_blocked)

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        """Reset per-episode tracking."""
        self.action_history = []
        self.block_history = []


class AngerModule:
    """Computes anger/frustration from goal obstruction.

    Key insight from Davidson: Anger is approach-motivated negative affect.
    Unlike fear (which motivates withdrawal), anger increases approach vigor.
    """

    def __init__(self, frustration_buildup: float = 0.3,
                 frustration_decay: float = 0.8,
                 goal_proximity_weight: float = 0.5):
        self.frustration = 0.0
        self.buildup = frustration_buildup
        self.decay = frustration_decay
        self.goal_weight = goal_proximity_weight

    def compute(self, context: EmotionalContext) -> float:
        """
        Frustration = f(blocked, goal_proximity, consecutive_blocks)

        Closer to goal + blocked = MORE frustration (Berkowitz)
        """
        if context.was_blocked:
            # Frustration increases more when close to goal
            proximity_factor = 1.0 / (1.0 + context.goal_distance)
            consecutive_factor = 1.0 + 0.2 * min(context.consecutive_blocks, 5)

            increment = self.buildup * proximity_factor * consecutive_factor
            self.frustration = min(1.0, self.frustration + increment)
        else:
            self.frustration *= self.decay

        return self.frustration

    def reset(self):
        self.frustration = 0.0


class FrustrationEDAgent:
    """Q-learning with anger/frustration channel.

    Key behavioral prediction:
    - Standard agent: Blocked → quickly learns to avoid → finds alternate path
    - Frustration agent: Blocked → frustration builds → PERSISTS at obstacle longer
                        → eventually reroutes but with more attempts first

    This models "approach under negative valence" - anger makes you try harder,
    not give up immediately.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 anger_weight: float = 0.5):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # Anger module
        self.anger_module = AngerModule()
        self.anger_weight = anger_weight

        # Track for analysis
        self.action_history: List[int] = []
        self.block_history: List[bool] = []
        self.frustration_history: List[float] = []
        self.current_frustration = 0.0

    def select_action(self, state: int) -> int:
        """Action selection with frustration-driven persistence.

        High frustration → bias toward repeating blocked action (persistence)
        This is counter-intuitive but models real anger behavior.
        """
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.Q.shape[1])
        else:
            q_values = self.Q[state].copy()

            # Frustration effect: boost recently-tried actions
            # (approach motivation = keep trying what was blocked)
            if self.current_frustration > 0.3 and len(self.action_history) > 0:
                last_action = self.action_history[-1]
                # Boost the action that was just blocked
                persistence_boost = self.current_frustration * 0.5
                q_values[last_action] += persistence_boost

            action = np.argmax(q_values)

        self.action_history.append(action)
        return action

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        """ED-style update with anger modulation."""
        self.block_history.append(context.was_blocked)

        # Compute frustration
        frustration = self.anger_module.compute(context)
        self.current_frustration = frustration
        self.frustration_history.append(frustration)

        # Standard TD error
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        # Anger modulation: frustration increases learning rate
        # (heightened attention/arousal under anger)
        anger_modulation = 1.0 + self.anger_weight * frustration

        # But ALSO: anger slows down negative learning for blocked action
        # (persistence = not immediately learning to avoid)
        if context.was_blocked and delta < 0:
            # Reduce negative learning when frustrated (don't give up so fast)
            persistence_factor = 1.0 - (0.5 * frustration)
            delta *= persistence_factor

        effective_lr = self.lr * anger_modulation
        self.Q[state, action] += effective_lr * delta

    def reset_episode(self):
        """Reset per-episode state."""
        self.anger_module.reset()
        self.action_history = []
        self.block_history = []
        self.frustration_history = []
        self.current_frustration = 0.0
