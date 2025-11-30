"""Agents for transfer/generalization testing.

Tests whether emotional learning transfers to novel situations.
"""

import numpy as np
from typing import Dict
from gridworld_transfer import TransferContext


class StandardQLearner:
    """Baseline Q-learning - no emotional transfer."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states

    def select_action(self, state: int) -> int:
        if state >= self.n_states:
            return np.random.randint(self.Q.shape[1])
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: TransferContext):
        if state >= self.n_states or next_state >= self.n_states:
            return
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += self.lr * (target - self.Q[state, action])

    def reset_episode(self):
        pass

    def resize_q_table(self, new_n_states: int):
        """Resize Q-table for different environment sizes."""
        if new_n_states > self.n_states:
            new_Q = np.zeros((new_n_states, self.Q.shape[1]))
            new_Q[:self.n_states] = self.Q
            self.Q = new_Q
            self.n_states = new_n_states


class EmotionalTransferAgent:
    """Agent that can transfer emotional learning.

    Key mechanism: Fear is learned as response to threat proximity,
    not specific state values. This allows generalization.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 0.8):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.fear_weight = fear_weight
        self.n_states = n_states

        # Fear is learned based on FEATURES, not states
        # This enables transfer
        self.fear_threshold = 2.0  # Learned safe distance
        self.fear_learning_rate = 0.1
        self.current_fear = 0.0

        # Track threat experiences
        self.threat_experiences = 0
        self.near_threat_outcomes = []

    def _compute_fear(self, context: TransferContext) -> float:
        """Compute fear based on proximity (generalizable feature)."""
        if context.threat_distance >= self.fear_threshold:
            return 0.0
        return 1.0 - context.threat_distance / self.fear_threshold

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        if state >= self.n_states:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Fear modulates action selection (transfers to novel situations)
        if self.current_fear > 0.3:
            q_values[np.argmax(q_values)] *= (1 + self.current_fear * self.fear_weight)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: TransferContext):
        self.current_fear = self._compute_fear(context)

        # Learn fear threshold from experience
        if context.threat_distance < 2.0:
            self.threat_experiences += 1
            self.near_threat_outcomes.append(reward)

            # If getting hurt near threats, increase fear threshold
            if reward < -0.2:
                self.fear_threshold = min(3.0, self.fear_threshold + self.fear_learning_rate)
            # If not getting hurt, can reduce threshold
            elif reward > 0:
                self.fear_threshold = max(1.0, self.fear_threshold - self.fear_learning_rate * 0.5)

        # Q-learning update
        if state < self.n_states and next_state < self.n_states:
            effective_lr = self.lr
            if self.current_fear > 0.3 and reward < 0:
                effective_lr *= (1 + self.current_fear * self.fear_weight)

            target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
            self.Q[state, action] += effective_lr * (target - self.Q[state, action])

    def reset_episode(self):
        self.current_fear = 0.0

    def resize_q_table(self, new_n_states: int):
        """Resize Q-table for different environment sizes."""
        if new_n_states > self.n_states:
            new_Q = np.zeros((new_n_states, self.Q.shape[1]))
            new_Q[:self.n_states] = self.Q
            self.Q = new_Q
            self.n_states = new_n_states

    def get_transfer_state(self) -> Dict:
        return {
            'fear_threshold': self.fear_threshold,
            'threat_experiences': self.threat_experiences,
            'current_fear': self.current_fear
        }


class NoTransferAgent:
    """Agent that learns state-specific responses (no transfer)."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states

        # State-specific fear (doesn't transfer)
        self.state_fear = np.zeros(n_states)
        self.current_fear = 0.0

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        if state >= self.n_states:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # State-specific fear
        if state < len(self.state_fear) and self.state_fear[state] > 0.3:
            q_values[np.argmax(q_values)] *= (1 + self.state_fear[state])

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: TransferContext):
        # Learn state-specific fear
        if state < len(self.state_fear):
            if reward < -0.2:
                self.state_fear[state] = min(1.0, self.state_fear[state] + 0.2)
            else:
                self.state_fear[state] *= 0.95

            self.current_fear = self.state_fear[state]

        # Q-learning update
        if state < self.n_states and next_state < self.n_states:
            target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
            self.Q[state, action] += self.lr * (target - self.Q[state, action])

    def reset_episode(self):
        self.current_fear = 0.0

    def resize_q_table(self, new_n_states: int):
        """Resize Q-table and fear array for different environment sizes."""
        if new_n_states > self.n_states:
            new_Q = np.zeros((new_n_states, self.Q.shape[1]))
            new_Q[:self.n_states] = self.Q
            self.Q = new_Q

            new_fear = np.zeros(new_n_states)
            new_fear[:len(self.state_fear)] = self.state_fear
            self.state_fear = new_fear

            self.n_states = new_n_states

    def get_transfer_state(self) -> Dict:
        return {
            'fearful_states': np.sum(self.state_fear > 0.3),
            'current_fear': self.current_fear
        }
