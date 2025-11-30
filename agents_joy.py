"""Agents for joy/curiosity experiments.

Tests positive emotions as approach-motivated parallel channels.
"""

import numpy as np
from typing import Dict, List
from gridworld_joy import JoyContext


class StandardQLearner:
    """Baseline Q-learning with epsilon-greedy exploration."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.2):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: JoyContext):
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += self.lr * (target - self.Q[state, action])


class CuriosityAgent:
    """Agent driven by curiosity (novelty-seeking).

    Curiosity modulates exploration: higher ε for novel states.
    Also provides intrinsic reward for novelty.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, base_epsilon: float = 0.1,
                 curiosity_weight: float = 0.5):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.base_epsilon = base_epsilon
        self.curiosity_weight = curiosity_weight

        # Track state visits for curiosity computation
        self.state_visits = np.zeros(n_states)
        self.current_curiosity = 0.0

    def _compute_curiosity(self, state: int, context: JoyContext) -> float:
        """Compute curiosity based on novelty."""
        return context.novelty * self.curiosity_weight

    def select_action(self, state: int) -> int:
        # Curiosity increases exploration
        effective_epsilon = self.base_epsilon + self.current_curiosity * 0.3
        effective_epsilon = min(effective_epsilon, 0.9)

        if np.random.random() < effective_epsilon:
            return np.random.randint(self.Q.shape[1])

        # Bias toward less-visited next states
        q_values = self.Q[state].copy()
        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: JoyContext):
        self.state_visits[state] += 1
        self.current_curiosity = self._compute_curiosity(state, context)

        # Intrinsic curiosity reward
        intrinsic_reward = context.novelty * self.curiosity_weight * 0.1

        # Q-learning with augmented reward
        total_reward = reward + intrinsic_reward
        target = total_reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += self.lr * (target - self.Q[state, action])

    def get_state(self) -> Dict:
        return {
            'current_curiosity': self.current_curiosity,
            'states_visited': np.sum(self.state_visits > 0)
        }


class JoyAgent:
    """Agent driven by joy (positive experience seeking).

    Joy builds from positive outcomes and biases toward
    states associated with past positive experiences.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 joy_weight: float = 0.5):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.joy_weight = joy_weight

        # Track joy associations with states
        self.state_joy = np.zeros(n_states)
        self.current_joy = 0.0
        self.joy_decay = 0.95

    def _update_joy(self, state: int, reward: float) -> float:
        """Update joy level based on positive outcomes."""
        if reward > 0:
            self.current_joy = min(1.0, self.current_joy + reward * self.joy_weight)
            self.state_joy[state] = min(1.0, self.state_joy[state] + reward * 0.3)
        else:
            self.current_joy *= self.joy_decay

        return self.current_joy

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Joy boosts exploitation of known-good actions
        if self.current_joy > 0.3:
            q_values[np.argmax(q_values)] *= (1 + self.current_joy * 0.5)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: JoyContext):
        self._update_joy(state, reward)

        # Enhanced learning for positive outcomes when joyful
        effective_lr = self.lr
        if self.current_joy > 0.3 and reward > 0:
            effective_lr *= (1 + self.current_joy * self.joy_weight)

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])

    def get_state(self) -> Dict:
        return {
            'current_joy': self.current_joy,
            'joyful_states': np.sum(self.state_joy > 0.3)
        }


class IntegratedJoyCuriosityAgent:
    """Agent with both joy and curiosity channels.

    Curiosity drives exploration to discover rewards.
    Joy drives return to and exploitation of positive states.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, base_epsilon: float = 0.1,
                 curiosity_weight: float = 0.4, joy_weight: float = 0.4):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.base_epsilon = base_epsilon
        self.curiosity_weight = curiosity_weight
        self.joy_weight = joy_weight

        # Curiosity state
        self.state_visits = np.zeros(n_states)
        self.current_curiosity = 0.0

        # Joy state
        self.state_joy = np.zeros(n_states)
        self.current_joy = 0.0
        self.joy_decay = 0.95

    def _compute_curiosity(self, context: JoyContext) -> float:
        return context.novelty * self.curiosity_weight

    def _update_joy(self, state: int, reward: float):
        if reward > 0:
            self.current_joy = min(1.0, self.current_joy + reward * self.joy_weight)
            self.state_joy[state] = min(1.0, self.state_joy[state] + reward * 0.3)
        else:
            self.current_joy *= self.joy_decay

    def select_action(self, state: int) -> int:
        # Curiosity increases exploration when novelty is high
        # Joy decreases exploration when positive experiences are recent
        effective_epsilon = self.base_epsilon
        effective_epsilon += self.current_curiosity * 0.3  # Curiosity → explore
        effective_epsilon -= self.current_joy * 0.2  # Joy → exploit
        effective_epsilon = max(0.05, min(0.9, effective_epsilon))

        if np.random.random() < effective_epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Joy boosts best action
        if self.current_joy > 0.3:
            q_values[np.argmax(q_values)] *= (1 + self.current_joy * 0.3)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: JoyContext):
        self.state_visits[state] += 1
        self.current_curiosity = self._compute_curiosity(context)
        self._update_joy(state, reward)

        # Intrinsic curiosity reward for novelty
        intrinsic = context.novelty * self.curiosity_weight * 0.1

        # Enhanced learning when emotional
        effective_lr = self.lr
        if self.current_curiosity > 0.3:
            effective_lr *= (1 + self.current_curiosity * 0.3)
        if self.current_joy > 0.3 and reward > 0:
            effective_lr *= (1 + self.current_joy * 0.3)

        total_reward = reward + intrinsic
        target = total_reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])

    def get_state(self) -> Dict:
        return {
            'current_curiosity': self.current_curiosity,
            'current_joy': self.current_joy,
            'states_visited': np.sum(self.state_visits > 0),
            'joyful_states': np.sum(self.state_joy > 0.3)
        }
