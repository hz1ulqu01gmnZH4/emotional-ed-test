"""Agents for temporal emotion dynamics testing.

Tests phasic (acute) vs tonic (mood) emotional responses.
"""

import numpy as np
from typing import Dict, List
from gridworld_temporal import TemporalEmotionalContext


class StandardQLearner:
    """Baseline Q-learning without temporal emotional dynamics."""

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
               next_state: int, done: bool, context: TemporalEmotionalContext):
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += self.lr * (target - self.Q[state, action])

    def reset_episode(self):
        # Intentionally empty - standard Q-learner has no emotional state to reset
        pass

    def get_emotional_state(self) -> Dict:
        return {'mood': 0.0, 'phasic': 0.0}


class PhasicOnlyAgent:
    """Agent with only phasic (acute) emotional responses.

    Emotions are immediate reactions that decay quickly.
    No lasting mood effects from sustained experience.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 phasic_decay: float = 0.7):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.phasic_decay = phasic_decay

        # Phasic emotion (immediate, fast-decaying)
        self.phasic_fear = 0.0
        self.phasic_joy = 0.0

    def _compute_phasic_emotion(self, context: TemporalEmotionalContext):
        """Compute immediate emotional response."""
        # Fear from immediate threat
        if context.threat_distance < 3.0:
            self.phasic_fear = 1.0 - context.threat_distance / 3.0
        else:
            self.phasic_fear *= self.phasic_decay

        # Joy from immediate reward
        if context.reward_obtained > 0.1:
            self.phasic_joy = min(1.0, context.reward_obtained)
        else:
            self.phasic_joy *= self.phasic_decay

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Phasic fear reduces exploration
        if self.phasic_fear > 0.3:
            # Stick with known-good actions
            q_values[np.argmax(q_values)] *= (1 + self.phasic_fear)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: TemporalEmotionalContext):
        self._compute_phasic_emotion(context)

        # Phasic emotion modulates learning rate
        effective_lr = self.lr
        if self.phasic_fear > 0.3 and reward < 0:
            effective_lr *= (1 + self.phasic_fear)
        if self.phasic_joy > 0.3 and reward > 0:
            effective_lr *= (1 + self.phasic_joy * 0.5)

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])

    def reset_episode(self):
        # Phasic emotions reset between episodes
        self.phasic_fear = 0.0
        self.phasic_joy = 0.0

    def get_emotional_state(self) -> Dict:
        return {
            'mood': 0.0,  # No tonic component
            'phasic_fear': self.phasic_fear,
            'phasic_joy': self.phasic_joy
        }


class TonicMoodAgent:
    """Agent with tonic (mood) emotional responses.

    Sustained experience shifts emotional baseline.
    Mood affects all subsequent perception and behavior.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 mood_inertia: float = 0.95, mood_sensitivity: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.mood_inertia = mood_inertia
        self.mood_sensitivity = mood_sensitivity

        # Tonic mood (slow-changing baseline)
        # Range: -1 (depressed) to +1 (elated)
        self.mood = 0.0

        # Track mood history
        self.mood_history = []

    def _update_mood(self, context: TemporalEmotionalContext):
        """Update tonic mood based on cumulative experience."""
        # Mood shifts based on recent cumulative experience
        # cumulative_reward is sum of last 20 rewards (float), divide by window size
        # Use episode_length capped at 20 to handle early-episode correctly
        window_size = max(1, min(context.episode_length, 20))
        recent_valence = context.cumulative_reward / window_size

        # Sustained negative shifts mood down
        if context.consecutive_negative > 5:
            mood_delta = -self.mood_sensitivity * (context.consecutive_negative / 10)
        elif context.consecutive_positive > 3:
            mood_delta = self.mood_sensitivity * (context.consecutive_positive / 10)
        else:
            # Gradual return to neutral
            mood_delta = -self.mood * 0.01

        # Apply with inertia
        self.mood = self.mood * self.mood_inertia + mood_delta
        self.mood = np.clip(self.mood, -1.0, 1.0)

        self.mood_history.append(self.mood)

    def select_action(self, state: int) -> int:
        # Mood affects exploration
        effective_epsilon = self.epsilon

        # Negative mood → more conservative (less exploration)
        if self.mood < -0.3:
            effective_epsilon *= (1 - abs(self.mood) * 0.5)

        # Positive mood → more exploration
        if self.mood > 0.3:
            effective_epsilon *= (1 + self.mood * 0.3)

        if np.random.random() < effective_epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Negative mood biases toward safe/known actions
        if self.mood < -0.2:
            q_values[np.argmax(q_values)] *= (1 + abs(self.mood) * 0.5)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: TemporalEmotionalContext):
        self._update_mood(context)

        # Mood biases reward perception
        perceived_reward = reward

        # Negative mood amplifies negative outcomes
        if self.mood < -0.2 and reward < 0:
            perceived_reward *= (1 + abs(self.mood) * 0.5)

        # Positive mood amplifies positive outcomes
        if self.mood > 0.2 and reward > 0:
            perceived_reward *= (1 + self.mood * 0.3)

        target = perceived_reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += self.lr * (target - self.Q[state, action])

    def reset_episode(self):
        # Intentionally empty - mood persists across episodes (key difference from phasic)
        # This is deliberate: tonic mood represents slow-changing baseline that doesn't reset
        pass

    def get_emotional_state(self) -> Dict:
        return {
            'mood': self.mood,
            'phasic': 0.0
        }


class IntegratedTemporalAgent:
    """Agent with both phasic and tonic emotional components.

    Phasic: Immediate reactions to events
    Tonic: Slow-shifting baseline mood

    Models biological emotion systems more completely.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 phasic_decay: float = 0.7,
                 mood_inertia: float = 0.95,
                 mood_sensitivity: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # Phasic components
        self.phasic_decay = phasic_decay
        self.phasic_fear = 0.0
        self.phasic_joy = 0.0

        # Tonic component
        self.mood_inertia = mood_inertia
        self.mood_sensitivity = mood_sensitivity
        self.mood = 0.0

        # Track history
        self.mood_history = []
        self.phasic_history = []

    def _compute_phasic(self, context: TemporalEmotionalContext):
        """Compute immediate emotional response."""
        if context.threat_distance < 3.0:
            self.phasic_fear = 1.0 - context.threat_distance / 3.0
        else:
            self.phasic_fear *= self.phasic_decay

        if context.reward_obtained > 0.1:
            self.phasic_joy = min(1.0, context.reward_obtained)
        else:
            self.phasic_joy *= self.phasic_decay

    def _update_mood(self, context: TemporalEmotionalContext):
        """Update tonic mood from sustained experience."""
        if context.consecutive_negative > 5:
            mood_delta = -self.mood_sensitivity * (context.consecutive_negative / 10)
        elif context.consecutive_positive > 3:
            mood_delta = self.mood_sensitivity * (context.consecutive_positive / 10)
        else:
            mood_delta = -self.mood * 0.01

        self.mood = self.mood * self.mood_inertia + mood_delta
        self.mood = np.clip(self.mood, -1.0, 1.0)

        self.mood_history.append(self.mood)
        self.phasic_history.append(self.phasic_fear - self.phasic_joy)

    def select_action(self, state: int) -> int:
        # Combined effect of mood and phasic
        effective_epsilon = self.epsilon

        # Mood affects baseline exploration
        if self.mood < -0.3:
            effective_epsilon *= 0.7
        elif self.mood > 0.3:
            effective_epsilon *= 1.3

        # Phasic fear further reduces exploration
        if self.phasic_fear > 0.5:
            effective_epsilon *= 0.5

        if np.random.random() < effective_epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Mood biases action selection
        if self.mood < -0.2:
            q_values[np.argmax(q_values)] *= (1 + abs(self.mood) * 0.3)

        # Phasic fear adds additional bias
        if self.phasic_fear > 0.3:
            q_values[np.argmax(q_values)] *= (1 + self.phasic_fear * 0.5)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: TemporalEmotionalContext):
        self._compute_phasic(context)
        self._update_mood(context)

        # Combined emotional modulation
        effective_lr = self.lr

        # Phasic modulation (immediate)
        if self.phasic_fear > 0.3 and reward < 0:
            effective_lr *= (1 + self.phasic_fear * 0.5)
        if self.phasic_joy > 0.3 and reward > 0:
            effective_lr *= (1 + self.phasic_joy * 0.3)

        # Mood modulation (baseline shift)
        perceived_reward = reward
        if self.mood < -0.2 and reward < 0:
            perceived_reward *= (1 + abs(self.mood) * 0.3)
        if self.mood > 0.2 and reward > 0:
            perceived_reward *= (1 + self.mood * 0.2)

        target = perceived_reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])

    def reset_episode(self):
        # Phasic resets, mood persists
        self.phasic_fear = 0.0
        self.phasic_joy = 0.0

    def get_emotional_state(self) -> Dict:
        return {
            'mood': self.mood,
            'phasic_fear': self.phasic_fear,
            'phasic_joy': self.phasic_joy,
            'combined': self.mood + (self.phasic_joy - self.phasic_fear) * 0.5
        }

    def get_mood_history(self) -> List[float]:
        return self.mood_history.copy()
