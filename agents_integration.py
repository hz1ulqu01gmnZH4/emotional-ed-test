"""Integrated multi-channel emotional agents.

Tests how fear, anger, and regret interact when active simultaneously.
"""

import numpy as np
from typing import Dict, Optional
from gridworld_integration import IntegratedEmotionalContext


class StandardQLearner:
    """Baseline Q-learning without emotional channels."""

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
               next_state: int, done: bool, context: IntegratedEmotionalContext):
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += self.lr * (target - self.Q[state, action])

    def reset_episode(self):
        pass

    def get_channel_states(self) -> Dict:
        return {}


class IntegratedEmotionalAgent:
    """Agent with fear, anger, and regret channels active simultaneously.

    Channel interactions:
    - Fear: Avoids threat, reduces approach to risky goal
    - Anger: Persistence at blocked path, may overcome fear
    - Regret: Learns from counterfactual outcomes, biases future choices

    Key question: How do channels compete and cooperate?
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 0.6,
                 anger_weight: float = 0.4,
                 regret_weight: float = 0.3):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # Channel weights
        self.fear_weight = fear_weight
        self.anger_weight = anger_weight
        self.regret_weight = regret_weight

        # Channel states
        self.current_fear = 0.0
        self.current_anger = 0.0
        self.current_regret = 0.0

        # Regret memory
        self.regret_memory = {}  # state -> cumulative regret

        # Anger buildup
        self.frustration = 0.0

    def _compute_fear(self, context: IntegratedEmotionalContext) -> float:
        """Fear from threat proximity."""
        if context.threat_distance >= 3.0:
            return 0.0
        return 1.0 - context.threat_distance / 3.0

    def _compute_anger(self, context: IntegratedEmotionalContext) -> float:
        """Anger from blocked path."""
        if not context.was_blocked:
            self.frustration = max(0, self.frustration - 0.1)
            return self.frustration

        # Anger builds with consecutive blocks
        self.frustration = min(1.0, self.frustration + 0.2 * context.consecutive_blocks)
        return self.frustration

    def _compute_regret(self, context: IntegratedEmotionalContext) -> float:
        """Regret from counterfactual comparison."""
        if not context.counterfactual_shown:
            return 0.0

        # Negative regret = missed better outcome
        regret = context.obtained_reward - context.foregone_reward
        return regret  # Can be negative (missed better) or positive (chose better)

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Fear biases away from threatening actions
        if self.current_fear > 0:
            q_min = q_values.min()
            q_max = q_values.max()
            if q_max > q_min:
                # Reduce value of actions toward threat
                normalized = (q_values - q_min) / (q_max - q_min)
                q_values -= self.current_fear * self.fear_weight * normalized

        # Anger biases toward persistence (exploit over explore)
        if self.current_anger > 0.3:
            # When angry, double down on current best action
            best_action = np.argmax(q_values)
            q_values[best_action] *= (1 + self.current_anger * self.anger_weight)

        # Regret memory affects exploration
        if state in self.regret_memory:
            cumulative_regret = self.regret_memory[state]
            if cumulative_regret < -0.5:  # Accumulated bad choices here
                # Increase exploration to find better options
                if np.random.random() < abs(cumulative_regret) * 0.3:
                    return np.random.randint(self.Q.shape[1])

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: IntegratedEmotionalContext):
        # Compute emotional states
        self.current_fear = self._compute_fear(context)
        self.current_anger = self._compute_anger(context)
        self.current_regret = self._compute_regret(context)

        # Update regret memory
        if context.counterfactual_shown:
            if state not in self.regret_memory:
                self.regret_memory[state] = 0.0
            self.regret_memory[state] += self.current_regret

        # Modulated learning rate
        effective_lr = self.lr

        # Fear amplifies negative learning
        if self.current_fear > 0 and reward < 0:
            effective_lr *= (1 + self.current_fear * self.fear_weight)

        # Anger reduces learning from failure (stubbornness)
        if self.current_anger > 0.3 and reward < 0:
            effective_lr *= (1 - self.current_anger * 0.5)

        # Regret modulates learning
        if self.current_regret < 0:  # Missed better outcome
            effective_lr *= (1 + abs(self.current_regret) * self.regret_weight)

        # Q-learning update
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])

    def reset_episode(self):
        self.current_fear = 0.0
        self.current_anger = 0.0
        self.current_regret = 0.0
        self.frustration = 0.0
        # Regret memory persists across episodes

    def get_channel_states(self) -> Dict:
        return {
            'fear': self.current_fear,
            'anger': self.current_anger,
            'regret': self.current_regret,
            'frustration': self.frustration,
            'regret_states': len(self.regret_memory)
        }


class FearDominantAgent(IntegratedEmotionalAgent):
    """Agent where fear channel dominates."""

    def __init__(self, n_states: int, n_actions: int, **kwargs):
        super().__init__(n_states, n_actions,
                         fear_weight=1.0,
                         anger_weight=0.2,
                         regret_weight=0.2,
                         **kwargs)


class AngerDominantAgent(IntegratedEmotionalAgent):
    """Agent where anger channel dominates."""

    def __init__(self, n_states: int, n_actions: int, **kwargs):
        super().__init__(n_states, n_actions,
                         fear_weight=0.2,
                         anger_weight=1.0,
                         regret_weight=0.2,
                         **kwargs)


class RegretDominantAgent(IntegratedEmotionalAgent):
    """Agent where regret channel dominates."""

    def __init__(self, n_states: int, n_actions: int, **kwargs):
        super().__init__(n_states, n_actions,
                         fear_weight=0.2,
                         anger_weight=0.2,
                         regret_weight=1.0,
                         **kwargs)


class BalancedAgent(IntegratedEmotionalAgent):
    """Agent with balanced emotional channels."""

    def __init__(self, n_states: int, n_actions: int, **kwargs):
        super().__init__(n_states, n_actions,
                         fear_weight=0.5,
                         anger_weight=0.5,
                         regret_weight=0.5,
                         **kwargs)


class AdaptiveEmotionalAgent(IntegratedEmotionalAgent):
    """Agent that learns to weight emotional channels.

    Meta-learning: adjusts channel weights based on outcomes.
    """

    def __init__(self, n_states: int, n_actions: int, meta_lr: float = 0.05, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.meta_lr = meta_lr
        self.channel_success = {'fear': 0.0, 'anger': 0.0, 'regret': 0.0}
        self.last_channel_active = None
        self.episode_reward = 0.0

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: IntegratedEmotionalContext):
        # Track which channel was most active
        fear = self._compute_fear(context)
        anger = self._compute_anger(context)
        regret = abs(self._compute_regret(context))

        max_channel = max(fear, anger, regret)
        if max_channel > 0.3:
            if fear == max_channel:
                self.last_channel_active = 'fear'
            elif anger == max_channel:
                self.last_channel_active = 'anger'
            else:
                self.last_channel_active = 'regret'

        self.episode_reward += reward

        # At episode end, update channel weights
        if done and self.last_channel_active:
            # Reward the active channel if outcome was good
            if self.episode_reward > 0:
                self.channel_success[self.last_channel_active] += self.episode_reward
            else:
                self.channel_success[self.last_channel_active] -= 0.1

            # Adjust weights based on success
            total_success = sum(max(0, v) for v in self.channel_success.values()) + 0.1
            self.fear_weight = 0.3 + 0.4 * max(0, self.channel_success['fear']) / total_success
            self.anger_weight = 0.3 + 0.4 * max(0, self.channel_success['anger']) / total_success
            self.regret_weight = 0.3 + 0.4 * max(0, self.channel_success['regret']) / total_success

        super().update(state, action, reward, next_state, done, context)

    def reset_episode(self):
        super().reset_episode()
        self.episode_reward = 0.0
        self.last_channel_active = None

    def get_channel_states(self) -> Dict:
        states = super().get_channel_states()
        states['channel_weights'] = {
            'fear': self.fear_weight,
            'anger': self.anger_weight,
            'regret': self.regret_weight
        }
        states['channel_success'] = self.channel_success.copy()
        return states
