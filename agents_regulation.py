"""Agents for emotion regulation testing."""

import numpy as np
from typing import List, Dict, Optional
from gridworld_regulation import EmotionalContext

class StandardQLearner:
    """Standard Q-learning baseline."""

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
               next_state: int, done: bool, context: EmotionalContext):
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        pass


class UnregulatedFearAgent:
    """Agent with fear response that cannot be regulated.

    Treats all threat-like stimuli the same - no discrimination
    between real and fake threats.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 0.8):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.fear_weight = fear_weight
        self.current_fear = 0.0

    def _compute_fear(self, context: EmotionalContext) -> float:
        """Fear based on threat proximity only - no discrimination."""
        if context.threat_distance >= 3.0:
            return 0.0
        return 1.0 - context.threat_distance / 3.0

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Fear biases toward safe actions
        if self.current_fear > 0:
            q_min, q_max = q_values.min(), q_values.max()
            if q_max > q_min:
                normalized = (q_values - q_min) / (q_max - q_min)
                q_values += self.current_fear * self.fear_weight * normalized

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        fear = self._compute_fear(context)
        self.current_fear = fear

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        # Fear amplifies negative learning
        if fear > 0 and delta < 0:
            delta *= (1 + fear * self.fear_weight)

        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        self.current_fear = 0.0


class RegulatedFearAgent:
    """Agent with emotion regulation capability.

    Implements cognitive reappraisal (Ochsner & Gross, 2005):
    - Initial fear response to all threats
    - Learns to discriminate real vs fake threats
    - Reduces fear response to fake threats over time

    Key mechanism: V(f(s)) instead of V(s)
    - f(s) = reappraised state representation
    - When threat is known to be fake, f(s) reduces threat salience
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 0.8, regulation_lr: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.fear_weight = fear_weight
        self.regulation_lr = regulation_lr

        # Learned threat discrimination
        # Maps state → learned "fakeness" (0 = real threat, 1 = definitely fake)
        self.threat_beliefs = np.zeros(n_states)  # Initially all threats seem real

        self.current_fear = 0.0
        self.current_regulation = 0.0

    def _compute_raw_fear(self, context: EmotionalContext) -> float:
        """Raw fear from threat proximity."""
        if context.threat_distance >= 3.0:
            return 0.0
        return 1.0 - context.threat_distance / 3.0

    def _compute_regulated_fear(self, state: int, raw_fear: float) -> float:
        """Fear after regulation (reappraisal)."""
        # Regulation reduces fear based on learned fakeness
        fakeness = self.threat_beliefs[state]
        regulated_fear = raw_fear * (1 - fakeness)
        return regulated_fear

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Use REGULATED fear for action selection
        if self.current_fear > 0:
            regulated_fear = self._compute_regulated_fear(state, self.current_fear)
            self.current_regulation = self.current_fear - regulated_fear

            q_min, q_max = q_values.min(), q_values.max()
            if q_max > q_min:
                normalized = (q_values - q_min) / (q_max - q_min)
                # Use regulated (reduced) fear
                q_values += regulated_fear * self.fear_weight * normalized

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        raw_fear = self._compute_raw_fear(context)
        self.current_fear = raw_fear

        # Learn threat discrimination
        # If we got positive outcome near threat → update belief toward "fake"
        if raw_fear > 0 and context.threat_type == 'fake':
            if reward > 0:  # Positive outcome near "threat"
                # Update belief: this threat is probably fake
                self.threat_beliefs[state] += self.regulation_lr * (1 - self.threat_beliefs[state])
        elif raw_fear > 0 and context.threat_type == 'real':
            if reward < -0.1:  # Negative outcome
                # Confirm threat is real
                self.threat_beliefs[state] *= (1 - self.regulation_lr)

        # Use regulated fear for learning
        regulated_fear = self._compute_regulated_fear(state, raw_fear)

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        # Regulated fear affects learning
        if regulated_fear > 0 and delta < 0:
            delta *= (1 + regulated_fear * self.fear_weight)

        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        self.current_fear = 0.0
        self.current_regulation = 0.0
        # Note: threat_beliefs persist across episodes (learned regulation)

    def get_regulation_stats(self) -> Dict:
        """Return statistics about learned regulation."""
        return {
            'mean_fakeness_belief': np.mean(self.threat_beliefs),
            'max_fakeness_belief': np.max(self.threat_beliefs),
            'states_regulated': np.sum(self.threat_beliefs > 0.3)
        }


class ExplicitReappraisalAgent(RegulatedFearAgent):
    """Agent that explicitly reframes threatening situations.

    More sophisticated regulation:
    - Learns multiple reappraisal strategies
    - Selects strategy based on context
    - Can generalize regulation across similar situations
    """

    def __init__(self, n_states: int, n_actions: int,
                 n_strategies: int = 3, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)

        # Multiple reappraisal strategies
        # Strategy 0: "It's not real" (distancing)
        # Strategy 1: "I can handle it" (self-efficacy)
        # Strategy 2: "This is an opportunity" (positive reframe)
        self.n_strategies = n_strategies
        self.strategy_effectiveness = np.ones((n_states, n_strategies)) * 0.5

        self.current_strategy = 0

    def _select_reappraisal_strategy(self, state: int) -> int:
        """Select best reappraisal strategy for this state."""
        return np.argmax(self.strategy_effectiveness[state])

    def _compute_regulated_fear(self, state: int, raw_fear: float) -> float:
        """Fear after applying best reappraisal strategy."""
        strategy = self._select_reappraisal_strategy(state)
        self.current_strategy = strategy

        # Effectiveness of chosen strategy
        effectiveness = self.strategy_effectiveness[state, strategy]

        # Regulated fear
        regulated_fear = raw_fear * (1 - effectiveness * 0.8)
        return max(0, regulated_fear)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        # Update strategy effectiveness based on outcome
        raw_fear = self._compute_raw_fear(context)

        if raw_fear > 0.3:  # Was in threatening situation
            strategy = self.current_strategy
            if reward > 0:  # Good outcome
                # Strategy worked - increase effectiveness
                self.strategy_effectiveness[state, strategy] += 0.1 * (1 - self.strategy_effectiveness[state, strategy])
            elif reward < -0.2:  # Bad outcome
                # Strategy failed - decrease effectiveness
                self.strategy_effectiveness[state, strategy] *= 0.9

        # Regular update
        super().update(state, action, reward, next_state, done, context)
