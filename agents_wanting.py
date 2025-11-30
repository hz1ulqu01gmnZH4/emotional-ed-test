"""Agents for wanting/liking dissociation testing.

Models Berridge's (2009) distinction:
- Wanting: Incentive salience (dopamine)
- Liking: Hedonic impact (opioid)
"""

import numpy as np
from typing import Dict, List
from gridworld_wanting import WantingLikingContext


class StandardQLearner:
    """Baseline Q-learning - no wanting/liking distinction."""

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
               next_state: int, done: bool, context: WantingLikingContext):
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += self.lr * (target - self.Q[state, action])

    def reset_episode(self):
        pass

    def get_motivation_state(self) -> Dict:
        return {'wanting': 0.0, 'liking': 0.0}


class WantingDominantAgent:
    """Agent where wanting (incentive salience) dominates behavior.

    Models addiction-like behavior:
    - Strong pull toward high-salience rewards
    - Wanting doesn't track actual hedonic value
    - Pursues even when satiated
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 wanting_weight: float = 1.5):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.wanting_weight = wanting_weight

        # Wanting state
        self.current_wanting = 0.0

        # Track salience of different rewards
        self.reward_salience = {
            'wanting': 1.5,  # High salience
            'liking': 0.7,   # Lower salience
            'regular': 1.0
        }

    def _compute_wanting(self, context: WantingLikingContext) -> float:
        """Compute wanting based on proximity to high-salience rewards."""
        # Wanting is proximity-weighted by salience
        wanting_pull = 0.0

        if context.high_wanting_distance < 4.0:
            pull = (1 - context.high_wanting_distance / 4.0) * self.reward_salience['wanting']
            wanting_pull = max(wanting_pull, pull)

        if context.high_liking_distance < 4.0:
            pull = (1 - context.high_liking_distance / 4.0) * self.reward_salience['liking']
            wanting_pull = max(wanting_pull, pull)

        if context.regular_distance < 4.0:
            pull = (1 - context.regular_distance / 4.0) * self.reward_salience['regular']
            wanting_pull = max(wanting_pull, pull)

        # Key insight: Wanting doesn't decrease much with satiation
        # (Unlike liking, which does)
        satiation_reduction = context.satiation_level * 0.2  # Minimal effect
        return wanting_pull * (1 - satiation_reduction)

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Wanting biases toward high-salience rewards
        if self.current_wanting > 0.3:
            # Boost exploitation when wanting is high
            q_values[np.argmax(q_values)] *= (1 + self.current_wanting * self.wanting_weight)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: WantingLikingContext):
        self.current_wanting = self._compute_wanting(context)

        # Wanting modulates learning - but tracks salience, not reward
        effective_lr = self.lr

        # High wanting → faster learning about high-salience rewards
        if context.just_consumed == 'wanting':
            effective_lr *= (1 + self.wanting_weight * 0.5)

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])

    def reset_episode(self):
        self.current_wanting = 0.0

    def get_motivation_state(self) -> Dict:
        return {'wanting': self.current_wanting, 'liking': 0.0}


class LikingDominantAgent:
    """Agent where liking (hedonic impact) dominates behavior.

    Models pleasure-seeking:
    - Pursues highest hedonic value
    - Liking decreases with satiation
    - Less motivated by salience
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 liking_weight: float = 1.5):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.liking_weight = liking_weight

        # Liking state
        self.current_liking = 0.0
        self.last_reward = 0.0

        # Track hedonic value of rewards
        self.reward_hedonic = {
            'wanting': 0.5,  # Modest pleasure
            'liking': 1.0,   # High pleasure
            'regular': 0.6
        }

    def _compute_liking(self, context: WantingLikingContext) -> float:
        """Compute liking - hedonic anticipation."""
        # Liking tracks expected hedonic value
        liking = 0.0

        if context.high_liking_distance < 4.0:
            liking = max(liking, (1 - context.high_liking_distance / 4.0) * self.reward_hedonic['liking'])

        if context.high_wanting_distance < 4.0:
            liking = max(liking, (1 - context.high_wanting_distance / 4.0) * self.reward_hedonic['wanting'])

        # Key insight: Liking decreases with satiation
        satiation_reduction = context.satiation_level * 0.6  # Strong effect
        return liking * (1 - satiation_reduction)

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Liking biases toward hedonic rewards
        if self.current_liking > 0.3:
            q_values[np.argmax(q_values)] *= (1 + self.current_liking * self.liking_weight)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: WantingLikingContext):
        self.current_liking = self._compute_liking(context)
        self.last_reward = reward

        # Liking modulates learning based on hedonic experience
        effective_lr = self.lr

        # High reward → positive reinforcement boost
        if reward > 0.5:
            effective_lr *= (1 + reward * self.liking_weight * 0.5)

        # Negative hedonic surprise → faster learning to avoid
        if reward < 0 and self.current_liking > 0.3:
            effective_lr *= 1.5  # Disappointment drives learning

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])

    def reset_episode(self):
        self.current_liking = 0.0
        self.last_reward = 0.0

    def get_motivation_state(self) -> Dict:
        return {'wanting': 0.0, 'liking': self.current_liking}


class IntegratedWantingLikingAgent:
    """Agent with both wanting and liking systems.

    Models healthy motivation:
    - Wanting provides drive
    - Liking provides feedback
    - Both influence but can dissociate
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 wanting_weight: float = 1.0, liking_weight: float = 1.0):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.wanting_weight = wanting_weight
        self.liking_weight = liking_weight

        self.current_wanting = 0.0
        self.current_liking = 0.0

        # Experience-based salience/hedonic learning
        self.learned_salience = {'wanting': 1.5, 'liking': 0.7, 'regular': 1.0}
        self.learned_hedonic = {'wanting': 0.5, 'liking': 1.0, 'regular': 0.6}

    def _compute_wanting(self, context: WantingLikingContext) -> float:
        wanting = 0.0
        if context.high_wanting_distance < 4.0:
            wanting = max(wanting, (1 - context.high_wanting_distance / 4.0) * self.learned_salience['wanting'])
        if context.high_liking_distance < 4.0:
            wanting = max(wanting, (1 - context.high_liking_distance / 4.0) * self.learned_salience['liking'])

        # Wanting less affected by satiation
        return wanting * (1 - context.satiation_level * 0.2)

    def _compute_liking(self, context: WantingLikingContext) -> float:
        liking = 0.0
        if context.high_liking_distance < 4.0:
            liking = max(liking, (1 - context.high_liking_distance / 4.0) * self.learned_hedonic['liking'])
        if context.high_wanting_distance < 4.0:
            liking = max(liking, (1 - context.high_wanting_distance / 4.0) * self.learned_hedonic['wanting'])

        # Liking strongly affected by satiation
        return liking * (1 - context.satiation_level * 0.6)

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Combined wanting + liking influence
        motivation = self.current_wanting * self.wanting_weight + self.current_liking * self.liking_weight
        if motivation > 0.5:
            q_values[np.argmax(q_values)] *= (1 + motivation * 0.5)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: WantingLikingContext):
        self.current_wanting = self._compute_wanting(context)
        self.current_liking = self._compute_liking(context)

        # Update salience/hedonic estimates from experience
        if context.just_consumed:
            consumed = context.just_consumed
            # If reward was good, increase salience slightly
            if reward > 0.3:
                self.learned_salience[consumed] = min(2.0, self.learned_salience[consumed] * 1.05)
            # Hedonic estimate tracks actual reward
            self.learned_hedonic[consumed] = 0.9 * self.learned_hedonic[consumed] + 0.1 * reward

        effective_lr = self.lr

        # Wanting-liking mismatch detection
        # If high wanting but low reward (addiction-like)
        if self.current_wanting > 0.5 and reward < 0.3:
            # Potential wanting-liking dissociation
            effective_lr *= 1.2  # Learn from the mismatch

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])

    def reset_episode(self):
        self.current_wanting = 0.0
        self.current_liking = 0.0

    def get_motivation_state(self) -> Dict:
        return {
            'wanting': self.current_wanting,
            'liking': self.current_liking,
            'salience': self.learned_salience.copy(),
            'hedonic': self.learned_hedonic.copy()
        }


class AddictionModelAgent(WantingDominantAgent):
    """Agent modeling addiction: High wanting, low liking.

    Key addiction features (Robinson & Berridge, 1993):
    - Wanting escalates with exposure
    - Liking diminishes (tolerance)
    - Pursues despite negative consequences
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 sensitization_rate: float = 0.05,
                 tolerance_rate: float = 0.1):
        super().__init__(n_states, n_actions, lr, gamma, epsilon)

        self.sensitization_rate = sensitization_rate
        self.tolerance_rate = tolerance_rate

        # Addiction-specific tracking
        self.wanting_baseline = 1.0  # Increases with exposure (sensitization)
        self.liking_baseline = 1.0   # Decreases with exposure (tolerance)
        self.high_wanting_exposures = 0

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: WantingLikingContext):
        # Track exposure to high-wanting reward
        if context.just_consumed == 'wanting':
            self.high_wanting_exposures += 1

            # Wanting sensitizes (increases)
            self.wanting_baseline *= (1 + self.sensitization_rate)
            self.reward_salience['wanting'] = 1.5 * self.wanting_baseline

            # Liking tolerates (decreases)
            self.liking_baseline *= (1 - self.tolerance_rate)

        super().update(state, action, reward, next_state, done, context)

    def get_motivation_state(self) -> Dict:
        base = super().get_motivation_state()
        base['wanting_baseline'] = self.wanting_baseline
        base['liking_baseline'] = self.liking_baseline
        base['exposures'] = self.high_wanting_exposures
        return base
