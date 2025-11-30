"""Two-door environment for regret/counterfactual testing (Coricelli paradigm)."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

@dataclass
class EmotionalContext:
    """Context for computing emotional signals."""
    threat_distance: float = float('inf')
    goal_distance: float = 0.0
    was_blocked: bool = False
    # Counterfactual information
    obtained_reward: float = 0.0
    foregone_reward: float = 0.0  # What the other choice would have given
    counterfactual_shown: bool = False

class TwoDoorEnv:
    """Two-door choice environment following Coricelli et al. (2005).

    Each trial:
    1. Agent sees two doors (A and B)
    2. Agent chooses one door
    3. Agent receives reward from chosen door
    4. Agent sees what was behind BOTH doors (counterfactual feedback)

    Key manipulation:
    - Standard RL: Only learns from obtained reward
    - Regret agent: Learns from comparison (obtained - foregone)
    """

    def __init__(self, n_trials: int = 100, reward_variance: float = 1.0,
                 correlation: float = 0.0):
        """
        Args:
            n_trials: Number of trials per episode
            reward_variance: Variance of reward distributions
            correlation: Correlation between door rewards (-1 to 1)
                        0 = independent, negative = anti-correlated
        """
        self.n_trials = n_trials
        self.reward_variance = reward_variance
        self.correlation = correlation
        self.reset()

    def reset(self) -> int:
        """Reset environment, return initial state (always 0 = choice state)."""
        self.trial = 0
        self.total_reward = 0.0
        self._generate_rewards()
        return 0  # Single state: choice point

    def _generate_rewards(self):
        """Pre-generate rewards for all trials."""
        # Generate correlated rewards
        mean = [0.5, 0.5]
        cov = [[self.reward_variance, self.correlation * self.reward_variance],
               [self.correlation * self.reward_variance, self.reward_variance]]

        rewards = np.random.multivariate_normal(mean, cov, self.n_trials)
        self.rewards_A = rewards[:, 0]
        self.rewards_B = rewards[:, 1]

    def step(self, action: int) -> Tuple[int, float, bool, EmotionalContext]:
        """
        Take action (0 = door A, 1 = door B).
        Returns (state, reward, done, context with counterfactual).
        """
        if action == 0:
            obtained = self.rewards_A[self.trial]
            foregone = self.rewards_B[self.trial]
        else:
            obtained = self.rewards_B[self.trial]
            foregone = self.rewards_A[self.trial]

        self.total_reward += obtained
        self.trial += 1
        done = self.trial >= self.n_trials

        context = EmotionalContext(
            obtained_reward=obtained,
            foregone_reward=foregone,
            counterfactual_shown=True
        )

        return 0, obtained, done, context  # Always return to state 0

    @property
    def n_states(self) -> int:
        return 1  # Single choice state

    @property
    def n_actions(self) -> int:
        return 2  # Door A or Door B


class PartialFeedbackTwoDoorEnv(TwoDoorEnv):
    """Two-door environment where counterfactual is only sometimes shown.

    This allows testing whether agents USE counterfactual information
    when available vs. when not available.
    """

    def __init__(self, n_trials: int = 100, feedback_prob: float = 0.5, **kwargs):
        """
        Args:
            feedback_prob: Probability of showing counterfactual feedback
        """
        self.feedback_prob = feedback_prob
        super().__init__(n_trials=n_trials, **kwargs)

    def step(self, action: int) -> Tuple[int, float, bool, EmotionalContext]:
        """Take action, maybe show counterfactual."""
        if action == 0:
            obtained = self.rewards_A[self.trial]
            foregone = self.rewards_B[self.trial]
        else:
            obtained = self.rewards_B[self.trial]
            foregone = self.rewards_A[self.trial]

        self.total_reward += obtained
        self.trial += 1
        done = self.trial >= self.n_trials

        # Only sometimes show counterfactual
        show_counterfactual = np.random.random() < self.feedback_prob

        context = EmotionalContext(
            obtained_reward=obtained,
            foregone_reward=foregone if show_counterfactual else 0.0,
            counterfactual_shown=show_counterfactual
        )

        return 0, obtained, done, context
