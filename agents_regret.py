"""Agents for regret/counterfactual testing."""

import numpy as np
from typing import List, Optional
from gridworld_regret import EmotionalContext

class StandardBanditAgent:
    """Standard bandit agent - learns only from obtained rewards."""

    def __init__(self, n_actions: int = 2, lr: float = 0.1, epsilon: float = 0.1):
        self.Q = np.zeros(n_actions)  # Action values
        self.lr = lr
        self.epsilon = epsilon
        self.n_actions = n_actions

        # Tracking
        self.choices: List[int] = []
        self.rewards: List[float] = []

    def select_action(self, state: int = 0) -> int:
        """Epsilon-greedy selection."""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q)
        self.choices.append(action)
        return action

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        """Standard update - ignores counterfactual information."""
        self.rewards.append(reward)

        # Only learn from obtained reward
        delta = reward - self.Q[action]
        self.Q[action] += self.lr * delta

    def reset_episode(self):
        self.choices = []
        self.rewards = []


class RegretModule:
    """Computes regret signal from counterfactual comparison.

    Regret = V(obtained) - V(best foregone)
    When regret < 0: We did worse than we could have
    When regret > 0: Relief - we did better than alternative

    Following Coricelli et al. (2005) and Camille et al. (2004).
    """

    def __init__(self, regret_sensitivity: float = 1.0):
        self.regret_sensitivity = regret_sensitivity
        self.regret_history: List[float] = []

    def compute(self, context: EmotionalContext) -> float:
        """Compute regret signal."""
        if not context.counterfactual_shown:
            return 0.0

        # Regret = obtained - foregone
        # Negative = regret (we did worse)
        # Positive = relief (we did better)
        regret = context.obtained_reward - context.foregone_reward
        regret = regret * self.regret_sensitivity

        self.regret_history.append(regret)
        return regret

    def reset(self):
        self.regret_history = []


class RegretEDAgent:
    """Bandit agent with regret channel.

    Key insight from Coricelli et al.:
    - Humans learn from BOTH obtained and foregone outcomes
    - Regret (choosing worse option) enhances learning
    - OFC lesion patients don't show regret-based learning

    Implementation:
    - Update chosen action based on obtained reward (standard)
    - ALSO update unchosen action based on foregone reward
    - Regret signal modulates learning rate
    """

    def __init__(self, n_actions: int = 2, lr: float = 0.1, epsilon: float = 0.1,
                 regret_weight: float = 0.5, counterfactual_lr: float = 0.05):
        self.Q = np.zeros(n_actions)
        self.lr = lr
        self.epsilon = epsilon
        self.n_actions = n_actions

        # Regret module
        self.regret_module = RegretModule()
        self.regret_weight = regret_weight
        self.counterfactual_lr = counterfactual_lr  # Learning rate for unchosen option

        # Tracking
        self.choices: List[int] = []
        self.rewards: List[float] = []
        self.regrets: List[float] = []

    def select_action(self, state: int = 0) -> int:
        """Epsilon-greedy, but regret history can influence exploration."""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q)
        self.choices.append(action)
        return action

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        """Update with regret-based counterfactual learning."""
        self.rewards.append(reward)

        # Compute regret signal
        regret = self.regret_module.compute(context)
        self.regrets.append(regret)

        # Standard update for chosen action
        delta = reward - self.Q[action]

        # Regret modulates learning rate
        # High regret (negative) → learn MORE from this mistake
        # Relief (positive) → normal learning
        if regret < 0:
            regret_modulation = 1.0 + self.regret_weight * abs(regret)
        else:
            regret_modulation = 1.0

        self.Q[action] += self.lr * regret_modulation * delta

        # KEY DIFFERENCE: Also update unchosen action from counterfactual
        if context.counterfactual_shown:
            unchosen = 1 - action
            foregone_delta = context.foregone_reward - self.Q[unchosen]
            self.Q[unchosen] += self.counterfactual_lr * foregone_delta

    def reset_episode(self):
        self.regret_module.reset()
        self.choices = []
        self.rewards = []
        self.regrets = []


class RegretAvoidanceAgent(RegretEDAgent):
    """Agent that actively avoids actions that led to regret.

    Beyond learning from counterfactuals, this agent:
    - Tracks which actions led to high regret
    - Becomes more exploratory after regret (seeking better options)
    - Shows "regret aversion" - avoiding previously regretted choices

    This models the behavioral finding that humans avoid choices
    that previously led to regret, even when those choices have
    equal expected value.
    """

    def __init__(self, n_actions: int = 2, regret_aversion: float = 0.3, **kwargs):
        super().__init__(n_actions=n_actions, **kwargs)
        self.regret_aversion = regret_aversion
        self.action_regret_history = [[] for _ in range(n_actions)]

    def select_action(self, state: int = 0) -> int:
        """Selection with regret aversion bias."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Compute regret-adjusted Q values
        q_adjusted = self.Q.copy()

        for a in range(self.n_actions):
            if self.action_regret_history[a]:
                # Recent regret from this action
                recent_regret = np.mean(self.action_regret_history[a][-5:])
                if recent_regret < 0:
                    # Penalize actions that led to regret
                    q_adjusted[a] -= self.regret_aversion * abs(recent_regret)

        action = np.argmax(q_adjusted)
        self.choices.append(action)
        return action

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        """Update with regret tracking per action."""
        super().update(state, action, reward, next_state, done, context)

        # Track regret per action
        if context.counterfactual_shown:
            regret = context.obtained_reward - context.foregone_reward
            self.action_regret_history[action].append(regret)

    def reset_episode(self):
        super().reset_episode()
        self.action_regret_history = [[] for _ in range(self.n_actions)]
