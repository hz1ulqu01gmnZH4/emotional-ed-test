"""Agents for grief/attachment loss testing."""

import numpy as np
from typing import List, Optional
from gridworld_grief import EmotionalContext

class StandardQLearner:
    """Standard Q-learning - immediate adaptation to changes."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # Tracking
        self.visits_to_resource_location: List[int] = []
        self.visit_times: List[int] = []

    def select_action(self, state: int) -> int:
        """Epsilon-greedy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        """Standard TD update."""
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.lr * delta

    def track_visit(self, at_resource_location: bool, step: int):
        """Track visits to resource location."""
        if at_resource_location:
            self.visits_to_resource_location.append(step)

    def reset_tracking(self):
        self.visits_to_resource_location = []
        self.visit_times = []


class GriefModule:
    """Models grief as attachment homeostasis disruption.

    Based on Panksepp's PANIC/GRIEF system:
    - Attachment creates "set point" for expected interaction
    - Loss creates deviation from set point → distress signal
    - Grief motivates seeking behavior (yearning)
    - Gradual adaptation (acceptance) as set point adjusts

    Key phases (Bowlby, Kübler-Ross):
    1. Yearning/Searching - elevated seeking of lost object
    2. Despair - continued absence, reduced activity
    3. Acceptance - set point adjusts, normal behavior resumes
    """

    def __init__(self, attachment_strength: float = 1.0,
                 yearning_duration: int = 30,
                 adaptation_rate: float = 0.05):
        self.attachment_strength = attachment_strength
        self.yearning_duration = yearning_duration
        self.adaptation_rate = adaptation_rate

        # State
        self.grief_level = 0.0
        self.yearning = 0.0
        self.attachment_baseline = 0.0  # Learned expectation
        self.loss_occurred = False

    def compute(self, context: EmotionalContext) -> dict:
        """Compute grief-related signals."""
        # Build attachment through successful resource collection
        if context.resource_obtained:
            self.attachment_baseline = min(1.0, self.attachment_baseline + 0.1)

        # Loss event triggers grief
        if context.resource_lost:
            self.loss_occurred = True
            self.grief_level = self.attachment_baseline * self.attachment_strength
            self.yearning = self.grief_level  # Initial yearning = grief level

        # Grief dynamics after loss
        if self.loss_occurred:
            # Yearning decays over time (adaptation)
            time_factor = context.time_since_loss / self.yearning_duration
            self.yearning = self.grief_level * max(0, 1 - time_factor)

            # Grief slowly reduces (acceptance)
            self.grief_level *= (1 - self.adaptation_rate)

        return {
            'grief': self.grief_level,
            'yearning': self.yearning,
            'attachment': self.attachment_baseline
        }

    def reset(self):
        self.grief_level = 0.0
        self.yearning = 0.0
        self.attachment_baseline = 0.0
        self.loss_occurred = False


class GriefEDAgent:
    """Q-learning with grief/attachment channel.

    Key behavioral predictions:
    - Before loss: Normal resource-seeking behavior
    - After loss: "Yearning phase" - continued visits to lost resource location
    - Gradual adaptation: Eventually stops visiting

    Standard agent: Immediately learns resource is gone, stops visiting
    Grief agent: Yearning maintains value of resource location temporarily
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 grief_weight: float = 0.5):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # Grief module
        self.grief_module = GriefModule()
        self.grief_weight = grief_weight

        # Track resource location for yearning behavior
        self.resource_state: Optional[int] = None

        # Tracking
        self.visits_to_resource_location: List[int] = []

    def select_action(self, state: int) -> int:
        """Action selection with yearning bias."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Yearning: bias toward actions that lead to lost resource location
        if self.grief_module.yearning > 0 and self.resource_state is not None:
            # Simple heuristic: boost Q-values for actions
            # This is approximate - proper implementation would use
            # successor features or model-based planning
            yearning_boost = self.grief_module.yearning * self.grief_weight
            # Boost all actions proportionally (simplified)
            q_values += yearning_boost * 0.1

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        """Update with grief modulation."""
        # Update grief module
        grief_signals = self.grief_module.compute(context)

        # Track resource location
        if context.resource_obtained:
            self.resource_state = next_state

        # Standard TD error
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        # Grief modulation: slow down negative learning for resource location
        # (Yearning maintains expected value longer)
        if grief_signals['yearning'] > 0:
            # For ANY state leading toward resource, slow negative learning
            if delta < 0:
                # Slow down all negative learning during yearning
                # This maintains the "pull" toward lost resource
                delta *= (1 - grief_signals['yearning'] * self.grief_weight * 0.8)

        self.Q[state, action] += self.lr * delta

    def track_visit(self, at_resource_location: bool, step: int):
        """Track visits to resource location."""
        if at_resource_location:
            self.visits_to_resource_location.append(step)

    def reset_episode(self):
        self.grief_module.reset()
        self.visits_to_resource_location = []
        self.resource_state = None
