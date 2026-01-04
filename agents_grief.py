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
                 yearning_duration: int = 80,  # Extended from 30 to 80
                 adaptation_rate: float = 0.02):  # Slower adaptation
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
            # Yearning decays over time (adaptation) - slower decay curve
            time_factor = context.time_since_loss / self.yearning_duration
            self.yearning = self.grief_level * max(0, 1 - time_factor ** 0.5)  # Square root for slower decay

            # Grief slowly reduces (acceptance)
            self.grief_level *= (1 - self.adaptation_rate)

        return {
            'grief': self.grief_level,
            'yearning': self.yearning,
            'attachment': self.attachment_baseline
        }

    def reset_episode(self):
        """Reset episode-specific state but PRESERVE attachment."""
        self.grief_level = 0.0
        self.yearning = 0.0
        self.loss_occurred = False
        # NOTE: attachment_baseline is NOT reset - it persists across episodes

    def reset(self):
        """Full reset including attachment (use for new agent only)."""
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
                 grief_weight: float = 0.5, grid_size: int = 5):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.grid_size = grid_size

        # Grief module
        self.grief_module = GriefModule()
        self.grief_weight = grief_weight

        # Track resource location for yearning behavior
        self.resource_state: Optional[int] = None
        self.resource_pos: Optional[tuple] = None

        # Tracking
        self.visits_to_resource_location: List[int] = []

    def _state_to_pos(self, state: int) -> tuple:
        """Convert state to (row, col) position."""
        return (state // self.grid_size, state % self.grid_size)

    def _get_action_toward_resource(self, state: int) -> Optional[int]:
        """Get action that moves toward resource location."""
        if self.resource_pos is None:
            return None

        current_pos = self._state_to_pos(state)
        target_pos = self.resource_pos

        # Actions: 0=up(-1,0), 1=down(+1,0), 2=left(0,-1), 3=right(0,+1)
        row_diff = target_pos[0] - current_pos[0]
        col_diff = target_pos[1] - current_pos[1]

        # Prefer larger difference direction
        if abs(row_diff) >= abs(col_diff):
            if row_diff < 0:
                return 0  # up
            elif row_diff > 0:
                return 1  # down
        if col_diff < 0:
            return 2  # left
        elif col_diff > 0:
            return 3  # right

        return None  # Already at resource

    def select_action(self, state: int) -> int:
        """Action selection with directional yearning bias."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Yearning: DIRECTIONAL bias toward lost resource location
        if self.grief_module.yearning > 0 and self.resource_pos is not None:
            best_action = self._get_action_toward_resource(state)
            if best_action is not None:
                # Boost Q-value for action that moves toward resource
                yearning_boost = self.grief_module.yearning * self.grief_weight * 0.5
                q_values[best_action] += yearning_boost

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        """Update with grief modulation."""
        # Update grief module
        grief_signals = self.grief_module.compute(context)

        # Track resource location
        if context.resource_obtained:
            self.resource_state = next_state
            self.resource_pos = self._state_to_pos(next_state)

        # Standard TD error
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        # Grief modulation: slow down negative learning for resource location
        # (Yearning maintains expected value longer)
        if grief_signals['yearning'] > 0:
            if delta < 0:
                # Slow down negative learning during yearning
                # This maintains the "pull" toward lost resource
                slowdown = 1 - grief_signals['yearning'] * self.grief_weight * 0.9
                delta *= max(0.1, slowdown)  # At least 10% learning

        self.Q[state, action] += self.lr * delta

    def track_visit(self, at_resource_location: bool, step: int):
        """Track visits to resource location."""
        if at_resource_location:
            self.visits_to_resource_location.append(step)

    def reset_episode(self):
        """Reset episode state but preserve attachment."""
        self.grief_module.reset_episode()  # Preserves attachment_baseline
        self.visits_to_resource_location = []
        # NOTE: resource_state and resource_pos are preserved for yearning

    def reset_full(self):
        """Full reset for new agent."""
        self.grief_module.reset()
        self.visits_to_resource_location = []
        self.resource_state = None
        self.resource_pos = None
