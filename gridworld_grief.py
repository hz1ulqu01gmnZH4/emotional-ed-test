"""Grid-world with attachment and loss for grief testing."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Set

@dataclass
class EmotionalContext:
    """Context for computing emotional signals."""
    threat_distance: float = float('inf')
    goal_distance: float = 0.0
    was_blocked: bool = False
    # Attachment/grief signals
    resource_present: bool = True
    resource_distance: float = 0.0
    resource_obtained: bool = False
    resource_lost: bool = False  # Did resource just disappear?
    time_since_loss: int = 0

class AttachmentGridWorld:
    """Grid-world with a resource the agent becomes attached to.

    Paradigm:
    1. Agent explores grid with renewable resource at fixed location
    2. Agent learns resource location, visits regularly
    3. Resource disappears permanently (loss event)
    4. Measure: How long does agent keep visiting old location?

    Key prediction:
    - Standard agent: Immediately updates, stops visiting
    - Grief agent: Shows "yearning" - continues visiting before adapting
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 5, resource_pos: Tuple[int, int] = (2, 2),
                 loss_step: int = 50, resource_value: float = 0.5):
        """
        Args:
            size: Grid size
            resource_pos: Location of renewable resource
            loss_step: Step at which resource disappears permanently
            resource_value: Reward from collecting resource
        """
        self.size = size
        self.resource_pos = np.array(resource_pos)
        self.loss_step = loss_step
        self.resource_value = resource_value
        self.reset()

    def reset(self) -> int:
        """Reset environment."""
        self.agent_pos = np.array([0, 0])
        self.step_count = 0
        self.resource_available = True
        self.resource_present = True  # Resource exists in world
        self.resource_cooldown = 0  # Steps until resource regenerates
        self.loss_occurred = False
        self.time_since_loss = 0
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        """Convert position to state index."""
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Euclidean distance."""
        return np.linalg.norm(pos1 - pos2)

    def step(self, action: int) -> Tuple[int, float, bool, EmotionalContext]:
        """Take action."""
        self.step_count += 1

        # Check for loss event
        resource_just_lost = False
        if self.step_count >= self.loss_step and self.resource_present:
            self.resource_present = False
            self.loss_occurred = True
            resource_just_lost = True

        if self.loss_occurred:
            self.time_since_loss += 1

        # Move agent
        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        was_blocked = False
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            was_blocked = True
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        # Check resource collection
        reward = -0.01  # Step penalty
        resource_obtained = False

        at_resource = np.array_equal(self.agent_pos, self.resource_pos)

        if at_resource and self.resource_present and self.resource_available:
            reward = self.resource_value
            resource_obtained = True
            self.resource_available = False
            self.resource_cooldown = 5  # Regenerates after 5 steps

        # Resource regeneration (before loss)
        if self.resource_cooldown > 0:
            self.resource_cooldown -= 1
            if self.resource_cooldown == 0 and self.resource_present:
                self.resource_available = True

        # Episode ends after fixed steps
        done = self.step_count >= 150

        context = EmotionalContext(
            resource_present=self.resource_present,
            resource_distance=self._distance(self.agent_pos, self.resource_pos),
            resource_obtained=resource_obtained,
            resource_lost=resource_just_lost,
            time_since_loss=self.time_since_loss if self.loss_occurred else 0,
            was_blocked=was_blocked
        )

        return self._pos_to_state(self.agent_pos), reward, done, context

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4

    def render(self) -> str:
        """ASCII render."""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        if self.resource_present:
            marker = 'R' if self.resource_available else 'r'
            grid[self.resource_pos[0]][self.resource_pos[1]] = marker
        else:
            grid[self.resource_pos[0]][self.resource_pos[1]] = '_'  # Lost resource location
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        return '\n'.join([' '.join(row) for row in grid])
