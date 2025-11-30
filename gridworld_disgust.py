"""Grid-world for disgust channel testing.

Disgust is distinct from fear: contamination avoidance vs threat avoidance.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Set

@dataclass
class DisgustContext:
    """Context for disgust-related information."""
    contaminant_distance: float
    threat_distance: float
    goal_distance: float
    touched_contaminant: bool = False
    is_contaminated: bool = False  # Agent's contamination state
    contamination_spread: int = 0  # How many cells are contaminated


class DisgustGridWorld:
    """Grid-world testing disgust vs fear.

    Key distinction (Rozin et al., 2008):
    - Fear: Threat avoidance (predator, danger)
    - Disgust: Contamination avoidance (pathogens, decay)

    Disgust properties:
    1. Contamination spreads (touching spreads to agent)
    2. One-contact permanent (small amount is enough)
    3. Negative contagion (contaminated things contaminate others)
    4. Disgust doesn't habituate like fear

    Environment:
    - Threat (X): Dangerous but doesn't spread
    - Contaminant (C): Not immediately harmful but spreads contamination
    - Safe food (+): Reward
    - Contaminated food (?): Was food, now contaminated

    Test: Do agents treat contaminants differently than threats?
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 6):
        self.size = size
        self.goal_pos = np.array([5, 5])

        # Threat position (fear-inducing)
        self.threat_pos = np.array([2, 2])

        # Initial contaminant position (disgust-inducing)
        self.initial_contaminant = np.array([2, 4])

        # Food positions
        self.food_positions = [
            np.array([1, 1]),  # Far from both
            np.array([1, 5]),  # Near contaminant
            np.array([4, 2]),  # Near threat
        ]

        self.reset()

    def reset(self) -> int:
        self.agent_pos = np.array([0, 0])
        self.step_count = 0

        # Contamination state
        self.agent_contaminated = False
        self.contaminated_cells: Set[Tuple[int, int]] = set()
        self.contaminated_cells.add(tuple(self.initial_contaminant))

        # Food state
        self.collected_food = set()
        self.contaminated_food = set()

        # Check if any food starts contaminated
        for i, food in enumerate(self.food_positions):
            if tuple(food) in self.contaminated_cells:
                self.contaminated_food.add(i)

        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def _spread_contamination(self, pos: np.ndarray):
        """Contamination spreads to adjacent cells."""
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                new_pos = (pos[0] + dx, pos[1] + dy)
                if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
                    self.contaminated_cells.add(new_pos)

        # Check if any food is now contaminated
        for i, food in enumerate(self.food_positions):
            if tuple(food) in self.contaminated_cells:
                self.contaminated_food.add(i)

    def step(self, action: int) -> Tuple[int, float, bool, DisgustContext]:
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        # Boundary check
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        # Compute distances
        threat_dist = self._distance(self.agent_pos, self.threat_pos)
        contam_dist = min(self._distance(self.agent_pos, np.array(c))
                          for c in self.contaminated_cells) if self.contaminated_cells else float('inf')
        goal_dist = self._distance(self.agent_pos, self.goal_pos)

        # Base reward
        reward = -0.01

        # Threat penalty (immediate harm, no spread)
        if threat_dist < 1.0:
            reward -= 0.5

        # Contamination check (spreads but less immediate harm)
        touched_contaminant = False
        if tuple(self.agent_pos) in self.contaminated_cells:
            touched_contaminant = True
            if not self.agent_contaminated:
                self.agent_contaminated = True
                reward -= 0.1  # Less immediate than threat

            # Agent spreads contamination
            self._spread_contamination(self.agent_pos)

        # Food collection
        for i, food in enumerate(self.food_positions):
            if np.array_equal(self.agent_pos, food) and i not in self.collected_food:
                self.collected_food.add(i)
                if i in self.contaminated_food:
                    # Eating contaminated food
                    reward -= 0.3  # Worse than touching
                    self.agent_contaminated = True
                else:
                    # Clean food
                    reward += 0.4

        # Goal reached
        done = np.array_equal(self.agent_pos, self.goal_pos) or self.step_count >= 100
        if np.array_equal(self.agent_pos, self.goal_pos):
            if self.agent_contaminated:
                reward += 0.5  # Reduced goal reward if contaminated
            else:
                reward += 1.0

        context = DisgustContext(
            contaminant_distance=contam_dist,
            threat_distance=threat_dist,
            goal_distance=goal_dist,
            touched_contaminant=touched_contaminant,
            is_contaminated=self.agent_contaminated,
            contamination_spread=len(self.contaminated_cells)
        )

        return self._pos_to_state(self.agent_pos), reward, done, context

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4

    def render(self) -> str:
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]

        # Mark contaminated cells
        for c in self.contaminated_cells:
            grid[c[0]][c[1]] = '~'

        # Mark threat
        grid[self.threat_pos[0]][self.threat_pos[1]] = 'X'

        # Mark food
        for i, food in enumerate(self.food_positions):
            if i not in self.collected_food:
                if i in self.contaminated_food:
                    grid[food[0]][food[1]] = '?'  # Contaminated food
                else:
                    grid[food[0]][food[1]] = '+'  # Clean food

        # Mark goal
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'

        # Mark agent
        agent_marker = 'a' if self.agent_contaminated else 'A'
        grid[self.agent_pos[0]][self.agent_pos[1]] = agent_marker

        return '\n'.join([' '.join(row) for row in grid])
