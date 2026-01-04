"""Experiment 17: Pitch Black - Sparse Reward Key Search

A FAIR test environment for emotional learning where curiosity should help.

Environment:
- 10x10 grid, completely dark (no reward signal until goal)
- Key hidden somewhere, door at opposite corner
- Must find key first, then reach door
- Reward = 0 until door opened with key (maximally sparse)

Why Emotional Channels Should Help:
- Standard RL relies on epsilon-greedy luck (random walk)
- Curiosity channel broadcasts positive signal for unvisited states
- Agent systematically searches instead of random walking

Key Insight:
"Dense emotional signals are noise in dense reward environments,
but they become signal in sparse/harsh environments."

This environment is MAXIMALLY SPARSE - no reward until task complete.
Curiosity-ED should explore systematically and find key+door faster.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Set, Optional
from scipy import stats
import sys


@dataclass
class ExplorationContext:
    """Context for computing curiosity signals."""
    state: Tuple[int, int]
    visit_count: int
    has_key: bool
    steps_since_new_state: int
    total_unique_visited: int


class PitchBlackEnv:
    """
    Pitch Black Key Search Environment.

    10x10 grid with:
    - Agent starts at (0, 0)
    - Key at random position
    - Door at (9, 9)
    - Reward = 0 for all steps UNTIL door opened with key
    - Episode ends when door opened OR max steps reached

    This is MAXIMALLY SPARSE - no reward shaping, no intermediate signals.
    Only curiosity-driven exploration can help find the key efficiently.
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

    def __init__(self, size: int = 10, max_steps: int = 500):
        self.size = size
        self.max_steps = max_steps

        self.start_pos = (0, 0)
        self.door_pos = (size - 1, size - 1)

        # Key position will be randomized each reset
        self.key_pos = None

        # State tracking
        self.agent_pos = None
        self.has_key = False
        self.steps = 0
        self.visited_states: Set[Tuple[int, int]] = set()
        self.steps_since_new = 0

    def reset(self, key_pos: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Reset environment. Optionally specify key position."""
        self.agent_pos = self.start_pos
        self.has_key = False
        self.steps = 0
        self.visited_states = {self.start_pos}
        self.steps_since_new = 0

        # Random key position (not at start or door)
        if key_pos is not None:
            self.key_pos = key_pos
        else:
            while True:
                self.key_pos = (np.random.randint(0, self.size),
                               np.random.randint(0, self.size))
                if self.key_pos != self.start_pos and self.key_pos != self.door_pos:
                    break

        return self.agent_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, ExplorationContext]:
        """
        Take action in pitch black environment.

        Returns: (next_state, reward, done, context)

        Reward is ALWAYS 0 except:
        - +10.0 when opening door with key (success)
        """
        self.steps += 1

        # Move
        delta = self.ACTIONS[action]
        new_row = self.agent_pos[0] + delta[0]
        new_col = self.agent_pos[1] + delta[1]

        # Boundary check
        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            self.agent_pos = (new_row, new_col)

        # Check if new state
        is_new_state = self.agent_pos not in self.visited_states
        if is_new_state:
            self.visited_states.add(self.agent_pos)
            self.steps_since_new = 0
        else:
            self.steps_since_new += 1

        # Pick up key (silently - no reward)
        if self.agent_pos == self.key_pos and not self.has_key:
            self.has_key = True

        # Check door
        reward = 0.0
        done = False

        if self.agent_pos == self.door_pos and self.has_key:
            reward = 10.0  # SUCCESS!
            done = True
        elif self.steps >= self.max_steps:
            done = True  # Failed - ran out of time

        # Context for curiosity computation
        visit_count = sum(1 for s in self.visited_states if s == self.agent_pos)
        context = ExplorationContext(
            state=self.agent_pos,
            visit_count=visit_count,
            has_key=self.has_key,
            steps_since_new_state=self.steps_since_new,
            total_unique_visited=len(self.visited_states)
        )

        return self.agent_pos, reward, done, context

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) to flat index."""
        return state[0] * self.size + state[1]

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4


class CuriosityModule:
    """
    Computes curiosity signal based on state novelty.

    Curiosity = 1 / (1 + visit_count)
    Also considers time since last new state discovery.
    """

    def __init__(self, novelty_weight: float = 1.0, stagnation_threshold: int = 20):
        self.novelty_weight = novelty_weight
        self.stagnation_threshold = stagnation_threshold

        # Track visit counts per state
        self.visit_counts: dict = {}

    def compute(self, context: ExplorationContext) -> float:
        """
        Compute curiosity signal.

        High curiosity when:
        - State is rarely visited
        - Haven't found new states recently (stagnation → explore more)
        """
        # Update visit count
        state = context.state
        self.visit_counts[state] = self.visit_counts.get(state, 0) + 1

        # Novelty-based curiosity
        visit_count = self.visit_counts[state]
        novelty = 1.0 / (1.0 + visit_count)

        # Stagnation penalty → boost curiosity
        stagnation_bonus = 0.0
        if context.steps_since_new_state > self.stagnation_threshold:
            stagnation_bonus = min(0.5, context.steps_since_new_state / 100.0)

        curiosity = novelty + stagnation_bonus
        return min(1.0, curiosity * self.novelty_weight)

    def get_novelty_bonus(self, state: Tuple[int, int]) -> float:
        """Get curiosity bonus for a specific state."""
        visit_count = self.visit_counts.get(state, 0)
        return 1.0 / (1.0 + visit_count)

    def reset(self):
        """Reset for new episode."""
        self.visit_counts = {}


class StandardQLearner:
    """Standard Q-learning with epsilon-greedy - no curiosity."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.3):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions

    def select_action(self, state: int, context: Optional[ExplorationContext] = None) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: Optional[ExplorationContext] = None):
        """Standard TD update."""
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        """No per-episode state to reset."""
        pass


class CuriosityEDAgent:
    """
    Curiosity-driven Emotional ED Agent.

    Uses curiosity signal to:
    1. Provide intrinsic reward for novel states
    2. Bias action selection toward unexplored directions
    3. Increase exploration when stagnating
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.3,
                 curiosity_weight: float = 0.5, intrinsic_reward_scale: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions

        # Curiosity parameters
        self.curiosity_weight = curiosity_weight
        self.intrinsic_reward_scale = intrinsic_reward_scale

        # Curiosity module
        self.curiosity_module = CuriosityModule(novelty_weight=curiosity_weight)

        # Track current curiosity for action selection
        self.current_curiosity = 0.0

        # Track last state to compute exploration bonuses
        self.last_state = None

        # Grid size for direction biasing (assume square)
        self.grid_size = int(np.sqrt(n_states))

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert flat state index to (row, col)."""
        return (state // self.grid_size, state % self.grid_size)

    def _get_neighbor_novelty(self, state: int, action: int) -> float:
        """Get novelty of state reached by action."""
        pos = self._state_to_pos(state)
        delta = PitchBlackEnv.ACTIONS[action]
        new_row = pos[0] + delta[0]
        new_col = pos[1] + delta[1]

        # Check bounds
        if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
            return 0.0  # No novelty for invalid moves

        neighbor_pos = (new_row, new_col)
        return self.curiosity_module.get_novelty_bonus(neighbor_pos)

    def select_action(self, state: int, context: Optional[ExplorationContext] = None) -> int:
        """
        Curiosity-biased action selection.

        - Epsilon-greedy base
        - When curious, bias toward unexplored neighbors
        """
        # Update curiosity
        if context is not None:
            self.current_curiosity = self.curiosity_module.compute(context)

        # Random exploration
        if np.random.random() < self.epsilon:
            # Even random actions are curiosity-biased when curiosity high
            if self.current_curiosity > 0.3:
                # Compute novelty for each neighbor
                novelties = [self._get_neighbor_novelty(state, a) for a in range(self.n_actions)]

                # Softmax over novelties
                novelties = np.array(novelties)
                if novelties.sum() > 0:
                    probs = novelties / novelties.sum()
                    return np.random.choice(self.n_actions, p=probs)

            return np.random.randint(self.n_actions)

        # Greedy with curiosity bonus
        q_values = self.Q[state].copy()

        # Add curiosity bonus to Q-values for novel neighbors
        if self.current_curiosity > 0.2:
            for a in range(self.n_actions):
                novelty = self._get_neighbor_novelty(state, a)
                q_values[a] += novelty * self.curiosity_weight * 5

        return int(np.argmax(q_values))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: ExplorationContext):
        """
        TD update with intrinsic curiosity reward.

        Adds small reward for discovering new states.
        """
        # Compute intrinsic reward from novelty
        if context is not None:
            novelty = self.curiosity_module.get_novelty_bonus(context.state)
            intrinsic_reward = novelty * self.intrinsic_reward_scale
        else:
            intrinsic_reward = 0.0

        # Total reward = extrinsic + intrinsic
        total_reward = reward + intrinsic_reward

        # Standard TD update with augmented reward
        target = total_reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        # Curiosity modulates learning rate (learn more from novel experiences)
        effective_lr = self.lr * (1 + self.current_curiosity * 0.5)
        self.Q[state, action] += effective_lr * delta

        self.last_state = next_state

    def reset_episode(self):
        """Reset curiosity module for new episode."""
        self.curiosity_module.reset()
        self.current_curiosity = 0.0
        self.last_state = None


def run_episode(env: PitchBlackEnv, agent, training: bool = True) -> dict:
    """Run one episode and return metrics."""
    state = env.reset()
    state_idx = env.state_to_index(state)

    total_reward = 0.0
    steps = 0
    found_key = False
    reached_door = False
    key_found_step = None

    agent.reset_episode()

    # Initial context
    context = ExplorationContext(
        state=state,
        visit_count=1,
        has_key=False,
        steps_since_new_state=0,
        total_unique_visited=1
    )

    while True:
        # Select action
        action = agent.select_action(state_idx, context)

        # Take action
        next_state, reward, done, context = env.step(action)
        next_state_idx = env.state_to_index(next_state)

        # Track key finding
        if env.has_key and not found_key:
            found_key = True
            key_found_step = steps

        # Store transition and update
        if training:
            agent.update(state_idx, action, reward, next_state_idx, done, context)

        total_reward += reward
        steps += 1

        if reward > 0:  # Reached door with key
            reached_door = True

        if done:
            break

        state = next_state
        state_idx = next_state_idx

    return {
        'reward': total_reward,
        'steps': steps,
        'found_key': found_key,
        'key_found_step': key_found_step,
        'reached_door': reached_door,
        'unique_states_visited': len(env.visited_states),
        'exploration_efficiency': len(env.visited_states) / steps if steps > 0 else 0
    }


def run_experiment(n_seeds: int = 50, n_episodes: int = 200):
    """
    Run the Pitch Black experiment with statistical validation.

    Args:
        n_seeds: Number of random seeds for statistical validation
        n_episodes: Training episodes per seed
    """
    print("=" * 70)
    print("EXPERIMENT 17: PITCH BLACK (Sparse Reward / Curiosity)")
    print("=" * 70)
    print("\nEnvironment: 10x10 grid with hidden key, door at corner")
    print("Reward: ZERO until door opened with key (maximally sparse)")
    print("Hypothesis: Curiosity-ED explores systematically, finds solution faster")
    print(f"Running {n_seeds} seeds x {n_episodes} episodes")
    print()

    standard_results = []
    curiosity_results = []

    # Track success rates over time
    standard_success_curves = []
    curiosity_success_curves = []

    for seed in range(n_seeds):
        np.random.seed(seed)

        print(f"\rSeed {seed+1}/{n_seeds}", end='', flush=True)

        # Create consistent key positions across agents for this seed
        env_template = PitchBlackEnv(size=10, max_steps=500)
        key_positions = []
        for _ in range(n_episodes):
            while True:
                kp = (np.random.randint(0, 10), np.random.randint(0, 10))
                if kp != (0, 0) and kp != (9, 9):
                    key_positions.append(kp)
                    break

        # Reset RNG for fair comparison
        np.random.seed(seed)

        # --- Standard Q-learner ---
        env = PitchBlackEnv(size=10, max_steps=500)
        standard_agent = StandardQLearner(env.n_states, env.n_actions)

        seed_successes = []
        seed_steps_to_solve = []
        seed_exploration = []
        success_curve = []

        for ep in range(n_episodes):
            np.random.seed(seed * 10000 + ep)  # Reproducible per-episode
            result = run_episode(env, standard_agent, training=True)
            seed_successes.append(1 if result['reached_door'] else 0)
            success_curve.append(1 if result['reached_door'] else 0)
            if result['reached_door']:
                seed_steps_to_solve.append(result['steps'])
            seed_exploration.append(result['exploration_efficiency'])

        standard_success_curves.append(success_curve)
        standard_results.append({
            'success_rate': np.mean(seed_successes),
            'mean_steps': np.mean(seed_steps_to_solve) if seed_steps_to_solve else 500,
            'mean_exploration': np.mean(seed_exploration),
            'first_success_ep': next((i for i, s in enumerate(seed_successes) if s), n_episodes)
        })

        # --- Curiosity-ED Agent ---
        np.random.seed(seed)  # Reset for fair comparison

        env = PitchBlackEnv(size=10, max_steps=500)
        curiosity_agent = CuriosityEDAgent(env.n_states, env.n_actions)

        seed_successes = []
        seed_steps_to_solve = []
        seed_exploration = []
        success_curve = []

        for ep in range(n_episodes):
            np.random.seed(seed * 10000 + ep)  # Same per-episode seed
            result = run_episode(env, curiosity_agent, training=True)
            seed_successes.append(1 if result['reached_door'] else 0)
            success_curve.append(1 if result['reached_door'] else 0)
            if result['reached_door']:
                seed_steps_to_solve.append(result['steps'])
            seed_exploration.append(result['exploration_efficiency'])

        curiosity_success_curves.append(success_curve)
        curiosity_results.append({
            'success_rate': np.mean(seed_successes),
            'mean_steps': np.mean(seed_steps_to_solve) if seed_steps_to_solve else 500,
            'mean_exploration': np.mean(seed_exploration),
            'first_success_ep': next((i for i, s in enumerate(seed_successes) if s), n_episodes)
        })

    print("\n")

    # Aggregate results
    standard_success = [r['success_rate'] for r in standard_results]
    standard_steps = [r['mean_steps'] for r in standard_results]
    standard_exploration = [r['mean_exploration'] for r in standard_results]
    standard_first = [r['first_success_ep'] for r in standard_results]

    curiosity_success = [r['success_rate'] for r in curiosity_results]
    curiosity_steps = [r['mean_steps'] for r in curiosity_results]
    curiosity_exploration = [r['mean_exploration'] for r in curiosity_results]
    curiosity_first = [r['first_success_ep'] for r in curiosity_results]

    # Statistical tests
    success_t, success_p = stats.ttest_ind(curiosity_success, standard_success)
    steps_t, steps_p = stats.ttest_ind(curiosity_steps, standard_steps)
    explore_t, explore_p = stats.ttest_ind(curiosity_exploration, standard_exploration)
    first_t, first_p = stats.ttest_ind(curiosity_first, standard_first)

    # Effect sizes (Cohen's d)
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

    success_d = cohens_d(curiosity_success, standard_success)
    steps_d = cohens_d(curiosity_steps, standard_steps)
    explore_d = cohens_d(curiosity_exploration, standard_exploration)
    first_d = cohens_d(curiosity_first, standard_first)

    # Print results
    print("=" * 70)
    print("RESULTS: After Training")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Standard QL':<20} {'Curiosity-ED':<20} {'p-value':<10} {'Cohen d':<10}")
    print("-" * 85)

    print(f"{'Success Rate':<25} "
          f"{np.mean(standard_success):.3f} +/- {np.std(standard_success):.3f}  "
          f"{np.mean(curiosity_success):.3f} +/- {np.std(curiosity_success):.3f}  "
          f"{success_p:.4f}    {success_d:+.3f}")

    print(f"{'Mean Steps (solved)':<25} "
          f"{np.mean(standard_steps):.1f} +/- {np.std(standard_steps):.1f}    "
          f"{np.mean(curiosity_steps):.1f} +/- {np.std(curiosity_steps):.1f}    "
          f"{steps_p:.4f}    {steps_d:+.3f}")

    print(f"{'Exploration Efficiency':<25} "
          f"{np.mean(standard_exploration):.3f} +/- {np.std(standard_exploration):.3f}  "
          f"{np.mean(curiosity_exploration):.3f} +/- {np.std(curiosity_exploration):.3f}  "
          f"{explore_p:.4f}    {explore_d:+.3f}")

    print(f"{'First Success Episode':<25} "
          f"{np.mean(standard_first):.1f} +/- {np.std(standard_first):.1f}    "
          f"{np.mean(curiosity_first):.1f} +/- {np.std(curiosity_first):.1f}    "
          f"{first_p:.4f}    {first_d:+.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("\nHypothesis: Curiosity-ED should find key+door faster through")
    print("           systematic exploration rather than random walk.")

    # Success rate analysis
    if success_p < 0.05 and success_d > 0:
        print(f"\n[SUCCESS] Curiosity-ED has significantly higher success rate")
        print(f"  Effect size: d={success_d:.3f} ", end="")
        if abs(success_d) < 0.5:
            print("(small)")
        elif abs(success_d) < 0.8:
            print("(medium)")
        else:
            print("(large)")
    elif success_p < 0.05 and success_d < 0:
        print(f"\n[UNEXPECTED] Standard QL has higher success rate")
    else:
        print(f"\n[INCONCLUSIVE] No significant difference in success rate (p={success_p:.4f})")

    # Exploration analysis
    if explore_p < 0.05 and explore_d > 0:
        print(f"\n[SUCCESS] Curiosity-ED explores more efficiently")
        print(f"  (More unique states per step)")

    # First success analysis
    if first_p < 0.05 and first_d < 0:
        print(f"\n[SUCCESS] Curiosity-ED finds first solution faster")
        print(f"  (Episode {np.mean(curiosity_first):.0f} vs {np.mean(standard_first):.0f})")

    # Return results
    return {
        'standard': {
            'success_rate': standard_success,
            'mean_steps': standard_steps,
            'exploration': standard_exploration,
            'first_success': standard_first,
        },
        'curiosity': {
            'success_rate': curiosity_success,
            'mean_steps': curiosity_steps,
            'exploration': curiosity_exploration,
            'first_success': curiosity_first,
        },
        'stats': {
            'success': {'t': success_t, 'p': success_p, 'd': success_d},
            'steps': {'t': steps_t, 'p': steps_p, 'd': steps_d},
            'exploration': {'t': explore_t, 'p': explore_p, 'd': explore_d},
            'first': {'t': first_t, 'p': first_p, 'd': first_d},
        }
    }


if __name__ == "__main__":
    results = run_experiment(n_seeds=50, n_episodes=200)

    print("\n" + "=" * 70)
    print("EXPERIMENT 17 COMPLETE")
    print("=" * 70)
