"""Experiment 20: Bottleneck Trap - Jammed Door Persistence Test

A FAIR test environment for emotional learning where anger/persistence helps.

Environment:
- MiniGrid-DoorKey variant
- Door is "jammed" - requires 3-5 consecutive PUSH actions to open
- Standard navigation gives small negative rewards (time penalty)
- Goal is behind the jammed door

Why Emotional Channels Should Help:
- Standard DQN treats door as wall after 1-2 failed attempts
- Gives up and wanders, never opening the door
- Anger/Frustration accumulates when blocked at door
- High anger -> increase persistence (keep trying same action)
- Enables "pushing through" local minima

Key Insight:
"Dense emotional signals are noise in dense reward environments,
but they become signal in sparse/harsh environments."

This environment requires PERSISTENCE - trying the same action repeatedly.
Standard RL explores away from "failed" actions. Anger-ED persists.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Set, Optional
from scipy import stats


@dataclass
class FrustrationContext:
    """Context for computing anger/frustration signals."""
    was_blocked: bool
    at_door: bool
    door_pushes: int  # How many times door has been pushed
    pushes_needed: int  # How many pushes needed to open
    consecutive_failures: int
    total_door_attempts: int


class BottleneckEnv:
    """
    Bottleneck/Jammed Door Environment.

    6x6 grid with:
    - Agent starts at (0, 0)
    - Jammed door at (2, 3) - requires N consecutive pushes
    - Goal at (5, 5) - only reachable through door
    - Wall separating starting area from goal area

    Layout:
    A . . | . G
    . . . | . .
    . . . D . .   (D = jammed door in wall)
    . . . | . .
    . . . | . .
    . . . | . .

    Door mechanics:
    - Agent must be adjacent to door and push toward it
    - Push counter increments on each push
    - After N pushes, door opens (becomes passable)
    - If agent moves away, push counter resets (must be consecutive)
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

    def __init__(self, size: int = 6, pushes_needed: int = 4, max_steps: int = 200):
        self.size = size
        self.pushes_needed = pushes_needed
        self.max_steps = max_steps

        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.door_pos = (2, 3)  # In the middle of the wall

        # Wall positions (vertical wall at column 3, except door)
        self.wall_positions = set()
        for row in range(size):
            if row != 2:  # Door is at row 2
                self.wall_positions.add((row, 3))

        # Episode state
        self.agent_pos = None
        self.steps = 0
        self.door_open = False
        self.door_push_count = 0
        self.was_at_door = False
        self.consecutive_failures = 0
        self.total_door_attempts = 0

    def reset(self) -> Tuple[int, int]:
        """Reset environment."""
        self.agent_pos = self.start_pos
        self.steps = 0
        self.door_open = False
        self.door_push_count = 0
        self.was_at_door = False
        self.consecutive_failures = 0
        self.total_door_attempts = 0

        return self.agent_pos

    def _is_adjacent_to_door(self, pos: Tuple[int, int]) -> bool:
        """Check if position is adjacent to door."""
        row, col = pos
        door_row, door_col = self.door_pos
        return abs(row - door_row) + abs(col - door_col) == 1

    def _is_pushing_door(self, pos: Tuple[int, int], action: int) -> bool:
        """Check if action from position would push the door."""
        delta = self.ACTIONS[action]
        new_row = pos[0] + delta[0]
        new_col = pos[1] + delta[1]
        return (new_row, new_col) == self.door_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, FrustrationContext]:
        """
        Take action in the environment.

        Returns: (next_state, reward, done, context)
        """
        self.steps += 1
        was_blocked = False
        at_door = False

        # Compute intended move
        delta = self.ACTIONS[action]
        new_row = self.agent_pos[0] + delta[0]
        new_col = self.agent_pos[1] + delta[1]
        intended_pos = (new_row, new_col)

        # Check if trying to push door
        pushing_door = self._is_pushing_door(self.agent_pos, action)

        if pushing_door and not self.door_open:
            at_door = True
            self.door_push_count += 1
            self.total_door_attempts += 1
            self.consecutive_failures += 1  # Blocked until door opens

            # Check if door opens
            if self.door_push_count >= self.pushes_needed:
                self.door_open = True
                self.consecutive_failures = 0  # Success!
                # Move through door
                self.agent_pos = self.door_pos
            else:
                # Door still jammed, stay in place
                was_blocked = True

        elif intended_pos in self.wall_positions:
            # Hit wall
            was_blocked = True
            self.consecutive_failures += 1

        elif not (0 <= new_row < self.size and 0 <= new_col < self.size):
            # Out of bounds
            was_blocked = True
            self.consecutive_failures += 1

        elif intended_pos == self.door_pos and not self.door_open:
            # Door position but not pushing (shouldn't happen with adjacency)
            was_blocked = True
            self.consecutive_failures += 1

        else:
            # Valid move
            old_pos = self.agent_pos
            self.agent_pos = intended_pos
            self.consecutive_failures = 0

            # If we moved away from door without opening it, reset push count
            if self.was_at_door and not at_door and not self.door_open:
                self.door_push_count = 0  # Reset! Must be consecutive

        self.was_at_door = at_door

        # Compute reward
        reward = -0.01  # Step cost

        done = False
        if self.agent_pos == self.goal_pos:
            reward = 10.0  # Success!
            done = True
        elif self.steps >= self.max_steps:
            done = True

        # Context for frustration computation
        context = FrustrationContext(
            was_blocked=was_blocked,
            at_door=at_door,
            door_pushes=self.door_push_count,
            pushes_needed=self.pushes_needed,
            consecutive_failures=self.consecutive_failures,
            total_door_attempts=self.total_door_attempts
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


class AngerModule:
    """
    Computes anger/frustration signal from blocking events.

    Anger builds when:
    - Agent is blocked repeatedly
    - Agent is at door but can't open it
    - Progress toward goal is stalled

    High anger -> increase persistence (keep trying blocked action).
    """

    def __init__(self, frustration_decay: float = 0.85, max_anger: float = 1.0):
        self.frustration_decay = frustration_decay
        self.max_anger = max_anger

        # Anger level
        self.anger = 0.0

        # Track repeated blocking
        self.consecutive_blocks = 0
        self.last_blocked_action = None

    def compute(self, context: FrustrationContext) -> float:
        """
        Compute anger signal from context.

        Anger increases when blocked, especially at door.
        """
        # Phasic anger from blocking
        phasic_anger = 0.0

        if context.was_blocked:
            self.consecutive_blocks += 1
            phasic_anger = min(0.5, self.consecutive_blocks * 0.1)

            # Extra anger when at door (we KNOW it should open)
            if context.at_door:
                progress = context.door_pushes / context.pushes_needed
                phasic_anger += 0.3 * progress  # More angry as we get closer

        else:
            self.consecutive_blocks = 0

        # Update tonic anger
        if phasic_anger > 0:
            self.anger = min(self.max_anger, self.anger + 0.2)
        else:
            self.anger *= self.frustration_decay

        return max(self.anger, phasic_anger)

    def get_persistence_boost(self) -> float:
        """
        Get persistence multiplier based on anger.

        High anger -> higher probability of repeating blocked action.
        """
        return 1.0 + self.anger * 2.0  # Up to 3x persistence

    def reset(self):
        """Reset for new episode."""
        self.anger = 0.0
        self.consecutive_blocks = 0
        self.last_blocked_action = None


class StandardQLearner:
    """Standard Q-learning - gives up after blocked actions."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.2):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions

    def select_action(self, state: int, context: Optional[FrustrationContext] = None) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: Optional[FrustrationContext] = None):
        """Standard TD update."""
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        """No per-episode state to reset."""
        pass


class AngerEDAgent:
    """
    Anger/Persistence-driven Emotional ED Agent.

    Uses anger signal to:
    1. Detect blocking events (door not opening)
    2. Increase persistence when angry (keep pushing door)
    3. Learn that persistence pays off

    Key insight: Standard RL penalizes "failed" actions and explores away.
    Anger-ED recognizes that some failures require persistence, not avoidance.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.2,
                 anger_weight: float = 0.5):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions
        self.anger_weight = anger_weight

        # Anger module
        self.anger_module = AngerModule()

        # Track current anger and last action
        self.current_anger = 0.0
        self.last_action = None
        self.last_was_blocked = False

        # Grid size for action reasoning
        self.grid_size = int(np.sqrt(n_states))

    def select_action(self, state: int, context: Optional[FrustrationContext] = None) -> int:
        """
        Anger-modulated action selection.

        High anger + blocked -> repeat last action (persistence).
        """
        # Update anger from context
        if context is not None:
            self.current_anger = self.anger_module.compute(context)
            was_blocked = context.was_blocked
            at_door = context.at_door
        else:
            was_blocked = False
            at_door = False

        # Persistence mechanism: if angry and was blocked, repeat action
        if self.last_action is not None and was_blocked and self.current_anger > 0.3:
            persistence_prob = min(0.8, self.current_anger * 0.6)
            if np.random.random() < persistence_prob:
                return self.last_action  # Keep pushing!

        # Standard epsilon-greedy with anger influence
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Greedy selection with potential anger bias
        q_values = self.Q[state].copy()

        # If we know we were at door and making progress, boost that action
        if at_door and self.last_action is not None and self.current_anger > 0.2:
            # Boost Q-value of last action (persistence bias)
            q_values[self.last_action] += self.current_anger * 5

        action = int(np.argmax(q_values))
        self.last_action = action
        return action

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: FrustrationContext):
        """
        TD update with anger-modulated learning.

        Key difference from standard RL:
        - When blocked at door, don't fully penalize the action
        - Recognize that persistence is building toward success
        """
        # Standard TD target
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        # Anger modulation: reduce negative learning when at door
        # (We're making progress even if reward is negative)
        if context.at_door and delta < 0:
            # Reduce magnitude of negative update proportional to progress
            progress = context.door_pushes / context.pushes_needed
            delta *= (1 - progress * 0.5)  # Up to 50% reduction in penalty

        # Also boost learning rate when angry (learn faster from persistence)
        lr_multiplier = 1.0 + self.current_anger * 0.5
        effective_lr = self.lr * lr_multiplier

        self.Q[state, action] += effective_lr * delta

        # Track last blocked state
        self.last_was_blocked = context.was_blocked

    def reset_episode(self):
        """Reset anger module for new episode."""
        self.anger_module.reset()
        self.current_anger = 0.0
        self.last_action = None
        self.last_was_blocked = False


def run_episode(env: BottleneckEnv, agent, training: bool = True) -> dict:
    """Run one episode and return metrics."""
    state = env.reset()
    state_idx = env.state_to_index(state)

    total_reward = 0.0
    steps = 0
    reached_goal = False
    door_opened = False
    max_consecutive_pushes = 0

    agent.reset_episode()

    # Initial context
    context = FrustrationContext(
        was_blocked=False,
        at_door=False,
        door_pushes=0,
        pushes_needed=env.pushes_needed,
        consecutive_failures=0,
        total_door_attempts=0
    )

    while True:
        # Select action
        action = agent.select_action(state_idx, context)

        # Take action
        next_state, reward, done, context = env.step(action)
        next_state_idx = env.state_to_index(next_state)

        # Track door progress
        max_consecutive_pushes = max(max_consecutive_pushes, env.door_push_count)

        # Store transition and update
        if training:
            agent.update(state_idx, action, reward, next_state_idx, done, context)

        total_reward += reward
        steps += 1

        if env.door_open:
            door_opened = True

        if reward > 5:  # Reached goal
            reached_goal = True

        if done:
            break

        state = next_state
        state_idx = next_state_idx

    return {
        'reward': total_reward,
        'steps': steps,
        'reached_goal': reached_goal,
        'door_opened': door_opened,
        'max_consecutive_pushes': max_consecutive_pushes,
        'total_door_attempts': env.total_door_attempts
    }


def run_experiment(n_seeds: int = 50, n_episodes: int = 300, pushes_needed: int = 4):
    """
    Run the Bottleneck experiment with statistical validation.

    Args:
        n_seeds: Number of random seeds for statistical validation
        n_episodes: Training episodes per seed
        pushes_needed: Consecutive pushes required to open door
    """
    print("=" * 70)
    print("EXPERIMENT 20: BOTTLENECK TRAP (Persistence / Anger)")
    print("=" * 70)
    print(f"\nEnvironment: 6x6 grid with jammed door requiring {pushes_needed} consecutive pushes")
    print("Standard RL: Treats blocked action as failure, explores away")
    print("Hypothesis: Anger-ED persists at door and opens it")
    print(f"Running {n_seeds} seeds x {n_episodes} episodes")
    print()

    standard_results = []
    anger_results = []

    for seed in range(n_seeds):
        np.random.seed(seed)

        print(f"\rSeed {seed+1}/{n_seeds}", end='', flush=True)

        # --- Standard Q-learner ---
        np.random.seed(seed)
        env = BottleneckEnv(size=6, pushes_needed=pushes_needed, max_steps=200)
        standard_agent = StandardQLearner(env.n_states, env.n_actions)

        seed_success = []
        seed_door_opened = []
        seed_max_pushes = []
        first_success = n_episodes

        for ep in range(n_episodes):
            result = run_episode(env, standard_agent, training=True)
            seed_success.append(1 if result['reached_goal'] else 0)
            seed_door_opened.append(1 if result['door_opened'] else 0)
            seed_max_pushes.append(result['max_consecutive_pushes'])

            if result['reached_goal'] and first_success == n_episodes:
                first_success = ep

        standard_results.append({
            'success_rate': np.mean(seed_success),
            'door_open_rate': np.mean(seed_door_opened),
            'mean_max_pushes': np.mean(seed_max_pushes),
            'first_success_ep': first_success,
            'final_success_rate': np.mean(seed_success[-50:]) if len(seed_success) >= 50 else np.mean(seed_success)
        })

        # --- Anger-ED Agent ---
        np.random.seed(seed)  # Reset for fair comparison
        env = BottleneckEnv(size=6, pushes_needed=pushes_needed, max_steps=200)
        anger_agent = AngerEDAgent(env.n_states, env.n_actions)

        seed_success = []
        seed_door_opened = []
        seed_max_pushes = []
        first_success = n_episodes

        for ep in range(n_episodes):
            result = run_episode(env, anger_agent, training=True)
            seed_success.append(1 if result['reached_goal'] else 0)
            seed_door_opened.append(1 if result['door_opened'] else 0)
            seed_max_pushes.append(result['max_consecutive_pushes'])

            if result['reached_goal'] and first_success == n_episodes:
                first_success = ep

        anger_results.append({
            'success_rate': np.mean(seed_success),
            'door_open_rate': np.mean(seed_door_opened),
            'mean_max_pushes': np.mean(seed_max_pushes),
            'first_success_ep': first_success,
            'final_success_rate': np.mean(seed_success[-50:]) if len(seed_success) >= 50 else np.mean(seed_success)
        })

    print("\n")

    # Aggregate results
    standard_success = [r['success_rate'] for r in standard_results]
    standard_door = [r['door_open_rate'] for r in standard_results]
    standard_pushes = [r['mean_max_pushes'] for r in standard_results]
    standard_first = [r['first_success_ep'] for r in standard_results]
    standard_final = [r['final_success_rate'] for r in standard_results]

    anger_success = [r['success_rate'] for r in anger_results]
    anger_door = [r['door_open_rate'] for r in anger_results]
    anger_pushes = [r['mean_max_pushes'] for r in anger_results]
    anger_first = [r['first_success_ep'] for r in anger_results]
    anger_final = [r['final_success_rate'] for r in anger_results]

    # Statistical tests
    success_t, success_p = stats.ttest_ind(anger_success, standard_success)
    door_t, door_p = stats.ttest_ind(anger_door, standard_door)
    pushes_t, pushes_p = stats.ttest_ind(anger_pushes, standard_pushes)
    first_t, first_p = stats.ttest_ind(anger_first, standard_first)
    final_t, final_p = stats.ttest_ind(anger_final, standard_final)

    # Effect sizes (Cohen's d)
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

    success_d = cohens_d(anger_success, standard_success)
    door_d = cohens_d(anger_door, standard_door)
    pushes_d = cohens_d(anger_pushes, standard_pushes)
    first_d = cohens_d(anger_first, standard_first)
    final_d = cohens_d(anger_final, standard_final)

    # Print results
    print("=" * 70)
    print("RESULTS: After Training")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Standard QL':<22} {'Anger-ED':<22} {'p-value':<10} {'Cohen d':<10}")
    print("-" * 90)

    print(f"{'Overall Success Rate':<25} "
          f"{np.mean(standard_success):.3f} +/- {np.std(standard_success):.3f}  "
          f"{np.mean(anger_success):.3f} +/- {np.std(anger_success):.3f}  "
          f"{success_p:.4f}    {success_d:+.3f}")

    print(f"{'Door Open Rate':<25} "
          f"{np.mean(standard_door):.3f} +/- {np.std(standard_door):.3f}  "
          f"{np.mean(anger_door):.3f} +/- {np.std(anger_door):.3f}  "
          f"{door_p:.4f}    {door_d:+.3f}")

    print(f"{'Mean Max Consecutive Push':<25} "
          f"{np.mean(standard_pushes):.2f} +/- {np.std(standard_pushes):.2f}    "
          f"{np.mean(anger_pushes):.2f} +/- {np.std(anger_pushes):.2f}    "
          f"{pushes_p:.4f}    {pushes_d:+.3f}")

    print(f"{'First Success Episode':<25} "
          f"{np.mean(standard_first):.1f} +/- {np.std(standard_first):.1f}    "
          f"{np.mean(anger_first):.1f} +/- {np.std(anger_first):.1f}    "
          f"{first_p:.4f}    {first_d:+.3f}")

    print(f"{'Final 50-ep Success Rate':<25} "
          f"{np.mean(standard_final):.3f} +/- {np.std(standard_final):.3f}  "
          f"{np.mean(anger_final):.3f} +/- {np.std(anger_final):.3f}  "
          f"{final_p:.4f}    {final_d:+.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print(f"\nHypothesis: Anger-ED should persist at jammed door and succeed")
    print(f"           where Standard QL gives up after 1-2 failed attempts.")

    # Door opening analysis (primary metric)
    if door_p < 0.05 and door_d > 0:
        print(f"\n[SUCCESS] Anger-ED opens door significantly more often")
        print(f"  Door open rate: {np.mean(anger_door):.1%} vs {np.mean(standard_door):.1%}")
        print(f"  Effect size: d={door_d:.3f} ", end="")
        if abs(door_d) < 0.5:
            print("(small)")
        elif abs(door_d) < 0.8:
            print("(medium)")
        else:
            print("(large)")
    elif door_p < 0.05 and door_d < 0:
        print(f"\n[UNEXPECTED] Standard QL opens door more often")
    else:
        print(f"\n[INCONCLUSIVE] No significant difference in door opening (p={door_p:.4f})")

    # Persistence analysis
    if pushes_p < 0.05 and pushes_d > 0:
        print(f"\n[SUCCESS] Anger-ED shows more persistence (more consecutive pushes)")
        print(f"  Max pushes: {np.mean(anger_pushes):.2f} vs {np.mean(standard_pushes):.2f}")

    # Success rate analysis
    if success_p < 0.05 and success_d > 0:
        print(f"\n[SUCCESS] Anger-ED has higher overall success rate")
        improvement = (np.mean(anger_success) - np.mean(standard_success)) * 100
        print(f"  Improvement: +{improvement:.1f} percentage points")

    # Behavioral insight
    print("\n" + "-" * 40)
    print("Behavioral Analysis:")
    print(f"  Standard QL avg max pushes: {np.mean(standard_pushes):.2f} (needs {pushes_needed})")
    print(f"  Anger-ED avg max pushes: {np.mean(anger_pushes):.2f}")

    if np.mean(standard_pushes) < pushes_needed - 1:
        print("  -> Standard QL gives up before door can open")
    if np.mean(anger_pushes) >= pushes_needed - 0.5:
        print("  -> Anger-ED persists long enough to open door")

    # Return results
    return {
        'standard': {
            'success_rate': standard_success,
            'door_open_rate': standard_door,
            'max_pushes': standard_pushes,
            'first_success': standard_first,
            'final_success': standard_final,
        },
        'anger': {
            'success_rate': anger_success,
            'door_open_rate': anger_door,
            'max_pushes': anger_pushes,
            'first_success': anger_first,
            'final_success': anger_final,
        },
        'stats': {
            'success': {'t': success_t, 'p': success_p, 'd': success_d},
            'door': {'t': door_t, 'p': door_p, 'd': door_d},
            'pushes': {'t': pushes_t, 'p': pushes_p, 'd': pushes_d},
            'first': {'t': first_t, 'p': first_p, 'd': first_d},
            'final': {'t': final_t, 'p': final_p, 'd': final_d},
        }
    }


if __name__ == "__main__":
    results = run_experiment(n_seeds=50, n_episodes=300, pushes_needed=4)

    print("\n" + "=" * 70)
    print("EXPERIMENT 20 COMPLETE")
    print("=" * 70)
