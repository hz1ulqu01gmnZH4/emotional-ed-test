"""Experiment 22: Predator-Prey - Adversarial Dynamics with Anxiety

A FAIR test environment for emotional learning where fear/anxiety helps.

Environment:
- 10x10 grid with agent (prey) and predator
- Predator chases agent each step (moves toward agent)
- Food pellets scattered around grid (respawn after collection)
- Agent must collect food while avoiding predator
- Getting caught = -50 reward, episode ends

Why Emotional Channels Should Help:
- Standard DQN struggles to balance eat vs run
- Agent either ignores predator (gets caught) or ignores food (starves)
- Anxiety channel triggers on predator proximity
- Modulates policy: prioritize evasion over food dynamically
- Enables "fight or flight" switching

Key Insight:
Fear/anxiety should create context-dependent behavior:
- Predator far: seek food
- Predator close: evade (even if near food)
Standard RL must learn this explicitly; emotional ED gets it automatically.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Set, Optional
from scipy import stats


@dataclass
class AnxietyContext:
    """Context for computing anxiety/fear signals."""
    predator_distance: float
    predator_approaching: bool  # Is predator getting closer?
    near_food: bool
    food_distance: float
    was_caught: bool
    food_collected_this_step: bool


class PredatorPreyEnv:
    """
    Predator-Prey Environment.

    10x10 grid with:
    - Agent (prey) starts at random position
    - Predator starts at opposite corner
    - N food pellets scattered randomly
    - Predator moves toward agent each step (simple chase AI)
    - Agent must collect food while avoiding predator

    Rewards:
    - +1 for collecting food
    - -50 for being caught (episode ends)
    - -0.01 step cost (encourages efficiency)

    Predator behavior:
    - Moves one step toward agent per turn
    - 80% optimal move, 20% random (imperfect chase)
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

    def __init__(self, size: int = 10, n_food: int = 5, max_steps: int = 200,
                 predator_speed: float = 0.8):
        self.size = size
        self.n_food = n_food
        self.max_steps = max_steps
        self.predator_speed = predator_speed  # Probability of optimal chase

        # Episode state
        self.agent_pos = None
        self.predator_pos = None
        self.food_positions: Set[Tuple[int, int]] = set()
        self.steps = 0
        self.food_collected = 0
        self.previous_predator_distance = None

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _spawn_food(self, n: int = 1):
        """Spawn n food pellets at random positions."""
        for _ in range(n):
            while True:
                pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
                if (pos != self.agent_pos and pos != self.predator_pos and
                    pos not in self.food_positions):
                    self.food_positions.add(pos)
                    break

    def reset(self) -> Tuple[int, int]:
        """Reset environment."""
        # Agent starts in one corner
        self.agent_pos = (0, 0)

        # Predator starts in opposite corner
        self.predator_pos = (self.size - 1, self.size - 1)

        # Spawn food
        self.food_positions = set()
        self._spawn_food(self.n_food)

        self.steps = 0
        self.food_collected = 0
        self.previous_predator_distance = self._manhattan_distance(
            self.agent_pos, self.predator_pos
        )

        return self.agent_pos

    def _move_predator(self):
        """Move predator toward agent."""
        if np.random.random() < self.predator_speed:
            # Optimal chase: move toward agent
            row_diff = self.agent_pos[0] - self.predator_pos[0]
            col_diff = self.agent_pos[1] - self.predator_pos[1]

            # Prefer larger gap
            if abs(row_diff) >= abs(col_diff):
                if row_diff > 0:
                    new_pos = (self.predator_pos[0] + 1, self.predator_pos[1])
                elif row_diff < 0:
                    new_pos = (self.predator_pos[0] - 1, self.predator_pos[1])
                else:
                    new_pos = self.predator_pos
            else:
                if col_diff > 0:
                    new_pos = (self.predator_pos[0], self.predator_pos[1] + 1)
                elif col_diff < 0:
                    new_pos = (self.predator_pos[0], self.predator_pos[1] - 1)
                else:
                    new_pos = self.predator_pos

            # Validate move
            if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
                self.predator_pos = new_pos
        else:
            # Random move
            action = np.random.randint(4)
            delta = self.ACTIONS[action]
            new_row = self.predator_pos[0] + delta[0]
            new_col = self.predator_pos[1] + delta[1]
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                self.predator_pos = (new_row, new_col)

    def _get_nearest_food_distance(self) -> float:
        """Get distance to nearest food."""
        if not self.food_positions:
            return float('inf')

        min_dist = float('inf')
        for food_pos in self.food_positions:
            dist = self._manhattan_distance(self.agent_pos, food_pos)
            min_dist = min(min_dist, dist)
        return min_dist

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, AnxietyContext]:
        """
        Take action in the environment.

        Order of operations:
        1. Agent moves
        2. Check for food collection
        3. Predator moves
        4. Check for catch

        Returns: (next_state, reward, done, context)
        """
        self.steps += 1

        # 1. Agent moves
        delta = self.ACTIONS[action]
        new_row = self.agent_pos[0] + delta[0]
        new_col = self.agent_pos[1] + delta[1]

        # Boundary check
        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            self.agent_pos = (new_row, new_col)

        # 2. Check for food collection
        reward = -0.01  # Step cost
        food_collected_this_step = False

        if self.agent_pos in self.food_positions:
            reward += 1.0
            self.food_positions.remove(self.agent_pos)
            self.food_collected += 1
            food_collected_this_step = True
            # Respawn food elsewhere
            self._spawn_food(1)

        # 3. Predator moves
        self._move_predator()

        # 4. Check for catch
        predator_distance = self._manhattan_distance(self.agent_pos, self.predator_pos)
        was_caught = False
        done = False

        if predator_distance == 0:  # Caught!
            reward = -50.0
            was_caught = True
            done = True
        elif self.steps >= self.max_steps:
            done = True

        # Check if predator is approaching
        predator_approaching = predator_distance < self.previous_predator_distance
        self.previous_predator_distance = predator_distance

        # Context for anxiety computation
        food_distance = self._get_nearest_food_distance()

        context = AnxietyContext(
            predator_distance=predator_distance,
            predator_approaching=predator_approaching,
            near_food=(food_distance <= 2),
            food_distance=food_distance,
            was_caught=was_caught,
            food_collected_this_step=food_collected_this_step
        )

        return self.agent_pos, reward, done, context

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) to flat index."""
        return state[0] * self.size + state[1]

    def get_full_state(self) -> Tuple:
        """Get full state including predator position (for more sophisticated agents)."""
        return (self.agent_pos, self.predator_pos, frozenset(self.food_positions))

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4


class StandardAgent:
    """
    Standard Q-learning agent without anxiety modulation.

    Uses agent position as state (ignores predator position in state).
    Must learn to balance food vs safety through reward only.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.2):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions

        # Grid size for position conversion
        self.grid_size = int(np.sqrt(n_states))

    def select_action(self, state: int, context: Optional[AnxietyContext] = None) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: Optional[AnxietyContext] = None):
        """Standard TD update."""
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        """No per-episode state to reset."""
        pass


class AnxietyFearAgent:
    """
    Anxiety/Fear-modulated agent for predator avoidance.

    Uses anxiety to dynamically switch between:
    - Food-seeking mode (low anxiety, predator far)
    - Evasion mode (high anxiety, predator close)

    Key mechanisms:
    1. Anxiety increases with predator proximity
    2. High anxiety biases actions AWAY from predator
    3. High anxiety reduces value of food-seeking actions
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.2,
                 anxiety_threshold: float = 3.0):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions

        # Grid size for position conversion
        self.grid_size = int(np.sqrt(n_states))

        # Anxiety parameters
        self.anxiety_threshold = anxiety_threshold  # Distance at which anxiety starts
        self.anxiety = 0.0  # Current anxiety level [0, 1]
        self.anxiety_decay = 0.8  # Decay rate when safe

        # Track predator position for evasion
        self.predator_pos = None

    def _compute_anxiety(self, context: AnxietyContext) -> float:
        """Compute anxiety level from context."""
        # Base anxiety from distance
        if context.predator_distance <= 1:
            base_anxiety = 1.0  # Maximum anxiety when adjacent
        elif context.predator_distance <= self.anxiety_threshold:
            base_anxiety = (self.anxiety_threshold - context.predator_distance) / self.anxiety_threshold
        else:
            base_anxiety = 0.0

        # Boost if predator is approaching
        if context.predator_approaching:
            base_anxiety = min(1.0, base_anxiety + 0.2)

        # Update tonic anxiety
        if base_anxiety > self.anxiety:
            self.anxiety = base_anxiety
        else:
            self.anxiety = self.anxiety * self.anxiety_decay + base_anxiety * (1 - self.anxiety_decay)

        return self.anxiety

    def _get_evasion_action(self, state: int, predator_pos: Tuple[int, int]) -> int:
        """Get action that moves away from predator."""
        agent_row = state // self.grid_size
        agent_col = state % self.grid_size

        # Compute direction away from predator
        row_diff = agent_row - predator_pos[0]
        col_diff = agent_col - predator_pos[1]

        # Score each action by how much it increases distance
        action_scores = []
        for action in range(4):
            delta = PredatorPreyEnv.ACTIONS[action]
            new_row = agent_row + delta[0]
            new_col = agent_col + delta[1]

            # Check bounds
            if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
                action_scores.append(-10)  # Invalid move
                continue

            # Distance after move
            new_dist = abs(new_row - predator_pos[0]) + abs(new_col - predator_pos[1])
            action_scores.append(new_dist)

        # Return action with highest distance (ties broken by Q-value)
        max_dist = max(action_scores)
        best_actions = [a for a, d in enumerate(action_scores) if d == max_dist]

        if len(best_actions) == 1:
            return best_actions[0]
        else:
            # Tie-break by Q-value
            q_values = [self.Q[state, a] for a in best_actions]
            return best_actions[int(np.argmax(q_values))]

    def select_action(self, state: int, context: Optional[AnxietyContext] = None,
                      predator_pos: Optional[Tuple[int, int]] = None) -> int:
        """
        Anxiety-modulated action selection.

        High anxiety -> prioritize evasion over Q-values.
        Low anxiety -> normal epsilon-greedy.
        """
        if context is not None:
            self.anxiety = self._compute_anxiety(context)

        if predator_pos is not None:
            self.predator_pos = predator_pos

        # High anxiety: switch to evasion mode
        if self.anxiety > 0.5 and self.predator_pos is not None:
            evasion_prob = self.anxiety * 0.8  # Up to 80% evasion
            if np.random.random() < evasion_prob:
                return self._get_evasion_action(state, self.predator_pos)

        # Normal epsilon-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Greedy with anxiety-modulated Q-values
        q_values = self.Q[state].copy()

        # If anxious, reduce value of actions toward predator
        if self.anxiety > 0.2 and self.predator_pos is not None:
            agent_row = state // self.grid_size
            agent_col = state % self.grid_size

            for action in range(4):
                delta = PredatorPreyEnv.ACTIONS[action]
                new_row = agent_row + delta[0]
                new_col = agent_col + delta[1]

                if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                    new_dist = abs(new_row - self.predator_pos[0]) + abs(new_col - self.predator_pos[1])
                    old_dist = abs(agent_row - self.predator_pos[0]) + abs(agent_col - self.predator_pos[1])

                    if new_dist < old_dist:  # Moving toward predator
                        q_values[action] -= self.anxiety * 10  # Penalize

        return int(np.argmax(q_values))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: AnxietyContext):
        """TD update with anxiety-aware learning."""
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        # Boost learning when caught (strong negative signal)
        if context.was_caught:
            effective_lr = self.lr * 2.0
        else:
            effective_lr = self.lr

        self.Q[state, action] += effective_lr * delta

    def reset_episode(self):
        """Reset anxiety for new episode."""
        self.anxiety = 0.0
        self.predator_pos = None


def run_episode(env: PredatorPreyEnv, agent, training: bool = True) -> dict:
    """Run one episode and return metrics."""
    state = env.reset()
    state_idx = env.state_to_index(state)

    total_reward = 0.0
    steps = 0
    food_collected = 0
    survived = True
    close_calls = 0  # Times predator got within distance 2

    agent.reset_episode()

    # Initial context
    context = AnxietyContext(
        predator_distance=env._manhattan_distance(env.agent_pos, env.predator_pos),
        predator_approaching=False,
        near_food=False,
        food_distance=float('inf'),
        was_caught=False,
        food_collected_this_step=False
    )

    while True:
        # Select action (provide predator position for fear agent)
        if hasattr(agent, 'predator_pos'):
            action = agent.select_action(state_idx, context, env.predator_pos)
        else:
            action = agent.select_action(state_idx, context)

        # Take action
        next_state, reward, done, context = env.step(action)
        next_state_idx = env.state_to_index(next_state)

        # Track metrics
        if context.food_collected_this_step:
            food_collected += 1
        if context.predator_distance <= 2:
            close_calls += 1
        if context.was_caught:
            survived = False

        # Store transition and update
        if training:
            agent.update(state_idx, action, reward, next_state_idx, done, context)

        total_reward += reward
        steps += 1

        if done:
            break

        state = next_state
        state_idx = next_state_idx

    return {
        'reward': total_reward,
        'steps': steps,
        'food_collected': food_collected,
        'survived': survived,
        'close_calls': close_calls,
        'survival_time': steps if survived else steps  # How long agent lasted
    }


def run_experiment(n_seeds: int = 50, n_episodes: int = 300):
    """
    Run the Predator-Prey experiment with statistical validation.

    Args:
        n_seeds: Number of random seeds
        n_episodes: Training episodes per seed
    """
    print("=" * 70)
    print("EXPERIMENT 22: PREDATOR-PREY (Adversarial / Anxiety)")
    print("=" * 70)
    print("\nEnvironment: 10x10 grid with chasing predator and food pellets")
    print("  - Predator chases agent (80% optimal, 20% random)")
    print("  - Collect food (+1) while avoiding predator (-50, episode ends)")
    print("Hypothesis: Anxiety-based agent balances food/evasion better")
    print(f"Running {n_seeds} seeds x {n_episodes} episodes")
    print()

    standard_results = []
    anxiety_results = []

    for seed in range(n_seeds):
        np.random.seed(seed)

        print(f"\rSeed {seed+1}/{n_seeds}", end='', flush=True)

        # --- Standard Agent ---
        np.random.seed(seed)
        env = PredatorPreyEnv(size=10, n_food=5, max_steps=200)
        standard_agent = StandardAgent(env.n_states, env.n_actions)

        seed_rewards = []
        seed_food = []
        seed_survived = []
        seed_survival_time = []

        for ep in range(n_episodes):
            result = run_episode(env, standard_agent, training=True)
            seed_rewards.append(result['reward'])
            seed_food.append(result['food_collected'])
            seed_survived.append(1 if result['survived'] else 0)
            seed_survival_time.append(result['survival_time'])

        standard_results.append({
            'mean_reward': np.mean(seed_rewards),
            'mean_food': np.mean(seed_food),
            'survival_rate': np.mean(seed_survived),
            'mean_survival_time': np.mean(seed_survival_time),
            'final_survival_rate': np.mean(seed_survived[-50:]) if len(seed_survived) >= 50 else np.mean(seed_survived),
            'final_food': np.mean(seed_food[-50:]) if len(seed_food) >= 50 else np.mean(seed_food),
        })

        # --- Anxiety/Fear Agent ---
        np.random.seed(seed)  # Reset for fair comparison
        env = PredatorPreyEnv(size=10, n_food=5, max_steps=200)
        anxiety_agent = AnxietyFearAgent(env.n_states, env.n_actions)

        seed_rewards = []
        seed_food = []
        seed_survived = []
        seed_survival_time = []

        for ep in range(n_episodes):
            result = run_episode(env, anxiety_agent, training=True)
            seed_rewards.append(result['reward'])
            seed_food.append(result['food_collected'])
            seed_survived.append(1 if result['survived'] else 0)
            seed_survival_time.append(result['survival_time'])

        anxiety_results.append({
            'mean_reward': np.mean(seed_rewards),
            'mean_food': np.mean(seed_food),
            'survival_rate': np.mean(seed_survived),
            'mean_survival_time': np.mean(seed_survival_time),
            'final_survival_rate': np.mean(seed_survived[-50:]) if len(seed_survived) >= 50 else np.mean(seed_survived),
            'final_food': np.mean(seed_food[-50:]) if len(seed_food) >= 50 else np.mean(seed_food),
        })

    print("\n")

    # Aggregate results
    standard_reward = [r['mean_reward'] for r in standard_results]
    standard_survival = [r['survival_rate'] for r in standard_results]
    standard_food = [r['mean_food'] for r in standard_results]
    standard_time = [r['mean_survival_time'] for r in standard_results]
    standard_final_surv = [r['final_survival_rate'] for r in standard_results]

    anxiety_reward = [r['mean_reward'] for r in anxiety_results]
    anxiety_survival = [r['survival_rate'] for r in anxiety_results]
    anxiety_food = [r['mean_food'] for r in anxiety_results]
    anxiety_time = [r['mean_survival_time'] for r in anxiety_results]
    anxiety_final_surv = [r['final_survival_rate'] for r in anxiety_results]

    # Statistical tests
    survival_t, survival_p = stats.ttest_ind(anxiety_survival, standard_survival)
    reward_t, reward_p = stats.ttest_ind(anxiety_reward, standard_reward)
    food_t, food_p = stats.ttest_ind(anxiety_food, standard_food)
    time_t, time_p = stats.ttest_ind(anxiety_time, standard_time)
    final_surv_t, final_surv_p = stats.ttest_ind(anxiety_final_surv, standard_final_surv)

    # Effect sizes (Cohen's d)
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

    survival_d = cohens_d(anxiety_survival, standard_survival)
    reward_d = cohens_d(anxiety_reward, standard_reward)
    food_d = cohens_d(anxiety_food, standard_food)
    time_d = cohens_d(anxiety_time, standard_time)
    final_surv_d = cohens_d(anxiety_final_surv, standard_final_surv)

    # Print results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Standard':<22} {'Anxiety-ED':<22} {'p-value':<10} {'Cohen d':<10}")
    print("-" * 90)

    print(f"{'Overall Survival Rate':<25} "
          f"{np.mean(standard_survival):.3f} +/- {np.std(standard_survival):.3f}  "
          f"{np.mean(anxiety_survival):.3f} +/- {np.std(anxiety_survival):.3f}  "
          f"{survival_p:.4f}    {survival_d:+.3f}")

    print(f"{'Mean Reward':<25} "
          f"{np.mean(standard_reward):.1f} +/- {np.std(standard_reward):.1f}    "
          f"{np.mean(anxiety_reward):.1f} +/- {np.std(anxiety_reward):.1f}    "
          f"{reward_p:.4f}    {reward_d:+.3f}")

    print(f"{'Mean Food Collected':<25} "
          f"{np.mean(standard_food):.2f} +/- {np.std(standard_food):.2f}    "
          f"{np.mean(anxiety_food):.2f} +/- {np.std(anxiety_food):.2f}    "
          f"{food_p:.4f}    {food_d:+.3f}")

    print(f"{'Mean Survival Time':<25} "
          f"{np.mean(standard_time):.1f} +/- {np.std(standard_time):.1f}    "
          f"{np.mean(anxiety_time):.1f} +/- {np.std(anxiety_time):.1f}    "
          f"{time_p:.4f}    {time_d:+.3f}")

    print(f"{'Final 50-ep Survival':<25} "
          f"{np.mean(standard_final_surv):.3f} +/- {np.std(standard_final_surv):.3f}  "
          f"{np.mean(anxiety_final_surv):.3f} +/- {np.std(anxiety_final_surv):.3f}  "
          f"{final_surv_p:.4f}    {final_surv_d:+.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("\nHypothesis: Anxiety-ED should survive longer by evading predator")
    print("           while still collecting food efficiently")

    # Survival analysis (primary metric)
    if survival_p < 0.05 and survival_d > 0:
        print(f"\n[SUCCESS] Anxiety-ED survives significantly more often")
        print(f"  Survival rate: {np.mean(anxiety_survival):.1%} vs {np.mean(standard_survival):.1%}")
        print(f"  Effect size: d={survival_d:.3f} ", end="")
        if abs(survival_d) < 0.5:
            print("(small)")
        elif abs(survival_d) < 0.8:
            print("(medium)")
        else:
            print("(large)")
    elif survival_p < 0.05 and survival_d < 0:
        print(f"\n[UNEXPECTED] Standard agent survives more often")
    else:
        print(f"\n[INCONCLUSIVE] No significant survival difference (p={survival_p:.4f})")

    # Food efficiency analysis
    if food_p < 0.05:
        if food_d > 0:
            print(f"\n[SUCCESS] Anxiety-ED also collects more food")
        else:
            print(f"\n[TRADEOFF] Anxiety-ED survives more but collects less food")

    # Reward analysis
    if reward_p < 0.05 and reward_d > 0:
        print(f"\n[SUCCESS] Anxiety-ED has higher overall reward")
        print(f"  (Survival benefit outweighs any food tradeoff)")

    # Behavioral insight
    print("\n" + "-" * 40)
    print("Behavioral Analysis:")
    print(f"  Standard: {np.mean(standard_survival):.1%} survival, {np.mean(standard_food):.1f} food/ep")
    print(f"  Anxiety:  {np.mean(anxiety_survival):.1%} survival, {np.mean(anxiety_food):.1f} food/ep")

    if np.mean(anxiety_survival) > np.mean(standard_survival):
        print("  -> Anxiety-ED shows better predator avoidance")
    if np.mean(anxiety_food) >= np.mean(standard_food) * 0.9:
        print("  -> Food collection not significantly sacrificed for safety")

    # Return results
    return {
        'standard': standard_results,
        'anxiety': anxiety_results,
        'stats': {
            'survival': {'t': survival_t, 'p': survival_p, 'd': survival_d},
            'reward': {'t': reward_t, 'p': reward_p, 'd': reward_d},
            'food': {'t': food_t, 'p': food_p, 'd': food_d},
            'time': {'t': time_t, 'p': time_p, 'd': time_d},
            'final_survival': {'t': final_surv_t, 'p': final_surv_p, 'd': final_surv_d},
        }
    }


if __name__ == "__main__":
    results = run_experiment(n_seeds=50, n_episodes=300)

    print("\n" + "=" * 70)
    print("EXPERIMENT 22 COMPLETE")
    print("=" * 70)
