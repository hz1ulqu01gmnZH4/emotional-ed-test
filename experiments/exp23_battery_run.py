"""Experiment 23: Battery Run - Long-Horizon Fuel Management

A FAIR test environment for emotional learning where patience/delayed gratification helps.

Environment:
- 12x12 grid with agent navigating to waypoints
- Agent has limited battery (100 units), each move costs 1 unit
- Charging stations scattered around grid (restore full battery)
- Goal: visit N waypoints in sequence
- Empty battery = crash (episode ends, -50 reward)

Why Emotional Channels Should Help:
- Standard DQN gets greedy for next waypoint
- Ignores battery level, crashes before completing mission
- Patience/Tonic Anxiety should modulate based on battery level:
  - High battery: pursue waypoints aggressively
  - Low battery: prioritize charging (even if waypoint is close)
- Enables strategic abort-to-refuel decisions

Key Insight:
Patience/delayed gratification enables long-horizon planning.
Standard RL discounts future too heavily to plan battery management.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Set, Optional
from scipy import stats


@dataclass
class BatteryContext:
    """Context for computing patience/anxiety signals."""
    battery_level: float  # 0.0 to 1.0
    battery_units: int  # Actual units remaining
    distance_to_nearest_charger: float
    distance_to_next_waypoint: float
    waypoints_remaining: int
    can_reach_charger: bool  # Is battery enough to reach nearest charger?
    charger_on_path: bool  # Is there a charger between agent and waypoint?


class BatteryRunEnv:
    """
    Battery Run Environment.

    12x12 grid with:
    - Agent starts at (0, 0) with full battery (100 units)
    - N waypoints to visit in sequence
    - M charging stations scattered around
    - Each move costs 1 battery unit
    - Visiting charger restores full battery

    Rewards:
    - +10 for reaching each waypoint
    - -50 for running out of battery (crash)
    - -0.01 step cost
    - +5 bonus for completing all waypoints

    Battery mechanics:
    - Starts at 100
    - Each step costs 1
    - Charger restores to 100
    - At 0, episode ends (crash)
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

    def __init__(self, size: int = 12, n_waypoints: int = 5, n_chargers: int = 4,
                 max_battery: int = 100, max_steps: int = 500):
        self.size = size
        self.n_waypoints = n_waypoints
        self.n_chargers = n_chargers
        self.max_battery = max_battery
        self.max_steps = max_steps

        self.start_pos = (0, 0)

        # Will be set on reset
        self.waypoints: List[Tuple[int, int]] = []
        self.chargers: Set[Tuple[int, int]] = set()
        self.current_waypoint_idx = 0

        # Episode state
        self.agent_pos = None
        self.battery = 0
        self.steps = 0
        self.waypoints_collected = 0

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _generate_positions(self):
        """Generate waypoint and charger positions."""
        occupied = {self.start_pos}

        # Generate waypoints (spread around grid)
        self.waypoints = []
        for i in range(self.n_waypoints):
            while True:
                # Bias waypoints toward edges and far from start
                row = np.random.randint(2, self.size)
                col = np.random.randint(2, self.size)
                pos = (row, col)
                if pos not in occupied:
                    self.waypoints.append(pos)
                    occupied.add(pos)
                    break

        # Generate chargers (spread evenly)
        self.chargers = set()
        regions = [
            (0, self.size//2, 0, self.size//2),  # Top-left
            (0, self.size//2, self.size//2, self.size),  # Top-right
            (self.size//2, self.size, 0, self.size//2),  # Bottom-left
            (self.size//2, self.size, self.size//2, self.size),  # Bottom-right
        ]

        for i in range(self.n_chargers):
            region = regions[i % len(regions)]
            while True:
                row = np.random.randint(region[0], region[1])
                col = np.random.randint(region[2], region[3])
                pos = (row, col)
                if pos not in occupied:
                    self.chargers.add(pos)
                    occupied.add(pos)
                    break

    def reset(self) -> Tuple[int, int]:
        """Reset environment."""
        self.agent_pos = self.start_pos
        self.battery = self.max_battery
        self.steps = 0
        self.waypoints_collected = 0
        self.current_waypoint_idx = 0

        self._generate_positions()

        return self.agent_pos

    def _get_nearest_charger_distance(self) -> float:
        """Get distance to nearest charger."""
        if not self.chargers:
            return float('inf')

        min_dist = float('inf')
        for charger_pos in self.chargers:
            dist = self._manhattan_distance(self.agent_pos, charger_pos)
            min_dist = min(min_dist, dist)
        return min_dist

    def _get_current_waypoint_distance(self) -> float:
        """Get distance to current target waypoint."""
        if self.current_waypoint_idx >= len(self.waypoints):
            return 0.0
        return self._manhattan_distance(
            self.agent_pos, self.waypoints[self.current_waypoint_idx]
        )

    def _is_charger_on_path(self) -> bool:
        """Check if any charger is roughly on the path to waypoint."""
        if self.current_waypoint_idx >= len(self.waypoints):
            return False

        waypoint = self.waypoints[self.current_waypoint_idx]
        direct_dist = self._manhattan_distance(self.agent_pos, waypoint)

        for charger in self.chargers:
            dist_to_charger = self._manhattan_distance(self.agent_pos, charger)
            charger_to_waypoint = self._manhattan_distance(charger, waypoint)
            # If going via charger adds <= 2 steps, consider it "on path"
            if dist_to_charger + charger_to_waypoint <= direct_dist + 2:
                return True
        return False

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, BatteryContext]:
        """
        Take action in the environment.

        Returns: (next_state, reward, done, context)
        """
        self.steps += 1

        # Move
        delta = self.ACTIONS[action]
        new_row = self.agent_pos[0] + delta[0]
        new_col = self.agent_pos[1] + delta[1]

        # Boundary check
        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            self.agent_pos = (new_row, new_col)

        # Battery drain
        self.battery -= 1

        # Check for charger
        if self.agent_pos in self.chargers:
            self.battery = self.max_battery

        # Check for waypoint
        reward = -0.01  # Step cost
        done = False

        if (self.current_waypoint_idx < len(self.waypoints) and
            self.agent_pos == self.waypoints[self.current_waypoint_idx]):
            reward += 10.0  # Waypoint reached!
            self.waypoints_collected += 1
            self.current_waypoint_idx += 1

            # All waypoints collected?
            if self.current_waypoint_idx >= len(self.waypoints):
                reward += 5.0  # Completion bonus
                done = True

        # Check for battery death
        if self.battery <= 0:
            reward = -50.0  # Crash!
            done = True

        # Check for max steps
        if self.steps >= self.max_steps:
            done = True

        # Context for patience computation
        charger_dist = self._get_nearest_charger_distance()

        context = BatteryContext(
            battery_level=self.battery / self.max_battery,
            battery_units=self.battery,
            distance_to_nearest_charger=charger_dist,
            distance_to_next_waypoint=self._get_current_waypoint_distance(),
            waypoints_remaining=len(self.waypoints) - self.current_waypoint_idx,
            can_reach_charger=(self.battery >= charger_dist + 5),  # +5 safety margin
            charger_on_path=self._is_charger_on_path()
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


class ImpulsiveAgent:
    """
    Standard Q-learning agent without patience modulation.

    Greedy for immediate rewards (waypoints).
    Doesn't adequately plan for battery management.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.95, epsilon: float = 0.2):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma  # Moderate discount factor
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions

        # Grid size for position conversion
        self.grid_size = int(np.sqrt(n_states))

    def select_action(self, state: int, context: Optional[BatteryContext] = None) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: Optional[BatteryContext] = None):
        """Standard TD update."""
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        """No per-episode state to reset."""
        pass


class PatienceAgent:
    """
    Patience-modulated agent for battery management.

    Uses battery-based tonic state to modulate behavior:
    - High battery: pursue waypoints aggressively (low patience needed)
    - Low battery: prioritize charging (high patience/caution)

    Key mechanisms:
    1. Tonic patience increases as battery drops
    2. High patience biases actions toward chargers
    3. High patience reduces impulsive waypoint-chasing
    4. Effectively implements "abort-to-refuel" behavior
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.95, epsilon: float = 0.2,
                 low_battery_threshold: float = 0.3):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions

        # Grid size for position conversion
        self.grid_size = int(np.sqrt(n_states))

        # Patience parameters
        self.low_battery_threshold = low_battery_threshold
        self.patience = 0.0  # Current patience level [0, 1]

        # Track charger and waypoint positions for action biasing
        self.charger_positions: Set[Tuple[int, int]] = set()
        self.current_waypoint: Optional[Tuple[int, int]] = None

    def _compute_patience(self, context: BatteryContext) -> float:
        """
        Compute patience level from battery context.

        Low battery -> high patience (prioritize safety/charging).
        """
        battery_level = context.battery_level

        if battery_level <= 0.15:
            # Critical: maximum patience (must charge!)
            patience = 1.0
        elif battery_level <= self.low_battery_threshold:
            # Low: increasing patience
            patience = (self.low_battery_threshold - battery_level) / self.low_battery_threshold
            patience = min(1.0, patience * 2)
        elif battery_level <= 0.5:
            # Medium: some caution
            patience = 0.2
        else:
            # High: low patience (pursue goals)
            patience = 0.0

        # Boost patience if we can't safely reach charger
        if not context.can_reach_charger:
            patience = max(patience, 0.8)

        self.patience = patience
        return patience

    def _get_direction_to_target(self, state: int, target: Tuple[int, int]) -> List[int]:
        """Get actions that move toward target, sorted by preference."""
        agent_row = state // self.grid_size
        agent_col = state % self.grid_size

        row_diff = target[0] - agent_row
        col_diff = target[1] - agent_col

        actions_with_score = []
        for action in range(4):
            delta = BatteryRunEnv.ACTIONS[action]
            new_row = agent_row + delta[0]
            new_col = agent_col + delta[1]

            # Check bounds
            if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
                continue

            # Distance after move
            new_dist = abs(new_row - target[0]) + abs(new_col - target[1])
            old_dist = abs(agent_row - target[0]) + abs(agent_col - target[1])

            if new_dist < old_dist:
                actions_with_score.append((action, old_dist - new_dist))

        # Sort by improvement
        actions_with_score.sort(key=lambda x: -x[1])
        return [a for a, _ in actions_with_score]

    def _get_nearest_charger(self, state: int) -> Optional[Tuple[int, int]]:
        """Get nearest charger position."""
        if not self.charger_positions:
            return None

        agent_row = state // self.grid_size
        agent_col = state % self.grid_size

        min_dist = float('inf')
        nearest = None
        for charger in self.charger_positions:
            dist = abs(charger[0] - agent_row) + abs(charger[1] - agent_col)
            if dist < min_dist:
                min_dist = dist
                nearest = charger
        return nearest

    def select_action(self, state: int, context: Optional[BatteryContext] = None) -> int:
        """
        Patience-modulated action selection.

        High patience -> move toward charger.
        Low patience -> pursue waypoint.
        """
        if context is not None:
            self._compute_patience(context)

        # If patience is high, prioritize moving toward charger
        if self.patience > 0.5:
            nearest_charger = self._get_nearest_charger(state)
            if nearest_charger is not None:
                charger_actions = self._get_direction_to_target(state, nearest_charger)
                if charger_actions:
                    # High probability of moving toward charger
                    if np.random.random() < self.patience:
                        return charger_actions[0]

        # Normal epsilon-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Greedy selection with patience-modulated Q-values
        q_values = self.Q[state].copy()

        # If patient, boost value of charger-directed actions
        if self.patience > 0.2:
            nearest_charger = self._get_nearest_charger(state)
            if nearest_charger is not None:
                charger_actions = self._get_direction_to_target(state, nearest_charger)
                for i, action in enumerate(charger_actions):
                    q_values[action] += self.patience * 10 * (1 - i * 0.3)

        return int(np.argmax(q_values))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: BatteryContext):
        """TD update with patience-aware learning."""
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        # Boost learning from battery crashes
        if reward < -10:  # Crash
            effective_lr = self.lr * 2.0
        else:
            effective_lr = self.lr

        self.Q[state, action] += effective_lr * delta

    def observe_environment(self, chargers: Set[Tuple[int, int]],
                           current_waypoint: Optional[Tuple[int, int]]):
        """Observe environment layout for action biasing."""
        self.charger_positions = chargers
        self.current_waypoint = current_waypoint

    def reset_episode(self):
        """Reset patience for new episode."""
        self.patience = 0.0


def run_episode(env: BatteryRunEnv, agent, training: bool = True) -> dict:
    """Run one episode and return metrics."""
    state = env.reset()
    state_idx = env.state_to_index(state)

    total_reward = 0.0
    steps = 0
    waypoints_collected = 0
    times_charged = 0
    crashed = False
    completed_all = False

    # Give patience agent environment info
    if hasattr(agent, 'observe_environment'):
        current_wp = env.waypoints[0] if env.waypoints else None
        agent.observe_environment(env.chargers, current_wp)

    agent.reset_episode()

    # Initial context
    context = BatteryContext(
        battery_level=1.0,
        battery_units=env.battery,
        distance_to_nearest_charger=env._get_nearest_charger_distance(),
        distance_to_next_waypoint=env._get_current_waypoint_distance(),
        waypoints_remaining=len(env.waypoints),
        can_reach_charger=True,
        charger_on_path=env._is_charger_on_path()
    )

    prev_battery = env.battery

    while True:
        # Select action
        action = agent.select_action(state_idx, context)

        # Take action
        next_state, reward, done, context = env.step(action)
        next_state_idx = env.state_to_index(next_state)

        # Track charging
        if env.battery > prev_battery:
            times_charged += 1
        prev_battery = env.battery

        # Track waypoints
        if env.waypoints_collected > waypoints_collected:
            waypoints_collected = env.waypoints_collected
            # Update patience agent's waypoint target
            if hasattr(agent, 'observe_environment'):
                if env.current_waypoint_idx < len(env.waypoints):
                    current_wp = env.waypoints[env.current_waypoint_idx]
                else:
                    current_wp = None
                agent.observe_environment(env.chargers, current_wp)

        # Check outcomes
        if reward < -10:  # Crash
            crashed = True
        if env.current_waypoint_idx >= len(env.waypoints):
            completed_all = True

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
        'waypoints_collected': waypoints_collected,
        'times_charged': times_charged,
        'crashed': crashed,
        'completed_all': completed_all,
        'completion_rate': waypoints_collected / env.n_waypoints
    }


def run_experiment(n_seeds: int = 50, n_episodes: int = 400):
    """
    Run the Battery Run experiment with statistical validation.

    Args:
        n_seeds: Number of random seeds
        n_episodes: Training episodes per seed
    """
    print("=" * 70)
    print("EXPERIMENT 23: BATTERY RUN (Long-Horizon / Patience)")
    print("=" * 70)
    print("\nEnvironment: 12x12 grid with waypoints and charging stations")
    print("  - Battery starts at 100, each move costs 1")
    print("  - Chargers restore full battery")
    print("  - Collect all 5 waypoints in sequence")
    print("  - Running out of battery = crash (-50)")
    print("Hypothesis: Patience-based agent manages battery better, completes more")
    print(f"Running {n_seeds} seeds x {n_episodes} episodes")
    print()

    impulsive_results = []
    patience_results = []

    for seed in range(n_seeds):
        np.random.seed(seed)

        print(f"\rSeed {seed+1}/{n_seeds}", end='', flush=True)

        # --- Impulsive Agent ---
        np.random.seed(seed)
        env = BatteryRunEnv(size=12, n_waypoints=5, n_chargers=4, max_battery=100)
        impulsive_agent = ImpulsiveAgent(env.n_states, env.n_actions)

        seed_rewards = []
        seed_waypoints = []
        seed_crashed = []
        seed_completed = []
        seed_charged = []

        for ep in range(n_episodes):
            result = run_episode(env, impulsive_agent, training=True)
            seed_rewards.append(result['reward'])
            seed_waypoints.append(result['waypoints_collected'])
            seed_crashed.append(1 if result['crashed'] else 0)
            seed_completed.append(1 if result['completed_all'] else 0)
            seed_charged.append(result['times_charged'])

        impulsive_results.append({
            'mean_reward': np.mean(seed_rewards),
            'mean_waypoints': np.mean(seed_waypoints),
            'crash_rate': np.mean(seed_crashed),
            'completion_rate': np.mean(seed_completed),
            'mean_charges': np.mean(seed_charged),
            'final_completion': np.mean(seed_completed[-50:]) if len(seed_completed) >= 50 else np.mean(seed_completed),
            'final_crash': np.mean(seed_crashed[-50:]) if len(seed_crashed) >= 50 else np.mean(seed_crashed),
        })

        # --- Patience Agent ---
        np.random.seed(seed)  # Reset for fair comparison
        env = BatteryRunEnv(size=12, n_waypoints=5, n_chargers=4, max_battery=100)
        patience_agent = PatienceAgent(env.n_states, env.n_actions)

        seed_rewards = []
        seed_waypoints = []
        seed_crashed = []
        seed_completed = []
        seed_charged = []

        for ep in range(n_episodes):
            result = run_episode(env, patience_agent, training=True)
            seed_rewards.append(result['reward'])
            seed_waypoints.append(result['waypoints_collected'])
            seed_crashed.append(1 if result['crashed'] else 0)
            seed_completed.append(1 if result['completed_all'] else 0)
            seed_charged.append(result['times_charged'])

        patience_results.append({
            'mean_reward': np.mean(seed_rewards),
            'mean_waypoints': np.mean(seed_waypoints),
            'crash_rate': np.mean(seed_crashed),
            'completion_rate': np.mean(seed_completed),
            'mean_charges': np.mean(seed_charged),
            'final_completion': np.mean(seed_completed[-50:]) if len(seed_completed) >= 50 else np.mean(seed_completed),
            'final_crash': np.mean(seed_crashed[-50:]) if len(seed_crashed) >= 50 else np.mean(seed_crashed),
        })

    print("\n")

    # Aggregate results
    impulsive_reward = [r['mean_reward'] for r in impulsive_results]
    impulsive_waypoints = [r['mean_waypoints'] for r in impulsive_results]
    impulsive_crash = [r['crash_rate'] for r in impulsive_results]
    impulsive_complete = [r['completion_rate'] for r in impulsive_results]
    impulsive_charges = [r['mean_charges'] for r in impulsive_results]
    impulsive_final_complete = [r['final_completion'] for r in impulsive_results]
    impulsive_final_crash = [r['final_crash'] for r in impulsive_results]

    patience_reward = [r['mean_reward'] for r in patience_results]
    patience_waypoints = [r['mean_waypoints'] for r in patience_results]
    patience_crash = [r['crash_rate'] for r in patience_results]
    patience_complete = [r['completion_rate'] for r in patience_results]
    patience_charges = [r['mean_charges'] for r in patience_results]
    patience_final_complete = [r['final_completion'] for r in patience_results]
    patience_final_crash = [r['final_crash'] for r in patience_results]

    # Statistical tests
    crash_t, crash_p = stats.ttest_ind(patience_crash, impulsive_crash)
    complete_t, complete_p = stats.ttest_ind(patience_complete, impulsive_complete)
    waypoint_t, waypoint_p = stats.ttest_ind(patience_waypoints, impulsive_waypoints)
    reward_t, reward_p = stats.ttest_ind(patience_reward, impulsive_reward)
    charges_t, charges_p = stats.ttest_ind(patience_charges, impulsive_charges)
    final_complete_t, final_complete_p = stats.ttest_ind(patience_final_complete, impulsive_final_complete)

    # Effect sizes (Cohen's d)
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

    crash_d = cohens_d(patience_crash, impulsive_crash)
    complete_d = cohens_d(patience_complete, impulsive_complete)
    waypoint_d = cohens_d(patience_waypoints, impulsive_waypoints)
    reward_d = cohens_d(patience_reward, impulsive_reward)
    charges_d = cohens_d(patience_charges, impulsive_charges)
    final_complete_d = cohens_d(patience_final_complete, impulsive_final_complete)

    # Print results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Impulsive':<22} {'Patience':<22} {'p-value':<10} {'Cohen d':<10}")
    print("-" * 90)

    print(f"{'Crash Rate':<25} "
          f"{np.mean(impulsive_crash):.3f} +/- {np.std(impulsive_crash):.3f}  "
          f"{np.mean(patience_crash):.3f} +/- {np.std(patience_crash):.3f}  "
          f"{crash_p:.4f}    {crash_d:+.3f}")

    print(f"{'Completion Rate':<25} "
          f"{np.mean(impulsive_complete):.3f} +/- {np.std(impulsive_complete):.3f}  "
          f"{np.mean(patience_complete):.3f} +/- {np.std(patience_complete):.3f}  "
          f"{complete_p:.4f}    {complete_d:+.3f}")

    print(f"{'Mean Waypoints':<25} "
          f"{np.mean(impulsive_waypoints):.2f} +/- {np.std(impulsive_waypoints):.2f}    "
          f"{np.mean(patience_waypoints):.2f} +/- {np.std(patience_waypoints):.2f}    "
          f"{waypoint_p:.4f}    {waypoint_d:+.3f}")

    print(f"{'Mean Reward':<25} "
          f"{np.mean(impulsive_reward):.1f} +/- {np.std(impulsive_reward):.1f}    "
          f"{np.mean(patience_reward):.1f} +/- {np.std(patience_reward):.1f}    "
          f"{reward_p:.4f}    {reward_d:+.3f}")

    print(f"{'Mean Charges/Episode':<25} "
          f"{np.mean(impulsive_charges):.2f} +/- {np.std(impulsive_charges):.2f}    "
          f"{np.mean(patience_charges):.2f} +/- {np.std(patience_charges):.2f}    "
          f"{charges_p:.4f}    {charges_d:+.3f}")

    print(f"{'Final 50-ep Completion':<25} "
          f"{np.mean(impulsive_final_complete):.3f} +/- {np.std(impulsive_final_complete):.3f}  "
          f"{np.mean(patience_final_complete):.3f} +/- {np.std(patience_final_complete):.3f}  "
          f"{final_complete_p:.4f}    {final_complete_d:+.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("\nHypothesis: Patience agent should crash less and complete more")
    print("           by prioritizing charging when battery is low")

    # Crash rate analysis (primary safety metric)
    if crash_p < 0.05 and crash_d < 0:
        print(f"\n[SUCCESS] Patience agent crashes significantly less often")
        print(f"  Crash rate: {np.mean(patience_crash):.1%} vs {np.mean(impulsive_crash):.1%}")
        print(f"  Effect size: d={abs(crash_d):.3f} ", end="")
        if abs(crash_d) < 0.5:
            print("(small)")
        elif abs(crash_d) < 0.8:
            print("(medium)")
        else:
            print("(large)")
    elif crash_p < 0.05 and crash_d > 0:
        print(f"\n[UNEXPECTED] Impulsive agent crashes less")
    else:
        print(f"\n[INCONCLUSIVE] No significant crash difference (p={crash_p:.4f})")

    # Completion analysis
    if complete_p < 0.05 and complete_d > 0:
        print(f"\n[SUCCESS] Patience agent completes missions more often")
        print(f"  Completion: {np.mean(patience_complete):.1%} vs {np.mean(impulsive_complete):.1%}")

    # Charging behavior analysis
    if charges_p < 0.05 and charges_d > 0:
        print(f"\n[BEHAVIOR] Patience agent charges more frequently")
        print(f"  This indicates proactive battery management")

    # Behavioral insight
    print("\n" + "-" * 40)
    print("Battery Management Analysis:")
    print(f"  Impulsive: {np.mean(impulsive_crash):.1%} crashes, {np.mean(impulsive_charges):.1f} charges/ep")
    print(f"  Patience:  {np.mean(patience_crash):.1%} crashes, {np.mean(patience_charges):.1f} charges/ep")

    if np.mean(patience_charges) > np.mean(impulsive_charges):
        print("  -> Patience agent shows proactive refueling behavior")
    if np.mean(patience_crash) < np.mean(impulsive_crash):
        print("  -> Patience reduces battery-related failures")

    # Return results
    return {
        'impulsive': impulsive_results,
        'patience': patience_results,
        'stats': {
            'crash': {'t': crash_t, 'p': crash_p, 'd': crash_d},
            'completion': {'t': complete_t, 'p': complete_p, 'd': complete_d},
            'waypoints': {'t': waypoint_t, 'p': waypoint_p, 'd': waypoint_d},
            'reward': {'t': reward_t, 'p': reward_p, 'd': reward_d},
            'charges': {'t': charges_t, 'p': charges_p, 'd': charges_d},
            'final_completion': {'t': final_complete_t, 'p': final_complete_p, 'd': final_complete_d},
        }
    }


if __name__ == "__main__":
    results = run_experiment(n_seeds=50, n_episodes=400)

    print("\n" + "=" * 70)
    print("EXPERIMENT 23 COMPLETE")
    print("=" * 70)
