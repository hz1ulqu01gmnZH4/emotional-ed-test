"""Experiment 19: Changing Seasons - Non-Stationary Rewards

A FAIR test environment for emotional learning where surprise detection helps.

Environment:
- Grid with two food sources (Red Berries and Blue Berries)
- Every N episodes (default 100), preference FLIPS:
  - Season A: Red = +1, Blue = -1
  - Season B: Red = -1, Blue = +1
- Agent must detect the change and adapt

Why Emotional Channels Should Help:
- Standard DQN suffers catastrophic forgetting / slow adaptation
- Surprise/Anger channel detects prediction error (expected vs actual reward)
- High surprise -> increase learning rate locally
- Enables rapid adaptation to regime change

Key Insight:
"Dense emotional signals are noise in dense reward environments,
but they become signal in sparse/harsh environments."

This environment is NON-STATIONARY - optimal policy changes suddenly.
Surprise-ED should detect reward prediction errors and adapt faster.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from scipy import stats


@dataclass
class SurpriseContext:
    """Context for computing surprise signals."""
    expected_reward: float
    actual_reward: float
    prediction_error: float
    is_season_change: bool
    current_season: str  # 'A' or 'B'
    episodes_in_season: int


class ChangingSeasonsEnv:
    """
    Changing Seasons Environment.

    5x5 grid with:
    - Agent starts at center (2, 2)
    - Red berries at (0, 0) and (0, 4)
    - Blue berries at (4, 0) and (4, 4)
    - Season flips every N episodes

    Season A: Red = +1, Blue = -1
    Season B: Red = -1, Blue = +1
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

    def __init__(self, size: int = 5, season_length: int = 100, max_steps: int = 50):
        self.size = size
        self.season_length = season_length
        self.max_steps = max_steps

        self.start_pos = (size // 2, size // 2)

        # Berry positions
        self.red_positions = {(0, 0), (0, size - 1)}
        self.blue_positions = {(size - 1, 0), (size - 1, size - 1)}

        # Season tracking
        self.current_season = 'A'
        self.total_episodes = 0
        self.episodes_in_current_season = 0

        # Episode state
        self.agent_pos = None
        self.steps = 0
        self.collected_berries = set()

    def _get_berry_reward(self, berry_type: str) -> float:
        """Get reward for berry type based on current season."""
        if self.current_season == 'A':
            return 1.0 if berry_type == 'red' else -1.0
        else:  # Season B
            return -1.0 if berry_type == 'red' else 1.0

    def reset(self) -> Tuple[int, int]:
        """Reset episode and potentially flip season."""
        # Check for season change
        is_change = False
        if self.total_episodes > 0 and self.total_episodes % self.season_length == 0:
            self.current_season = 'B' if self.current_season == 'A' else 'A'
            self.episodes_in_current_season = 0
            is_change = True

        self.total_episodes += 1
        self.episodes_in_current_season += 1

        # Reset episode state
        self.agent_pos = self.start_pos
        self.steps = 0
        self.collected_berries = set()

        return self.agent_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, SurpriseContext]:
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

        # Check for berries
        reward = -0.01  # Small step cost
        berry_collected = False

        if self.agent_pos in self.red_positions and self.agent_pos not in self.collected_berries:
            reward += self._get_berry_reward('red')
            self.collected_berries.add(self.agent_pos)
            berry_collected = True

        if self.agent_pos in self.blue_positions and self.agent_pos not in self.collected_berries:
            reward += self._get_berry_reward('blue')
            self.collected_berries.add(self.agent_pos)
            berry_collected = True

        # Episode ends when all berries collected or max steps
        all_collected = len(self.collected_berries) == 4
        done = all_collected or self.steps >= self.max_steps

        # Context for surprise computation
        # Note: expected_reward will be filled in by the agent based on Q-values
        context = SurpriseContext(
            expected_reward=0.0,  # Placeholder - agent fills this
            actual_reward=reward,
            prediction_error=0.0,  # Placeholder
            is_season_change=(self.episodes_in_current_season == 1),
            current_season=self.current_season,
            episodes_in_season=self.episodes_in_current_season
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

    def get_season_info(self) -> dict:
        """Get current season information."""
        return {
            'season': self.current_season,
            'episodes_in_season': self.episodes_in_current_season,
            'total_episodes': self.total_episodes,
            'is_first_in_season': self.episodes_in_current_season == 1
        }


class SurpriseModule:
    """
    Computes surprise signal from reward prediction error.

    High surprise when:
    - Actual reward differs significantly from expected
    - Happens after season change
    """

    def __init__(self, threshold: float = 0.3, max_surprise: float = 1.0,
                 decay: float = 0.9):
        self.threshold = threshold
        self.max_surprise = max_surprise
        self.decay = decay

        # Running surprise level (decays over time)
        self.surprise_level = 0.0

        # Track recent prediction errors for adaptation
        self.recent_errors = []
        self.window_size = 10

    def compute(self, expected: float, actual: float) -> float:
        """
        Compute surprise signal from prediction error.

        Returns value in [0, 1] where:
        - 0 = no surprise (prediction matched reality)
        - 1 = maximum surprise (large prediction error)
        """
        prediction_error = abs(expected - actual)

        # Update recent errors
        self.recent_errors.append(prediction_error)
        if len(self.recent_errors) > self.window_size:
            self.recent_errors.pop(0)

        # Phasic surprise (immediate response to error)
        if prediction_error > self.threshold:
            phasic_surprise = min(self.max_surprise, prediction_error / 2.0)
        else:
            phasic_surprise = 0.0

        # Update tonic surprise (persistent when errors are common)
        if phasic_surprise > 0:
            self.surprise_level = min(1.0, self.surprise_level + 0.3)
        else:
            self.surprise_level *= self.decay

        # Combined surprise
        total_surprise = max(self.surprise_level, phasic_surprise)
        return total_surprise

    def get_adaptive_lr_multiplier(self) -> float:
        """
        Get learning rate multiplier based on surprise.

        High surprise -> higher learning rate (faster adaptation).
        """
        return 1.0 + self.surprise_level * 2.0  # Up to 3x learning rate

    def reset(self):
        """Reset for new episode (keeps some surprise memory)."""
        # Don't fully reset - maintain some surprise awareness across episodes
        self.surprise_level *= 0.8


class StandardQLearner:
    """Standard Q-learning - no surprise adaptation."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.2):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions

    def select_action(self, state: int, context: Optional[SurpriseContext] = None) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def get_expected_reward(self, state: int, action: int) -> float:
        """Get expected reward for state-action pair."""
        return self.Q[state, action]

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: Optional[SurpriseContext] = None):
        """Standard TD update - fixed learning rate."""
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        """No per-episode state to reset."""
        pass


class SurpriseEDAgent:
    """
    Surprise-driven Emotional ED Agent.

    Uses surprise signal to:
    1. Detect reward prediction errors (sign of environment change)
    2. Increase learning rate when surprised (faster adaptation)
    3. Increase exploration when surprised (re-explore environment)
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon_base: float = 0.2,
                 surprise_weight: float = 0.5):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon_base = epsilon_base
        self.n_states = n_states
        self.n_actions = n_actions
        self.surprise_weight = surprise_weight

        # Surprise module
        self.surprise_module = SurpriseModule()

        # Track current surprise for action selection
        self.current_surprise = 0.0

        # Running average of rewards (for detecting changes)
        self.reward_ema = 0.0
        self.ema_alpha = 0.1

    def select_action(self, state: int, context: Optional[SurpriseContext] = None) -> int:
        """
        Surprise-modulated action selection.

        High surprise -> higher epsilon (more exploration to re-learn environment).
        """
        # Dynamic epsilon based on surprise
        epsilon = self.epsilon_base + self.current_surprise * 0.3
        epsilon = min(0.8, epsilon)  # Cap at 80% exploration

        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def get_expected_reward(self, state: int, action: int) -> float:
        """Get expected reward for state-action pair."""
        return self.Q[state, action]

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: SurpriseContext):
        """
        TD update with surprise-modulated learning rate.

        High surprise -> higher learning rate -> faster adaptation.
        """
        # Get expected value from Q-table
        expected = self.Q[state, action]

        # Compute surprise from prediction error
        self.current_surprise = self.surprise_module.compute(expected, reward)

        # Get adaptive learning rate
        lr_multiplier = self.surprise_module.get_adaptive_lr_multiplier()
        effective_lr = min(0.5, self.lr * lr_multiplier)  # Cap to prevent instability

        # Standard TD update with adaptive learning rate
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += effective_lr * delta

        # Update reward EMA for general tracking
        self.reward_ema = (1 - self.ema_alpha) * self.reward_ema + self.ema_alpha * reward

    def reset_episode(self):
        """Reset surprise module (partial reset)."""
        self.surprise_module.reset()


def run_episode(env: ChangingSeasonsEnv, agent, training: bool = True) -> dict:
    """Run one episode and return metrics."""
    state = env.reset()
    state_idx = env.state_to_index(state)

    total_reward = 0.0
    steps = 0
    positive_berries = 0
    negative_berries = 0

    agent.reset_episode()

    season_info = env.get_season_info()
    is_season_change = season_info['is_first_in_season'] and season_info['total_episodes'] > 1

    while True:
        # Select action
        action = agent.select_action(state_idx)

        # Get expected reward for surprise computation
        expected = agent.get_expected_reward(state_idx, action)

        # Take action
        next_state, reward, done, context = env.step(action)
        next_state_idx = env.state_to_index(next_state)

        # Track berry outcomes
        if reward > 0.5:
            positive_berries += 1
        elif reward < -0.5:
            negative_berries += 1

        # Update context with expected reward
        context.expected_reward = expected
        context.prediction_error = abs(expected - reward)

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
        'positive_berries': positive_berries,
        'negative_berries': negative_berries,
        'is_season_change': is_season_change,
        'season': season_info['season']
    }


def run_experiment(n_seeds: int = 50, n_episodes: int = 600, season_length: int = 100):
    """
    Run the Changing Seasons experiment with statistical validation.

    Args:
        n_seeds: Number of random seeds for statistical validation
        n_episodes: Total training episodes per seed (should span multiple seasons)
        season_length: Episodes per season before flip
    """
    print("=" * 70)
    print("EXPERIMENT 19: CHANGING SEASONS (Non-Stationary / Surprise)")
    print("=" * 70)
    print(f"\nEnvironment: 5x5 grid with Red and Blue berries")
    print(f"Seasons flip every {season_length} episodes")
    print("  Season A: Red = +1, Blue = -1")
    print("  Season B: Red = -1, Blue = +1")
    print("Hypothesis: Surprise-ED adapts faster after season change")
    print(f"Running {n_seeds} seeds x {n_episodes} episodes ({n_episodes // season_length} season changes)")
    print()

    standard_results = []
    surprise_results = []

    # Track performance around season changes
    standard_recovery = []  # Episodes to recover after change
    surprise_recovery = []

    for seed in range(n_seeds):
        np.random.seed(seed)

        print(f"\rSeed {seed+1}/{n_seeds}", end='', flush=True)

        # --- Standard Q-learner ---
        np.random.seed(seed)
        env = ChangingSeasonsEnv(size=5, season_length=season_length, max_steps=50)
        standard_agent = StandardQLearner(env.n_states, env.n_actions)

        seed_rewards = []
        seed_recovery_times = []
        last_season = None
        recovery_counter = 0
        recovered = True

        for ep in range(n_episodes):
            result = run_episode(env, standard_agent, training=True)
            seed_rewards.append(result['reward'])

            # Track recovery after season change
            if result['is_season_change']:
                last_season = result['season']
                recovery_counter = 0
                recovered = False

            if not recovered:
                recovery_counter += 1
                # Consider "recovered" when reward > 0 (collecting more good than bad)
                if result['reward'] > 0:
                    seed_recovery_times.append(recovery_counter)
                    recovered = True
                elif recovery_counter > season_length // 2:
                    # Didn't recover in time
                    seed_recovery_times.append(season_length // 2)
                    recovered = True

        standard_recovery.append(np.mean(seed_recovery_times) if seed_recovery_times else season_length // 2)
        standard_results.append({
            'mean_reward': np.mean(seed_rewards),
            'total_reward': np.sum(seed_rewards),
            'recovery_time': np.mean(seed_recovery_times) if seed_recovery_times else season_length // 2
        })

        # --- Surprise-ED Agent ---
        np.random.seed(seed)  # Reset for fair comparison
        env = ChangingSeasonsEnv(size=5, season_length=season_length, max_steps=50)
        surprise_agent = SurpriseEDAgent(env.n_states, env.n_actions)

        seed_rewards = []
        seed_recovery_times = []
        last_season = None
        recovery_counter = 0
        recovered = True

        for ep in range(n_episodes):
            result = run_episode(env, surprise_agent, training=True)
            seed_rewards.append(result['reward'])

            # Track recovery after season change
            if result['is_season_change']:
                last_season = result['season']
                recovery_counter = 0
                recovered = False

            if not recovered:
                recovery_counter += 1
                if result['reward'] > 0:
                    seed_recovery_times.append(recovery_counter)
                    recovered = True
                elif recovery_counter > season_length // 2:
                    seed_recovery_times.append(season_length // 2)
                    recovered = True

        surprise_recovery.append(np.mean(seed_recovery_times) if seed_recovery_times else season_length // 2)
        surprise_results.append({
            'mean_reward': np.mean(seed_rewards),
            'total_reward': np.sum(seed_rewards),
            'recovery_time': np.mean(seed_recovery_times) if seed_recovery_times else season_length // 2
        })

    print("\n")

    # Aggregate results
    standard_mean_reward = [r['mean_reward'] for r in standard_results]
    standard_total_reward = [r['total_reward'] for r in standard_results]
    standard_recovery_time = [r['recovery_time'] for r in standard_results]

    surprise_mean_reward = [r['mean_reward'] for r in surprise_results]
    surprise_total_reward = [r['total_reward'] for r in surprise_results]
    surprise_recovery_time = [r['recovery_time'] for r in surprise_results]

    # Statistical tests
    reward_t, reward_p = stats.ttest_ind(surprise_mean_reward, standard_mean_reward)
    total_t, total_p = stats.ttest_ind(surprise_total_reward, standard_total_reward)
    recovery_t, recovery_p = stats.ttest_ind(surprise_recovery_time, standard_recovery_time)

    # Effect sizes (Cohen's d)
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

    reward_d = cohens_d(surprise_mean_reward, standard_mean_reward)
    total_d = cohens_d(surprise_total_reward, standard_total_reward)
    recovery_d = cohens_d(surprise_recovery_time, standard_recovery_time)

    # Print results
    print("=" * 70)
    print("RESULTS: After Training")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Standard QL':<22} {'Surprise-ED':<22} {'p-value':<10} {'Cohen d':<10}")
    print("-" * 90)

    print(f"{'Mean Reward/Episode':<25} "
          f"{np.mean(standard_mean_reward):.3f} +/- {np.std(standard_mean_reward):.3f}  "
          f"{np.mean(surprise_mean_reward):.3f} +/- {np.std(surprise_mean_reward):.3f}  "
          f"{reward_p:.4f}    {reward_d:+.3f}")

    print(f"{'Total Cumulative Reward':<25} "
          f"{np.mean(standard_total_reward):.1f} +/- {np.std(standard_total_reward):.1f}    "
          f"{np.mean(surprise_total_reward):.1f} +/- {np.std(surprise_total_reward):.1f}    "
          f"{total_p:.4f}    {total_d:+.3f}")

    print(f"{'Recovery Time (episodes)':<25} "
          f"{np.mean(standard_recovery_time):.1f} +/- {np.std(standard_recovery_time):.1f}      "
          f"{np.mean(surprise_recovery_time):.1f} +/- {np.std(surprise_recovery_time):.1f}      "
          f"{recovery_p:.4f}    {recovery_d:+.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("\nHypothesis: Surprise-ED should detect reward changes and adapt faster,")
    print("           leading to shorter recovery time after season flips.")

    # Recovery time analysis (primary metric)
    if recovery_p < 0.05 and recovery_d < 0:
        print(f"\n[SUCCESS] Surprise-ED recovers faster after season change")
        print(f"  Recovery: {np.mean(surprise_recovery_time):.1f} vs {np.mean(standard_recovery_time):.1f} episodes")
        print(f"  Effect size: d={abs(recovery_d):.3f} ", end="")
        if abs(recovery_d) < 0.5:
            print("(small)")
        elif abs(recovery_d) < 0.8:
            print("(medium)")
        else:
            print("(large)")
    elif recovery_p < 0.05 and recovery_d > 0:
        print(f"\n[UNEXPECTED] Standard QL recovers faster")
    else:
        print(f"\n[INCONCLUSIVE] No significant difference in recovery time (p={recovery_p:.4f})")

    # Reward analysis
    if reward_p < 0.05 and reward_d > 0:
        print(f"\n[SUCCESS] Surprise-ED has higher mean reward")
        improvement = (np.mean(surprise_mean_reward) - np.mean(standard_mean_reward)) / abs(np.mean(standard_mean_reward)) * 100
        print(f"  Improvement: {improvement:.1f}%")

    # Total reward analysis
    if total_p < 0.05 and total_d > 0:
        print(f"\n[SUCCESS] Surprise-ED has higher cumulative reward")
        print(f"  This indicates better overall performance across season changes")

    # Return results
    return {
        'standard': {
            'mean_reward': standard_mean_reward,
            'total_reward': standard_total_reward,
            'recovery_time': standard_recovery_time,
        },
        'surprise': {
            'mean_reward': surprise_mean_reward,
            'total_reward': surprise_total_reward,
            'recovery_time': surprise_recovery_time,
        },
        'stats': {
            'reward': {'t': reward_t, 'p': reward_p, 'd': reward_d},
            'total': {'t': total_t, 'p': total_p, 'd': total_d},
            'recovery': {'t': recovery_t, 'p': recovery_p, 'd': recovery_d},
        }
    }


if __name__ == "__main__":
    results = run_experiment(n_seeds=50, n_episodes=600, season_length=100)

    print("\n" + "=" * 70)
    print("EXPERIMENT 19 COMPLETE")
    print("=" * 70)
