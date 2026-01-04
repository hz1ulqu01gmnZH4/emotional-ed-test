"""Experiment 20: Transfer V2 with Feature-Based Q-Function

Tests whether feature-based Q-learning enables fear generalization
to novel threat locations.

Setup:
1. Train on environment with threat at position A
2. Test on environment with threat at NEW position B
3. Measure: Does threat avoidance transfer?

Hypothesis: FeatureBasedFearAgent should show transfer because
features (threat_distance, fear_level) are position-invariant.
TabularBaselineAgent should NOT transfer (new states = Q=0).

Previous result (V1 tabular): d=0.12 (no significant transfer)
Expected result (V2 feature): d>0.5 (significant transfer)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ak/emotional-ed-test')

from scipy import stats
from dataclasses import dataclass
from typing import List, Tuple

from agents_v2.agents_feature_based import (
    FeatureBasedFearAgent, TabularBaselineAgent, FeatureContext
)


class TransferGridWorld:
    """Gridworld with configurable threat position for transfer testing."""

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

    def __init__(self, size: int = 6, threat_pos: Tuple[int, int] = (2, 2)):
        self.size = size
        self.threat_pos = np.array(threat_pos)
        self.goal_pos = np.array([5, 5])
        self.reset()

    def reset(self) -> Tuple[int, int]:
        self.agent_pos = np.array([0, 0])
        self.step_count = 0
        return tuple(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def _get_direction(self, from_pos: np.ndarray, to_pos: np.ndarray) -> Tuple[int, int]:
        diff = to_pos - from_pos
        return (int(diff[0]), int(diff[1]))

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, FeatureContext]:
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        # Boundary check
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        # Distances
        threat_dist = self._distance(self.agent_pos, self.threat_pos)
        goal_dist = self._distance(self.agent_pos, self.goal_pos)

        # Rewards
        reward = -0.01  # Step cost

        # Threat penalty
        if threat_dist < 1.0:
            reward -= 0.5

        # Goal
        done = np.array_equal(self.agent_pos, self.goal_pos) or self.step_count >= 100
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 1.0

        context = FeatureContext(
            threat_distance=threat_dist,
            goal_distance=goal_dist,
            threat_direction=self._get_direction(self.agent_pos, self.threat_pos),
            goal_direction=self._get_direction(self.agent_pos, self.goal_pos),
            near_threat=threat_dist < 2.0,
            near_wall=(
                self.agent_pos[0] == 0 or self.agent_pos[0] == self.size - 1 or
                self.agent_pos[1] == 0 or self.agent_pos[1] == self.size - 1
            )
        )

        return tuple(self.agent_pos), reward, done, context

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4


@dataclass
class TransferResult:
    feature_train_hits: List[float]
    feature_test_hits: List[float]
    tabular_train_hits: List[float]
    tabular_test_hits: List[float]
    feature_transfer_ratio: List[float]
    tabular_transfer_ratio: List[float]


def run_single_trial(n_train_episodes: int = 50, n_test_episodes: int = 20,
                     seed: int = None) -> Tuple[float, float, float, float]:
    """Run a single transfer trial."""
    if seed is not None:
        np.random.seed(seed)

    # Training environment: threat at (2, 2)
    train_env = TransferGridWorld(size=6, threat_pos=(2, 2))

    # Test environment: threat at NEW position (4, 3)
    test_env = TransferGridWorld(size=6, threat_pos=(4, 3))

    # Feature-based agent
    feature_agent = FeatureBasedFearAgent(n_actions=4, lr=0.05, epsilon=0.1)

    # Tabular baseline
    tabular_agent = TabularBaselineAgent(n_states=36, n_actions=4, lr=0.1, epsilon=0.1)

    # --- Train feature agent ---
    feature_train_hits = 0
    for ep in range(n_train_episodes):
        state_pos = train_env.reset()
        state = train_env._pos_to_state(np.array(state_pos))
        feature_agent.reset_episode()
        done = False

        while not done:
            _, _, _, context = train_env.step(0)  # Get context for current position
            train_env.agent_pos = np.array(state_pos)  # Reset position
            train_env.step_count -= 1

            action = feature_agent.select_action(state_pos, context)
            next_state_pos, reward, done, next_context = train_env.step(action)

            feature_agent.update(state_pos, action, reward, next_state_pos, done,
                                context, next_context)

            if context.threat_distance < 1.0:
                feature_train_hits += 1

            state_pos = next_state_pos

    # --- Test feature agent (NEW threat location) ---
    feature_test_hits = 0
    feature_agent.epsilon = 0.0  # Greedy evaluation

    for ep in range(n_test_episodes):
        state_pos = test_env.reset()
        done = False

        while not done:
            # Get context (threat at new location)
            test_env.agent_pos = np.array(state_pos)
            _, _, _, context = test_env.step(0)
            test_env.agent_pos = np.array(state_pos)
            test_env.step_count -= 1

            action = feature_agent.select_action(state_pos, context)
            next_state_pos, reward, done, next_context = test_env.step(action)

            if context.threat_distance < 1.0:
                feature_test_hits += 1

            state_pos = next_state_pos

    # --- Train tabular agent ---
    if seed is not None:
        np.random.seed(seed + 5000)

    train_env = TransferGridWorld(size=6, threat_pos=(2, 2))

    tabular_train_hits = 0
    for ep in range(n_train_episodes):
        state_pos = train_env.reset()
        state = train_env._pos_to_state(np.array(state_pos))
        tabular_agent.reset_episode()
        done = False

        while not done:
            _, _, _, context = train_env.step(0)
            train_env.agent_pos = np.array(state_pos)
            train_env.step_count -= 1

            action = tabular_agent.select_action(state, context)
            next_state_pos, reward, done, next_context = train_env.step(action)
            next_state = train_env._pos_to_state(np.array(next_state_pos))

            tabular_agent.update(state, action, reward, next_state, done)

            if context.threat_distance < 1.0:
                tabular_train_hits += 1

            state_pos = next_state_pos
            state = next_state

    # --- Test tabular agent (NEW threat location) ---
    test_env = TransferGridWorld(size=6, threat_pos=(4, 3))

    tabular_test_hits = 0
    tabular_agent.epsilon = 0.0

    for ep in range(n_test_episodes):
        state_pos = test_env.reset()
        state = test_env._pos_to_state(np.array(state_pos))
        done = False

        while not done:
            test_env.agent_pos = np.array(state_pos)
            _, _, _, context = test_env.step(0)
            test_env.agent_pos = np.array(state_pos)
            test_env.step_count -= 1

            action = tabular_agent.select_action(state, context)
            next_state_pos, reward, done, next_context = test_env.step(action)
            next_state = test_env._pos_to_state(np.array(next_state_pos))

            if context.threat_distance < 1.0:
                tabular_test_hits += 1

            state_pos = next_state_pos
            state = next_state

    return (
        feature_train_hits / n_train_episodes,
        feature_test_hits / n_test_episodes,
        tabular_train_hits / n_train_episodes,
        tabular_test_hits / n_test_episodes
    )


def run_experiment(n_trials: int = 50) -> TransferResult:
    """Run full transfer experiment."""
    print(f"Running Transfer V2 Experiment: {n_trials} trials")
    print("=" * 60)
    print("Training: threat at (2,2)")
    print("Testing: threat at (4,3) - NEW location")
    print("=" * 60)

    feature_train_hits = []
    feature_test_hits = []
    tabular_train_hits = []
    tabular_test_hits = []

    for trial in range(n_trials):
        f_train, f_test, t_train, t_test = run_single_trial(
            n_train_episodes=50,
            n_test_episodes=20,
            seed=trial * 1000
        )

        feature_train_hits.append(f_train)
        feature_test_hits.append(f_test)
        tabular_train_hits.append(t_train)
        tabular_test_hits.append(t_test)

        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{n_trials} complete")

    # Compute transfer ratios (test_hits / train_hits, capped)
    feature_transfer = []
    tabular_transfer = []

    for i in range(n_trials):
        if feature_train_hits[i] > 0:
            feature_transfer.append(feature_test_hits[i] / max(0.1, feature_train_hits[i]))
        else:
            feature_transfer.append(1.0 if feature_test_hits[i] == 0 else 0.0)

        if tabular_train_hits[i] > 0:
            tabular_transfer.append(tabular_test_hits[i] / max(0.1, tabular_train_hits[i]))
        else:
            tabular_transfer.append(1.0 if tabular_test_hits[i] == 0 else 0.0)

    return TransferResult(
        feature_train_hits=feature_train_hits,
        feature_test_hits=feature_test_hits,
        tabular_train_hits=tabular_train_hits,
        tabular_test_hits=tabular_test_hits,
        feature_transfer_ratio=feature_transfer,
        tabular_transfer_ratio=tabular_transfer
    )


def analyze_results(result: TransferResult):
    """Statistical analysis."""
    print("\n" + "=" * 60)
    print("RESULTS: Transfer V2 (Feature-Based Q)")
    print("=" * 60)

    # Training hits (should be similar - both learn in training env)
    print("\n--- Training Phase (threat at 2,2) ---")
    print(f"Feature-based: {np.mean(result.feature_train_hits):.3f} ± {np.std(result.feature_train_hits):.3f} hits/ep")
    print(f"Tabular: {np.mean(result.tabular_train_hits):.3f} ± {np.std(result.tabular_train_hits):.3f} hits/ep")

    # Test hits (feature should be lower - transfer works)
    print("\n--- Test Phase (NEW threat at 4,3) ---")
    print(f"Feature-based: {np.mean(result.feature_test_hits):.3f} ± {np.std(result.feature_test_hits):.3f} hits/ep")
    print(f"Tabular: {np.mean(result.tabular_test_hits):.3f} ± {np.std(result.tabular_test_hits):.3f} hits/ep")

    # Compare test hits (primary measure)
    f_test = np.array(result.feature_test_hits)
    t_test = np.array(result.tabular_test_hits)

    t_stat, p_value = stats.ttest_ind(f_test, t_test)
    pooled_std = np.sqrt((np.std(f_test)**2 + np.std(t_test)**2) / 2)
    cohens_d = (np.mean(f_test) - np.mean(t_test)) / pooled_std if pooled_std > 0 else 0

    print(f"\n--- Transfer Comparison (test hits) ---")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Cohen's d: {cohens_d:.3f}")

    if cohens_d < 0:
        print("Direction: CORRECT (feature-based has fewer hits = better avoidance)")
    else:
        print("Direction: Feature-based did NOT transfer better")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if p_value < 0.05 and cohens_d < 0:
        print("✓ SUCCESS: Feature-based agent transfers threat avoidance")
        print(f"  Effect size: d={abs(cohens_d):.2f}")
    elif p_value < 0.05 and cohens_d > 0:
        print("✗ REVERSED: Tabular somehow better at transfer")
    else:
        print("~ INCONCLUSIVE: No significant transfer difference")

    # Compare to V1
    print("\n--- Comparison to V1 ---")
    print(f"V1 result (tabular only): d=0.12 (no transfer)")
    print(f"V2 result (feature vs tabular): d={cohens_d:.2f}")

    return {
        'test_hits_cohens_d': cohens_d,
        'test_hits_p_value': p_value,
        'feature_test_mean': np.mean(f_test),
        'tabular_test_mean': np.mean(t_test)
    }


if __name__ == '__main__':
    result = run_experiment(n_trials=50)
    stats_result = analyze_results(result)
