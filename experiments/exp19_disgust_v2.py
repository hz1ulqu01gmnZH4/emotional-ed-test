"""Experiment 19: Disgust V2 with Directional Repulsion

Tests the fixed disgust mechanism that uses directional repulsion
instead of argmax boost.

Hypothesis: DisgustOnlyAgentV2 should have FEWER contamination touches
than FearOnlyAgentV2 (which habituates) because disgust doesn't habituate
and now correctly repels from contaminants.

Previous result (V1): d=0.25 REVERSED (disgust touched MORE)
Expected result (V2): d>0.5 in correct direction (disgust touches LESS)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ak/emotional-ed-test')

from scipy import stats
from dataclasses import dataclass
from typing import List, Tuple

from agents_v2.agents_disgust_v2 import DisgustOnlyAgentV2, FearOnlyAgentV2
from gridworld_disgust import DisgustGridWorld


@dataclass
class ExperimentResult:
    disgust_touches: List[float]
    fear_touches: List[float]
    disgust_rewards: List[float]
    fear_rewards: List[float]


def run_single_trial(n_episodes: int = 100, seed: int = None) -> Tuple[float, float, float, float]:
    """Run a single trial comparing disgust vs fear agent."""
    if seed is not None:
        np.random.seed(seed)

    env = DisgustGridWorld(size=6)

    # Create agents
    disgust_agent = DisgustOnlyAgentV2(
        n_states=env.n_states,
        n_actions=env.n_actions,
        grid_size=6,
        disgust_weight=0.5
    )

    fear_agent = FearOnlyAgentV2(
        n_states=env.n_states,
        n_actions=env.n_actions,
        grid_size=6,
        habituation_rate=0.05
    )

    # Run disgust agent
    disgust_touches = 0
    disgust_total_reward = 0

    for ep in range(n_episodes):
        state = env.reset()
        disgust_agent.reset_episode()
        done = False

        while not done:
            action = disgust_agent.select_action(state)
            next_state, reward, done, context = env.step(action)
            disgust_agent.update(state, action, reward, next_state, done, context)

            if context.touched_contaminant:
                disgust_touches += 1
            disgust_total_reward += reward

            state = next_state

    # Reset environment seed for fair comparison
    if seed is not None:
        np.random.seed(seed + 10000)

    env = DisgustGridWorld(size=6)

    # Run fear agent
    fear_touches = 0
    fear_total_reward = 0

    for ep in range(n_episodes):
        state = env.reset()
        fear_agent.reset_episode()
        done = False

        while not done:
            action = fear_agent.select_action(state)
            next_state, reward, done, context = env.step(action)
            fear_agent.update(state, action, reward, next_state, done, context)

            if context.touched_contaminant:
                fear_touches += 1
            fear_total_reward += reward

            state = next_state

    return (
        disgust_touches,
        fear_touches,
        disgust_total_reward / n_episodes,
        fear_total_reward / n_episodes
    )


def run_experiment(n_trials: int = 50, n_episodes: int = 100) -> ExperimentResult:
    """Run full experiment with N trials."""
    print(f"Running Disgust V2 Experiment: {n_trials} trials, {n_episodes} episodes each")
    print("=" * 60)

    disgust_touches = []
    fear_touches = []
    disgust_rewards = []
    fear_rewards = []

    for trial in range(n_trials):
        d_touch, f_touch, d_reward, f_reward = run_single_trial(
            n_episodes=n_episodes,
            seed=trial * 1000
        )

        disgust_touches.append(d_touch)
        fear_touches.append(f_touch)
        disgust_rewards.append(d_reward)
        fear_rewards.append(f_reward)

        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{n_trials} complete")

    return ExperimentResult(
        disgust_touches=disgust_touches,
        fear_touches=fear_touches,
        disgust_rewards=disgust_rewards,
        fear_rewards=fear_rewards
    )


def analyze_results(result: ExperimentResult):
    """Statistical analysis of results."""
    print("\n" + "=" * 60)
    print("RESULTS: Disgust V2 (Directional Repulsion)")
    print("=" * 60)

    # Contamination touches
    d_touches = np.array(result.disgust_touches)
    f_touches = np.array(result.fear_touches)

    print("\n--- Contamination Touches ---")
    print(f"Disgust V2: {np.mean(d_touches):.2f} ± {np.std(d_touches):.2f}")
    print(f"Fear (habituating): {np.mean(f_touches):.2f} ± {np.std(f_touches):.2f}")

    # t-test (expect disgust < fear)
    t_stat, p_value = stats.ttest_ind(d_touches, f_touches)

    # Cohen's d (negative = disgust fewer touches = CORRECT direction)
    pooled_std = np.sqrt((np.std(d_touches)**2 + np.std(f_touches)**2) / 2)
    cohens_d = (np.mean(d_touches) - np.mean(f_touches)) / pooled_std

    print(f"\nt-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Cohen's d: {cohens_d:.3f}")

    if cohens_d < 0:
        print("Direction: CORRECT (disgust has fewer touches)")
    else:
        print("Direction: REVERSED (disgust has more touches)")

    # Rewards
    d_rewards = np.array(result.disgust_rewards)
    f_rewards = np.array(result.fear_rewards)

    print("\n--- Average Rewards ---")
    print(f"Disgust V2: {np.mean(d_rewards):.3f} ± {np.std(d_rewards):.3f}")
    print(f"Fear: {np.mean(f_rewards):.3f} ± {np.std(f_rewards):.3f}")

    t_stat_r, p_value_r = stats.ttest_ind(d_rewards, f_rewards)
    pooled_std_r = np.sqrt((np.std(d_rewards)**2 + np.std(f_rewards)**2) / 2)
    cohens_d_r = (np.mean(d_rewards) - np.mean(f_rewards)) / pooled_std_r

    print(f"\nt-statistic (reward): {t_stat_r:.3f}")
    print(f"p-value (reward): {p_value_r:.4f}")
    print(f"Cohen's d (reward): {cohens_d_r:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if p_value < 0.05 and cohens_d < 0:
        print("✓ SUCCESS: Disgust V2 significantly reduces contamination touches")
        print(f"  Effect size: d={abs(cohens_d):.2f} (large if >0.8)")
    elif p_value < 0.05 and cohens_d > 0:
        print("✗ STILL REVERSED: Disgust V2 still has more touches")
    else:
        print("~ INCONCLUSIVE: No significant difference")

    # Compare to V1
    print("\n--- Comparison to V1 ---")
    print(f"V1 result: d=0.25 (reversed, disgust touched MORE)")
    print(f"V2 result: d={cohens_d:.2f}")

    if cohens_d < -0.5:
        print("✓ MAJOR IMPROVEMENT: Fix worked!")
    elif cohens_d < 0:
        print("~ PARTIAL IMPROVEMENT: Correct direction but small effect")
    else:
        print("✗ NO IMPROVEMENT: Still reversed")

    return {
        'touches_cohens_d': cohens_d,
        'touches_p_value': p_value,
        'reward_cohens_d': cohens_d_r,
        'reward_p_value': p_value_r
    }


if __name__ == '__main__':
    result = run_experiment(n_trials=50, n_episodes=100)
    stats_result = analyze_results(result)
