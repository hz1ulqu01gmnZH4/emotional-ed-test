"""Experiment 16: Sample Efficiency Comparison.

Tests whether emotional channels provide faster learning
through direct supervision signals.

Hypothesis:
- Emotional ED learns target behaviors faster than standard RL
- Direct emotional signals provide better sample efficiency
- This advantage persists across different emotional types
"""

import numpy as np
from typing import Dict, List

from gridworld import GridWorld
from agents import StandardQLearner, EmotionalEDAgent
from gridworld_anger import BlockedPathGridWorld
from agents_anger import StandardQLearner as AngerStandard, FrustrationEDAgent


def measure_learning_curve(env_class, agent_class, n_seeds: int = 30,
                           n_episodes: int = 300, metric: str = 'reward',
                           agent_kwargs: dict = None) -> Dict:
    """Measure learning curve for an agent type."""
    curves = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        env = env_class()
        agent = agent_class(env.n_states, env.n_actions, **(agent_kwargs or {}))

        episode_metrics = []
        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0
            min_threat_dist = float('inf')

            for step in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                agent.update(state, action, reward, next_state, done, ctx)

                total_reward += reward
                if hasattr(ctx, 'threat_distance'):
                    min_threat_dist = min(min_threat_dist, ctx.threat_distance)

                if done:
                    break
                state = next_state

            if metric == 'reward':
                episode_metrics.append(total_reward)
            elif metric == 'threat_dist':
                episode_metrics.append(min_threat_dist)

        curves.append(episode_metrics)

    curves = np.array(curves)
    return {
        'mean': np.mean(curves, axis=0),
        'std': np.std(curves, axis=0),
        'raw': curves
    }


def episodes_to_criterion(curves: np.ndarray, criterion: float,
                          direction: str = 'above') -> List[int]:
    """Find episode where criterion is first met."""
    n_seeds = curves.shape[0]
    n_episodes = curves.shape[1]
    episodes = []

    for seed in range(n_seeds):
        found = False
        # Use rolling average to smooth
        window = 10
        for ep in range(window, n_episodes):
            avg = np.mean(curves[seed, ep-window:ep])
            if direction == 'above' and avg >= criterion:
                episodes.append(ep)
                found = True
                break
            elif direction == 'below' and avg <= criterion:
                episodes.append(ep)
                found = True
                break

        if not found:
            episodes.append(n_episodes)

    return episodes


def main():
    np.random.seed(42)

    print("=" * 70)
    print("EXPERIMENT 16: SAMPLE EFFICIENCY COMPARISON")
    print("=" * 70)

    print("\nHYPOTHESIS: Emotional ED learns faster through direct signals")
    print("- Standard RL: Learns via reward signal only")
    print("- Emotional ED: Direct supervision from emotional channels")
    print()

    # Test 1: Fear Learning Efficiency
    print("=" * 70)
    print("TEST 1: Fear Channel Learning Efficiency")
    print("=" * 70)
    print("\nTask: Learn to avoid threat while reaching goal")
    print("Metric: Min threat distance (higher = safer)")

    print("\nMeasuring learning curves (30 seeds, 300 episodes)...")

    standard_fear = measure_learning_curve(
        GridWorld, StandardQLearner,
        n_seeds=30, n_episodes=300, metric='threat_dist'
    )
    emotional_fear = measure_learning_curve(
        GridWorld, EmotionalEDAgent,
        n_seeds=30, n_episodes=300, metric='threat_dist'
    )

    # Episodes to reach safe behavior (threat dist > 0.5)
    std_eps = episodes_to_criterion(standard_fear['raw'], 0.5, 'above')
    emo_eps = episodes_to_criterion(emotional_fear['raw'], 0.5, 'above')

    print(f"\nEpisodes to safe behavior (threat dist > 0.5):")
    print(f"  Standard: {np.mean(std_eps):.1f} ± {np.std(std_eps):.1f}")
    print(f"  Emotional ED: {np.mean(emo_eps):.1f} ± {np.std(emo_eps):.1f}")

    if np.mean(emo_eps) < np.mean(std_eps):
        speedup = np.mean(std_eps) / np.mean(emo_eps)
        print(f"  → Emotional ED is {speedup:.1f}x faster")
    else:
        print(f"  → No speedup for Emotional ED")

    # Learning curve at key points
    print(f"\nMean threat distance at key episodes:")
    print(f"{'Episode':<10} {'Standard':<18} {'Emotional ED':<18}")
    print("-" * 50)
    for ep in [10, 50, 100, 200, 300]:
        idx = ep - 1
        std_mean = standard_fear['mean'][idx]
        std_std = standard_fear['std'][idx]
        emo_mean = emotional_fear['mean'][idx]
        emo_std = emotional_fear['std'][idx]
        print(f"{ep:<10} {std_mean:.3f} ± {std_std:.3f}    {emo_mean:.3f} ± {emo_std:.3f}")

    # Test 2: Reward Efficiency
    print("\n" + "=" * 70)
    print("TEST 2: Overall Reward Learning Efficiency")
    print("=" * 70)
    print("\nTask: Reach goal while avoiding threat")
    print("Metric: Episode reward (higher = better)")

    standard_reward = measure_learning_curve(
        GridWorld, StandardQLearner,
        n_seeds=30, n_episodes=300, metric='reward'
    )
    emotional_reward = measure_learning_curve(
        GridWorld, EmotionalEDAgent,
        n_seeds=30, n_episodes=300, metric='reward'
    )

    # Episodes to reach good performance (reward > 0.8)
    std_eps_r = episodes_to_criterion(standard_reward['raw'], 0.8, 'above')
    emo_eps_r = episodes_to_criterion(emotional_reward['raw'], 0.8, 'above')

    print(f"\nEpisodes to good reward (> 0.8):")
    print(f"  Standard: {np.mean(std_eps_r):.1f} ± {np.std(std_eps_r):.1f}")
    print(f"  Emotional ED: {np.mean(emo_eps_r):.1f} ± {np.std(emo_eps_r):.1f}")

    print(f"\nMean reward at key episodes:")
    print(f"{'Episode':<10} {'Standard':<18} {'Emotional ED':<18}")
    print("-" * 50)
    for ep in [10, 50, 100, 200, 300]:
        idx = ep - 1
        std_mean = standard_reward['mean'][idx]
        std_std = standard_reward['std'][idx]
        emo_mean = emotional_reward['mean'][idx]
        emo_std = emotional_reward['std'][idx]
        print(f"{ep:<10} {std_mean:.3f} ± {std_std:.3f}    {emo_mean:.3f} ± {emo_std:.3f}")

    # Test 3: Anger/Frustration Learning
    print("\n" + "=" * 70)
    print("TEST 3: Anger Channel Learning Efficiency")
    print("=" * 70)
    print("\nTask: Navigate around obstacles to reach goal")

    standard_anger = measure_learning_curve(
        BlockedPathGridWorld, AngerStandard,
        n_seeds=30, n_episodes=300, metric='reward'
    )
    emotional_anger = measure_learning_curve(
        BlockedPathGridWorld, FrustrationEDAgent,
        n_seeds=30, n_episodes=300, metric='reward'
    )

    std_eps_a = episodes_to_criterion(standard_anger['raw'], 0.8, 'above')
    emo_eps_a = episodes_to_criterion(emotional_anger['raw'], 0.8, 'above')

    print(f"\nEpisodes to good reward (> 0.8):")
    print(f"  Standard: {np.mean(std_eps_a):.1f} ± {np.std(std_eps_a):.1f}")
    print(f"  Frustration ED: {np.mean(emo_eps_a):.1f} ± {np.std(emo_eps_a):.1f}")

    print(f"\nMean reward at key episodes:")
    print(f"{'Episode':<10} {'Standard':<18} {'Frustration ED':<18}")
    print("-" * 50)
    for ep in [10, 50, 100, 200, 300]:
        idx = ep - 1
        std_mean = standard_anger['mean'][idx]
        std_std = standard_anger['std'][idx]
        emo_mean = emotional_anger['mean'][idx]
        emo_std = emotional_anger['std'][idx]
        print(f"{ep:<10} {std_mean:.3f} ± {std_std:.3f}    {emo_mean:.3f} ± {emo_std:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Sample Efficiency")
    print("=" * 70)

    print(f"\n{'Task':<25} {'Standard Eps':<18} {'ED Eps':<18} {'Speedup':<10}")
    print("-" * 70)

    tasks = [
        ("Fear (threat avoidance)", np.mean(std_eps), np.mean(emo_eps)),
        ("Reward (goal reaching)", np.mean(std_eps_r), np.mean(emo_eps_r)),
        ("Anger (obstacle nav)", np.mean(std_eps_a), np.mean(emo_eps_a))
    ]

    total_speedup = 0
    for name, std, emo in tasks:
        if emo > 0:
            speedup = std / emo
            total_speedup += speedup
            print(f"{name:<25} {std:.1f}              {emo:.1f}              {speedup:.2f}x")
        else:
            print(f"{name:<25} {std:.1f}              {emo:.1f}              N/A")

    avg_speedup = total_speedup / len(tasks)
    print(f"\nAverage speedup: {avg_speedup:.2f}x")

    # Hypothesis test
    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST")
    print("=" * 70)

    print(f"\nH: Emotional ED provides faster learning than Standard RL")
    print(f"   Average speedup: {avg_speedup:.2f}x")

    if avg_speedup > 1.5:
        print("   ✓ Emotional ED learns SIGNIFICANTLY faster (>1.5x)")
    elif avg_speedup > 1.1:
        print("   ~ Emotional ED learns somewhat faster (1.1-1.5x)")
    else:
        print("   ✗ No significant sample efficiency advantage")

    print("\nConclusion:")
    if avg_speedup > 1.2:
        print("Emotional channels provide direct supervision that")
        print("accelerates learning compared to reward-only signal.")
    else:
        print("Results inconclusive - efficiency advantage not clearly demonstrated.")


if __name__ == "__main__":
    main()
