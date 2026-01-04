"""Experiment 21: Regulation V2 with Bayesian Reappraisal

Tests the fixed regulation mechanism with:
1. Bayesian belief update about threat safety
2. Credit assignment fix (belief flows to TD target)
3. Environment with FAKE threats (look scary but give bonus)

Hypothesis: RegulatedFearAgentV2 should outperform UnregulatedFearAgentV2
because it learns that fake threats are safe and approaches them for bonus.

Previous result (V1): d=-0.36 REVERSED (regulation hurt performance)
Expected result (V2): d>0.5 in correct direction (regulation helps)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ak/emotional-ed-test')

from scipy import stats
from dataclasses import dataclass
from typing import List, Tuple

from agents_v2.agents_regulation_v2 import (
    RegulatedFearAgentV2, UnregulatedFearAgentV2, RegulationGridWorldV2
)


@dataclass
class RegulationResult:
    regulated_rewards: List[float]
    unregulated_rewards: List[float]
    regulated_fake_bonuses: List[int]
    unregulated_fake_bonuses: List[int]
    regulated_real_harm: List[int]
    unregulated_real_harm: List[int]


def run_single_trial(n_episodes: int = 100, seed: int = None) -> Tuple[float, float, int, int, int, int]:
    """Run a single trial comparing regulated vs unregulated."""
    if seed is not None:
        np.random.seed(seed)

    env = RegulationGridWorldV2(size=6)

    # Regulated agent (with Bayesian reappraisal)
    regulated = RegulatedFearAgentV2(
        n_states=env.n_states,
        n_actions=env.n_actions,
        fear_weight=0.5
    )

    # Unregulated agent (no reappraisal)
    unregulated = UnregulatedFearAgentV2(
        n_states=env.n_states,
        n_actions=env.n_actions,
        fear_weight=0.5
    )

    # --- Run regulated agent ---
    reg_total_reward = 0
    reg_fake_bonus = 0
    reg_real_harm = 0

    for ep in range(n_episodes):
        state = env.reset()
        regulated.reset_episode()
        done = False

        while not done:
            # Get initial context
            context = env._get_context(state)
            action = regulated.select_action(state, context)
            next_state, reward, done, context = env.step(action)

            regulated.update(state, action, reward, next_state, done, context)

            reg_total_reward += reward

            if context.is_fake_threat and reward > 0.3:
                reg_fake_bonus += 1
            if context.was_harmed:
                reg_real_harm += 1

            state = next_state

    # Reset for fair comparison
    if seed is not None:
        np.random.seed(seed + 10000)

    env = RegulationGridWorldV2(size=6)

    # --- Run unregulated agent ---
    unreg_total_reward = 0
    unreg_fake_bonus = 0
    unreg_real_harm = 0

    for ep in range(n_episodes):
        state = env.reset()
        unregulated.reset_episode()
        done = False

        while not done:
            context = env._get_context(state)
            action = unregulated.select_action(state, context)
            next_state, reward, done, context = env.step(action)

            unregulated.update(state, action, reward, next_state, done, context)

            unreg_total_reward += reward

            if context.is_fake_threat and reward > 0.3:
                unreg_fake_bonus += 1
            if context.was_harmed:
                unreg_real_harm += 1

            state = next_state

    return (
        reg_total_reward / n_episodes,
        unreg_total_reward / n_episodes,
        reg_fake_bonus,
        unreg_fake_bonus,
        reg_real_harm,
        unreg_real_harm
    )


def run_experiment(n_trials: int = 50, n_episodes: int = 100) -> RegulationResult:
    """Run full regulation experiment."""
    print(f"Running Regulation V2 Experiment: {n_trials} trials, {n_episodes} episodes each")
    print("=" * 60)
    print("Environment: Real threat (-0.5) and Fake threat (+0.4 bonus)")
    print("Hypothesis: Regulated agent learns fake is safe → collects bonus")
    print("=" * 60)

    regulated_rewards = []
    unregulated_rewards = []
    regulated_fake = []
    unregulated_fake = []
    regulated_harm = []
    unregulated_harm = []

    for trial in range(n_trials):
        reg_r, unreg_r, reg_f, unreg_f, reg_h, unreg_h = run_single_trial(
            n_episodes=n_episodes,
            seed=trial * 1000
        )

        regulated_rewards.append(reg_r)
        unregulated_rewards.append(unreg_r)
        regulated_fake.append(reg_f)
        unregulated_fake.append(unreg_f)
        regulated_harm.append(reg_h)
        unregulated_harm.append(unreg_h)

        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{n_trials} complete")

    return RegulationResult(
        regulated_rewards=regulated_rewards,
        unregulated_rewards=unregulated_rewards,
        regulated_fake_bonuses=regulated_fake,
        unregulated_fake_bonuses=unregulated_fake,
        regulated_real_harm=regulated_harm,
        unregulated_real_harm=unregulated_harm
    )


def analyze_results(result: RegulationResult):
    """Statistical analysis."""
    print("\n" + "=" * 60)
    print("RESULTS: Regulation V2 (Bayesian Reappraisal)")
    print("=" * 60)

    # Rewards
    reg_r = np.array(result.regulated_rewards)
    unreg_r = np.array(result.unregulated_rewards)

    print("\n--- Average Rewards ---")
    print(f"Regulated (reappraisal): {np.mean(reg_r):.3f} ± {np.std(reg_r):.3f}")
    print(f"Unregulated: {np.mean(unreg_r):.3f} ± {np.std(unreg_r):.3f}")

    t_stat, p_value = stats.ttest_ind(reg_r, unreg_r)
    pooled_std = np.sqrt((np.std(reg_r)**2 + np.std(unreg_r)**2) / 2)
    cohens_d = (np.mean(reg_r) - np.mean(unreg_r)) / pooled_std if pooled_std > 0 else 0

    print(f"\nt-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Cohen's d: {cohens_d:.3f}")

    if cohens_d > 0:
        print("Direction: CORRECT (regulated has higher reward)")
    else:
        print("Direction: REVERSED (regulated has lower reward)")

    # Fake bonus collection
    reg_f = np.array(result.regulated_fake_bonuses)
    unreg_f = np.array(result.unregulated_fake_bonuses)

    print("\n--- Fake Threat Bonuses Collected ---")
    print(f"Regulated: {np.mean(reg_f):.2f} ± {np.std(reg_f):.2f}")
    print(f"Unregulated: {np.mean(unreg_f):.2f} ± {np.std(unreg_f):.2f}")

    t_stat_f, p_value_f = stats.ttest_ind(reg_f, unreg_f)
    pooled_std_f = np.sqrt((np.std(reg_f)**2 + np.std(unreg_f)**2) / 2)
    cohens_d_f = (np.mean(reg_f) - np.mean(unreg_f)) / pooled_std_f if pooled_std_f > 0 else 0

    print(f"Cohen's d (fake bonus): {cohens_d_f:.3f}")

    # Real harm
    reg_h = np.array(result.regulated_real_harm)
    unreg_h = np.array(result.unregulated_real_harm)

    print("\n--- Real Threat Harm ---")
    print(f"Regulated: {np.mean(reg_h):.2f} ± {np.std(reg_h):.2f}")
    print(f"Unregulated: {np.mean(unreg_h):.2f} ± {np.std(unreg_h):.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if p_value < 0.05 and cohens_d > 0:
        print("✓ SUCCESS: Regulation V2 improves performance")
        print(f"  Effect size: d={cohens_d:.2f}")
    elif p_value < 0.05 and cohens_d < 0:
        print("✗ STILL REVERSED: Regulation V2 still hurts performance")
    else:
        print("~ INCONCLUSIVE: No significant difference")

    # Compare to V1
    print("\n--- Comparison to V1 ---")
    print(f"V1 result: d=-0.36 (reversed, regulation hurt)")
    print(f"V2 result: d={cohens_d:.2f}")

    if cohens_d > 0 and p_value < 0.05:
        print("✓ MAJOR IMPROVEMENT: Fix worked!")
    elif cohens_d > 0:
        print("~ PARTIAL IMPROVEMENT: Correct direction but not significant")
    else:
        print("✗ NO IMPROVEMENT: Still reversed")

    return {
        'reward_cohens_d': cohens_d,
        'reward_p_value': p_value,
        'fake_bonus_cohens_d': cohens_d_f
    }


# Add helper method to environment for getting context at arbitrary state
def _get_context(env, state):
    """Get context for a state without stepping."""
    from agents_v2.agents_regulation_v2 import RegulationContext

    row = state // env.size
    col = state % env.size
    pos = np.array([row, col])

    real_dist = env._distance(pos, env.real_threat_pos)
    fake_dist = env._distance(pos, env.fake_threat_pos)
    goal_dist = env._distance(pos, env.goal_pos)

    # CRITICAL FIX: Use same threshold (1.0) as in environment step()
    near_real = real_dist < 1.0
    near_fake = fake_dist < 1.0

    if near_real:
        threat_type = 'real'
        is_real = True
        is_fake = False
    elif near_fake:
        threat_type = 'fake'
        is_real = False
        is_fake = True
    else:
        threat_type = 'none'
        is_real = False
        is_fake = False

    return RegulationContext(
        threat_distance=min(real_dist, fake_dist),
        goal_distance=goal_dist,
        is_real_threat=is_real,
        is_fake_threat=is_fake,
        threat_type=threat_type
    )


# Patch the environment class
RegulationGridWorldV2._get_context = _get_context


if __name__ == '__main__':
    result = run_experiment(n_trials=50, n_episodes=100)
    stats_result = analyze_results(result)
