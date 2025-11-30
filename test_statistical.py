"""Experiment 12: Statistical Validation of Key Results.

Re-run key experiments with N=50 seeds to establish:
- Mean and standard deviation
- 95% confidence intervals
- p-values vs baseline (using permutation test)
- Cohen's d effect sizes

Validates: Fear, Anger, Regret, Multi-channel, Transfer
"""

import numpy as np
from typing import Dict, List, Tuple
import math

# Import from existing experiments
from gridworld import GridWorld
from agents import StandardQLearner, EmotionalEDAgent
from gridworld_anger import BlockedPathGridWorld
from agents_anger import StandardQLearner as AngerStandardAgent, FrustrationEDAgent
from gridworld_regret import TwoDoorEnv
from agents_regret import StandardBanditAgent, RegretEDAgent
from gridworld_integration import IntegrationGridWorld
from agents_integration import (StandardQLearner as IntegrationStandard, FearDominantAgent,
                                AngerDominantAgent)
from gridworld_transfer import TrainingGridWorld, NovelThreatGridWorld
from agents_transfer import StandardQLearner as TransferStandard, EmotionalTransferAgent, NoTransferAgent


def t_test_ind(arr1: np.ndarray, arr2: np.ndarray) -> Tuple[float, float]:
    """Independent samples t-test (numpy only implementation)."""
    n1, n2 = len(arr1), len(arr2)
    mean1, mean2 = np.mean(arr1), np.mean(arr2)
    var1, var2 = np.var(arr1, ddof=1), np.var(arr2, ddof=1)

    # Pooled standard error
    se = np.sqrt(var1/n1 + var2/n2)
    if se == 0:
        return 0.0, 1.0

    t_stat = (mean1 - mean2) / se

    # Welch's degrees of freedom
    df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

    # Approximate p-value using permutation test for simplicity
    combined = np.concatenate([arr1, arr2])
    observed_diff = abs(mean1 - mean2)
    n_perm = 1000
    count = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_diff = abs(np.mean(combined[:n1]) - np.mean(combined[n1:]))
        if perm_diff >= observed_diff:
            count += 1
    p_value = count / n_perm

    return t_stat, max(p_value, 1/n_perm)  # Avoid p=0


def ci_95(arr: np.ndarray) -> Tuple[float, float]:
    """Compute 95% CI using bootstrap."""
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    # Approximate t-value for 95% CI with large n
    t_val = 1.96 if n > 30 else 2.0
    margin = t_val * std / np.sqrt(n)
    return (mean - margin, mean + margin)


def compute_statistics(data1: List[float], data2: List[float],
                       name1: str = "Control", name2: str = "Treatment") -> Dict:
    """Compute comprehensive statistics for two groups."""
    arr1 = np.array(data1)
    arr2 = np.array(data2)

    # Basic stats
    mean1, mean2 = np.mean(arr1), np.mean(arr2)
    std1, std2 = np.std(arr1, ddof=1), np.std(arr2, ddof=1)
    n1, n2 = len(arr1), len(arr2)

    # 95% CI
    ci1 = ci_95(arr1)
    ci2 = ci_95(arr2)

    # t-test (independent samples)
    t_stat, p_value = t_test_ind(arr1, arr2)

    # Cohen's d
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0

    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_interp = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interp = "small"
    elif abs(cohens_d) < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"

    return {
        name1: {"mean": mean1, "std": std1, "ci_95": ci1, "n": n1},
        name2: {"mean": mean2, "std": std2, "ci_95": ci2, "n": n2},
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "effect_interpretation": effect_interp,
        "significant": p_value < 0.05
    }


def run_fear_experiment(n_seeds: int = 50, n_train: int = 200, n_eval: int = 50) -> Dict:
    """Run fear avoidance experiment with multiple seeds."""
    standard_distances = []
    emotional_distances = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        env = GridWorld()

        # Standard agent
        std_agent = StandardQLearner(env.n_states, env.n_actions)
        for _ in range(n_train):
            state = env.reset()
            for _ in range(100):
                action = std_agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                std_agent.update(state, action, reward, next_state, done, ctx)
                if done:
                    break
                state = next_state

        # Evaluate
        std_agent.epsilon = 0.05
        min_dists = []
        for _ in range(n_eval):
            state = env.reset()
            episode_min = float('inf')
            for _ in range(100):
                action = std_agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                episode_min = min(episode_min, ctx.threat_distance)
                if done:
                    break
                state = next_state
            min_dists.append(episode_min)
        standard_distances.append(np.mean(min_dists))

        # Emotional agent
        np.random.seed(seed)
        emo_agent = EmotionalEDAgent(env.n_states, env.n_actions)
        for _ in range(n_train):
            state = env.reset()
            for _ in range(100):
                action = emo_agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                emo_agent.update(state, action, reward, next_state, done, ctx)
                if done:
                    break
                state = next_state

        emo_agent.epsilon = 0.05
        min_dists = []
        for _ in range(n_eval):
            state = env.reset()
            episode_min = float('inf')
            for _ in range(100):
                action = emo_agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                episode_min = min(episode_min, ctx.threat_distance)
                if done:
                    break
                state = next_state
            min_dists.append(episode_min)
        emotional_distances.append(np.mean(min_dists))

    return compute_statistics(standard_distances, emotional_distances,
                              "Standard", "Emotional ED")


def run_anger_experiment(n_seeds: int = 50, n_episodes: int = 20) -> Dict:
    """Run anger/frustration experiment with multiple seeds."""
    standard_hits = []
    frustrated_hits = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        env = BlockedPathGridWorld()

        # Standard agent
        std_agent = AngerStandardAgent(env.n_states, env.n_actions)
        total_hits = 0
        for _ in range(n_episodes):
            state = env.reset()
            for _ in range(100):
                action = std_agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                std_agent.update(state, action, reward, next_state, done, ctx)
                if ctx.was_blocked:
                    total_hits += 1
                if done:
                    break
                state = next_state
        standard_hits.append(total_hits)

        # Frustrated agent
        np.random.seed(seed)
        frust_agent = FrustrationEDAgent(env.n_states, env.n_actions)
        total_hits = 0
        for _ in range(n_episodes):
            state = env.reset()
            for _ in range(100):
                action = frust_agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                frust_agent.update(state, action, reward, next_state, done, ctx)
                if ctx.was_blocked:
                    total_hits += 1
                if done:
                    break
                state = next_state
        frustrated_hits.append(total_hits)

    return compute_statistics(standard_hits, frustrated_hits,
                              "Standard", "Frustrated")


def run_regret_experiment(n_seeds: int = 50, n_trials: int = 100) -> Dict:
    """Run regret/counterfactual experiment with multiple seeds."""
    standard_optimal = []
    regret_optimal = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        env = TwoDoorEnv()

        # Standard agent
        std_agent = StandardBanditAgent()
        optimal_choices = 0
        for _ in range(n_trials):
            action = std_agent.select_action()
            state, reward, done, ctx = env.step(action)
            std_agent.update(state, action, reward, state, done, ctx)
            # Door 1 has higher mean (0.7 vs 0.3)
            if action == 1:
                optimal_choices += 1
            if done:
                env = TwoDoorEnv()  # Reset for more trials
        standard_optimal.append(optimal_choices / n_trials)

        # Regret agent
        np.random.seed(seed)
        env = TwoDoorEnv()  # Reset env
        reg_agent = RegretEDAgent()
        optimal_choices = 0
        for _ in range(n_trials):
            action = reg_agent.select_action()
            state, reward, done, ctx = env.step(action)
            reg_agent.update(state, action, reward, state, done, ctx)
            if action == 1:
                optimal_choices += 1
            if done:
                env = TwoDoorEnv()
        regret_optimal.append(optimal_choices / n_trials)

    return compute_statistics(standard_optimal, regret_optimal,
                              "Standard", "Regret ED")


def run_integration_experiment(n_seeds: int = 50, n_train: int = 500,
                                n_eval: int = 100) -> Dict:
    """Run multi-channel integration experiment with multiple seeds."""
    fear_risky = []
    anger_risky = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        env = IntegrationGridWorld()

        # Fear-dominant agent
        fear_agent = FearDominantAgent(env.n_states, env.n_actions)
        for _ in range(n_train):
            state = env.reset()
            for _ in range(100):
                action = fear_agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                fear_agent.update(state, action, reward, next_state, done, ctx)
                if done:
                    break
                state = next_state

        fear_agent.epsilon = 0.05
        risky_count = 0
        for _ in range(n_eval):
            state = env.reset()
            for _ in range(100):
                action = fear_agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                if done and 'risky' in env.visited_goals:
                    risky_count += 1
                if done:
                    break
                state = next_state
        fear_risky.append(risky_count / n_eval)

        # Anger-dominant agent
        np.random.seed(seed)
        anger_agent = AngerDominantAgent(env.n_states, env.n_actions)
        for _ in range(n_train):
            state = env.reset()
            for _ in range(100):
                action = anger_agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                anger_agent.update(state, action, reward, next_state, done, ctx)
                if done:
                    break
                state = next_state

        anger_agent.epsilon = 0.05
        risky_count = 0
        for _ in range(n_eval):
            state = env.reset()
            for _ in range(100):
                action = anger_agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                if done and 'risky' in env.visited_goals:
                    risky_count += 1
                if done:
                    break
                state = next_state
        anger_risky.append(risky_count / n_eval)

    return compute_statistics(fear_risky, anger_risky,
                              "Fear-dominant", "Anger-dominant")


def run_transfer_experiment(n_seeds: int = 50, n_train: int = 200,
                            n_eval: int = 25) -> Dict:
    """Run transfer/generalization experiment with multiple seeds."""
    no_transfer_hits = []
    emotional_transfer_hits = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        train_env = TrainingGridWorld()
        test_env = NovelThreatGridWorld()

        # No transfer agent
        no_trans = NoTransferAgent(train_env.n_states, train_env.n_actions)
        for _ in range(n_train):
            state = train_env.reset()
            for _ in range(100):
                action = no_trans.select_action(state)
                next_state, reward, done, ctx = train_env.step(action)
                no_trans.update(state, action, reward, next_state, done, ctx)
                if done:
                    break
                state = next_state

        no_trans.epsilon = 0.05
        hits = []
        for _ in range(n_eval):
            state = test_env.reset()
            episode_hits = 0
            for _ in range(100):
                action = no_trans.select_action(state)
                next_state, reward, done, ctx = test_env.step(action)
                if reward < -0.2:
                    episode_hits += 1
                if done:
                    break
                state = next_state
            hits.append(episode_hits)
        no_transfer_hits.append(np.mean(hits))

        # Emotional transfer agent
        np.random.seed(seed)
        emo_trans = EmotionalTransferAgent(train_env.n_states, train_env.n_actions)
        for _ in range(n_train):
            state = train_env.reset()
            for _ in range(100):
                action = emo_trans.select_action(state)
                next_state, reward, done, ctx = train_env.step(action)
                emo_trans.update(state, action, reward, next_state, done, ctx)
                if done:
                    break
                state = next_state

        emo_trans.epsilon = 0.05
        hits = []
        for _ in range(n_eval):
            state = test_env.reset()
            episode_hits = 0
            for _ in range(100):
                action = emo_trans.select_action(state)
                next_state, reward, done, ctx = test_env.step(action)
                if reward < -0.2:
                    episode_hits += 1
                if done:
                    break
                state = next_state
            hits.append(episode_hits)
        emotional_transfer_hits.append(np.mean(hits))

    return compute_statistics(no_transfer_hits, emotional_transfer_hits,
                              "No Transfer", "Emotional Transfer")


def print_results(name: str, results: Dict):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print('='*60)

    for key in results:
        if isinstance(results[key], dict) and 'mean' in results[key]:
            d = results[key]
            print(f"\n{key}:")
            print(f"  Mean: {d['mean']:.4f}")
            print(f"  SD: {d['std']:.4f}")
            print(f"  95% CI: [{d['ci_95'][0]:.4f}, {d['ci_95'][1]:.4f}]")
            print(f"  N: {d['n']}")

    print(f"\nStatistical Test:")
    print(f"  t-statistic: {results['t_statistic']:.4f}")
    print(f"  p-value: {results['p_value']:.6f}")
    print(f"  Cohen's d: {results['cohens_d']:.4f} ({results['effect_interpretation']})")
    print(f"  Significant (p<0.05): {'YES' if results['significant'] else 'NO'}")


def main():
    print("="*70)
    print("EXPERIMENT 12: STATISTICAL VALIDATION")
    print("="*70)
    print("\nRunning experiments with N=50 seeds each...")
    print("This may take several minutes.\n")

    # Run all validation experiments
    print("Running Fear experiment...")
    fear_results = run_fear_experiment(n_seeds=50)
    print_results("FEAR AVOIDANCE (Min Threat Distance)", fear_results)

    print("\nRunning Anger experiment...")
    anger_results = run_anger_experiment(n_seeds=50)
    print_results("ANGER/FRUSTRATION (Wall Hits in Early Learning)", anger_results)

    print("\nRunning Regret experiment...")
    regret_results = run_regret_experiment(n_seeds=50)
    print_results("REGRET (Optimal Choice Rate)", regret_results)

    print("\nRunning Integration experiment...")
    integration_results = run_integration_experiment(n_seeds=50)
    print_results("MULTI-CHANNEL (Risky Goal Rate)", integration_results)

    print("\nRunning Transfer experiment...")
    transfer_results = run_transfer_experiment(n_seeds=50)
    print_results("TRANSFER (Threat Hits on Novel Environment)", transfer_results)

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n{'Experiment':<25} {'Cohens d':<12} {'p-value':<12} {'Effect':<12} {'Sig?':<6}")
    print("-"*70)

    all_results = [
        ("Fear Avoidance", fear_results),
        ("Anger Persistence", anger_results),
        ("Regret Learning", regret_results),
        ("Multi-channel", integration_results),
        ("Transfer", transfer_results)
    ]

    for name, r in all_results:
        sig = "YES" if r['significant'] else "NO"
        print(f"{name:<25} {r['cohens_d']:<12.3f} {r['p_value']:<12.6f} "
              f"{r['effect_interpretation']:<12} {sig:<6}")

    # Overall assessment
    print("\n" + "="*70)
    print("OVERALL ASSESSMENT")
    print("="*70)

    significant_count = sum(1 for _, r in all_results if r['significant'])
    large_effect_count = sum(1 for _, r in all_results if abs(r['cohens_d']) >= 0.8)

    print(f"\nSignificant results (p<0.05): {significant_count}/5")
    print(f"Large effects (|d|>=0.8): {large_effect_count}/5")

    if significant_count >= 4 and large_effect_count >= 3:
        print("\n✓ STRONG statistical support for Emotional ED claims")
    elif significant_count >= 3:
        print("\n~ MODERATE statistical support - some effects may need larger N")
    else:
        print("\n✗ WEAK statistical support - results may not replicate")


if __name__ == "__main__":
    main()
