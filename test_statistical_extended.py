"""Experiment 12b: Extended Statistical Validation.

Extends statistical validation to remaining experiments:
- Exp 4: Grief/Attachment
- Exp 5: Approach-Avoidance Conflict
- Exp 6: Emotion Regulation
- Exp 8: Temporal Dynamics (Phasic vs Tonic)
- Exp 9: Disgust/Contamination
- Exp 10: Wanting/Liking Dissociation

Reports: Mean±SD, 95% CI, p-value, Cohen's d for each.
"""

import numpy as np
from typing import Dict, List, Tuple

# Import from existing experiments - using correct class names
from gridworld_grief import AttachmentGridWorld
from agents_grief import StandardQLearner as GriefStandardAgent, GriefEDAgent

from gridworld_conflict import ApproachAvoidanceGridWorld
from agents_conflict import (StandardQLearner as ConflictStandard,
                             FearDominantAgent, ApproachDominantAgent, BalancedConflictAgent)

from gridworld_regulation import RegulationGridWorld
from agents_regulation import (StandardQLearner as RegulationStandard,
                               UnregulatedFearAgent, RegulatedFearAgent)

from gridworld_temporal import TemporalGridWorld
from agents_temporal import (StandardQLearner as TemporalStandard,
                             PhasicOnlyAgent, TonicMoodAgent, IntegratedTemporalAgent)

from gridworld_disgust import DisgustGridWorld
from agents_disgust import (StandardQLearner as DisgustStandard,
                            FearOnlyAgent, DisgustOnlyAgent, IntegratedFearDisgustAgent)

from gridworld_wanting import WantingLikingGridWorld
from agents_wanting import (StandardQLearner as WantingStandard,
                            WantingDominantAgent, LikingDominantAgent, AddictionModelAgent)


def permutation_test(arr1: np.ndarray, arr2: np.ndarray, n_perm: int = 1000) -> float:
    """Permutation test for difference in means."""
    n1 = len(arr1)
    combined = np.concatenate([arr1, arr2])
    observed_diff = abs(np.mean(arr1) - np.mean(arr2))

    count = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_diff = abs(np.mean(combined[:n1]) - np.mean(combined[n1:]))
        if perm_diff >= observed_diff:
            count += 1

    return max(count / n_perm, 1/n_perm)  # Avoid p=0


def ci_95(arr: np.ndarray) -> Tuple[float, float]:
    """Compute 95% CI."""
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    t_val = 1.96 if n > 30 else 2.0
    margin = t_val * std / np.sqrt(n)
    return (mean - margin, mean + margin)


def compute_statistics(data1: List[float], data2: List[float],
                       name1: str = "Control", name2: str = "Treatment") -> Dict:
    """Compute comprehensive statistics for two groups."""
    arr1 = np.array(data1)
    arr2 = np.array(data2)

    mean1, mean2 = np.mean(arr1), np.mean(arr2)
    std1, std2 = np.std(arr1, ddof=1), np.std(arr2, ddof=1)
    n1, n2 = len(arr1), len(arr2)

    ci1 = ci_95(arr1)
    ci2 = ci_95(arr2)

    p_value = permutation_test(arr1, arr2)

    # Cohen's d
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0

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
        "p_value": p_value,
        "cohens_d": cohens_d,
        "effect_interpretation": effect_interp,
        "significant": p_value < 0.05
    }


def run_grief_experiment(n_seeds: int = 50) -> Dict:
    """Exp 4: Grief/Attachment - visits after loss.

    Test in MID-LEARNING: Agent has learned resource location but Q-values not converged.
    This is when yearning should have most effect:
    - Standard agent: Q-values update quickly, stops visiting lost resource
    - Grief agent: Yearning slows negative learning, maintains visits longer

    Key: Only minimal pre-training (5 episodes) so Q-values are still malleable.
    """
    print("  Running Grief experiment...")
    standard_visits = []
    grief_visits = []

    grid_size = 7
    resource_pos = (3, 3)
    loss_step = 30  # Early loss
    n_pretrain = 5  # MINIMAL pre-training - mid-learning state
    eval_epsilon = 0.1  # Keep exploration to see learning dynamics

    for seed in range(n_seeds):
        np.random.seed(seed)

        # Standard agent - minimal pre-training
        env = AttachmentGridWorld(size=grid_size, resource_pos=resource_pos, loss_step=loss_step)
        agent = GriefStandardAgent(n_states=env.n_states, n_actions=env.n_actions,
                                   lr=0.1, epsilon=0.1)

        # Minimal pre-train (just learn resource exists, Q-values not converged)
        for _ in range(n_pretrain):
            state = env.reset()
            for step in range(loss_step):  # Only before loss
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                agent.update(state, action, reward, next_state, done, ctx)
                state = next_state

        # Test episode with loss - count visits AFTER loss
        env = AttachmentGridWorld(size=grid_size, resource_pos=resource_pos, loss_step=loss_step)
        state = env.reset()
        resource_state = env._pos_to_state(env.resource_pos)
        visits_after_loss = 0

        for step in range(120):
            action = agent.select_action(state)
            next_state, reward, done, ctx = env.step(action)
            agent.update(state, action, reward, next_state, done, ctx)

            # Count visits after loss (yearning window = steps 30-90)
            if next_state == resource_state and env.loss_occurred:
                visits_after_loss += 1

            if done:
                break
            state = next_state
        standard_visits.append(visits_after_loss)

        # Grief agent - same minimal pre-training
        np.random.seed(seed)
        env = AttachmentGridWorld(size=grid_size, resource_pos=resource_pos, loss_step=loss_step)
        agent = GriefEDAgent(n_states=env.n_states, n_actions=env.n_actions,
                             lr=0.1, epsilon=0.1, grief_weight=0.8, grid_size=grid_size)

        # Minimal pre-train - attachment builds during this
        for _ in range(n_pretrain):
            state = env.reset()
            agent.reset_episode()  # Preserves attachment
            for step in range(loss_step):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                agent.update(state, action, reward, next_state, done, ctx)
                state = next_state

        # Test episode with loss
        env = AttachmentGridWorld(size=grid_size, resource_pos=resource_pos, loss_step=loss_step)
        state = env.reset()
        resource_state = env._pos_to_state(env.resource_pos)
        visits_after_loss = 0

        for step in range(120):
            action = agent.select_action(state)
            next_state, reward, done, ctx = env.step(action)
            agent.update(state, action, reward, next_state, done, ctx)

            if next_state == resource_state and env.loss_occurred:
                visits_after_loss += 1

            if done:
                break
            state = next_state
        grief_visits.append(visits_after_loss)

    return compute_statistics(standard_visits, grief_visits, "Standard", "Grief ED")


def run_conflict_experiment(n_seeds: int = 50, n_train: int = 300, n_eval: int = 100) -> Dict:
    """Exp 5: Approach-Avoidance Conflict - risky vs safe choice."""
    print("  Running Conflict experiment...")
    fear_risky = []
    approach_risky = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        env = ApproachAvoidanceGridWorld()

        # Fear-dominant agent
        agent = FearDominantAgent(env.n_states, env.n_actions)
        for _ in range(n_train):
            state = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                agent.update(state, action, reward, next_state, done, ctx)
                if done:
                    break
                state = next_state

        agent.epsilon = 0.05
        risky_count = 0
        for _ in range(n_eval):
            state = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                if done:
                    if env.risky_collected:
                        risky_count += 1
                    break
                state = next_state
        fear_risky.append(risky_count / n_eval)

        # Approach-dominant agent
        np.random.seed(seed)
        env = ApproachAvoidanceGridWorld()
        agent = ApproachDominantAgent(env.n_states, env.n_actions)
        for _ in range(n_train):
            state = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                agent.update(state, action, reward, next_state, done, ctx)
                if done:
                    break
                state = next_state

        agent.epsilon = 0.05
        risky_count = 0
        for _ in range(n_eval):
            state = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                if done:
                    if env.risky_collected:
                        risky_count += 1
                    break
                state = next_state
        approach_risky.append(risky_count / n_eval)

    return compute_statistics(fear_risky, approach_risky, "Fear-dominant", "Approach-dominant")


def run_regulation_experiment(n_seeds: int = 50, n_train: int = 500, n_eval: int = 100) -> Dict:
    """Exp 6: Emotion Regulation - fake bonus collection."""
    print("  Running Regulation experiment...")
    unregulated_reward = []
    regulated_reward = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        env = RegulationGridWorld()

        # Unregulated fear agent
        agent = UnregulatedFearAgent(env.n_states, env.n_actions)
        for _ in range(n_train):
            state = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                agent.update(state, action, reward, next_state, done, ctx)
                if done:
                    break
                state = next_state

        agent.epsilon = 0.05
        total_reward = 0
        for _ in range(n_eval):
            state = env.reset()
            ep_reward = 0
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                ep_reward += reward
                if done:
                    break
                state = next_state
            total_reward += ep_reward
        unregulated_reward.append(total_reward / n_eval)

        # Regulated fear agent
        np.random.seed(seed)
        env = RegulationGridWorld()
        agent = RegulatedFearAgent(env.n_states, env.n_actions)
        for _ in range(n_train):
            state = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                agent.update(state, action, reward, next_state, done, ctx)
                if done:
                    break
                state = next_state

        agent.epsilon = 0.05
        total_reward = 0
        for _ in range(n_eval):
            state = env.reset()
            ep_reward = 0
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                ep_reward += reward
                if done:
                    break
                state = next_state
            total_reward += ep_reward
        regulated_reward.append(total_reward / n_eval)

    return compute_statistics(unregulated_reward, regulated_reward, "Unregulated", "Regulated")


def run_temporal_experiment(n_seeds: int = 50) -> Dict:
    """Exp 8: Temporal Dynamics - mood during negative phase."""
    print("  Running Temporal experiment...")
    phasic_mood = []
    tonic_mood = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        env = TemporalGridWorld()

        # Phasic only agent
        agent = PhasicOnlyAgent(env.n_states, env.n_actions)
        state = env.reset()

        # Run through phases and record mood during negative phase
        mood_readings = []
        for step in range(400):  # Enough to hit negative phase
            action = agent.select_action(state)
            next_state, reward, done, ctx = env.step(action)
            agent.update(state, action, reward, next_state, done, ctx)

            # Record mood during negative phase (steps 100-200)
            if 100 <= step < 200:
                if hasattr(agent, 'tonic_mood'):
                    mood_readings.append(agent.tonic_mood)
                elif hasattr(agent, 'mood'):
                    mood_readings.append(agent.mood)
                else:
                    mood_readings.append(0.0)

            if done:
                state = env.reset()
            else:
                state = next_state

        assert mood_readings, f"BUG: No mood readings collected for phasic agent seed {seed}"
        phasic_mood.append(np.mean(mood_readings))

        # Tonic mood agent
        np.random.seed(seed)
        env = TemporalGridWorld()
        agent = TonicMoodAgent(env.n_states, env.n_actions)
        state = env.reset()

        mood_readings = []
        for step in range(400):
            action = agent.select_action(state)
            next_state, reward, done, ctx = env.step(action)
            agent.update(state, action, reward, next_state, done, ctx)

            if 100 <= step < 200:
                if hasattr(agent, 'tonic_mood'):
                    mood_readings.append(agent.tonic_mood)
                elif hasattr(agent, 'mood'):
                    mood_readings.append(agent.mood)
                else:
                    mood_readings.append(0.0)

            if done:
                state = env.reset()
            else:
                state = next_state

        assert mood_readings, f"BUG: No mood readings collected for tonic agent seed {seed}"
        tonic_mood.append(np.mean(mood_readings))

    return compute_statistics(phasic_mood, tonic_mood, "Phasic Only", "Tonic Mood")


def run_disgust_experiment(n_seeds: int = 50, n_train: int = 300, n_eval: int = 100) -> Dict:
    """Exp 9: Disgust - contaminant approach rate (fear vs disgust)."""
    print("  Running Disgust experiment...")
    fear_contam_approach = []
    disgust_contam_approach = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        env = DisgustGridWorld()

        # Fear-only agent
        agent = FearOnlyAgent(env.n_states, env.n_actions)
        for _ in range(n_train):
            state = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                agent.update(state, action, reward, next_state, done, ctx)
                if done:
                    break
                state = next_state

        agent.epsilon = 0.05
        contam_approaches = 0
        for _ in range(n_eval):
            state = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                # Count contamination touches (actual contamination exposure)
                if ctx.touched_contaminant:
                    contam_approaches += 1
                if done:
                    break
                state = next_state
        fear_contam_approach.append(contam_approaches / n_eval)

        # Disgust-only agent
        np.random.seed(seed)
        env = DisgustGridWorld()
        agent = DisgustOnlyAgent(env.n_states, env.n_actions)
        for _ in range(n_train):
            state = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                agent.update(state, action, reward, next_state, done, ctx)
                if done:
                    break
                state = next_state

        agent.epsilon = 0.05
        contam_approaches = 0
        for _ in range(n_eval):
            state = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                # Count contamination touches (actual contamination exposure)
                if ctx.touched_contaminant:
                    contam_approaches += 1
                if done:
                    break
                state = next_state
        disgust_contam_approach.append(contam_approaches / n_eval)

    return compute_statistics(fear_contam_approach, disgust_contam_approach, "Fear Only", "Disgust Only")


def run_wanting_experiment(n_seeds: int = 50, n_train: int = 300, n_eval: int = 100) -> Dict:
    """Exp 10: Wanting/Liking Dissociation - high-wanting vs high-liking preference."""
    print("  Running Wanting/Liking experiment...")
    wanting_high_wanting_pref = []
    liking_high_wanting_pref = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        env = WantingLikingGridWorld()

        # Wanting-dominant agent - should prefer high-wanting reward
        agent = WantingDominantAgent(env.n_states, env.n_actions)
        for _ in range(n_train):
            state = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                agent.update(state, action, reward, next_state, done, ctx)
                if done:
                    break
                state = next_state

        agent.epsilon = 0.05
        high_wanting_choices = 0
        total_goals = 0
        for _ in range(n_eval):
            state = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                if done:
                    total_goals += 1
                    # Check which reward was collected
                    if env.collected['wanting']:
                        high_wanting_choices += 1
                    break
                state = next_state
        wanting_high_wanting_pref.append(high_wanting_choices / max(total_goals, 1))

        # Liking-dominant agent - should prefer high-liking reward
        np.random.seed(seed)
        env = WantingLikingGridWorld()
        agent = LikingDominantAgent(env.n_states, env.n_actions)
        for _ in range(n_train):
            state = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                agent.update(state, action, reward, next_state, done, ctx)
                if done:
                    break
                state = next_state

        agent.epsilon = 0.05
        high_wanting_choices = 0
        total_goals = 0
        for _ in range(n_eval):
            state = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                if done:
                    total_goals += 1
                    # Check which reward was collected
                    if env.collected['wanting']:
                        high_wanting_choices += 1
                    break
                state = next_state
        liking_high_wanting_pref.append(high_wanting_choices / max(total_goals, 1))

    return compute_statistics(wanting_high_wanting_pref, liking_high_wanting_pref,
                              "Wanting-dominant", "Liking-dominant")


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
    print(f"  p-value: {results['p_value']:.6f}")
    print(f"  Cohen's d: {results['cohens_d']:.4f} ({results['effect_interpretation']})")
    print(f"  Significant (p<0.05): {'YES' if results['significant'] else 'NO'}")


def main():
    print("="*70)
    print("EXPERIMENT 12b: EXTENDED STATISTICAL VALIDATION")
    print("="*70)
    print("\nRunning experiments with N=50 seeds each...")
    print("This may take several minutes.\n")

    all_results = []

    # Run all validation experiments
    print("Running Grief experiment (Exp 4)...")
    grief_results = run_grief_experiment(n_seeds=50)
    print_results("EXP 4: GRIEF (Visits After Loss)", grief_results)
    all_results.append(("Grief (visits after loss)", grief_results))

    print("\nRunning Conflict experiment (Exp 5)...")
    conflict_results = run_conflict_experiment(n_seeds=50)
    print_results("EXP 5: CONFLICT (Risky Choice Rate)", conflict_results)
    all_results.append(("Conflict (risky rate)", conflict_results))

    print("\nRunning Regulation experiment (Exp 6)...")
    regulation_results = run_regulation_experiment(n_seeds=50)
    print_results("EXP 6: REGULATION (Mean Episode Reward)", regulation_results)
    all_results.append(("Regulation (reward)", regulation_results))

    print("\nRunning Temporal experiment (Exp 8)...")
    temporal_results = run_temporal_experiment(n_seeds=50)
    print_results("EXP 8: TEMPORAL (Mood During Negative Phase)", temporal_results)
    all_results.append(("Temporal (negative mood)", temporal_results))

    print("\nRunning Disgust experiment (Exp 9)...")
    disgust_results = run_disgust_experiment(n_seeds=50)
    print_results("EXP 9: DISGUST (Contamination Approaches)", disgust_results)
    all_results.append(("Disgust (contamination)", disgust_results))

    print("\nRunning Wanting/Liking experiment (Exp 10)...")
    wanting_results = run_wanting_experiment(n_seeds=50)
    print_results("EXP 10: WANTING/LIKING (High-Wanting Preference)", wanting_results)
    all_results.append(("Wanting/Liking (preference)", wanting_results))

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n{'Experiment':<30} {'Cohens d':<12} {'p-value':<12} {'Effect':<12} {'Sig?':<6}")
    print("-"*70)

    for name, r in all_results:
        sig = "YES" if r['significant'] else "NO"
        print(f"{name:<30} {r['cohens_d']:<12.3f} {r['p_value']:<12.6f} "
              f"{r['effect_interpretation']:<12} {sig:<6}")

    # Overall assessment
    print("\n" + "="*70)
    print("OVERALL ASSESSMENT")
    print("="*70)

    significant_count = sum(1 for _, r in all_results if r['significant'])
    large_effect_count = sum(1 for _, r in all_results if abs(r['cohens_d']) >= 0.8)

    print(f"\nSignificant results (p<0.05): {significant_count}/6")
    print(f"Large effects (|d|>=0.8): {large_effect_count}/6")

    if significant_count >= 5:
        print("\n[OK] STRONG statistical support for these Emotional ED claims")
    elif significant_count >= 3:
        print("\n[~] MODERATE statistical support - some effects validated")
    else:
        print("\n[X] WEAK statistical support - effects may not replicate")

    # Combined with original Exp 12 results
    print("\n" + "="*70)
    print("COMBINED RESULTS (All 11 experiments)")
    print("="*70)
    print("\nPreviously validated (Exp 12):")
    print("  - Fear: p=0.013, d=1.09 [OK]")
    print("  - Anger: p=0.001, d=0.75 [OK]")
    print("  - Regret: p=0.058, d=1.06 (marginal)")
    print("  - Integration: p=0.001, d=1.56 [OK]")
    print("  - Transfer: p=0.32, d=0.12 (NS)")
    print(f"\nNewly validated: {significant_count}/6 significant")
    print(f"\nTotal significant across all 11: {3 + significant_count}/11")


if __name__ == "__main__":
    main()
