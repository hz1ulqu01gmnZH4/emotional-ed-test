"""Experiment 22: CVaR Fear with Distributional RL

Tests CVaR-based fear which uses principled risk-sensitivity:
- Maintains distribution of returns (quantiles)
- Fear level controls CVaR alpha (risk level)
- High fear → optimize worst-case outcomes

Hypothesis: CVaRFearAgent should avoid threats MORE than RiskNeutralAgent
because it focuses on worst-case outcomes when fearful.

Comparison:
- CVaR fear (principled risk-sensitivity)
- Risk-neutral (standard expected value)
- Original heuristic fear (from Exp 1)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ak/emotional-ed-test')

from scipy import stats
from dataclasses import dataclass
from typing import List, Tuple

from agents_v2.agents_cvar_fear import CVaRFearAgent, RiskNeutralAgent, FearContext


class StochasticThreatGridWorld:
    """Gridworld with STOCHASTIC threat outcomes.

    This is where CVaR should excel over expected value:
    - Threat has variable damage (sometimes mild, sometimes severe)
    - Risk-neutral agent cares only about average
    - CVaR agent focuses on worst cases
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 6, threat_severity_std: float = 0.3):
        self.size = size
        self.threat_pos = np.array([2, 2])
        self.goal_pos = np.array([5, 5])
        self.threat_severity_std = threat_severity_std  # Variance in damage
        self.reset()

    def reset(self) -> int:
        self.agent_pos = np.array([0, 0])
        self.step_count = 0
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def _get_direction(self, from_pos: np.ndarray, to_pos: np.ndarray) -> Tuple[int, int]:
        diff = to_pos - from_pos
        return (int(diff[0]), int(diff[1]))

    def step(self, action: int) -> Tuple[int, float, bool, FearContext]:
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        threat_dist = self._distance(self.agent_pos, self.threat_pos)
        goal_dist = self._distance(self.agent_pos, self.goal_pos)

        reward = -0.01  # Step cost
        was_harmed = False

        # STOCHASTIC threat damage
        if threat_dist < 1.0:
            # Damage varies: sometimes mild (-0.2), sometimes severe (-0.8)
            base_damage = 0.5
            damage_variation = np.random.normal(0, self.threat_severity_std)
            actual_damage = base_damage + damage_variation
            actual_damage = np.clip(actual_damage, 0.1, 1.0)  # Clamp
            reward -= actual_damage
            was_harmed = True

        # Goal
        done = np.array_equal(self.agent_pos, self.goal_pos) or self.step_count >= 100
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 1.0

        context = FearContext(
            threat_distance=threat_dist,
            goal_distance=goal_dist,
            threat_direction=self._get_direction(self.agent_pos, self.threat_pos),
            near_threat=threat_dist < 2.0,
            was_harmed=was_harmed
        )

        return self._pos_to_state(self.agent_pos), reward, done, context

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4


@dataclass
class CVaRResult:
    cvar_rewards: List[float]
    neutral_rewards: List[float]
    cvar_threat_hits: List[int]
    neutral_threat_hits: List[int]
    cvar_goals: List[int]
    neutral_goals: List[int]


def run_single_trial(n_episodes: int = 100, seed: int = None) -> Tuple[float, float, int, int, int, int]:
    """Run single trial comparing CVaR vs risk-neutral."""
    if seed is not None:
        np.random.seed(seed)

    env = StochasticThreatGridWorld(size=6, threat_severity_std=0.3)

    # CVaR fear agent
    # FIX: base_alpha=1.0 means risk-neutral when calm
    # Fear reduces alpha toward min_alpha=0.1 (very risk-averse)
    # This allows proper exploration when safe, caution when threatened
    cvar_agent = CVaRFearAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        n_quantiles=21,
        base_alpha=1.0,  # Risk-neutral when calm (was 0.5)
        min_alpha=0.1    # Very risk-averse when maximally fearful
    )

    # Risk-neutral agent
    neutral_agent = RiskNeutralAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        n_quantiles=21
    )

    # --- Run CVaR agent ---
    cvar_total_reward = 0
    cvar_hits = 0
    cvar_goals = 0

    for ep in range(n_episodes):
        state = env.reset()
        cvar_agent.reset_episode()
        done = False

        while not done:
            # Get context
            threat_dist = env._distance(env.agent_pos, env.threat_pos)
            context = FearContext(
                threat_distance=threat_dist,
                goal_distance=env._distance(env.agent_pos, env.goal_pos),
                threat_direction=env._get_direction(env.agent_pos, env.threat_pos),
                near_threat=threat_dist < 2.0
            )

            action = cvar_agent.select_action(state, context)
            next_state, reward, done, next_context = env.step(action)

            # Pass next_context to enable proper fear-based bootstrapping
            cvar_agent.update(state, action, reward, next_state, done, context, next_context)

            cvar_total_reward += reward

            if next_context.was_harmed:
                cvar_hits += 1

            state = next_state

        if np.array_equal(env.agent_pos, env.goal_pos):
            cvar_goals += 1

    # Reset for neutral agent
    if seed is not None:
        np.random.seed(seed + 10000)

    env = StochasticThreatGridWorld(size=6, threat_severity_std=0.3)

    # --- Run risk-neutral agent ---
    neutral_total_reward = 0
    neutral_hits = 0
    neutral_goals = 0

    for ep in range(n_episodes):
        state = env.reset()
        neutral_agent.reset_episode()
        done = False

        while not done:
            context = FearContext(
                threat_distance=env._distance(env.agent_pos, env.threat_pos),
                goal_distance=env._distance(env.agent_pos, env.goal_pos),
                threat_direction=env._get_direction(env.agent_pos, env.threat_pos),
                near_threat=env._distance(env.agent_pos, env.threat_pos) < 2.0
            )

            action = neutral_agent.select_action(state, context)
            next_state, reward, done, next_context = env.step(action)

            neutral_agent.update(state, action, reward, next_state, done, context)

            neutral_total_reward += reward

            if next_context.was_harmed:
                neutral_hits += 1

            state = next_state

        if np.array_equal(env.agent_pos, env.goal_pos):
            neutral_goals += 1

    return (
        cvar_total_reward / n_episodes,
        neutral_total_reward / n_episodes,
        cvar_hits,
        neutral_hits,
        cvar_goals,
        neutral_goals
    )


def run_experiment(n_trials: int = 50, n_episodes: int = 100) -> CVaRResult:
    """Run full CVaR experiment."""
    print(f"Running CVaR Fear Experiment: {n_trials} trials, {n_episodes} episodes each")
    print("=" * 60)
    print("Environment: Stochastic threat (damage varies)")
    print("Hypothesis: CVaR should avoid threats more (risk-averse)")
    print("=" * 60)

    cvar_rewards = []
    neutral_rewards = []
    cvar_hits = []
    neutral_hits = []
    cvar_goals = []
    neutral_goals = []

    for trial in range(n_trials):
        c_r, n_r, c_h, n_h, c_g, n_g = run_single_trial(
            n_episodes=n_episodes,
            seed=trial * 1000
        )

        cvar_rewards.append(c_r)
        neutral_rewards.append(n_r)
        cvar_hits.append(c_h)
        neutral_hits.append(n_h)
        cvar_goals.append(c_g)
        neutral_goals.append(n_g)

        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{n_trials} complete")

    return CVaRResult(
        cvar_rewards=cvar_rewards,
        neutral_rewards=neutral_rewards,
        cvar_threat_hits=cvar_hits,
        neutral_threat_hits=neutral_hits,
        cvar_goals=cvar_goals,
        neutral_goals=neutral_goals
    )


def analyze_results(result: CVaRResult):
    """Statistical analysis."""
    print("\n" + "=" * 60)
    print("RESULTS: CVaR Fear (Principled Risk-Sensitivity)")
    print("=" * 60)

    # Threat hits (primary measure)
    c_hits = np.array(result.cvar_threat_hits)
    n_hits = np.array(result.neutral_threat_hits)

    print("\n--- Threat Hits ---")
    print(f"CVaR Fear: {np.mean(c_hits):.2f} ± {np.std(c_hits):.2f}")
    print(f"Risk-neutral: {np.mean(n_hits):.2f} ± {np.std(n_hits):.2f}")

    t_stat, p_value = stats.ttest_ind(c_hits, n_hits)
    pooled_std = np.sqrt((np.std(c_hits)**2 + np.std(n_hits)**2) / 2)
    cohens_d = (np.mean(c_hits) - np.mean(n_hits)) / pooled_std if pooled_std > 0 else 0

    print(f"\nt-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Cohen's d: {cohens_d:.3f}")

    if cohens_d < 0:
        print("Direction: CORRECT (CVaR has fewer hits = more risk-averse)")
    else:
        print("Direction: Unexpected (CVaR has more hits)")

    # Rewards
    c_r = np.array(result.cvar_rewards)
    n_r = np.array(result.neutral_rewards)

    print("\n--- Average Rewards ---")
    print(f"CVaR Fear: {np.mean(c_r):.3f} ± {np.std(c_r):.3f}")
    print(f"Risk-neutral: {np.mean(n_r):.3f} ± {np.std(n_r):.3f}")

    t_stat_r, p_value_r = stats.ttest_ind(c_r, n_r)
    pooled_std_r = np.sqrt((np.std(c_r)**2 + np.std(n_r)**2) / 2)
    cohens_d_r = (np.mean(c_r) - np.mean(n_r)) / pooled_std_r if pooled_std_r > 0 else 0

    print(f"Cohen's d (reward): {cohens_d_r:.3f}")

    # Goals
    c_g = np.array(result.cvar_goals)
    n_g = np.array(result.neutral_goals)

    print("\n--- Goal Completions ---")
    print(f"CVaR Fear: {np.mean(c_g):.2f} ± {np.std(c_g):.2f}")
    print(f"Risk-neutral: {np.mean(n_g):.2f} ± {np.std(n_g):.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if p_value < 0.05 and cohens_d < 0:
        print("✓ SUCCESS: CVaR Fear is more risk-averse (fewer threat hits)")
        print(f"  Effect size: d={abs(cohens_d):.2f}")
    elif p_value < 0.05 and cohens_d > 0:
        print("✗ UNEXPECTED: CVaR has more hits (less risk-averse)")
    else:
        print("~ INCONCLUSIVE: No significant difference in risk-aversion")

    # Note on reward trade-off
    if cohens_d < 0 and cohens_d_r < 0:
        print("\nNote: Lower reward is expected for risk-averse behavior")
        print("      (Avoiding threats means longer paths)")

    return {
        'hits_cohens_d': cohens_d,
        'hits_p_value': p_value,
        'reward_cohens_d': cohens_d_r,
        'reward_p_value': p_value_r
    }


if __name__ == '__main__':
    result = run_experiment(n_trials=50, n_episodes=100)
    stats_result = analyze_results(result)
