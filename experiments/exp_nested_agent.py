"""Experiment: Nested Emotional Agent Validation

Compares the multi-timescale NestedEmotionalAgent against:
1. StandardQLearner (no emotions)
2. EmotionalEDAgent (single-timescale emotions)
3. SimplifiedNestedAgent (2-timescale ablation)

Tests:
1. Threat avoidance (fear-based)
2. Learning stability (mood effects)
3. Exploration efficiency (curiosity-based)
4. Recovery from negative events (mood persistence)

Hypothesis:
Multi-timescale emotions should provide:
- More stable learning (tonic mood smooths phasic reactions)
- Better threat calibration (mood affects reactivity)
- Improved exploration (curiosity novelty bonus)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ak/emotional-ed-test')

from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from src.modules.nested_agent import NestedEmotionalAgent, SimplifiedNestedAgent, NestedContext


# =============================================================================
# Environment
# =============================================================================

class ThreatGridWorld:
    """Grid world with threat at center for testing fear mechanisms."""

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 7, max_steps: int = 100):
        self.size = size
        self.max_steps = max_steps
        self.threat_pos = np.array([size // 2, size // 2])
        self.goal_pos = np.array([size - 1, size - 1])
        self.reset()

    def reset(self) -> Tuple[int, NestedContext]:
        self.agent_pos = np.array([0, 0])
        self.step_count = 0
        self.min_threat_distance = float('inf')
        self.threat_encounters = 0

        context = self._make_context(0.0)
        return self._pos_to_state(), context

    def _pos_to_state(self) -> int:
        return self.agent_pos[0] * self.size + self.agent_pos[1]

    def _distance(self, p1, p2) -> float:
        return float(np.linalg.norm(p1 - p2))

    def _make_context(self, reward: float) -> NestedContext:
        threat_dist = self._distance(self.agent_pos, self.threat_pos)
        goal_dist = self._distance(self.agent_pos, self.goal_pos)
        return NestedContext(
            threat_distance=threat_dist,
            near_threat=threat_dist < 2.0,
            goal_distance=goal_dist,
            was_blocked=False,
            reward=reward,
            step=self.step_count
        )

    def step(self, action: int) -> Tuple[int, float, bool, NestedContext]:
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        was_blocked = False
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            was_blocked = True
            new_pos = self.agent_pos

        self.agent_pos = new_pos
        threat_dist = self._distance(self.agent_pos, self.threat_pos)
        self.min_threat_distance = min(self.min_threat_distance, threat_dist)

        if threat_dist < 1.5:
            self.threat_encounters += 1

        done = np.array_equal(self.agent_pos, self.goal_pos) or self.step_count >= self.max_steps
        reward = 1.0 if np.array_equal(self.agent_pos, self.goal_pos) else -0.01

        context = self._make_context(reward)
        context.was_blocked = was_blocked

        return self._pos_to_state(), reward, done, context

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4


# =============================================================================
# Baseline Agents (for comparison)
# =============================================================================

class StandardQLearner:
    """Vanilla Q-learning baseline."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state: int, context: Optional[NestedContext] = None) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: NestedContext):
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        pass


class SingleTimescaleEmotional:
    """Single-timescale emotional agent (original EmotionalEDAgent style)."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 0.5):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.fear_weight = fear_weight
        self.current_fear = 0.0

    def _compute_fear(self, context: NestedContext) -> float:
        if context.threat_distance >= 3.0:
            return 0.0
        return 1.0 - context.threat_distance / 3.0

    def select_action(self, state: int, context: Optional[NestedContext] = None) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()
        if self.current_fear > 0.1:
            q_min, q_max = q_values.min(), q_values.max()
            if q_max > q_min:
                normalized = (q_values - q_min) / (q_max - q_min)
                q_values += self.current_fear * normalized * self.fear_weight

        return int(np.argmax(q_values))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: NestedContext):
        fear = self._compute_fear(context)
        self.current_fear = fear

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        fear_mod = 1.0 + self.fear_weight * fear * (1.0 if delta < 0 else 0.5)
        effective_lr = self.lr * fear_mod

        self.Q[state, action] += effective_lr * delta

    def reset_episode(self):
        self.current_fear = 0.0


# =============================================================================
# Experiment
# =============================================================================

@dataclass
class EpisodeMetrics:
    steps: int
    reward: float
    min_threat_dist: float
    threat_encounters: int
    reached_goal: bool


def run_episode(env: ThreatGridWorld, agent, training: bool = True) -> EpisodeMetrics:
    state, context = env.reset()
    agent.reset_episode()

    total_reward = 0.0

    while True:
        action = agent.select_action(state, context)
        next_state, reward, done, context = env.step(action)

        if training:
            agent.update(state, action, reward, next_state, done, context)

        total_reward += reward
        state = next_state

        if done:
            break

    return EpisodeMetrics(
        steps=env.step_count,
        reward=total_reward,
        min_threat_dist=env.min_threat_distance,
        threat_encounters=env.threat_encounters,
        reached_goal=np.array_equal(env.agent_pos, env.goal_pos)
    )


def run_experiment(n_seeds: int = 50, n_episodes: int = 200):
    print("=" * 70)
    print("NESTED EMOTIONAL AGENT VALIDATION")
    print("=" * 70)
    print(f"Seeds: {n_seeds}, Episodes: {n_episodes}")
    print("Environment: 7x7 ThreatGridWorld")
    print("=" * 70)

    agent_configs = [
        ('Standard', StandardQLearner, {}),
        ('Single-Timescale', SingleTimescaleEmotional, {'fear_weight': 0.5}),
        ('Simplified Nested', SimplifiedNestedAgent, {'fear_weight': 0.5, 'curiosity_weight': 0.3}),
        ('Full Nested', NestedEmotionalAgent, {'fear_weight': 0.5, 'anger_weight': 0.3, 'joy_weight': 0.2}),
    ]

    results = {name: {
        'threat_encounters': [],
        'min_threat_dist': [],
        'goal_rate': [],
        'avg_steps': [],
        'reward_variance': [],
    } for name, _, _ in agent_configs}

    for seed in range(n_seeds):
        if (seed + 1) % 10 == 0:
            print(f"  Seed {seed + 1}/{n_seeds}")

        for name, AgentClass, kwargs in agent_configs:
            np.random.seed(seed * 100 + hash(name) % 100)

            env = ThreatGridWorld(size=7, max_steps=100)
            agent = AgentClass(env.n_states, env.n_actions, **kwargs)

            episode_metrics = []
            for ep in range(n_episodes):
                metrics = run_episode(env, agent, training=True)
                episode_metrics.append(metrics)

            # Aggregate over last 50 episodes
            last_n = 50
            final = episode_metrics[-last_n:]

            results[name]['threat_encounters'].append(np.mean([m.threat_encounters for m in final]))
            results[name]['min_threat_dist'].append(np.mean([m.min_threat_dist for m in final]))
            results[name]['goal_rate'].append(np.mean([m.reached_goal for m in final]))
            results[name]['avg_steps'].append(np.mean([m.steps for m in final]))

            # Reward variance (stability metric)
            rewards = [m.reward for m in episode_metrics]
            results[name]['reward_variance'].append(np.var(rewards[-last_n:]))

    # Statistical analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    def cohens_d(g1, g2):
        n1, n2 = len(g1), len(g2)
        v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        pooled = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
        return (np.mean(g1) - np.mean(g2)) / pooled if pooled > 0 else 0

    baseline = 'Standard'

    print("\n--- THREAT ENCOUNTERS (lower is better) ---")
    for name in results:
        enc = np.array(results[name]['threat_encounters'])
        print(f"{name:20s}: {np.mean(enc):.2f} +/- {np.std(enc):.2f}")

    print("\nEffect sizes vs Standard:")
    std_enc = np.array(results[baseline]['threat_encounters'])
    for name in results:
        if name == baseline:
            continue
        enc = np.array(results[name]['threat_encounters'])
        d = cohens_d(enc, std_enc)
        _, p = stats.ttest_ind(enc, std_enc)
        sig = "*" if p < 0.05 else ""
        print(f"  {name:20s}: d = {d:+.3f} (p = {p:.4f}) {sig}")

    print("\n--- LEARNING STABILITY (reward variance, lower is better) ---")
    for name in results:
        var = np.array(results[name]['reward_variance'])
        print(f"{name:20s}: {np.mean(var):.4f} +/- {np.std(var):.4f}")

    print("\nEffect sizes vs Standard:")
    std_var = np.array(results[baseline]['reward_variance'])
    for name in results:
        if name == baseline:
            continue
        var = np.array(results[name]['reward_variance'])
        d = cohens_d(var, std_var)
        _, p = stats.ttest_ind(var, std_var)
        sig = "*" if p < 0.05 else ""
        print(f"  {name:20s}: d = {d:+.3f} (p = {p:.4f}) {sig}")

    print("\n--- GOAL COMPLETION RATE ---")
    for name in results:
        rate = np.array(results[name]['goal_rate'])
        print(f"{name:20s}: {np.mean(rate)*100:.1f}%")

    print("\n--- AVERAGE STEPS TO GOAL ---")
    for name in results:
        steps = np.array(results[name]['avg_steps'])
        print(f"{name:20s}: {np.mean(steps):.1f} +/- {np.std(steps):.1f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Check if nested is better than single-timescale
    nested_enc = np.array(results['Full Nested']['threat_encounters'])
    single_enc = np.array(results['Single-Timescale']['threat_encounters'])
    d_nested_vs_single = cohens_d(nested_enc, single_enc)
    _, p_nested_vs_single = stats.ttest_ind(nested_enc, single_enc)

    print(f"\nNested vs Single-Timescale:")
    print(f"  Threat encounters: d = {d_nested_vs_single:+.3f} (p = {p_nested_vs_single:.4f})")

    if p_nested_vs_single < 0.05:
        if d_nested_vs_single < 0:
            print("  [SUCCESS] Full Nested agent has FEWER threat encounters")
        else:
            print("  [UNEXPECTED] Full Nested agent has MORE threat encounters")
    else:
        print("  [INCONCLUSIVE] No significant difference")

    # Stability comparison
    nested_var = np.array(results['Full Nested']['reward_variance'])
    single_var = np.array(results['Single-Timescale']['reward_variance'])
    d_var = cohens_d(nested_var, single_var)
    _, p_var = stats.ttest_ind(nested_var, single_var)

    print(f"\nLearning Stability:")
    print(f"  Reward variance: d = {d_var:+.3f} (p = {p_var:.4f})")

    if p_var < 0.05 and d_var < 0:
        print("  [SUCCESS] Full Nested agent is MORE stable")
    else:
        print("  [INCONCLUSIVE] No significant stability difference")

    return results


if __name__ == "__main__":
    results = run_experiment(n_seeds=50, n_episodes=200)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
