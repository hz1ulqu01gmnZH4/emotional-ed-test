"""Experiment: Ablation Studies for Emotional Modulation Components

Separates LR-only vs Policy-only effects of emotional modulation to determine
which component contributes more to threat avoidance behavior.

Agent Types:
1. LR-Only Agent: Emotional modulation ONLY affects learning rate
   - Action selection uses standard argmax (no emotional bias)

2. Policy-Only Agent: Learning rate is fixed (no emotional modulation)
   - Action selection is biased by emotional state (fear->risk-averse)

3. Full Emotional Agent (baseline): Both LR and policy modulation

4. Standard Agent (control): No emotional modulation

Metrics:
- Threat avoidance (minimum distance to threat)
- Learning speed (episodes to convergence)
- Final performance (steps to goal)

Statistical Analysis:
- Effect sizes for each ablation vs standard
- Component contribution analysis
- Additivity vs synergy test
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ak/emotional-ed-test')

from scipy import stats
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Environment
# =============================================================================

@dataclass
class EmotionalContext:
    """Context for computing emotional signals."""
    threat_distance: float
    goal_distance: float
    was_blocked: bool


class FearGridWorld:
    """Grid-world with threat at center, goal at corner.

    Layout (configurable size, default 7x7):
    - Start: (0, 0) - top-left
    - Threat: center position
    - Goal: bottom-right corner

    The optimal safe path goes around the threat.
    A larger grid makes the threat more salient because the direct path
    passes closer to it.
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

    def __init__(self, size: int = 7, threat_penalty: float = 0.0):
        """
        Args:
            size: Grid size (default 7 for better threat differentiation)
            threat_penalty: Optional reward penalty for being near threat
                           (default 0 - threat is implicit/emotional only)
        """
        self.size = size
        self.threat_pos = np.array([size // 2, size // 2])  # Center
        self.goal_pos = np.array([size - 1, size - 1])  # Bottom-right corner
        self.threat_penalty = threat_penalty
        self.reset()

    def reset(self) -> int:
        self.agent_pos = np.array([0, 0])
        self.step_count = 0
        self.min_threat_distance = float('inf')
        self.threat_encounters = 0  # Count how many times agent is near threat
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return float(np.linalg.norm(pos1 - pos2))

    def step(self, action: int) -> Tuple[int, float, bool, EmotionalContext]:
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        was_blocked = False
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            was_blocked = True
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        threat_dist = self._distance(self.agent_pos, self.threat_pos)
        goal_dist = self._distance(self.agent_pos, self.goal_pos)

        # Track minimum threat distance for this episode
        self.min_threat_distance = min(self.min_threat_distance, threat_dist)

        # Track threat encounters (within 1.5 cells)
        if threat_dist < 1.5:
            self.threat_encounters += 1

        # Reward structure
        done = np.array_equal(self.agent_pos, self.goal_pos) or self.step_count >= 100
        reward = 1.0 if np.array_equal(self.agent_pos, self.goal_pos) else -0.01

        # Optional threat penalty (only if configured)
        if self.threat_penalty > 0 and threat_dist < 1.5:
            reward -= self.threat_penalty

        context = EmotionalContext(
            threat_distance=threat_dist,
            goal_distance=goal_dist,
            was_blocked=was_blocked
        )

        return self._pos_to_state(self.agent_pos), reward, done, context

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4


# =============================================================================
# Fear Module (shared by agents that use it)
# =============================================================================

class FearModule:
    """Computes fear signal based on threat proximity."""

    def __init__(self, safe_distance: float = 3.0, max_fear: float = 1.0):
        self.safe_distance = safe_distance
        self.max_fear = max_fear

    def compute(self, context: EmotionalContext) -> float:
        if context.threat_distance >= self.safe_distance:
            return 0.0
        return self.max_fear * (1 - context.threat_distance / self.safe_distance)


# =============================================================================
# Agent Implementations
# =============================================================================

class StandardAgent:
    """Control: Vanilla Q-learning with no emotional modulation."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state: int, context: EmotionalContext = None) -> int:
        """Standard epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext = None):
        """Standard TD update. Context ignored."""
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        pass


class LROnlyAgent:
    """Emotional modulation ONLY affects learning rate.

    Action selection uses standard argmax (no emotional bias).
    Fear modulates learning rate: higher fear -> learn faster from negative outcomes.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 0.5):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.fear_weight = fear_weight
        self.fear_module = FearModule()

    def select_action(self, state: int, context: EmotionalContext = None) -> int:
        """Standard epsilon-greedy - NO emotional bias in action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        """TD update with fear-modulated learning rate."""
        fear = self.fear_module.compute(context)

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        # Fear increases learning from negative outcomes
        fear_modulation = 1.0 + self.fear_weight * fear * (1.0 if delta < 0 else 0.5)
        effective_lr = self.lr * fear_modulation

        self.Q[state, action] += effective_lr * delta

    def reset_episode(self):
        pass


class PolicyOnlyAgent:
    """Learning rate is fixed (no emotional modulation).

    Action selection is biased by emotional state:
    - Fear -> risk-averse (prefer higher Q-value actions)

    Key difference: Fear is computed FROM CONTEXT before action selection,
    not stored from previous step. This makes policy modulation immediate.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 1.0):  # Higher weight for policy-only
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.fear_weight = fear_weight
        self.fear_module = FearModule()

    def select_action(self, state: int, context: EmotionalContext = None) -> int:
        """Action selection biased by fear (risk aversion).

        Fear is computed immediately from context, not stored from previous step.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Compute fear from current context
        fear = 0.0
        if context is not None:
            fear = self.fear_module.compute(context)

        # Fear biases toward higher-Q actions (risk aversion)
        # When near threat, prefer safer (higher Q) actions
        if fear > 0:
            q_min, q_max = q_values.min(), q_values.max()
            if q_max > q_min:
                normalized = (q_values - q_min) / (q_max - q_min)
                # Fear adds bonus to high-Q actions (amplifies preference)
                q_values = q_values + fear * normalized * self.fear_weight
            else:
                # All Q-values equal: fear adds random tie-breaking with preference
                # for actions that don't lead toward threat
                # This is a soft bias that will be refined through learning
                pass

        return int(np.argmax(q_values))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        """Standard TD update - NO emotional modulation of learning rate."""
        # Standard learning (no modulation)
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        pass


class FullEmotionalAgent:
    """Full emotional modulation: Both LR and policy modulation.

    This is the baseline that combines both mechanisms:
    1. LR modulation: Fear increases learning from negative outcomes
    2. Policy modulation: Fear biases toward higher-Q (safer) actions
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 0.5):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.fear_weight = fear_weight
        self.fear_module = FearModule()

    def select_action(self, state: int, context: EmotionalContext = None) -> int:
        """Action selection biased by fear (computed from current context)."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Compute fear from current context
        fear = 0.0
        if context is not None:
            fear = self.fear_module.compute(context)

        if fear > 0:
            q_min, q_max = q_values.min(), q_values.max()
            if q_max > q_min:
                normalized = (q_values - q_min) / (q_max - q_min)
                q_values = q_values + fear * normalized * self.fear_weight

        return int(np.argmax(q_values))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: EmotionalContext):
        """TD update with fear-modulated learning rate."""
        fear = self.fear_module.compute(context)

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        # Fear modulates learning rate (same as LR-Only)
        fear_modulation = 1.0 + self.fear_weight * fear * (1.0 if delta < 0 else 0.5)
        effective_lr = self.lr * fear_modulation

        self.Q[state, action] += effective_lr * delta

    def reset_episode(self):
        pass


# =============================================================================
# Experiment Running
# =============================================================================

@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    steps_to_goal: int
    min_threat_distance: float
    threat_encounters: int
    total_reward: float
    reached_goal: bool


@dataclass
class AgentResults:
    """Aggregated results for an agent across all seeds."""
    name: str
    min_threat_distances: List[float] = field(default_factory=list)
    threat_encounters: List[float] = field(default_factory=list)
    episodes_to_converge: List[int] = field(default_factory=list)
    final_steps: List[float] = field(default_factory=list)
    goal_rates: List[float] = field(default_factory=list)


def run_episode(env: FearGridWorld, agent, max_steps: int = 100) -> EpisodeMetrics:
    """Run a single episode and return metrics."""
    state = env.reset()
    agent.reset_episode()

    total_reward = 0.0

    for step in range(max_steps):
        # Get context for action selection
        context = EmotionalContext(
            threat_distance=env._distance(env.agent_pos, env.threat_pos),
            goal_distance=env._distance(env.agent_pos, env.goal_pos),
            was_blocked=False
        )

        action = agent.select_action(state, context)
        next_state, reward, done, context = env.step(action)

        agent.update(state, action, reward, next_state, done, context)
        total_reward += reward

        if done:
            break

        state = next_state

    reached_goal = np.array_equal(env.agent_pos, env.goal_pos)

    return EpisodeMetrics(
        steps_to_goal=env.step_count,
        min_threat_distance=env.min_threat_distance,
        threat_encounters=env.threat_encounters,
        total_reward=total_reward,
        reached_goal=reached_goal
    )


def detect_convergence(rewards: List[float], window: int = 20,
                       stability_threshold: float = 0.05) -> int:
    """Detect episode where agent converges.

    Convergence = first episode where reward variance stabilizes
    (std deviation in window drops below threshold of the mean).
    Returns n_episodes if never converges.
    """
    if len(rewards) < window:
        return len(rewards)

    for i in range(window, len(rewards)):
        window_rewards = rewards[i-window:i]
        avg = np.mean(window_rewards)
        std = np.std(window_rewards)

        # Convergence when relative std is low and average is positive
        if avg > 0 and std / abs(avg) < stability_threshold:
            return i - window

    return len(rewards)


def run_single_seed(seed: int, n_episodes: int = 200) -> Dict[str, Dict]:
    """Run all agents for a single seed.

    Each agent gets its own unique seed derived from the base seed.
    This ensures:
    1. Different agents have different exploration trajectories
    2. Results are reproducible given the same base seed
    3. We can fairly compare agent behaviors under similar but not identical conditions
    """
    results = {}

    agent_configs = [
        ('standard', StandardAgent, {}),
        ('lr_only', LROnlyAgent, {'fear_weight': 0.5}),
        ('policy_only', PolicyOnlyAgent, {'fear_weight': 1.0}),
        ('full_emotional', FullEmotionalAgent, {'fear_weight': 0.5}),
    ]

    for idx, (name, AgentClass, kwargs) in enumerate(agent_configs):
        # Each agent gets a unique seed: base_seed * 4 + agent_index
        agent_seed = seed * 4 + idx
        np.random.seed(agent_seed)

        env = FearGridWorld(size=7)  # Larger grid for better threat differentiation
        agent = AgentClass(env.n_states, env.n_actions, **kwargs)

        episode_metrics = []

        for ep in range(n_episodes):
            metrics = run_episode(env, agent)
            episode_metrics.append(metrics)

        # Aggregate metrics
        rewards = [m.total_reward for m in episode_metrics]
        convergence_ep = detect_convergence(rewards)

        # Use last 20 episodes for final performance
        last_n = 20
        final_episodes = episode_metrics[-last_n:]

        results[name] = {
            'avg_min_threat_dist': np.mean([m.min_threat_distance for m in final_episodes]),
            'avg_threat_encounters': np.mean([m.threat_encounters for m in final_episodes]),
            'convergence_episode': convergence_ep,
            'final_steps': np.mean([m.steps_to_goal for m in final_episodes]),
            'goal_rate': np.mean([m.reached_goal for m in final_episodes]),
            'all_min_threat_dists': [m.min_threat_distance for m in episode_metrics],
        }

    return results


def run_experiment(n_seeds: int = 50, n_episodes: int = 200) -> Dict[str, AgentResults]:
    """Run full ablation experiment."""
    print("=" * 70)
    print("ABLATION STUDIES: LR-Only vs Policy-Only Effects")
    print("=" * 70)
    print(f"Seeds: {n_seeds}, Episodes per seed: {n_episodes}")
    print(f"Environment: 7x7 FearGridWorld (threat at center, goal at corner)")
    print("=" * 70)

    all_results = {
        'standard': AgentResults(name='Standard (Control)'),
        'lr_only': AgentResults(name='LR-Only'),
        'policy_only': AgentResults(name='Policy-Only'),
        'full_emotional': AgentResults(name='Full Emotional'),
    }

    for seed in range(n_seeds):
        seed_results = run_single_seed(seed, n_episodes)

        for agent_name, metrics in seed_results.items():
            all_results[agent_name].min_threat_distances.append(metrics['avg_min_threat_dist'])
            all_results[agent_name].threat_encounters.append(metrics['avg_threat_encounters'])
            all_results[agent_name].episodes_to_converge.append(metrics['convergence_episode'])
            all_results[agent_name].final_steps.append(metrics['final_steps'])
            all_results[agent_name].goal_rates.append(metrics['goal_rate'])

        if (seed + 1) % 10 == 0:
            print(f"  Completed {seed + 1}/{n_seeds} seeds")

    return all_results


# =============================================================================
# Statistical Analysis
# =============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def analyze_results(results: Dict[str, AgentResults]) -> Dict:
    """Perform statistical analysis on results."""
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    standard = results['standard']
    lr_only = results['lr_only']
    policy_only = results['policy_only']
    full_emotional = results['full_emotional']

    analysis = {}

    # -------------------------------------------------------------------------
    # 1. Threat Avoidance (Primary Metric)
    # -------------------------------------------------------------------------
    print("\n--- THREAT AVOIDANCE (Min Distance to Threat) ---")
    print("Higher is better (more threat avoidance)\n")

    for name, agent in results.items():
        dists = np.array(agent.min_threat_distances)
        print(f"{agent.name:25s}: {np.mean(dists):.3f} +/- {np.std(dists):.3f}")

    # Effect sizes vs standard
    print("\nEffect Sizes vs Standard (Cohen's d):")

    std_dists = np.array(standard.min_threat_distances)

    for name in ['lr_only', 'policy_only', 'full_emotional']:
        agent = results[name]
        dists = np.array(agent.min_threat_distances)
        d = cohens_d(dists, std_dists)
        _, p = stats.ttest_ind(dists, std_dists)
        sig = "*" if p < 0.05 else ""
        print(f"  {agent.name:25s}: d = {d:+.3f} (p = {p:.4f}) {sig}")
        analysis[f'{name}_threat_d'] = d
        analysis[f'{name}_threat_p'] = p

    # -------------------------------------------------------------------------
    # 2. Learning Speed (Episodes to Convergence)
    # -------------------------------------------------------------------------
    print("\n--- LEARNING SPEED (Episodes to Convergence) ---")
    print("Lower is better (faster learning)\n")

    for name, agent in results.items():
        eps = np.array(agent.episodes_to_converge)
        print(f"{agent.name:25s}: {np.mean(eps):.1f} +/- {np.std(eps):.1f}")

    print("\nEffect Sizes vs Standard (Cohen's d):")

    std_eps = np.array(standard.episodes_to_converge)

    for name in ['lr_only', 'policy_only', 'full_emotional']:
        agent = results[name]
        eps = np.array(agent.episodes_to_converge)
        d = cohens_d(eps, std_eps)
        _, p = stats.ttest_ind(eps, std_eps)
        sig = "*" if p < 0.05 else ""
        # Negative d means faster learning (fewer episodes)
        print(f"  {agent.name:25s}: d = {d:+.3f} (p = {p:.4f}) {sig}")
        analysis[f'{name}_converge_d'] = d
        analysis[f'{name}_converge_p'] = p

    # -------------------------------------------------------------------------
    # 3. Final Performance (Steps to Goal)
    # -------------------------------------------------------------------------
    print("\n--- FINAL PERFORMANCE (Avg Steps to Goal, last 20 episodes) ---")
    print("Lower is better (more efficient paths)\n")

    for name, agent in results.items():
        steps = np.array(agent.final_steps)
        print(f"{agent.name:25s}: {np.mean(steps):.2f} +/- {np.std(steps):.2f}")

    print("\nEffect Sizes vs Standard (Cohen's d):")

    std_steps = np.array(standard.final_steps)

    for name in ['lr_only', 'policy_only', 'full_emotional']:
        agent = results[name]
        steps = np.array(agent.final_steps)
        d = cohens_d(steps, std_steps)
        _, p = stats.ttest_ind(steps, std_steps)
        sig = "*" if p < 0.05 else ""
        print(f"  {agent.name:25s}: d = {d:+.3f} (p = {p:.4f}) {sig}")
        analysis[f'{name}_steps_d'] = d
        analysis[f'{name}_steps_p'] = p

    # -------------------------------------------------------------------------
    # 4. Component Contribution Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 70)

    # Which component contributes more to threat avoidance?
    lr_effect = analysis['lr_only_threat_d']
    policy_effect = analysis['policy_only_threat_d']
    full_effect = analysis['full_emotional_threat_d']

    print("\nThreat Avoidance Contributions:")
    print(f"  LR-Only effect:     d = {lr_effect:+.3f}")
    print(f"  Policy-Only effect: d = {policy_effect:+.3f}")
    print(f"  Full Emotional:     d = {full_effect:+.3f}")

    if abs(lr_effect) > abs(policy_effect):
        dominant = "LR modulation"
        contribution_ratio = abs(lr_effect) / abs(policy_effect) if abs(policy_effect) > 0.01 else float('inf')
    else:
        dominant = "Policy modulation"
        contribution_ratio = abs(policy_effect) / abs(lr_effect) if abs(lr_effect) > 0.01 else float('inf')

    print(f"\n  Dominant Component: {dominant}")
    if contribution_ratio < float('inf'):
        print(f"  Contribution Ratio: {contribution_ratio:.2f}x")

    analysis['dominant_component'] = dominant
    analysis['contribution_ratio'] = contribution_ratio

    # -------------------------------------------------------------------------
    # 5. Additivity vs Synergy Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ADDITIVITY vs SYNERGY ANALYSIS")
    print("=" * 70)

    # If effects are additive: Full = LR + Policy
    # If synergistic: Full > LR + Policy
    # If redundant: Full < LR + Policy

    expected_additive = lr_effect + policy_effect
    actual_combined = full_effect

    print(f"\nExpected if additive (LR + Policy): d = {expected_additive:+.3f}")
    print(f"Actual Full Emotional:              d = {actual_combined:+.3f}")

    difference = actual_combined - expected_additive

    if abs(difference) < 0.1:
        interaction_type = "ADDITIVE"
        interaction_desc = "Components contribute independently"
    elif difference > 0.1:
        interaction_type = "SYNERGISTIC"
        interaction_desc = "Combined effect exceeds sum of parts"
    else:
        interaction_type = "REDUNDANT/INTERFERING"
        interaction_desc = "Combined effect less than sum (possible interference)"

    print(f"\nDifference: {difference:+.3f}")
    print(f"Interaction Type: {interaction_type}")
    print(f"Interpretation: {interaction_desc}")

    analysis['expected_additive'] = expected_additive
    analysis['actual_combined'] = actual_combined
    analysis['interaction_type'] = interaction_type

    # -------------------------------------------------------------------------
    # 6. Threat Encounters (Secondary Metric)
    # -------------------------------------------------------------------------
    print("\n--- THREAT ENCOUNTERS (Times within 1.5 cells of threat) ---")
    print("Lower is better (fewer dangerous encounters)\n")

    for name, agent in results.items():
        encounters = np.array(agent.threat_encounters)
        print(f"{agent.name:25s}: {np.mean(encounters):.2f} +/- {np.std(encounters):.2f}")

    print("\nEffect Sizes vs Standard (Cohen's d):")

    std_encounters = np.array(standard.threat_encounters)

    for name in ['lr_only', 'policy_only', 'full_emotional']:
        agent = results[name]
        encounters = np.array(agent.threat_encounters)
        d = cohens_d(encounters, std_encounters)
        _, p = stats.ttest_ind(encounters, std_encounters)
        sig = "*" if p < 0.05 else ""
        # Negative d means fewer encounters (better avoidance)
        print(f"  {agent.name:25s}: d = {d:+.3f} (p = {p:.4f}) {sig}")
        analysis[f'{name}_encounters_d'] = d
        analysis[f'{name}_encounters_p'] = p

    # -------------------------------------------------------------------------
    # 7. Goal Rate Comparison
    # -------------------------------------------------------------------------
    print("\n--- GOAL COMPLETION RATE ---")

    for name, agent in results.items():
        rates = np.array(agent.goal_rates)
        print(f"{agent.name:25s}: {np.mean(rates)*100:.1f}%")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n1. THREAT AVOIDANCE:")
    if analysis['lr_only_threat_p'] < 0.05 and lr_effect > 0:
        print("   - LR-Only: SIGNIFICANT improvement over standard")
    else:
        print("   - LR-Only: No significant improvement")

    if analysis['policy_only_threat_p'] < 0.05 and policy_effect > 0:
        print("   - Policy-Only: SIGNIFICANT improvement over standard")
    else:
        print("   - Policy-Only: No significant improvement")

    if analysis['full_emotional_threat_p'] < 0.05 and full_effect > 0:
        print("   - Full Emotional: SIGNIFICANT improvement over standard")
    else:
        print("   - Full Emotional: No significant improvement")

    print(f"\n2. DOMINANT COMPONENT: {dominant}")

    print(f"\n3. INTERACTION: {interaction_type}")

    print("\n4. CONCLUSIONS:")
    if interaction_type == "SYNERGISTIC":
        print("   The LR and Policy components work together synergistically.")
        print("   Full emotional agent > sum of individual components.")
    elif interaction_type == "ADDITIVE":
        print("   The LR and Policy components contribute independently.")
        print("   Full emotional agent = sum of individual components.")
    else:
        print("   There may be interference between LR and Policy modulation.")
        print("   Consider using only the dominant component.")

    return analysis


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Ablation Studies for Emotional Modulation')
    parser.add_argument('--seeds', type=int, default=50, help='Number of random seeds')
    parser.add_argument('--episodes', type=int, default=200, help='Episodes per seed')
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("# EMOTIONAL MODULATION ABLATION STUDIES")
    print("# Separating LR-Only vs Policy-Only Effects")
    print("#" * 70 + "\n")

    results = run_experiment(n_seeds=args.seeds, n_episodes=args.episodes)
    analysis = analyze_results(results)

    print("\n" + "#" * 70)
    print("# EXPERIMENT COMPLETE")
    print("#" * 70)
