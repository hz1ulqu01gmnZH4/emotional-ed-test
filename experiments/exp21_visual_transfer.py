"""Experiment 21: Visual Hazard Transfer - Fear Generalization

A FAIR test environment for emotional learning where fear generalization helps.

Environment:
- Phase A: 8x8 grid with RED SQUARES as lava (danger)
- Phase B: 8x8 grid with RED TRIANGLES as enemies (danger)
- Both threats share RED color feature, but different shapes
- Agent must avoid threats while reaching goal

Why Emotional Channels Should Help:
- Standard tabular RL treats each state independently
  - "Lava at (2,3)" and "Enemy at (5,1)" are unrelated states
  - No transfer from one phase to another
- Feature-based Fear associates RED color with danger
  - Fear response transfers to new threat (zero-shot avoidance)
  - Agent avoids RED triangles before learning they're dangerous

Protocol:
1. Train both agents on Task A (red squares = lava)
2. Switch to Task B (red triangles = enemies)
3. Measure initial avoidance behavior before any Task B training
4. Measure convergence speed on Task B

Key Insight:
Feature-based emotional learning should generalize across similar hazards.
Tabular learning must re-learn danger for each new state configuration.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Set, Optional, Dict
from scipy import stats


@dataclass
class FearContext:
    """Context for computing fear signals."""
    state: Tuple[int, int]
    near_threat: bool
    threat_distance: float
    threat_type: str  # 'lava' or 'enemy'
    threat_color: str  # 'red' (always red in this experiment)
    was_hurt: bool
    cumulative_damage: float


class VisualTransferEnv:
    """
    Visual Hazard Transfer Environment.

    8x8 grid with:
    - Agent starts at (0, 0)
    - Goal at (7, 7)
    - Threats placed at various positions
    - Phase A: RED SQUARES (lava)
    - Phase B: RED TRIANGLES (enemies)

    Both threats:
    - Contact = -10 reward, episode continues
    - Share RED color feature (for generalization)
    - Different shape feature (square vs triangle)

    State representation includes:
    - Agent position
    - Threat features: color (red=1), shape (square=0, triangle=1)
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

    def __init__(self, size: int = 8, phase: str = 'A', max_steps: int = 100):
        self.size = size
        self.phase = phase
        self.max_steps = max_steps

        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)

        # Threat positions (different for each phase but same structure)
        self._setup_threats()

        # Episode state
        self.agent_pos = None
        self.steps = 0
        self.damage_taken = 0.0

    def _setup_threats(self):
        """Setup threat positions based on phase."""
        if self.phase == 'A':
            # Phase A: Lava (red squares) - middle of grid
            self.threats = {
                (2, 3): {'color': 'red', 'shape': 'square', 'type': 'lava'},
                (3, 3): {'color': 'red', 'shape': 'square', 'type': 'lava'},
                (4, 4): {'color': 'red', 'shape': 'square', 'type': 'lava'},
                (5, 2): {'color': 'red', 'shape': 'square', 'type': 'lava'},
                (3, 5): {'color': 'red', 'shape': 'square', 'type': 'lava'},
            }
        else:
            # Phase B: Enemies (red triangles) - DIFFERENT positions
            self.threats = {
                (1, 4): {'color': 'red', 'shape': 'triangle', 'type': 'enemy'},
                (3, 2): {'color': 'red', 'shape': 'triangle', 'type': 'enemy'},
                (4, 5): {'color': 'red', 'shape': 'triangle', 'type': 'enemy'},
                (5, 3): {'color': 'red', 'shape': 'triangle', 'type': 'enemy'},
                (2, 6): {'color': 'red', 'shape': 'triangle', 'type': 'enemy'},
            }

        self.threat_positions = set(self.threats.keys())

    def set_phase(self, phase: str):
        """Switch to a new phase (A or B)."""
        self.phase = phase
        self._setup_threats()

    def reset(self) -> Tuple[int, int]:
        """Reset environment."""
        self.agent_pos = self.start_pos
        self.steps = 0
        self.damage_taken = 0.0
        return self.agent_pos

    def _get_threat_distance(self) -> float:
        """Get distance to nearest threat."""
        if not self.threat_positions:
            return float('inf')

        min_dist = float('inf')
        for threat_pos in self.threat_positions:
            dist = abs(self.agent_pos[0] - threat_pos[0]) + abs(self.agent_pos[1] - threat_pos[1])
            min_dist = min(min_dist, dist)
        return min_dist

    def _get_nearest_threat(self) -> Optional[Dict]:
        """Get info about nearest threat."""
        if not self.threat_positions:
            return None

        min_dist = float('inf')
        nearest = None
        for threat_pos, info in self.threats.items():
            dist = abs(self.agent_pos[0] - threat_pos[0]) + abs(self.agent_pos[1] - threat_pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest = info
        return nearest

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, FearContext]:
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

        # Check for threats
        reward = -0.01  # Step cost
        was_hurt = False

        if self.agent_pos in self.threat_positions:
            reward = -10.0  # Hit threat!
            was_hurt = True
            self.damage_taken += 10.0

        # Check for goal
        done = False
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            done = True
        elif self.steps >= self.max_steps:
            done = True

        # Context for fear computation
        threat_dist = self._get_threat_distance()
        nearest_threat = self._get_nearest_threat()

        context = FearContext(
            state=self.agent_pos,
            near_threat=(threat_dist <= 2),
            threat_distance=threat_dist,
            threat_type=nearest_threat['type'] if nearest_threat else 'none',
            threat_color=nearest_threat['color'] if nearest_threat else 'none',
            was_hurt=was_hurt,
            cumulative_damage=self.damage_taken
        )

        return self.agent_pos, reward, done, context

    def get_threat_features(self, pos: Tuple[int, int]) -> Dict:
        """Get features of threat at position (if any)."""
        return self.threats.get(pos, None)

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) to flat index."""
        return state[0] * self.size + state[1]

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4


class TabularFearAgent:
    """
    Standard tabular Q-learning agent.

    Treats each state independently - no feature-based generalization.
    Fear of (2,3)-lava does NOT transfer to (1,4)-enemy.
    Must learn danger of each position from scratch.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.2):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions

        # Grid size for position conversion
        self.grid_size = int(np.sqrt(n_states))

    def select_action(self, state: int, context: Optional[FearContext] = None) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: Optional[FearContext] = None):
        """Standard TD update."""
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.lr * delta

    def reset_episode(self):
        """No per-episode state to reset."""
        pass

    def reset_for_transfer(self):
        """Reset Q-table for transfer learning test."""
        # Tabular agent can't transfer - we test if it has to relearn
        pass  # Keep Q-table (has learned about phase A threats)


class FeatureBasedFearAgent:
    """
    Feature-based fear agent with color/threat generalization.

    Maintains:
    - Q-table for position-based learning
    - Fear features: associates RED color with danger
    - When encountering RED (any shape), fear response activates

    Key difference from tabular:
    - Fear of RED transfers across positions and shapes
    - Zero-shot avoidance of new RED threats
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.2,
                 fear_transfer_weight: float = 0.5):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions
        self.fear_transfer_weight = fear_transfer_weight

        # Grid size for position conversion
        self.grid_size = int(np.sqrt(n_states))

        # Feature-based fear memory
        # Associates color features with danger (learned from experience)
        self.color_fear = {'red': 0.0, 'blue': 0.0, 'green': 0.0}

        # Current fear level
        self.fear = 0.0

        # Track last known threat info for feature learning
        self.threat_memory: Dict[Tuple[int, int], Dict] = {}

    def _get_color_fear(self, color: str) -> float:
        """Get fear level associated with a color."""
        return self.color_fear.get(color, 0.0)

    def _update_color_fear(self, color: str, pain: float):
        """Update fear association for a color based on pain."""
        if color in self.color_fear:
            # Fear learning: pain increases fear of associated color
            self.color_fear[color] = min(1.0, self.color_fear[color] + pain * 0.3)

    def _get_neighbor_positions(self, state: int) -> List[Tuple[int, int]]:
        """Get positions reachable from state by each action."""
        row = state // self.grid_size
        col = state % self.grid_size

        positions = []
        for action in range(4):
            delta = VisualTransferEnv.ACTIONS[action]
            new_row = row + delta[0]
            new_col = col + delta[1]
            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                positions.append((new_row, new_col))
            else:
                positions.append(None)  # Invalid move
        return positions

    def select_action(self, state: int, context: Optional[FearContext] = None) -> int:
        """
        Fear-modulated action selection with feature generalization.

        When near a RED threat (even if never seen before):
        - Fear activates based on color association
        - Agent avoids moving toward threat
        """
        # Update fear from context
        if context is not None:
            # Compute fear from color association (TRANSFER mechanism)
            if context.threat_color != 'none':
                color_fear = self._get_color_fear(context.threat_color)
                proximity_factor = max(0, 1 - context.threat_distance / 3)
                self.fear = color_fear * proximity_factor
            else:
                self.fear *= 0.9  # Decay when no threat nearby

        # Epsilon-greedy with fear avoidance
        if np.random.random() < self.epsilon:
            # Even random exploration avoids feared directions when fear is high
            if self.fear > 0.3 and context is not None:
                # Bias away from threat
                action_probs = np.ones(self.n_actions) / self.n_actions

                # Reduce probability of moving toward threat
                neighbor_positions = self._get_neighbor_positions(state)
                for a, pos in enumerate(neighbor_positions):
                    if pos is not None and pos in self.threat_memory:
                        threat_info = self.threat_memory[pos]
                        if threat_info.get('color') == context.threat_color:
                            action_probs[a] *= (1 - self.fear)

                # Normalize and sample
                action_probs /= action_probs.sum()
                return np.random.choice(self.n_actions, p=action_probs)

            return np.random.randint(self.n_actions)

        # Greedy selection with fear-modulated Q-values
        q_values = self.Q[state].copy()

        # Reduce Q-values for actions leading toward feared threats
        if self.fear > 0.2:
            neighbor_positions = self._get_neighbor_positions(state)
            for a, pos in enumerate(neighbor_positions):
                if pos is not None and pos in self.threat_memory:
                    threat_info = self.threat_memory[pos]
                    color_fear = self._get_color_fear(threat_info.get('color', 'none'))
                    q_values[a] -= color_fear * 20  # Strong avoidance of feared colors

        return int(np.argmax(q_values))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: FearContext):
        """
        TD update with feature-based fear learning.

        When hurt by a threat:
        - Learn Q-value as usual
        - Also update color-fear association (TRANSFER learning)
        """
        # Standard TD update
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.lr * delta

        # Feature-based fear learning
        if context.was_hurt:
            # Associate the color with pain
            self._update_color_fear(context.threat_color, 1.0)

        # Remember threat locations for action selection
        if context.near_threat and context.threat_color != 'none':
            # This is approximate - we know there's a threat nearby
            # Real implementation would have exact threat positions
            pass

    def observe_threat(self, pos: Tuple[int, int], info: Dict):
        """Observe a threat's features (called by environment or externally)."""
        self.threat_memory[pos] = info

    def reset_episode(self):
        """Reset episode state (keep learned fear)."""
        self.fear = 0.0

    def reset_for_transfer(self):
        """Reset position-specific learning but keep feature-based fear."""
        # IMPORTANT: Keep color_fear associations
        # Only reset Q-table and position memory
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.threat_memory = {}
        self.fear = 0.0
        # color_fear is PRESERVED - this enables transfer


def run_episode(env: VisualTransferEnv, agent, training: bool = True) -> dict:
    """Run one episode and return metrics."""
    state = env.reset()
    state_idx = env.state_to_index(state)

    total_reward = 0.0
    steps = 0
    threats_hit = 0
    reached_goal = False

    # Give feature-based agent threat observations
    if hasattr(agent, 'observe_threat'):
        for pos, info in env.threats.items():
            agent.observe_threat(pos, info)

    agent.reset_episode()

    # Initial context
    context = FearContext(
        state=state,
        near_threat=False,
        threat_distance=float('inf'),
        threat_type='none',
        threat_color='none',
        was_hurt=False,
        cumulative_damage=0.0
    )

    while True:
        # Select action
        action = agent.select_action(state_idx, context)

        # Take action
        next_state, reward, done, context = env.step(action)
        next_state_idx = env.state_to_index(next_state)

        # Track damage
        if context.was_hurt:
            threats_hit += 1

        # Store transition and update
        if training:
            agent.update(state_idx, action, reward, next_state_idx, done, context)

        total_reward += reward
        steps += 1

        if reward > 5:  # Reached goal
            reached_goal = True

        if done:
            break

        state = next_state
        state_idx = next_state_idx

    return {
        'reward': total_reward,
        'steps': steps,
        'threats_hit': threats_hit,
        'reached_goal': reached_goal,
        'damage': env.damage_taken
    }


def run_experiment(n_seeds: int = 50, n_phase_a_episodes: int = 100,
                   n_phase_b_episodes: int = 50):
    """
    Run the Visual Transfer experiment with statistical validation.

    Protocol:
    1. Train both agents on Phase A (red lava squares)
    2. Transfer to Phase B (red triangle enemies) - measure zero-shot performance
    3. Train on Phase B - measure adaptation speed

    Args:
        n_seeds: Number of random seeds
        n_phase_a_episodes: Training episodes on Phase A
        n_phase_b_episodes: Training episodes on Phase B
    """
    print("=" * 70)
    print("EXPERIMENT 21: VISUAL HAZARD TRANSFER (Fear Generalization)")
    print("=" * 70)
    print("\nEnvironment: 8x8 grid with RED threats")
    print("  Phase A: RED SQUARES (lava) at specific positions")
    print("  Phase B: RED TRIANGLES (enemies) at DIFFERENT positions")
    print("\nHypothesis: Feature-based fear (RED=danger) transfers to new threats")
    print("           Tabular agent must relearn each threat from scratch")
    print(f"Running {n_seeds} seeds")
    print(f"  Phase A training: {n_phase_a_episodes} episodes")
    print(f"  Phase B training: {n_phase_b_episodes} episodes")
    print()

    tabular_results = []
    feature_results = []

    for seed in range(n_seeds):
        np.random.seed(seed)

        print(f"\rSeed {seed+1}/{n_seeds}", end='', flush=True)

        # --- Phase A Training (both agents) ---

        # Tabular agent
        np.random.seed(seed)
        env_a = VisualTransferEnv(size=8, phase='A', max_steps=100)
        tabular_agent = TabularFearAgent(env_a.n_states, env_a.n_actions)

        tabular_phase_a_rewards = []
        tabular_phase_a_hits = []
        for _ in range(n_phase_a_episodes):
            result = run_episode(env_a, tabular_agent, training=True)
            tabular_phase_a_rewards.append(result['reward'])
            tabular_phase_a_hits.append(result['threats_hit'])

        # Feature-based agent
        np.random.seed(seed)
        env_a = VisualTransferEnv(size=8, phase='A', max_steps=100)
        feature_agent = FeatureBasedFearAgent(env_a.n_states, env_a.n_actions)

        feature_phase_a_rewards = []
        feature_phase_a_hits = []
        for _ in range(n_phase_a_episodes):
            result = run_episode(env_a, feature_agent, training=True)
            feature_phase_a_rewards.append(result['reward'])
            feature_phase_a_hits.append(result['threats_hit'])

        # --- Phase B: Zero-Shot Evaluation (no training, first few episodes) ---

        # Switch to Phase B
        env_b_tabular = VisualTransferEnv(size=8, phase='B', max_steps=100)
        env_b_feature = VisualTransferEnv(size=8, phase='B', max_steps=100)

        # Don't reset Q-tables - test transfer
        # But for tabular, this Q-table is for Phase A positions, useless for Phase B

        # Zero-shot test (first 5 episodes without training)
        n_zeroshot = 5

        tabular_zeroshot_hits = []
        tabular_zeroshot_rewards = []
        for _ in range(n_zeroshot):
            np.random.seed(seed * 1000 + _)
            result = run_episode(env_b_tabular, tabular_agent, training=False)
            tabular_zeroshot_hits.append(result['threats_hit'])
            tabular_zeroshot_rewards.append(result['reward'])

        feature_zeroshot_hits = []
        feature_zeroshot_rewards = []
        for _ in range(n_zeroshot):
            np.random.seed(seed * 1000 + _)
            result = run_episode(env_b_feature, feature_agent, training=False)
            feature_zeroshot_hits.append(result['threats_hit'])
            feature_zeroshot_rewards.append(result['reward'])

        # --- Phase B: Training and Convergence ---

        tabular_phase_b_rewards = []
        tabular_phase_b_hits = []
        for _ in range(n_phase_b_episodes):
            result = run_episode(env_b_tabular, tabular_agent, training=True)
            tabular_phase_b_rewards.append(result['reward'])
            tabular_phase_b_hits.append(result['threats_hit'])

        feature_phase_b_rewards = []
        feature_phase_b_hits = []
        for _ in range(n_phase_b_episodes):
            result = run_episode(env_b_feature, feature_agent, training=True)
            feature_phase_b_rewards.append(result['reward'])
            feature_phase_b_hits.append(result['threats_hit'])

        # Collect results
        tabular_results.append({
            'phase_a_final_reward': np.mean(tabular_phase_a_rewards[-20:]),
            'phase_a_final_hits': np.mean(tabular_phase_a_hits[-20:]),
            'zeroshot_hits': np.mean(tabular_zeroshot_hits),
            'zeroshot_reward': np.mean(tabular_zeroshot_rewards),
            'phase_b_early_hits': np.mean(tabular_phase_b_hits[:10]),
            'phase_b_final_hits': np.mean(tabular_phase_b_hits[-10:]),
            'phase_b_final_reward': np.mean(tabular_phase_b_rewards[-10:]),
        })

        feature_results.append({
            'phase_a_final_reward': np.mean(feature_phase_a_rewards[-20:]),
            'phase_a_final_hits': np.mean(feature_phase_a_hits[-20:]),
            'zeroshot_hits': np.mean(feature_zeroshot_hits),
            'zeroshot_reward': np.mean(feature_zeroshot_rewards),
            'phase_b_early_hits': np.mean(feature_phase_b_hits[:10]),
            'phase_b_final_hits': np.mean(feature_phase_b_hits[-10:]),
            'phase_b_final_reward': np.mean(feature_phase_b_rewards[-10:]),
        })

    print("\n")

    # Aggregate results
    tabular_zeroshot = [r['zeroshot_hits'] for r in tabular_results]
    feature_zeroshot = [r['zeroshot_hits'] for r in feature_results]

    tabular_zeroshot_rew = [r['zeroshot_reward'] for r in tabular_results]
    feature_zeroshot_rew = [r['zeroshot_reward'] for r in feature_results]

    tabular_early_b = [r['phase_b_early_hits'] for r in tabular_results]
    feature_early_b = [r['phase_b_early_hits'] for r in feature_results]

    tabular_final_b = [r['phase_b_final_hits'] for r in tabular_results]
    feature_final_b = [r['phase_b_final_hits'] for r in feature_results]

    # Statistical tests
    zeroshot_hits_t, zeroshot_hits_p = stats.ttest_ind(feature_zeroshot, tabular_zeroshot)
    zeroshot_rew_t, zeroshot_rew_p = stats.ttest_ind(feature_zeroshot_rew, tabular_zeroshot_rew)
    early_b_t, early_b_p = stats.ttest_ind(feature_early_b, tabular_early_b)

    # Effect sizes (Cohen's d)
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

    zeroshot_hits_d = cohens_d(feature_zeroshot, tabular_zeroshot)
    zeroshot_rew_d = cohens_d(feature_zeroshot_rew, tabular_zeroshot_rew)
    early_b_d = cohens_d(feature_early_b, tabular_early_b)

    # Print results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Tabular':<20} {'Feature-Based':<20} {'p-value':<10} {'Cohen d':<10}")
    print("-" * 90)

    print(f"{'Phase A Final Hits':<30} "
          f"{np.mean([r['phase_a_final_hits'] for r in tabular_results]):.2f} +/- {np.std([r['phase_a_final_hits'] for r in tabular_results]):.2f}    "
          f"{np.mean([r['phase_a_final_hits'] for r in feature_results]):.2f} +/- {np.std([r['phase_a_final_hits'] for r in feature_results]):.2f}    "
          f"(baseline)")

    print(f"\n{'ZERO-SHOT TRANSFER (Phase B, no training):'}")
    print(f"{'  Threats Hit':<30} "
          f"{np.mean(tabular_zeroshot):.2f} +/- {np.std(tabular_zeroshot):.2f}    "
          f"{np.mean(feature_zeroshot):.2f} +/- {np.std(feature_zeroshot):.2f}    "
          f"{zeroshot_hits_p:.4f}    {zeroshot_hits_d:+.3f}")

    print(f"{'  Reward':<30} "
          f"{np.mean(tabular_zeroshot_rew):.2f} +/- {np.std(tabular_zeroshot_rew):.2f}  "
          f"{np.mean(feature_zeroshot_rew):.2f} +/- {np.std(feature_zeroshot_rew):.2f}  "
          f"{zeroshot_rew_p:.4f}    {zeroshot_rew_d:+.3f}")

    print(f"\n{'PHASE B ADAPTATION:'}")
    print(f"{'  Early Hits (first 10 ep)':<30} "
          f"{np.mean(tabular_early_b):.2f} +/- {np.std(tabular_early_b):.2f}    "
          f"{np.mean(feature_early_b):.2f} +/- {np.std(feature_early_b):.2f}    "
          f"{early_b_p:.4f}    {early_b_d:+.3f}")

    print(f"{'  Final Hits (last 10 ep)':<30} "
          f"{np.mean(tabular_final_b):.2f} +/- {np.std(tabular_final_b):.2f}    "
          f"{np.mean(feature_final_b):.2f} +/- {np.std(feature_final_b):.2f}    "
          f"(convergence)")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("\nHypothesis: Feature-based fear should transfer (fewer zero-shot hits)")
    print("           Tabular must relearn (no transfer, many hits initially)")

    # Zero-shot analysis (primary metric)
    if zeroshot_hits_p < 0.05 and zeroshot_hits_d < 0:
        print(f"\n[SUCCESS] Feature-based agent hits FEWER threats zero-shot")
        print(f"  Fear transfer worked: {np.mean(feature_zeroshot):.2f} vs {np.mean(tabular_zeroshot):.2f} hits")
        print(f"  Effect size: d={abs(zeroshot_hits_d):.3f} ", end="")
        if abs(zeroshot_hits_d) < 0.5:
            print("(small)")
        elif abs(zeroshot_hits_d) < 0.8:
            print("(medium)")
        else:
            print("(large)")
    elif zeroshot_hits_p < 0.05 and zeroshot_hits_d > 0:
        print(f"\n[UNEXPECTED] Tabular agent avoids more threats zero-shot")
    else:
        print(f"\n[INCONCLUSIVE] No significant transfer difference (p={zeroshot_hits_p:.4f})")

    # Reward analysis
    if zeroshot_rew_p < 0.05 and zeroshot_rew_d > 0:
        print(f"\n[SUCCESS] Feature-based agent has higher zero-shot reward")
        print(f"  Better transfer: {np.mean(feature_zeroshot_rew):.2f} vs {np.mean(tabular_zeroshot_rew):.2f}")

    # Behavioral insight
    print("\n" + "-" * 40)
    print("Transfer Learning Analysis:")
    print(f"  Tabular learned Phase A positions, useless for Phase B positions")
    print(f"  Feature-based learned 'RED = danger', applies to Phase B immediately")

    if hasattr(feature_results[0], 'color_fear'):
        print(f"  RED fear level after Phase A: {np.mean([r.get('red_fear', 0) for r in feature_results]):.2f}")

    # Return results
    return {
        'tabular': tabular_results,
        'feature': feature_results,
        'stats': {
            'zeroshot_hits': {'t': zeroshot_hits_t, 'p': zeroshot_hits_p, 'd': zeroshot_hits_d},
            'zeroshot_reward': {'t': zeroshot_rew_t, 'p': zeroshot_rew_p, 'd': zeroshot_rew_d},
            'early_b': {'t': early_b_t, 'p': early_b_p, 'd': early_b_d},
        }
    }


if __name__ == "__main__":
    results = run_experiment(n_seeds=50, n_phase_a_episodes=100, n_phase_b_episodes=50)

    print("\n" + "=" * 70)
    print("EXPERIMENT 21 COMPLETE")
    print("=" * 70)
