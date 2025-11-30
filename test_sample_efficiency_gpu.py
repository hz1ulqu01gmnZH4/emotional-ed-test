"""Experiment 16b: Sample Efficiency Comparison (GPU/Neural Network Version).

GPU-accelerated version using PyTorch with neural network function approximation.
Tests whether emotional channels provide faster learning in deep RL setting.

Hypothesis:
- Emotional ED's broadcast modulation may show stronger effects with neural nets
- Direct emotional signals may improve sample efficiency in function approximation
- GPU parallelization enables larger-scale testing

Requirements:
    uv pip install torch

Run with:
    uv run python test_sample_efficiency_gpu.py
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: uv pip install torch")


@dataclass
class Context:
    """Environment context for emotional processing."""
    threat_distance: float = float('inf')
    was_blocked: bool = False
    goal_distance: float = float('inf')
    reward: float = 0.0


class GridWorldGPU:
    """GPU-compatible gridworld environment."""

    def __init__(self, size: int = 5, device: str = 'cuda'):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, down, left, right
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Positions
        self.threat_pos = (size // 2, size // 2)  # Center
        self.goal_pos = (size - 1, size - 1)  # Bottom-right
        self.start_pos = (0, 0)  # Top-left

        self.agent_pos = self.start_pos

    def reset(self) -> int:
        self.agent_pos = self.start_pos
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.size + pos[1]

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        return (state // self.size, state % self.size)

    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self, action: int) -> Tuple[int, float, bool, Context]:
        # Movement deltas
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        dx, dy = deltas[action]

        new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))

        was_blocked = (new_x, new_y) == self.agent_pos
        self.agent_pos = (new_x, new_y)

        # Compute distances
        threat_dist = self._distance(self.agent_pos, self.threat_pos)
        goal_dist = self._distance(self.agent_pos, self.goal_pos)

        # Reward
        reward = 0.0
        done = False

        if self.agent_pos == self.goal_pos:
            reward = 1.0
            done = True
        elif self.agent_pos == self.threat_pos:
            reward = -1.0
            done = True
        else:
            reward = -0.01  # Step penalty

        ctx = Context(
            threat_distance=threat_dist,
            was_blocked=was_blocked,
            goal_distance=goal_dist,
            reward=reward
        )

        return self._pos_to_state(self.agent_pos), reward, done, ctx

    def get_state_features(self, state: int) -> torch.Tensor:
        """Convert state to feature vector for neural network."""
        pos = self._state_to_pos(state)

        # Normalized features
        features = [
            pos[0] / self.size,  # x position
            pos[1] / self.size,  # y position
            self._distance(pos, self.threat_pos) / (2 * self.size),  # threat distance
            self._distance(pos, self.goal_pos) / (2 * self.size),  # goal distance
        ]

        return torch.tensor(features, dtype=torch.float32, device=self.device)


class DQNNetwork(nn.Module):
    """Deep Q-Network for function approximation."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int = 10000, device: str = 'cuda'):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0

    def push(self, state: torch.Tensor, action: int, reward: float,
             next_state: torch.Tensor, done: bool, fear: float = 0.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, fear)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = torch.stack([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch], device=self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([b[3] for b in batch])
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)
        fears = torch.tensor([b[5] for b in batch], dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones, fears

    def __len__(self):
        return len(self.buffer)


class StandardDQNAgent:
    """Standard DQN agent without emotional modulation."""

    def __init__(self, input_dim: int, n_actions: int, device: str = 'cuda',
                 lr: float = 1e-3, gamma: float = 0.99, epsilon: float = 0.1):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_network = DQNNetwork(input_dim, n_actions).to(device)
        self.target_network = DQNNetwork(input_dim, n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(device=device)

        self.update_counter = 0
        self.target_update_freq = 100

    def select_action(self, state_features: torch.Tensor) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            q_values = self.q_network(state_features.unsqueeze(0))
            return q_values.argmax().item()

    def update(self, state: torch.Tensor, action: int, reward: float,
               next_state: torch.Tensor, done: bool, ctx: Context):
        self.buffer.push(state, action, reward, next_state, done)

        if len(self.buffer) < 64:
            return

        # Sample batch
        states, actions, rewards, next_states, dones, _ = self.buffer.sample(64)

        # Compute Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Standard MSE loss
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())


class EmotionalDQNAgent:
    """DQN agent with emotional ED modulation (fear channel).

    Key differences from standard DQN:
    1. Fear signal modulates learning rate (higher fear = faster learning from threats)
    2. Fear signal modulates action selection (higher fear = more risk aversion)
    3. Emotional broadcast affects network updates (not just reward shaping)
    """

    def __init__(self, input_dim: int, n_actions: int, device: str = 'cuda',
                 lr: float = 1e-3, gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 0.5):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.base_epsilon = epsilon
        self.fear_weight = fear_weight

        # Extended input to include fear signal
        self.q_network = DQNNetwork(input_dim + 1, n_actions).to(device)  # +1 for fear
        self.target_network = DQNNetwork(input_dim + 1, n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.base_lr = lr
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(device=device)

        self.update_counter = 0
        self.target_update_freq = 100

        self.current_fear = 0.0
        self.safe_distance = 2.0
        self.max_fear = 1.0

    def _compute_fear(self, ctx: Context) -> float:
        """Compute fear signal from context."""
        if ctx.threat_distance >= self.safe_distance:
            return 0.0
        return self.max_fear * (1 - ctx.threat_distance / self.safe_distance)

    def _augment_features(self, features: torch.Tensor, fear: float) -> torch.Tensor:
        """Add fear signal to feature vector."""
        fear_tensor = torch.tensor([fear], dtype=torch.float32, device=self.device)
        return torch.cat([features, fear_tensor])

    def select_action(self, state_features: torch.Tensor, ctx: Optional[Context] = None) -> int:
        # Compute fear
        fear = self._compute_fear(ctx) if ctx else self.current_fear
        self.current_fear = fear

        # Fear modulates exploration (more fear = less exploration near threats)
        effective_epsilon = self.base_epsilon * (1 - fear * 0.5)

        if np.random.random() < effective_epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            augmented = self._augment_features(state_features, fear)
            q_values = self.q_network(augmented.unsqueeze(0))

            # Fear biases toward safer actions (higher Q variance aversion)
            if fear > 0.3:
                # Softer action selection under fear
                probs = F.softmax(q_values / (1 + fear), dim=1)
                return torch.multinomial(probs, 1).item()

            return q_values.argmax().item()

    def update(self, state: torch.Tensor, action: int, reward: float,
               next_state: torch.Tensor, done: bool, ctx: Context):
        fear = self._compute_fear(ctx)

        # Augment states with fear
        aug_state = self._augment_features(state, fear)
        aug_next = self._augment_features(next_state, fear)

        self.buffer.push(aug_state, action, reward, aug_next, done, fear)

        if len(self.buffer) < 64:
            return

        # Sample batch
        states, actions, rewards, next_states, dones, fears = self.buffer.sample(64)

        # Compute Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Emotional modulation of loss
        # Higher fear = higher learning rate for negative outcomes
        fear_modulation = 1 + fears * self.fear_weight
        weights = torch.where(target_q < current_q, fear_modulation, torch.ones_like(fears))

        # Weighted MSE loss
        loss = (weights * (current_q - target_q) ** 2).mean()

        # Dynamic learning rate based on average fear in batch
        avg_fear = fears.mean().item()
        effective_lr = self.base_lr * (1 + avg_fear * self.fear_weight)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = effective_lr

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())


def run_episode(env: GridWorldGPU, agent, max_steps: int = 100) -> Dict:
    """Run single episode and return metrics."""
    state = env.reset()
    state_features = env.get_state_features(state)

    total_reward = 0.0
    min_threat_dist = float('inf')

    # Initial context for first action
    ctx = Context(
        threat_distance=env._distance(env.agent_pos, env.threat_pos),
        goal_distance=env._distance(env.agent_pos, env.goal_pos)
    )

    for step in range(max_steps):
        if isinstance(agent, EmotionalDQNAgent):
            action = agent.select_action(state_features, ctx)
        else:
            action = agent.select_action(state_features)

        next_state, reward, done, ctx = env.step(action)
        next_features = env.get_state_features(next_state)

        agent.update(state_features, action, reward, next_features, done, ctx)

        total_reward += reward
        min_threat_dist = min(min_threat_dist, ctx.threat_distance)

        if done:
            break

        state_features = next_features

    return {
        'reward': total_reward,
        'threat_dist': min_threat_dist,
        'steps': step + 1
    }


def measure_learning_curve_gpu(agent_class, n_seeds: int = 30, n_episodes: int = 500,
                               device: str = 'cuda') -> Dict:
    """Measure learning curve for agent type (GPU version)."""
    reward_curves = []
    threat_curves = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = GridWorldGPU(device=device)
        agent = agent_class(input_dim=4, n_actions=4, device=device)

        episode_rewards = []
        episode_threats = []

        for ep in range(n_episodes):
            result = run_episode(env, agent)
            episode_rewards.append(result['reward'])
            episode_threats.append(result['threat_dist'])

        reward_curves.append(episode_rewards)
        threat_curves.append(episode_threats)

    return {
        'reward_mean': np.mean(reward_curves, axis=0),
        'reward_std': np.std(reward_curves, axis=0),
        'reward_raw': np.array(reward_curves),
        'threat_mean': np.mean(threat_curves, axis=0),
        'threat_std': np.std(threat_curves, axis=0),
        'threat_raw': np.array(threat_curves)
    }


def episodes_to_criterion(curves: np.ndarray, criterion: float,
                          direction: str = 'above', window: int = 20) -> List[int]:
    """Find episode where criterion is first met (rolling average)."""
    n_seeds, n_episodes = curves.shape
    episodes = []

    for seed in range(n_seeds):
        found = False
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
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch required. Install with: uv pip install torch")
        return

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("=" * 70)
    print("EXPERIMENT 16b: SAMPLE EFFICIENCY (GPU/NEURAL NETWORK)")
    print("=" * 70)

    print("\nHYPOTHESIS: Emotional ED shows stronger effects with neural networks")
    print("- Standard DQN: Learns via reward signal only")
    print("- Emotional DQN: Fear modulates learning rate, action selection, loss weighting")
    print()

    n_seeds = 30
    n_episodes = 500

    # Test: Fear Learning with Neural Networks
    print("=" * 70)
    print("TEST: Fear Channel with Deep Q-Learning")
    print("=" * 70)
    print(f"\nRunning {n_seeds} seeds × {n_episodes} episodes each...")

    start_time = time.time()

    print("\nTraining Standard DQN...")
    standard_results = measure_learning_curve_gpu(
        StandardDQNAgent, n_seeds=n_seeds, n_episodes=n_episodes, device=device
    )

    print("Training Emotional DQN...")
    emotional_results = measure_learning_curve_gpu(
        EmotionalDQNAgent, n_seeds=n_seeds, n_episodes=n_episodes, device=device
    )

    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed:.1f}s")

    # Analyze threat avoidance
    print("\n" + "=" * 70)
    print("RESULTS: Threat Avoidance (min threat distance)")
    print("=" * 70)

    std_threat_eps = episodes_to_criterion(standard_results['threat_raw'], 1.5, 'above')
    emo_threat_eps = episodes_to_criterion(emotional_results['threat_raw'], 1.5, 'above')

    print(f"\nEpisodes to safe behavior (threat dist > 1.5):")
    print(f"  Standard DQN:  {np.mean(std_threat_eps):.1f} ± {np.std(std_threat_eps):.1f}")
    print(f"  Emotional DQN: {np.mean(emo_threat_eps):.1f} ± {np.std(emo_threat_eps):.1f}")

    if np.mean(emo_threat_eps) < np.mean(std_threat_eps):
        speedup = np.mean(std_threat_eps) / np.mean(emo_threat_eps)
        print(f"  → Emotional DQN is {speedup:.2f}x faster")
    else:
        ratio = np.mean(emo_threat_eps) / np.mean(std_threat_eps)
        print(f"  → Standard DQN is {ratio:.2f}x faster")

    print(f"\nMean threat distance at key episodes:")
    print(f"{'Episode':<10} {'Standard DQN':<20} {'Emotional DQN':<20}")
    print("-" * 55)
    for ep in [50, 100, 200, 300, 500]:
        idx = ep - 1
        std_mean = standard_results['threat_mean'][idx]
        std_std = standard_results['threat_std'][idx]
        emo_mean = emotional_results['threat_mean'][idx]
        emo_std = emotional_results['threat_std'][idx]
        print(f"{ep:<10} {std_mean:.3f} ± {std_std:.3f}      {emo_mean:.3f} ± {emo_std:.3f}")

    # Analyze reward learning
    print("\n" + "=" * 70)
    print("RESULTS: Reward Learning")
    print("=" * 70)

    std_reward_eps = episodes_to_criterion(standard_results['reward_raw'], 0.5, 'above')
    emo_reward_eps = episodes_to_criterion(emotional_results['reward_raw'], 0.5, 'above')

    print(f"\nEpisodes to good reward (> 0.5):")
    print(f"  Standard DQN:  {np.mean(std_reward_eps):.1f} ± {np.std(std_reward_eps):.1f}")
    print(f"  Emotional DQN: {np.mean(emo_reward_eps):.1f} ± {np.std(emo_reward_eps):.1f}")

    print(f"\nMean reward at key episodes:")
    print(f"{'Episode':<10} {'Standard DQN':<20} {'Emotional DQN':<20}")
    print("-" * 55)
    for ep in [50, 100, 200, 300, 500]:
        idx = ep - 1
        std_mean = standard_results['reward_mean'][idx]
        std_std = standard_results['reward_std'][idx]
        emo_mean = emotional_results['reward_mean'][idx]
        emo_std = emotional_results['reward_std'][idx]
        print(f"{ep:<10} {std_mean:.3f} ± {std_std:.3f}      {emo_mean:.3f} ± {emo_std:.3f}")

    # Statistical comparison
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)

    # Final performance comparison
    final_std_threat = standard_results['threat_raw'][:, -50:].mean(axis=1)
    final_emo_threat = emotional_results['threat_raw'][:, -50:].mean(axis=1)

    final_std_reward = standard_results['reward_raw'][:, -50:].mean(axis=1)
    final_emo_reward = emotional_results['reward_raw'][:, -50:].mean(axis=1)

    # Cohen's d
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        pooled_std = np.sqrt(((nx-1)*np.std(x)**2 + (ny-1)*np.std(y)**2) / (nx+ny-2))
        return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0

    # Permutation test
    def permutation_test(x, y, n_perm=1000):
        observed = abs(np.mean(x) - np.mean(y))
        combined = np.concatenate([x, y])
        count = 0
        for _ in range(n_perm):
            np.random.shuffle(combined)
            perm_diff = abs(np.mean(combined[:len(x)]) - np.mean(combined[len(x):]))
            if perm_diff >= observed:
                count += 1
        return max(count / n_perm, 1/n_perm)

    print(f"\nFinal Performance (last 50 episodes):")
    print(f"\nThreat Distance:")
    print(f"  Standard: {final_std_threat.mean():.3f} ± {final_std_threat.std():.3f}")
    print(f"  Emotional: {final_emo_threat.mean():.3f} ± {final_emo_threat.std():.3f}")
    d_threat = cohens_d(final_emo_threat, final_std_threat)
    p_threat = permutation_test(final_emo_threat, final_std_threat)
    print(f"  Cohen's d: {d_threat:.3f}, p-value: {p_threat:.3f}")

    print(f"\nReward:")
    print(f"  Standard: {final_std_reward.mean():.3f} ± {final_std_reward.std():.3f}")
    print(f"  Emotional: {final_emo_reward.mean():.3f} ± {final_emo_reward.std():.3f}")
    d_reward = cohens_d(final_emo_reward, final_std_reward)
    p_reward = permutation_test(final_emo_reward, final_std_reward)
    print(f"  Cohen's d: {d_reward:.3f}, p-value: {p_reward:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    threat_speedup = np.mean(std_threat_eps) / np.mean(emo_threat_eps) if np.mean(emo_threat_eps) > 0 else 0
    reward_speedup = np.mean(std_reward_eps) / np.mean(emo_reward_eps) if np.mean(emo_reward_eps) > 0 else 0

    print(f"\n{'Metric':<25} {'Speedup':<15} {'Effect Size':<15} {'p-value':<10}")
    print("-" * 65)
    print(f"{'Threat Avoidance':<25} {threat_speedup:.2f}x          d={d_threat:.2f}          {p_threat:.3f}")
    print(f"{'Reward Learning':<25} {reward_speedup:.2f}x          d={d_reward:.2f}          {p_reward:.3f}")

    avg_speedup = (threat_speedup + reward_speedup) / 2
    print(f"\nAverage speedup: {avg_speedup:.2f}x")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if avg_speedup > 1.5 and (p_threat < 0.05 or p_reward < 0.05):
        print("\n✓ Emotional ED shows SIGNIFICANT sample efficiency advantage")
        print("  with neural network function approximation.")
    elif avg_speedup > 1.1:
        print("\n~ Emotional ED shows MODEST sample efficiency advantage")
        print("  with neural network function approximation.")
    else:
        print("\n✗ No significant sample efficiency advantage demonstrated")
        print("  in neural network setting.")

    print(f"\nKey finding: Neural network + emotional modulation")
    if d_threat > 0.5:
        print(f"  produces medium-large effect on threat avoidance (d={d_threat:.2f})")
    else:
        print(f"  produces small effect on threat avoidance (d={d_threat:.2f})")


if __name__ == "__main__":
    main()
