"""Experiment 18: Slippery Cliff (Safety / Fear)

A FAIR test environment for emotional learning where fear should help.

Environment:
- Modified CliffWalking: Path from Start to Goal flanked by cliff
- 20% chance agent moves in random direction (stochastic)
- Falling = -100 reward, episode terminates
- Standard navigation gives small negative rewards (time penalty)

Why Emotional Channels Should Help:
- Standard DQN averages Q-values, hugs cliff edge for optimality
- High stochasticity causes frequent deaths
- Fear channel spikes on negative reward / cliff proximity
- Acts as "safety bias" forcing safer sub-optimal path

Key Insight:
"Dense emotional signals are noise in dense reward environments,
but they become signal in sparse/harsh environments."

This environment is HARSH (cliff = death) and STOCHASTIC (20% slip).
Standard DQN will optimize for expected value and frequently die.
Fear-ED should learn safer paths even if slightly longer.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional
from scipy import stats


@dataclass
class Context:
    """Environment context for emotional computation."""
    near_cliff: bool
    cliff_distance: int
    reward: float
    fell: bool


class SlipperyCliffEnv:
    """
    Slippery Cliff Walking Environment.

    Grid layout (4x12):
    S = Start, G = Goal, C = Cliff (death), . = Safe path

    Row 0: . . . . . . . . . . . G
    Row 1: . . . . . . . . . . . .
    Row 2: . . . . . . . . . . . .
    Row 3: S C C C C C C C C C C .

    Agent starts at S, must reach G.
    The bottom row has cliffs - falling = -100 and episode ends.
    20% chance of slipping to random direction.
    """

    def __init__(self, slip_prob: float = 0.2, time_penalty: float = -1.0):
        self.height = 4
        self.width = 12
        self.slip_prob = slip_prob
        self.time_penalty = time_penalty

        # Positions
        self.start_pos = (3, 0)  # Bottom-left
        self.goal_pos = (0, 11)   # Top-right

        # Cliff positions: bottom row from column 1 to 10
        self.cliff_positions = {(3, c) for c in range(1, 11)}

        # Actions: 0=up, 1=right, 2=down, 3=left
        self.n_actions = 4
        self.action_deltas = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1),  # left
        }

        self.state = self.start_pos

    def reset(self) -> Tuple[int, int]:
        """Reset environment and return initial state."""
        self.state = self.start_pos
        return self.state

    def _get_cliff_distance(self, pos: Tuple[int, int]) -> int:
        """Manhattan distance to nearest cliff cell."""
        if not self.cliff_positions:
            return float('inf')
        return min(abs(pos[0] - c[0]) + abs(pos[1] - c[1])
                   for c in self.cliff_positions)

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Context]:
        """
        Take action with slip probability.

        Returns: (next_state, reward, done, context)
        """
        # Slip: with 20% probability, take random action instead
        if random.random() < self.slip_prob:
            action = random.randint(0, self.n_actions - 1)

        # Compute next position
        delta = self.action_deltas[action]
        next_row = self.state[0] + delta[0]
        next_col = self.state[1] + delta[1]

        # Boundary check
        next_row = max(0, min(self.height - 1, next_row))
        next_col = max(0, min(self.width - 1, next_col))
        next_pos = (next_row, next_col)

        # Check outcomes
        fell = next_pos in self.cliff_positions
        reached_goal = next_pos == self.goal_pos

        # Compute reward
        if fell:
            reward = -100.0
            done = True
            next_pos = self.start_pos  # Reset position (but episode done)
        elif reached_goal:
            reward = 10.0  # Positive reward for reaching goal
            done = True
        else:
            reward = self.time_penalty  # Small penalty for each step
            done = False

        self.state = next_pos

        # Context for emotional computation
        cliff_dist = self._get_cliff_distance(next_pos)
        context = Context(
            near_cliff=(cliff_dist <= 1),
            cliff_distance=cliff_dist,
            reward=reward,
            fell=fell
        )

        return next_pos, reward, done, context

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) to flat index."""
        return state[0] * self.width + state[1]

    @property
    def n_states(self) -> int:
        return self.height * self.width


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: int, action: int, reward: float, next_state: int,
             done: bool, context: Context):
        self.buffer.append((state, action, reward, next_state, done, context))

    def sample(self, batch_size: int) -> List:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNetwork(nn.Module):
    """Simple DQN network."""

    def __init__(self, n_states: int, n_actions: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StandardDQNAgent:
    """Standard DQN agent without emotional modulation."""

    def __init__(self, n_states: int, n_actions: int, device: torch.device,
                 lr: float = 1e-3, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Networks
        self.policy_net = DQNetwork(n_states, n_actions).to(device)
        self.target_net = DQNetwork(n_states, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

        self.batch_size = 64
        self.target_update_freq = 100
        self.steps = 0

    def _state_to_tensor(self, state: int) -> torch.Tensor:
        """One-hot encode state."""
        tensor = torch.zeros(self.n_states, device=self.device)
        tensor[state] = 1.0
        return tensor

    def select_action(self, state: int, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_tensor = self._state_to_tensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state: int, action: int, reward: float,
                        next_state: int, done: bool, context: Context):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done, context)

    def update(self):
        """Perform one step of optimization."""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, _ = zip(*batch)

        # Convert to tensors
        state_batch = torch.stack([self._state_to_tensor(s) for s in states])
        action_batch = torch.tensor(actions, device=self.device, dtype=torch.long)
        reward_batch = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_state_batch = torch.stack([self._state_to_tensor(s) for s in next_states])
        done_batch = torch.tensor(dones, device=self.device, dtype=torch.float32)

        # Compute current Q values
        current_q = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(dim=1)[0]
            target_q = reward_batch + self.gamma * next_q * (1 - done_batch)

        # Compute loss and optimize
        loss = nn.functional.mse_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class FearEDAgent:
    """
    Fear-driven Emotional ED Agent.

    Uses gradient blocking architecture:
    - Fear computed from context (cliff proximity, negative rewards)
    - Fear modulates as INPUT to Q-network (not gradient interference)
    - Higher fear → bias toward safer actions
    """

    def __init__(self, n_states: int, n_actions: int, device: torch.device,
                 lr: float = 1e-3, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 fear_weight: float = 0.5, max_fear: float = 1.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Fear parameters
        self.fear_weight = fear_weight
        self.max_fear = max_fear
        self.fear_level = 0.0  # Tonic fear (decays slowly)
        self.fear_decay = 0.9

        # Networks - input includes fear level (state augmentation)
        input_size = n_states + 1  # +1 for fear level
        self.policy_net = DQNetwork(input_size, n_actions).to(device)
        self.target_net = DQNetwork(input_size, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

        self.batch_size = 64
        self.target_update_freq = 100
        self.steps = 0

    def _compute_fear(self, context: Context) -> float:
        """
        Compute fear signal from context.

        Fear spikes when:
        - Near cliff (proximity danger)
        - After negative reward (pain signal)
        - After falling (trauma - persistent high fear)
        """
        phasic_fear = 0.0

        # Proximity-based fear
        if context.near_cliff:
            phasic_fear = self.max_fear
        elif context.cliff_distance <= 2:
            phasic_fear = self.max_fear * 0.5

        # Pain-triggered fear
        if context.reward < -10:  # Fell off cliff
            self.fear_level = min(1.0, self.fear_level + 0.5)

        # Combine tonic (persistent) and phasic (immediate) fear
        total_fear = max(self.fear_level, phasic_fear)

        # Decay tonic fear
        self.fear_level *= self.fear_decay

        return total_fear

    def _state_to_tensor(self, state: int, fear: float) -> torch.Tensor:
        """One-hot encode state with fear as additional input."""
        tensor = torch.zeros(self.n_states + 1, device=self.device)
        tensor[state] = 1.0
        tensor[-1] = fear  # Fear as last element
        return tensor

    def select_action(self, state: int, context: Optional[Context] = None,
                     training: bool = True) -> int:
        """
        Epsilon-greedy with fear-biased action selection.

        Fear biases selection away from actions toward cliff.
        """
        # Compute current fear
        fear = self._compute_fear(context) if context else self.fear_level

        if training and random.random() < self.epsilon:
            # Even random actions are fear-biased when fear is high
            if fear > 0.5 and random.random() < fear:
                # Prefer "away from danger" actions (up = 0)
                return 0  # Move up (away from cliff row)
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_tensor = self._state_to_tensor(state, fear).unsqueeze(0)
            q_values = self.policy_net(state_tensor)

            # Fear-biased action selection
            if fear > 0.3:
                # Boost Q-value of "safe" actions (up = 0)
                q_values[0, 0] += fear * self.fear_weight * 10

            return q_values.argmax(dim=1).item()

    def store_transition(self, state: int, action: int, reward: float,
                        next_state: int, done: bool, context: Context):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done, context)

    def update(self):
        """Perform one step of optimization with fear modulation."""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, contexts = zip(*batch)

        # Compute fears for each transition
        fears = torch.tensor([self._compute_fear(ctx) for ctx in contexts],
                            device=self.device, dtype=torch.float32)

        # Convert to tensors (with fear augmentation)
        state_batch = torch.stack([self._state_to_tensor(s, f.item())
                                   for s, f in zip(states, fears)])
        action_batch = torch.tensor(actions, device=self.device, dtype=torch.long)
        reward_batch = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_state_batch = torch.stack([self._state_to_tensor(s, f.item())
                                        for s, f in zip(next_states, fears)])
        done_batch = torch.tensor(dones, device=self.device, dtype=torch.float32)

        # Compute current Q values
        current_q = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(dim=1)[0]
            target_q = reward_batch + self.gamma * next_q * (1 - done_batch)

        # Fear-weighted loss: higher fear = more weight on avoiding negative outcomes
        fear_weights = 1 + fears * self.fear_weight
        loss = (fear_weights * (current_q.squeeze() - target_q) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def reset_fear(self):
        """Reset fear level at episode start."""
        self.fear_level = 0.0


def run_episode(env: SlipperyCliffEnv, agent, training: bool = True,
               max_steps: int = 200) -> dict:
    """Run one episode and return metrics."""
    state = env.reset()
    state_idx = env.state_to_index(state)

    total_reward = 0.0
    steps = 0
    fell = False
    reached_goal = False
    min_cliff_dist = float('inf')

    # Initial context
    context = Context(near_cliff=False, cliff_distance=env._get_cliff_distance(state),
                     reward=0.0, fell=False)

    # Reset fear for emotional agent
    if hasattr(agent, 'reset_fear'):
        agent.reset_fear()

    for step in range(max_steps):
        # Select action
        if hasattr(agent, 'select_action'):
            if isinstance(agent, FearEDAgent):
                action = agent.select_action(state_idx, context, training)
            else:
                action = agent.select_action(state_idx, training)

        # Take action
        next_state, reward, done, context = env.step(action)
        next_state_idx = env.state_to_index(next_state)

        # Store transition
        if training:
            agent.store_transition(state_idx, action, reward, next_state_idx,
                                  done, context)
            agent.update()

        # Track metrics
        total_reward += reward
        steps += 1
        min_cliff_dist = min(min_cliff_dist, context.cliff_distance)

        if context.fell:
            fell = True
        if next_state == env.goal_pos:
            reached_goal = True

        if done:
            break

        state = next_state
        state_idx = next_state_idx

    if training:
        agent.decay_epsilon()

    return {
        'reward': total_reward,
        'steps': steps,
        'fell': fell,
        'reached_goal': reached_goal,
        'min_cliff_dist': min_cliff_dist,
    }


def evaluate_agent(env: SlipperyCliffEnv, agent, n_episodes: int = 100) -> dict:
    """Evaluate agent performance without training."""
    metrics = {'rewards': [], 'survival': [], 'goals': [], 'steps': []}

    for _ in range(n_episodes):
        result = run_episode(env, agent, training=False)
        metrics['rewards'].append(result['reward'])
        metrics['survival'].append(1 if not result['fell'] else 0)
        metrics['goals'].append(1 if result['reached_goal'] else 0)
        metrics['steps'].append(result['steps'])

    return {
        'mean_reward': np.mean(metrics['rewards']),
        'std_reward': np.std(metrics['rewards']),
        'survival_rate': np.mean(metrics['survival']),
        'goal_rate': np.mean(metrics['goals']),
        'mean_steps': np.mean(metrics['steps']),
        'raw': metrics
    }


def run_experiment(n_seeds: int = 30, n_episodes: int = 500,
                   eval_episodes: int = 100, slip_prob: float = 0.2):
    """
    Run the Slippery Cliff experiment with statistical validation.

    Args:
        n_seeds: Number of random seeds for statistical validation
        n_episodes: Training episodes per seed
        eval_episodes: Evaluation episodes per seed
        slip_prob: Probability of slipping (stochasticity)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("=" * 70)
    print("EXPERIMENT 18: SLIPPERY CLIFF (Safety / Fear)")
    print("=" * 70)
    print(f"\nEnvironment: 4x12 CliffWalking with {slip_prob*100:.0f}% slip chance")
    print("Hypothesis: Fear-ED learns safer paths by avoiding cliff proximity")
    print(f"Running {n_seeds} seeds × {n_episodes} episodes")
    print()

    standard_results = []
    fearED_results = []

    # Learning curves for plotting
    standard_curves = {'survival': [], 'reward': []}
    fearED_curves = {'survival': [], 'reward': []}

    for seed in range(n_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        print(f"\rSeed {seed+1}/{n_seeds}", end='', flush=True)

        # Train Standard DQN
        env = SlipperyCliffEnv(slip_prob=slip_prob)
        standard_agent = StandardDQNAgent(env.n_states, env.n_actions, device)

        seed_survival = []
        seed_reward = []
        for ep in range(n_episodes):
            result = run_episode(env, standard_agent, training=True)
            seed_survival.append(1 if not result['fell'] else 0)
            seed_reward.append(result['reward'])

        standard_curves['survival'].append(seed_survival)
        standard_curves['reward'].append(seed_reward)

        # Evaluate
        standard_eval = evaluate_agent(env, standard_agent, eval_episodes)
        standard_results.append(standard_eval)

        # Train Fear-ED
        env = SlipperyCliffEnv(slip_prob=slip_prob)
        fearED_agent = FearEDAgent(env.n_states, env.n_actions, device)

        seed_survival = []
        seed_reward = []
        for ep in range(n_episodes):
            result = run_episode(env, fearED_agent, training=True)
            seed_survival.append(1 if not result['fell'] else 0)
            seed_reward.append(result['reward'])

        fearED_curves['survival'].append(seed_survival)
        fearED_curves['reward'].append(seed_reward)

        # Evaluate
        fearED_eval = evaluate_agent(env, fearED_agent, eval_episodes)
        fearED_results.append(fearED_eval)

    print("\n")

    # Aggregate results
    standard_survival = [r['survival_rate'] for r in standard_results]
    standard_goals = [r['goal_rate'] for r in standard_results]
    standard_rewards = [r['mean_reward'] for r in standard_results]

    fearED_survival = [r['survival_rate'] for r in fearED_results]
    fearED_goals = [r['goal_rate'] for r in fearED_results]
    fearED_rewards = [r['mean_reward'] for r in fearED_results]

    # Statistical tests
    survival_t, survival_p = stats.ttest_ind(fearED_survival, standard_survival)
    goals_t, goals_p = stats.ttest_ind(fearED_goals, standard_goals)
    reward_t, reward_p = stats.ttest_ind(fearED_rewards, standard_rewards)

    # Effect sizes (Cohen's d)
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        assert pooled_std > 0, f"BUG: Zero variance in cohens_d calculation"
        return (np.mean(group1) - np.mean(group2)) / pooled_std

    survival_d = cohens_d(fearED_survival, standard_survival)
    goals_d = cohens_d(fearED_goals, standard_goals)
    reward_d = cohens_d(fearED_rewards, standard_rewards)

    # Print results
    print("=" * 70)
    print("RESULTS: After Training")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Standard DQN':<20} {'Fear-ED':<20} {'p-value':<12} {'Cohen d':<10}")
    print("-" * 82)

    print(f"{'Survival Rate':<20} "
          f"{np.mean(standard_survival):.3f} ± {np.std(standard_survival):.3f}    "
          f"{np.mean(fearED_survival):.3f} ± {np.std(fearED_survival):.3f}    "
          f"{survival_p:.4f}      {survival_d:+.3f}")

    print(f"{'Goal Rate':<20} "
          f"{np.mean(standard_goals):.3f} ± {np.std(standard_goals):.3f}    "
          f"{np.mean(fearED_goals):.3f} ± {np.std(fearED_goals):.3f}    "
          f"{goals_p:.4f}      {goals_d:+.3f}")

    print(f"{'Mean Reward':<20} "
          f"{np.mean(standard_rewards):.1f} ± {np.std(standard_rewards):.1f}      "
          f"{np.mean(fearED_rewards):.1f} ± {np.std(fearED_rewards):.1f}      "
          f"{reward_p:.4f}      {reward_d:+.3f}")

    # Learning curve analysis (episodes to reach 70% survival)
    def episodes_to_threshold(curves, threshold=0.7, window=50):
        """Find first episode where rolling survival exceeds threshold."""
        episodes = []
        for curve in curves:
            found = False
            for i in range(window, len(curve)):
                rolling_avg = np.mean(curve[i-window:i])
                if rolling_avg >= threshold:
                    episodes.append(i)
                    found = True
                    break
            if not found:
                episodes.append(len(curve))  # Never reached
        return episodes

    standard_eps_to_safe = episodes_to_threshold(standard_curves['survival'], 0.7)
    fearED_eps_to_safe = episodes_to_threshold(fearED_curves['survival'], 0.7)

    print(f"\n{'Episodes to 70% survival':<25} "
          f"{np.mean(standard_eps_to_safe):.0f} ± {np.std(standard_eps_to_safe):.0f}    "
          f"{np.mean(fearED_eps_to_safe):.0f} ± {np.std(fearED_eps_to_safe):.0f}")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("\nHypothesis: Fear-ED should demonstrate safer behavior (higher survival)")
    print("           in stochastic cliff environment by avoiding cliff proximity.")

    if survival_p < 0.05 and survival_d > 0:
        print(f"\n✓ CONFIRMED: Fear-ED shows significantly higher survival")
        print(f"  Effect size: {survival_d:.3f} (", end="")
        if abs(survival_d) < 0.5:
            print("small)")
        elif abs(survival_d) < 0.8:
            print("medium)")
        else:
            print("large)")
    elif survival_p < 0.05 and survival_d < 0:
        print(f"\n✗ REVERSED: Standard DQN survives better (unexpected)")
        print(f"  Fear may be EXCESSIVE, causing suboptimal avoidance")
    else:
        print(f"\n~ INCONCLUSIVE: No significant survival difference (p={survival_p:.4f})")

    if np.mean(fearED_goals) > np.mean(standard_goals):
        print(f"\n✓ Fear-ED reaches goal more often ({np.mean(fearED_goals):.1%} vs {np.mean(standard_goals):.1%})")
    else:
        print(f"\n~ Goal rates similar or Standard DQN better")

    # Variance analysis (Fear should reduce variance)
    print("\n" + "-" * 40)
    print("Variance Analysis (Fear should REDUCE variance):")
    print(f"  Standard reward variance: {np.var(standard_rewards):.1f}")
    print(f"  Fear-ED reward variance:  {np.var(fearED_rewards):.1f}")

    if np.var(fearED_rewards) < np.var(standard_rewards):
        reduction = (1 - np.var(fearED_rewards) / np.var(standard_rewards)) * 100
        print(f"  → Fear-ED reduces variance by {reduction:.0f}%")

    # Return results for further analysis
    return {
        'standard': {
            'survival': standard_survival,
            'goals': standard_goals,
            'rewards': standard_rewards,
            'curves': standard_curves,
        },
        'fearED': {
            'survival': fearED_survival,
            'goals': fearED_goals,
            'rewards': fearED_rewards,
            'curves': fearED_curves,
        },
        'stats': {
            'survival': {'t': survival_t, 'p': survival_p, 'd': survival_d},
            'goals': {'t': goals_t, 'p': goals_p, 'd': goals_d},
            'rewards': {'t': reward_t, 'p': reward_p, 'd': reward_d},
        }
    }


if __name__ == "__main__":
    results = run_experiment(n_seeds=30, n_episodes=500, eval_episodes=100)

    print("\n" + "=" * 70)
    print("EXPERIMENT 18 COMPLETE")
    print("=" * 70)
