"""Compare standard Q-learning vs emotional ED agent on fear-relevant task."""

import numpy as np
from gridworld import GridWorld
from agents import StandardQLearner, EmotionalEDAgent

def run_episode(env: GridWorld, agent, max_steps: int = 100, train: bool = True,
                use_threat_penalty: bool = False):
    """Run single episode, return (total_reward, steps, min_threat_distance, path)."""
    state = env.reset()
    if hasattr(agent, 'reset_episode'):
        agent.reset_episode()

    total_reward = 0
    min_threat_dist = float('inf')
    path = [tuple(env.agent_pos)]

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, context = env.step(action, include_threat_penalty=use_threat_penalty)

        if train:
            agent.update(state, action, reward, next_state, done, context)

        total_reward += reward
        min_threat_dist = min(min_threat_dist, context.threat_distance)
        path.append(tuple(env.agent_pos))

        if done:
            break
        state = next_state

    return total_reward, step + 1, min_threat_dist, path


def evaluate(env: GridWorld, agent, n_episodes: int = 100):
    """Evaluate agent without training."""
    rewards, steps, threat_dists = [], [], []

    for _ in range(n_episodes):
        r, s, d, _ = run_episode(env, agent, train=False)
        rewards.append(r)
        steps.append(s)
        threat_dists.append(d)

    return {
        'mean_reward': np.mean(rewards),
        'mean_steps': np.mean(steps),
        'mean_min_threat_dist': np.mean(threat_dists),
        'success_rate': sum(1 for r in rewards if r > 0) / n_episodes
    }


def train_agent(env: GridWorld, agent, n_episodes: int = 500):
    """Train agent."""
    for _ in range(n_episodes):
        run_episode(env, agent, train=True)


def visualize_path(env: GridWorld, agent):
    """Show agent path with full trace."""
    state = env.reset()
    if hasattr(agent, 'reset_episode'):
        agent.reset_episode()

    path = [tuple(env.agent_pos)]

    for step in range(50):
        action = agent.select_action(state)
        state, _, done, _ = env.step(action)
        path.append(tuple(env.agent_pos))
        if done:
            break

    # Render with path
    grid = [['.' for _ in range(env.size)] for _ in range(env.size)]
    grid[env.threat_pos[0]][env.threat_pos[1]] = 'X'
    grid[env.goal_pos[0]][env.goal_pos[1]] = 'G'

    # Mark path with numbers
    for i, (x, y) in enumerate(path[:-1]):  # Exclude goal
        if grid[x][y] == '.':
            grid[x][y] = str(i) if i < 10 else '+'

    print("Path (numbers show order):")
    print('\n'.join([' '.join(row) for row in grid]))
    print(f"\nPath: {' → '.join([str(p) for p in path])}")
    print(f"Steps: {len(path) - 1}")


def main():
    np.random.seed(42)

    # Environment: threat ON the diagonal path, forcing choice
    # Threat at (2,2) blocks direct diagonal path from (0,0) to (4,4)
    env = GridWorld(size=5, threat_pos=(2, 2))

    print("=" * 50)
    print("EMOTIONAL ED TEST: Fear Channel")
    print("=" * 50)
    print(f"\nGrid: {env.size}x{env.size}")
    print(f"Threat at: {tuple(env.threat_pos)}")
    print(f"Goal at: {tuple(env.goal_pos)}")
    print("\nInitial grid:")
    print(env.render())
    print()

    print("TEST DESIGN:")
    print("- Standard agent: NO threat penalty in reward")
    print("- Emotional agent: Fear channel provides threat signal")
    print("- Question: Does fear alone cause threat avoidance?")
    print()

    # Create agents
    standard = StandardQLearner(env.n_states, env.n_actions)
    emotional = EmotionalEDAgent(env.n_states, env.n_actions, fear_weight=1.0)

    # Train standard WITHOUT threat penalty (blind to threat)
    print("Training standard Q-learner (no threat signal)...")
    for _ in range(500):
        run_episode(env, standard, train=True, use_threat_penalty=False)

    # Train emotional WITHOUT threat penalty (only fear channel sees threat)
    print("Training emotional ED agent (fear channel only)...")
    for _ in range(500):
        run_episode(env, emotional, train=True, use_threat_penalty=False)

    # Evaluate
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    # Set epsilon to 0 for evaluation
    standard.epsilon = 0
    emotional.epsilon = 0

    std_results = evaluate(env, standard)
    emo_results = evaluate(env, emotional)

    print(f"\n{'Metric':<25} {'Standard':<15} {'Emotional ED':<15}")
    print("-" * 55)
    print(f"{'Mean reward':<25} {std_results['mean_reward']:<15.3f} {emo_results['mean_reward']:<15.3f}")
    print(f"{'Mean steps':<25} {std_results['mean_steps']:<15.1f} {emo_results['mean_steps']:<15.1f}")
    print(f"{'Mean min threat dist':<25} {std_results['mean_min_threat_dist']:<15.2f} {emo_results['mean_min_threat_dist']:<15.2f}")
    print(f"{'Success rate':<25} {std_results['success_rate']:<15.1%} {emo_results['success_rate']:<15.1%}")

    # Key metric: does emotional agent maintain greater distance from threat?
    print("\n" + "=" * 50)
    print("HYPOTHESIS TEST")
    print("=" * 50)
    dist_diff = emo_results['mean_min_threat_dist'] - std_results['mean_min_threat_dist']
    print(f"\nThreat distance difference: {dist_diff:+.3f}")

    if dist_diff > 0.1:
        print("✓ Emotional agent maintains GREATER distance from threat")
        print("  → Fear channel produces risk-averse behavior WITHOUT explicit reward")
    elif dist_diff < -0.1:
        print("✗ Emotional agent gets CLOSER to threat (unexpected)")
    else:
        print("~ No significant difference in threat avoidance")

    # Visualize paths
    print("\n" + "=" * 50)
    print("STANDARD AGENT PATH")
    print("=" * 50)
    visualize_path(env, standard)

    print("\n" + "=" * 50)
    print("EMOTIONAL ED AGENT PATH")
    print("=" * 50)
    visualize_path(env, emotional)


if __name__ == "__main__":
    main()
