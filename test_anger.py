"""Test anger/frustration: Does frustration cause persistence at blocked paths?

Hypothesis (Davidson, Berkowitz):
- Anger is approach-motivated negative affect
- Frustrated agents should TRY HARDER before giving up
- Standard RL: blocked → immediately seek alternate path
- Emotional ED: blocked → persist → eventually reroute

Key metric: "attempts at wall" before finding alternate route
"""

import numpy as np
from gridworld_anger import BlockedPathGridWorld
from agents_anger import StandardQLearner, FrustrationEDAgent

def run_episode(env: BlockedPathGridWorld, agent, max_steps: int = 100):
    """Run episode, tracking wall-hitting behavior."""
    state = env.reset()
    agent.reset_episode()

    total_reward = 0
    wall_hits = 0
    steps_to_reroute = None  # When agent first goes around wall

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, context = env.step(action)
        agent.update(state, action, reward, next_state, done, context)

        total_reward += reward
        if context.was_blocked:
            wall_hits += 1

        # Detect rerouting: agent moved away from wall area
        if steps_to_reroute is None:
            # Check if agent is going around (y > 3 or x > 3 and not blocked)
            pos = env.agent_pos
            if (pos[0] > 3 or pos[1] > 3) and not context.was_blocked:
                steps_to_reroute = step

        if done:
            break
        state = next_state

    return {
        'reward': total_reward,
        'steps': step + 1,
        'wall_hits': wall_hits,
        'steps_to_reroute': steps_to_reroute,
        'success': done
    }


def train_and_evaluate(env: BlockedPathGridWorld, agent, train_episodes: int = 300,
                       eval_episodes: int = 50):
    """Train agent, then evaluate persistence behavior."""

    # Training phase
    for _ in range(train_episodes):
        run_episode(env, agent)

    # Evaluation phase - measure persistence
    agent.epsilon = 0  # Greedy for evaluation
    results = []

    for _ in range(eval_episodes):
        result = run_episode(env, agent)
        results.append(result)

    return {
        'mean_reward': np.mean([r['reward'] for r in results]),
        'mean_steps': np.mean([r['steps'] for r in results]),
        'mean_wall_hits': np.mean([r['wall_hits'] for r in results]),
        'success_rate': np.mean([r['success'] for r in results]),
        'results': results
    }


def analyze_early_behavior(env: BlockedPathGridWorld, agent, n_episodes: int = 20):
    """Analyze behavior in early episodes (before learning alternate path).

    This is where frustration effect should be most visible:
    - Standard: Quickly learns wall is bad, stops trying
    - Frustrated: Keeps hitting wall longer (persistence)
    """
    agent.epsilon = 0.2  # Some exploration
    wall_hits_per_episode = []

    for ep in range(n_episodes):
        state = env.reset()
        agent.reset_episode()
        episode_wall_hits = 0

        for step in range(50):
            action = agent.select_action(state)
            next_state, reward, done, context = env.step(action)
            agent.update(state, action, reward, next_state, done, context)

            if context.was_blocked:
                episode_wall_hits += 1

            if done:
                break
            state = next_state

        wall_hits_per_episode.append(episode_wall_hits)

    return wall_hits_per_episode


def main():
    np.random.seed(42)

    # Environment: wall blocks direct path, permanent walls
    # Agent starts at (0,0), goal at (4,4)
    # Walls at (2,2), (2,3), (3,2) block diagonal
    env = BlockedPathGridWorld(size=5, wall_duration=0)  # Permanent walls

    print("=" * 60)
    print("EMOTIONAL ED TEST: Anger/Frustration Channel")
    print("=" * 60)
    print("\nGrid layout (# = wall):")
    print(env.render())
    print()
    print("HYPOTHESIS:")
    print("- Standard agent: Blocked → quickly avoid → find alternate path")
    print("- Frustrated agent: Blocked → persist longer → eventually reroute")
    print("- Key metric: Wall hits during learning (persistence)")
    print()

    # Test 1: Early learning behavior (most sensitive to frustration)
    print("=" * 60)
    print("TEST 1: Early Learning Behavior (first 20 episodes)")
    print("=" * 60)

    # Fresh agents
    standard = StandardQLearner(env.n_states, env.n_actions)
    frustrated = FrustrationEDAgent(env.n_states, env.n_actions, anger_weight=0.8)

    std_early = analyze_early_behavior(env, standard, n_episodes=20)
    env2 = BlockedPathGridWorld(size=5, wall_duration=0)
    frust_early = analyze_early_behavior(env2, frustrated, n_episodes=20)

    print(f"\n{'Episode':<10} {'Standard Hits':<15} {'Frustrated Hits':<15}")
    print("-" * 40)
    for i in range(min(10, len(std_early))):
        print(f"{i+1:<10} {std_early[i]:<15} {frust_early[i]:<15}")

    print(f"\n{'Total wall hits (20 eps)':<25} {sum(std_early):<15} {sum(frust_early):<15}")
    print(f"{'Mean hits per episode':<25} {np.mean(std_early):<15.2f} {np.mean(frust_early):<15.2f}")

    # Test 2: Converged behavior
    print("\n" + "=" * 60)
    print("TEST 2: Converged Behavior (after 300 training episodes)")
    print("=" * 60)

    # Fresh agents, full training
    standard = StandardQLearner(env.n_states, env.n_actions)
    frustrated = FrustrationEDAgent(env.n_states, env.n_actions, anger_weight=0.8)

    std_results = train_and_evaluate(env, standard)

    env2 = BlockedPathGridWorld(size=5, wall_duration=0)
    frust_results = train_and_evaluate(env2, frustrated)

    print(f"\n{'Metric':<25} {'Standard':<15} {'Frustrated ED':<15}")
    print("-" * 55)
    print(f"{'Success rate':<25} {std_results['success_rate']:<15.1%} {frust_results['success_rate']:<15.1%}")
    print(f"{'Mean steps':<25} {std_results['mean_steps']:<15.1f} {frust_results['mean_steps']:<15.1f}")
    print(f"{'Mean wall hits':<25} {std_results['mean_wall_hits']:<15.2f} {frust_results['mean_wall_hits']:<15.2f}")

    # Hypothesis test
    print("\n" + "=" * 60)
    print("HYPOTHESIS TEST")
    print("=" * 60)

    early_diff = sum(frust_early) - sum(std_early)
    print(f"\nEarly learning wall hits difference: {early_diff:+d}")

    if early_diff > 5:
        print("✓ Frustrated agent shows MORE persistence at wall")
        print("  → Anger channel produces approach-under-negative-valence")
    elif early_diff < -5:
        print("✗ Frustrated agent shows LESS persistence (unexpected)")
    else:
        print("~ Similar persistence levels")

    # Visualize learned paths
    print("\n" + "=" * 60)
    print("LEARNED PATHS")
    print("=" * 60)

    for name, agent in [("Standard", standard), ("Frustrated", frustrated)]:
        env_vis = BlockedPathGridWorld(size=5, wall_duration=0)
        state = env_vis.reset()
        agent.reset_episode()
        agent.epsilon = 0

        path = [tuple(env_vis.agent_pos)]
        for _ in range(20):
            action = agent.select_action(state)
            state, _, done, _ = env_vis.step(action)
            path.append(tuple(env_vis.agent_pos))
            if done:
                break

        print(f"\n{name} agent path:")
        grid = [['.' for _ in range(5)] for _ in range(5)]
        for wx, wy in env_vis.wall_positions:
            grid[wx][wy] = '#'
        grid[4][4] = 'G'
        for i, (x, y) in enumerate(path[:-1]):
            if grid[x][y] == '.':
                grid[x][y] = str(i) if i < 10 else '+'
        print('\n'.join([' '.join(row) for row in grid]))
        print(f"Steps: {len(path) - 1}")


if __name__ == "__main__":
    main()
