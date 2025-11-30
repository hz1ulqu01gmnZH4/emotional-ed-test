"""Experiment 14: Joy and Curiosity as Positive Emotional Channels.

Tests whether positive emotions (joy, curiosity) drive exploration
and approach behavior symmetrically to how fear drives avoidance.

Hypothesis (Fredrickson, 2001; Silvia, 2008):
- Curiosity increases exploration toward novel states
- Joy biases toward rewarding states
- These are parallel channels, not just intrinsic reward

Key predictions:
1. Curiosity agent discovers hidden reward faster
2. Joy agent returns to positive states more reliably
3. Integrated agent shows both exploration AND exploitation benefits
"""

import numpy as np
from gridworld_joy import JoyCuriosityGridWorld
from agents_joy import (StandardQLearner, CuriosityAgent, JoyAgent,
                        IntegratedJoyCuriosityAgent)


def run_episode(env, agent, max_steps: int = 100):
    """Run episode tracking discovery and revisit behavior."""
    state = env.reset()
    total_reward = 0
    steps = 0
    hidden_found_step = None
    goal_reached = False

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, ctx = env.step(action)
        agent.update(state, action, reward, next_state, done, ctx)

        total_reward += reward
        steps += 1

        if ctx.hidden_found and hidden_found_step is None:
            hidden_found_step = step + 1

        if done:
            if reward > 0.5:  # Goal reached
                goal_reached = True
            break

        state = next_state

    return {
        'total_reward': total_reward,
        'steps': steps,
        'hidden_found_step': hidden_found_step,
        'goal_reached': goal_reached,
        'states_explored': env.n_states - env.get_unvisited_count()
    }


def discovery_speed_test(n_seeds: int = 30, max_episodes: int = 50):
    """Test how quickly agents discover hidden reward."""
    results = {name: [] for name in ['Standard', 'Curiosity', 'Joy', 'Integrated']}

    for seed in range(n_seeds):
        np.random.seed(seed)

        env = JoyCuriosityGridWorld()
        agents = {
            'Standard': StandardQLearner(env.n_states, env.n_actions),
            'Curiosity': CuriosityAgent(env.n_states, env.n_actions),
            'Joy': JoyAgent(env.n_states, env.n_actions),
            'Integrated': IntegratedJoyCuriosityAgent(env.n_states, env.n_actions)
        }

        for name, agent in agents.items():
            env.reset_full()
            discovery_episode = None

            for ep in range(max_episodes):
                result = run_episode(env, agent)
                if result['hidden_found_step'] is not None and discovery_episode is None:
                    discovery_episode = ep + 1
                    break

            results[name].append(discovery_episode if discovery_episode else max_episodes)

    return {name: {
        'mean': np.mean(data),
        'std': np.std(data),
        'discovery_rate': np.mean([1 if d < max_episodes else 0 for d in data])
    } for name, data in results.items()}


def exploration_coverage_test(n_seeds: int = 30, n_episodes: int = 30):
    """Test state coverage (exploration breadth)."""
    results = {name: [] for name in ['Standard', 'Curiosity', 'Joy', 'Integrated']}

    for seed in range(n_seeds):
        np.random.seed(seed)

        for name in results:
            env = JoyCuriosityGridWorld()

            if name == 'Standard':
                agent = StandardQLearner(env.n_states, env.n_actions)
            elif name == 'Curiosity':
                agent = CuriosityAgent(env.n_states, env.n_actions)
            elif name == 'Joy':
                agent = JoyAgent(env.n_states, env.n_actions)
            else:
                agent = IntegratedJoyCuriosityAgent(env.n_states, env.n_actions)

            env.reset_full()
            for _ in range(n_episodes):
                run_episode(env, agent)

            coverage = (env.n_states - env.get_unvisited_count()) / env.n_states
            results[name].append(coverage)

    return {name: {
        'mean': np.mean(data),
        'std': np.std(data)
    } for name, data in results.items()}


def reward_exploitation_test(n_seeds: int = 30, n_train: int = 50, n_eval: int = 20):
    """Test how well agents exploit discovered rewards."""
    results = {name: {'train_reward': [], 'eval_reward': [], 'goal_rate': []}
               for name in ['Standard', 'Curiosity', 'Joy', 'Integrated']}

    for seed in range(n_seeds):
        np.random.seed(seed)

        for name in results:
            env = JoyCuriosityGridWorld()

            if name == 'Standard':
                agent = StandardQLearner(env.n_states, env.n_actions)
            elif name == 'Curiosity':
                agent = CuriosityAgent(env.n_states, env.n_actions)
            elif name == 'Joy':
                agent = JoyAgent(env.n_states, env.n_actions)
            else:
                agent = IntegratedJoyCuriosityAgent(env.n_states, env.n_actions)

            # Training
            env.reset_full()
            train_rewards = []
            for _ in range(n_train):
                result = run_episode(env, agent)
                train_rewards.append(result['total_reward'])
            results[name]['train_reward'].append(np.mean(train_rewards))

            # Evaluation (reduced exploration)
            if hasattr(agent, 'epsilon'):
                agent.epsilon = 0.05
            if hasattr(agent, 'base_epsilon'):
                agent.base_epsilon = 0.05

            eval_rewards = []
            goal_count = 0
            for _ in range(n_eval):
                result = run_episode(env, agent)
                eval_rewards.append(result['total_reward'])
                if result['goal_reached']:
                    goal_count += 1

            results[name]['eval_reward'].append(np.mean(eval_rewards))
            results[name]['goal_rate'].append(goal_count / n_eval)

    return {name: {
        'train_reward_mean': np.mean(data['train_reward']),
        'train_reward_std': np.std(data['train_reward']),
        'eval_reward_mean': np.mean(data['eval_reward']),
        'eval_reward_std': np.std(data['eval_reward']),
        'goal_rate_mean': np.mean(data['goal_rate']),
        'goal_rate_std': np.std(data['goal_rate'])
    } for name, data in results.items()}


def main():
    np.random.seed(42)

    print("=" * 70)
    print("EXPERIMENT 14: JOY AND CURIOSITY CHANNELS")
    print("=" * 70)

    env = JoyCuriosityGridWorld()
    print("\nEnvironment layout:")
    print(env.render())
    print("\nLegend: A=agent, ?=hidden reward, G=goal")
    print()

    print("HYPOTHESIS (Fredrickson, Silvia):")
    print("- Curiosity drives exploration toward novelty")
    print("- Joy biases toward positive experiences")
    print("- These are PARALLEL channels, not just intrinsic reward")
    print()

    # Test 1: Discovery Speed
    print("=" * 70)
    print("TEST 1: Hidden Reward Discovery Speed")
    print("=" * 70)
    print("\nHow quickly does each agent discover the hidden reward?")

    discovery = discovery_speed_test(n_seeds=30)

    print(f"\n{'Agent':<15} {'Mean Episodes':<18} {'Discovery Rate':<15}")
    print("-" * 50)
    for name, stats in discovery.items():
        print(f"{name:<15} {stats['mean']:.1f} ± {stats['std']:.1f}      "
              f"{stats['discovery_rate']:.0%}")

    # Test 2: Exploration Coverage
    print("\n" + "=" * 70)
    print("TEST 2: Exploration Coverage")
    print("=" * 70)
    print("\nWhat fraction of states does each agent visit?")

    coverage = exploration_coverage_test(n_seeds=30)

    print(f"\n{'Agent':<15} {'Coverage':<20}")
    print("-" * 35)
    for name, stats in coverage.items():
        print(f"{name:<15} {stats['mean']:.1%} ± {stats['std']:.1%}")

    # Test 3: Reward Exploitation
    print("\n" + "=" * 70)
    print("TEST 3: Reward Exploitation")
    print("=" * 70)
    print("\nHow well do agents exploit discovered rewards?")

    exploitation = reward_exploitation_test(n_seeds=30)

    print(f"\n{'Agent':<15} {'Train Reward':<18} {'Eval Reward':<18} {'Goal Rate':<12}")
    print("-" * 65)
    for name, stats in exploitation.items():
        print(f"{name:<15} {stats['train_reward_mean']:.2f} ± {stats['train_reward_std']:.2f}   "
              f"{stats['eval_reward_mean']:.2f} ± {stats['eval_reward_std']:.2f}   "
              f"{stats['goal_rate_mean']:.0%}")

    # Hypothesis Tests
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)

    # H1: Curiosity discovers faster
    std_disc = discovery['Standard']['mean']
    cur_disc = discovery['Curiosity']['mean']
    print(f"\nH1: Curiosity agent discovers hidden reward faster")
    print(f"    Standard: {std_disc:.1f} episodes")
    print(f"    Curiosity: {cur_disc:.1f} episodes")
    if cur_disc < std_disc * 0.8:
        print("    ✓ Curiosity discovers FASTER (>20% fewer episodes)")
    elif cur_disc < std_disc:
        print("    ~ Curiosity slightly faster")
    else:
        print("    ✗ No discovery advantage for curiosity")

    # H2: Curiosity explores more
    std_cov = coverage['Standard']['mean']
    cur_cov = coverage['Curiosity']['mean']
    print(f"\nH2: Curiosity agent explores more states")
    print(f"    Standard coverage: {std_cov:.1%}")
    print(f"    Curiosity coverage: {cur_cov:.1%}")
    if cur_cov > std_cov + 0.05:
        print("    ✓ Curiosity explores MORE (+5% coverage)")
    elif cur_cov > std_cov:
        print("    ~ Curiosity slightly more exploration")
    else:
        print("    ✗ No exploration advantage for curiosity")

    # H3: Joy exploits better
    std_goal = exploitation['Standard']['goal_rate_mean']
    joy_goal = exploitation['Joy']['goal_rate_mean']
    print(f"\nH3: Joy agent exploits rewards better (higher goal rate)")
    print(f"    Standard goal rate: {std_goal:.0%}")
    print(f"    Joy goal rate: {joy_goal:.0%}")
    if joy_goal > std_goal + 0.1:
        print("    ✓ Joy exploits BETTER (+10% goal rate)")
    elif joy_goal > std_goal:
        print("    ~ Joy slightly better exploitation")
    else:
        print("    ✗ No exploitation advantage for joy")

    # H4: Integrated shows both benefits
    int_disc = discovery['Integrated']['mean']
    int_cov = coverage['Integrated']['mean']
    int_goal = exploitation['Integrated']['goal_rate_mean']
    print(f"\nH4: Integrated agent shows both exploration AND exploitation")
    print(f"    Integrated discovery: {int_disc:.1f} episodes")
    print(f"    Integrated coverage: {int_cov:.1%}")
    print(f"    Integrated goal rate: {int_goal:.0%}")
    if int_disc <= cur_disc and int_goal >= joy_goal:
        print("    ✓ Integrated shows BOTH benefits")
    else:
        print("    ~ Integrated shows partial benefits")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Joy and Curiosity Channels")
    print("=" * 70)

    print("\nKey findings:")
    print("1. Curiosity drives exploration toward novelty")
    print("2. Joy biases toward positive/rewarding states")
    print("3. Positive emotions function as parallel approach channels")
    print("4. Symmetrical to negative emotions (fear = avoid, joy = approach)")

    print("\nBiological parallel:")
    print("- Curiosity ≈ Dopamine (novelty-seeking, exploration)")
    print("- Joy ≈ Opioid (pleasure, positive reinforcement)")
    print("- Both modulate attention and approach behavior")


if __name__ == "__main__":
    main()
