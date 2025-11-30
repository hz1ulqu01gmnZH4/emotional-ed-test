"""Test approach-avoidance conflict: fear vs desire.

Hypothesis (Miller, 1944; Gray, 1982):
- Fear and approach are competing motivational systems
- Fear-dominant: Risk-averse, prefers safe reward
- Approach-dominant: Risk-seeking, pursues high-value despite danger
- Balanced: Shows conflict behavior (hesitation, exploration)

Environment:
- Safe reward (0.3) far from threat
- Risky reward (1.0) near threat
- Agent must choose which to pursue first

Key metrics:
- Which reward collected first (safe vs risky)
- Time spent near threat
- Hesitation under conflict
"""

import numpy as np
from gridworld_conflict import ApproachAvoidanceGridWorld
from agents_conflict import (StandardQLearner, FearDominantAgent,
                             ApproachDominantAgent, BalancedConflictAgent)

def run_episode(env: ApproachAvoidanceGridWorld, agent, max_steps: int = 100):
    """Run episode tracking conflict behavior."""
    state = env.reset()
    if hasattr(agent, 'reset_episode'):
        agent.reset_episode()

    first_reward = None
    time_near_threat = 0
    total_reward = 0

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, context = env.step(action)
        agent.update(state, action, reward, next_state, done, context)

        total_reward += reward

        # Track which reward collected first
        if first_reward is None:
            if env.safe_collected and not env.risky_collected:
                first_reward = 'safe'
            elif env.risky_collected and not env.safe_collected:
                first_reward = 'risky'

        # Track time near threat
        if context.threat_distance < 2.0:
            time_near_threat += 1

        if done:
            break
        state = next_state

    return {
        'first_reward': first_reward or 'none',
        'time_near_threat': time_near_threat,
        'total_reward': total_reward,
        'steps': step + 1,
        'safe_collected': env.safe_collected,
        'risky_collected': env.risky_collected
    }


def train_and_evaluate(agent_class, n_train: int = 300, n_eval: int = 100, **kwargs):
    """Train agent and evaluate conflict behavior."""
    env = ApproachAvoidanceGridWorld()
    agent = agent_class(n_states=env.n_states, n_actions=env.n_actions, **kwargs)

    # Training
    for _ in range(n_train):
        run_episode(env, agent)

    # Evaluation
    agent.epsilon = 0.05  # Small exploration for evaluation
    results = []
    for _ in range(n_eval):
        result = run_episode(env, agent)
        results.append(result)

    # Aggregate
    safe_first = sum(1 for r in results if r['first_reward'] == 'safe')
    risky_first = sum(1 for r in results if r['first_reward'] == 'risky')
    both_collected = sum(1 for r in results if r['safe_collected'] and r['risky_collected'])

    return {
        'safe_first_rate': safe_first / n_eval,
        'risky_first_rate': risky_first / n_eval,
        'both_collected_rate': both_collected / n_eval,
        'mean_time_near_threat': np.mean([r['time_near_threat'] for r in results]),
        'mean_total_reward': np.mean([r['total_reward'] for r in results]),
        'mean_steps': np.mean([r['steps'] for r in results])
    }


def analyze_conflict_behavior(n_runs: int = 50):
    """Analyze hesitation/conflict patterns."""
    env = ApproachAvoidanceGridWorld()

    agents = {
        'Standard': StandardQLearner(env.n_states, env.n_actions),
        'Fear-dominant': FearDominantAgent(env.n_states, env.n_actions),
        'Approach-dominant': ApproachDominantAgent(env.n_states, env.n_actions),
        'Balanced (conflict)': BalancedConflictAgent(env.n_states, env.n_actions)
    }

    results = {}
    for name, agent in agents.items():
        # Train
        for _ in range(300):
            run_episode(env, agent)

        # Evaluate
        agent.epsilon = 0.05
        eval_results = [run_episode(env, agent) for _ in range(n_runs)]

        results[name] = {
            'safe_first': sum(1 for r in eval_results if r['first_reward'] == 'safe') / n_runs,
            'risky_first': sum(1 for r in eval_results if r['first_reward'] == 'risky') / n_runs,
            'threat_time': np.mean([r['time_near_threat'] for r in eval_results]),
            'total_reward': np.mean([r['total_reward'] for r in eval_results])
        }

    return results


def main():
    np.random.seed(42)

    print("=" * 65)
    print("EMOTIONAL ED TEST: Approach-Avoidance Conflict (Fear vs Desire)")
    print("=" * 65)

    env = ApproachAvoidanceGridWorld()
    print("\nEnvironment layout:")
    print(env.render())
    print("\nLegend: S=safe reward (0.3), R=risky reward (1.0), X=threat, A=agent")
    print()
    print("HYPOTHESIS (Miller, Gray):")
    print("- Fear-dominant: Prefers safe reward, avoids threat")
    print("- Approach-dominant: Pursues risky reward despite threat")
    print("- Balanced: Shows conflict (hesitation, more exploration)")
    print()

    # Test 1: First reward choice
    print("=" * 65)
    print("TEST 1: Which Reward Collected First")
    print("=" * 65)

    results = analyze_conflict_behavior(n_runs=100)

    print(f"\n{'Agent':<22} {'Safe First':<12} {'Risky First':<12} {'Threat Time':<12} {'Reward':<10}")
    print("-" * 68)
    for name, r in results.items():
        print(f"{name:<22} {r['safe_first']:<12.1%} {r['risky_first']:<12.1%} "
              f"{r['threat_time']:<12.1f} {r['total_reward']:<10.2f}")

    # Test 2: Detailed comparison
    print("\n" + "=" * 65)
    print("TEST 2: Detailed Agent Comparison")
    print("=" * 65)

    comparisons = [
        ("Standard", StandardQLearner, {}),
        ("Fear-dominant", FearDominantAgent, {'fear_weight': 1.0, 'approach_weight': 0.3}),
        ("Approach-dominant", ApproachDominantAgent, {'fear_weight': 0.3, 'approach_weight': 1.0}),
        ("Balanced", BalancedConflictAgent, {'fear_weight': 0.7, 'approach_weight': 0.7})
    ]

    detailed = {}
    for name, agent_class, kwargs in comparisons:
        detailed[name] = train_and_evaluate(agent_class, **kwargs)

    print(f"\n{'Metric':<25} {'Standard':<12} {'Fear-dom':<12} {'Approach-dom':<12} {'Balanced':<12}")
    print("-" * 73)

    metrics = ['safe_first_rate', 'risky_first_rate', 'both_collected_rate',
               'mean_time_near_threat', 'mean_total_reward']
    metric_names = ['Safe first %', 'Risky first %', 'Both collected %',
                    'Time near threat', 'Total reward']

    for metric, mname in zip(metrics, metric_names):
        row = f"{mname:<25}"
        for name in ['Standard', 'Fear-dominant', 'Approach-dominant', 'Balanced']:
            val = detailed[name][metric]
            if 'rate' in metric or '%' in mname:
                row += f"{val:<12.1%}"
            else:
                row += f"{val:<12.2f}"
        print(row)

    # Hypothesis test
    print("\n" + "=" * 65)
    print("HYPOTHESIS TEST")
    print("=" * 65)

    fear_safe = detailed['Fear-dominant']['safe_first_rate']
    approach_risky = detailed['Approach-dominant']['risky_first_rate']
    standard_safe = detailed['Standard']['safe_first_rate']

    print(f"\nFear-dominant safe-first rate: {fear_safe:.1%}")
    print(f"Approach-dominant risky-first rate: {approach_risky:.1%}")
    print(f"Standard safe-first rate: {standard_safe:.1%}")

    if fear_safe > standard_safe + 0.05:
        print("\n✓ Fear-dominant agent is MORE risk-averse than standard")
    else:
        print("\n~ Fear-dominant agent similar to standard")

    if approach_risky > 0.3:
        print("✓ Approach-dominant agent pursues risky reward")
    else:
        print("~ Approach-dominant agent avoids risk")

    balanced_threat = detailed['Balanced']['mean_time_near_threat']
    fear_threat = detailed['Fear-dominant']['mean_time_near_threat']

    if balanced_threat > fear_threat:
        print("✓ Balanced agent spends MORE time near threat (conflict/hesitation)")
    else:
        print("~ Balanced agent doesn't show clear conflict pattern")

    # Visualize single episodes
    print("\n" + "=" * 65)
    print("SINGLE EPISODE PATHS")
    print("=" * 65)

    for name, agent_class, kwargs in [("Fear-dominant", FearDominantAgent, {}),
                                       ("Approach-dominant", ApproachDominantAgent, {})]:
        env = ApproachAvoidanceGridWorld()
        agent = agent_class(n_states=env.n_states, n_actions=env.n_actions, lr=0.1, **kwargs)

        # Quick train
        for _ in range(200):
            run_episode(env, agent)

        # Show path
        env.reset()
        agent.epsilon = 0
        path = [tuple(env.agent_pos)]

        state = env._pos_to_state(env.agent_pos)
        for _ in range(30):
            action = agent.select_action(state)
            state, _, done, _ = env.step(action)
            path.append(tuple(env.agent_pos))
            if done:
                break

        print(f"\n{name} path: {' → '.join([str(p) for p in path[:10]])}...")
        print(f"First reward: {'Safe' if env.safe_collected and not env.risky_collected else 'Risky' if env.risky_collected else 'None'}")


if __name__ == "__main__":
    main()
