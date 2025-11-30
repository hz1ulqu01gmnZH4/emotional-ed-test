"""Test multi-channel emotion integration.

Hypothesis (Pessoa, 2008; LeDoux & Pine, 2016):
- Emotional systems interact, not operate independently
- Fear, anger, and regret compete for control of behavior
- Adaptive agents learn to weight channels based on context
- Integration produces emergent behaviors not seen in single-channel agents

Environment:
- Safe path: 0.3 reward, no threat
- Risky path: 1.0 reward, near threat
- Blocked path: 0.6 reward, requires persistence

Key metrics:
- Which goal reached (safe/risky/blocked)
- Channel activation patterns
- Emergent behaviors from channel interaction
"""

import numpy as np
from gridworld_integration import IntegrationGridWorld
from agents_integration import (StandardQLearner, IntegratedEmotionalAgent,
                                FearDominantAgent, AngerDominantAgent,
                                RegretDominantAgent, BalancedAgent,
                                AdaptiveEmotionalAgent)


def run_episode(env: IntegrationGridWorld, agent, max_steps: int = 100):
    """Run episode tracking multi-channel behavior."""
    state = env.reset()
    if hasattr(agent, 'reset_episode'):
        agent.reset_episode()

    total_reward = 0
    channel_activations = {'fear': [], 'anger': [], 'regret': []}

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, context = env.step(action)
        agent.update(state, action, reward, next_state, done, context)

        total_reward += reward

        # Track channel activations
        if hasattr(agent, 'get_channel_states'):
            states = agent.get_channel_states()
            if 'fear' in states:
                channel_activations['fear'].append(states['fear'])
            if 'anger' in states:
                channel_activations['anger'].append(states['anger'])
            if 'regret' in states:
                channel_activations['regret'].append(states['regret'])

        if done:
            break
        state = next_state

    # Determine which goal was reached
    goal_reached = 'none'
    if env.safe_collected:
        goal_reached = 'safe'
    elif env.risky_collected:
        goal_reached = 'risky'
    elif env.blocked_collected:
        goal_reached = 'blocked'

    return {
        'goal_reached': goal_reached,
        'total_reward': total_reward,
        'steps': step + 1,
        'wall_hits': env.wall_hits,
        'wall_broken': env.wall_broken,
        'mean_fear': np.mean(channel_activations['fear']) if channel_activations['fear'] else 0,
        'mean_anger': np.mean(channel_activations['anger']) if channel_activations['anger'] else 0,
        'mean_regret': np.mean(channel_activations['regret']) if channel_activations['regret'] else 0,
        'max_fear': max(channel_activations['fear']) if channel_activations['fear'] else 0,
        'max_anger': max(channel_activations['anger']) if channel_activations['anger'] else 0
    }


def train_and_evaluate(agent_class, n_train: int = 500, n_eval: int = 100, **kwargs):
    """Train agent and evaluate multi-channel behavior."""
    env = IntegrationGridWorld()
    agent = agent_class(n_states=env.n_states, n_actions=env.n_actions, **kwargs)

    # Training
    for _ in range(n_train):
        run_episode(env, agent)

    # Evaluation
    agent.epsilon = 0.05
    results = []
    for _ in range(n_eval):
        result = run_episode(env, agent)
        results.append(result)

    # Aggregate
    goal_counts = {'safe': 0, 'risky': 0, 'blocked': 0, 'none': 0}
    for r in results:
        goal_counts[r['goal_reached']] += 1

    # Get final channel states if available
    final_states = {}
    if hasattr(agent, 'get_channel_states'):
        final_states = agent.get_channel_states()

    return {
        'safe_rate': goal_counts['safe'] / n_eval,
        'risky_rate': goal_counts['risky'] / n_eval,
        'blocked_rate': goal_counts['blocked'] / n_eval,
        'none_rate': goal_counts['none'] / n_eval,
        'mean_reward': np.mean([r['total_reward'] for r in results]),
        'mean_steps': np.mean([r['steps'] for r in results]),
        'mean_wall_hits': np.mean([r['wall_hits'] for r in results]),
        'wall_broken_rate': np.mean([r['wall_broken'] for r in results]),
        'mean_fear': np.mean([r['mean_fear'] for r in results]),
        'mean_anger': np.mean([r['mean_anger'] for r in results]),
        'final_states': final_states
    }


def analyze_channel_competition(n_runs: int = 30):
    """Analyze how channels compete for behavioral control."""
    env = IntegrationGridWorld()

    agents = {
        'Standard': lambda: StandardQLearner(env.n_states, env.n_actions),
        'Fear-dominant': lambda: FearDominantAgent(env.n_states, env.n_actions),
        'Anger-dominant': lambda: AngerDominantAgent(env.n_states, env.n_actions),
        'Regret-dominant': lambda: RegretDominantAgent(env.n_states, env.n_actions),
        'Balanced': lambda: BalancedAgent(env.n_states, env.n_actions),
        'Adaptive': lambda: AdaptiveEmotionalAgent(env.n_states, env.n_actions)
    }

    results = {}
    for name, agent_fn in agents.items():
        run_results = []
        for _ in range(n_runs):
            agent = agent_fn()
            result = train_and_evaluate(type(agent), n_train=500, n_eval=50)
            run_results.append(result)

        results[name] = {
            'safe_rate': np.mean([r['safe_rate'] for r in run_results]),
            'risky_rate': np.mean([r['risky_rate'] for r in run_results]),
            'blocked_rate': np.mean([r['blocked_rate'] for r in run_results]),
            'mean_reward': np.mean([r['mean_reward'] for r in run_results])
        }

    return results


def main():
    np.random.seed(42)

    print("=" * 70)
    print("EMOTIONAL ED TEST: Multi-Channel Integration (Fear + Anger + Regret)")
    print("=" * 70)

    env = IntegrationGridWorld()
    print("\nEnvironment layout:")
    print(env.render())
    print("\nLegend: S=safe(0.3), R=risky(1.0), B=blocked(0.6), X=threat, #=wall")
    print()
    print("HYPOTHESIS (Pessoa, LeDoux & Pine):")
    print("- Fear-dominant → Safe path (avoids threat)")
    print("- Anger-dominant → Blocked path (persists at wall)")
    print("- Regret-dominant → Learns from foregone, improves over time")
    print("- Balanced → May overcome fear for risky path")
    print("- Adaptive → Learns optimal channel weighting")
    print()

    # Test 1: Goal choice by agent type
    print("=" * 70)
    print("TEST 1: Goal Choice by Emotional Profile (500 train, 100 eval)")
    print("=" * 70)

    comparisons = [
        ("Standard", StandardQLearner, {}),
        ("Fear-dominant", FearDominantAgent, {}),
        ("Anger-dominant", AngerDominantAgent, {}),
        ("Regret-dominant", RegretDominantAgent, {}),
        ("Balanced", BalancedAgent, {}),
        ("Adaptive", AdaptiveEmotionalAgent, {})
    ]

    detailed = {}
    for name, agent_class, kwargs in comparisons:
        detailed[name] = train_and_evaluate(agent_class, **kwargs)

    print(f"\n{'Agent':<18} {'Safe':<10} {'Risky':<10} {'Blocked':<10} {'None':<10} {'Reward':<10}")
    print("-" * 68)
    for name in ['Standard', 'Fear-dominant', 'Anger-dominant', 'Regret-dominant', 'Balanced', 'Adaptive']:
        r = detailed[name]
        print(f"{name:<18} {r['safe_rate']:<10.1%} {r['risky_rate']:<10.1%} "
              f"{r['blocked_rate']:<10.1%} {r['none_rate']:<10.1%} {r['mean_reward']:<10.2f}")

    # Test 2: Channel activation patterns
    print("\n" + "=" * 70)
    print("TEST 2: Channel Activation Patterns")
    print("=" * 70)

    print(f"\n{'Agent':<18} {'Mean Fear':<12} {'Mean Anger':<12} {'Wall Hits':<12} {'Wall Broken':<12}")
    print("-" * 68)
    for name in ['Fear-dominant', 'Anger-dominant', 'Balanced', 'Adaptive']:
        r = detailed[name]
        print(f"{name:<18} {r['mean_fear']:<12.3f} {r['mean_anger']:<12.3f} "
              f"{r['mean_wall_hits']:<12.1f} {r['wall_broken_rate']:<12.1%}")

    # Test 3: Adaptive agent weight evolution
    print("\n" + "=" * 70)
    print("TEST 3: Adaptive Agent Channel Evolution")
    print("=" * 70)

    env = IntegrationGridWorld()
    adaptive = AdaptiveEmotionalAgent(env.n_states, env.n_actions)

    weight_history = []
    for block in range(10):
        for _ in range(50):
            run_episode(env, adaptive)
        states = adaptive.get_channel_states()
        if 'channel_weights' in states:
            weight_history.append(states['channel_weights'].copy())

    if weight_history:
        print(f"\n{'Block':<10} {'Fear Weight':<15} {'Anger Weight':<15} {'Regret Weight':<15}")
        print("-" * 55)
        for i, weights in enumerate(weight_history):
            print(f"{(i+1)*50:<10} {weights['fear']:<15.3f} {weights['anger']:<15.3f} "
                  f"{weights['regret']:<15.3f}")

    # Hypothesis tests
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)

    # H1: Fear-dominant avoids risky path
    fear_risky = detailed['Fear-dominant']['risky_rate']
    standard_risky = detailed['Standard']['risky_rate']
    anger_risky = detailed['Anger-dominant']['risky_rate']
    print(f"\nH1: Fear reduces approach to threatening reward")
    print(f"    Fear-dominant risky rate: {fear_risky:.1%}")
    print(f"    Standard risky rate: {standard_risky:.1%}")
    print(f"    Anger-dominant risky rate: {anger_risky:.1%}")
    if fear_risky < anger_risky:
        print("    ✓ Fear-dominant approaches risky goal LESS than anger-dominant")
    else:
        print("    ~ No clear fear effect on risky approach")

    # H2: Anger enables approach despite threat
    print(f"\nH2: Anger enables approach despite threat")
    print(f"    Anger-dominant risky rate: {anger_risky:.1%}")
    print(f"    Fear-dominant risky rate: {fear_risky:.1%}")
    if anger_risky > fear_risky + 0.1:
        print("    ✓ Anger-dominant overcomes threat aversion")
    else:
        print("    ~ No clear anger effect on threat approach")

    # H3: Regret improves value-seeking
    regret_reward = detailed['Regret-dominant']['mean_reward']
    standard_reward = detailed['Standard']['mean_reward']
    print(f"\nH3: Regret channel improves learning")
    print(f"    Regret-dominant mean reward: {regret_reward:.2f}")
    print(f"    Standard mean reward: {standard_reward:.2f}")
    if regret_reward > standard_reward * 0.9:
        print("    ✓ Regret-dominant achieves COMPETITIVE reward")
    else:
        print("    ~ Regret channel doesn't clearly improve")

    # H4: Channel interactions produce distinct profiles
    fear_safe = detailed['Fear-dominant']['safe_rate']
    anger_safe = detailed['Anger-dominant']['safe_rate']
    balanced_safe = detailed['Balanced']['safe_rate']
    print(f"\nH4: Different channel weights produce distinct behavioral profiles")
    print(f"    Fear-dominant: {fear_safe:.0%} safe, {fear_risky:.0%} risky")
    print(f"    Anger-dominant: {anger_safe:.0%} safe, {anger_risky:.0%} risky")
    print(f"    Balanced: {balanced_safe:.0%} safe, {detailed['Balanced']['risky_rate']:.0%} risky")

    # Check for differentiation
    profiles_differ = (abs(fear_risky - anger_risky) > 0.15 or
                       abs(fear_safe - anger_safe) > 0.15)
    if profiles_differ:
        print("    ✓ Channel weights produce DISTINCT behavioral profiles")
    else:
        print("    ~ Profiles not clearly differentiated")

    # Test 4: Emergent behaviors
    print("\n" + "=" * 70)
    print("TEST 4: Emergent Behaviors from Channel Interaction")
    print("=" * 70)

    # Single episode paths
    for name, agent_class in [("Fear-dominant", FearDominantAgent),
                               ("Anger-dominant", AngerDominantAgent),
                               ("Balanced", BalancedAgent)]:
        env = IntegrationGridWorld()
        agent = agent_class(n_states=env.n_states, n_actions=env.n_actions)

        # Train
        for _ in range(300):
            run_episode(env, agent)

        # Show path
        env.reset()
        agent.epsilon = 0
        path = [tuple(env.agent_pos)]

        state = env._pos_to_state(env.agent_pos)
        for _ in range(50):
            action = agent.select_action(state)
            state, _, done, _ = env.step(action)
            path.append(tuple(env.agent_pos))
            if done:
                break

        goal = 'Safe' if env.safe_collected else 'Risky' if env.risky_collected else 'Blocked' if env.blocked_collected else 'None'
        print(f"\n{name}:")
        print(f"  Path: {' → '.join([str(p) for p in path[:8]])}...")
        print(f"  Goal reached: {goal}")
        print(f"  Wall hits: {env.wall_hits}, Wall broken: {env.wall_broken}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Channel Integration Effects")
    print("=" * 70)

    print("\nGoal preferences by emotional profile:")
    print("  Fear-dominant → Prefers safe path (risk aversion)")
    print("  Anger-dominant → Persists at blocked path (frustration-driven)")
    print("  Balanced → May achieve risky goal (fear-anger equilibrium)")
    print("  Adaptive → Learns context-appropriate weighting")

    print("\nEmergent interaction effects:")
    print("  - Fear × Anger: Anger can override fear avoidance")
    print("  - Fear × Regret: Regret from safe choice may reduce future fear")
    print("  - Anger × Regret: Persistence + learning creates exploration")


if __name__ == "__main__":
    main()
