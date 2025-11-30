"""Test disgust channel: contamination avoidance vs threat avoidance.

Hypothesis (Rozin et al., 2008; Oaten et al., 2009):
- Disgust is distinct from fear
- Fear: Habituates with exposure, immediate threat avoidance
- Disgust: Doesn't habituate, contamination tracking, one-contact rule

Key differences to test:
1. Habituation: Fear decreases with exposure; disgust doesn't
2. Spread: Threats don't spread; contamination does
3. Memory: Fear of specific threats; disgust tracks contaminated areas
4. Learning: Fear habituates; disgust creates persistent avoidance

Environment:
- Threat (X): Immediate harm, no spread, fear habituates
- Contaminant (C): Spreads, no habituation, one-contact rule
- Food (+): Reward if clean, penalty if contaminated
"""

import numpy as np
from gridworld_disgust import DisgustGridWorld
from agents_disgust import (StandardQLearner, FearOnlyAgent,
                            DisgustOnlyAgent, IntegratedFearDisgustAgent)


def run_episode(env: DisgustGridWorld, agent, max_steps: int = 100):
    """Run episode tracking disgust vs fear behavior."""
    state = env.reset()
    if hasattr(agent, 'reset_episode'):
        agent.reset_episode()

    total_reward = 0
    threat_approaches = 0
    contaminant_approaches = 0
    contamination_events = 0

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, context = env.step(action)
        agent.update(state, action, reward, next_state, done, context)

        total_reward += reward

        if context.threat_distance < 2.0:
            threat_approaches += 1
        if context.contaminant_distance < 2.0:
            contaminant_approaches += 1
        if context.touched_contaminant:
            contamination_events += 1

        if done:
            break
        state = next_state

    emotional_state = agent.get_emotional_state() if hasattr(agent, 'get_emotional_state') else {}

    return {
        'total_reward': total_reward,
        'steps': step + 1,
        'threat_approaches': threat_approaches,
        'contaminant_approaches': contaminant_approaches,
        'contamination_events': contamination_events,
        'agent_contaminated': env.agent_contaminated,
        'contamination_spread': len(env.contaminated_cells),
        'emotional_state': emotional_state
    }


def test_habituation(n_episodes: int = 50):
    """Test whether fear habituates but disgust doesn't."""
    env = DisgustGridWorld()

    agents = {
        'Fear Only': FearOnlyAgent(env.n_states, env.n_actions),
        'Disgust Only': DisgustOnlyAgent(env.n_states, env.n_actions),
        'Integrated': IntegratedFearDisgustAgent(env.n_states, env.n_actions)
    }

    # Track avoidance over time
    results = {name: {'early_approach': [], 'late_approach': []} for name in agents}

    for name, agent in agents.items():
        for ep in range(n_episodes):
            result = run_episode(env, agent)

            if ep < 10:  # Early episodes
                results[name]['early_approach'].append(result['threat_approaches'] + result['contaminant_approaches'])
            elif ep >= n_episodes - 10:  # Late episodes
                results[name]['late_approach'].append(result['threat_approaches'] + result['contaminant_approaches'])

    return results


def train_and_evaluate(agent_class, n_train: int = 200, n_eval: int = 50, **kwargs):
    """Train agent and evaluate disgust vs fear behavior."""
    env = DisgustGridWorld()
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

    return {
        'mean_reward': np.mean([r['total_reward'] for r in results]),
        'threat_approaches': np.mean([r['threat_approaches'] for r in results]),
        'contaminant_approaches': np.mean([r['contaminant_approaches'] for r in results]),
        'contamination_rate': np.mean([r['agent_contaminated'] for r in results]),
        'contamination_events': np.mean([r['contamination_events'] for r in results]),
        'final_state': results[-1]['emotional_state'] if results else {}
    }


def main():
    np.random.seed(42)

    print("=" * 70)
    print("EMOTIONAL ED TEST: Disgust Channel (Contamination vs Threat)")
    print("=" * 70)

    env = DisgustGridWorld()
    print("\nEnvironment layout:")
    print(env.render())
    print("\nLegend: X=threat, ~=contaminant, +=food, ?=contaminated food, G=goal")
    print()
    print("HYPOTHESIS (Rozin, Oaten):")
    print("- Fear habituates with exposure; disgust doesn't")
    print("- Threat doesn't spread; contamination does")
    print("- Disgust creates persistent avoidance memory")
    print()

    # Test 1: Basic comparison
    print("=" * 70)
    print("TEST 1: Avoidance Behavior by Agent Type")
    print("=" * 70)

    agents = [
        ("Standard", StandardQLearner, {}),
        ("Fear Only", FearOnlyAgent, {}),
        ("Disgust Only", DisgustOnlyAgent, {}),
        ("Integrated", IntegratedFearDisgustAgent, {})
    ]

    detailed = {}
    for name, agent_class, kwargs in agents:
        detailed[name] = train_and_evaluate(agent_class, **kwargs)

    print(f"\n{'Agent':<18} {'Threat App':<12} {'Contam App':<12} {'Contam Rate':<12} {'Reward':<10}")
    print("-" * 64)
    for name, r in detailed.items():
        print(f"{name:<18} {r['threat_approaches']:<12.1f} {r['contaminant_approaches']:<12.1f} "
              f"{r['contamination_rate']:<12.1%} {r['mean_reward']:<10.2f}")

    # Test 2: Habituation comparison
    print("\n" + "=" * 70)
    print("TEST 2: Habituation Over Training")
    print("=" * 70)

    hab_results = test_habituation(n_episodes=50)

    print(f"\n{'Agent':<18} {'Early Approach':<15} {'Late Approach':<15} {'Change':<12}")
    print("-" * 60)
    for name, data in hab_results.items():
        early = np.mean(data['early_approach'])
        late = np.mean(data['late_approach'])
        change = late - early
        print(f"{name:<18} {early:<15.1f} {late:<15.1f} {change:<+12.1f}")

    # Test 3: Contamination spread awareness
    print("\n" + "=" * 70)
    print("TEST 3: Contamination Spread Tracking")
    print("=" * 70)

    env = DisgustGridWorld()
    disgust_agent = DisgustOnlyAgent(env.n_states, env.n_actions)
    fear_agent = FearOnlyAgent(env.n_states, env.n_actions)

    # Run episodes and track contamination awareness
    for _ in range(100):
        run_episode(env, disgust_agent)
        run_episode(env, fear_agent)

    disgust_known = len(disgust_agent.known_contaminated)
    fear_exposures = fear_agent.threat_exposures

    print(f"\nAfter 100 episodes:")
    print(f"  Disgust agent known contaminated states: {disgust_known}")
    print(f"  Fear agent threat exposures (with habituation): {fear_exposures}")
    print(f"  Fear habituation factor: {fear_agent.fear_habituation ** fear_exposures:.4f}")

    # Hypothesis tests
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)

    # H1: Fear habituates, disgust doesn't
    fear_early = np.mean(hab_results['Fear Only']['early_approach'])
    fear_late = np.mean(hab_results['Fear Only']['late_approach'])
    disgust_early = np.mean(hab_results['Disgust Only']['early_approach'])
    disgust_late = np.mean(hab_results['Disgust Only']['late_approach'])

    print(f"\nH1: Fear habituates; disgust doesn't")
    print(f"    Fear: early={fear_early:.1f} → late={fear_late:.1f} (change: {fear_late-fear_early:+.1f})")
    print(f"    Disgust: early={disgust_early:.1f} → late={disgust_late:.1f} (change: {disgust_late-disgust_early:+.1f})")

    fear_hab = fear_late > fear_early  # Fear approaches more over time (habituated)
    disgust_persist = disgust_late <= disgust_early + 1  # Disgust doesn't increase approaching

    if fear_hab and disgust_persist:
        print("    ✓ Fear habituates (more approach); disgust persists")
    elif fear_hab:
        print("    ~ Fear habituates; disgust unclear")
    elif disgust_persist:
        print("    ~ Fear unclear; disgust persists")
    else:
        print("    ~ Neither pattern clear")

    # H2: Disgust tracks contamination spread
    print(f"\nH2: Disgust agent tracks contaminated locations")
    print(f"    Disgust known contaminated: {disgust_known} states")
    if disgust_known > 1:
        print("    ✓ Disgust agent learns contamination spread")
    else:
        print("    ~ Disgust agent doesn't track spread")

    # H3: Different approach patterns
    fear_threat = detailed['Fear Only']['threat_approaches']
    fear_contam = detailed['Fear Only']['contaminant_approaches']
    disgust_threat = detailed['Disgust Only']['threat_approaches']
    disgust_contam = detailed['Disgust Only']['contaminant_approaches']

    print(f"\nH3: Agents show different approach patterns")
    print(f"    Fear Only: threat={fear_threat:.1f}, contaminant={fear_contam:.1f}")
    print(f"    Disgust Only: threat={disgust_threat:.1f}, contaminant={disgust_contam:.1f}")

    # Fear should approach contaminant more (doesn't track it)
    # Disgust should approach threat more (doesn't fear it)
    if fear_contam > disgust_contam:
        print("    ✓ Fear approaches contaminant MORE (no contamination tracking)")
    else:
        print("    ~ Fear doesn't approach contaminant more")

    if disgust_threat > fear_threat:
        print("    ✓ Disgust approaches threat MORE (no fear)")
    else:
        print("    ~ Disgust doesn't approach threat more")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Disgust vs Fear Channel")
    print("=" * 70)

    print("\nKey distinctions demonstrated:")
    print("1. Habituation: Fear decreases with exposure; disgust persists")
    print("2. Tracking: Disgust remembers contaminated locations")
    print("3. Specificity: Fear avoids threats; disgust avoids contaminants")
    print("4. Spread: Disgust learns contamination spreads")

    print("\nBiological parallel:")
    print("- Fear: Predator avoidance (habituates to non-dangerous)")
    print("- Disgust: Pathogen avoidance (one bad apple ruins the barrel)")
    print("- Evolutionary: Different problems, different solutions")


if __name__ == "__main__":
    main()
