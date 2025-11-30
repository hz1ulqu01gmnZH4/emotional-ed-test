"""Test wanting/liking dissociation.

Hypothesis (Berridge, 2009; Robinson & Berridge, 1993):
- Wanting (incentive salience): Motivational drive, dopamine
- Liking (hedonic impact): Pleasure, opioid
- These dissociate: Can want what you don't like

Key predictions:
1. Wanting-dominant → Pursues high-salience even when low hedonic
2. Liking-dominant → Pursues highest pleasure
3. Addiction model → Wanting escalates, liking diminishes

Environment:
- High-wanting reward (W): High salience, modest pleasure
- High-liking reward (L): Low salience, high pleasure
- Regular reward (R): Baseline
"""

import numpy as np
from gridworld_wanting import WantingLikingGridWorld
from agents_wanting import (StandardQLearner, WantingDominantAgent,
                            LikingDominantAgent, IntegratedWantingLikingAgent,
                            AddictionModelAgent)


def run_episode(env: WantingLikingGridWorld, agent, max_steps: int = 100):
    """Run episode tracking wanting/liking behavior."""
    state = env.reset()
    if hasattr(agent, 'reset_episode'):
        agent.reset_episode()

    total_reward = 0
    first_choice = None
    choices = []

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, context = env.step(action)
        agent.update(state, action, reward, next_state, done, context)

        total_reward += reward

        if context.just_consumed:
            choices.append(context.just_consumed)
            if first_choice is None:
                first_choice = context.just_consumed

        if done:
            break
        state = next_state

    motivation_state = agent.get_motivation_state() if hasattr(agent, 'get_motivation_state') else {}

    return {
        'total_reward': total_reward,
        'steps': step + 1,
        'first_choice': first_choice,
        'choices': choices,
        'wanting_count': choices.count('wanting'),
        'liking_count': choices.count('liking'),
        'regular_count': choices.count('regular'),
        'motivation_state': motivation_state
    }


def train_and_evaluate(agent_class, n_train: int = 300, n_eval: int = 100, **kwargs):
    """Train agent and evaluate wanting/liking behavior."""
    env = WantingLikingGridWorld()
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

    # First choice distribution
    first_choices = {'wanting': 0, 'liking': 0, 'regular': 0, 'none': 0}
    for r in results:
        if r['first_choice']:
            first_choices[r['first_choice']] += 1
        else:
            first_choices['none'] += 1

    return {
        'mean_reward': np.mean([r['total_reward'] for r in results]),
        'wanting_first': first_choices['wanting'] / n_eval,
        'liking_first': first_choices['liking'] / n_eval,
        'regular_first': first_choices['regular'] / n_eval,
        'wanting_total': np.mean([r['wanting_count'] for r in results]),
        'liking_total': np.mean([r['liking_count'] for r in results]),
        'final_state': results[-1]['motivation_state'] if results else {}
    }


def test_addiction_progression(n_episodes: int = 100):
    """Test addiction model: wanting escalates, liking diminishes."""
    env = WantingLikingGridWorld()
    addiction_agent = AddictionModelAgent(env.n_states, env.n_actions)
    normal_agent = IntegratedWantingLikingAgent(env.n_states, env.n_actions)

    addiction_trace = {'wanting_baseline': [], 'liking_baseline': [], 'choices': []}
    normal_trace = {'choices': []}

    for ep in range(n_episodes):
        add_result = run_episode(env, addiction_agent)
        norm_result = run_episode(env, normal_agent)

        state = addiction_agent.get_motivation_state()
        addiction_trace['wanting_baseline'].append(state.get('wanting_baseline', 1.0))
        addiction_trace['liking_baseline'].append(state.get('liking_baseline', 1.0))
        addiction_trace['choices'].append(add_result['wanting_count'])
        normal_trace['choices'].append(norm_result['wanting_count'])

    return addiction_trace, normal_trace


def main():
    np.random.seed(42)

    print("=" * 70)
    print("EMOTIONAL ED TEST: Wanting/Liking Dissociation")
    print("=" * 70)

    env = WantingLikingGridWorld()
    print("\nEnvironment layout:")
    print(env.render())
    print("\nLegend: W=high-wanting(0.5), L=high-liking(1.0), R=regular(0.6), G=goal")
    print()
    print("HYPOTHESIS (Berridge, Robinson):")
    print("- Wanting = incentive salience (motivational pull)")
    print("- Liking = hedonic impact (pleasure)")
    print("- These can dissociate: want ≠ like")
    print()

    # Test 1: First choice by agent type
    print("=" * 70)
    print("TEST 1: First Reward Choice by Agent Type")
    print("=" * 70)

    agents = [
        ("Standard", StandardQLearner, {}),
        ("Wanting-dominant", WantingDominantAgent, {}),
        ("Liking-dominant", LikingDominantAgent, {}),
        ("Integrated", IntegratedWantingLikingAgent, {}),
        ("Addiction Model", AddictionModelAgent, {})
    ]

    detailed = {}
    for name, agent_class, kwargs in agents:
        detailed[name] = train_and_evaluate(agent_class, **kwargs)

    print(f"\n{'Agent':<20} {'Want 1st':<12} {'Like 1st':<12} {'Reg 1st':<12} {'Reward':<10}")
    print("-" * 66)
    for name, r in detailed.items():
        print(f"{name:<20} {r['wanting_first']:<12.1%} {r['liking_first']:<12.1%} "
              f"{r['regular_first']:<12.1%} {r['mean_reward']:<10.2f}")

    # Test 2: Total collection patterns
    print("\n" + "=" * 70)
    print("TEST 2: Total Collection Patterns")
    print("=" * 70)

    print(f"\n{'Agent':<20} {'Wanting Total':<15} {'Liking Total':<15}")
    print("-" * 50)
    for name, r in detailed.items():
        print(f"{name:<20} {r['wanting_total']:<15.2f} {r['liking_total']:<15.2f}")

    # Test 3: Addiction progression
    print("\n" + "=" * 70)
    print("TEST 3: Addiction Model Progression")
    print("=" * 70)

    addiction_trace, normal_trace = test_addiction_progression(n_episodes=100)

    print("\nWanting baseline over time (addiction model):")
    for i in [0, 24, 49, 74, 99]:
        if i < len(addiction_trace['wanting_baseline']):
            w = addiction_trace['wanting_baseline'][i]
            l = addiction_trace['liking_baseline'][i]
            print(f"  Episode {i+1:3d}: Wanting={w:.2f}, Liking={l:.2f}")

    # Check for sensitization and tolerance
    early_wanting = np.mean(addiction_trace['wanting_baseline'][:20])
    late_wanting = np.mean(addiction_trace['wanting_baseline'][-20:])
    early_liking = np.mean(addiction_trace['liking_baseline'][:20])
    late_liking = np.mean(addiction_trace['liking_baseline'][-20:])

    print(f"\nEarly vs Late baselines:")
    print(f"  Wanting: {early_wanting:.2f} → {late_wanting:.2f} (change: {late_wanting-early_wanting:+.2f})")
    print(f"  Liking: {early_liking:.2f} → {late_liking:.2f} (change: {late_liking-early_liking:+.2f})")

    # Hypothesis tests
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)

    # H1: Wanting-dominant pursues high-salience first
    want_dom_wanting = detailed['Wanting-dominant']['wanting_first']
    want_dom_liking = detailed['Wanting-dominant']['liking_first']
    print(f"\nH1: Wanting-dominant pursues high-salience (W) first")
    print(f"    Wanting-dominant: W first={want_dom_wanting:.1%}, L first={want_dom_liking:.1%}")
    if want_dom_wanting > want_dom_liking:
        print("    ✓ Wanting-dominant prefers high-salience reward")
    else:
        print("    ~ No clear salience preference")

    # H2: Liking-dominant pursues highest pleasure first
    like_dom_wanting = detailed['Liking-dominant']['wanting_first']
    like_dom_liking = detailed['Liking-dominant']['liking_first']
    print(f"\nH2: Liking-dominant pursues high-hedonic (L) first")
    print(f"    Liking-dominant: W first={like_dom_wanting:.1%}, L first={like_dom_liking:.1%}")
    if like_dom_liking > like_dom_wanting:
        print("    ✓ Liking-dominant prefers high-pleasure reward")
    else:
        print("    ~ No clear hedonic preference")

    # H3: Addiction shows sensitization (wanting up) and tolerance (liking down)
    print(f"\nH3: Addiction model shows sensitization and tolerance")
    print(f"    Wanting baseline: {early_wanting:.2f} → {late_wanting:.2f}")
    print(f"    Liking baseline: {early_liking:.2f} → {late_liking:.2f}")

    if late_wanting > early_wanting + 0.1:
        print("    ✓ Wanting SENSITIZES (increases with exposure)")
    else:
        print("    ~ No clear sensitization")

    if late_liking < early_liking - 0.05:
        print("    ✓ Liking shows TOLERANCE (decreases with exposure)")
    else:
        print("    ~ No clear tolerance")

    # H4: Wanting ≠ Liking (they dissociate)
    # Check if reward prediction (wanting) mismatches hedonic outcome (liking)
    addiction_reward = detailed['Addiction Model']['mean_reward']
    liking_reward = detailed['Liking-dominant']['mean_reward']
    print(f"\nH4: Wanting and liking can dissociate")
    print(f"    Addiction model reward: {addiction_reward:.2f}")
    print(f"    Liking-dominant reward: {liking_reward:.2f}")

    if liking_reward > addiction_reward + 0.1:
        print("    ✓ Following liking yields MORE reward than following wanting")
        print("    (Addiction: high wanting, suboptimal outcomes)")
    else:
        print("    ~ Wanting and liking produce similar outcomes")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Wanting/Liking Dissociation")
    print("=" * 70)

    print("\nKey findings:")
    print("1. Wanting-dominant: Pursues high-salience rewards")
    print("2. Liking-dominant: Pursues high-pleasure rewards")
    print("3. Addiction model: Wanting escalates, liking diminishes")
    print("4. Dissociation: Can want what doesn't maximize pleasure")

    print("\nBiological parallel:")
    print("- Wanting ≈ Dopamine (mesolimbic) - incentive salience")
    print("- Liking ≈ Opioid (nucleus accumbens) - hedonic impact")
    print("- Addiction: Wanting sensitizes, liking tolerates")


if __name__ == "__main__":
    main()
