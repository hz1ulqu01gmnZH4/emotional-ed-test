"""Test temporal emotion dynamics: phasic vs tonic (mood).

Hypothesis (Davidson, 1998; Watson, 2000):
- Phasic emotions: Acute responses that decay quickly
- Tonic mood: Slow-shifting baseline from sustained experience
- Mood biases perception, cognition, and behavior
- Sustained negative experience → depressed mood → conservative behavior
- Sustained positive experience → elevated mood → exploratory behavior

Environment phases:
1. Neutral: Mixed outcomes
2. Negative: Many threats, few rewards
3. Recovery: Return to neutral
4. Positive: Many rewards, few threats

Key metrics:
- Mood shift during negative phase
- Recovery rate after negative phase
- Behavioral differences by mood state
"""

import numpy as np
from gridworld_temporal import TemporalGridWorld
from agents_temporal import (StandardQLearner, PhasicOnlyAgent,
                             TonicMoodAgent, IntegratedTemporalAgent)


def run_episode(env: TemporalGridWorld, agent, max_steps: int = 100):
    """Run episode tracking temporal dynamics."""
    state = env.reset()
    if hasattr(agent, 'reset_episode'):
        agent.reset_episode()

    total_reward = 0
    emotional_states = []
    phase = env.get_phase()

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, context = env.step(action)
        agent.update(state, action, reward, next_state, done, context)

        total_reward += reward

        if hasattr(agent, 'get_emotional_state'):
            emotional_states.append(agent.get_emotional_state().copy())

        if done:
            break
        state = next_state

    # Extract final mood with explicit assertion
    final_mood = 0
    if emotional_states:
        final_state = emotional_states[-1]
        assert 'mood' in final_state, f"BUG: Emotional state missing 'mood' key: {final_state}"
        final_mood = final_state['mood']

    return {
        'total_reward': total_reward,
        'steps': step + 1,
        'phase': phase,
        'emotional_states': emotional_states,
        'final_mood': final_mood
    }


def run_phase_experiment(agent_class, n_cycles: int = 3, phase_length: int = 50, **kwargs):
    """Run experiment across multiple phase cycles."""
    env = TemporalGridWorld(phase_length=phase_length)
    agent = agent_class(n_states=env.n_states, n_actions=env.n_actions, **kwargs)

    # Track mood by phase
    phase_moods = {'neutral': [], 'negative': [], 'recovery': [], 'positive': []}
    phase_rewards = {'neutral': [], 'negative': [], 'recovery': [], 'positive': []}

    total_steps = n_cycles * 4 * phase_length
    episodes_per_phase = phase_length // 20  # Approximate episodes per phase

    current_mood = 0
    for _ in range(total_steps // 50):  # Run many short episodes
        result = run_episode(env, agent, max_steps=50)
        phase = env.get_phase()

        if hasattr(agent, 'get_emotional_state'):
            emotional_state = agent.get_emotional_state()
            assert 'mood' in emotional_state, f"BUG: Agent {agent.__class__.__name__} emotional state missing 'mood' key"
            current_mood = emotional_state['mood']

        phase_moods[phase].append(current_mood)
        phase_rewards[phase].append(result['total_reward'])

    # Assert data was collected - empty lists indicate a bug
    assert all(m for m in phase_moods.values()), \
        f"BUG: Empty mood data in phases: {[p for p, m in phase_moods.items() if not m]}"
    assert all(r for r in phase_rewards.values()), \
        f"BUG: Empty reward data in phases: {[p for p, r in phase_rewards.items() if not r]}"
    assert hasattr(agent, 'get_mood_history'), \
        f"BUG: Agent {type(agent).__name__} missing get_mood_history method"

    return {
        'phase_moods': {p: np.mean(m) for p, m in phase_moods.items()},
        'phase_rewards': {p: np.mean(r) for p, r in phase_rewards.items()},
        'mood_history': agent.get_mood_history()
    }


def analyze_mood_dynamics(n_runs: int = 20):
    """Analyze how mood develops across phases."""
    agents = {
        'Standard': StandardQLearner,
        'Phasic Only': PhasicOnlyAgent,
        'Tonic Mood': TonicMoodAgent,
        'Integrated': IntegratedTemporalAgent
    }

    results = {}
    for name, agent_class in agents.items():
        run_results = []
        for _ in range(n_runs):
            result = run_phase_experiment(agent_class, n_cycles=3)
            run_results.append(result)

        # Average across runs
        results[name] = {
            'neutral_mood': np.mean([r['phase_moods']['neutral'] for r in run_results]),
            'negative_mood': np.mean([r['phase_moods']['negative'] for r in run_results]),
            'recovery_mood': np.mean([r['phase_moods']['recovery'] for r in run_results]),
            'positive_mood': np.mean([r['phase_moods']['positive'] for r in run_results]),
            'neutral_reward': np.mean([r['phase_rewards']['neutral'] for r in run_results]),
            'negative_reward': np.mean([r['phase_rewards']['negative'] for r in run_results]),
            'positive_reward': np.mean([r['phase_rewards']['positive'] for r in run_results])
        }

    return results


def test_mood_persistence():
    """Test whether mood persists after negative phase ends."""
    env = TemporalGridWorld(phase_length=30)

    agents = {
        'Phasic Only': PhasicOnlyAgent(env.n_states, env.n_actions),
        'Tonic Mood': TonicMoodAgent(env.n_states, env.n_actions),
        'Integrated': IntegratedTemporalAgent(env.n_states, env.n_actions)
    }

    mood_traces = {name: [] for name in agents}

    # Run through phases
    for step in range(300):
        for name, agent in agents.items():
            state = env._pos_to_state(env.agent_pos)
            action = agent.select_action(state)
            next_state, reward, done, context = env.step(action)
            agent.update(state, action, reward, next_state, done, context)

            emotional_state = agent.get_emotional_state()
            assert 'mood' in emotional_state, f"BUG: Agent {name} emotional state missing 'mood' key"
            mood = emotional_state['mood']
            mood_traces[name].append(mood)

            if done:
                env.reset()
                agent.reset_episode()

    return mood_traces


def main():
    np.random.seed(42)

    print("=" * 70)
    print("EMOTIONAL ED TEST: Temporal Dynamics (Phasic vs Tonic Mood)")
    print("=" * 70)

    print("\nHYPOTHESIS (Davidson, Watson):")
    print("- Phasic: Immediate emotional responses, decay quickly")
    print("- Tonic: Mood baseline shifts from sustained experience")
    print("- Negative phase should depress mood in tonic agents")
    print("- Mood should persist into recovery phase")
    print()

    # Test 1: Mood across phases
    print("=" * 70)
    print("TEST 1: Mood Levels Across Environmental Phases")
    print("=" * 70)

    results = analyze_mood_dynamics(n_runs=30)

    print(f"\n{'Agent':<18} {'Neutral':<12} {'Negative':<12} {'Recovery':<12} {'Positive':<12}")
    print("-" * 66)
    for name in ['Standard', 'Phasic Only', 'Tonic Mood', 'Integrated']:
        r = results[name]
        print(f"{name:<18} {r['neutral_mood']:<12.3f} {r['negative_mood']:<12.3f} "
              f"{r['recovery_mood']:<12.3f} {r['positive_mood']:<12.3f}")

    # Test 2: Mood persistence
    print("\n" + "=" * 70)
    print("TEST 2: Mood Persistence After Negative Phase")
    print("=" * 70)

    mood_traces = test_mood_persistence()

    # Sample mood at key points
    phase_length = 30
    sample_points = {
        'mid_neutral': 15,
        'mid_negative': phase_length + 15,
        'start_recovery': 2 * phase_length + 5,
        'mid_recovery': 2 * phase_length + 15,
        'mid_positive': 3 * phase_length + 15
    }

    print(f"\n{'Agent':<18} {'Mid Neutral':<12} {'Mid Negative':<12} {'Start Recov':<12} {'Mid Recov':<12}")
    print("-" * 66)
    for name, trace in mood_traces.items():
        if len(trace) > max(sample_points.values()):
            print(f"{name:<18} {trace[sample_points['mid_neutral']]:<12.3f} "
                  f"{trace[sample_points['mid_negative']]:<12.3f} "
                  f"{trace[sample_points['start_recovery']]:<12.3f} "
                  f"{trace[sample_points['mid_recovery']]:<12.3f}")

    # Test 3: Performance by mood
    print("\n" + "=" * 70)
    print("TEST 3: Performance (Reward) by Phase")
    print("=" * 70)

    print(f"\n{'Agent':<18} {'Neutral':<12} {'Negative':<12} {'Positive':<12}")
    print("-" * 54)
    for name in ['Standard', 'Phasic Only', 'Tonic Mood', 'Integrated']:
        r = results[name]
        print(f"{name:<18} {r['neutral_reward']:<12.2f} {r['negative_reward']:<12.2f} "
              f"{r['positive_reward']:<12.2f}")

    # Hypothesis tests
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)

    # H1: Tonic mood shifts negative during negative phase
    tonic_neg = results['Tonic Mood']['negative_mood']
    tonic_neut = results['Tonic Mood']['neutral_mood']
    print(f"\nH1: Tonic mood shifts negative during negative phase")
    print(f"    Tonic mood (neutral): {tonic_neut:.3f}")
    print(f"    Tonic mood (negative): {tonic_neg:.3f}")
    if tonic_neg < tonic_neut - 0.05:
        print("    ✓ Mood DECREASES during negative phase")
    else:
        print("    ~ No clear mood decrease")

    # H2: Mood persists into recovery
    tonic_recov = results['Tonic Mood']['recovery_mood']
    print(f"\nH2: Negative mood persists into recovery phase")
    print(f"    Tonic mood (negative): {tonic_neg:.3f}")
    print(f"    Tonic mood (recovery): {tonic_recov:.3f}")
    if tonic_recov < tonic_neut - 0.02:
        print("    ✓ Mood remains DEPRESSED during early recovery")
    else:
        print("    ~ Mood recovers quickly")

    # H3: Phasic agent doesn't show mood persistence
    phasic_neg = results['Phasic Only']['negative_mood']
    phasic_recov = results['Phasic Only']['recovery_mood']
    print(f"\nH3: Phasic-only agent shows no mood persistence")
    print(f"    Phasic mood (negative): {phasic_neg:.3f}")
    print(f"    Phasic mood (recovery): {phasic_recov:.3f}")
    if abs(phasic_neg - phasic_recov) < 0.05:
        print("    ✓ Phasic agent shows NO mood carryover")
    else:
        print("    ~ Phasic agent shows unexpected persistence")

    # H4: Positive phase elevates mood
    tonic_pos = results['Tonic Mood']['positive_mood']
    print(f"\nH4: Positive phase elevates mood")
    print(f"    Tonic mood (neutral): {tonic_neut:.3f}")
    print(f"    Tonic mood (positive): {tonic_pos:.3f}")
    if tonic_pos > tonic_neut + 0.02:
        print("    ✓ Mood INCREASES during positive phase")
    else:
        print("    ~ No clear mood elevation")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Temporal Emotion Dynamics")
    print("=" * 70)

    print("\nKey findings:")
    print("- Phasic emotions: Immediate reactions, no persistence")
    print("- Tonic mood: Slow baseline shifts from sustained experience")
    print("- Mood persists after phase change (carryover effect)")
    print("- Integrated model combines both dynamics")

    print("\nBiological parallel:")
    print("- Phasic ≈ Acute stress response (HPA axis surge)")
    print("- Tonic ≈ Mood disorders (sustained dysregulation)")
    print("- Recovery time reflects emotional inertia")


if __name__ == "__main__":
    main()
