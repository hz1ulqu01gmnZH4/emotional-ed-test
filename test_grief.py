"""Test grief/attachment loss behavior.

Hypothesis (Panksepp, Bowlby):
- Attachment creates expected interaction pattern
- Loss triggers PANIC/GRIEF system
- Yearning phase: continued seeking of lost object
- Standard RL: Immediate adaptation (no yearning)
- Grief agent: Prolonged visits to lost resource location

Key metric: Visits to resource location AFTER loss
"""

import numpy as np
from gridworld_grief import AttachmentGridWorld
from agents_grief import StandardQLearner, GriefEDAgent

def run_episode(env: AttachmentGridWorld, agent, track_visits: bool = True):
    """Run episode tracking visits to resource location."""
    state = env.reset()
    if hasattr(agent, 'reset_episode'):
        agent.reset_episode()
    if hasattr(agent, 'reset_tracking'):
        agent.reset_tracking()

    resource_state = env._pos_to_state(env.resource_pos)

    visits_before_loss = 0
    visits_after_loss = 0
    visit_times_after_loss = []

    for step in range(150):
        action = agent.select_action(state)
        next_state, reward, done, context = env.step(action)
        agent.update(state, action, reward, next_state, done, context)

        # Track visits to resource location
        if next_state == resource_state:
            if env.loss_occurred:
                visits_after_loss += 1
                visit_times_after_loss.append(step - env.loss_step)
            else:
                visits_before_loss += 1

        if done:
            break
        state = next_state

    return {
        'visits_before_loss': visits_before_loss,
        'visits_after_loss': visits_after_loss,
        'visit_times_after_loss': visit_times_after_loss,
        'total_reward': env.step_count
    }


def analyze_post_loss_visits(agent_class, n_runs: int = 100, **agent_kwargs):
    """Analyze visiting pattern after resource loss."""
    all_visits_after = []
    all_visit_times = []

    for run in range(n_runs):
        env = AttachmentGridWorld(size=5, resource_pos=(2, 2), loss_step=50)
        agent = agent_class(n_states=env.n_states, n_actions=env.n_actions, **agent_kwargs)

        # Pre-train to learn resource location (minimal training)
        for _ in range(5):  # Less pre-training so effect is visible
            state = env.reset()
            if hasattr(agent, 'reset_episode'):
                agent.reset_episode()
            for step in range(50):  # Only before loss
                action = agent.select_action(state)
                next_state, reward, done, context = env.step(action)
                agent.update(state, action, reward, next_state, done, context)
                state = next_state

        # Now run full episode with loss
        env = AttachmentGridWorld(size=5, resource_pos=(2, 2), loss_step=50)
        result = run_episode(env, agent)

        all_visits_after.append(result['visits_after_loss'])
        all_visit_times.extend(result['visit_times_after_loss'])

    return {
        'mean_visits_after': np.mean(all_visits_after),
        'std_visits_after': np.std(all_visits_after),
        'visit_times': all_visit_times
    }


def analyze_yearning_decay(agent_class, n_runs: int = 50, **agent_kwargs):
    """Analyze how visits decay over time after loss."""
    # Bin visits by time since loss
    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    bin_visits = {b: [] for b in bins}

    for run in range(n_runs):
        env = AttachmentGridWorld(size=5, resource_pos=(2, 2), loss_step=50)
        agent = agent_class(n_states=env.n_states, n_actions=env.n_actions, **agent_kwargs)

        # Pre-train
        for _ in range(20):
            state = env.reset()
            if hasattr(agent, 'reset_episode'):
                agent.reset_episode()
            for step in range(50):
                action = agent.select_action(state)
                next_state, reward, done, context = env.step(action)
                agent.update(state, action, reward, next_state, done, context)
                state = next_state

        # Full episode with loss
        env = AttachmentGridWorld(size=5, resource_pos=(2, 2), loss_step=50)
        result = run_episode(env, agent)

        # Bin the visit times
        for vt in result['visit_times_after_loss']:
            for (lo, hi) in bins:
                if lo <= vt < hi:
                    bin_visits[(lo, hi)].append(1)
                    break

    # Compute mean visits per bin
    return {b: len(v) / n_runs for b, v in bin_visits.items()}


def main():
    np.random.seed(42)

    print("=" * 60)
    print("EMOTIONAL ED TEST: Grief/Attachment Channel")
    print("=" * 60)
    print("\nParadigm: Resource attachment and loss")
    print("- Agent learns location of renewable resource")
    print("- Resource disappears at step 50 (loss event)")
    print("- Measure: Visits to resource location AFTER loss")
    print()
    print("HYPOTHESIS (Panksepp PANIC/GRIEF system):")
    print("- Standard: Immediately adapts, stops visiting")
    print("- Grief agent: Yearning phase → continued visits → gradual adaptation")
    print()

    # Test 1: Post-loss visits comparison
    print("=" * 60)
    print("TEST 1: Visits to Lost Resource Location")
    print("=" * 60)

    std_results = analyze_post_loss_visits(
        StandardQLearner, n_runs=100, lr=0.1, epsilon=0.1
    )

    grief_results = analyze_post_loss_visits(
        GriefEDAgent, n_runs=100, lr=0.1, epsilon=0.1, grief_weight=0.7
    )

    print(f"\n{'Metric':<30} {'Standard':<15} {'Grief ED':<15}")
    print("-" * 60)
    print(f"{'Mean visits after loss':<30} {std_results['mean_visits_after']:<15.2f} {grief_results['mean_visits_after']:<15.2f}")
    print(f"{'Std dev':<30} {std_results['std_visits_after']:<15.2f} {grief_results['std_visits_after']:<15.2f}")

    # Test 2: Yearning decay analysis
    print("\n" + "=" * 60)
    print("TEST 2: Yearning Decay Over Time")
    print("=" * 60)

    std_decay = analyze_yearning_decay(StandardQLearner, n_runs=50, lr=0.1, epsilon=0.1)
    grief_decay = analyze_yearning_decay(GriefEDAgent, n_runs=50, lr=0.1, epsilon=0.1, grief_weight=0.7)

    print(f"\n{'Time after loss':<20} {'Standard visits':<20} {'Grief visits':<20}")
    print("-" * 60)
    for (lo, hi) in sorted(std_decay.keys()):
        print(f"{lo}-{hi} steps{'':<10} {std_decay[(lo, hi)]:<20.2f} {grief_decay[(lo, hi)]:<20.2f}")

    # Test 3: Single episode visualization
    print("\n" + "=" * 60)
    print("TEST 3: Single Episode Behavior")
    print("=" * 60)

    for name, agent_class, kwargs in [
        ("Standard", StandardQLearner, {}),
        ("Grief ED", GriefEDAgent, {'grief_weight': 0.7})
    ]:
        env = AttachmentGridWorld(size=5, resource_pos=(2, 2), loss_step=50)
        agent = agent_class(n_states=env.n_states, n_actions=env.n_actions,
                           lr=0.1, epsilon=0.1, **kwargs)

        # Pre-train
        for _ in range(30):
            state = env.reset()
            if hasattr(agent, 'reset_episode'):
                agent.reset_episode()
            for step in range(50):
                action = agent.select_action(state)
                next_state, reward, done, context = env.step(action)
                agent.update(state, action, reward, next_state, done, context)
                state = next_state

        # Full episode
        env = AttachmentGridWorld(size=5, resource_pos=(2, 2), loss_step=50)
        result = run_episode(env, agent)

        print(f"\n{name} agent:")
        print(f"  Visits before loss: {result['visits_before_loss']}")
        print(f"  Visits after loss: {result['visits_after_loss']}")
        if result['visit_times_after_loss']:
            print(f"  Visit times after loss: {result['visit_times_after_loss'][:10]}...")

    # Hypothesis test
    print("\n" + "=" * 60)
    print("HYPOTHESIS TEST")
    print("=" * 60)

    visit_diff = grief_results['mean_visits_after'] - std_results['mean_visits_after']
    print(f"\nPost-loss visits difference: {visit_diff:+.2f}")

    if visit_diff > 0.5:
        print("✓ Grief agent shows MORE visits to lost resource (yearning)")
        print("  → Grief channel produces seeking behavior after loss")
    elif visit_diff < -0.5:
        print("✗ Grief agent shows FEWER visits (unexpected)")
    else:
        print("~ Similar visiting patterns")

    # Check decay pattern
    early_grief = grief_decay[(0, 20)]
    late_grief = grief_decay[(60, 80)]
    early_std = std_decay[(0, 20)]
    late_std = std_decay[(60, 80)]

    print(f"\nYearning decay pattern:")
    print(f"  Standard: {early_std:.2f} (early) → {late_std:.2f} (late)")
    print(f"  Grief ED: {early_grief:.2f} (early) → {late_grief:.2f} (late)")

    if early_grief > early_std and (early_grief - late_grief) > (early_std - late_std):
        print("✓ Grief agent shows yearning followed by adaptation")
    else:
        print("~ Yearning pattern not clearly distinguished")


if __name__ == "__main__":
    main()
