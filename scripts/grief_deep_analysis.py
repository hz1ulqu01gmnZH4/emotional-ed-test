"""Deep analysis of Grief experiment to understand why no effect is observed.

Hypotheses for null result:
1. Both agents visit equally because grid is small (5x5) and resource is center
2. Grief module not building attachment during pre-training
3. Yearning signal too weak to affect behavior
4. Q-values not sufficiently learned to show difference
5. Both agents have high epsilon (exploration) drowning out effect
"""

import sys
sys.path.insert(0, '/home/ak/emotional-ed-test')

import numpy as np
from gridworld_grief import AttachmentGridWorld
from agents_grief import StandardQLearner, GriefEDAgent


def diagnose_attachment_building():
    """Check if grief module builds attachment during pre-training."""
    print("=" * 70)
    print("DIAGNOSIS 1: Attachment Building During Pre-training")
    print("=" * 70)

    env = AttachmentGridWorld(size=5, resource_pos=(2, 2), loss_step=50)
    agent = GriefEDAgent(n_states=env.n_states, n_actions=env.n_actions,
                         lr=0.1, epsilon=0.1, grief_weight=0.7)

    print("\nTracking attachment over 20 pre-training episodes...")

    for ep in range(20):
        state = env.reset()
        agent.reset_episode()  # This resets grief module!

        resources_collected = 0
        for step in range(50):
            action = agent.select_action(state)
            next_state, reward, done, ctx = env.step(action)
            agent.update(state, action, reward, next_state, done, ctx)
            if ctx.resource_obtained:
                resources_collected += 1
            state = next_state

        attachment = agent.grief_module.attachment_baseline
        print(f"  Episode {ep+1}: collected {resources_collected}, attachment={attachment:.3f}")

    print(f"\nFinal attachment after pre-training: {agent.grief_module.attachment_baseline:.3f}")
    print("\nPROBLEM IDENTIFIED: reset_episode() resets attachment to 0!")
    print("Attachment never accumulates across episodes.")


def diagnose_grief_signals_during_loss():
    """Check grief signals when loss occurs."""
    print("\n" + "=" * 70)
    print("DIAGNOSIS 2: Grief Signals During Loss Event")
    print("=" * 70)

    env = AttachmentGridWorld(size=5, resource_pos=(2, 2), loss_step=50)
    agent = GriefEDAgent(n_states=env.n_states, n_actions=env.n_actions,
                         lr=0.1, epsilon=0.1, grief_weight=0.7)

    # DON'T reset episode so attachment accumulates
    state = env.reset()

    print("\nRunning single long episode (150 steps, loss at step 50)...")
    print("Tracking grief module state:\n")

    for step in range(150):
        action = agent.select_action(state)
        next_state, reward, done, ctx = env.step(action)
        agent.update(state, action, reward, next_state, done, ctx)

        # Print key moments
        if step == 49:
            print(f"Step {step} (before loss):")
            print(f"  Attachment: {agent.grief_module.attachment_baseline:.3f}")
            print(f"  Grief: {agent.grief_module.grief_level:.3f}")
            print(f"  Yearning: {agent.grief_module.yearning:.3f}")

        if step == 50:
            print(f"\nStep {step} (LOSS OCCURS):")
            print(f"  resource_lost in context: {ctx.resource_lost}")
            print(f"  Attachment: {agent.grief_module.attachment_baseline:.3f}")
            print(f"  Grief: {agent.grief_module.grief_level:.3f}")
            print(f"  Yearning: {agent.grief_module.yearning:.3f}")

        if step in [60, 70, 80, 100, 120]:
            print(f"\nStep {step} (post-loss):")
            print(f"  Grief: {agent.grief_module.grief_level:.3f}")
            print(f"  Yearning: {agent.grief_module.yearning:.3f}")

        if done:
            break
        state = next_state


def diagnose_visit_pattern():
    """Analyze visit patterns in detail."""
    print("\n" + "=" * 70)
    print("DIAGNOSIS 3: Detailed Visit Pattern Analysis")
    print("=" * 70)

    resource_state = 2 * 5 + 2  # (2,2) in 5x5 grid = state 12

    for name, AgentClass, kwargs in [
        ("Standard", StandardQLearner, {}),
        ("Grief ED", GriefEDAgent, {"grief_weight": 0.7})
    ]:
        print(f"\n{name} Agent:")

        # Collect visits across multiple runs
        all_visits_by_step = {i: 0 for i in range(150)}
        n_runs = 50

        for _ in range(n_runs):
            env = AttachmentGridWorld(size=5, resource_pos=(2, 2), loss_step=50)
            agent = AgentClass(n_states=env.n_states, n_actions=env.n_actions,
                               lr=0.1, epsilon=0.1, **kwargs)

            # NO pre-training - start fresh each run
            state = env.reset()
            if hasattr(agent, 'reset_episode'):
                agent.reset_episode()

            for step in range(150):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                agent.update(state, action, reward, next_state, done, ctx)

                if next_state == resource_state:
                    all_visits_by_step[step] += 1

                if done:
                    break
                state = next_state

        # Summarize by phase
        before_loss = sum(all_visits_by_step[i] for i in range(50)) / n_runs
        early_after = sum(all_visits_by_step[i] for i in range(50, 80)) / n_runs
        mid_after = sum(all_visits_by_step[i] for i in range(80, 110)) / n_runs
        late_after = sum(all_visits_by_step[i] for i in range(110, 150)) / n_runs

        print(f"  Visits before loss (0-49):    {before_loss:.2f}")
        print(f"  Visits early after (50-79):   {early_after:.2f}")
        print(f"  Visits mid after (80-109):    {mid_after:.2f}")
        print(f"  Visits late after (110-149):  {late_after:.2f}")


def diagnose_q_values():
    """Check Q-values for resource state."""
    print("\n" + "=" * 70)
    print("DIAGNOSIS 4: Q-Value Analysis for Resource State")
    print("=" * 70)

    resource_state = 2 * 5 + 2  # state 12

    for name, AgentClass, kwargs in [
        ("Standard", StandardQLearner, {}),
        ("Grief ED", GriefEDAgent, {"grief_weight": 0.7})
    ]:
        print(f"\n{name} Agent Q-values for resource state (12):")

        env = AttachmentGridWorld(size=5, resource_pos=(2, 2), loss_step=50)
        agent = AgentClass(n_states=env.n_states, n_actions=env.n_actions,
                           lr=0.1, epsilon=0.1, **kwargs)

        # Pre-train
        for ep in range(20):
            state = env.reset()
            if hasattr(agent, 'reset_episode'):
                agent.reset_episode()
            for step in range(50):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                agent.update(state, action, reward, next_state, done, ctx)
                state = next_state

        print(f"  After pre-training: {agent.Q[resource_state]}")

        # Run with loss
        env = AttachmentGridWorld(size=5, resource_pos=(2, 2), loss_step=50)
        state = env.reset()
        if hasattr(agent, 'reset_episode'):
            agent.reset_episode()

        for step in range(150):
            action = agent.select_action(state)
            next_state, reward, done, ctx = env.step(action)
            agent.update(state, action, reward, next_state, done, ctx)

            if step == 50:
                print(f"  At loss (step 50): {agent.Q[resource_state]}")
            if step == 100:
                print(f"  Post-loss (step 100): {agent.Q[resource_state]}")

            if done:
                break
            state = next_state

        print(f"  Final (step 150): {agent.Q[resource_state]}")


def diagnose_statistical_test_setup():
    """Reproduce the exact statistical test setup."""
    print("\n" + "=" * 70)
    print("DIAGNOSIS 5: Reproducing Statistical Test Setup")
    print("=" * 70)

    # This matches test_statistical_extended.py exactly
    standard_visits = []
    grief_visits = []

    for seed in range(50):
        np.random.seed(seed)

        # Standard agent
        env = AttachmentGridWorld(size=5, resource_pos=(2, 2), loss_step=50)
        agent = StandardQLearner(n_states=env.n_states, n_actions=env.n_actions,
                                 lr=0.1, epsilon=0.1)

        # Pre-train
        for _ in range(20):
            state = env.reset()
            for step in range(50):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                agent.update(state, action, reward, next_state, done, ctx)
                state = next_state

        # Full episode with loss
        env = AttachmentGridWorld(size=5, resource_pos=(2, 2), loss_step=50)
        state = env.reset()
        resource_state = env._pos_to_state(env.resource_pos)
        visits_after = 0

        for step in range(150):
            action = agent.select_action(state)
            next_state, reward, done, ctx = env.step(action)
            agent.update(state, action, reward, next_state, done, ctx)
            if next_state == resource_state and env.loss_occurred:
                visits_after += 1
            if done:
                break
            state = next_state
        standard_visits.append(visits_after)

        # Grief agent
        np.random.seed(seed)
        env = AttachmentGridWorld(size=5, resource_pos=(2, 2), loss_step=50)
        agent = GriefEDAgent(n_states=env.n_states, n_actions=env.n_actions,
                             lr=0.1, epsilon=0.1, grief_weight=0.7)

        for _ in range(20):
            state = env.reset()
            if hasattr(agent, 'reset_episode'):
                agent.reset_episode()
            for step in range(50):
                action = agent.select_action(state)
                next_state, reward, done, ctx = env.step(action)
                agent.update(state, action, reward, next_state, done, ctx)
                state = next_state

        env = AttachmentGridWorld(size=5, resource_pos=(2, 2), loss_step=50)
        state = env.reset()
        resource_state = env._pos_to_state(env.resource_pos)
        visits_after = 0

        for step in range(150):
            action = agent.select_action(state)
            next_state, reward, done, ctx = env.step(action)
            agent.update(state, action, reward, next_state, done, ctx)
            if next_state == resource_state and env.loss_occurred:
                visits_after += 1
            if done:
                break
            state = next_state
        grief_visits.append(visits_after)

    print(f"\nStandard: mean={np.mean(standard_visits):.2f}, std={np.std(standard_visits):.2f}")
    print(f"Grief ED: mean={np.mean(grief_visits):.2f}, std={np.std(grief_visits):.2f}")

    # CRITICAL: Check if values are nearly identical
    print(f"\nDifference: {np.mean(grief_visits) - np.mean(standard_visits):.4f}")

    # Check distribution
    print(f"\nStandard visits distribution: min={min(standard_visits)}, max={max(standard_visits)}")
    print(f"Grief visits distribution: min={min(grief_visits)}, max={max(grief_visits)}")


def main():
    diagnose_attachment_building()
    diagnose_grief_signals_during_loss()
    diagnose_visit_pattern()
    diagnose_q_values()
    diagnose_statistical_test_setup()

    print("\n" + "=" * 70)
    print("SUMMARY: Root Cause Analysis")
    print("=" * 70)
    print("""
Key Issues Identified:

1. ATTACHMENT RESET BUG: reset_episode() resets grief_module, which resets
   attachment_baseline to 0. This means attachment never builds across
   pre-training episodes.

2. GRIEF NEVER TRIGGERS: Since attachment_baseline is 0 when loss occurs,
   grief_level = 0 * attachment_strength = 0. No grief signal!

3. YEARNING BOOST INEFFECTIVE: Even if yearning existed, the boost of
   yearning * 0.1 added to ALL Q-values doesn't create directional bias
   toward the resource location.

4. HIGH BASELINE VISITS: With epsilon=0.1 and small grid (5x5), both
   agents visit the center (resource) state ~47 times just by random
   exploration, drowning any potential effect.

FIXES NEEDED:
1. Don't reset attachment between episodes during pre-training
2. Or build attachment within-episode faster
3. Implement directional yearning (boost actions toward resource)
4. Use larger grid or lower epsilon to reduce baseline visits
""")


if __name__ == "__main__":
    main()
