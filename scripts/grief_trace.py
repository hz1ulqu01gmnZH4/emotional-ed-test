"""Trace grief module behavior to find the bug."""

import sys
sys.path.insert(0, '/home/ak/emotional-ed-test')

import numpy as np
from gridworld_grief import AttachmentGridWorld
from agents_grief import GriefEDAgent, GriefModule


def trace_grief_module():
    """Trace grief module state."""
    print("=" * 60)
    print("TRACE 1: Grief module during pre-training")
    print("=" * 60)

    gm = GriefModule()
    print(f"Initial state: grief={gm.grief_level}, yearning={gm.yearning}, "
          f"attachment={gm.attachment_baseline}, loss_occurred={gm.loss_occurred}")

    # Simulate resource obtained
    from gridworld_grief import EmotionalContext
    ctx = EmotionalContext(resource_obtained=True)
    gm.compute(ctx)
    print(f"After resource_obtained: grief={gm.grief_level}, yearning={gm.yearning}, "
          f"attachment={gm.attachment_baseline}, loss_occurred={gm.loss_occurred}")

    # Call reset_episode
    gm.reset_episode()
    print(f"After reset_episode: grief={gm.grief_level}, yearning={gm.yearning}, "
          f"attachment={gm.attachment_baseline}, loss_occurred={gm.loss_occurred}")

    print("\n" + "=" * 60)
    print("TRACE 2: What happens at loss event")
    print("=" * 60)

    # Build up attachment first
    gm = GriefModule()
    for _ in range(10):
        ctx = EmotionalContext(resource_obtained=True)
        gm.compute(ctx)
    print(f"After 10 resource collections: attachment={gm.attachment_baseline}")

    # Now simulate loss
    ctx = EmotionalContext(resource_lost=True, time_since_loss=0)
    result = gm.compute(ctx)
    print(f"After loss event: grief={gm.grief_level}, yearning={gm.yearning}, "
          f"loss_occurred={gm.loss_occurred}")

    # Simulate time passing
    for t in [10, 20, 40, 60, 80]:
        ctx = EmotionalContext(resource_lost=False, time_since_loss=t)
        result = gm.compute(ctx)
        print(f"time_since_loss={t}: grief={gm.grief_level:.3f}, yearning={gm.yearning:.3f}")

    print("\n" + "=" * 60)
    print("TRACE 3: Check episode structure")
    print("=" * 60)

    grid_size = 7
    env = AttachmentGridWorld(size=grid_size, resource_pos=(3, 3), loss_step=60)
    agent = GriefEDAgent(n_states=env.n_states, n_actions=env.n_actions,
                         lr=0.1, epsilon=0.1, grief_weight=0.8, grid_size=grid_size)

    # Run ONE pre-train episode
    state = env.reset()
    print(f"Start of episode: step_count={env.step_count}, loss_occurred={env.loss_occurred}")

    for step in range(70):  # Run past loss_step
        action = agent.select_action(state)
        next_state, reward, done, ctx = env.step(action)
        agent.update(state, action, reward, next_state, done, ctx)

        if step in [58, 59, 60, 61]:
            print(f"Step {step}: env.step_count={env.step_count}, "
                  f"ctx.resource_lost={ctx.resource_lost}, "
                  f"agent.grief={agent.grief_module.grief_level:.3f}, "
                  f"agent.yearning={agent.grief_module.yearning:.3f}")

        state = next_state

    print("\n" + "=" * 60)
    print("TRACE 4: Check why grief builds up BEFORE loss")
    print("=" * 60)

    env = AttachmentGridWorld(size=grid_size, resource_pos=(3, 3), loss_step=60)
    agent = GriefEDAgent(n_states=env.n_states, n_actions=env.n_actions,
                         lr=0.1, epsilon=0.5, grief_weight=0.8, grid_size=grid_size)

    # Episode 1 - runs past loss
    state = env.reset()
    for step in range(100):
        action = agent.select_action(state)
        next_state, reward, done, ctx = env.step(action)
        agent.update(state, action, reward, next_state, done, ctx)
        if done:
            break
        state = next_state

    print(f"After episode 1: grief={agent.grief_module.grief_level:.3f}, "
          f"yearning={agent.grief_module.yearning:.3f}, "
          f"loss_occurred={agent.grief_module.loss_occurred}")

    # reset_episode
    agent.reset_episode()
    print(f"After reset_episode: grief={agent.grief_module.grief_level:.3f}, "
          f"yearning={agent.grief_module.yearning:.3f}, "
          f"loss_occurred={agent.grief_module.loss_occurred}")

    # Episode 2 - new env, but check initial state
    env = AttachmentGridWorld(size=grid_size, resource_pos=(3, 3), loss_step=60)
    state = env.reset()

    # Check BEFORE any steps
    print(f"\nEpisode 2 at step 0 (before any action):")
    print(f"  env.loss_occurred={env.loss_occurred}")
    print(f"  agent.grief_module.loss_occurred={agent.grief_module.loss_occurred}")

    for step in range(10):
        action = agent.select_action(state)
        next_state, reward, done, ctx = env.step(action)
        agent.update(state, action, reward, next_state, done, ctx)
        state = next_state

    print(f"\nAfter 10 steps in episode 2:")
    print(f"  grief={agent.grief_module.grief_level:.3f}")
    print(f"  yearning={agent.grief_module.yearning:.3f}")


if __name__ == "__main__":
    trace_grief_module()
