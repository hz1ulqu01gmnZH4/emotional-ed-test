"""Quick debug of grief experiment after fixes."""

import sys
sys.path.insert(0, '/home/ak/emotional-ed-test')

import numpy as np
from gridworld_grief import AttachmentGridWorld
from agents_grief import StandardQLearner, GriefEDAgent


def debug_single_run():
    """Debug a single run to see what's happening."""
    np.random.seed(42)

    grid_size = 7
    resource_pos = (3, 3)
    loss_step = 60

    # Create grief agent
    env = AttachmentGridWorld(size=grid_size, resource_pos=resource_pos, loss_step=loss_step)
    agent = GriefEDAgent(n_states=env.n_states, n_actions=env.n_actions,
                         lr=0.1, epsilon=0.1, grief_weight=0.8, grid_size=grid_size)

    print("Pre-training (30 episodes)...")
    for ep in range(30):
        state = env.reset()
        agent.reset_episode()
        for step in range(loss_step):
            action = agent.select_action(state)
            next_state, reward, done, ctx = env.step(action)
            agent.update(state, action, reward, next_state, done, ctx)
            state = next_state

        if ep % 10 == 9:
            print(f"  Ep {ep+1}: attachment={agent.grief_module.attachment_baseline:.3f}, "
                  f"resource_pos={agent.resource_pos}")

    print(f"\nAfter pre-training:")
    print(f"  attachment_baseline = {agent.grief_module.attachment_baseline:.3f}")
    print(f"  resource_pos = {agent.resource_pos}")
    print(f"  resource_state = {agent.resource_state}")

    # Now run test episode
    print("\n" + "=" * 50)
    print("TEST EPISODE (with loss at step 60)")
    print("=" * 50)

    env = AttachmentGridWorld(size=grid_size, resource_pos=resource_pos, loss_step=loss_step)
    state = env.reset()
    resource_state = env._pos_to_state(env.resource_pos)
    agent.epsilon = 0.02

    visits_before = 0
    visits_after = 0

    for step in range(200):
        action = agent.select_action(state)
        next_state, reward, done, ctx = env.step(action)
        agent.update(state, action, reward, next_state, done, ctx)

        at_resource = (next_state == resource_state)
        if at_resource:
            if env.loss_occurred:
                visits_after += 1
            else:
                visits_before += 1

        # Print key moments
        if step == loss_step - 1:
            print(f"\nStep {step} (just before loss):")
            print(f"  grief={agent.grief_module.grief_level:.3f}")
            print(f"  yearning={agent.grief_module.yearning:.3f}")
            print(f"  attachment={agent.grief_module.attachment_baseline:.3f}")
            print(f"  visits_before={visits_before}")

        if step == loss_step:
            print(f"\nStep {step} (loss event):")
            print(f"  ctx.resource_lost={ctx.resource_lost}")
            print(f"  grief={agent.grief_module.grief_level:.3f}")
            print(f"  yearning={agent.grief_module.yearning:.3f}")

        if step in [70, 80, 100, 120, 150]:
            print(f"\nStep {step}:")
            print(f"  grief={agent.grief_module.grief_level:.3f}")
            print(f"  yearning={agent.grief_module.yearning:.3f}")
            print(f"  visits_after={visits_after}")

        if done:
            break
        state = next_state

    print(f"\n" + "=" * 50)
    print(f"FINAL: visits_before={visits_before}, visits_after={visits_after}")
    print("=" * 50)

    # Compare with standard agent
    print("\n\nNow running STANDARD agent for comparison...")
    np.random.seed(42)

    env = AttachmentGridWorld(size=grid_size, resource_pos=resource_pos, loss_step=loss_step)
    std_agent = StandardQLearner(n_states=env.n_states, n_actions=env.n_actions,
                                  lr=0.1, epsilon=0.1)

    for ep in range(30):
        state = env.reset()
        for step in range(loss_step):
            action = std_agent.select_action(state)
            next_state, reward, done, ctx = env.step(action)
            std_agent.update(state, action, reward, next_state, done, ctx)
            state = next_state

    env = AttachmentGridWorld(size=grid_size, resource_pos=resource_pos, loss_step=loss_step)
    state = env.reset()
    std_agent.epsilon = 0.02
    std_visits_after = 0

    for step in range(200):
        action = std_agent.select_action(state)
        next_state, reward, done, ctx = env.step(action)
        std_agent.update(state, action, reward, next_state, done, ctx)

        if next_state == resource_state and env.loss_occurred:
            std_visits_after += 1

        if done:
            break
        state = next_state

    print(f"\nStandard agent visits_after={std_visits_after}")
    print(f"Grief agent visits_after={visits_after}")
    print(f"Difference: {visits_after - std_visits_after}")


if __name__ == "__main__":
    debug_single_run()
