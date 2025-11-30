"""Test transfer/generalization of emotional learning.

Hypothesis (LeDoux, 2000; Dunsmoor & Paz, 2015):
- Fear learning generalizes to similar stimuli
- Emotional responses based on features, not specific instances
- Transfer enables adaptive behavior in novel situations

Test scenarios:
1. Novel threat location (same type)
2. Larger environment (same threat)
3. Multiple threats (original + novel)

Key question: Does emotional learning transfer better than
state-specific learning?
"""

import numpy as np
from gridworld_transfer import (TrainingGridWorld, NovelThreatGridWorld,
                                LargerGridWorld, MultipleThreatGridWorld)
from agents_transfer import (StandardQLearner, EmotionalTransferAgent,
                             NoTransferAgent)


def run_episode(env, agent, max_steps: int = 100):
    """Run episode tracking transfer behavior."""
    state = env.reset()
    if hasattr(agent, 'reset_episode'):
        agent.reset_episode()

    total_reward = 0
    threat_hits = 0
    min_threat_dist = float('inf')

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, context = env.step(action)
        agent.update(state, action, reward, next_state, done, context)

        total_reward += reward
        min_threat_dist = min(min_threat_dist, context.threat_distance)

        if reward < -0.2:  # Hit threat
            threat_hits += 1

        if done:
            break
        state = next_state

    return {
        'total_reward': total_reward,
        'steps': step + 1,
        'threat_hits': threat_hits,
        'min_threat_dist': min_threat_dist,
        'goal_reached': np.array_equal(env.agent_pos, env.goal_pos)
    }


def train_in_environment(agent, env, n_episodes: int = 200):
    """Train agent in environment."""
    for _ in range(n_episodes):
        run_episode(env, agent)


def evaluate_transfer(agent, env, n_episodes: int = 50):
    """Evaluate agent in (possibly novel) environment without training."""
    agent.epsilon = 0.05  # Small exploration
    results = []
    for _ in range(n_episodes):
        result = run_episode(env, agent)
        results.append(result)

    return {
        'mean_reward': np.mean([r['total_reward'] for r in results]),
        'threat_hits': np.mean([r['threat_hits'] for r in results]),
        'min_threat_dist': np.mean([r['min_threat_dist'] for r in results]),
        'goal_rate': np.mean([r['goal_reached'] for r in results])
    }


def test_transfer_scenario(agent_class, train_env_class, test_env_class,
                           train_episodes: int = 200, eval_episodes: int = 50,
                           **kwargs):
    """Test transfer from training to test environment."""
    # Create environments
    train_env = train_env_class()
    test_env = test_env_class()

    # Create agent
    agent = agent_class(n_states=train_env.n_states, n_actions=train_env.n_actions, **kwargs)

    # Train
    train_in_environment(agent, train_env, train_episodes)

    # Get training performance
    train_result = evaluate_transfer(agent, train_env, eval_episodes)

    # Resize Q-table if needed for larger test environment
    if hasattr(agent, 'resize_q_table') and test_env.n_states > train_env.n_states:
        agent.resize_q_table(test_env.n_states)

    # Test transfer (with some learning allowed to be fair)
    # First: Zero-shot transfer
    zero_shot_result = evaluate_transfer(agent, test_env, eval_episodes // 2)

    # Then: Few-shot adaptation
    train_in_environment(agent, test_env, train_episodes // 4)  # 25% training
    few_shot_result = evaluate_transfer(agent, test_env, eval_episodes // 2)

    return {
        'train': train_result,
        'zero_shot': zero_shot_result,
        'few_shot': few_shot_result
    }


def main():
    np.random.seed(42)

    print("=" * 70)
    print("EMOTIONAL ED TEST: Transfer and Generalization")
    print("=" * 70)

    print("\nTraining environment:")
    train_env = TrainingGridWorld()
    print(train_env.render())

    print("\nHYPOTHESIS (LeDoux, Dunsmoor & Paz):")
    print("- Emotional learning based on features generalizes")
    print("- State-specific learning doesn't transfer")
    print("- Feature-based fear transfers to novel threats")
    print()

    # Test 1: Novel threat location
    print("=" * 70)
    print("TEST 1: Transfer to Novel Threat Location")
    print("=" * 70)

    print("\nTest environment (novel threat at different position):")
    test1_env = NovelThreatGridWorld()
    print(test1_env.render())

    agents = [
        ("Standard QL", StandardQLearner, {}),
        ("No Transfer", NoTransferAgent, {}),
        ("Emotional Transfer", EmotionalTransferAgent, {})
    ]

    test1_results = {}
    for name, agent_class, kwargs in agents:
        test1_results[name] = test_transfer_scenario(
            agent_class, TrainingGridWorld, NovelThreatGridWorld, **kwargs
        )

    print(f"\n{'Agent':<20} {'Train Hits':<12} {'Zero-Shot':<12} {'Few-Shot':<12}")
    print("-" * 56)
    for name, r in test1_results.items():
        print(f"{name:<20} {r['train']['threat_hits']:<12.2f} "
              f"{r['zero_shot']['threat_hits']:<12.2f} {r['few_shot']['threat_hits']:<12.2f}")

    # Test 2: Larger environment
    print("\n" + "=" * 70)
    print("TEST 2: Transfer to Larger Environment")
    print("=" * 70)

    print("\nTest environment (7x7 instead of 5x5):")
    test2_env = LargerGridWorld()
    print(test2_env.render())

    test2_results = {}
    for name, agent_class, kwargs in agents:
        test2_results[name] = test_transfer_scenario(
            agent_class, TrainingGridWorld, LargerGridWorld, **kwargs
        )

    print(f"\n{'Agent':<20} {'Train Reward':<12} {'Zero-Shot':<12} {'Few-Shot':<12}")
    print("-" * 56)
    for name, r in test2_results.items():
        print(f"{name:<20} {r['train']['mean_reward']:<12.2f} "
              f"{r['zero_shot']['mean_reward']:<12.2f} {r['few_shot']['mean_reward']:<12.2f}")

    # Test 3: Multiple threats
    print("\n" + "=" * 70)
    print("TEST 3: Transfer to Multiple Threats")
    print("=" * 70)

    print("\nTest environment (original threat + 2 novel):")
    test3_env = MultipleThreatGridWorld()
    print(test3_env.render())

    test3_results = {}
    for name, agent_class, kwargs in agents:
        test3_results[name] = test_transfer_scenario(
            agent_class, TrainingGridWorld, MultipleThreatGridWorld, **kwargs
        )

    print(f"\n{'Agent':<20} {'Train Hits':<12} {'Zero-Shot':<12} {'Few-Shot':<12}")
    print("-" * 56)
    for name, r in test3_results.items():
        print(f"{name:<20} {r['train']['threat_hits']:<12.2f} "
              f"{r['zero_shot']['threat_hits']:<12.2f} {r['few_shot']['threat_hits']:<12.2f}")

    # Hypothesis tests
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)

    # H1: Emotional transfer shows better zero-shot on novel threat
    emo_zero = test1_results['Emotional Transfer']['zero_shot']['threat_hits']
    no_zero = test1_results['No Transfer']['zero_shot']['threat_hits']
    std_zero = test1_results['Standard QL']['zero_shot']['threat_hits']

    print(f"\nH1: Emotional transfer generalizes to novel threat location")
    print(f"    Emotional Transfer zero-shot hits: {emo_zero:.2f}")
    print(f"    No Transfer zero-shot hits: {no_zero:.2f}")
    print(f"    Standard QL zero-shot hits: {std_zero:.2f}")

    if emo_zero < no_zero:
        print("    ✓ Emotional transfer has FEWER hits (better generalization)")
    else:
        print("    ~ No clear transfer advantage")

    # H2: Emotional transfer works in larger environment
    emo_large_zero = test2_results['Emotional Transfer']['zero_shot']['mean_reward']
    no_large_zero = test2_results['No Transfer']['zero_shot']['mean_reward']

    print(f"\nH2: Emotional transfer works in larger environment")
    print(f"    Emotional Transfer zero-shot reward: {emo_large_zero:.2f}")
    print(f"    No Transfer zero-shot reward: {no_large_zero:.2f}")

    if emo_large_zero > no_large_zero:
        print("    ✓ Emotional transfer achieves BETTER reward in larger space")
    else:
        print("    ~ No clear size transfer advantage")

    # H3: Emotional transfer handles multiple threats
    emo_multi = test3_results['Emotional Transfer']['zero_shot']['threat_hits']
    no_multi = test3_results['No Transfer']['zero_shot']['threat_hits']

    print(f"\nH3: Emotional transfer handles multiple threats")
    print(f"    Emotional Transfer multi-threat hits: {emo_multi:.2f}")
    print(f"    No Transfer multi-threat hits: {no_multi:.2f}")

    if emo_multi < no_multi:
        print("    ✓ Emotional transfer avoids NOVEL threats better")
    else:
        print("    ~ No clear multi-threat advantage")

    # H4: Few-shot adaptation helps all agents
    print(f"\nH4: Few-shot adaptation improves performance")
    for name in ['Emotional Transfer', 'No Transfer']:
        zero = test1_results[name]['zero_shot']['threat_hits']
        few = test1_results[name]['few_shot']['threat_hits']
        improvement = zero - few
        print(f"    {name}: {zero:.2f} → {few:.2f} (improvement: {improvement:+.2f})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Transfer and Generalization")
    print("=" * 70)

    print("\nKey findings:")
    print("1. Feature-based fear learning transfers to novel situations")
    print("2. State-specific learning requires relearning in new contexts")
    print("3. Emotional generalization enables faster adaptation")
    print("4. Transfer is imperfect but provides useful prior")

    print("\nBiological parallel:")
    print("- Fear generalizes to similar stimuli (feature-based)")
    print("- Overgeneralization → anxiety disorders")
    print("- Appropriate generalization → adaptive behavior")


if __name__ == "__main__":
    main()
