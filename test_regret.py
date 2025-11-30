"""Test regret/counterfactual learning (Coricelli paradigm).

Hypothesis:
- Standard agent: Only learns from obtained rewards
- Regret agent: Learns from BOTH obtained and foregone outcomes
- Regret agent should show:
  1. Faster learning (uses more information)
  2. Better final performance
  3. Regret-modulated behavior (avoids previously regretted choices)

Following Coricelli et al. (2005) Nature Neuroscience.
"""

import numpy as np
from gridworld_regret import TwoDoorEnv, PartialFeedbackTwoDoorEnv
from agents_regret import StandardBanditAgent, RegretEDAgent, RegretAvoidanceAgent

def run_episode(env, agent):
    """Run single episode of two-door task."""
    state = env.reset()
    agent.reset_episode()

    while True:
        action = agent.select_action(state)
        next_state, reward, done, context = env.step(action)
        agent.update(state, action, reward, next_state, done, context)
        state = next_state
        if done:
            break

    return {
        'total_reward': env.total_reward,
        'choices': agent.choices.copy(),
        'q_values': agent.Q.copy()
    }


def evaluate_learning_speed(agent_class, env_class, n_runs: int = 50,
                            n_trials: int = 100, **agent_kwargs):
    """Measure how quickly agent learns optimal choice."""
    optimal_choice_rates = []

    for run in range(n_runs):
        env = env_class(n_trials=n_trials, correlation=-0.3)  # Anti-correlated rewards
        agent = agent_class(**agent_kwargs)

        state = env.reset()
        agent.reset_episode()

        choices = []
        optimal_choices = []

        for trial in range(n_trials):
            action = agent.select_action(state)
            next_state, reward, done, context = env.step(action)
            agent.update(state, action, reward, next_state, done, context)

            choices.append(action)
            # Track if agent chose the better door for this trial
            better_door = 0 if env.rewards_A[trial-1] > env.rewards_B[trial-1] else 1
            optimal_choices.append(1 if action == better_door else 0)

            state = next_state

        optimal_choice_rates.append(optimal_choices)

    return np.array(optimal_choice_rates)


def analyze_regret_response(agent_class, n_runs: int = 50, **agent_kwargs):
    """Analyze how agent responds to regret-inducing outcomes."""
    switch_after_regret = []
    switch_after_relief = []

    for run in range(n_runs):
        env = TwoDoorEnv(n_trials=100, correlation=-0.5)
        agent = agent_class(**agent_kwargs)

        state = env.reset()
        agent.reset_episode()

        prev_action = None
        prev_regret = None

        for trial in range(100):
            action = agent.select_action(state)
            next_state, reward, done, context = env.step(action)
            agent.update(state, action, reward, next_state, done, context)

            # Track switching behavior after regret vs relief
            if prev_action is not None and prev_regret is not None:
                switched = (action != prev_action)
                if prev_regret < -0.2:  # Experienced regret
                    switch_after_regret.append(1 if switched else 0)
                elif prev_regret > 0.2:  # Experienced relief
                    switch_after_relief.append(1 if switched else 0)

            prev_action = action
            prev_regret = context.obtained_reward - context.foregone_reward
            state = next_state

    return {
        'switch_after_regret': np.mean(switch_after_regret) if switch_after_regret else 0,
        'switch_after_relief': np.mean(switch_after_relief) if switch_after_relief else 0
    }


def main():
    np.random.seed(42)

    print("=" * 60)
    print("EMOTIONAL ED TEST: Regret/Counterfactual Channel")
    print("=" * 60)
    print("\nParadigm: Two-door choice task (Coricelli et al., 2005)")
    print("- Agent chooses door A or B")
    print("- Receives reward from chosen door")
    print("- Sees reward from BOTH doors (counterfactual feedback)")
    print()
    print("HYPOTHESIS:")
    print("- Standard: Only learns from obtained reward")
    print("- Regret agent: Learns from obtained AND foregone")
    print("- Regret agent should learn faster and perform better")
    print()

    # Test 1: Learning speed comparison
    print("=" * 60)
    print("TEST 1: Learning Speed (100 trials, 50 runs)")
    print("=" * 60)

    std_rates = evaluate_learning_speed(
        StandardBanditAgent, TwoDoorEnv,
        n_actions=2, lr=0.1, epsilon=0.1
    )

    regret_rates = evaluate_learning_speed(
        RegretEDAgent, TwoDoorEnv,
        n_actions=2, lr=0.1, epsilon=0.1, regret_weight=0.5
    )

    # Compute learning curves (mean over runs)
    std_curve = std_rates.mean(axis=0)
    regret_curve = regret_rates.mean(axis=0)

    # Smooth with rolling window
    window = 10
    std_smooth = np.convolve(std_curve, np.ones(window)/window, mode='valid')
    regret_smooth = np.convolve(regret_curve, np.ones(window)/window, mode='valid')

    print(f"\n{'Trial Block':<15} {'Standard':<15} {'Regret ED':<15}")
    print("-" * 45)
    for i in range(0, len(std_smooth), 20):
        print(f"{i+1}-{i+10:<10} {std_smooth[i]:<15.1%} {regret_smooth[i]:<15.1%}")

    print(f"\n{'Overall optimal rate':<25} {std_curve.mean():<15.1%} {regret_curve.mean():<15.1%}")

    # Test 2: Final performance comparison
    print("\n" + "=" * 60)
    print("TEST 2: Final Performance (after 500 episodes)")
    print("=" * 60)

    # Train for longer
    std_rewards = []
    regret_rewards = []

    for _ in range(100):
        env = TwoDoorEnv(n_trials=50, correlation=-0.3)
        std_agent = StandardBanditAgent(n_actions=2, lr=0.1, epsilon=0.05)
        regret_agent = RegretEDAgent(n_actions=2, lr=0.1, epsilon=0.05)

        # Train both
        for ep in range(500):
            run_episode(env, std_agent)
            env.reset()
            run_episode(env, regret_agent)
            env.reset()

        # Evaluate (greedy)
        std_agent.epsilon = 0
        regret_agent.epsilon = 0

        std_result = run_episode(env, std_agent)
        env.reset()
        regret_result = run_episode(env, regret_agent)

        std_rewards.append(std_result['total_reward'])
        regret_rewards.append(regret_result['total_reward'])

    print(f"\n{'Metric':<25} {'Standard':<15} {'Regret ED':<15}")
    print("-" * 55)
    print(f"{'Mean total reward':<25} {np.mean(std_rewards):<15.2f} {np.mean(regret_rewards):<15.2f}")
    print(f"{'Std dev':<25} {np.std(std_rewards):<15.2f} {np.std(regret_rewards):<15.2f}")

    # Test 3: Regret-induced switching
    print("\n" + "=" * 60)
    print("TEST 3: Behavioral Response to Regret")
    print("=" * 60)

    std_response = analyze_regret_response(StandardBanditAgent, n_actions=2, lr=0.1, epsilon=0.1)
    regret_response = analyze_regret_response(RegretEDAgent, n_actions=2, lr=0.1, epsilon=0.1)
    aversion_response = analyze_regret_response(RegretAvoidanceAgent, n_actions=2, lr=0.1, epsilon=0.1)

    print(f"\n{'Agent':<20} {'Switch after regret':<20} {'Switch after relief':<20}")
    print("-" * 60)
    print(f"{'Standard':<20} {std_response['switch_after_regret']:<20.1%} {std_response['switch_after_relief']:<20.1%}")
    print(f"{'Regret ED':<20} {regret_response['switch_after_regret']:<20.1%} {regret_response['switch_after_relief']:<20.1%}")
    print(f"{'Regret Aversion':<20} {aversion_response['switch_after_regret']:<20.1%} {aversion_response['switch_after_relief']:<20.1%}")

    # Hypothesis test
    print("\n" + "=" * 60)
    print("HYPOTHESIS TEST")
    print("=" * 60)

    # Learning speed
    early_std = std_curve[:20].mean()
    early_regret = regret_curve[:20].mean()
    learning_diff = early_regret - early_std

    print(f"\nEarly learning (first 20 trials):")
    print(f"  Standard: {early_std:.1%}")
    print(f"  Regret ED: {early_regret:.1%}")
    print(f"  Difference: {learning_diff:+.1%}")

    if learning_diff > 0.02:
        print("✓ Regret agent learns FASTER (uses counterfactual information)")
    else:
        print("~ No significant learning speed difference")

    # Regret response
    regret_switch_diff = regret_response['switch_after_regret'] - regret_response['switch_after_relief']
    std_switch_diff = std_response['switch_after_regret'] - std_response['switch_after_relief']

    print(f"\nRegret-induced switching (switch_regret - switch_relief):")
    print(f"  Standard: {std_switch_diff:+.1%}")
    print(f"  Regret ED: {regret_switch_diff:+.1%}")

    if regret_switch_diff > std_switch_diff + 0.05:
        print("✓ Regret agent shows MORE regret-sensitive switching")
    else:
        print("~ Similar switching patterns")

    # Q-value comparison
    print("\n" + "=" * 60)
    print("Q-VALUE COMPARISON (after training)")
    print("=" * 60)

    env = TwoDoorEnv(n_trials=50)
    std_agent = StandardBanditAgent(n_actions=2, lr=0.1, epsilon=0.1)
    regret_agent = RegretEDAgent(n_actions=2, lr=0.1, epsilon=0.1)

    for _ in range(200):
        run_episode(env, std_agent)
        env.reset()
        run_episode(env, regret_agent)
        env.reset()

    print(f"\n{'Agent':<20} {'Q(A)':<15} {'Q(B)':<15}")
    print("-" * 50)
    print(f"{'Standard':<20} {std_agent.Q[0]:<15.3f} {std_agent.Q[1]:<15.3f}")
    print(f"{'Regret ED':<20} {regret_agent.Q[0]:<15.3f} {regret_agent.Q[1]:<15.3f}")


if __name__ == "__main__":
    main()
