"""Test emotion regulation: Can agents learn to overcome unnecessary fear?

Hypothesis (Ochsner & Gross, 2005):
- Cognitive reappraisal modifies emotional response by changing interpretation
- V(f(s)) ≠ V(s) when state is reframed
- Regulated agents should:
  1. Initially fear both real and fake threats
  2. Learn to discriminate based on outcomes
  3. Approach fake threats to get bonus
  4. Continue avoiding real threats

Environment:
- Real threat: Always harmful (avoid)
- Fake threat: Looks scary but gives bonus (overcome fear)
- Goal: Reach goal, ideally collecting fake threat bonus

Key metric: Does regulated agent collect fake threat bonus more often?
"""

import numpy as np
from gridworld_regulation import RegulationGridWorld
from agents_regulation import (StandardQLearner, UnregulatedFearAgent,
                               RegulatedFearAgent, ExplicitReappraisalAgent)

def run_episode(env: RegulationGridWorld, agent, max_steps: int = 100):
    """Run episode tracking regulation behavior."""
    state = env.reset()
    if hasattr(agent, 'reset_episode'):
        agent.reset_episode()

    total_reward = 0
    fake_threat_visited = False
    real_threat_visited = False

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, context = env.step(action)
        agent.update(state, action, reward, next_state, done, context)

        total_reward += reward

        # Track threat visits
        if context.threat_type == 'fake' and context.threat_distance < 1.0:
            fake_threat_visited = True
        if context.threat_type == 'real' and context.threat_distance < 1.0:
            real_threat_visited = True

        if done:
            break
        state = next_state

    return {
        'total_reward': total_reward,
        'fake_bonus_collected': env.fake_bonus_collected,
        'fake_threat_visited': fake_threat_visited,
        'real_threat_visited': real_threat_visited,
        'steps': step + 1,
        'goal_reached': np.array_equal(env.agent_pos, env.goal_pos)
    }


def train_and_evaluate(agent_class, n_train: int = 500, n_eval: int = 100, **kwargs):
    """Train agent and evaluate regulation effectiveness."""
    env = RegulationGridWorld()
    agent = agent_class(n_states=env.n_states, n_actions=env.n_actions, **kwargs)

    # Track learning over time
    fake_bonus_over_time = []

    # Training
    for ep in range(n_train):
        result = run_episode(env, agent)
        if ep % 50 == 49:
            fake_bonus_over_time.append(result['fake_bonus_collected'])

    # Evaluation
    agent.epsilon = 0.05
    results = []
    for _ in range(n_eval):
        result = run_episode(env, agent)
        results.append(result)

    # Get regulation stats if available
    reg_stats = {}
    if hasattr(agent, 'get_regulation_stats'):
        reg_stats = agent.get_regulation_stats()

    return {
        'fake_bonus_rate': np.mean([r['fake_bonus_collected'] for r in results]),
        'real_threat_rate': np.mean([r['real_threat_visited'] for r in results]),
        'goal_rate': np.mean([r['goal_reached'] for r in results]),
        'mean_reward': np.mean([r['total_reward'] for r in results]),
        'learning_curve': fake_bonus_over_time,
        'regulation_stats': reg_stats
    }


def analyze_regulation_learning(n_runs: int = 30):
    """Analyze how regulation develops over training."""
    env = RegulationGridWorld()

    agents = {
        'Standard': lambda: StandardQLearner(env.n_states, env.n_actions),
        'Unregulated Fear': lambda: UnregulatedFearAgent(env.n_states, env.n_actions),
        'Regulated Fear': lambda: RegulatedFearAgent(env.n_states, env.n_actions),
        'Explicit Reappraisal': lambda: ExplicitReappraisalAgent(env.n_states, env.n_actions)
    }

    results = {}
    for name, agent_fn in agents.items():
        run_results = []
        for _ in range(n_runs):
            agent = agent_fn()
            result = train_and_evaluate(type(agent), n_train=500, n_eval=50)
            run_results.append(result)

        results[name] = {
            'fake_bonus_rate': np.mean([r['fake_bonus_rate'] for r in run_results]),
            'real_threat_rate': np.mean([r['real_threat_rate'] for r in run_results]),
            'goal_rate': np.mean([r['goal_rate'] for r in run_results]),
            'mean_reward': np.mean([r['mean_reward'] for r in run_results])
        }

    return results


def main():
    np.random.seed(42)

    print("=" * 65)
    print("EMOTIONAL ED TEST: Emotion Regulation (Cognitive Reappraisal)")
    print("=" * 65)

    env = RegulationGridWorld()
    print("\nEnvironment layout:")
    print(env.render())
    print("\nLegend: X=real threat (harmful), F=fake threat (bonus), G=goal, A=agent")
    print()
    print("HYPOTHESIS (Ochsner & Gross):")
    print("- Cognitive reappraisal changes V(f(s)) by modifying state interpretation")
    print("- Unregulated: Fears all threats equally, misses fake threat bonus")
    print("- Regulated: Learns fake threat is safe, collects bonus")
    print()

    # Test 1: Regulation effectiveness
    print("=" * 65)
    print("TEST 1: Regulation Effectiveness (500 training, 100 eval)")
    print("=" * 65)

    comparisons = [
        ("Standard", StandardQLearner, {}),
        ("Unregulated Fear", UnregulatedFearAgent, {'fear_weight': 0.8}),
        ("Regulated Fear", RegulatedFearAgent, {'fear_weight': 0.8}),
        ("Explicit Reappraisal", ExplicitReappraisalAgent, {'fear_weight': 0.8})
    ]

    detailed = {}
    for name, agent_class, kwargs in comparisons:
        detailed[name] = train_and_evaluate(agent_class, **kwargs)

    print(f"\n{'Agent':<22} {'Fake Bonus':<12} {'Real Threat':<12} {'Goal Rate':<12} {'Reward':<10}")
    print("-" * 68)
    for name in ['Standard', 'Unregulated Fear', 'Regulated Fear', 'Explicit Reappraisal']:
        r = detailed[name]
        print(f"{name:<22} {r['fake_bonus_rate']:<12.1%} {r['real_threat_rate']:<12.1%} "
              f"{r['goal_rate']:<12.1%} {r['mean_reward']:<10.2f}")

    # Test 2: Regulation stats
    print("\n" + "=" * 65)
    print("TEST 2: Learned Regulation (Threat Discrimination)")
    print("=" * 65)

    for name in ['Regulated Fear', 'Explicit Reappraisal']:
        if detailed[name]['regulation_stats']:
            stats = detailed[name]['regulation_stats']
            print(f"\n{name}:")
            for k, v in stats.items():
                print(f"  {k}: {v:.3f}")

    # Hypothesis test
    print("\n" + "=" * 65)
    print("HYPOTHESIS TEST")
    print("=" * 65)

    unreg_fake = detailed['Unregulated Fear']['fake_bonus_rate']
    reg_fake = detailed['Regulated Fear']['fake_bonus_rate']
    explicit_fake = detailed['Explicit Reappraisal']['fake_bonus_rate']

    print(f"\nFake threat bonus collection rate:")
    print(f"  Unregulated Fear: {unreg_fake:.1%}")
    print(f"  Regulated Fear: {reg_fake:.1%}")
    print(f"  Explicit Reappraisal: {explicit_fake:.1%}")

    if reg_fake > unreg_fake + 0.05:
        print("\n✓ Regulated agent collects MORE fake bonuses (learned reappraisal)")
    else:
        print("\n~ Regulation effect not clearly demonstrated")

    unreg_real = detailed['Unregulated Fear']['real_threat_rate']
    reg_real = detailed['Regulated Fear']['real_threat_rate']

    print(f"\nReal threat approach rate:")
    print(f"  Unregulated Fear: {unreg_real:.1%}")
    print(f"  Regulated Fear: {reg_real:.1%}")

    if abs(reg_real - unreg_real) < 0.1:
        print("✓ Regulated agent still avoids REAL threats (selective regulation)")
    else:
        print("~ Regulation may be overgeneralized")

    # Test 3: Learning curve
    print("\n" + "=" * 65)
    print("TEST 3: Regulation Learning Curve")
    print("=" * 65)

    env = RegulationGridWorld()
    reg_agent = RegulatedFearAgent(env.n_states, env.n_actions)
    unreg_agent = UnregulatedFearAgent(env.n_states, env.n_actions)

    reg_curve = []
    unreg_curve = []

    for block in range(10):
        reg_bonuses = 0
        unreg_bonuses = 0

        for _ in range(50):
            reg_result = run_episode(env, reg_agent)
            unreg_result = run_episode(env, unreg_agent)

            reg_bonuses += reg_result['fake_bonus_collected']
            unreg_bonuses += unreg_result['fake_bonus_collected']

        reg_curve.append(reg_bonuses / 50)
        unreg_curve.append(unreg_bonuses / 50)

    print(f"\n{'Block':<10} {'Unregulated':<15} {'Regulated':<15}")
    print("-" * 40)
    for i, (u, r) in enumerate(zip(unreg_curve, reg_curve)):
        print(f"{(i+1)*50:<10} {u:<15.1%} {r:<15.1%}")

    # Check if regulation improves over time
    early_diff = np.mean(reg_curve[:3]) - np.mean(unreg_curve[:3])
    late_diff = np.mean(reg_curve[-3:]) - np.mean(unreg_curve[-3:])

    print(f"\nEarly advantage (blocks 1-3): {early_diff:+.1%}")
    print(f"Late advantage (blocks 8-10): {late_diff:+.1%}")

    if late_diff > early_diff:
        print("✓ Regulation advantage GROWS over training (learning to reappraise)")
    else:
        print("~ Regulation advantage doesn't clearly increase")


if __name__ == "__main__":
    main()
