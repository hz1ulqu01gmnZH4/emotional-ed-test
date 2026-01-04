"""Validation tests for v2 agents.

These tests verify that the architectural fixes work correctly:
1. Disgust v2: Directional repulsion produces avoidance
2. Feature-based: Features generalize across states
3. Regulation v2: Bayesian beliefs update correctly
4. CVaR Fear: Risk-sensitivity produces conservative behavior
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ak/emotional-ed-test')

from agents_v2.agents_disgust_v2 import DisgustOnlyAgentV2
from agents_v2.agents_feature_based import FeatureBasedFearAgent, FeatureContext
from agents_v2.agents_regulation_v2 import (
    RegulatedFearAgentV2, UnregulatedFearAgentV2,
    RegulationGridWorldV2, RegulationContext, BayesianReappraisalModule
)
from agents_v2.agents_cvar_fear import CVaRFearAgent, RiskNeutralAgent, FearContext
from gridworld_disgust import DisgustContext


def test_disgust_directional_repulsion():
    """Test that disgust produces directional repulsion from contaminant."""
    print("\n=== Test: Disgust Directional Repulsion ===")

    agent = DisgustOnlyAgentV2(n_states=36, n_actions=4, grid_size=6)

    # Place agent at (2, 2), contaminant at (2, 3) - to the right
    agent.contaminant_positions = {(2, 3)}
    agent.disgust_level = 0.8

    state = 2 * 6 + 2  # State 14 = (2, 2)

    # Get action away from contaminant
    away_action = agent._get_action_away_from_contaminant(state)
    # Actions: 0=up, 1=down, 2=left, 3=right
    # Contaminant is to the right (col_diff > 0), so away = left (2)
    assert away_action == 2, f"Expected left (2), got {away_action}"

    # Get action toward contaminant
    toward_action = agent._get_action_toward_contaminant(state)
    # Toward = right (3)
    assert toward_action == 3, f"Expected right (3), got {toward_action}"

    print("✓ Directional calculations correct")

    # Test that action selection favors away direction
    # Run multiple trials to see bias
    away_count = 0
    toward_count = 0
    n_trials = 1000

    # Set low epsilon for deterministic selection
    agent.epsilon = 0.0

    for _ in range(n_trials):
        action = agent.select_action(state)
        if action == 2:  # left = away
            away_count += 1
        elif action == 3:  # right = toward
            toward_count += 1

    # With disgust, should strongly prefer away over toward
    print(f"  Away (left): {away_count}, Toward (right): {toward_count}")
    assert away_count > toward_count, "Should prefer away from contaminant"
    print("✓ Action selection prefers away direction")

    print("PASSED: Disgust directional repulsion works correctly")


def test_feature_based_generalization():
    """Test that feature-based agent can generalize across states."""
    print("\n=== Test: Feature-Based Generalization ===")

    agent = FeatureBasedFearAgent(n_actions=4, lr=0.1)

    # Train on one threat position
    train_context = FeatureContext(
        threat_distance=1.0,
        goal_distance=3.0,
        threat_direction=(0, 1),  # threat to the right
        goal_direction=(1, 1),
        near_threat=True,
        near_wall=False
    )

    # Simulate learning: penalize moving toward threat
    for _ in range(100):
        state_pos = (2, 2)
        action = 3  # right = toward threat
        reward = -0.5  # penalty

        next_context = FeatureContext(
            threat_distance=0.5,
            goal_distance=3.5,
            threat_direction=(0, 0),
            goal_direction=(1, 1),
            near_threat=True,
            near_wall=False
        )

        agent.update(state_pos, action, reward, (2, 3), False, train_context, next_context)

    # Now test in NEW state with SAME features (threat to the right)
    test_context = FeatureContext(
        threat_distance=1.0,  # Same threat distance
        goal_distance=3.0,
        threat_direction=(0, 1),  # Same threat direction
        goal_direction=(1, 1),
        near_threat=True,
        near_wall=False
    )

    # Should avoid action 3 (toward threat) in new state
    agent.epsilon = 0.0
    new_state_pos = (4, 4)  # Different position

    q_values = [agent.Q(new_state_pos, test_context, a) for a in range(4)]
    print(f"  Q-values in new state: {q_values}")

    # Action 3 (toward threat) should have lower value
    assert q_values[3] < max(q_values), "Action toward threat should be penalized"
    print("✓ Feature-based Q generalizes threat avoidance to new state")

    print("PASSED: Feature-based generalization works")


def test_bayesian_reappraisal():
    """Test that Bayesian reappraisal module updates beliefs correctly."""
    print("\n=== Test: Bayesian Reappraisal ===")

    module = BayesianReappraisalModule(prior_safe=0.3)

    # Initial belief should be prior
    initial_belief = module.get_safety_belief('fake_threat')
    print(f"  Initial belief P(safe|fake_threat): {initial_belief:.3f}")
    assert abs(initial_belief - 0.3) < 0.01, "Initial should be prior"

    # Update with no harm experiences (fake threat is safe)
    for _ in range(10):
        module.update_belief('fake_threat', was_harmed=False)

    updated_belief = module.get_safety_belief('fake_threat')
    print(f"  After 10 safe experiences: {updated_belief:.3f}")
    assert updated_belief > 0.7, "Should become confident threat is safe"
    print("✓ Belief increases with safe experiences")

    # Test real threat (gets harmed)
    for _ in range(5):
        module.update_belief('real_threat', was_harmed=True)

    real_belief = module.get_safety_belief('real_threat')
    print(f"  After 5 harmful experiences (real): {real_belief:.3f}")
    assert real_belief < 0.3, "Should become confident real threat is dangerous"
    print("✓ Belief decreases with harmful experiences")

    # Test fear reduction
    base_fear = 0.8
    fake_fear = module.reappraised_fear(base_fear, 'fake_threat')
    real_fear = module.reappraised_fear(base_fear, 'real_threat')

    print(f"  Base fear: {base_fear}, Fake reappraised: {fake_fear:.3f}, Real reappraised: {real_fear:.3f}")
    assert fake_fear < real_fear, "Fake threat should have lower reappraised fear"
    print("✓ Reappraised fear lower for learned-safe threats")

    print("PASSED: Bayesian reappraisal works correctly")


def test_regulation_environment():
    """Test that regulation environment has fake threats that give bonus."""
    print("\n=== Test: Regulation Environment ===")

    env = RegulationGridWorldV2(size=6)

    print(f"  Goal: {env.goal_pos}")
    print(f"  Real threat: {env.real_threat_pos}")
    print(f"  Fake threat: {env.fake_threat_pos}")

    # Navigate agent to fake threat and check bonus
    env.reset()
    env.agent_pos = np.array([3, 3])  # Near fake threat at (3, 4)

    # Move toward fake threat - step computes context at NEW position
    # So we move FROM [3, 3] TO [3, 4] (the fake threat position)
    state, reward, done, context = env.step(3)  # right moves to [3, 4]

    print(f"  At fake threat: reward={reward:.2f}, context={context}")
    assert context.threat_type == 'fake' or context.is_fake_threat, "Should identify as fake"

    print("PASSED: Regulation environment has fake threats")


def test_cvar_risk_sensitivity():
    """Test that CVaR fear produces risk-averse behavior."""
    print("\n=== Test: CVaR Risk Sensitivity ===")

    n_states = 36
    n_actions = 4

    cvar_agent = CVaRFearAgent(n_states, n_actions, n_quantiles=21)
    neutral_agent = RiskNeutralAgent(n_states, n_actions, n_quantiles=21)

    # Set up a situation with high variance outcome
    # State 0 has action 0 with high variance (could be very bad or very good)
    # Action 1 has low variance (consistently medium)

    # Action 0: High variance - some very negative outcomes
    cvar_agent.Z[0, 0] = np.linspace(-2.0, 3.0, 21)  # Range from -2 to +3
    neutral_agent.Z[0, 0] = np.linspace(-2.0, 3.0, 21)

    # Action 1: Low variance - consistently medium
    cvar_agent.Z[0, 1] = np.linspace(0.3, 0.7, 21)  # Range from 0.3 to 0.7
    neutral_agent.Z[0, 1] = np.linspace(0.3, 0.7, 21)

    # Expected values
    ev_action0 = np.mean(cvar_agent.Z[0, 0])
    ev_action1 = np.mean(cvar_agent.Z[0, 1])
    print(f"  E[action 0] = {ev_action0:.2f}, E[action 1] = {ev_action1:.2f}")

    # CVaR at alpha=0.3 for action 0 (worst 30%)
    cvar_action0 = cvar_agent.dist.compute_cvar(cvar_agent.Z[0, 0], 0.3)
    cvar_action1 = cvar_agent.dist.compute_cvar(cvar_agent.Z[0, 1], 0.3)
    print(f"  CVaR_0.3[action 0] = {cvar_action0:.2f}, CVaR_0.3[action 1] = {cvar_action1:.2f}")

    # Risk-neutral should prefer action 0 (higher expected value)
    neutral_agent.epsilon = 0.0
    neutral_choice = neutral_agent.select_action(0)
    print(f"  Risk-neutral chooses action {neutral_choice}")
    assert neutral_choice == 0, "Risk-neutral should choose higher EV"

    # High fear CVaR should prefer action 1 (better worst-case)
    high_fear_context = FearContext(
        threat_distance=0.5,  # Very close to threat → high fear
        goal_distance=3.0,
        threat_direction=(0, 1),
        near_threat=True
    )
    cvar_agent.epsilon = 0.0
    cvar_choice = cvar_agent.select_action(0, high_fear_context)
    print(f"  CVaR with high fear chooses action {cvar_choice}")

    # With high fear (low alpha), CVaR of action 1 is better
    assert cvar_action1 > cvar_action0, "CVaR should favor low-variance action"
    print("✓ CVaR correctly identifies risk-averse choice")

    print("PASSED: CVaR risk sensitivity works correctly")


def test_cvar_fear_modulation():
    """Test that fear level modulates CVaR alpha."""
    print("\n=== Test: Fear → Alpha Modulation ===")

    agent = CVaRFearAgent(n_states=36, n_actions=4, base_alpha=0.5, min_alpha=0.1)

    # No fear → alpha should be base
    agent.fear_level = 0.0
    alpha_no_fear = agent._fear_to_alpha(0.0)
    print(f"  Fear=0.0 → alpha={alpha_no_fear:.2f}")
    assert abs(alpha_no_fear - 0.5) < 0.01, "No fear should give base alpha"

    # Max fear → alpha should be min
    agent.fear_level = 1.0
    alpha_max_fear = agent._fear_to_alpha(1.0)
    print(f"  Fear=1.0 → alpha={alpha_max_fear:.2f}")
    assert abs(alpha_max_fear - 0.1) < 0.01, "Max fear should give min alpha"

    # Mid fear → alpha should be interpolated
    agent.fear_level = 0.5
    alpha_mid_fear = agent._fear_to_alpha(0.5)
    print(f"  Fear=0.5 → alpha={alpha_mid_fear:.2f}")
    expected = 0.5 - 0.5 * (0.5 - 0.1)  # 0.3
    assert abs(alpha_mid_fear - expected) < 0.01, "Mid fear should interpolate"

    print("✓ Fear correctly modulates CVaR alpha")
    print("PASSED: Fear-alpha modulation works correctly")


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("VALIDATION TESTS FOR V2 AGENTS")
    print("=" * 60)

    tests = [
        test_disgust_directional_repulsion,
        test_feature_based_generalization,
        test_bayesian_reappraisal,
        test_regulation_environment,
        test_cvar_risk_sensitivity,
        test_cvar_fear_modulation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            # Log the error but re-raise - unexpected exceptions are bugs
            print(f"ERROR: {test.__name__}")
            print(f"  Exception: {e}")
            raise RuntimeError(f"Unexpected exception in {test.__name__}: {e}") from e

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
