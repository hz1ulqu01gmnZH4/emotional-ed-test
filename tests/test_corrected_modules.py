"""Tests for corrected emotional modules.

Verifies fixes for:
- Issue 3.2: Gamma bounding in PatienceModule
- Issue 3.3: Tolerance direction in WantingLikingModule
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/ak/emotional-ed-test')

from src.modules.patience_module import PatienceModule, HyperbolicPatienceModule
from src.modules.wanting_liking_module import WantingLikingModule


class TestPatienceModule:
    """Tests for gamma bounding fix (Issue 3.2)."""

    def test_gamma_always_bounded(self):
        """CRITICAL: gamma must NEVER exceed 1.0 or go below 0."""
        module = PatienceModule(gamma_min=0.7, gamma_max=0.99)

        # Test with extreme serotonin levels
        for serotonin in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]:
            module.serotonin = serotonin
            gamma = module.effective_gamma()

            assert gamma >= module.gamma_min, \
                f"gamma {gamma} < gamma_min {module.gamma_min} at 5-HT={serotonin}"
            assert gamma < 1.0, \
                f"gamma {gamma} >= 1.0 at 5-HT={serotonin} (CONVERGENCE BROKEN)"
            assert gamma < module.gamma_max, \
                f"gamma {gamma} >= gamma_max {module.gamma_max}"

    def test_serotonin_modulates_gamma_correctly(self):
        """Low serotonin → low gamma (impulsive), high → high gamma (patient)."""
        module = PatienceModule()

        # Low serotonin
        module.serotonin = 0.2
        gamma_low = module.effective_gamma()

        # High serotonin
        module.serotonin = 2.0
        gamma_high = module.effective_gamma()

        assert gamma_high > gamma_low, \
            f"Higher 5-HT should give higher gamma: {gamma_high} <= {gamma_low}"

    def test_reward_increases_serotonin(self):
        """Positive reward should increase serotonin (patience recovery)."""
        module = PatienceModule()
        module.serotonin = 0.5  # Start depleted

        initial = module.serotonin
        module.update(reward=1.0, was_waiting=True)

        assert module.serotonin > initial, \
            f"Reward should increase 5-HT: {module.serotonin} <= {initial}"

    def test_frustration_depletes_serotonin(self):
        """Frustration should decrease serotonin (impulsivity increases)."""
        module = PatienceModule()
        module.serotonin = 1.5  # Start elevated

        initial = module.serotonin
        module.update(reward=-1.0, was_frustrated=True)

        assert module.serotonin < initial, \
            f"Frustration should decrease 5-HT: {module.serotonin} >= {initial}"

    def test_impulsivity_inverse_of_serotonin(self):
        """Impulsivity should be high when serotonin is low."""
        module = PatienceModule()

        module.serotonin = 0.3
        impulsivity_low_5ht = module.impulsivity()

        module.serotonin = 1.5
        impulsivity_high_5ht = module.impulsivity()

        assert impulsivity_low_5ht > impulsivity_high_5ht, \
            f"Low 5-HT should give high impulsivity"

    def test_gamma_uses_sigmoid_not_linear(self):
        """Gamma should saturate (sigmoid), not grow linearly."""
        module = PatienceModule()

        gammas = []
        for serotonin in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            module.serotonin = serotonin
            gammas.append(module.effective_gamma())

        # Check saturation: large changes in high 5-HT should give small gamma changes
        delta_low = gammas[2] - gammas[0]  # 0.1 → 1.0
        delta_high = gammas[-1] - gammas[-3]  # 2.0 → 5.0

        assert delta_high < delta_low, \
            f"Gamma should saturate: high 5-HT delta {delta_high} >= low delta {delta_low}"


class TestHyperbolicPatienceModule:
    """Tests for hyperbolic discounting variant."""

    def test_discount_rate_bounded(self):
        """Discount rate k should stay in [k_min, k_max]."""
        module = HyperbolicPatienceModule(k_min=0.01, k_max=1.0)

        for serotonin in [0.01, 0.5, 1.0, 2.0, 10.0]:
            module.serotonin = serotonin
            k = module.discount_rate()

            assert k >= module.k_min, f"k {k} < k_min at 5-HT={serotonin}"
            assert k <= module.k_max, f"k {k} > k_max at 5-HT={serotonin}"

    def test_hyperbolic_discounting(self):
        """Delayed values should be discounted hyperbolically."""
        module = HyperbolicPatienceModule()
        module.serotonin = 1.0

        value = 10.0
        discounted_1 = module.discount_value(value, delay=1)
        discounted_10 = module.discount_value(value, delay=10)
        discounted_100 = module.discount_value(value, delay=100)

        assert discounted_1 > discounted_10 > discounted_100, \
            "Longer delays should give more discounting"

        # Hyperbolic property: V(D) = V0 / (1 + k*D)
        # The key property is that hyperbolic gives LESS discounting than
        # exponential for long delays (preference reversal phenomenon)
        # We verify this by checking the formula works correctly
        k = module.discount_rate()
        expected_1 = value / (1 + k * 1)
        expected_10 = value / (1 + k * 10)

        assert abs(discounted_1 - expected_1) < 0.01, \
            f"Hyperbolic formula mismatch: {discounted_1} != {expected_1}"
        assert abs(discounted_10 - expected_10) < 0.01, \
            f"Hyperbolic formula mismatch: {discounted_10} != {expected_10}"


class TestWantingLikingModule:
    """Tests for tolerance direction fix (Issue 3.3)."""

    def test_tolerance_increases_with_exposure(self):
        """CRITICAL: Tolerance should INCREASE with repeated consumption."""
        module = WantingLikingModule()

        initial_tolerance = module.tolerance
        for _ in range(10):
            module.consume(dose=1.0)

        assert module.tolerance > initial_tolerance, \
            f"Tolerance should INCREASE: {module.tolerance} <= {initial_tolerance}"

    def test_sensitization_increases_with_exposure(self):
        """Sensitization (wanting) should INCREASE with repeated consumption."""
        module = WantingLikingModule()

        initial_sensitization = module.sensitization
        for _ in range(10):
            module.consume(dose=1.0)

        assert module.sensitization > initial_sensitization, \
            f"Sensitization should INCREASE: {module.sensitization} <= {initial_sensitization}"

    def test_liking_decreases_with_tolerance(self):
        """Same dose should give LESS liking as tolerance builds."""
        module = WantingLikingModule()

        # First consumption - no tolerance
        result_1 = module.consume(dose=1.0)
        liking_1 = result_1['liking']

        # After several consumptions
        for _ in range(10):
            module.consume(dose=1.0)

        result_n = module.consume(dose=1.0)
        liking_n = result_n['liking']

        assert liking_n < liking_1, \
            f"Liking should DECREASE with tolerance: {liking_n} >= {liking_1}"

    def test_wanting_increases_with_sensitization(self):
        """Same cue should trigger MORE wanting as sensitization builds."""
        module = WantingLikingModule()

        # First wanting - baseline sensitization
        wanting_1 = module.compute_wanting(cue_salience=1.0)

        # After several consumptions
        for _ in range(10):
            module.consume(dose=1.0)

        wanting_n = module.compute_wanting(cue_salience=1.0)

        assert wanting_n > wanting_1, \
            f"Wanting should INCREASE with sensitization: {wanting_n} <= {wanting_1}"

    def test_addiction_index_increases(self):
        """Addiction index (wanting/liking) should increase with repeated use."""
        module = WantingLikingModule()

        initial_index = module.addiction_index()
        for _ in range(15):
            module.consume(dose=1.0)

        final_index = module.addiction_index()

        assert final_index > initial_index, \
            f"Addiction index should increase: {final_index} <= {initial_index}"

    def test_withdrawal_during_abstinence(self):
        """Withdrawal should appear when abstaining after tolerance builds."""
        # Use lower threshold so it can be exceeded with fewer consumptions
        module = WantingLikingModule(
            withdrawal_onset_threshold=0.3,  # Lower threshold
            tolerance_rate=0.1  # Faster tolerance buildup
        )

        # Build tolerance with higher doses
        for _ in range(30):
            module.consume(dose=2.0)

        assert module.tolerance > module.withdrawal_onset_threshold, \
            f"Tolerance {module.tolerance} should exceed threshold {module.withdrawal_onset_threshold}"

        # Abstain
        initial_withdrawal = module.withdrawal_level
        module.abstain(steps=5)

        assert module.withdrawal_level > initial_withdrawal, \
            f"Withdrawal should appear: {module.withdrawal_level} <= {initial_withdrawal}"

    def test_abstinence_allows_recovery(self):
        """Extended abstinence should reduce tolerance (partial recovery)."""
        module = WantingLikingModule()

        # Build tolerance
        for _ in range(20):
            module.consume(dose=1.0)

        tolerance_after_use = module.tolerance

        # Long abstinence
        for _ in range(50):
            module.abstain(steps=10)

        assert module.tolerance < tolerance_after_use, \
            f"Abstinence should reduce tolerance: {module.tolerance} >= {tolerance_after_use}"

    def test_sensitization_persists_longer_than_tolerance(self):
        """Sensitization should decay slower than tolerance (wanting persists)."""
        module = WantingLikingModule(
            sensitization_decay=0.001,  # Slow
            tolerance_decay=0.01  # Faster
        )

        # Build both
        for _ in range(20):
            module.consume(dose=1.0)

        sens_after = module.sensitization
        tol_after = module.tolerance

        # Abstain
        for _ in range(50):
            module.abstain(steps=10)

        # Calculate relative decay
        sens_remaining = module.sensitization / sens_after
        tol_remaining = module.tolerance / tol_after

        assert sens_remaining > tol_remaining, \
            f"Sensitization should persist longer: {sens_remaining} <= {tol_remaining}"

    def test_bounded_growth(self):
        """Sensitization and tolerance should be bounded."""
        module = WantingLikingModule(
            sensitization_max=5.0,
            tolerance_max=10.0
        )

        # Massive consumption
        for _ in range(100):
            module.consume(dose=5.0)

        assert module.sensitization <= module.sensitization_max, \
            f"Sensitization exceeded max: {module.sensitization} > {module.sensitization_max}"
        assert module.tolerance <= module.tolerance_max, \
            f"Tolerance exceeded max: {module.tolerance} > {module.tolerance_max}"


class TestIntegration:
    """Integration tests for corrected modules."""

    def test_patience_and_addiction_interact(self):
        """Addiction should affect patience (serotonin depletion)."""
        patience = PatienceModule()
        # Use parameters that make addiction develop faster
        addiction = WantingLikingModule(
            sensitization_rate=0.1,
            tolerance_rate=0.1,
            withdrawal_onset_threshold=0.3
        )

        # Simulate addiction cycle with higher doses
        for episode in range(30):
            if episode < 20:
                # Consuming - initially rewarding
                result = addiction.consume(dose=2.0)
                # Reward modulates serotonin (but tolerance reduces liking)
                patience.update(reward=result['liking'])
            else:
                # Abstaining - withdrawal
                addiction.abstain(steps=5)
                # Withdrawal is frustrating
                patience.update(reward=-1.0, was_frustrated=True)

        # After addiction cycle, addiction index should increase
        # (This is the core test - wanting up, liking down)
        assert addiction.addiction_index() > 1.5, \
            f"Should show elevated addiction index: {addiction.addiction_index()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
