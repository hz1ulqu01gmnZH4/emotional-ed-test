"""Wanting/Liking dissociation module with corrected tolerance dynamics.

Fixes issue 3.3: Tolerance should INCREASE with exposure (not decrease).

Based on:
- Berridge (2007): Incentive Salience Theory
- Robinson & Berridge (2008): Incentive Sensitization Theory of Addiction
- GPT-5/Gemini review recommendations

Key distinction:
- WANTING (DA, mesolimbic): Incentive salience, motivational pull
  → SENSITIZES with repeated exposure (wanting increases)

- LIKING (Opioids, NAc): Hedonic impact, actual pleasure
  → TOLERATES with repeated exposure (same dose gives less pleasure)

This creates addiction dynamics: Wanting ↑↑ while Liking ↓↓
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ConsumptionEvent:
    """Record of a consumption event."""
    dose: float
    wanting_at_consumption: float
    liking_received: float
    timestamp: int


class WantingLikingModule:
    """Proper wanting/liking dissociation with bounded dynamics.

    Key dynamics:
    1. SENSITIZATION: Wanting increases with repeated exposure
       Sens_{t+1} = Sens_t + η_s × dose × (1 - Sens_t/S_max) - ρ_s × Sens_t

    2. TOLERANCE: Effective pleasure decreases with repeated exposure
       Tol_{t+1} = Tol_t + η_t × dose × (1 - Tol_t/T_max) - ρ_t × Tol_t
       Liking_effective = Liking_base / (1 + Tol)

    3. WITHDRAWAL: Negative state when abstaining after dependence
    """

    def __init__(
        self,
        # Sensitization parameters (wanting increases)
        sensitization_rate: float = 0.05,
        sensitization_max: float = 5.0,
        sensitization_decay: float = 0.001,  # Slow decay during abstinence

        # Tolerance parameters (liking decreases)
        tolerance_rate: float = 0.03,
        tolerance_max: float = 10.0,
        tolerance_decay: float = 0.01,  # Faster recovery than sensitization

        # Withdrawal parameters
        withdrawal_onset_threshold: float = 3.0,  # Tolerance level to trigger withdrawal
        withdrawal_severity_scale: float = 0.5,
        withdrawal_decay: float = 0.05
    ):
        """Initialize wanting/liking module.

        Args:
            sensitization_rate: How fast wanting sensitizes per dose
            sensitization_max: Maximum sensitization level
            sensitization_decay: How fast sensitization decays during abstinence
            tolerance_rate: How fast tolerance builds per dose
            tolerance_max: Maximum tolerance level
            tolerance_decay: How fast tolerance decays during abstinence
            withdrawal_onset_threshold: Tolerance level at which withdrawal starts
            withdrawal_severity_scale: How severe withdrawal is relative to tolerance
            withdrawal_decay: How fast withdrawal subsides
        """
        # Sensitization (wanting multiplier)
        self.sensitization = 1.0  # Starts at baseline
        self.sensitization_rate = sensitization_rate
        self.sensitization_max = sensitization_max
        self.sensitization_decay = sensitization_decay

        # Tolerance (liking divisor)
        self.tolerance = 0.0  # Starts at zero (no tolerance)
        self.tolerance_rate = tolerance_rate
        self.tolerance_max = tolerance_max
        self.tolerance_decay = tolerance_decay

        # Withdrawal
        self.withdrawal_level = 0.0
        self.withdrawal_onset_threshold = withdrawal_onset_threshold
        self.withdrawal_severity_scale = withdrawal_severity_scale
        self.withdrawal_decay = withdrawal_decay

        # Tracking
        self.time_since_last_use = 0
        self.total_consumption = 0.0
        self.consumption_history: list = []

    def compute_wanting(self, cue_salience: float,
                        baseline_wanting: float = 1.0) -> float:
        """Compute current wanting level.

        Wanting = baseline × cue_salience × sensitization

        Args:
            cue_salience: How salient/attractive the cue is
            baseline_wanting: Base wanting level

        Returns:
            Current wanting (can exceed 1.0 with sensitization)
        """
        # Sensitization amplifies wanting
        wanting = baseline_wanting * cue_salience * self.sensitization

        # Withdrawal also increases wanting (craving)
        if self.withdrawal_level > 0:
            wanting *= (1.0 + self.withdrawal_level)

        return float(wanting)

    def compute_liking(self, hedonic_value: float,
                       baseline_liking: float = 1.0) -> float:
        """Compute actual pleasure received from consumption.

        Liking = baseline × hedonic_value / (1 + tolerance)

        Args:
            hedonic_value: Inherent pleasurability of stimulus
            baseline_liking: Base liking level

        Returns:
            Actual pleasure experienced (diminishes with tolerance)
        """
        # Tolerance reduces actual pleasure
        liking = baseline_liking * hedonic_value / (1.0 + self.tolerance)

        return float(max(0.0, liking))

    def consume(self, dose: float, cue_salience: float = 1.0,
                hedonic_value: float = 1.0, timestamp: int = 0) -> Dict:
        """Process a consumption event.

        Updates sensitization, tolerance, and resets withdrawal.

        Args:
            dose: Amount consumed (higher = more effect)
            cue_salience: Salience of consumption cue
            hedonic_value: Inherent pleasure of stimulus
            timestamp: Current timestep (for logging)

        Returns:
            Dict with wanting, liking, and addiction metrics
        """
        # Compute wanting and liking at consumption
        wanting = self.compute_wanting(cue_salience)
        liking = self.compute_liking(hedonic_value)

        # Update SENSITIZATION (wanting increases)
        # Bounded growth: slows as approaching max
        sensitization_delta = (
            self.sensitization_rate * dose *
            (1.0 - self.sensitization / self.sensitization_max)
        )
        self.sensitization = min(
            self.sensitization_max,
            self.sensitization + sensitization_delta
        )

        # Update TOLERANCE (liking decreases)
        # Bounded growth: tolerance increases toward max
        tolerance_delta = (
            self.tolerance_rate * dose *
            (1.0 - self.tolerance / self.tolerance_max)
        )
        self.tolerance = min(
            self.tolerance_max,
            self.tolerance + tolerance_delta
        )

        # Reset withdrawal (consumption relieves it)
        self.withdrawal_level = 0.0
        self.time_since_last_use = 0
        self.total_consumption += dose

        # Log consumption
        event = ConsumptionEvent(
            dose=dose,
            wanting_at_consumption=wanting,
            liking_received=liking,
            timestamp=timestamp
        )
        self.consumption_history.append(event)

        return {
            'wanting': wanting,
            'liking': liking,
            'total_value': wanting * liking,
            'addiction_index': self.addiction_index(),
            'sensitization': self.sensitization,
            'tolerance': self.tolerance
        }

    def abstain(self, steps: int = 1):
        """Process abstinence period.

        Updates withdrawal and allows partial recovery.

        Args:
            steps: Number of steps of abstinence
        """
        self.time_since_last_use += steps

        # Withdrawal kicks in if tolerant
        if self.tolerance > self.withdrawal_onset_threshold:
            # Withdrawal severity proportional to tolerance
            target_withdrawal = (
                self.withdrawal_severity_scale *
                (self.tolerance - self.withdrawal_onset_threshold)
            )
            # Withdrawal builds up then decays
            if self.time_since_last_use < 10:
                # Building phase
                self.withdrawal_level = min(
                    target_withdrawal,
                    self.withdrawal_level + 0.1 * target_withdrawal
                )
            else:
                # Decay phase
                self.withdrawal_level = max(
                    0.0,
                    self.withdrawal_level - self.withdrawal_decay * steps
                )
        else:
            self.withdrawal_level = max(
                0.0,
                self.withdrawal_level - self.withdrawal_decay * steps
            )

        # Sensitization decays slowly (wanting decreases slowly)
        self.sensitization = max(
            1.0,  # Cannot go below baseline
            self.sensitization - self.sensitization_decay * steps
        )

        # Tolerance decays faster (liking recovers)
        self.tolerance = max(
            0.0,  # Cannot go below zero
            self.tolerance - self.tolerance_decay * steps
        )

    def addiction_index(self) -> float:
        """Compute addiction index: wanting / liking ratio.

        High addiction index = high wanting with low liking
        (hallmark of addiction: want it but don't enjoy it)

        Returns:
            Addiction index (1.0 = normal, >1 = addicted)
        """
        # Avoid division by zero
        effective_liking = 1.0 / (1.0 + self.tolerance)
        return self.sensitization / max(0.01, effective_liking)

    def craving_level(self) -> float:
        """Current craving level.

        Craving = sensitization × withdrawal

        Returns:
            Craving intensity
        """
        return self.sensitization * (1.0 + self.withdrawal_level)

    def is_addicted(self, threshold: float = 2.0) -> bool:
        """Check if agent shows addiction pattern.

        Args:
            threshold: Addiction index threshold

        Returns:
            True if addiction index exceeds threshold
        """
        return self.addiction_index() > threshold

    def reset(self):
        """Reset to initial state."""
        self.sensitization = 1.0
        self.tolerance = 0.0
        self.withdrawal_level = 0.0
        self.time_since_last_use = 0
        self.total_consumption = 0.0
        self.consumption_history.clear()

    def get_state(self) -> Dict:
        """Return current state for logging."""
        return {
            'sensitization': self.sensitization,
            'tolerance': self.tolerance,
            'withdrawal': self.withdrawal_level,
            'addiction_index': self.addiction_index(),
            'craving': self.craving_level(),
            'time_since_use': self.time_since_last_use,
            'total_consumption': self.total_consumption
        }


# Example usage demonstrating addiction dynamics
if __name__ == "__main__":
    module = WantingLikingModule()

    print("=== Addiction Dynamics Demo ===\n")

    # Simulate repeated consumption
    for episode in range(20):
        if episode < 15:
            # Regular use
            result = module.consume(dose=1.0, timestamp=episode)
            print(f"Ep {episode:2d} CONSUME: "
                  f"Wanting={result['wanting']:.2f}, "
                  f"Liking={result['liking']:.2f}, "
                  f"Addiction={result['addiction_index']:.2f}")
        else:
            # Abstinence
            module.abstain(steps=10)
            state = module.get_state()
            print(f"Ep {episode:2d} ABSTAIN: "
                  f"Withdrawal={state['withdrawal']:.2f}, "
                  f"Craving={state['craving']:.2f}, "
                  f"Addiction={state['addiction_index']:.2f}")

    print(f"\nFinal state: {module.get_state()}")
    print(f"Is addicted: {module.is_addicted()}")
