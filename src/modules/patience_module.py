"""Serotonin-based patience module with proper gamma bounding.

Fixes issue 3.2: γ_effective must be bounded in [γ_min, γ_max) for convergence.

Based on:
- Schweighofer et al. (2008): Low serotonin → higher discounting
- Miyazaki et al. (2018): 5-HT neurons encode patience for delayed rewards
- GPT-5/Gemini review recommendations
"""

import numpy as np
from typing import Optional


class PatienceModule:
    """Serotonin-modulated temporal discounting with proper bounds.

    Key insight: Serotonin modulates patience (discount factor γ).
    - Low 5-HT → Lower γ → Impulsive (discount future heavily)
    - High 5-HT → Higher γ → Patient (value future rewards)

    CRITICAL: γ must stay in [0, 1) for TD convergence.
    We use a saturating sigmoid transform, not linear addition.
    """

    def __init__(
        self,
        gamma_min: float = 0.7,     # Minimum discount (impulsive)
        gamma_max: float = 0.99,    # Maximum discount (very patient)
        serotonin_baseline: float = 1.0,
        recovery_rate: float = 0.05,
        depletion_rate: float = 0.1,
        sigmoid_steepness: float = 2.0
    ):
        """Initialize patience module.

        Args:
            gamma_min: Floor discount factor (when 5-HT depleted)
            gamma_max: Ceiling discount factor (when 5-HT high)
            serotonin_baseline: Starting 5-HT level (normalized)
            recovery_rate: How fast 5-HT recovers after reward
            depletion_rate: How fast 5-HT depletes during frustration
            sigmoid_steepness: Controls transition sharpness
        """
        assert 0.0 < gamma_min < gamma_max < 1.0, \
            f"Require 0 < gamma_min ({gamma_min}) < gamma_max ({gamma_max}) < 1"

        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.serotonin = serotonin_baseline
        self.serotonin_baseline = serotonin_baseline
        self.recovery_rate = recovery_rate
        self.depletion_rate = depletion_rate
        self.sigmoid_steepness = sigmoid_steepness

        # Bounds for serotonin level
        self.serotonin_min = 0.1
        self.serotonin_max = 2.0

    def effective_gamma(self) -> float:
        """Compute patience (discount factor) modulated by serotonin.

        Uses saturating sigmoid to GUARANTEE γ ∈ [γ_min, γ_max).

        Returns:
            Bounded discount factor
        """
        # Normalize serotonin to [-1, 1] range centered at baseline
        normalized = (self.serotonin - self.serotonin_baseline) / self.serotonin_baseline

        # Sigmoid maps (-∞, +∞) → (0, 1)
        sigmoid = 1.0 / (1.0 + np.exp(-self.sigmoid_steepness * normalized))

        # Map [0, 1] → [γ_min, γ_max)
        # Note: We never reach exactly γ_max (asymptotic)
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * sigmoid

        # Extra safety clamp (should never trigger if math is right)
        return float(np.clip(gamma, self.gamma_min, self.gamma_max - 1e-6))

    def update(self, reward: float, was_waiting: bool = False,
               was_frustrated: bool = False):
        """Update serotonin level based on outcomes.

        Args:
            reward: Reward received
            was_waiting: Whether agent was waiting for delayed reward
            was_frustrated: Whether agent experienced blocking/failure
        """
        if reward > 0:
            # Positive outcome → serotonin recovery
            # Especially if reward came after waiting (patience paid off)
            recovery = self.recovery_rate * (1 + 0.5 * float(was_waiting))
            self.serotonin = min(self.serotonin_max,
                                 self.serotonin + recovery)

        elif was_frustrated or (was_waiting and reward <= 0):
            # Frustration or unrewarded waiting → serotonin depletion
            self.serotonin = max(self.serotonin_min,
                                 self.serotonin - self.depletion_rate)

        else:
            # Neutral outcome → slow drift toward baseline
            drift = 0.01 * (self.serotonin_baseline - self.serotonin)
            self.serotonin += drift

    def impulsivity(self) -> float:
        """Current impulsivity level (inverse of patience).

        Returns:
            Impulsivity in [0, 1] where 1 = maximally impulsive
        """
        return max(0.0, 1.0 - self.serotonin / self.serotonin_baseline)

    def reset(self):
        """Reset to baseline state."""
        self.serotonin = self.serotonin_baseline

    def get_state(self) -> dict:
        """Return current state for logging."""
        return {
            'serotonin': self.serotonin,
            'gamma_effective': self.effective_gamma(),
            'impulsivity': self.impulsivity()
        }


class HyperbolicPatienceModule(PatienceModule):
    """Alternative: Hyperbolic discounting modulated by serotonin.

    Human/animal discounting is often hyperbolic, not exponential:
    V(D) = V0 / (1 + k * D)

    Where k is the discount rate (higher k = more impulsive).
    Serotonin modulates k.
    """

    def __init__(
        self,
        k_min: float = 0.01,   # Low discounting (patient)
        k_max: float = 1.0,    # High discounting (impulsive)
        **kwargs
    ):
        super().__init__(**kwargs)
        self.k_min = k_min
        self.k_max = k_max

    def discount_rate(self) -> float:
        """Hyperbolic discount rate k modulated by serotonin.

        Low serotonin → High k → Impulsive
        High serotonin → Low k → Patient
        """
        # Inverse relationship: low 5-HT → high k
        normalized = self.serotonin / self.serotonin_baseline

        # Exponential decay of k with serotonin
        k = self.k_max * np.exp(-2.0 * (normalized - 0.5))

        return float(np.clip(k, self.k_min, self.k_max))

    def discount_value(self, value: float, delay: int) -> float:
        """Discount a future value hyperbolically.

        Args:
            value: Future value
            delay: Steps until value received

        Returns:
            Present value (discounted)
        """
        k = self.discount_rate()
        return value / (1.0 + k * delay)
