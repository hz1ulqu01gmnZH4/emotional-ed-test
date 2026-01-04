"""
Specialized Fear Module.

Mirrors the FearEDAgent._compute_fear() logic from the RL emotional agents.
"""

import torch
import torch.nn as nn
from typing import Optional


class FearModule(nn.Module):
    """
    Specialized fear module for detecting danger/risk in LLM outputs.

    Similar to FearEDAgent._compute_fear() in the RL implementation.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        fear_decay: float = 0.9,
    ):
        """
        Initialize fear module.

        Args:
            hidden_dim: Dimension of LLM hidden states
            fear_decay: Decay rate for tonic fear
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fear_decay_rate = fear_decay

        # Danger detector network
        self.danger_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Pain/negative reward detector
        self.pain_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Uncertainty detector (epistemic fear)
        self.uncertainty_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Tonic fear state
        self.tonic_fear = 0.0

        # Weights for combining fear components
        self.danger_weight = 0.4
        self.pain_weight = 0.3
        self.uncertainty_weight = 0.3

    def forward(
        self,
        hidden_states: torch.Tensor,
        feedback: Optional[float] = None,
    ) -> float:
        """
        Compute fear level from LLM hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] from LLM
            feedback: Optional recent feedback signal (-1 to 1)

        Returns:
            Fear level (0 to 1)
        """
        # Pool hidden states (mean over sequence and batch)
        if hidden_states.dim() == 3:
            pooled = hidden_states.mean(dim=(0, 1))
        elif hidden_states.dim() == 2:
            pooled = hidden_states.mean(dim=0)
        else:
            pooled = hidden_states

        # Compute phasic fear components
        danger = self.danger_net(pooled).item()
        pain = self.pain_net(pooled).item()
        uncertainty = self.uncertainty_net(pooled).item()

        # Combine phasic fear
        phasic_fear = (
            self.danger_weight * danger +
            self.pain_weight * pain +
            self.uncertainty_weight * uncertainty
        )

        # Update tonic fear from feedback
        if feedback is not None:
            if feedback < -0.5:
                # Strong negative feedback → increase tonic fear
                self.tonic_fear = min(1.0, self.tonic_fear + 0.3)
            elif feedback < -0.2:
                # Mild negative feedback → slight increase
                self.tonic_fear = min(1.0, self.tonic_fear + 0.1)
            elif feedback > 0.3:
                # Positive feedback → reduce tonic fear
                self.tonic_fear = max(0.0, self.tonic_fear - 0.1)

        # Combine phasic and tonic fear
        total_fear = max(phasic_fear, self.tonic_fear)

        # Decay tonic fear
        self.tonic_fear *= self.fear_decay_rate

        return total_fear

    def compute_phasic_only(self, hidden_states: torch.Tensor) -> float:
        """
        Compute only phasic fear without updating tonic state.

        Args:
            hidden_states: LLM hidden states

        Returns:
            Phasic fear level (0 to 1)
        """
        with torch.no_grad():
            if hidden_states.dim() == 3:
                pooled = hidden_states.mean(dim=(0, 1))
            elif hidden_states.dim() == 2:
                pooled = hidden_states.mean(dim=0)
            else:
                pooled = hidden_states

            danger = self.danger_net(pooled).item()
            pain = self.pain_net(pooled).item()
            uncertainty = self.uncertainty_net(pooled).item()

            return (
                self.danger_weight * danger +
                self.pain_weight * pain +
                self.uncertainty_weight * uncertainty
            )

    def get_tonic_fear(self) -> float:
        """Get current tonic fear level."""
        return self.tonic_fear

    def set_tonic_fear(self, value: float) -> None:
        """Set tonic fear level."""
        self.tonic_fear = max(0.0, min(1.0, value))

    def reset(self) -> None:
        """Reset tonic fear to zero."""
        self.tonic_fear = 0.0

    def decay(self) -> None:
        """Manually apply decay to tonic fear."""
        self.tonic_fear *= self.fear_decay_rate

    def set_component_weights(
        self,
        danger: float = 0.4,
        pain: float = 0.3,
        uncertainty: float = 0.3,
    ) -> None:
        """
        Set weights for fear components.

        Args:
            danger: Weight for danger detection
            pain: Weight for pain/negative reward
            uncertainty: Weight for uncertainty
        """
        total = danger + pain + uncertainty
        self.danger_weight = danger / total
        self.pain_weight = pain / total
        self.uncertainty_weight = uncertainty / total

    def get_fear_breakdown(self, hidden_states: torch.Tensor) -> dict:
        """
        Get breakdown of fear components.

        Args:
            hidden_states: LLM hidden states

        Returns:
            Dict with danger, pain, uncertainty, tonic, and total fear
        """
        with torch.no_grad():
            if hidden_states.dim() == 3:
                pooled = hidden_states.mean(dim=(0, 1))
            elif hidden_states.dim() == 2:
                pooled = hidden_states.mean(dim=0)
            else:
                pooled = hidden_states

            danger = self.danger_net(pooled).item()
            pain = self.pain_net(pooled).item()
            uncertainty = self.uncertainty_net(pooled).item()

            phasic = (
                self.danger_weight * danger +
                self.pain_weight * pain +
                self.uncertainty_weight * uncertainty
            )

            return {
                "danger": danger,
                "pain": pain,
                "uncertainty": uncertainty,
                "phasic": phasic,
                "tonic": self.tonic_fear,
                "total": max(phasic, self.tonic_fear),
            }
