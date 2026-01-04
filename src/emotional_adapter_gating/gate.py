"""
Emotional Gate module for computing gating values.

Different emotions can have different effects:
- Fear: reduces adapter contribution (conservative)
- Curiosity: increases adapter contribution (exploratory)
- Anger: modulates specific dimensions (persistence)
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class EmotionalGate(nn.Module):
    """
    Computes gating values from emotional state.

    Supports three gate types:
    - scalar: Single gate value per layer
    - vector: Per-dimension gating
    - attention: Emotion-conditioned attention over hidden dims
    """

    GATE_TYPES = ["scalar", "vector", "attention"]

    def __init__(
        self,
        emotion_dim: int = 6,
        hidden_dim: int = 768,
        gate_type: str = "scalar",
        device: Optional[torch.device] = None,
    ):
        """
        Initialize emotional gate.

        Args:
            emotion_dim: Dimension of emotional state vector
            hidden_dim: Hidden dimension of the LLM
            gate_type: One of "scalar", "vector", "attention"
            device: Device to use
        """
        super().__init__()

        if gate_type not in self.GATE_TYPES:
            raise ValueError(f"gate_type must be one of {self.GATE_TYPES}")

        self.gate_type = gate_type
        self.hidden_dim = hidden_dim
        self.emotion_dim = emotion_dim
        self.device = device or torch.device("cpu")

        if gate_type == "scalar":
            # Single scalar gate per layer
            self.gate_net = nn.Sequential(
                nn.Linear(emotion_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        elif gate_type == "vector":
            # Per-dimension gating
            self.gate_net = nn.Sequential(
                nn.Linear(emotion_dim, 64),
                nn.ReLU(),
                nn.Linear(64, hidden_dim),
                nn.Sigmoid()
            )
        elif gate_type == "attention":
            # Emotion-conditioned attention over hidden dims
            self.emotion_query = nn.Linear(emotion_dim, 64)
            self.hidden_key = nn.Linear(hidden_dim, 64)
            self.gate_proj = nn.Linear(64, 1)

        self.to(self.device)

    def forward(
        self,
        emotional_state: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gate values.

        Args:
            emotional_state: [batch, emotion_dim]
            hidden: [batch, seq_len, hidden_dim] (for attention gate)

        Returns:
            gate: shape depends on gate_type
                - scalar: [batch, 1]
                - vector: [batch, hidden_dim]
                - attention: [batch, seq_len, 1]
        """
        if self.gate_type == "scalar":
            return self.gate_net(emotional_state)

        elif self.gate_type == "vector":
            return self.gate_net(emotional_state)

        elif self.gate_type == "attention":
            if hidden is None:
                raise ValueError("hidden required for attention gate")

            # Emotion-conditioned attention
            q = self.emotion_query(emotional_state)  # [batch, 64]
            k = self.hidden_key(hidden)  # [batch, seq_len, 64]

            # Compute gate based on emotion-hidden interaction
            q_expanded = q.unsqueeze(1)  # [batch, 1, 64]
            interaction = q_expanded * k  # [batch, seq_len, 64]
            gate = torch.sigmoid(self.gate_proj(interaction))  # [batch, seq_len, 1]

            return gate

    def get_gate_info(self) -> dict:
        """Get information about gate configuration."""
        return {
            "type": self.gate_type,
            "emotion_dim": self.emotion_dim,
            "hidden_dim": self.hidden_dim,
        }
