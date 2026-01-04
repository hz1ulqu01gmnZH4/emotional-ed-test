"""
Emotional Adapter module with LoRA-style architecture and emotional gating.

Structure: down_proj → activation → up_proj, gated by emotion
"""

import torch
import torch.nn as nn
from typing import Optional

from .gate import EmotionalGate


class EmotionalAdapter(nn.Module):
    """
    LoRA-style adapter with emotional gating.

    The adapter output is multiplied by a gate that depends on
    emotional state, allowing emotional control over adapter contribution.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        adapter_dim: int = 64,
        emotion_dim: int = 6,
        gate_type: str = "scalar",
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize emotional adapter.

        Args:
            hidden_dim: Hidden dimension of the LLM
            adapter_dim: Bottleneck dimension (rank)
            emotion_dim: Dimension of emotional state
            gate_type: One of "scalar", "vector", "attention"
            dropout: Dropout probability
            device: Device to use
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim
        self.device = device or torch.device("cpu")

        # Adapter layers (trainable)
        self.down_proj = nn.Linear(hidden_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Emotional gate (trainable)
        self.gate = EmotionalGate(
            emotion_dim=emotion_dim,
            hidden_dim=hidden_dim,
            gate_type=gate_type,
            device=device,
        )

        # Initialize up_proj to zero for stable training start
        # This ensures the adapter has no effect initially
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        # Small init for down_proj
        nn.init.normal_(self.down_proj.weight, std=0.02)

        self.to(self.device)

    def forward(
        self,
        hidden: torch.Tensor,
        emotional_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply emotionally-gated adapter.

        Args:
            hidden: [batch, seq_len, hidden_dim]
            emotional_state: [batch, emotion_dim]

        Returns:
            modulated hidden: [batch, seq_len, hidden_dim]
        """
        # Adapter transformation
        adapter_out = self.down_proj(hidden)
        adapter_out = self.activation(adapter_out)
        adapter_out = self.dropout(adapter_out)
        adapter_out = self.up_proj(adapter_out)

        # Compute emotional gate
        gate = self.gate(emotional_state, hidden)

        # Apply gate based on its shape
        if gate.dim() == 2:  # Scalar gate [batch, 1] or vector [batch, hidden]
            if gate.size(-1) == 1:
                # Scalar gate - expand to match adapter_out
                gate = gate.unsqueeze(1)  # [batch, 1, 1]
            else:
                # Vector gate - expand seq dim
                gate = gate.unsqueeze(1)  # [batch, 1, hidden_dim]
        # gate.dim() == 3 means attention gate [batch, seq_len, 1] - no change needed

        # Apply gated adapter as residual
        gated_adapter = gate * adapter_out

        return hidden + gated_adapter

    def get_adapter_norm(self) -> float:
        """Get L2 norm of adapter weights (for monitoring)."""
        down_norm = torch.norm(self.down_proj.weight).item()
        up_norm = torch.norm(self.up_proj.weight).item()
        return (down_norm + up_norm) / 2

    def count_parameters(self) -> dict:
        """Count parameters in adapter."""
        down_params = sum(p.numel() for p in [self.down_proj.weight, self.down_proj.bias])
        up_params = sum(p.numel() for p in [self.up_proj.weight, self.up_proj.bias])
        gate_params = sum(p.numel() for p in self.gate.parameters())

        return {
            "down_proj": down_params,
            "up_proj": up_params,
            "gate": gate_params,
            "total": down_params + up_params + gate_params,
        }
