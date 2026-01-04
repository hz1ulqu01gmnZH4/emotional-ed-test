"""
Emotional Encoder for computing emotional state from context.

Maps environmental signals to emotional dimensions, similar to
FearEDAgent._compute_fear() but multi-dimensional.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .context import EmotionalContext


class EmotionalEncoder(nn.Module):
    """
    Computes emotional state from context.

    Maintains both phasic (immediate) and tonic (persistent) emotional states.
    Each emotion has its own encoder for interpretability.
    """

    EMOTIONS = ["fear", "curiosity", "anger", "joy"]

    def __init__(
        self,
        context_dim: int = 10,
        hidden_dim: int = 32,
        emotion_dim: int = 4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.emotion_dim = emotion_dim
        self.device = device or torch.device("cpu")
        self._dtype = dtype or torch.float32

        # Individual emotion encoders (interpretable)
        self.fear_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.curiosity_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.anger_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.joy_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Tonic state (persistent, decays slowly)
        self._tonic_fear = 0.0
        self._tonic_joy = 0.0
        self.fear_decay = 0.9
        self.joy_decay = 0.95

        # Move to device
        self.to(self.device)

    def context_to_tensor(self, ctx: EmotionalContext) -> torch.Tensor:
        """Convert EmotionalContext to tensor."""
        # Get dtype from model parameters
        param_dtype = next(self.parameters()).dtype
        tensor = torch.tensor([
            ctx.last_reward,
            float(ctx.safety_flag),
            ctx.user_satisfaction,
            float(ctx.repeated_query),
            ctx.topic_novelty,
            float(ctx.contradiction_detected),
            min(ctx.cumulative_negative / 5.0, 1.0),  # Normalize
            min(ctx.cumulative_positive / 5.0, 1.0),  # Normalize
            min(ctx.failed_attempts / 5.0, 1.0),      # Normalize
            0.0  # Reserved for future signals
        ], dtype=param_dtype, device=self.device)
        return tensor

    def forward(
        self,
        context: EmotionalContext,
        update_tonic: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute emotional state from context.

        Args:
            context: EmotionalContext with environmental signals
            update_tonic: Whether to update tonic emotional state

        Returns:
            Dictionary with individual emotions and combined vector:
            {
                'fear': Tensor[1],
                'curiosity': Tensor[1],
                'anger': Tensor[1],
                'joy': Tensor[1],
                'combined': Tensor[4]
            }
        """
        ctx_tensor = self.context_to_tensor(context).unsqueeze(0)

        # Phasic emotions (immediate response)
        fear_phasic = self.fear_net(ctx_tensor)
        curiosity = self.curiosity_net(ctx_tensor)
        anger = self.anger_net(ctx_tensor)
        joy_phasic = self.joy_net(ctx_tensor)

        if update_tonic:
            # Update tonic state based on strong signals
            if context.safety_flag or context.last_reward < -0.5:
                self._tonic_fear = min(1.0, self._tonic_fear + 0.3)
            if context.last_reward > 0.5:
                self._tonic_joy = min(1.0, self._tonic_joy + 0.2)

            # Decay tonic emotions
            self._tonic_fear *= self.fear_decay
            self._tonic_joy *= self.joy_decay

        # Combine phasic and tonic (max of each)
        param_dtype = next(self.parameters()).dtype
        tonic_fear_tensor = torch.tensor(
            [[self._tonic_fear]], dtype=param_dtype, device=self.device
        )
        tonic_joy_tensor = torch.tensor(
            [[self._tonic_joy]], dtype=param_dtype, device=self.device
        )

        fear = torch.maximum(fear_phasic, tonic_fear_tensor)
        joy = torch.maximum(joy_phasic, tonic_joy_tensor)

        # Combine all emotions into single vector
        combined = torch.cat([fear, curiosity, anger, joy], dim=-1)

        return {
            'fear': fear.squeeze(-1),
            'curiosity': curiosity.squeeze(-1),
            'anger': anger.squeeze(-1),
            'joy': joy.squeeze(-1),
            'combined': combined.squeeze(0),
        }

    def reset_tonic(self) -> None:
        """Reset tonic state (e.g., at conversation start)."""
        self._tonic_fear = 0.0
        self._tonic_joy = 0.0

    @property
    def tonic_fear(self) -> float:
        """Get current tonic fear level."""
        return self._tonic_fear

    @property
    def tonic_joy(self) -> float:
        """Get current tonic joy level."""
        return self._tonic_joy

    def get_emotional_summary(self) -> Dict[str, float]:
        """Get human-readable summary of tonic state."""
        return {
            "tonic_fear": self._tonic_fear,
            "tonic_joy": self._tonic_joy,
        }
