"""
Emotional Encoder for Adapter Gating.

More sophisticated than prefix version - tracks multi-turn dynamics
and integrates external signals.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .state import EmotionalState


class EmotionalEncoderForAdapter(nn.Module):
    """
    Emotional encoder that tracks conversation state.

    Computes emotional state from:
    1. Hidden representations (internal)
    2. External feedback signals (external)
    3. Temporal dynamics via GRU (tonic state)
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        emotion_dim: int = 6,
        signal_dim: int = 5,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize encoder.

        Args:
            hidden_dim: Hidden dimension of LLM
            emotion_dim: Output emotion dimension
            signal_dim: Dimension of external signals
            device: Device to use
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emotion_dim = emotion_dim
        self.signal_dim = signal_dim
        self.device = device or torch.device("cpu")

        # Encode current turn hidden states to emotional features
        self.hidden_to_emotion = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emotion_dim),
            nn.Sigmoid()
        )

        # Temporal smoothing (tonic state) via GRU
        self.emotion_rnn = nn.GRU(
            input_size=emotion_dim,
            hidden_size=emotion_dim,
            batch_first=True
        )

        # External signal integration
        self.signal_encoder = nn.Sequential(
            nn.Linear(signal_dim, 32),
            nn.ReLU(),
            nn.Linear(32, emotion_dim),
        )

        # Combine internal and external emotions
        self.fusion = nn.Sequential(
            nn.Linear(emotion_dim * 2, emotion_dim),
            nn.Sigmoid()
        )

        # Tonic state tracking
        self._tonic_state: Optional[torch.Tensor] = None

        # Weights for phasic vs tonic
        self.phasic_weight = 0.7
        self.tonic_weight = 0.3

        self.to(self.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        external_signals: Optional[torch.Tensor] = None,
        update_tonic: bool = True,
    ) -> torch.Tensor:
        """
        Compute emotional state from hidden representations.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] - LLM hidden states
            external_signals: [batch, signal_dim] - external feedback signals
            update_tonic: Whether to update tonic state

        Returns:
            emotional_state: [batch, emotion_dim]
        """
        batch_size = hidden_states.size(0)

        # Pool hidden states (mean pooling)
        pooled = hidden_states.mean(dim=1)  # [batch, hidden_dim]

        # Compute phasic emotions from current context
        phasic_emotion = self.hidden_to_emotion(pooled)  # [batch, emotion_dim]

        if update_tonic:
            # Initialize tonic state if needed
            if self._tonic_state is None:
                self._tonic_state = torch.zeros(
                    1, batch_size, self.emotion_dim,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )

            # Update tonic state with GRU
            phasic_seq = phasic_emotion.unsqueeze(1)  # [batch, 1, emotion_dim]
            _, self._tonic_state = self.emotion_rnn(phasic_seq, self._tonic_state)

        # Get tonic emotion
        if self._tonic_state is not None:
            tonic_emotion = self._tonic_state.squeeze(0)  # [batch, emotion_dim]
        else:
            tonic_emotion = torch.zeros_like(phasic_emotion)

        # Combine phasic and tonic
        internal_emotion = (
            self.phasic_weight * phasic_emotion +
            self.tonic_weight * tonic_emotion
        )

        # Integrate external signals if provided
        if external_signals is not None:
            external_emotion = torch.sigmoid(self.signal_encoder(external_signals))
            combined = torch.cat([internal_emotion, external_emotion], dim=-1)
            emotional_state = self.fusion(combined)
        else:
            emotional_state = internal_emotion

        return emotional_state

    def forward_from_state(
        self,
        state: EmotionalState,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convenience method to convert EmotionalState to tensor.

        Args:
            state: EmotionalState object
            device: Device for tensor

        Returns:
            emotional_state: [1, emotion_dim] tensor
        """
        d = device or self.device
        return state.to_tensor(d).unsqueeze(0)

    def reset_tonic(self) -> None:
        """Reset tonic state for new conversation."""
        self._tonic_state = None

    def get_tonic_state(self) -> Optional[torch.Tensor]:
        """Get current tonic emotional state."""
        return self._tonic_state

    def set_tonic_state(self, state: torch.Tensor) -> None:
        """Set tonic emotional state (for resuming conversations)."""
        self._tonic_state = state


class SimpleEmotionalEncoder(nn.Module):
    """
    Simplified emotional encoder that just uses external signals.

    Useful when you want to directly control emotional state
    rather than deriving it from hidden states.
    """

    def __init__(
        self,
        emotion_dim: int = 6,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.emotion_dim = emotion_dim
        self.device = device or torch.device("cpu")

        # Tonic state
        self._tonic = torch.zeros(emotion_dim, device=self.device)
        self.decay = 0.9

    def forward(
        self,
        state: EmotionalState,
        update_tonic: bool = True,
    ) -> torch.Tensor:
        """
        Convert EmotionalState to tensor with tonic integration.

        Args:
            state: EmotionalState object
            update_tonic: Whether to update tonic state

        Returns:
            emotional_state: [1, emotion_dim] tensor
        """
        phasic = state.to_tensor(self.device)

        # Ensure tonic is on same device
        if self._tonic.device != self.device:
            self._tonic = self._tonic.to(self.device)

        if update_tonic:
            # Update tonic with decay
            self._tonic = self.decay * self._tonic + (1 - self.decay) * phasic

        # Combine phasic and tonic
        combined = 0.7 * phasic + 0.3 * self._tonic

        return combined.unsqueeze(0)

    def reset_tonic(self) -> None:
        """Reset tonic state."""
        self._tonic = torch.zeros(self.emotion_dim, device=self.device)
