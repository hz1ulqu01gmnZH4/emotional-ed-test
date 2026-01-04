"""
Emotional Reward Model - Observes LLM and outputs emotional signals.

Similar to a reward model in RLHF, but outputs emotional dimensions
instead of a single scalar reward.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .signals import EmotionalSignals


class EmotionalRewardModel(nn.Module):
    """
    Emotional Reward Model - Observes LLM hidden states and outputs emotions.

    Trained separately from the LLM (which remains frozen).
    Uses attention pooling over sequence and separate heads for each emotion.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        n_emotions: int = 6,
        dropout: float = 0.1,
    ):
        """
        Initialize Emotional Reward Model.

        Args:
            hidden_dim: Dimension of LLM hidden states
            n_emotions: Number of emotion dimensions
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_emotions = n_emotions

        # Encoder: processes LLM hidden states
        self.hidden_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Attention pooling over sequence
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True,
            dropout=dropout,
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, 128))

        # Individual emotion heads (for interpretability)
        self.emotion_heads = nn.ModuleDict({
            "fear": nn.Linear(128, 1),
            "curiosity": nn.Linear(128, 1),
            "anger": nn.Linear(128, 1),
            "joy": nn.Linear(128, 1),
            "anxiety": nn.Linear(128, 1),
            "confidence": nn.Linear(128, 1),
        })

        # Temporal state (tonic emotions) - GRU for sequence modeling
        self.tonic_state = nn.GRU(
            input_size=n_emotions,
            hidden_size=n_emotions,
            batch_first=True,
        )
        self.tonic_hidden: Optional[torch.Tensor] = None

        # Phasic/tonic mixing ratio
        self.phasic_weight = 0.6
        self.tonic_weight = 0.4

    def forward(
        self,
        hidden_states: torch.Tensor,
        update_tonic: bool = True,
    ) -> Tuple[EmotionalSignals, torch.Tensor]:
        """
        Compute emotional signals from LLM hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] from LLM
            update_tonic: Whether to update tonic state

        Returns:
            Tuple of (EmotionalSignals, raw emotion tensor [batch, n_emotions])
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device

        # Encode hidden states
        encoded = self.hidden_encoder(hidden_states)  # [batch, seq, 128]

        # Attention pooling to single vector
        query = self.pool_query.expand(batch_size, -1, -1).to(device)
        pooled, _ = self.attention_pool(query, encoded, encoded)
        pooled = pooled.squeeze(1)  # [batch, 128]

        # Compute phasic emotions
        phasic_emotions = {}
        for emotion, head in self.emotion_heads.items():
            phasic_emotions[emotion] = torch.sigmoid(head(pooled))

        # Stack phasic emotions in consistent order
        phasic_tensor = torch.cat([
            phasic_emotions["fear"],
            phasic_emotions["curiosity"],
            phasic_emotions["anger"],
            phasic_emotions["joy"],
            phasic_emotions["anxiety"],
            phasic_emotions["confidence"],
        ], dim=-1)  # [batch, n_emotions]

        # Update tonic state
        if update_tonic:
            if self.tonic_hidden is None:
                self.tonic_hidden = torch.zeros(
                    1, batch_size, self.n_emotions,
                    device=device, dtype=phasic_tensor.dtype
                )

            # Ensure tonic_hidden matches batch size
            if self.tonic_hidden.size(1) != batch_size:
                self.tonic_hidden = torch.zeros(
                    1, batch_size, self.n_emotions,
                    device=device, dtype=phasic_tensor.dtype
                )

            # Detach tonic hidden to prevent backward through multiple iterations
            tonic_hidden_detached = self.tonic_hidden.detach()

            phasic_seq = phasic_tensor.unsqueeze(1)  # [batch, 1, n_emotions]
            _, new_tonic_hidden = self.tonic_state(phasic_seq, tonic_hidden_detached)

            # Store detached for next iteration
            self.tonic_hidden = new_tonic_hidden.detach()

            # Combine phasic and tonic (use new_tonic_hidden for gradient)
            tonic = new_tonic_hidden.squeeze(0)  # [batch, n_emotions]
            combined = self.phasic_weight * phasic_tensor + self.tonic_weight * tonic
        else:
            combined = phasic_tensor

        # Return as EmotionalSignals (for batch size 1) and raw tensor
        signals = EmotionalSignals.from_tensor(combined[0])
        return signals, combined

    def reset_tonic(self) -> None:
        """Reset tonic state for new conversation."""
        self.tonic_hidden = None

    def get_phasic_only(self, hidden_states: torch.Tensor) -> EmotionalSignals:
        """Get only phasic emotions without tonic state update."""
        signals, _ = self.forward(hidden_states, update_tonic=False)
        return signals

    def set_tonic_weights(self, phasic: float, tonic: float) -> None:
        """
        Set phasic/tonic mixing weights.

        Args:
            phasic: Weight for phasic emotions (0-1)
            tonic: Weight for tonic emotions (0-1)
        """
        assert abs((phasic + tonic) - 1.0) < 0.01, "Weights should sum to 1"
        self.phasic_weight = phasic
        self.tonic_weight = tonic
