"""
Emotional Prefix Generator for creating soft prompt tokens.

The core innovation: prefixes are CONDITIONED on emotional state,
enabling dynamic behavior modulation without changing LLM weights.
"""

import torch
import torch.nn as nn
from typing import Optional


class EmotionalPrefixGenerator(nn.Module):
    """
    Generates soft prefix tokens from emotional state.

    Architecture:
    1. Base prefix embeddings (learned starting point)
    2. Emotional modulation network (maps emotions to adjustments)
    3. Combined output = base + scale * modulation
    """

    def __init__(
        self,
        emotion_dim: int = 4,
        hidden_dim: int = 768,
        prefix_length: int = 10,
        n_layers: int = 12,
        n_heads: int = 12,
        modulation_hidden: int = 128,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize prefix generator.

        Args:
            emotion_dim: Dimension of emotional state vector
            hidden_dim: Hidden dimension of the LLM
            prefix_length: Number of prefix tokens
            n_layers: Number of LLM layers
            n_heads: Number of attention heads in LLM
            modulation_hidden: Hidden dim for modulation network
            device: Device to use
        """
        super().__init__()
        self.emotion_dim = emotion_dim
        self.hidden_dim = hidden_dim
        self.prefix_length = prefix_length
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.device = device or torch.device("cpu")

        # Base prefix embeddings (learned starting point)
        # Shape: [n_layers, 2, prefix_length, hidden_dim]
        # The "2" is for key and value
        self.base_prefix = nn.Parameter(
            torch.randn(n_layers, 2, prefix_length, hidden_dim) * 0.01
        )

        # Emotional modulation network
        # Maps emotion vector to prefix adjustments
        total_size = n_layers * 2 * prefix_length * hidden_dim
        self.emotion_to_prefix = nn.Sequential(
            nn.Linear(emotion_dim, modulation_hidden),
            nn.ReLU(),
            nn.Linear(modulation_hidden, modulation_hidden * 2),
            nn.ReLU(),
            nn.Linear(modulation_hidden * 2, total_size)
        )

        # Learnable scaling factor for emotional modulation
        # Starts small to not disrupt base prefix too much
        self.modulation_scale = nn.Parameter(torch.tensor(0.1))

        # Move to device
        self.to(self.device)

    def forward(self, emotional_state: torch.Tensor) -> torch.Tensor:
        """
        Generate prefix tokens conditioned on emotional state.

        Args:
            emotional_state: [batch, emotion_dim] tensor

        Returns:
            prefix: [batch, n_layers, 2, prefix_length, hidden_dim] tensor
                   The "2" dimension is for (key, value)
        """
        batch_size = emotional_state.size(0)

        # Generate emotional modulation
        modulation = self.emotion_to_prefix(emotional_state)
        modulation = modulation.view(
            batch_size,
            self.n_layers,
            2,
            self.prefix_length,
            self.hidden_dim
        )

        # Combine base prefix with emotional modulation
        # Base provides general prefix behavior
        # Modulation adjusts based on current emotional state
        base_expanded = self.base_prefix.unsqueeze(0).expand(
            batch_size, -1, -1, -1, -1
        )
        prefix = base_expanded + self.modulation_scale * modulation

        return prefix

    def get_past_key_values(
        self,
        emotional_state: torch.Tensor
    ) -> tuple:
        """
        Generate past_key_values for LLM from emotional state.

        This converts the prefix to the format expected by HuggingFace
        models' past_key_values argument.

        Args:
            emotional_state: [batch, emotion_dim] tensor

        Returns:
            past_key_values: Tuple of (key, value) tensors for each layer
                            Each has shape [batch, n_heads, prefix_length, head_dim]
        """
        prefix = self.forward(emotional_state)
        batch_size = prefix.size(0)

        past_key_values = []
        for layer_idx in range(self.n_layers):
            # Get key and value for this layer
            # Shape: [batch, prefix_length, hidden_dim]
            layer_key = prefix[:, layer_idx, 0, :, :]
            layer_value = prefix[:, layer_idx, 1, :, :]

            # Reshape to [batch, n_heads, prefix_length, head_dim]
            layer_key = layer_key.view(
                batch_size, self.prefix_length, self.n_heads, self.head_dim
            ).transpose(1, 2)

            layer_value = layer_value.view(
                batch_size, self.prefix_length, self.n_heads, self.head_dim
            ).transpose(1, 2)

            past_key_values.append((layer_key, layer_value))

        return tuple(past_key_values)

    def get_modulation_strength(self) -> float:
        """Get current modulation scale."""
        return self.modulation_scale.item()

    def count_parameters(self) -> dict:
        """Count trainable parameters."""
        base_params = self.base_prefix.numel()
        modulation_params = sum(
            p.numel() for p in self.emotion_to_prefix.parameters()
        )
        scale_params = 1

        return {
            "base_prefix": base_params,
            "modulation_network": modulation_params,
            "scale": scale_params,
            "total": base_params + modulation_params + scale_params,
        }
