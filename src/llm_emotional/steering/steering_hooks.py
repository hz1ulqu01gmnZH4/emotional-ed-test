"""
Steering Hooks - PyTorch forward hooks for activation modification.

These hooks intercept hidden states during forward pass and add
emotional steering vectors to modify the model's behavior.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn

from .direction_bank import EmotionalDirectionBank


class SteeringHookError(Exception):
    """Raised when steering hook encounters an error. NO FALLBACK."""
    pass


class ActivationSteeringHook:
    """
    Forward hook that modifies layer activations based on emotional state.

    Usage:
        hook = ActivationSteeringHook(direction_bank, layer_idx=5)
        handle = layer.register_forward_hook(hook)

        # Set emotional state before inference
        hook.emotional_state = {'fear': 0.7, 'curiosity': 0.3}

        # Run inference - activations will be modified
        output = model(input)

        # Cleanup
        handle.remove()
    """

    def __init__(
        self,
        direction_bank: EmotionalDirectionBank,
        layer_idx: int,
        scale: float = 1.0,
    ):
        """
        Initialize steering hook.

        Args:
            direction_bank: Bank containing emotional direction vectors
            layer_idx: Which layer this hook is attached to
            scale: Global scaling factor for steering magnitude
        """
        self.direction_bank = direction_bank
        self.layer_idx = layer_idx
        self.scale = scale

        # Current emotional state (set before inference)
        self.emotional_state: Dict[str, float] = {}

        # Enable/disable steering
        self.enabled = True

        # Statistics for debugging
        self.call_count = 0
        self.last_steering_norm: Optional[float] = None

    def __call__(
        self,
        module: nn.Module,
        input: Tuple[Tensor, ...],
        output: Union[Tensor, Tuple[Tensor, ...]]
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Forward hook called after layer computes output.

        Adds steering vector to hidden states.

        Args:
            module: The layer module (unused but required by hook signature)
            input: Layer input tensors
            output: Layer output (modified in place conceptually)

        Returns:
            Modified output with steering applied
        """
        self.call_count += 1

        # Skip if disabled or no emotional state
        if not self.enabled or not self.emotional_state:
            return output

        # Extract hidden states from output
        # Most transformer layers return (hidden_states, ...) tuple
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Compute steering vector
        steering = self.direction_bank.get_combined_steering(
            self.emotional_state,
            self.layer_idx
        )

        # Apply scaling
        steering = steering * self.scale

        # Track for debugging
        self.last_steering_norm = steering.norm().item()

        # Add steering to hidden states
        # Shape: hidden_states is [batch, seq_len, hidden_dim]
        # steering is [hidden_dim]
        # Broadcasting adds steering to all positions
        device = hidden_states.device
        dtype = hidden_states.dtype
        steering = steering.to(device=device, dtype=dtype)

        modified_hidden = hidden_states + steering

        # Reconstruct output
        if rest is not None:
            return (modified_hidden,) + rest
        else:
            return modified_hidden

    def set_emotional_state(self, **emotions: float) -> None:
        """
        Set emotional state for steering.

        Args:
            **emotions: Emotion intensities (e.g., fear=0.7, joy=0.3)
        """
        self.emotional_state.update(emotions)

    def clear_emotional_state(self) -> None:
        """Clear all emotional state (no steering)."""
        self.emotional_state = {}

    def __repr__(self) -> str:
        state_str = ", ".join(
            f"{e}={v:.2f}"
            for e, v in self.emotional_state.items()
            if v != 0
        )
        return (
            f"ActivationSteeringHook("
            f"layer={self.layer_idx}, "
            f"enabled={self.enabled}, "
            f"state=[{state_str}], "
            f"calls={self.call_count})"
        )


class MultiLayerSteeringManager:
    """
    Manages steering hooks across multiple layers.

    Provides convenient API for installing/removing hooks and
    synchronizing emotional state across all layers.
    """

    def __init__(
        self,
        direction_bank: EmotionalDirectionBank,
        layers: Optional[list] = None,
        scale: float = 1.0,
    ):
        """
        Initialize steering manager.

        Args:
            direction_bank: Shared direction bank for all hooks
            layers: Optional list of layer indices to steer (default: all)
            scale: Global steering magnitude scale
        """
        self.direction_bank = direction_bank
        self.scale = scale
        self.target_layers = layers  # None means all layers

        # Hooks and handles (populated by install())
        self.hooks: list[ActivationSteeringHook] = []
        self.handles: list = []

        # Current emotional state
        self.emotional_state: Dict[str, float] = {
            'fear': 0.0,
            'curiosity': 0.0,
            'anger': 0.0,
            'joy': 0.0,
        }

    def install(self, model: nn.Module) -> None:
        """
        Install steering hooks on model layers.

        Args:
            model: The transformer model to steer

        Raises:
            SteeringHookError: If model structure not recognized
        """
        # Get transformer layers
        layers = self._get_layers(model)

        if not layers:
            raise SteeringHookError(
                "Could not find transformer layers in model. "
                "Model must have .model.layers, .transformer.h, or similar."
            )

        # Determine which layers to steer
        if self.target_layers is not None:
            layer_indices = self.target_layers
        else:
            layer_indices = list(range(len(layers)))

        # Validate layer indices
        for idx in layer_indices:
            if not 0 <= idx < len(layers):
                raise SteeringHookError(
                    f"Layer index {idx} out of range [0, {len(layers)})"
                )

        # Install hooks
        for layer_idx in layer_indices:
            hook = ActivationSteeringHook(
                self.direction_bank,
                layer_idx,
                scale=self.scale,
            )
            handle = layers[layer_idx].register_forward_hook(hook)
            self.hooks.append(hook)
            self.handles.append(handle)

    def _get_layers(self, model: nn.Module) -> list:
        """
        Extract transformer layers from model.

        Handles different model architectures.
        """
        # Qwen3 / Qwen2.5 / Llama style
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return list(model.model.layers)

        # GPT-2 style
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return list(model.transformer.h)

        # BERT style
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            return list(model.encoder.layer)

        # Direct layers attribute
        if hasattr(model, 'layers'):
            return list(model.layers)

        return []

    def uninstall(self) -> None:
        """Remove all steering hooks from model."""
        for handle in self.handles:
            handle.remove()
        self.hooks = []
        self.handles = []

    def set_emotional_state(self, **emotions: float) -> None:
        """
        Set emotional state across all hooks.

        Args:
            **emotions: Emotion intensities (e.g., fear=0.7, joy=0.3)
        """
        self.emotional_state.update(emotions)
        for hook in self.hooks:
            hook.emotional_state = self.emotional_state.copy()

    def clear_emotional_state(self) -> None:
        """Clear emotional state on all hooks."""
        self.emotional_state = {e: 0.0 for e in self.emotional_state}
        for hook in self.hooks:
            hook.clear_emotional_state()

    def enable(self) -> None:
        """Enable steering on all hooks."""
        for hook in self.hooks:
            hook.enabled = True

    def disable(self) -> None:
        """Disable steering on all hooks (passthrough mode)."""
        for hook in self.hooks:
            hook.enabled = False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all hooks."""
        return {
            'n_hooks': len(self.hooks),
            'emotional_state': self.emotional_state.copy(),
            'call_counts': [h.call_count for h in self.hooks],
            'last_steering_norms': [h.last_steering_norm for h in self.hooks],
        }

    def __repr__(self) -> str:
        state_str = ", ".join(
            f"{e}={v:.2f}"
            for e, v in self.emotional_state.items()
            if v != 0
        )
        return (
            f"MultiLayerSteeringManager("
            f"n_hooks={len(self.hooks)}, "
            f"state=[{state_str}])"
        )
