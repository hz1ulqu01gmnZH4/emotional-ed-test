"""
Layer-Weighted Steering Hooks (Version 2).

Applies different steering strengths to different layers based on
learned importance weights from PCA analysis.
"""

from typing import Optional

import torch
from torch import Tensor, nn


class LayerWeightedSteeringHook:
    """
    Forward hook that applies layer-specific emotional steering.

    Different layers receive different steering strengths based on
    their learned importance weights.
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_dim: int,
        layer_weight: float = 1.0,
    ):
        """
        Initialize layer-weighted steering hook.

        Args:
            layer_idx: Index of the layer this hook is attached to
            hidden_dim: Dimension of hidden states
            layer_weight: Importance weight for this layer (from PCA analysis)
        """
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        self.layer_weight = layer_weight

        # Current steering vector (set by manager)
        self.steering_vector: Optional[Tensor] = None
        self.enabled = True

    def __call__(
        self,
        module: nn.Module,
        input: tuple,
        output: tuple,
    ) -> tuple:
        """Apply steering to layer output."""
        if not self.enabled or self.steering_vector is None:
            return output

        # Handle different output formats
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Apply weighted steering
        steering = self.steering_vector.to(hidden_states.device, hidden_states.dtype)
        weighted_steering = steering * self.layer_weight

        # Add to all positions
        hidden_states = hidden_states + weighted_steering.unsqueeze(0).unsqueeze(0)

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    def set_steering(self, vector: Optional[Tensor]):
        """Set the steering vector for this layer."""
        self.steering_vector = vector

    def disable(self):
        """Disable steering for this layer."""
        self.enabled = False

    def enable(self):
        """Enable steering for this layer."""
        self.enabled = True


class LayerWeightedSteeringManager:
    """
    Manages layer-weighted steering across all layers.

    Uses importance weights from PCA analysis to apply stronger
    steering to layers that have clearer emotional directions.
    """

    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        layer_weights: Optional[dict[int, float]] = None,
    ):
        """
        Initialize the layer-weighted steering manager.

        Args:
            n_layers: Number of layers in the model
            hidden_dim: Hidden dimension size
            layer_weights: Dictionary mapping layer_idx to importance weight
        """
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Default to uniform weights if not provided
        if layer_weights is None:
            layer_weights = {i: 1.0 / n_layers for i in range(n_layers)}

        self.layer_weights = layer_weights
        self.scale = 1.0

        # Create hooks for each layer
        self.hooks: dict[int, LayerWeightedSteeringHook] = {}
        self._handles: list = []

        for layer_idx in range(n_layers):
            weight = layer_weights.get(layer_idx, 0.0)
            self.hooks[layer_idx] = LayerWeightedSteeringHook(
                layer_idx=layer_idx,
                hidden_dim=hidden_dim,
                layer_weight=weight,
            )

    def install(self, model: nn.Module):
        """Install hooks on all model layers."""
        # Find layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        else:
            raise ValueError("Cannot find layers in model architecture")

        for layer_idx, layer in enumerate(layers):
            if layer_idx in self.hooks:
                handle = layer.register_forward_hook(self.hooks[layer_idx])
                self._handles.append(handle)

    def uninstall(self):
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def set_steering(self, direction: Tensor):
        """
        Set steering direction for all layers.

        The direction is scaled by the global scale and per-layer weights.

        Args:
            direction: Steering direction vector [hidden_dim]
        """
        scaled_direction = direction * self.scale

        for layer_idx, hook in self.hooks.items():
            hook.set_steering(scaled_direction)

    def clear_steering(self):
        """Clear steering from all layers."""
        for hook in self.hooks.values():
            hook.set_steering(None)

    def set_layer_weights(self, weights: dict[int, float]):
        """Update layer importance weights."""
        self.layer_weights = weights
        for layer_idx, hook in self.hooks.items():
            hook.layer_weight = weights.get(layer_idx, 0.0)

    def enable_all(self):
        """Enable all hooks."""
        for hook in self.hooks.values():
            hook.enable()

    def disable_all(self):
        """Disable all hooks."""
        for hook in self.hooks.values():
            hook.disable()

    def enable_layers(self, layer_indices: list[int]):
        """Enable only specific layers."""
        for layer_idx, hook in self.hooks.items():
            if layer_idx in layer_indices:
                hook.enable()
            else:
                hook.disable()
