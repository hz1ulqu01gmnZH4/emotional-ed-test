"""
Error Diffusion Steering Hooks (Version 3).

Implements error diffusion mechanism inspired by Floyd-Steinberg dithering:
- When steering at one layer creates residual error, diffuse to neighboring layers
- Temporal error accumulation across tokens with leaky integration
- Attractor-based steering that measures deviation from target state

Key insight from research: Emotions function as "somatic markers" (Damasio) -
they provide continuous error signals for decision-making. V3 implements this
by treating steering as a control system with feedback.
"""

from typing import Optional, Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ErrorState:
    """Tracks accumulated error for diffusion."""
    temporal_error: Tensor  # [hidden_dim] - accumulated across tokens
    layer_residuals: dict[int, Tensor] = field(default_factory=dict)  # [layer_idx -> hidden_dim]
    token_count: int = 0

    def reset(self):
        """Reset error state for new sequence."""
        self.temporal_error.zero_()
        self.layer_residuals.clear()
        self.token_count = 0


class ErrorDiffusionSteeringHook:
    """
    Forward hook with error diffusion for emotional steering.

    Unlike V2 which applies fixed steering, V3 hooks:
    1. Measure "error" between current hidden state and target attractor
    2. Apply steering + diffused error from previous layer
    3. Compute residual and pass to next layer

    This implements a feedback control system where emotional steering
    self-corrects based on how well it's achieving the target state.
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_dim: int,
        n_layers: int,
        layer_weight: float = 1.0,
        diffusion_rate: float = 0.25,  # Floyd-Steinberg uses 7/16 ≈ 0.44, we use less
        temporal_decay: float = 0.9,  # Leaky integrator decay
        max_steering_norm: float = 2.0,  # Clip steering to prevent runaway
    ):
        """
        Initialize error diffusion steering hook.

        Args:
            layer_idx: Index of this layer
            hidden_dim: Hidden state dimension
            n_layers: Total number of layers (for diffusion matrix)
            layer_weight: Base importance weight for this layer
            diffusion_rate: How much error to diffuse to next layer (0-1)
            temporal_decay: Decay rate for temporal error accumulation
            max_steering_norm: Maximum norm for applied steering
        """
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.layer_weight = layer_weight
        self.diffusion_rate = diffusion_rate
        self.temporal_decay = temporal_decay
        self.max_steering_norm = max_steering_norm

        # Steering vector (set by manager)
        self.steering_vector: Optional[Tensor] = None

        # Target attractor for error computation (set by manager)
        self.target_attractor: Optional[Tensor] = None

        # Error state shared across layers (set by manager)
        self.error_state: Optional[ErrorState] = None

        self.enabled = True

        # Metrics for analysis
        self.last_error_magnitude: float = 0.0
        self.last_steering_magnitude: float = 0.0
        self.last_diffused_error: float = 0.0

    def __call__(
        self,
        module: nn.Module,
        input: tuple,
        output: tuple,
    ) -> tuple:
        """Apply error-diffusion steering to layer output."""
        if not self.enabled or self.steering_vector is None:
            return output

        # Extract hidden states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Get steering vector on correct device
        steering = self.steering_vector.to(device, dtype)

        # === Step 1: Get diffused error from previous layer ===
        diffused_error = torch.zeros(self.hidden_dim, device=device, dtype=dtype)
        if self.error_state is not None:
            prev_layer = self.layer_idx - 1
            if prev_layer in self.error_state.layer_residuals:
                diffused_error = self.error_state.layer_residuals[prev_layer].to(device, dtype)
                self.last_diffused_error = diffused_error.norm().item()

        # === Step 2: Add temporal error (accumulated across tokens) ===
        temporal_error = torch.zeros(self.hidden_dim, device=device, dtype=dtype)
        if self.error_state is not None:
            temporal_error = self.error_state.temporal_error.to(device, dtype) * self.temporal_decay

        # === Step 3: Compute total steering ===
        # Base steering + diffused error from previous layer + temporal correction
        total_steering = (
            steering * self.layer_weight +
            diffused_error * self.diffusion_rate +
            temporal_error * 0.1  # Small temporal contribution
        )

        # Clip to prevent runaway steering
        steering_norm = total_steering.norm()
        if steering_norm > self.max_steering_norm:
            total_steering = total_steering * (self.max_steering_norm / steering_norm)

        self.last_steering_magnitude = total_steering.norm().item()

        # === Step 4: Apply steering ===
        # Apply to all positions in sequence
        modified_hidden = hidden_states + total_steering.unsqueeze(0).unsqueeze(0)

        # === Step 5: Compute error for diffusion ===
        if self.target_attractor is not None and self.error_state is not None:
            attractor = self.target_attractor.to(device, dtype)

            # Mean hidden state across batch and sequence
            mean_hidden = modified_hidden.mean(dim=(0, 1))

            # Error = how far we are from target attractor
            error = attractor - mean_hidden
            error_magnitude = error.norm().item()
            self.last_error_magnitude = error_magnitude

            # === Step 6: Diffuse error to next layer ===
            # Floyd-Steinberg style: pass fraction of error forward
            self.error_state.layer_residuals[self.layer_idx] = error * self.diffusion_rate

            # === Step 7: Accumulate temporal error ===
            # Leaky integrator: new_error = decay * old_error + (1-decay) * current_error
            self.error_state.temporal_error = (
                self.error_state.temporal_error.to(device, dtype) * self.temporal_decay +
                error * (1 - self.temporal_decay)
            )
            self.error_state.token_count += 1

        # Return modified output
        if isinstance(output, tuple):
            return (modified_hidden,) + output[1:]
        return modified_hidden

    def set_steering(self, vector: Optional[Tensor]):
        """Set the steering vector."""
        self.steering_vector = vector

    def set_attractor(self, attractor: Optional[Tensor]):
        """Set the target attractor for error computation."""
        self.target_attractor = attractor

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def get_metrics(self) -> dict:
        """Get last-computed metrics."""
        return {
            "layer_idx": self.layer_idx,
            "error_magnitude": self.last_error_magnitude,
            "steering_magnitude": self.last_steering_magnitude,
            "diffused_error": self.last_diffused_error,
        }


class ErrorDiffusionManager:
    """
    Manages error-diffusion steering across all layers.

    Implements the core error diffusion algorithm:
    1. Each layer receives diffused error from the previous layer
    2. Temporal errors accumulate across tokens (leaky integrator)
    3. Attractors define target emotional states in activation space

    The diffusion pattern is inspired by Floyd-Steinberg dithering:
    - When steering "overshoots" at one layer, the error propagates forward
    - This creates smoother, more stable emotional transitions
    - Prevents oscillation that can occur with pure feedback control
    """

    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        layer_weights: Optional[dict[int, float]] = None,
        diffusion_rate: float = 0.25,
        temporal_decay: float = 0.9,
        max_steering_norm: float = 2.0,
    ):
        """
        Initialize error diffusion manager.

        Args:
            n_layers: Number of transformer layers
            hidden_dim: Hidden dimension size
            layer_weights: Per-layer importance weights
            diffusion_rate: Error diffusion rate between layers (0-1)
            temporal_decay: Temporal error decay rate (0-1)
            max_steering_norm: Maximum steering norm per layer
        """
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.diffusion_rate = diffusion_rate
        self.temporal_decay = temporal_decay
        self.scale = 1.0

        # Default layer weights (middle layers often most effective)
        if layer_weights is None:
            layer_weights = self._default_layer_weights()
        self.layer_weights = layer_weights

        # Shared error state across all hooks
        self.error_state = ErrorState(
            temporal_error=torch.zeros(hidden_dim),
            layer_residuals={},
            token_count=0,
        )

        # Create hooks for each layer
        self.hooks: dict[int, ErrorDiffusionSteeringHook] = {}
        self._handles: list = []

        for layer_idx in range(n_layers):
            weight = layer_weights.get(layer_idx, 0.0)
            self.hooks[layer_idx] = ErrorDiffusionSteeringHook(
                layer_idx=layer_idx,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                layer_weight=weight,
                diffusion_rate=diffusion_rate,
                temporal_decay=temporal_decay,
                max_steering_norm=max_steering_norm,
            )
            self.hooks[layer_idx].error_state = self.error_state

        # Attractor storage
        self.attractors: dict[str, Tensor] = {}
        self.current_attractor: Optional[Tensor] = None

    def _default_layer_weights(self) -> dict[int, float]:
        """
        Generate default layer weights.

        Research shows middle-to-late layers are most effective for
        emotional steering (Emotion Circuits paper, Wang et al. 2025).
        """
        weights = {}
        for i in range(self.n_layers):
            # Bell curve centered at 2/3 through the model
            center = self.n_layers * 0.67
            spread = self.n_layers * 0.25
            weights[i] = torch.exp(
                torch.tensor(-((i - center) ** 2) / (2 * spread ** 2))
            ).item()

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def install(self, model: nn.Module):
        """Install error diffusion hooks on all model layers."""
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

        Args:
            direction: Steering direction vector [hidden_dim]
        """
        scaled = direction * self.scale
        for hook in self.hooks.values():
            hook.set_steering(scaled)

    def clear_steering(self):
        """Clear all steering."""
        for hook in self.hooks.values():
            hook.set_steering(None)
            hook.set_attractor(None)
        self.reset_error_state()

    def reset_error_state(self):
        """Reset accumulated errors (call between sequences)."""
        self.error_state.reset()

    def set_attractor(self, name: str, attractor: Tensor):
        """
        Store a named attractor.

        Attractors define target points in activation space that
        represent desired emotional states.

        Args:
            name: Attractor name (e.g., "fear", "joy")
            attractor: Target activation vector [hidden_dim]
        """
        self.attractors[name] = attractor

    def activate_attractor(self, name: str):
        """
        Activate a named attractor for error computation.

        Args:
            name: Name of attractor to activate
        """
        if name not in self.attractors:
            raise KeyError(f"Unknown attractor: {name}")

        self.current_attractor = self.attractors[name]
        for hook in self.hooks.values():
            hook.set_attractor(self.current_attractor)

    def deactivate_attractor(self):
        """Deactivate attractor-based error computation."""
        self.current_attractor = None
        for hook in self.hooks.values():
            hook.set_attractor(None)

    def set_layer_weights(self, weights: dict[int, float]):
        """Update layer importance weights."""
        self.layer_weights = weights
        for layer_idx, hook in self.hooks.items():
            hook.layer_weight = weights.get(layer_idx, 0.0)

    def get_all_metrics(self) -> list[dict]:
        """Get metrics from all hooks."""
        return [hook.get_metrics() for hook in self.hooks.values()]

    def get_error_summary(self) -> dict:
        """Get summary of current error state."""
        return {
            "temporal_error_norm": self.error_state.temporal_error.norm().item(),
            "token_count": self.error_state.token_count,
            "active_residuals": len(self.error_state.layer_residuals),
            "mean_layer_error": sum(
                r.norm().item() for r in self.error_state.layer_residuals.values()
            ) / max(1, len(self.error_state.layer_residuals)),
        }

    def enable_all(self):
        for hook in self.hooks.values():
            hook.enable()

    def disable_all(self):
        for hook in self.hooks.values():
            hook.disable()
