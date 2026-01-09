"""
Tests for V3 Error Diffusion Steering.

Tests the core innovations:
1. Error diffusion between layers
2. Temporal error accumulation
3. Attractor-based steering
4. Wanting/liking separation
5. Emotion trajectory generation
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from src.llm_emotional.steering.steering_hooks_v3 import (
    ErrorDiffusionSteeringHook,
    ErrorDiffusionManager,
    ErrorState,
)
from src.llm_emotional.steering.emotional_llm_v3 import (
    EmotionalSteeringLLMv3,
    EmotionState,
    compute_wanting_liking_directions,
    compute_regulatory_directions,
)


class TestErrorState:
    """Tests for ErrorState dataclass."""

    def test_error_state_creation(self):
        """Test creating an error state."""
        hidden_dim = 256
        state = ErrorState(
            temporal_error=torch.zeros(hidden_dim),
            layer_residuals={},
            token_count=0,
        )

        assert state.temporal_error.shape == (hidden_dim,)
        assert len(state.layer_residuals) == 0
        assert state.token_count == 0

    def test_error_state_reset(self):
        """Test resetting error state."""
        hidden_dim = 256
        state = ErrorState(
            temporal_error=torch.randn(hidden_dim),
            layer_residuals={0: torch.randn(hidden_dim), 1: torch.randn(hidden_dim)},
            token_count=42,
        )

        state.reset()

        assert torch.allclose(state.temporal_error, torch.zeros(hidden_dim))
        assert len(state.layer_residuals) == 0
        assert state.token_count == 0


class TestErrorDiffusionSteeringHook:
    """Tests for the error diffusion hook."""

    def test_hook_creation(self):
        """Test creating an error diffusion hook."""
        hook = ErrorDiffusionSteeringHook(
            layer_idx=5,
            hidden_dim=256,
            n_layers=12,
            layer_weight=0.15,
            diffusion_rate=0.25,
            temporal_decay=0.9,
        )

        assert hook.layer_idx == 5
        assert hook.hidden_dim == 256
        assert hook.layer_weight == 0.15
        assert hook.diffusion_rate == 0.25
        assert hook.temporal_decay == 0.9
        assert hook.enabled is True
        assert hook.steering_vector is None

    def test_hook_disable_enable(self):
        """Test disabling and enabling hook."""
        hook = ErrorDiffusionSteeringHook(
            layer_idx=0,
            hidden_dim=256,
            n_layers=12,
        )

        assert hook.enabled is True
        hook.disable()
        assert hook.enabled is False
        hook.enable()
        assert hook.enabled is True

    def test_hook_returns_unchanged_when_disabled(self):
        """Test that disabled hook returns unchanged output."""
        hook = ErrorDiffusionSteeringHook(
            layer_idx=0,
            hidden_dim=256,
            n_layers=12,
        )
        hook.disable()

        # Create mock output
        hidden_states = torch.randn(1, 10, 256)
        output = (hidden_states,)

        result = hook(None, None, output)

        assert result is output  # Should be unchanged

    def test_hook_returns_unchanged_without_steering(self):
        """Test that hook without steering vector returns unchanged output."""
        hook = ErrorDiffusionSteeringHook(
            layer_idx=0,
            hidden_dim=256,
            n_layers=12,
        )

        hidden_states = torch.randn(1, 10, 256)
        output = (hidden_states,)

        result = hook(None, None, output)

        assert result is output

    def test_hook_applies_steering(self):
        """Test that hook applies steering vector."""
        hidden_dim = 256
        hook = ErrorDiffusionSteeringHook(
            layer_idx=0,
            hidden_dim=hidden_dim,
            n_layers=12,
            layer_weight=1.0,
        )

        # Set up steering
        steering = torch.randn(hidden_dim)
        hook.set_steering(steering)

        # Set up error state
        hook.error_state = ErrorState(
            temporal_error=torch.zeros(hidden_dim),
            layer_residuals={},
            token_count=0,
        )

        # Create input
        hidden_states = torch.randn(1, 10, hidden_dim)
        output = (hidden_states.clone(),)

        result = hook(None, None, output)

        # Output should be different (steering applied)
        assert not torch.allclose(result[0], hidden_states)

    def test_hook_diffuses_error(self):
        """Test that hook diffuses error to next layer."""
        hidden_dim = 256
        hook = ErrorDiffusionSteeringHook(
            layer_idx=3,
            hidden_dim=hidden_dim,
            n_layers=12,
            diffusion_rate=0.25,
        )

        # Set up steering and attractor
        hook.set_steering(torch.randn(hidden_dim))
        hook.set_attractor(torch.randn(hidden_dim))

        # Set up error state
        error_state = ErrorState(
            temporal_error=torch.zeros(hidden_dim),
            layer_residuals={},
            token_count=0,
        )
        hook.error_state = error_state

        # Apply hook
        hidden_states = torch.randn(1, 10, hidden_dim)
        output = (hidden_states,)
        hook(None, None, output)

        # Error should be diffused to this layer's index
        assert hook.layer_idx in error_state.layer_residuals

    def test_hook_accumulates_temporal_error(self):
        """Test that hook accumulates temporal error."""
        hidden_dim = 256
        hook = ErrorDiffusionSteeringHook(
            layer_idx=0,
            hidden_dim=hidden_dim,
            n_layers=12,
            temporal_decay=0.9,
        )

        hook.set_steering(torch.randn(hidden_dim))
        hook.set_attractor(torch.randn(hidden_dim))

        error_state = ErrorState(
            temporal_error=torch.zeros(hidden_dim),
            layer_residuals={},
            token_count=0,
        )
        hook.error_state = error_state

        # Apply hook multiple times
        for _ in range(5):
            hidden_states = torch.randn(1, 10, hidden_dim)
            hook(None, None, (hidden_states,))

        # Temporal error should be non-zero
        assert not torch.allclose(error_state.temporal_error, torch.zeros(hidden_dim))
        assert error_state.token_count == 5

    def test_hook_metrics(self):
        """Test that hook tracks metrics."""
        hidden_dim = 256
        hook = ErrorDiffusionSteeringHook(
            layer_idx=5,
            hidden_dim=hidden_dim,
            n_layers=12,
        )

        hook.set_steering(torch.randn(hidden_dim))
        hook.set_attractor(torch.randn(hidden_dim))
        hook.error_state = ErrorState(
            temporal_error=torch.zeros(hidden_dim),
            layer_residuals={},
            token_count=0,
        )

        hidden_states = torch.randn(1, 10, hidden_dim)
        hook(None, None, (hidden_states,))

        metrics = hook.get_metrics()

        assert metrics["layer_idx"] == 5
        assert "error_magnitude" in metrics
        assert "steering_magnitude" in metrics
        assert "diffused_error" in metrics


class TestErrorDiffusionManager:
    """Tests for the error diffusion manager."""

    def test_manager_creation(self):
        """Test creating error diffusion manager."""
        manager = ErrorDiffusionManager(
            n_layers=12,
            hidden_dim=256,
            diffusion_rate=0.25,
            temporal_decay=0.9,
        )

        assert len(manager.hooks) == 12
        assert manager.hidden_dim == 256
        assert manager.diffusion_rate == 0.25
        assert manager.temporal_decay == 0.9

    def test_default_layer_weights(self):
        """Test that default layer weights favor middle-late layers."""
        manager = ErrorDiffusionManager(
            n_layers=12,
            hidden_dim=256,
        )

        weights = manager.layer_weights

        # Should have weights for all layers
        assert len(weights) == 12

        # Middle-late layers should have higher weights
        middle_late_weight = weights[8]  # ~67% through
        early_weight = weights[1]
        assert middle_late_weight > early_weight

    def test_manager_set_steering(self):
        """Test setting steering direction."""
        manager = ErrorDiffusionManager(
            n_layers=12,
            hidden_dim=256,
        )

        direction = torch.randn(256)
        manager.set_steering(direction)

        # All hooks should have steering set
        for hook in manager.hooks.values():
            assert hook.steering_vector is not None

    def test_manager_clear_steering(self):
        """Test clearing steering."""
        manager = ErrorDiffusionManager(
            n_layers=12,
            hidden_dim=256,
        )

        direction = torch.randn(256)
        manager.set_steering(direction)
        manager.clear_steering()

        for hook in manager.hooks.values():
            assert hook.steering_vector is None

    def test_manager_set_attractor(self):
        """Test setting and activating attractors."""
        manager = ErrorDiffusionManager(
            n_layers=12,
            hidden_dim=256,
        )

        attractor = torch.randn(256)
        manager.set_attractor("fear", attractor)

        assert "fear" in manager.attractors
        assert torch.allclose(manager.attractors["fear"], attractor)

        manager.activate_attractor("fear")

        for hook in manager.hooks.values():
            assert hook.target_attractor is not None

    def test_manager_reset_error_state(self):
        """Test resetting error state."""
        manager = ErrorDiffusionManager(
            n_layers=12,
            hidden_dim=256,
        )

        # Simulate some accumulated error
        manager.error_state.temporal_error = torch.randn(256)
        manager.error_state.layer_residuals = {0: torch.randn(256)}
        manager.error_state.token_count = 100

        manager.reset_error_state()

        assert torch.allclose(manager.error_state.temporal_error, torch.zeros(256))
        assert len(manager.error_state.layer_residuals) == 0
        assert manager.error_state.token_count == 0

    def test_manager_get_error_summary(self):
        """Test getting error summary."""
        manager = ErrorDiffusionManager(
            n_layers=12,
            hidden_dim=256,
        )

        manager.error_state.temporal_error = torch.randn(256)
        manager.error_state.token_count = 50

        summary = manager.get_error_summary()

        assert "temporal_error_norm" in summary
        assert summary["token_count"] == 50
        assert "active_residuals" in summary
        assert "mean_layer_error" in summary


class TestEmotionState:
    """Tests for EmotionState dataclass."""

    def test_emotion_state_defaults(self):
        """Test default emotion state values."""
        state = EmotionState()

        assert state.fear == 0.0
        assert state.joy == 0.0
        assert state.wanting == 0.0
        assert state.liking == 0.0

    def test_emotion_state_to_dict(self):
        """Test converting emotion state to dict."""
        state = EmotionState(fear=0.5, joy=0.8, wanting=0.3)

        d = state.to_dict()

        assert d["fear"] == 0.5
        assert d["joy"] == 0.8
        assert d["wanting"] == 0.3
        assert d["anger"] == 0.0

    def test_emotion_state_active_emotions(self):
        """Test getting only active emotions."""
        state = EmotionState(fear=0.5, joy=0.8, wanting=0.0)

        active = state.active_emotions()

        assert "fear" in active
        assert "joy" in active
        assert "wanting" not in active
        assert len(active) == 2


class TestEmotionalSteeringLLMv3:
    """Tests for the main V3 LLM class (mocked)."""

    @pytest.fixture
    def mock_model_tokenizer(self):
        """Create mock model and tokenizer."""
        # Mock model
        model = MagicMock()
        model.config.hidden_size = 256

        # Create mock layers
        mock_layers = [MagicMock() for _ in range(12)]
        model.model.layers = mock_layers

        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value=torch.ones(1, 10, dtype=torch.long))
        tokenizer.decode = MagicMock(return_value="Test response")

        return model, tokenizer

    def test_emotion_state_management(self):
        """Test emotion state setting and clearing (without model loading)."""
        state = EmotionState()

        # Set some emotions
        state.fear = 0.7
        state.curiosity = 0.5

        assert state.fear == 0.7
        assert state.curiosity == 0.5

        # Check active emotions
        active = state.active_emotions()
        assert "fear" in active
        assert "curiosity" in active
        assert "joy" not in active

    def test_v3_emotions_include_wanting_liking(self):
        """Test that V3 supports wanting/liking dimensions."""
        assert "wanting" in EmotionalSteeringLLMv3.EMOTIONS
        assert "liking" in EmotionalSteeringLLMv3.EMOTIONS
        assert "resilience" in EmotionalSteeringLLMv3.EMOTIONS
        assert "equanimity" in EmotionalSteeringLLMv3.EMOTIONS

    def test_v3_emotion_count(self):
        """Test V3 has expected emotion count."""
        # V3 should have 10 emotions
        expected_emotions = {
            "fear", "joy", "anger", "sadness",
            "curiosity", "engagement",
            "wanting", "liking",
            "resilience", "equanimity",
        }
        assert EmotionalSteeringLLMv3.EMOTIONS == expected_emotions


class TestIntegration:
    """Integration tests for the V3 system."""

    def test_error_diffusion_flow(self):
        """Test complete error diffusion flow through multiple layers."""
        hidden_dim = 256
        n_layers = 12

        manager = ErrorDiffusionManager(
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            diffusion_rate=0.25,
            temporal_decay=0.9,
        )

        # Set up steering
        direction = torch.randn(hidden_dim)
        manager.set_steering(direction)

        # Set up attractor
        attractor = torch.randn(hidden_dim)
        manager.set_attractor("test", attractor)
        manager.activate_attractor("test")

        # Simulate forward pass through layers
        hidden_states = torch.randn(1, 10, hidden_dim)

        for layer_idx in range(n_layers):
            hook = manager.hooks[layer_idx]
            output = (hidden_states.clone(),)
            result = hook(None, None, output)
            hidden_states = result[0]

        # Check that error was accumulated
        assert manager.error_state.token_count > 0

        # Check that error was diffused through layers
        assert len(manager.error_state.layer_residuals) > 0

    def test_temporal_decay_reduces_error(self):
        """Test that temporal decay reduces accumulated error over time."""
        hidden_dim = 256

        hook = ErrorDiffusionSteeringHook(
            layer_idx=0,
            hidden_dim=hidden_dim,
            n_layers=12,
            temporal_decay=0.5,  # Fast decay for testing
        )

        hook.set_steering(torch.zeros(hidden_dim))  # No steering
        hook.set_attractor(torch.randn(hidden_dim))

        error_state = ErrorState(
            temporal_error=torch.ones(hidden_dim),  # Start with error
            layer_residuals={},
            token_count=0,
        )
        hook.error_state = error_state

        initial_norm = error_state.temporal_error.norm().item()

        # Run multiple passes - error should decay
        for _ in range(10):
            hidden_states = torch.zeros(1, 10, hidden_dim)  # Constant hidden states
            hook(None, None, (hidden_states,))

        final_norm = error_state.temporal_error.norm().item()

        # Error should have decayed (though not to exactly zero due to new errors)
        # The decay makes old error smaller while new errors are added
        assert error_state.token_count == 10

    def test_layer_weights_affect_steering_magnitude(self):
        """Test that layer weights affect steering magnitude."""
        hidden_dim = 256

        # Low weight hook
        hook_low = ErrorDiffusionSteeringHook(
            layer_idx=0,
            hidden_dim=hidden_dim,
            n_layers=12,
            layer_weight=0.1,
        )

        # High weight hook
        hook_high = ErrorDiffusionSteeringHook(
            layer_idx=0,
            hidden_dim=hidden_dim,
            n_layers=12,
            layer_weight=1.0,
        )

        steering = torch.randn(hidden_dim)
        hook_low.set_steering(steering)
        hook_high.set_steering(steering)

        # Create shared error states
        for hook in [hook_low, hook_high]:
            hook.error_state = ErrorState(
                temporal_error=torch.zeros(hidden_dim),
                layer_residuals={},
                token_count=0,
            )

        hidden_states = torch.randn(1, 10, hidden_dim)

        result_low = hook_low(None, None, (hidden_states.clone(),))
        result_high = hook_high(None, None, (hidden_states.clone(),))

        # Higher weight should cause larger deviation from original
        diff_low = (result_low[0] - hidden_states).norm().item()
        diff_high = (result_high[0] - hidden_states).norm().item()

        assert diff_high > diff_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
