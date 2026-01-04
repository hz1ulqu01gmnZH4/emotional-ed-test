"""
Tests for steering hooks.

Tests:
- ActivationSteeringHook modifies outputs correctly
- Hook respects enabled/disabled state
- Hook handles different output formats (tuple vs tensor)
- MultiLayerSteeringManager coordinates hooks
"""

import pytest
import torch
import torch.nn as nn

from src.llm_emotional.steering.direction_bank import EmotionalDirectionBank
from src.llm_emotional.steering.steering_hooks import (
    ActivationSteeringHook,
    MultiLayerSteeringManager,
    SteeringHookError,
)


class DummyLayer(nn.Module):
    """Dummy layer for testing hooks."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # Return tuple like transformer layers
        return (x, None, None)


class DummyModel(nn.Module):
    """Dummy model with multiple layers."""

    def __init__(self, hidden_dim: int, n_layers: int):
        super().__init__()
        self.config = type('Config', (), {
            'hidden_size': hidden_dim,
            'num_hidden_layers': n_layers,
        })()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            DummyLayer(hidden_dim) for _ in range(n_layers)
        ])


class TestActivationSteeringHook:
    """Test individual steering hook."""

    def test_hook_disabled_returns_unchanged(self):
        """Disabled hook should not modify output."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        hook = ActivationSteeringHook(bank, layer_idx=0)
        hook.enabled = False
        hook.emotional_state = {'fear': 0.8}

        input_tensor = torch.randn(2, 10, 64)  # batch, seq, hidden
        output = (input_tensor.clone(), None)

        result = hook(None, None, output)

        assert torch.allclose(result[0], input_tensor)

    def test_hook_empty_state_returns_unchanged(self):
        """Empty emotional state should not modify output."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        hook = ActivationSteeringHook(bank, layer_idx=0)
        hook.emotional_state = {}

        input_tensor = torch.randn(2, 10, 64)
        output = (input_tensor.clone(), None)

        result = hook(None, None, output)

        assert torch.allclose(result[0], input_tensor)

    def test_hook_adds_steering_to_hidden_states(self):
        """Hook should add steering vector to hidden states."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        direction = torch.randn(4, 64)
        bank.set_direction('fear', direction)

        hook = ActivationSteeringHook(bank, layer_idx=1)
        hook.emotional_state = {'fear': 1.0}

        input_tensor = torch.zeros(2, 10, 64)
        output = (input_tensor.clone(), None)

        result = hook(None, None, output)

        # Result should be direction broadcast across batch and sequence
        expected_steering = direction[1]  # layer 1
        assert result[0].shape == (2, 10, 64)
        # All positions should have the steering added
        assert torch.allclose(result[0][0, 0], expected_steering)
        assert torch.allclose(result[0][1, 5], expected_steering)

    def test_hook_handles_tensor_output(self):
        """Hook should handle non-tuple output."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        hook = ActivationSteeringHook(bank, layer_idx=0)
        hook.emotional_state = {'fear': 0.5}

        input_tensor = torch.randn(2, 10, 64)

        result = hook(None, None, input_tensor)

        assert isinstance(result, torch.Tensor)
        assert result.shape == input_tensor.shape

    def test_hook_respects_scale(self):
        """Hook should scale steering by scale factor."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        direction = torch.ones(4, 64)
        bank.set_direction('fear', direction)

        hook = ActivationSteeringHook(bank, layer_idx=0, scale=0.1)
        hook.emotional_state = {'fear': 1.0}

        input_tensor = torch.zeros(1, 1, 64)
        output = (input_tensor.clone(), None)

        result = hook(None, None, output)

        # Steering should be scaled down by 0.1
        assert torch.allclose(result[0][0, 0], torch.ones(64) * 0.1)

    def test_hook_tracks_statistics(self):
        """Hook should track call count and steering norm."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        direction = torch.randn(4, 64)
        bank.set_direction('joy', direction)

        hook = ActivationSteeringHook(bank, layer_idx=2)
        hook.emotional_state = {'joy': 0.5}

        assert hook.call_count == 0
        assert hook.last_steering_norm is None

        output = (torch.randn(1, 5, 64), None)
        hook(None, None, output)

        assert hook.call_count == 1
        assert hook.last_steering_norm is not None
        assert hook.last_steering_norm > 0


class TestMultiLayerSteeringManager:
    """Test multi-layer hook management."""

    def test_install_hooks_on_model(self):
        """Should install hooks on all layers."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        model = DummyModel(hidden_dim=64, n_layers=4)

        manager = MultiLayerSteeringManager(bank)
        manager.install(model)

        assert len(manager.hooks) == 4
        assert len(manager.handles) == 4

    def test_install_specific_layers(self):
        """Should install hooks only on specified layers."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        model = DummyModel(hidden_dim=64, n_layers=4)

        manager = MultiLayerSteeringManager(bank, layers=[0, 2])
        manager.install(model)

        assert len(manager.hooks) == 2
        assert manager.hooks[0].layer_idx == 0
        assert manager.hooks[1].layer_idx == 2

    def test_uninstall_removes_hooks(self):
        """Uninstall should remove all hooks."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        model = DummyModel(hidden_dim=64, n_layers=4)

        manager = MultiLayerSteeringManager(bank)
        manager.install(model)
        manager.uninstall()

        assert len(manager.hooks) == 0
        assert len(manager.handles) == 0

    def test_set_emotional_state_propagates(self):
        """Setting state should propagate to all hooks."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        model = DummyModel(hidden_dim=64, n_layers=4)

        manager = MultiLayerSteeringManager(bank)
        manager.install(model)
        manager.set_emotional_state(fear=0.7, joy=0.3)

        for hook in manager.hooks:
            assert hook.emotional_state == {'fear': 0.7, 'curiosity': 0.0, 'anger': 0.0, 'joy': 0.3}

    def test_clear_emotional_state_clears_all(self):
        """Clearing state should clear all hooks."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        model = DummyModel(hidden_dim=64, n_layers=4)

        manager = MultiLayerSteeringManager(bank)
        manager.install(model)
        manager.set_emotional_state(anger=0.9)
        manager.clear_emotional_state()

        for hook in manager.hooks:
            assert hook.emotional_state == {}

    def test_enable_disable(self):
        """Enable/disable should affect all hooks."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        model = DummyModel(hidden_dim=64, n_layers=4)

        manager = MultiLayerSteeringManager(bank)
        manager.install(model)

        manager.disable()
        assert all(not hook.enabled for hook in manager.hooks)

        manager.enable()
        assert all(hook.enabled for hook in manager.hooks)

    def test_get_stats(self):
        """Should return aggregated statistics."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        model = DummyModel(hidden_dim=64, n_layers=4)

        manager = MultiLayerSteeringManager(bank)
        manager.install(model)
        manager.set_emotional_state(fear=0.5)

        stats = manager.get_stats()

        assert stats['n_hooks'] == 4
        assert stats['emotional_state']['fear'] == 0.5
        assert len(stats['call_counts']) == 4

    def test_invalid_layer_index_fails(self):
        """Should fail on out-of-range layer index."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        model = DummyModel(hidden_dim=64, n_layers=4)

        manager = MultiLayerSteeringManager(bank, layers=[0, 5])  # 5 is out of range

        with pytest.raises(SteeringHookError, match="out of range"):
            manager.install(model)
