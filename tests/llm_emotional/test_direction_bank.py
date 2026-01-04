"""
Tests for EmotionalDirectionBank.

Tests:
- Initialization with valid dimensions
- Setting and getting directions
- Combined steering computation
- Save and load persistence
- Error handling (no fallback - must fail loudly)
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from src.llm_emotional.steering.direction_bank import (
    EmotionalDirectionBank,
    DirectionBankError,
)


class TestDirectionBankInit:
    """Test initialization."""

    def test_init_valid_dimensions(self):
        """Should initialize with valid dimensions."""
        bank = EmotionalDirectionBank(hidden_dim=768, n_layers=12)

        assert bank.hidden_dim == 768
        assert bank.n_layers == 12
        assert len(bank.directions) == 4  # fear, curiosity, anger, joy
        assert all(
            bank.directions[e].shape == (12, 768)
            for e in bank.EMOTIONS
        )

    def test_init_creates_small_random_directions(self):
        """Initial directions should be small (close to zero)."""
        bank = EmotionalDirectionBank(hidden_dim=100, n_layers=6)

        for emotion in bank.EMOTIONS:
            norm = bank.directions[emotion].norm().item()
            # Random init with 0.01 scale, should have small norm
            assert norm < 10.0  # sqrt(600) * 0.01 ≈ 0.24, allow headroom

    def test_init_invalid_hidden_dim_fails(self):
        """Should fail loudly on invalid hidden_dim."""
        with pytest.raises(DirectionBankError, match="hidden_dim must be positive"):
            EmotionalDirectionBank(hidden_dim=0, n_layers=12)

        with pytest.raises(DirectionBankError, match="hidden_dim must be positive"):
            EmotionalDirectionBank(hidden_dim=-1, n_layers=12)

    def test_init_invalid_n_layers_fails(self):
        """Should fail loudly on invalid n_layers."""
        with pytest.raises(DirectionBankError, match="n_layers must be positive"):
            EmotionalDirectionBank(hidden_dim=768, n_layers=0)


class TestDirectionBankSetGet:
    """Test setting and getting directions."""

    def test_set_direction_valid(self):
        """Should set direction for known emotion."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        direction = torch.randn(4, 64)

        bank.set_direction('fear', direction)

        assert torch.allclose(bank.directions['fear'], direction)
        assert bank.learned['fear'] is True

    def test_set_direction_unknown_emotion_fails(self):
        """Should fail on unknown emotion."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        direction = torch.randn(4, 64)

        with pytest.raises(DirectionBankError, match="Unknown emotion"):
            bank.set_direction('sadness', direction)

    def test_set_direction_wrong_shape_fails(self):
        """Should fail on shape mismatch."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        wrong_direction = torch.randn(4, 32)  # Wrong hidden_dim

        with pytest.raises(DirectionBankError, match="shape mismatch"):
            bank.set_direction('fear', wrong_direction)

    def test_get_direction_valid(self):
        """Should get direction for specific layer."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        direction = torch.randn(4, 64)
        bank.set_direction('curiosity', direction)

        layer_dir = bank.get_direction('curiosity', layer_idx=2)

        assert layer_dir.shape == (64,)
        assert torch.allclose(layer_dir, direction[2])

    def test_get_direction_invalid_layer_fails(self):
        """Should fail on out-of-range layer."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)

        with pytest.raises(DirectionBankError, match="out of range"):
            bank.get_direction('fear', layer_idx=4)

        with pytest.raises(DirectionBankError, match="out of range"):
            bank.get_direction('fear', layer_idx=-1)


class TestCombinedSteering:
    """Test combined steering computation."""

    def test_empty_state_returns_zeros(self):
        """Empty emotional state should return zero steering."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)

        steering = bank.get_combined_steering({}, layer_idx=0)

        assert steering.shape == (64,)
        assert torch.allclose(steering, torch.zeros(64))

    def test_zero_intensity_skipped(self):
        """Zero intensity should not contribute to steering."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)

        steering = bank.get_combined_steering({'fear': 0.0, 'joy': 0.0}, layer_idx=0)

        assert torch.allclose(steering, torch.zeros(64))

    def test_single_emotion_steering(self):
        """Single emotion should return scaled direction."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        direction = torch.randn(4, 64)
        bank.set_direction('fear', direction)

        steering = bank.get_combined_steering({'fear': 0.5}, layer_idx=1)

        # steering = intensity * weight * direction
        expected = 0.5 * 1.0 * direction[1]  # weight is 1.0 by default
        assert torch.allclose(steering, expected)

    def test_multiple_emotions_sum(self):
        """Multiple emotions should sum their steering."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        fear_dir = torch.randn(4, 64)
        joy_dir = torch.randn(4, 64)
        bank.set_direction('fear', fear_dir)
        bank.set_direction('joy', joy_dir)

        steering = bank.get_combined_steering(
            {'fear': 0.3, 'joy': 0.7},
            layer_idx=0
        )

        expected = 0.3 * fear_dir[0] + 0.7 * joy_dir[0]
        assert torch.allclose(steering, expected)

    def test_unknown_emotion_fails_loudly(self):
        """Unknown emotions should FAIL - no silent ignore."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)

        # Must raise on unknown emotion - typos should not be silently ignored
        with pytest.raises(DirectionBankError, match="Unknown emotion"):
            bank.get_combined_steering(
                {'sadness': 0.5, 'fear': 0.0},
                layer_idx=0
            )


class TestPersistence:
    """Test save and load functionality."""

    def test_save_load_roundtrip(self):
        """Saved bank should load identically."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)

        # Set custom directions
        fear_dir = torch.randn(4, 64)
        joy_dir = torch.randn(4, 64)
        bank.set_direction('fear', fear_dir)
        bank.set_direction('joy', joy_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bank.json"
            bank.save(path)

            loaded = EmotionalDirectionBank.load(path)

        assert loaded.hidden_dim == bank.hidden_dim
        assert loaded.n_layers == bank.n_layers
        assert torch.allclose(loaded.directions['fear'], fear_dir, atol=1e-5)
        assert torch.allclose(loaded.directions['joy'], joy_dir, atol=1e-5)
        assert loaded.learned['fear'] is True
        assert loaded.learned['joy'] is True

    def test_load_nonexistent_fails(self):
        """Loading from nonexistent path should fail loudly."""
        with pytest.raises(DirectionBankError, match="not found"):
            EmotionalDirectionBank.load(Path("/nonexistent/path.json"))

    def test_load_corrupted_json_fails(self):
        """Loading corrupted JSON should fail loudly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text("not valid json {{{")

            with pytest.raises(DirectionBankError, match="Corrupted"):
                EmotionalDirectionBank.load(path)

    def test_load_missing_fields_fails(self):
        """Loading JSON missing required fields should fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "incomplete.json"
            path.write_text('{"hidden_dim": 64}')  # Missing n_layers, etc.

            with pytest.raises(DirectionBankError, match="Missing fields"):
                EmotionalDirectionBank.load(path)


class TestRepr:
    """Test string representation."""

    def test_repr_shows_learned_status(self):
        """Repr should show which directions are learned."""
        bank = EmotionalDirectionBank(hidden_dim=64, n_layers=4)
        bank.set_direction('fear', torch.randn(4, 64))

        repr_str = repr(bank)

        assert "fear=Y" in repr_str
        assert "curiosity=N" in repr_str
        assert "hidden_dim=64" in repr_str
        assert "n_layers=4" in repr_str
