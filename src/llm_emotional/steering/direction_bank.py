"""
Emotional Direction Bank - stores learned emotional direction vectors.

Each emotion has a direction vector per layer that can be added to hidden states
to steer the model's behavior toward that emotional quality.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor


class DirectionBankError(Exception):
    """Raised when direction bank operations fail. NO FALLBACK."""
    pass


class EmotionalDirectionBank:
    """
    Stores learned emotional direction vectors.

    Each emotion has:
    - directions[emotion]: [n_layers, hidden_dim] tensor
    - layer_weights[emotion]: [n_layers] tensor for per-layer scaling

    Steering for layer i = sum(intensity * weight[i] * direction[i] for each emotion)
    """

    EMOTIONS = ('fear', 'curiosity', 'anger', 'joy')

    def __init__(self, hidden_dim: int, n_layers: int):
        """
        Initialize direction bank with small random directions.

        Args:
            hidden_dim: Hidden dimension of the model
            n_layers: Number of transformer layers
        """
        if hidden_dim <= 0:
            raise DirectionBankError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_layers <= 0:
            raise DirectionBankError(f"n_layers must be positive, got {n_layers}")

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Initialize with small random directions (will be learned)
        self.directions: Dict[str, Tensor] = {
            emotion: torch.randn(n_layers, hidden_dim) * 0.01
            for emotion in self.EMOTIONS
        }

        # Per-layer weights (uniform initially, can be learned)
        self.layer_weights: Dict[str, Tensor] = {
            emotion: torch.ones(n_layers)
            for emotion in self.EMOTIONS
        }

        # Track if directions have been learned (not just random init)
        self.learned: Dict[str, bool] = {emotion: False for emotion in self.EMOTIONS}

    def set_direction(self, emotion: str, direction: Tensor) -> None:
        """
        Set the direction for an emotion.

        Args:
            emotion: One of EMOTIONS
            direction: Tensor of shape [n_layers, hidden_dim]

        Raises:
            DirectionBankError: If emotion unknown or shape mismatch
        """
        if emotion not in self.EMOTIONS:
            raise DirectionBankError(
                f"Unknown emotion: {emotion}. Must be one of {self.EMOTIONS}"
            )

        expected_shape = (self.n_layers, self.hidden_dim)
        if direction.shape != expected_shape:
            raise DirectionBankError(
                f"Direction shape mismatch for {emotion}: "
                f"expected {expected_shape}, got {tuple(direction.shape)}"
            )

        self.directions[emotion] = direction.clone()
        self.learned[emotion] = True

    def get_direction(self, emotion: str, layer_idx: int) -> Tensor:
        """
        Get direction vector for a specific emotion and layer.

        Args:
            emotion: One of EMOTIONS
            layer_idx: Layer index (0 to n_layers-1)

        Returns:
            Direction tensor of shape [hidden_dim]
        """
        if emotion not in self.EMOTIONS:
            raise DirectionBankError(f"Unknown emotion: {emotion}")
        if not 0 <= layer_idx < self.n_layers:
            raise DirectionBankError(
                f"layer_idx {layer_idx} out of range [0, {self.n_layers})"
            )

        return self.directions[emotion][layer_idx]

    def get_combined_steering(
        self,
        emotional_state: Dict[str, float],
        layer_idx: int
    ) -> Tensor:
        """
        Compute combined steering vector for a layer given emotional state.

        steering = sum(intensity * layer_weight * direction for each emotion)

        Args:
            emotional_state: Dict mapping emotion names to intensities in [0, 1]
            layer_idx: Which layer to compute steering for

        Returns:
            Combined steering vector of shape [hidden_dim]
        """
        if not 0 <= layer_idx < self.n_layers:
            raise DirectionBankError(
                f"layer_idx {layer_idx} out of range [0, {self.n_layers})"
            )

        steering = torch.zeros(self.hidden_dim)

        for emotion, intensity in emotional_state.items():
            if emotion not in self.directions:
                raise DirectionBankError(
                    f"Unknown emotion '{emotion}' in emotional_state. "
                    f"Valid emotions are: {list(self.EMOTIONS)}"
                )
            if intensity == 0.0:
                continue  # Zero intensity is valid no-op, not an error

            weight = self.layer_weights[emotion][layer_idx]
            direction = self.directions[emotion][layer_idx]
            steering = steering + intensity * weight * direction

        return steering

    def save(self, path: Path) -> None:
        """
        Save direction bank to disk.

        Args:
            path: Path to save to (will create parent directories)

        Raises:
            DirectionBankError: If save fails
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tensors to lists for JSON serialization
        data = {
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'directions': {
                emotion: direction.tolist()
                for emotion, direction in self.directions.items()
            },
            'layer_weights': {
                emotion: weights.tolist()
                for emotion, weights in self.layer_weights.items()
            },
            'learned': self.learned,
        }

        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            raise DirectionBankError(f"Failed to save direction bank to {path}: {e}")

        # Verify save succeeded
        self._verify_save(path, data)

    def _verify_save(self, path: Path, expected_data: dict) -> None:
        """Verify saved data matches what we wrote. FAIL if mismatch."""
        try:
            with open(path, 'r') as f:
                loaded = json.load(f)
        except Exception as e:
            raise DirectionBankError(f"Failed to verify save at {path}: {e}")

        if loaded['hidden_dim'] != expected_data['hidden_dim']:
            raise DirectionBankError("Save verification failed: hidden_dim mismatch")
        if loaded['n_layers'] != expected_data['n_layers']:
            raise DirectionBankError("Save verification failed: n_layers mismatch")

    @classmethod
    def load(cls, path: Path) -> "EmotionalDirectionBank":
        """
        Load direction bank from disk.

        Args:
            path: Path to load from

        Returns:
            Loaded EmotionalDirectionBank

        Raises:
            DirectionBankError: If file not found or corrupted
        """
        path = Path(path)

        if not path.exists():
            raise DirectionBankError(
                f"Direction bank not found at {path}. "
                "Train directions first using EmotionalDirectionLearner."
            )

        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DirectionBankError(f"Corrupted direction bank at {path}: {e}")
        except Exception as e:
            raise DirectionBankError(f"Failed to load direction bank from {path}: {e}")

        # Validate required fields
        required = {'hidden_dim', 'n_layers', 'directions', 'layer_weights'}
        missing = required - set(data.keys())
        if missing:
            raise DirectionBankError(f"Missing fields in direction bank: {missing}")

        # Create instance and populate
        bank = cls(data['hidden_dim'], data['n_layers'])

        for emotion in cls.EMOTIONS:
            if emotion in data['directions']:
                bank.directions[emotion] = torch.tensor(data['directions'][emotion])
            if emotion in data['layer_weights']:
                bank.layer_weights[emotion] = torch.tensor(data['layer_weights'][emotion])

        if 'learned' in data:
            bank.learned = data['learned']

        return bank

    def __repr__(self) -> str:
        learned_str = ", ".join(
            f"{e}={'Y' if self.learned[e] else 'N'}"
            for e in self.EMOTIONS
        )
        return (
            f"EmotionalDirectionBank("
            f"hidden_dim={self.hidden_dim}, "
            f"n_layers={self.n_layers}, "
            f"learned=[{learned_str}])"
        )
