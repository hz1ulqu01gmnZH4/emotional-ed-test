"""
Emotional State dataclass for adapter gating.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class EmotionalState:
    """
    Current emotional state vector.

    Represents the emotional context that gates adapter contributions.
    """

    # Primary emotions (0-1 scale)
    fear: float = 0.0
    curiosity: float = 0.0
    anger: float = 0.0
    joy: float = 0.0

    # Secondary emotional states
    anxiety: float = 0.0      # Tonic fear / persistent worry
    confidence: float = 0.5   # Self-assessment of capability

    # Device for tensor conversion
    _device: Optional[torch.device] = None

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert to tensor for network input."""
        d = device or self._device or torch.device("cpu")
        return torch.tensor([
            self.fear,
            self.curiosity,
            self.anger,
            self.joy,
            self.anxiety,
            self.confidence
        ], dtype=torch.float32, device=d)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "EmotionalState":
        """Create from tensor."""
        values = tensor.detach().cpu().tolist()
        return cls(
            fear=values[0],
            curiosity=values[1],
            anger=values[2],
            joy=values[3],
            anxiety=values[4] if len(values) > 4 else 0.0,
            confidence=values[5] if len(values) > 5 else 0.5,
        )

    @classmethod
    def neutral(cls) -> "EmotionalState":
        """Create neutral emotional state."""
        return cls()

    @classmethod
    def fearful(cls, intensity: float = 1.0) -> "EmotionalState":
        """Create fearful state."""
        return cls(fear=intensity, anxiety=intensity * 0.5)

    @classmethod
    def curious(cls, intensity: float = 1.0) -> "EmotionalState":
        """Create curious state."""
        return cls(curiosity=intensity, confidence=0.7)

    @classmethod
    def angry(cls, intensity: float = 1.0) -> "EmotionalState":
        """Create angry state."""
        return cls(anger=intensity, anxiety=intensity * 0.3)

    @classmethod
    def joyful(cls, intensity: float = 1.0) -> "EmotionalState":
        """Create joyful state."""
        return cls(joy=intensity, confidence=0.8)

    def blend(self, other: "EmotionalState", weight: float = 0.5) -> "EmotionalState":
        """Blend two emotional states."""
        w = weight
        return EmotionalState(
            fear=self.fear * (1 - w) + other.fear * w,
            curiosity=self.curiosity * (1 - w) + other.curiosity * w,
            anger=self.anger * (1 - w) + other.anger * w,
            joy=self.joy * (1 - w) + other.joy * w,
            anxiety=self.anxiety * (1 - w) + other.anxiety * w,
            confidence=self.confidence * (1 - w) + other.confidence * w,
        )

    def dominant_emotion(self) -> str:
        """Get the dominant emotion."""
        emotions = {
            "fear": self.fear,
            "curiosity": self.curiosity,
            "anger": self.anger,
            "joy": self.joy,
        }
        return max(emotions, key=emotions.get)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "fear": self.fear,
            "curiosity": self.curiosity,
            "anger": self.anger,
            "joy": self.joy,
            "anxiety": self.anxiety,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionalState":
        """Create from dictionary."""
        return cls(**data)
