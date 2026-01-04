"""
Emotional Signals dataclass for representing emotional state.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmotionalSignals:
    """
    Output from the Emotional Reward Model.

    Represents the emotional state with multiple dimensions.
    """

    fear: float = 0.0
    curiosity: float = 0.0
    anger: float = 0.0
    joy: float = 0.0
    anxiety: float = 0.0  # Tonic fear
    confidence: float = 0.5

    # Number of emotion dimensions
    N_EMOTIONS: int = field(default=6, repr=False)

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convert to tensor.

        Args:
            device: Optional device to place tensor on

        Returns:
            Tensor of shape [6] with emotion values
        """
        tensor = torch.tensor([
            self.fear,
            self.curiosity,
            self.anger,
            self.joy,
            self.anxiety,
            self.confidence,
        ])
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "EmotionalSignals":
        """
        Create from tensor.

        Args:
            t: Tensor of shape [6] with emotion values

        Returns:
            EmotionalSignals instance
        """
        # Handle different tensor shapes
        t = t.detach().cpu()
        if t.dim() > 1:
            t = t.squeeze()

        return cls(
            fear=t[0].item(),
            curiosity=t[1].item(),
            anger=t[2].item(),
            joy=t[3].item(),
            anxiety=t[4].item(),
            confidence=t[5].item(),
        )

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
    def from_dict(cls, d: dict) -> "EmotionalSignals":
        """Create from dictionary."""
        return cls(
            fear=d.get("fear", 0.0),
            curiosity=d.get("curiosity", 0.0),
            anger=d.get("anger", 0.0),
            joy=d.get("joy", 0.0),
            anxiety=d.get("anxiety", 0.0),
            confidence=d.get("confidence", 0.5),
        )

    def dominant_emotion(self) -> str:
        """Get the dominant emotion (excluding confidence)."""
        emotions = {
            "fear": self.fear,
            "curiosity": self.curiosity,
            "anger": self.anger,
            "joy": self.joy,
            "anxiety": self.anxiety,
        }
        return max(emotions, key=emotions.get)

    def overall_valence(self) -> float:
        """
        Get overall emotional valence (-1 to +1).

        Positive: joy, curiosity, confidence
        Negative: fear, anger, anxiety
        """
        positive = self.joy + self.curiosity + self.confidence
        negative = self.fear + self.anger + self.anxiety
        total = positive + negative
        if total == 0:
            return 0.0
        return (positive - negative) / total

    def copy(self) -> "EmotionalSignals":
        """Create a copy of this signals object."""
        return EmotionalSignals(
            fear=self.fear,
            curiosity=self.curiosity,
            anger=self.anger,
            joy=self.joy,
            anxiety=self.anxiety,
            confidence=self.confidence,
        )

    @classmethod
    def neutral(cls) -> "EmotionalSignals":
        """Create neutral emotional signals."""
        return cls(confidence=0.5)

    @classmethod
    def fearful(cls, intensity: float = 0.8) -> "EmotionalSignals":
        """Create fearful emotional signals."""
        return cls(
            fear=intensity,
            anxiety=intensity * 0.5,
            confidence=0.3,
        )

    @classmethod
    def curious(cls, intensity: float = 0.8) -> "EmotionalSignals":
        """Create curious emotional signals."""
        return cls(
            curiosity=intensity,
            joy=intensity * 0.3,
            confidence=0.6,
        )

    @classmethod
    def joyful(cls, intensity: float = 0.8) -> "EmotionalSignals":
        """Create joyful emotional signals."""
        return cls(
            joy=intensity,
            curiosity=intensity * 0.2,
            confidence=0.8,
        )

    @classmethod
    def angry(cls, intensity: float = 0.8) -> "EmotionalSignals":
        """Create angry emotional signals."""
        return cls(
            anger=intensity,
            anxiety=intensity * 0.3,
            confidence=0.4,
        )
