"""
Tonic Emotional State for persistent emotional baseline.

Decays slowly over conversation and carries emotional momentum.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TonicEmotionalState:
    """
    Persistent emotional state that decays slowly.

    Similar to cumulative_fear in Emotional-ED, but multi-dimensional.
    """

    # Primary emotions (0-1 scale)
    fear: float = 0.0
    anxiety: float = 0.0
    joy: float = 0.0
    trust: float = 0.5
    frustration: float = 0.0
    curiosity: float = 0.0

    # Decay rates (per turn) - how quickly emotions fade
    fear_decay: float = 0.9
    anxiety_decay: float = 0.95
    joy_decay: float = 0.9
    frustration_decay: float = 0.85
    curiosity_decay: float = 0.85

    def update_from_feedback(self, feedback: float) -> None:
        """
        Update tonic state based on feedback.

        Args:
            feedback: User feedback from -1 (negative) to +1 (positive)
        """
        if feedback < -0.3:
            # Negative feedback increases fear and frustration
            self.fear = min(1.0, self.fear + 0.2)
            self.frustration = min(1.0, self.frustration + 0.3)
            self.trust = max(0.0, self.trust - 0.1)
        elif feedback > 0.3:
            # Positive feedback increases joy and trust
            self.joy = min(1.0, self.joy + 0.2)
            self.trust = min(1.0, self.trust + 0.05)
            # Reduce fear when things go well
            self.fear = max(0.0, self.fear - 0.1)

    def decay(self) -> None:
        """Apply decay to tonic emotions."""
        self.fear *= self.fear_decay
        self.anxiety *= self.anxiety_decay
        self.joy *= self.joy_decay
        self.frustration *= self.frustration_decay
        self.curiosity *= self.curiosity_decay

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'fear': self.fear,
            'anxiety': self.anxiety,
            'joy': self.joy,
            'trust': self.trust,
            'frustration': self.frustration,
            'curiosity': self.curiosity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "TonicEmotionalState":
        """Create from dictionary."""
        state = cls()
        for key, value in data.items():
            if hasattr(state, key) and not key.endswith('_decay'):
                setattr(state, key, value)
        return state

    def reset(self) -> None:
        """Reset to neutral state."""
        self.fear = 0.0
        self.anxiety = 0.0
        self.joy = 0.0
        self.trust = 0.5
        self.frustration = 0.0
        self.curiosity = 0.0

    def dominant_emotion(self) -> str:
        """Get the dominant emotion."""
        emotions = {
            'fear': self.fear,
            'anxiety': self.anxiety,
            'joy': self.joy,
            'frustration': self.frustration,
            'curiosity': self.curiosity,
        }
        return max(emotions, key=emotions.get)

    def overall_valence(self) -> float:
        """Get overall emotional valence (-1 to +1)."""
        positive = self.joy + self.trust + self.curiosity
        negative = self.fear + self.anxiety + self.frustration
        total = positive + negative
        if total == 0:
            return 0.0
        return (positive - negative) / total
