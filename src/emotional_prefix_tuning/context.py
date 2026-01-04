"""
Emotional Context dataclass for prefix tuning.

Stores signals from the environment that inform emotional state.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EmotionalContext:
    """
    Context for computing emotional state.

    Captures signals from the environment that should influence
    the model's emotional state and thus its prefix generation.
    """

    # Recent feedback signals (-1 to 1 scale)
    last_reward: float = 0.0

    # Safety flag (True if safety concern detected)
    safety_flag: bool = False

    # Estimated user satisfaction (0 to 1)
    user_satisfaction: float = 0.5

    # Conversation dynamics
    repeated_query: bool = False  # User asked similar thing before
    topic_novelty: float = 0.5    # How novel is current topic (0-1)
    contradiction_detected: bool = False

    # Historical state (tonic emotions)
    cumulative_negative: float = 0.0  # Accumulated negative feedback
    cumulative_positive: float = 0.0  # Accumulated positive feedback
    failed_attempts: int = 0          # Consecutive failures

    # Optional: explicit emotional targets for training
    target_fear: Optional[float] = None
    target_curiosity: Optional[float] = None
    target_anger: Optional[float] = None
    target_joy: Optional[float] = None

    # Conversation history (for more sophisticated encoding)
    history: List[str] = field(default_factory=list)

    def update_from_feedback(self, feedback: float) -> None:
        """Update context based on user feedback."""
        self.last_reward = feedback
        if feedback > 0:
            self.cumulative_positive += feedback
            self.failed_attempts = 0
        elif feedback < 0:
            self.cumulative_negative += abs(feedback)
            self.failed_attempts += 1

    def reset_episodic(self) -> None:
        """Reset episodic (phasic) signals, keep tonic state."""
        self.last_reward = 0.0
        self.safety_flag = False
        self.repeated_query = False
        self.contradiction_detected = False

    def reset_all(self) -> None:
        """Full reset for new conversation."""
        self.last_reward = 0.0
        self.safety_flag = False
        self.user_satisfaction = 0.5
        self.repeated_query = False
        self.topic_novelty = 0.5
        self.contradiction_detected = False
        self.cumulative_negative = 0.0
        self.cumulative_positive = 0.0
        self.failed_attempts = 0
        self.history.clear()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "last_reward": self.last_reward,
            "safety_flag": self.safety_flag,
            "user_satisfaction": self.user_satisfaction,
            "repeated_query": self.repeated_query,
            "topic_novelty": self.topic_novelty,
            "contradiction_detected": self.contradiction_detected,
            "cumulative_negative": self.cumulative_negative,
            "cumulative_positive": self.cumulative_positive,
            "failed_attempts": self.failed_attempts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionalContext":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k != "history"})
