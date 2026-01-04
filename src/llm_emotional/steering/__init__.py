"""Steering module - activation modification via learned direction vectors."""

from .direction_bank import EmotionalDirectionBank
from .steering_hooks import ActivationSteeringHook
from .direction_learner import EmotionalDirectionLearner
from .emotional_llm import EmotionalSteeringLLM

__all__ = [
    "EmotionalDirectionBank",
    "ActivationSteeringHook",
    "EmotionalDirectionLearner",
    "EmotionalSteeringLLM",
]
