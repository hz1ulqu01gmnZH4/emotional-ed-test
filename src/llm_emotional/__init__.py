"""
Emotional LLM Integration - Activation Steering Approach.

This package implements emotional activation steering for LLMs,
adding learned "emotional direction vectors" to hidden states at inference time.
"""

from .steering.direction_bank import EmotionalDirectionBank
from .steering.steering_hooks import ActivationSteeringHook
from .steering.emotional_llm import EmotionalSteeringLLM
from .emotions.context_computer import EmotionalContextComputer, ConversationContext

__version__ = "0.1.0"

__all__ = [
    "EmotionalDirectionBank",
    "ActivationSteeringHook",
    "EmotionalSteeringLLM",
    "EmotionalContextComputer",
    "ConversationContext",
]
