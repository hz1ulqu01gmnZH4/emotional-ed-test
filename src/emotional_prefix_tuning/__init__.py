"""
Emotional Prefix Tuning for Language Models

Extends prefix tuning by making prefixes dynamically conditioned
on emotional state. The LLM is frozen; only emotional modules train.

Based on:
- Li & Liang 2021: Prefix-Tuning
- Emotional-ED project: Emotional learning in RL
"""

from .context import EmotionalContext
from .encoder import EmotionalEncoder
from .prefix_generator import EmotionalPrefixGenerator
from .model import EmotionalPrefixLLM
from .trainer import EmotionalPrefixTrainer

__version__ = "0.1.0"
__all__ = [
    "EmotionalContext",
    "EmotionalEncoder",
    "EmotionalPrefixGenerator",
    "EmotionalPrefixLLM",
    "EmotionalPrefixTrainer",
]
