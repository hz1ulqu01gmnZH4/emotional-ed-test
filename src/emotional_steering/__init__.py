"""
Emotional Steering for Language Models

A library for steering LLM outputs toward specific emotional tones
using activation steering (representation engineering).

Based on research from:
- Turner et al. 2023: Activation Addition
- Zou et al. 2023: Representation Engineering
"""

from .model import EmotionalSteeringModel, GenerationConfig
from .directions import DirectionExtractor
from .emotions import EMOTIONS, EmotionConfig

__version__ = "0.1.0"
__all__ = ["EmotionalSteeringModel", "GenerationConfig", "DirectionExtractor", "EMOTIONS", "EmotionConfig"]
