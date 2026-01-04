"""
Emotional Adapter Gating for Language Models

Inserts small trainable adapter modules between frozen LLM layers,
where the adapter output is gated by emotional state. Provides
deeper integration than prefix tuning.

Based on:
- LoRA (Hu et al., 2021)
- Emotional-ED project
"""

from .state import EmotionalState
from .gate import EmotionalGate
from .adapter import EmotionalAdapter
from .encoder import EmotionalEncoderForAdapter
from .model import EmotionalAdapterLLM
from .trainer import EmotionalAdapterTrainer

__version__ = "0.1.0"
__all__ = [
    "EmotionalState",
    "EmotionalGate",
    "EmotionalAdapter",
    "EmotionalEncoderForAdapter",
    "EmotionalAdapterLLM",
    "EmotionalAdapterTrainer",
]
