"""
External Emotional Memory for Language Models

Maintains persistent emotional memory external to the LLM.
The memory stores emotional associations, past experiences, and
tonic emotional states. At each turn, relevant emotional context
is retrieved and provided to the frozen LLM.

Based on:
- Emotional-ED project: cumulative_fear and temporal tracking
- Memory-augmented neural networks
"""

from .tonic_state import TonicEmotionalState
from .episodic_memory import EpisodicMemoryEntry, EpisodicEmotionalMemory
from .semantic_memory import SemanticEmotionalMemory
from .context_generator import EmotionalContextGenerator
from .model import EmotionalMemoryLLM

__version__ = "0.1.0"
__all__ = [
    "TonicEmotionalState",
    "EpisodicMemoryEntry",
    "EpisodicEmotionalMemory",
    "SemanticEmotionalMemory",
    "EmotionalContextGenerator",
    "EmotionalMemoryLLM",
]
