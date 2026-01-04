"""Emotions module - context computation and dataset handling."""

from .context_computer import EmotionalContextComputer, ConversationContext
from .datasets import (
    load_dataset,
    get_contrastive_pairs,
    DatasetNotFoundError,
    DatasetValidationError,
)

__all__ = [
    "EmotionalContextComputer",
    "ConversationContext",
    "load_dataset",
    "get_contrastive_pairs",
    "DatasetNotFoundError",
    "DatasetValidationError",
]
