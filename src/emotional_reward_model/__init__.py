"""
Emotional Reward Model (Approach 5).

A side-channel approach that trains a separate Emotional Reward Model
to observe LLM outputs and provide emotional feedback signals.
"""

from .signals import EmotionalSignals
from .reward_model import EmotionalRewardModel
from .logit_modulator import LogitModulator
from .temperature_modulator import TemperatureModulator
from .fear_module import FearModule
from .model import EmotionalRewardLLM
from .trainer import ERMTrainer

__all__ = [
    "EmotionalSignals",
    "EmotionalRewardModel",
    "LogitModulator",
    "TemperatureModulator",
    "FearModule",
    "EmotionalRewardLLM",
    "ERMTrainer",
]
