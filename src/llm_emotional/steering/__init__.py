"""Steering module - activation modification via learned direction vectors."""

from .direction_bank import EmotionalDirectionBank
from .steering_hooks import ActivationSteeringHook
from .direction_learner import EmotionalDirectionLearner
from .emotional_llm import EmotionalSteeringLLM

# V2: Layer-weighted steering
from .steering_hooks_v2 import LayerWeightedSteeringHook, LayerWeightedSteeringManager
from .emotional_llm_v2 import EmotionalSteeringLLMv2

# V3: Error diffusion steering
from .steering_hooks_v3 import ErrorDiffusionSteeringHook, ErrorDiffusionManager, ErrorState
from .emotional_llm_v3 import (
    EmotionalSteeringLLMv3,
    EmotionState,
    compute_wanting_liking_directions,
    compute_regulatory_directions,
)

__all__ = [
    # V1
    "EmotionalDirectionBank",
    "ActivationSteeringHook",
    "EmotionalDirectionLearner",
    "EmotionalSteeringLLM",
    # V2
    "LayerWeightedSteeringHook",
    "LayerWeightedSteeringManager",
    "EmotionalSteeringLLMv2",
    # V3
    "ErrorDiffusionSteeringHook",
    "ErrorDiffusionManager",
    "ErrorState",
    "EmotionalSteeringLLMv3",
    "EmotionState",
    "compute_wanting_liking_directions",
    "compute_regulatory_directions",
]
