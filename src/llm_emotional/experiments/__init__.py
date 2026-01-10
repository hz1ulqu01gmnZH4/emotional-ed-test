"""
Experiments module for V3 Falsification Protocol.

Implements operationalized tests from V2 protocol:
- Quantitative metrics (classifier, logit shift, internal metrics)
- Behavioral tasks (Iowa Gambling Task, choice tasks)
- Falsification runner for all hypotheses
"""

from .metrics import (
    EmotionMetrics,
    InternalMetrics,
    FunctionalMetrics,
    EmotionClassifier,
    LogitShiftMeasurer,
    InternalMetricsMeasurer,
    FunctionalMetricsMeasurer,
    compute_effect_size,
    EMOTION_KEYWORDS,
    FUNCTIONAL_INDICATORS,
)

from .behavioral_tasks import (
    IGTResult,
    ChoiceTaskResult,
    WantingLikingResult,
    IowaGamblingTask,
    BinaryChoiceTask,
    WantingLikingTask,
    IntrinsicPersistenceTest,
    RobustnessTest,
)

from .falsification_runner import (
    HypothesisResult,
    FalsificationResults,
    FalsificationRunner,
)

__all__ = [
    # Metrics
    "EmotionMetrics",
    "InternalMetrics",
    "FunctionalMetrics",
    "EmotionClassifier",
    "LogitShiftMeasurer",
    "InternalMetricsMeasurer",
    "FunctionalMetricsMeasurer",
    "compute_effect_size",
    "EMOTION_KEYWORDS",
    "FUNCTIONAL_INDICATORS",
    # Behavioral tasks
    "IGTResult",
    "ChoiceTaskResult",
    "WantingLikingResult",
    "IowaGamblingTask",
    "BinaryChoiceTask",
    "WantingLikingTask",
    "IntrinsicPersistenceTest",
    "RobustnessTest",
    # Runner
    "HypothesisResult",
    "FalsificationResults",
    "FalsificationRunner",
]
