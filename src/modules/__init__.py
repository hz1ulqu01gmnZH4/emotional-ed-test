"""Corrected emotional modules for ED architecture.

These modules fix known issues from theoretical review:
- PatienceModule: Proper gamma bounding (fixes 3.2)
- WantingLikingModule: Correct tolerance direction (fixes 3.3)
- ContinuumAffect: Multi-timescale affect (Phase 4)
- LSSEmotionalTrigger: Prediction-error based emotions (Phase 4)
"""

from .patience_module import PatienceModule, HyperbolicPatienceModule
from .wanting_liking_module import WantingLikingModule
from .continuum_affect import ContinuumAffect, MultiChannelAffect, NestedLevelAffect
from .lss_trigger import LSSEmotionalTrigger, AdaptiveLSSThreshold, EmotionalContext
from .nested_agent import NestedEmotionalAgent, SimplifiedNestedAgent, NestedContext

__all__ = [
    # Fixes for issues 3.2 and 3.3
    'PatienceModule',
    'HyperbolicPatienceModule',
    'WantingLikingModule',
    # Phase 4: Multi-timescale architecture
    'ContinuumAffect',
    'MultiChannelAffect',
    'NestedLevelAffect',
    'LSSEmotionalTrigger',
    'AdaptiveLSSThreshold',
    'EmotionalContext',
    # Nested Emotional Agent
    'NestedEmotionalAgent',
    'SimplifiedNestedAgent',
    'NestedContext',
]
