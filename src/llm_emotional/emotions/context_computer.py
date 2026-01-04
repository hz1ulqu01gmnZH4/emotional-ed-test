"""
Emotional Context Computer - compute emotional state from context signals.

Translates patterns from existing RL agents:
- agents_fear.py: _compute_fear()
- agents_anger.py: AngerModule
- agents_temporal.py: TonicMoodAgent

Implements both phasic (immediate) and tonic (persistent) emotional states.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ConversationContext:
    """
    Context signals for emotion computation.

    These signals come from various sources during conversation:
    - Safety classifiers
    - Model output analysis
    - User feedback
    - Conversation history
    """
    # Safety/risk signals
    safety_flag: bool = False           # Content flagged by safety classifier
    model_uncertainty: float = 0.5      # Entropy of output distribution [0, 1]

    # Feedback signals
    user_feedback: float = 0.0          # Explicit feedback [-1, +1]
    task_success: Optional[bool] = None # Did the task succeed?

    # Conversation dynamics
    repeated_query: bool = False        # User asking similar thing again
    topic_novelty: float = 0.5          # How novel is this topic [0, 1]
    contradiction_detected: bool = False # Conflicting requirements found

    # Historical counters (updated per turn)
    consecutive_negative: int = 0       # Consecutive negative interactions
    consecutive_positive: int = 0       # Consecutive positive interactions

    # Danger signals (from content analysis)
    mentions_harm: bool = False         # Content mentions harmful topics
    high_stakes: bool = False           # High-stakes decision context


@dataclass
class EmotionalState:
    """Current emotional state with phasic and tonic components."""
    fear: float = 0.0
    curiosity: float = 0.0
    anger: float = 0.0
    joy: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for use with steering hooks."""
        return {
            'fear': self.fear,
            'curiosity': self.curiosity,
            'anger': self.anger,
            'joy': self.joy,
        }

    def dominant_emotion(self) -> str:
        """Return the emotion with highest intensity."""
        emotions = self.to_dict()
        return max(emotions, key=emotions.get)

    def is_neutral(self, threshold: float = 0.1) -> bool:
        """Check if all emotions are below threshold."""
        return all(v < threshold for v in self.to_dict().values())


class EmotionalContextComputer:
    """
    Compute emotional state from context signals.

    Mirrors patterns from Emotional-ED RL agents:
    - Fear: Risk/danger detection, uncertainty, negative outcomes
    - Curiosity: Novelty detection, exploration drive
    - Anger: Frustration from repeated failures, contradictions
    - Joy: Positive feedback, task success, progress

    Implements tonic (baseline) and phasic (immediate) emotion processing.
    Tonic states persist across turns and decay slowly.
    """

    def __init__(
        self,
        fear_decay: float = 0.9,
        joy_decay: float = 0.95,
        frustration_decay: float = 0.85,
        fear_boost_on_danger: float = 0.3,
        joy_boost_on_success: float = 0.2,
    ):
        """
        Initialize emotional computer.

        Args:
            fear_decay: Decay rate for tonic fear (0.9 = 10% decay per turn)
            joy_decay: Decay rate for tonic joy
            frustration_decay: Decay rate for frustration/anger
            fear_boost_on_danger: Fear increment on danger signals
            joy_boost_on_success: Joy increment on success signals
        """
        # Tonic states (persist across turns, decay slowly)
        self.tonic_fear = 0.0
        self.tonic_joy = 0.0
        self.frustration = 0.0  # Cumulative anger from frustration

        # Decay rates
        self.fear_decay = fear_decay
        self.joy_decay = joy_decay
        self.frustration_decay = frustration_decay

        # Boost amounts
        self.fear_boost_on_danger = fear_boost_on_danger
        self.joy_boost_on_success = joy_boost_on_success

        # Turn counter for temporal effects
        self.turn_count = 0

    def compute(self, context: ConversationContext) -> EmotionalState:
        """
        Compute emotional state from context signals.

        Returns EmotionalState with fear, curiosity, anger, joy in [0, 1].

        Args:
            context: Current conversation context

        Returns:
            Computed emotional state
        """
        self.turn_count += 1

        # Initialize phasic emotions
        phasic_fear = 0.0
        phasic_curiosity = 0.0
        phasic_anger = 0.0
        phasic_joy = 0.0

        # === FEAR (mirrors agents_fear.py, agents_cvar_fear.py) ===
        # Safety flag → immediate high fear + boost tonic
        if context.safety_flag:
            phasic_fear = max(phasic_fear, 0.8)
            self.tonic_fear = min(1.0, self.tonic_fear + self.fear_boost_on_danger)

        # Harm mentions → moderate fear
        if context.mentions_harm:
            phasic_fear = max(phasic_fear, 0.5)
            self.tonic_fear = min(1.0, self.tonic_fear + 0.15)

        # High stakes → elevated fear
        if context.high_stakes:
            phasic_fear = max(phasic_fear, 0.4)

        # High uncertainty → fear proportional to uncertainty
        if context.model_uncertainty > 0.7:
            phasic_fear = max(phasic_fear, context.model_uncertainty * 0.5)

        # Negative feedback → increase tonic fear
        if context.user_feedback < -0.5:
            self.tonic_fear = min(1.0, self.tonic_fear + 0.2)

        # Consecutive negatives → escalating fear
        if context.consecutive_negative >= 2:
            phasic_fear = max(phasic_fear, 0.3 + 0.1 * context.consecutive_negative)

        # === CURIOSITY (mirrors agents_joy.py exploration bonus) ===
        # Novelty drives curiosity
        phasic_curiosity = context.topic_novelty

        # Low uncertainty + high novelty = peak curiosity
        if context.model_uncertainty < 0.3 and context.topic_novelty > 0.7:
            phasic_curiosity = min(1.0, phasic_curiosity + 0.2)

        # === ANGER/FRUSTRATION (mirrors agents_anger.py) ===
        # Repeated queries → growing frustration
        if context.repeated_query:
            self.frustration = min(1.0, self.frustration + 0.3)

        # Contradictions → frustration
        if context.contradiction_detected:
            self.frustration = min(1.0, self.frustration + 0.2)

        # Task failure → frustration
        if context.task_success is False:
            self.frustration = min(1.0, self.frustration + 0.25)

        phasic_anger = self.frustration

        # === JOY (mirrors agents_temporal.py mood) ===
        # Positive feedback → joy
        if context.user_feedback > 0.5:
            phasic_joy = max(phasic_joy, 0.6)
            self.tonic_joy = min(1.0, self.tonic_joy + self.joy_boost_on_success)

        # Task success → joy
        if context.task_success is True:
            phasic_joy = max(phasic_joy, 0.5)
            self.tonic_joy = min(1.0, self.tonic_joy + 0.15)

        # Consecutive positives → building joy
        if context.consecutive_positive >= 2:
            phasic_joy = max(phasic_joy, 0.3 + 0.1 * context.consecutive_positive)

        # === COMBINE PHASIC AND TONIC ===
        # Final emotion = max(phasic, tonic) for each
        final_fear = max(phasic_fear, self.tonic_fear)
        final_curiosity = phasic_curiosity  # Curiosity is purely phasic
        final_anger = phasic_anger  # Anger is accumulated frustration
        final_joy = max(phasic_joy, self.tonic_joy)

        # === DECAY TONIC STATES ===
        self.tonic_fear *= self.fear_decay
        self.tonic_joy *= self.joy_decay
        self.frustration *= self.frustration_decay

        # Clamp to [0, 1]
        return EmotionalState(
            fear=min(1.0, max(0.0, final_fear)),
            curiosity=min(1.0, max(0.0, final_curiosity)),
            anger=min(1.0, max(0.0, final_anger)),
            joy=min(1.0, max(0.0, final_joy)),
        )

    def reset(self) -> None:
        """Reset all tonic states (new conversation)."""
        self.tonic_fear = 0.0
        self.tonic_joy = 0.0
        self.frustration = 0.0
        self.turn_count = 0

    def get_tonic_state(self) -> Dict[str, float]:
        """Get current tonic (baseline) emotional state."""
        return {
            'tonic_fear': self.tonic_fear,
            'tonic_joy': self.tonic_joy,
            'frustration': self.frustration,
        }

    def set_tonic_state(
        self,
        tonic_fear: Optional[float] = None,
        tonic_joy: Optional[float] = None,
        frustration: Optional[float] = None,
    ) -> None:
        """
        Manually set tonic states.

        Useful for initializing emotional baseline at conversation start.
        """
        if tonic_fear is not None:
            self.tonic_fear = min(1.0, max(0.0, tonic_fear))
        if tonic_joy is not None:
            self.tonic_joy = min(1.0, max(0.0, tonic_joy))
        if frustration is not None:
            self.frustration = min(1.0, max(0.0, frustration))

    def __repr__(self) -> str:
        return (
            f"EmotionalContextComputer("
            f"tonic_fear={self.tonic_fear:.2f}, "
            f"tonic_joy={self.tonic_joy:.2f}, "
            f"frustration={self.frustration:.2f}, "
            f"turns={self.turn_count})"
        )
