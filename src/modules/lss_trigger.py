"""Local Surprise Signal (LSS) emotional triggering module.

Based on:
- Behrouz et al. (2025): Nested Learning - LSS as prediction error
- Neuroscience: Prediction error as emotional trigger

Key insight: Emotions are triggered by prediction errors (surprise):
- Negative surprise near threat → Fear
- Negative surprise when blocked → Anger/Frustration
- Positive surprise → Joy
- Counterfactual negative → Regret

LSS = actual_outcome - predicted_outcome (TD error)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class EmotionalContext:
    """Context for emotion computation."""
    # Threat-related
    near_threat: bool = False
    threat_distance: float = float('inf')
    threat_direction: tuple = (0, 0)

    # Goal-related
    goal_distance: float = float('inf')
    goal_direction: tuple = (0, 0)
    was_blocked: bool = False
    consecutive_blocks: int = 0

    # Outcome-related
    reward: float = 0.0
    counterfactual_available: bool = False
    foregone_reward: float = 0.0

    # State
    is_terminal: bool = False


class LSSEmotionalTrigger:
    """Trigger emotions based on Local Surprise Signal (prediction error).

    Unlike heuristic approaches that use distance-based fear ramps,
    LSS-based triggering uses the actual prediction error as the
    emotional signal. This is more adaptive and generalizes better.
    """

    def __init__(
        self,
        surprise_threshold: float = 0.1,
        fear_sensitivity: float = 1.0,
        anger_sensitivity: float = 1.0,
        joy_sensitivity: float = 1.0,
        regret_sensitivity: float = 1.0
    ):
        """Initialize LSS emotional trigger.

        Args:
            surprise_threshold: Minimum |LSS| to trigger emotions
            fear_sensitivity: Scaling for fear response
            anger_sensitivity: Scaling for anger response
            joy_sensitivity: Scaling for joy response
            regret_sensitivity: Scaling for regret response
        """
        self.surprise_threshold = surprise_threshold
        self.fear_sensitivity = fear_sensitivity
        self.anger_sensitivity = anger_sensitivity
        self.joy_sensitivity = joy_sensitivity
        self.regret_sensitivity = regret_sensitivity

        # Track running prediction for LSS computation
        self.running_prediction = 0.0
        self.prediction_lr = 0.1

    def compute_lss(self, predicted: float, actual: float) -> float:
        """Compute Local Surprise Signal.

        LSS = actual - predicted

        Positive LSS: Better than expected (positive surprise)
        Negative LSS: Worse than expected (negative surprise)

        Args:
            predicted: Predicted outcome (Q-value, expected reward)
            actual: Actual outcome (received reward + discounted next value)

        Returns:
            Local Surprise Signal
        """
        return actual - predicted

    def update_prediction(self, outcome: float):
        """Update running prediction with observed outcome.

        Args:
            outcome: Observed outcome value
        """
        self.running_prediction = (
            (1 - self.prediction_lr) * self.running_prediction +
            self.prediction_lr * outcome
        )

    def trigger_emotions(
        self,
        lss: float,
        context: EmotionalContext
    ) -> Dict[str, float]:
        """Trigger emotions based on LSS and context.

        The key insight: Different emotions arise from the SAME
        prediction error, but in DIFFERENT contexts.

        Args:
            lss: Local Surprise Signal (prediction error)
            context: Environmental and state context

        Returns:
            Dict of emotion name → intensity
        """
        emotions = {}
        abs_lss = abs(lss)

        # Only trigger if surprise exceeds threshold
        if abs_lss < self.surprise_threshold:
            return emotions

        # === NEGATIVE SURPRISE (worse than expected) ===
        if lss < 0:
            # Fear: Negative surprise NEAR THREAT
            # "Something bad happened near danger" → heightened fear
            if context.near_threat:
                threat_proximity = 1.0 / (1.0 + context.threat_distance)
                fear = abs_lss * threat_proximity * self.fear_sensitivity
                emotions['fear'] = min(1.0, fear)

            # Anger/Frustration: Negative surprise WHEN BLOCKED
            # "I expected to make progress but was blocked" → frustration
            if context.was_blocked:
                goal_proximity = 1.0 / (1.0 + context.goal_distance)
                # Frustration intensifies with consecutive blocks
                block_factor = 1.0 + 0.2 * min(context.consecutive_blocks, 5)
                anger = abs_lss * goal_proximity * block_factor * self.anger_sensitivity
                emotions['anger'] = min(1.0, anger)

            # General disappointment (baseline negative emotion)
            emotions['disappointment'] = min(1.0, abs_lss * 0.5)

        # === POSITIVE SURPRISE (better than expected) ===
        if lss > 0:
            # Joy: Positive surprise anywhere
            joy = lss * self.joy_sensitivity
            emotions['joy'] = min(1.0, joy)

            # Relief: Positive surprise near threat (escaped danger)
            if context.near_threat:
                threat_proximity = 1.0 / (1.0 + context.threat_distance)
                relief = lss * threat_proximity * 0.5
                emotions['relief'] = min(1.0, relief)

        # === COUNTERFACTUAL EMOTIONS ===
        if context.counterfactual_available:
            # Regret: Could have done better
            cf_difference = context.foregone_reward - context.reward
            if cf_difference > self.surprise_threshold:
                regret = cf_difference * self.regret_sensitivity
                emotions['regret'] = min(1.0, regret)
            # Relief: Avoided worse outcome
            elif cf_difference < -self.surprise_threshold:
                emotions['relief'] = min(1.0, abs(cf_difference) * 0.5)

        # === UNCERTAINTY/AROUSAL ===
        # Large prediction error (either direction) = high uncertainty
        if abs_lss > self.surprise_threshold * 2:
            emotions['uncertainty'] = min(1.0, abs_lss / 2.0)

        return emotions

    def get_dominant_emotion(
        self,
        emotions: Dict[str, float]
    ) -> Optional[str]:
        """Get the most intense emotion.

        Args:
            emotions: Dict of emotion intensities

        Returns:
            Name of dominant emotion, or None if empty
        """
        if not emotions:
            return None
        return max(emotions.keys(), key=lambda k: emotions[k])

    def get_valence(self, emotions: Dict[str, float]) -> float:
        """Get overall emotional valence.

        Positive valence: joy, relief
        Negative valence: fear, anger, regret, disappointment

        Args:
            emotions: Dict of emotion intensities

        Returns:
            Valence in [-1, 1]
        """
        positive = emotions.get('joy', 0) + emotions.get('relief', 0)
        negative = (emotions.get('fear', 0) + emotions.get('anger', 0) +
                   emotions.get('regret', 0) + emotions.get('disappointment', 0))
        total = positive + negative
        if total == 0:
            return 0.0
        return (positive - negative) / total

    def get_arousal(self, emotions: Dict[str, float]) -> float:
        """Get overall emotional arousal.

        Arousal = total emotional intensity regardless of valence.

        Args:
            emotions: Dict of emotion intensities

        Returns:
            Arousal level (sum of all intensities)
        """
        return sum(emotions.values())


class AdaptiveLSSThreshold:
    """Adaptive surprise threshold based on recent experience.

    If environment is highly variable, raise threshold (less reactive).
    If environment is stable, lower threshold (more sensitive).
    """

    def __init__(
        self,
        initial_threshold: float = 0.1,
        min_threshold: float = 0.01,
        max_threshold: float = 1.0,
        adaptation_rate: float = 0.05
    ):
        """Initialize adaptive threshold.

        Args:
            initial_threshold: Starting threshold
            min_threshold: Floor for threshold
            max_threshold: Ceiling for threshold
            adaptation_rate: How fast threshold adapts
        """
        self.threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adaptation_rate = adaptation_rate

        # Track recent prediction errors
        self.recent_lss: list = []
        self.window_size = 50

    def update(self, lss: float):
        """Update threshold based on observed LSS.

        Args:
            lss: Observed Local Surprise Signal
        """
        self.recent_lss.append(abs(lss))
        if len(self.recent_lss) > self.window_size:
            self.recent_lss.pop(0)

        # Adapt threshold toward recent LSS variance
        if len(self.recent_lss) >= 10:
            target = np.std(self.recent_lss)
            self.threshold = (
                (1 - self.adaptation_rate) * self.threshold +
                self.adaptation_rate * target
            )
            self.threshold = np.clip(
                self.threshold,
                self.min_threshold,
                self.max_threshold
            )

    def get_threshold(self) -> float:
        """Return current threshold."""
        return self.threshold


# Example usage
if __name__ == "__main__":
    print("=== LSS Emotional Trigger Demo ===\n")

    trigger = LSSEmotionalTrigger()

    # Scenario 1: Negative surprise near threat
    print("Scenario 1: Negative surprise near threat")
    context = EmotionalContext(
        near_threat=True,
        threat_distance=1.5,
        reward=-1.0
    )
    lss = -0.5  # Worse than expected
    emotions = trigger.trigger_emotions(lss, context)
    print(f"  LSS: {lss}, Context: near_threat=True")
    print(f"  Emotions: {emotions}")
    print(f"  Dominant: {trigger.get_dominant_emotion(emotions)}")
    print(f"  Valence: {trigger.get_valence(emotions):.2f}")

    # Scenario 2: Negative surprise when blocked
    print("\nScenario 2: Negative surprise when blocked")
    context = EmotionalContext(
        was_blocked=True,
        consecutive_blocks=3,
        goal_distance=2.0,
        reward=0.0
    )
    lss = -0.3
    emotions = trigger.trigger_emotions(lss, context)
    print(f"  LSS: {lss}, Context: blocked, 3 consecutive")
    print(f"  Emotions: {emotions}")
    print(f"  Dominant: {trigger.get_dominant_emotion(emotions)}")

    # Scenario 3: Positive surprise (found reward)
    print("\nScenario 3: Positive surprise (found reward)")
    context = EmotionalContext(reward=1.0)
    lss = 0.8  # Much better than expected
    emotions = trigger.trigger_emotions(lss, context)
    print(f"  LSS: {lss}")
    print(f"  Emotions: {emotions}")
    print(f"  Valence: {trigger.get_valence(emotions):.2f}")

    # Scenario 4: Counterfactual regret
    print("\nScenario 4: Counterfactual regret")
    context = EmotionalContext(
        counterfactual_available=True,
        reward=0.2,
        foregone_reward=1.0  # Could have gotten 1.0
    )
    lss = 0.0  # No surprise about actual outcome
    emotions = trigger.trigger_emotions(lss, context)
    print(f"  Actual: 0.2, Foregone: 1.0")
    print(f"  Emotions: {emotions}")
