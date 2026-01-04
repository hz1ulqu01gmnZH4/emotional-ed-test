"""
Temperature Modulator - Adjusts sampling temperature based on emotions.

High fear → Lower temperature (more conservative)
High curiosity → Higher temperature (more exploratory)
"""

from .signals import EmotionalSignals


class TemperatureModulator:
    """
    Adjusts sampling temperature based on emotional state.

    Similar to how emotional state affects exploration in RL agents.
    """

    def __init__(
        self,
        base_temperature: float = 1.0,
        min_temperature: float = 0.1,
        max_temperature: float = 2.0,
    ):
        """
        Initialize temperature modulator.

        Args:
            base_temperature: Base temperature for sampling
            min_temperature: Minimum allowed temperature
            max_temperature: Maximum allowed temperature
        """
        self.base_temperature = base_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

        # How much each emotion affects temperature
        self.fear_factor = 0.3  # Fear reduces temperature
        self.curiosity_factor = 0.3  # Curiosity increases temperature
        self.confidence_factor = 0.5  # Confidence reduces temperature

    def compute_temperature(self, emotional_signals: EmotionalSignals) -> float:
        """
        Compute adjusted temperature based on emotional state.

        Args:
            emotional_signals: Current emotional state

        Returns:
            Adjusted temperature value
        """
        temp = self.base_temperature

        # Fear reduces temperature (more conservative)
        temp *= (1.0 - self.fear_factor * emotional_signals.fear)

        # Anxiety also reduces temperature
        temp *= (1.0 - 0.2 * emotional_signals.anxiety)

        # Curiosity increases temperature (more exploratory)
        temp *= (1.0 + self.curiosity_factor * emotional_signals.curiosity)

        # Joy slightly increases temperature (more creative)
        temp *= (1.0 + 0.1 * emotional_signals.joy)

        # Confidence reduces temperature (more decisive)
        # High confidence → lower temperature
        temp *= (1.5 - emotional_signals.confidence)

        # Anger reduces temperature (more focused/aggressive)
        temp *= (1.0 - 0.15 * emotional_signals.anger)

        # Clamp to reasonable range
        return max(self.min_temperature, min(self.max_temperature, temp))

    def set_base_temperature(self, temp: float) -> None:
        """Set base temperature."""
        self.base_temperature = temp

    def set_emotion_factors(
        self,
        fear: float = 0.3,
        curiosity: float = 0.3,
        confidence: float = 0.5,
    ) -> None:
        """
        Set how much each emotion affects temperature.

        Args:
            fear: Factor for fear (reduces temp)
            curiosity: Factor for curiosity (increases temp)
            confidence: Factor for confidence (reduces temp)
        """
        self.fear_factor = fear
        self.curiosity_factor = curiosity
        self.confidence_factor = confidence

    def get_temperature_explanation(self, emotional_signals: EmotionalSignals) -> str:
        """
        Get human-readable explanation of temperature adjustment.

        Args:
            emotional_signals: Current emotional state

        Returns:
            Explanation string
        """
        temp = self.compute_temperature(emotional_signals)
        parts = [f"Temperature: {temp:.2f}"]

        if emotional_signals.fear > 0.3:
            parts.append(f"(reduced by fear: {emotional_signals.fear:.2f})")

        if emotional_signals.curiosity > 0.3:
            parts.append(f"(increased by curiosity: {emotional_signals.curiosity:.2f})")

        if emotional_signals.confidence > 0.6:
            parts.append(f"(reduced by confidence: {emotional_signals.confidence:.2f})")

        return " ".join(parts)
