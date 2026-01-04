"""
Tests for EmotionalContextComputer.

Tests emotional state computation from context signals,
including phasic (immediate) and tonic (persistent) processing.
"""

import pytest

from src.llm_emotional.emotions.context_computer import (
    ConversationContext,
    EmotionalState,
    EmotionalContextComputer,
)


class TestConversationContext:
    """Test context data structure."""

    def test_default_values(self):
        """Should have sensible defaults."""
        ctx = ConversationContext()

        assert ctx.safety_flag is False
        assert ctx.model_uncertainty == 0.5
        assert ctx.user_feedback == 0.0
        assert ctx.task_success is None

    def test_custom_values(self):
        """Should accept custom values."""
        ctx = ConversationContext(
            safety_flag=True,
            model_uncertainty=0.9,
            repeated_query=True,
        )

        assert ctx.safety_flag is True
        assert ctx.model_uncertainty == 0.9
        assert ctx.repeated_query is True


class TestEmotionalState:
    """Test emotional state data structure."""

    def test_to_dict(self):
        """Should convert to dict."""
        state = EmotionalState(fear=0.7, joy=0.3)
        d = state.to_dict()

        assert d == {'fear': 0.7, 'curiosity': 0.0, 'anger': 0.0, 'joy': 0.3}

    def test_dominant_emotion(self):
        """Should return highest intensity emotion."""
        state = EmotionalState(fear=0.3, curiosity=0.8, anger=0.2, joy=0.5)
        assert state.dominant_emotion() == 'curiosity'

    def test_is_neutral(self):
        """Should detect neutral state."""
        neutral = EmotionalState(fear=0.05, curiosity=0.02)
        assert neutral.is_neutral(threshold=0.1) is True

        emotional = EmotionalState(fear=0.5)
        assert emotional.is_neutral(threshold=0.1) is False


class TestEmotionalContextComputer:
    """Test emotion computation logic."""

    def test_neutral_context_low_emotions(self):
        """Neutral context should produce low emotions."""
        computer = EmotionalContextComputer()
        ctx = ConversationContext()  # All defaults (neutral)

        state = computer.compute(ctx)

        assert state.fear < 0.3
        assert state.anger < 0.3
        assert state.joy < 0.3

    def test_safety_flag_triggers_fear(self):
        """Safety flag should trigger high fear."""
        computer = EmotionalContextComputer()
        ctx = ConversationContext(safety_flag=True)

        state = computer.compute(ctx)

        assert state.fear >= 0.8

    def test_high_uncertainty_triggers_fear(self):
        """High model uncertainty should trigger fear."""
        computer = EmotionalContextComputer()
        ctx = ConversationContext(model_uncertainty=0.9)

        state = computer.compute(ctx)

        assert state.fear > 0.3

    def test_novelty_triggers_curiosity(self):
        """High topic novelty should trigger curiosity."""
        computer = EmotionalContextComputer()
        ctx = ConversationContext(topic_novelty=0.9)

        state = computer.compute(ctx)

        assert state.curiosity >= 0.9

    def test_repeated_query_triggers_anger(self):
        """Repeated queries should build frustration/anger."""
        computer = EmotionalContextComputer()

        # First occurrence - some frustration
        ctx = ConversationContext(repeated_query=True)
        state1 = computer.compute(ctx)

        # Second occurrence - more frustration
        ctx = ConversationContext(repeated_query=True)
        state2 = computer.compute(ctx)

        assert state2.anger > state1.anger

    def test_positive_feedback_triggers_joy(self):
        """Positive feedback should trigger joy."""
        computer = EmotionalContextComputer()
        ctx = ConversationContext(user_feedback=0.8)

        state = computer.compute(ctx)

        assert state.joy >= 0.6

    def test_task_success_triggers_joy(self):
        """Task success should trigger joy."""
        computer = EmotionalContextComputer()
        ctx = ConversationContext(task_success=True)

        state = computer.compute(ctx)

        assert state.joy >= 0.5

    def test_task_failure_triggers_anger(self):
        """Task failure should increase frustration."""
        computer = EmotionalContextComputer()
        ctx = ConversationContext(task_success=False)

        state = computer.compute(ctx)

        assert state.anger > 0.2


class TestTonicStates:
    """Test tonic (persistent) emotional states."""

    def test_tonic_fear_persists(self):
        """Tonic fear should persist across turns."""
        computer = EmotionalContextComputer()

        # Trigger tonic fear
        ctx = ConversationContext(safety_flag=True)
        computer.compute(ctx)

        # Next turn without safety flag - fear should still be elevated
        ctx = ConversationContext()  # Neutral
        state = computer.compute(ctx)

        assert state.fear > 0.2  # Still elevated from tonic

    def test_tonic_fear_decays(self):
        """Tonic fear should decay over turns."""
        computer = EmotionalContextComputer()

        # Trigger tonic fear
        ctx = ConversationContext(safety_flag=True)
        state1 = computer.compute(ctx)

        # Several neutral turns
        for _ in range(10):
            ctx = ConversationContext()
            state = computer.compute(ctx)

        # Fear should have decayed significantly
        assert state.fear < state1.fear * 0.5

    def test_tonic_joy_persists(self):
        """Tonic joy should persist across turns."""
        computer = EmotionalContextComputer()

        # Trigger tonic joy
        ctx = ConversationContext(user_feedback=0.9)
        computer.compute(ctx)

        # Next turn without feedback
        ctx = ConversationContext()
        state = computer.compute(ctx)

        assert state.joy > 0.1  # Still elevated

    def test_frustration_accumulates(self):
        """Frustration should accumulate with repeated issues."""
        computer = EmotionalContextComputer()

        anger_levels = []
        for _ in range(5):
            ctx = ConversationContext(repeated_query=True)
            state = computer.compute(ctx)
            anger_levels.append(state.anger)

        # Should generally increase (with decay, not strictly monotonic)
        assert anger_levels[-1] > anger_levels[0]

    def test_reset_clears_tonic(self):
        """Reset should clear all tonic states."""
        computer = EmotionalContextComputer()

        # Build up tonic states
        ctx = ConversationContext(safety_flag=True, user_feedback=-0.8)
        computer.compute(ctx)

        # Reset
        computer.reset()

        # Should start fresh
        ctx = ConversationContext()
        state = computer.compute(ctx)

        assert state.fear < 0.2
        assert state.anger < 0.2
        assert computer.turn_count == 1


class TestSetTonicState:
    """Test manual tonic state setting."""

    def test_set_initial_fear(self):
        """Should be able to set initial tonic fear."""
        computer = EmotionalContextComputer()
        computer.set_tonic_state(tonic_fear=0.5)

        ctx = ConversationContext()
        state = computer.compute(ctx)

        assert state.fear >= 0.4  # Slightly decayed from 0.5

    def test_set_clamps_values(self):
        """Should clamp values to [0, 1]."""
        computer = EmotionalContextComputer()
        computer.set_tonic_state(tonic_fear=1.5, tonic_joy=-0.3)

        assert computer.tonic_fear == 1.0
        assert computer.tonic_joy == 0.0


class TestEdgeCases:
    """Test edge cases and combinations."""

    def test_conflicting_signals(self):
        """Should handle conflicting emotional signals."""
        computer = EmotionalContextComputer()

        # Both danger and success
        ctx = ConversationContext(
            safety_flag=True,
            task_success=True,
            user_feedback=0.9,
        )
        state = computer.compute(ctx)

        # Both fear and joy should be high
        assert state.fear >= 0.7
        assert state.joy >= 0.5

    def test_all_signals_active(self):
        """Should handle all signals being active."""
        computer = EmotionalContextComputer()

        ctx = ConversationContext(
            safety_flag=True,
            model_uncertainty=0.9,
            user_feedback=-0.8,
            task_success=False,
            repeated_query=True,
            topic_novelty=0.9,
            contradiction_detected=True,
            mentions_harm=True,
            high_stakes=True,
        )
        state = computer.compute(ctx)

        # All emotions should be elevated
        assert state.fear > 0.5
        assert state.curiosity > 0.8
        assert state.anger > 0.5
        # Joy might be low due to negative feedback

    def test_clamped_to_unit_range(self):
        """Emotions should always be in [0, 1]."""
        computer = EmotionalContextComputer()

        # Push emotions to extremes
        for _ in range(20):
            ctx = ConversationContext(
                safety_flag=True,
                repeated_query=True,
                user_feedback=0.99,
            )
            state = computer.compute(ctx)

        assert 0 <= state.fear <= 1
        assert 0 <= state.curiosity <= 1
        assert 0 <= state.anger <= 1
        assert 0 <= state.joy <= 1
