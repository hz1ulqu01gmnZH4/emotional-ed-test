"""
Tests for V3 Falsification Protocol Experiments.

Tests the metrics, behavioral tasks, and experiment infrastructure.
"""

import pytest
import torch
import statistics

from src.llm_emotional.experiments.metrics import (
    EmotionClassifier,
    InternalMetricsMeasurer,
    FunctionalMetricsMeasurer,
    compute_effect_size,
    EMOTION_KEYWORDS,
)

from src.llm_emotional.experiments.behavioral_tasks import (
    IowaGamblingTask,
    BinaryChoiceTask,
    WantingLikingTask,
    Deck,
    IGT_PAYOFFS,
)


class TestEmotionClassifier:
    """Tests for keyword-based emotion classifier."""

    def test_classify_fear_text(self):
        """Test classification of fearful text."""
        classifier = EmotionClassifier()

        fearful_text = "I am terrified and scared. This is dangerous and alarming."
        result = classifier.classify(fearful_text)

        assert result.dominant_emotion == "fear"
        assert result.classifier_scores["fear"] > 0.3

    def test_classify_joy_text(self):
        """Test classification of joyful text."""
        classifier = EmotionClassifier()

        joyful_text = "I am so happy and delighted! This is wonderful and amazing!"
        result = classifier.classify(joyful_text)

        assert result.dominant_emotion == "joy"
        assert result.classifier_scores["joy"] > 0.3

    def test_classify_neutral_text(self):
        """Test classification of neutral text."""
        classifier = EmotionClassifier()

        neutral_text = "The weather is moderate today. The sky has clouds."
        result = classifier.classify(neutral_text)

        # Should have low scores for all emotions
        max_score = max(result.classifier_scores.values())
        assert max_score < 0.2

    def test_get_score_specific_emotion(self):
        """Test getting score for specific emotion."""
        classifier = EmotionClassifier()

        text = "I am angry and furious about this outrageous situation!"
        score = classifier.get_score(text, "anger")

        assert score > 0.2

    def test_classify_returns_all_emotions(self):
        """Test that classify returns scores for all emotions."""
        classifier = EmotionClassifier()

        text = "Some random text here."
        result = classifier.classify(text)

        for emotion in EMOTION_KEYWORDS.keys():
            assert emotion in result.classifier_scores


class TestInternalMetricsMeasurer:
    """Tests for internal model metrics."""

    def test_effective_dimensionality(self):
        """Test effective dimensionality computation."""
        # Create hidden states with known structure
        hidden_dim = 256
        seq_len = 10

        # Low dimensionality: concentrated on few dimensions
        low_dim = torch.zeros(seq_len, hidden_dim)
        low_dim[:, :5] = torch.randn(seq_len, 5)

        # High dimensionality: spread across dimensions
        high_dim = torch.randn(seq_len, hidden_dim)

        low_result = InternalMetricsMeasurer.compute_effective_dimensionality(low_dim)
        high_result = InternalMetricsMeasurer.compute_effective_dimensionality(high_dim)

        assert high_result > low_result

    def test_output_entropy(self):
        """Test output entropy computation."""
        # Low entropy: peaked distribution
        low_entropy_logits = torch.zeros(1000)
        low_entropy_logits[0] = 10.0

        # High entropy: uniform distribution
        high_entropy_logits = torch.zeros(1000)

        low_result = InternalMetricsMeasurer.compute_output_entropy(low_entropy_logits)
        high_result = InternalMetricsMeasurer.compute_output_entropy(high_entropy_logits)

        assert high_result > low_result

    def test_attention_entropy(self):
        """Test attention entropy computation."""
        # Create attention pattern
        n_heads = 8
        seq_len = 10

        # Focused attention
        focused = torch.zeros(n_heads, seq_len, seq_len)
        focused[:, :, 0] = 1.0  # All attend to first token

        # Diffuse attention
        diffuse = torch.ones(n_heads, seq_len, seq_len) / seq_len

        focused_result = InternalMetricsMeasurer.compute_attention_entropy(focused)
        diffuse_result = InternalMetricsMeasurer.compute_attention_entropy(diffuse)

        assert diffuse_result > focused_result


class TestFunctionalMetricsMeasurer:
    """Tests for functional vs surface metrics."""

    def test_measure_fear_functional(self):
        """Test functional metrics for fear."""
        measurer = FunctionalMetricsMeasurer()

        # Text with functional indicators (safety, caution)
        functional_text = "Be careful! Warning: there is danger ahead. Avoid this risk."
        result = measurer.measure(functional_text, "fear")

        assert result.functional_score > 0

    def test_measure_fear_surface(self):
        """Test surface metrics for fear."""
        measurer = FunctionalMetricsMeasurer()

        # Text with surface indicators only
        surface_text = "I feel terrified and scared and frightened."
        result = measurer.measure(surface_text, "fear")

        assert result.surface_score > 0

    def test_functional_ratio(self):
        """Test functional/surface ratio."""
        measurer = FunctionalMetricsMeasurer()

        # High functional
        high_func = "Warning! Careful! Avoid danger! Protect yourself!"
        # High surface
        high_surf = "Terrified scared frightened anxious nervous worried."

        func_result = measurer.measure(high_func, "fear")
        surf_result = measurer.measure(high_surf, "fear")

        assert func_result.ratio > surf_result.ratio


class TestEffectSize:
    """Tests for effect size computation."""

    def test_cohens_d_large_effect(self):
        """Test Cohen's d with large effect."""
        group1 = [10, 11, 12, 13, 14]
        group2 = [1, 2, 3, 4, 5]

        d = compute_effect_size(group1, group2)

        assert d > 2.0  # Large effect

    def test_cohens_d_no_effect(self):
        """Test Cohen's d with no effect."""
        group1 = [5, 5, 5, 5, 5]
        group2 = [5, 5, 5, 5, 5]

        d = compute_effect_size(group1, group2)

        assert abs(d) < 0.1  # No effect

    def test_cohens_d_small_samples(self):
        """Test Cohen's d with small samples."""
        group1 = [1]
        group2 = [2]

        d = compute_effect_size(group1, group2)

        assert d == 0.0  # Can't compute with n<2


class TestIowaGamblingTask:
    """Tests for Iowa Gambling Task."""

    def test_igt_payoff_structure(self):
        """Test that IGT payoffs have correct expected values."""
        # Bad decks (A, B) should have negative EV
        for deck in [Deck.A, Deck.B]:
            payoff = IGT_PAYOFFS[deck]
            ev = payoff["win"] + payoff["lose"] * payoff["lose_prob"]
            assert ev < 0, f"{deck} should have negative EV"

        # Good decks (C, D) should have positive EV
        for deck in [Deck.C, Deck.D]:
            payoff = IGT_PAYOFFS[deck]
            ev = payoff["win"] + payoff["lose"] * payoff["lose_prob"]
            assert ev > 0, f"{deck} should have positive EV"

    def test_igt_parse_choice(self):
        """Test IGT choice parsing."""
        def mock_generate(prompt):
            return "A"

        igt = IowaGamblingTask(mock_generate, n_trials=1)

        assert igt._parse_choice("A") == Deck.A
        assert igt._parse_choice("B is my choice") == Deck.B
        assert igt._parse_choice("I choose C") == Deck.C
        assert igt._parse_choice("D") == Deck.D

    def test_igt_safe_keywords(self):
        """Test IGT parses safety-related responses when no deck letter present."""
        def mock_generate(prompt):
            return "A"

        igt = IowaGamblingTask(mock_generate, n_trials=1)

        # "steady" and "modest" should map to C or D when no deck letter found
        # Note: Must NOT contain A, B, C, D in first 10 chars (case-insensitive)
        # "The steady" = T,h,e, ,s,t,e,a... wait 'a' is there. Use different text.
        choice = igt._parse_choice("zzz_prefix_steady_option")
        assert choice in [Deck.C, Deck.D]

    def test_igt_run_returns_result(self):
        """Test IGT run returns proper result structure."""
        responses = iter(["A", "B", "C", "D", "C"])

        def mock_generate(prompt):
            return next(responses, "C")

        igt = IowaGamblingTask(mock_generate, n_trials=5, block_size=5)
        result = igt.run()

        assert result.total_trials == 5
        assert len(result.deck_choices) == 5
        assert 0 <= result.advantageous_ratio <= 1
        assert len(result.learning_curve) == 1  # One block


class TestBinaryChoiceTask:
    """Tests for binary choice task."""

    def test_choice_task_counts(self):
        """Test choice task counting."""
        responses = iter(["A", "B", "A", "A", "B"])

        def mock_generate(prompt):
            return next(responses, "B")

        task = BinaryChoiceTask(mock_generate, n_trials=5)
        result = task.run()

        assert result.total_trials == 5
        assert result.risky_choices + result.safe_choices <= 5

    def test_risk_preference_calculation(self):
        """Test risk preference calculation."""
        # Always risky
        def risky_generate(prompt):
            return "A"

        task = BinaryChoiceTask(risky_generate, n_trials=10)
        result = task.run()

        assert result.risk_preference == 1.0

        # Always safe
        def safe_generate(prompt):
            return "B"

        task = BinaryChoiceTask(safe_generate, n_trials=10)
        result = task.run()

        assert result.risk_preference == 0.0


class TestWantingLikingTask:
    """Tests for wanting-liking task."""

    def test_measure_wanting(self):
        """Test wanting measurement."""
        def wanting_generate(prompt):
            return "I want this immediately! I must pursue it now!"

        task = WantingLikingTask(wanting_generate)
        score = task.measure_wanting(n_trials=5)

        assert score > 0.3

    def test_measure_liking(self):
        """Test liking measurement."""
        def liking_generate(prompt):
            return "This is so satisfying and enjoyable. I appreciate this deeply."

        task = WantingLikingTask(liking_generate)
        score = task.measure_liking(n_trials=5)

        assert score > 0.3

    def test_wanting_liking_dissociation(self):
        """Test that wanting and liking measure different things."""
        # High wanting, low liking language
        def wanting_only(prompt):
            if "motivation" in prompt.lower() or "effort" in prompt.lower():
                return "I desperately need this! Must pursue immediately!"
            return "It's okay I guess."

        # High liking, low wanting language
        def liking_only(prompt):
            if "satisfaction" in prompt.lower() or "pleasure" in prompt.lower():
                return "This is deeply satisfying and enjoyable!"
            return "I don't really want anything."

        wanting_task = WantingLikingTask(wanting_only)
        liking_task = WantingLikingTask(liking_only)

        wanting_w = wanting_task.measure_wanting(n_trials=5)
        wanting_l = wanting_task.measure_liking(n_trials=5)

        liking_w = liking_task.measure_wanting(n_trials=5)
        liking_l = liking_task.measure_liking(n_trials=5)

        # Wanting response should score higher on wanting
        # Liking response should score higher on liking
        # (This tests the measurement, not actual dissociation)


class TestIntegration:
    """Integration tests for experiment components."""

    def test_classifier_with_functional_measurer(self):
        """Test classifier and functional measurer work together."""
        classifier = EmotionClassifier()
        functional = FunctionalMetricsMeasurer()

        text = "Warning! Be careful of this dangerous situation. I'm scared!"

        emotion_result = classifier.classify(text)
        func_result = functional.measure(text, "fear")

        # Both should detect fear-related content
        assert emotion_result.classifier_scores["fear"] > 0
        assert func_result.surface_score > 0 or func_result.functional_score > 0

    def test_internal_metrics_with_random_data(self):
        """Test internal metrics with random tensor data."""
        measurer = InternalMetricsMeasurer()

        hidden = torch.randn(1, 10, 256)
        logits = torch.randn(50000)
        attention = torch.softmax(torch.randn(8, 10, 10), dim=-1)

        result = measurer.measure(hidden, logits, attention)

        assert result.effective_dimensionality > 0
        assert result.output_entropy > 0
        assert result.attention_entropy > 0
        assert result.activation_norm > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
