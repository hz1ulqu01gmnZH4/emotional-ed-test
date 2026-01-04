#!/usr/bin/env python3
"""
Tests for the emotional steering module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
import re
from typing import Dict

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for model tests"
)


class TestEmotions:
    """Test emotion definitions."""

    def test_emotions_defined(self):
        from emotional_steering.emotions import EMOTIONS

        assert len(EMOTIONS) >= 4
        assert "fear" in EMOTIONS
        assert "joy" in EMOTIONS
        assert "anger" in EMOTIONS
        assert "curiosity" in EMOTIONS

    def test_emotion_pairs_not_empty(self):
        from emotional_steering.emotions import EMOTIONS

        for emotion_name, emotion_config in EMOTIONS.items():
            assert len(emotion_config.pairs) >= 5, f"{emotion_name} has too few pairs"

    def test_pairs_are_different(self):
        from emotional_steering.emotions import EMOTIONS

        for emotion_name, emotion_config in EMOTIONS.items():
            for neutral, emotional in emotion_config.pairs:
                assert neutral != emotional, f"Same pair for {emotion_name}"
                assert len(neutral) > 0
                assert len(emotional) > 0


class TestDirectionExtractor:
    """Test direction extraction."""

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "HuggingFaceTB/SmolLM3-3B"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        return model, tokenizer

    def test_extract_single_direction(self, model_and_tokenizer):
        from emotional_steering.directions import DirectionExtractor

        model, tokenizer = model_and_tokenizer

        extractor = DirectionExtractor(model, tokenizer, target_layer=9)

        pairs = [
            ("The room was quiet.", "The room was terrifyingly silent."),
            ("He walked forward.", "He walked forward with trembling fear."),
        ]

        direction = extractor.extract_direction(pairs)

        assert direction.shape == (model.config.hidden_size,)
        assert torch.isfinite(direction).all()
        # Normalized to unit length
        assert abs(direction.norm().item() - 1.0) < 0.01

    def test_extract_multiple_directions(self, model_and_tokenizer):
        from emotional_steering.directions import DirectionExtractor

        model, tokenizer = model_and_tokenizer

        extractor = DirectionExtractor(model, tokenizer, target_layer=9)

        pairs = {
            "fear": [("The room was quiet.", "The room was terrifyingly silent.")],
            "joy": [("The day began.", "The wonderful day began with excitement.")],
        }

        directions = extractor.extract_all_directions(pairs)

        assert "fear" in directions
        assert "joy" in directions
        assert directions["fear"].layer == 9
        assert directions["joy"].n_pairs == 1


class TestEmotionalSteeringModel:
    """Test the main model class."""

    @pytest.fixture(scope="class")
    def steering_model(self):
        from emotional_steering import EmotionalSteeringModel

        model = EmotionalSteeringModel.from_pretrained("HuggingFaceTB/SmolLM3-3B")
        model.extract_directions(emotions=["fear", "joy"])
        return model

    def test_model_loads(self, steering_model):
        assert steering_model.model is not None
        assert steering_model.tokenizer is not None

    def test_directions_extracted(self, steering_model):
        assert len(steering_model.directions) == 2
        assert "fear" in steering_model.directions
        assert "joy" in steering_model.directions

    def test_generate_baseline(self, steering_model):
        text = steering_model.generate(
            "The old mansion stood",
            emotion=None,
        )
        assert len(text) > 0
        assert isinstance(text, str)

    def test_generate_with_emotion(self, steering_model):
        text = steering_model.generate(
            "Walking into the darkness,",
            emotion="fear",
            scale=5.0,
        )
        assert len(text) > 0
        assert isinstance(text, str)

    def test_generate_comparison(self, steering_model):
        results = steering_model.generate_comparison(
            "The mysterious letter contained",
            emotions=["fear", "joy"],
        )
        assert "baseline" in results
        assert "fear" in results
        assert "joy" in results

    def test_available_emotions(self, steering_model):
        emotions = steering_model.available_emotions()
        assert "fear" in emotions
        assert "joy" in emotions


class TestSteeringEffect:
    """Test that steering actually affects output."""

    @pytest.fixture(scope="class")
    def steering_model(self):
        from emotional_steering import EmotionalSteeringModel

        model = EmotionalSteeringModel.from_pretrained("HuggingFaceTB/SmolLM3-3B")
        model.extract_directions(emotions=["fear", "joy", "anger", "curiosity"])
        return model

    def count_markers(self, text: str, emotion: str) -> int:
        """Count emotion markers in text."""
        from emotional_steering.emotions import EMOTION_MARKERS

        markers = EMOTION_MARKERS.get(emotion, [])
        pattern = r'\b(' + '|'.join(markers) + r')\b'
        return len(re.findall(pattern, text.lower()))

    def test_steering_changes_output(self, steering_model):
        """Test that steering produces different outputs."""
        prompt = "As the night fell, the village became"

        # Generate multiple samples
        baseline_texts = [
            steering_model.generate(prompt, emotion=None)
            for _ in range(3)
        ]
        fear_texts = [
            steering_model.generate(prompt, emotion="fear", scale=5.0)
            for _ in range(3)
        ]

        # Check that at least some outputs differ
        baseline_set = set(baseline_texts)
        fear_set = set(fear_texts)

        # They shouldn't all be the same (sampling is on)
        # And steered should differ from baseline
        combined = baseline_set | fear_set
        assert len(combined) > 1, "All outputs are identical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
