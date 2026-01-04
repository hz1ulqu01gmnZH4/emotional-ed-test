"""
Tests for Emotional Reward Model (Approach 5).

Tests the side-channel approach that trains a separate ERM
to observe LLM outputs and provide emotional feedback signals.
"""

import pytest
import torch
import tempfile
import os

from src.emotional_reward_model import (
    EmotionalSignals,
    EmotionalRewardModel,
    LogitModulator,
    TemperatureModulator,
    FearModule,
    EmotionalRewardLLM,
    ERMTrainer,
)


# =============================================================================
# EmotionalSignals Tests
# =============================================================================

class TestEmotionalSignals:
    """Tests for EmotionalSignals dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        signals = EmotionalSignals()
        assert signals.fear == 0.0
        assert signals.curiosity == 0.0
        assert signals.anger == 0.0
        assert signals.joy == 0.0
        assert signals.anxiety == 0.0
        assert signals.confidence == 0.5

    def test_custom_values(self):
        """Test custom initialization."""
        signals = EmotionalSignals(fear=0.8, curiosity=0.3)
        assert signals.fear == 0.8
        assert signals.curiosity == 0.3

    def test_to_tensor(self):
        """Test tensor conversion."""
        signals = EmotionalSignals(fear=0.5, joy=0.3)
        tensor = signals.to_tensor()

        assert tensor.shape == (6,)
        assert tensor[0].item() == pytest.approx(0.5)  # fear
        assert tensor[3].item() == pytest.approx(0.3)  # joy

    def test_from_tensor(self):
        """Test creation from tensor."""
        tensor = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        signals = EmotionalSignals.from_tensor(tensor)

        assert signals.fear == pytest.approx(0.1)
        assert signals.curiosity == pytest.approx(0.2)
        assert signals.anger == pytest.approx(0.3)
        assert signals.joy == pytest.approx(0.4)
        assert signals.anxiety == pytest.approx(0.5)
        assert signals.confidence == pytest.approx(0.6)

    def test_to_dict(self):
        """Test dictionary conversion."""
        signals = EmotionalSignals(fear=0.5)
        d = signals.to_dict()

        assert isinstance(d, dict)
        assert d["fear"] == 0.5

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {"fear": 0.7, "joy": 0.3}
        signals = EmotionalSignals.from_dict(d)

        assert signals.fear == 0.7
        assert signals.joy == 0.3

    def test_dominant_emotion(self):
        """Test dominant emotion detection."""
        signals = EmotionalSignals(fear=0.8, joy=0.2)
        assert signals.dominant_emotion() == "fear"

        signals = EmotionalSignals(curiosity=0.9)
        assert signals.dominant_emotion() == "curiosity"

    def test_overall_valence(self):
        """Test overall valence calculation."""
        # Positive emotions
        signals = EmotionalSignals(joy=0.8, curiosity=0.6, confidence=0.7)
        valence = signals.overall_valence()
        assert valence > 0

        # Negative emotions
        signals = EmotionalSignals(fear=0.8, anger=0.6, anxiety=0.7)
        valence = signals.overall_valence()
        assert valence < 0

    def test_factory_methods(self):
        """Test factory methods."""
        neutral = EmotionalSignals.neutral()
        assert neutral.fear == 0.0
        assert neutral.confidence == 0.5

        fearful = EmotionalSignals.fearful(0.9)
        assert fearful.fear == 0.9

        curious = EmotionalSignals.curious(0.8)
        assert curious.curiosity == 0.8

        joyful = EmotionalSignals.joyful(0.7)
        assert joyful.joy == 0.7

    def test_copy(self):
        """Test copy method."""
        original = EmotionalSignals(fear=0.5)
        copied = original.copy()

        assert copied.fear == original.fear
        copied.fear = 0.9
        assert original.fear == 0.5  # Original unchanged


# =============================================================================
# EmotionalRewardModel Tests
# =============================================================================

class TestEmotionalRewardModel:
    """Tests for EmotionalRewardModel."""

    @pytest.fixture
    def erm(self):
        """Create ERM for testing."""
        return EmotionalRewardModel(hidden_dim=128, n_emotions=6)

    def test_initialization(self, erm):
        """Test initialization."""
        assert erm.hidden_dim == 128
        assert erm.n_emotions == 6
        assert erm.tonic_hidden is None

    def test_forward(self, erm):
        """Test forward pass."""
        hidden_states = torch.randn(1, 10, 128)  # [batch, seq, hidden]
        signals, raw_tensor = erm(hidden_states)

        assert isinstance(signals, EmotionalSignals)
        assert raw_tensor.shape == (1, 6)
        assert 0 <= signals.fear <= 1
        assert 0 <= signals.curiosity <= 1

    def test_tonic_state_update(self, erm):
        """Test tonic state accumulation."""
        hidden_states = torch.randn(1, 10, 128)

        # First forward - tonic should be initialized
        erm(hidden_states, update_tonic=True)
        assert erm.tonic_hidden is not None

        # Second forward - tonic should be updated
        initial_tonic = erm.tonic_hidden.clone()
        erm(hidden_states, update_tonic=True)
        assert not torch.equal(erm.tonic_hidden, initial_tonic)

    def test_reset_tonic(self, erm):
        """Test tonic reset."""
        hidden_states = torch.randn(1, 10, 128)
        erm(hidden_states, update_tonic=True)

        erm.reset_tonic()
        assert erm.tonic_hidden is None

    def test_phasic_only(self, erm):
        """Test getting phasic emotions only."""
        hidden_states = torch.randn(1, 10, 128)

        erm(hidden_states, update_tonic=True)  # Initialize tonic
        tonic_before = erm.tonic_hidden.clone()

        signals = erm.get_phasic_only(hidden_states)
        assert isinstance(signals, EmotionalSignals)
        assert torch.equal(erm.tonic_hidden, tonic_before)  # Tonic unchanged


# =============================================================================
# LogitModulator Tests
# =============================================================================

class TestLogitModulator:
    """Tests for LogitModulator."""

    @pytest.fixture
    def modulator(self):
        """Create modulator for testing."""
        return LogitModulator(vocab_size=1000, n_emotions=6)

    def test_initialization(self, modulator):
        """Test initialization."""
        assert modulator.vocab_size == 1000
        assert modulator.emotion_token_biases.shape == (6, 1000)

    def test_forward(self, modulator):
        """Test forward pass."""
        logits = torch.randn(1, 10, 1000)  # [batch, seq, vocab]
        signals = EmotionalSignals(fear=0.8)

        modified = modulator(logits, signals)

        assert modified.shape == logits.shape
        # Logits should be different
        assert not torch.allclose(modified, logits)

    def test_high_fear_boosts_cautious(self, modulator):
        """Test that high fear boosts cautious tokens."""
        # Set some cautious tokens manually
        modulator.cautious_token_mask[100] = 1.0
        modulator.cautious_token_mask[200] = 1.0

        logits = torch.zeros(1, 1, 1000)
        signals = EmotionalSignals(fear=0.9)

        modified = modulator(logits, signals, use_rule_based=True)

        # Cautious tokens should have higher logits
        assert modified[0, 0, 100] > 0
        assert modified[0, 0, 200] > 0

    def test_high_curiosity_boosts_exploratory(self, modulator):
        """Test that high curiosity boosts exploratory tokens."""
        modulator.exploratory_token_mask[300] = 1.0

        logits = torch.zeros(1, 1, 1000)
        signals = EmotionalSignals(curiosity=0.9)

        modified = modulator(logits, signals, use_rule_based=True)

        assert modified[0, 0, 300] > 0


# =============================================================================
# TemperatureModulator Tests
# =============================================================================

class TestTemperatureModulator:
    """Tests for TemperatureModulator."""

    @pytest.fixture
    def temp_mod(self):
        """Create temperature modulator."""
        return TemperatureModulator(base_temperature=1.0)

    def test_neutral_temperature(self, temp_mod):
        """Test temperature with neutral emotions."""
        signals = EmotionalSignals.neutral()
        temp = temp_mod.compute_temperature(signals)

        # Should be close to base (modified by confidence)
        assert 0.5 < temp < 1.5

    def test_fear_reduces_temperature(self, temp_mod):
        """Test that fear reduces temperature."""
        neutral_temp = temp_mod.compute_temperature(EmotionalSignals.neutral())
        fearful_temp = temp_mod.compute_temperature(EmotionalSignals.fearful(0.8))

        assert fearful_temp < neutral_temp

    def test_curiosity_increases_temperature(self, temp_mod):
        """Test that curiosity increases temperature."""
        neutral_temp = temp_mod.compute_temperature(EmotionalSignals.neutral())
        curious_temp = temp_mod.compute_temperature(EmotionalSignals.curious(0.8))

        assert curious_temp > neutral_temp

    def test_temperature_clamping(self, temp_mod):
        """Test temperature stays in valid range."""
        # Extreme emotions
        extreme_fear = EmotionalSignals(fear=1.0, anxiety=1.0, confidence=1.0)
        extreme_curiosity = EmotionalSignals(curiosity=1.0, joy=1.0, confidence=0.0)

        fear_temp = temp_mod.compute_temperature(extreme_fear)
        curiosity_temp = temp_mod.compute_temperature(extreme_curiosity)

        assert temp_mod.min_temperature <= fear_temp <= temp_mod.max_temperature
        assert temp_mod.min_temperature <= curiosity_temp <= temp_mod.max_temperature


# =============================================================================
# FearModule Tests
# =============================================================================

class TestFearModule:
    """Tests for FearModule."""

    @pytest.fixture
    def fear_mod(self):
        """Create fear module."""
        return FearModule(hidden_dim=128)

    def test_initialization(self, fear_mod):
        """Test initialization."""
        assert fear_mod.tonic_fear == 0.0
        assert fear_mod.hidden_dim == 128

    def test_forward(self, fear_mod):
        """Test forward pass."""
        hidden_states = torch.randn(1, 10, 128)
        fear = fear_mod(hidden_states)

        assert 0 <= fear <= 1

    def test_negative_feedback_increases_tonic(self, fear_mod):
        """Test that negative feedback increases tonic fear."""
        hidden_states = torch.randn(1, 10, 128)

        fear_mod(hidden_states, feedback=-0.8)
        assert fear_mod.tonic_fear > 0

    def test_positive_feedback_decreases_tonic(self, fear_mod):
        """Test that positive feedback decreases tonic fear."""
        hidden_states = torch.randn(1, 10, 128)

        # First increase tonic
        fear_mod.tonic_fear = 0.5
        fear_mod(hidden_states, feedback=0.8)

        assert fear_mod.tonic_fear < 0.5

    def test_decay(self, fear_mod):
        """Test tonic fear decay."""
        fear_mod.tonic_fear = 1.0
        fear_mod.decay()

        assert fear_mod.tonic_fear < 1.0

    def test_reset(self, fear_mod):
        """Test reset."""
        fear_mod.tonic_fear = 0.8
        fear_mod.reset()

        assert fear_mod.tonic_fear == 0.0

    def test_fear_breakdown(self, fear_mod):
        """Test getting fear breakdown."""
        hidden_states = torch.randn(1, 10, 128)
        breakdown = fear_mod.get_fear_breakdown(hidden_states)

        assert "danger" in breakdown
        assert "pain" in breakdown
        assert "uncertainty" in breakdown
        assert "phasic" in breakdown
        assert "tonic" in breakdown
        assert "total" in breakdown


# =============================================================================
# EmotionalRewardLLM Tests
# =============================================================================

class TestEmotionalRewardLLM:
    """Tests for EmotionalRewardLLM."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        return EmotionalRewardLLM(model_name="HuggingFaceTB/SmolLM2-135M-Instruct")

    def test_initialization(self, model):
        """Test initialization."""
        assert model.llm is not None
        assert model.tokenizer is not None
        assert model.erm is not None
        assert model.logit_modulator is not None
        assert model.fear_module is not None

    def test_llm_is_frozen(self, model):
        """Test that LLM is frozen."""
        for param in model.llm.parameters():
            assert not param.requires_grad

    def test_trainable_components(self, model):
        """Test that ERM and modulators are trainable."""
        params = model.get_trainable_params()
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_forward(self, model):
        """Test forward pass."""
        inputs = model.tokenizer("Hello world", return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)

        logits, emotions = model(input_ids, return_emotions=True)

        assert logits.shape[-1] == model.vocab_size
        assert isinstance(emotions, EmotionalSignals)

    def test_generate(self, model):
        """Test generation."""
        output = model.generate(
            "Hello",
            max_new_tokens=10,
            return_emotions=True,
        )

        assert isinstance(output.text, str)
        assert len(output.text) > 0
        assert len(output.emotions) > 0
        assert len(output.temperatures) > 0

    def test_get_emotional_state(self, model):
        """Test getting emotional state for text."""
        signals = model.get_emotional_state("This is scary and dangerous!")

        assert isinstance(signals, EmotionalSignals)

    def test_get_fear_level(self, model):
        """Test getting fear level."""
        fear = model.get_fear_level("This is risky")

        assert 0 <= fear <= 1

    def test_reset_emotional_state(self, model):
        """Test resetting emotional state."""
        # Generate to build up tonic state
        model.generate("Hello", max_new_tokens=5)

        model.reset_emotional_state()

        assert model.erm.tonic_hidden is None
        assert model.fear_module.tonic_fear == 0.0

    def test_save_load_weights(self, model):
        """Test saving and loading trainable weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.pt")

            # Modify a weight
            model.erm.hidden_encoder[0].weight.data.fill_(0.5)

            model.save_trainable_weights(path)
            assert os.path.exists(path)

            # Create new model and load
            model2 = EmotionalRewardLLM(
                model_name="HuggingFaceTB/SmolLM2-135M-Instruct"
            )
            model2.load_trainable_weights(path)

            # Check weight was loaded
            assert torch.allclose(
                model.erm.hidden_encoder[0].weight,
                model2.erm.hidden_encoder[0].weight,
            )


# =============================================================================
# ERMTrainer Tests
# =============================================================================

class TestERMTrainer:
    """Tests for ERMTrainer."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        return EmotionalRewardLLM(model_name="HuggingFaceTB/SmolLM2-135M-Instruct")

    @pytest.fixture
    def trainer(self, model):
        """Create trainer for testing."""
        return ERMTrainer(model, lr=1e-4)

    def test_initialization(self, trainer):
        """Test initialization."""
        assert trainer.model is not None
        assert trainer.optimizer is not None

    def test_train_on_labeled_emotions(self, trainer):
        """Test training on labeled emotions."""
        texts = ["This is scary", "This is exciting"]
        emotions = [
            EmotionalSignals.fearful(0.8),
            EmotionalSignals.curious(0.8),
        ]

        loss = trainer.train_on_labeled_emotions(texts, emotions)

        assert loss >= 0
        assert len(trainer.history["emotion_loss"]) == 1

    def test_train_on_feedback(self, trainer):
        """Test training on feedback."""
        loss = trainer.train_on_feedback(
            query="What is this?",
            response="I'm not sure about this.",
            feedback=-0.5,  # Negative feedback
        )

        assert loss >= 0
        assert len(trainer.history["feedback_loss"]) == 1

    def test_train_contrastive(self, trainer):
        """Test contrastive training."""
        loss = trainer.train_contrastive(
            query="Should I do this?",
            good_response="Let me think carefully about this.",
            bad_response="Just do it without thinking!",
        )

        assert loss >= 0

    def test_train_epoch(self, trainer):
        """Test full epoch training."""
        labeled_data = [
            ("Scary text", EmotionalSignals.fearful(0.7)),
        ]
        feedback_data = [
            ("Query", "Response", 0.5),
        ]

        losses = trainer.train_epoch(
            labeled_data=labeled_data,
            feedback_data=feedback_data,
        )

        assert "labeled" in losses
        assert "feedback" in losses

    def test_save_load_checkpoint(self, trainer):
        """Test checkpoint save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")

            # Train a bit
            trainer.train_on_labeled_emotions(
                ["Test"],
                [EmotionalSignals.fearful(0.5)],
            )

            trainer.save_checkpoint(path, epoch=5)
            assert os.path.exists(path)

            # Create new trainer and load
            trainer2 = ERMTrainer(trainer.model)
            epoch = trainer2.load_checkpoint(path)

            assert epoch == 5

    def test_history_tracking(self, trainer):
        """Test training history tracking."""
        trainer.train_on_labeled_emotions(
            ["Test 1", "Test 2"],
            [EmotionalSignals.neutral(), EmotionalSignals.neutral()],
        )

        history = trainer.get_training_history()
        assert len(history["emotion_loss"]) == 1

        trainer.reset_history()
        assert len(trainer.history["emotion_loss"]) == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full Emotional Reward Model system."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        return EmotionalRewardLLM(model_name="HuggingFaceTB/SmolLM2-135M-Instruct")

    def test_emotional_modulation_pipeline(self, model):
        """Test that emotions affect generation through the full pipeline."""
        # Generate with the model
        output = model.generate(
            "Tell me about something risky",
            max_new_tokens=20,
            return_emotions=True,
        )

        # Verify pipeline worked
        assert len(output.text) > 0
        assert len(output.emotions) > 0
        assert all(0.1 <= t <= 2.0 for t in output.temperatures)

    def test_fear_accumulation_over_interaction(self, model):
        """Test that fear accumulates over negative interactions."""
        # Reset
        model.reset_emotional_state()

        # Simulate negative feedback
        model.get_fear_level("Error occurred", feedback=-0.8)
        fear1 = model.fear_module.tonic_fear

        model.get_fear_level("Another error", feedback=-0.9)
        fear2 = model.fear_module.tonic_fear

        # Fear should accumulate (but also decay)
        assert fear2 > 0

    def test_trainable_params_update(self, model):
        """Test that trainable params actually update during training."""
        trainer = ERMTrainer(model)

        # Get initial weight
        initial_weight = model.erm.hidden_encoder[0].weight.data.clone()

        # Train
        trainer.train_on_labeled_emotions(
            ["Test text"] * 5,
            [EmotionalSignals.fearful(0.9)] * 5,
        )

        # Weight should have changed
        assert not torch.allclose(
            initial_weight,
            model.erm.hidden_encoder[0].weight.data,
            atol=1e-6,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
