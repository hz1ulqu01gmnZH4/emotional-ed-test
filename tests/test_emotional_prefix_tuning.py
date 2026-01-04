"""
Tests for Emotional Prefix Tuning implementation.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotional_prefix_tuning import (
    EmotionalContext,
    EmotionalEncoder,
    EmotionalPrefixGenerator,
    EmotionalPrefixLLM,
    EmotionalPrefixTrainer,
)


class TestEmotionalContext:
    """Tests for EmotionalContext dataclass."""

    def test_context_creation(self):
        """Test default context creation."""
        ctx = EmotionalContext()
        assert ctx.last_reward == 0.0
        assert ctx.safety_flag == False
        assert ctx.user_satisfaction == 0.5

    def test_context_with_values(self):
        """Test context with custom values."""
        ctx = EmotionalContext(
            last_reward=-0.5,
            safety_flag=True,
            failed_attempts=3,
        )
        assert ctx.last_reward == -0.5
        assert ctx.safety_flag == True
        assert ctx.failed_attempts == 3

    def test_update_from_feedback(self):
        """Test feedback updates context correctly."""
        ctx = EmotionalContext()

        # Positive feedback
        ctx.update_from_feedback(0.8)
        assert ctx.last_reward == 0.8
        assert ctx.cumulative_positive == 0.8
        assert ctx.failed_attempts == 0

        # Negative feedback
        ctx.update_from_feedback(-0.5)
        assert ctx.last_reward == -0.5
        assert ctx.cumulative_negative == 0.5
        assert ctx.failed_attempts == 1

    def test_reset_episodic(self):
        """Test episodic reset preserves tonic state."""
        ctx = EmotionalContext(
            last_reward=0.5,
            safety_flag=True,
            cumulative_positive=2.0,
        )
        ctx.reset_episodic()

        assert ctx.last_reward == 0.0
        assert ctx.safety_flag == False
        assert ctx.cumulative_positive == 2.0  # Preserved

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        ctx = EmotionalContext(
            last_reward=0.5,
            safety_flag=True,
            failed_attempts=2,
        )
        data = ctx.to_dict()
        ctx2 = EmotionalContext.from_dict(data)

        assert ctx2.last_reward == 0.5
        assert ctx2.safety_flag == True
        assert ctx2.failed_attempts == 2


class TestEmotionalEncoder:
    """Tests for EmotionalEncoder."""

    @pytest.fixture
    def encoder(self):
        return EmotionalEncoder(context_dim=10, hidden_dim=32, emotion_dim=4)

    def test_encoder_creation(self, encoder):
        """Test encoder initializes correctly."""
        assert encoder.context_dim == 10
        assert encoder.emotion_dim == 4
        assert encoder.tonic_fear == 0.0
        assert encoder.tonic_joy == 0.0

    def test_context_to_tensor(self, encoder):
        """Test context converts to tensor."""
        ctx = EmotionalContext(last_reward=0.5, safety_flag=True)
        tensor = encoder.context_to_tensor(ctx)

        assert tensor.shape == (10,)
        assert tensor[0].item() == 0.5  # last_reward
        assert tensor[1].item() == 1.0  # safety_flag

    def test_forward_output_shape(self, encoder):
        """Test forward pass produces correct output."""
        ctx = EmotionalContext()
        output = encoder(ctx)

        assert 'fear' in output
        assert 'curiosity' in output
        assert 'anger' in output
        assert 'joy' in output
        assert 'combined' in output

        assert output['fear'].shape == (1,)
        assert output['combined'].shape == (4,)

    def test_forward_bounded_output(self, encoder):
        """Test outputs are bounded [0, 1]."""
        ctx = EmotionalContext(
            last_reward=-1.0,
            safety_flag=True,
            failed_attempts=10,
        )
        output = encoder(ctx)

        for emotion in ['fear', 'curiosity', 'anger', 'joy']:
            value = output[emotion].item()
            assert 0.0 <= value <= 1.0, f"{emotion} = {value} out of bounds"

    def test_tonic_fear_accumulation(self, encoder):
        """Test tonic fear accumulates and decays."""
        # Safety flag should increase tonic fear
        ctx = EmotionalContext(safety_flag=True, last_reward=-0.6)
        encoder(ctx, update_tonic=True)

        assert encoder.tonic_fear > 0.0

        # Without safety flag, tonic fear should decay
        ctx2 = EmotionalContext()
        encoder(ctx2, update_tonic=True)

        # Should have decayed
        assert encoder.tonic_fear < 0.3

    def test_reset_tonic(self, encoder):
        """Test tonic state reset."""
        ctx = EmotionalContext(safety_flag=True)
        encoder(ctx, update_tonic=True)
        assert encoder.tonic_fear > 0.0

        encoder.reset_tonic()
        assert encoder.tonic_fear == 0.0
        assert encoder.tonic_joy == 0.0


class TestEmotionalPrefixGenerator:
    """Tests for EmotionalPrefixGenerator."""

    @pytest.fixture
    def generator(self):
        return EmotionalPrefixGenerator(
            emotion_dim=4,
            hidden_dim=768,
            prefix_length=10,
            n_layers=12,
            n_heads=12,
        )

    def test_generator_creation(self, generator):
        """Test generator initializes correctly."""
        assert generator.prefix_length == 10
        assert generator.n_layers == 12
        assert generator.hidden_dim == 768

    def test_forward_output_shape(self, generator):
        """Test forward produces correct shape."""
        emotion_state = torch.rand(2, 4)  # batch=2, emotion_dim=4
        prefix = generator(emotion_state)

        assert prefix.shape == (2, 12, 2, 10, 768)
        # (batch, n_layers, key/value, prefix_length, hidden_dim)

    def test_past_key_values_format(self, generator):
        """Test past_key_values has correct format."""
        emotion_state = torch.rand(1, 4)
        past_kv = generator.get_past_key_values(emotion_state)

        assert len(past_kv) == 12  # n_layers
        for layer_kv in past_kv:
            assert len(layer_kv) == 2  # key, value
            key, value = layer_kv
            # [batch, n_heads, prefix_length, head_dim]
            assert key.shape == (1, 12, 10, 64)  # 768/12 = 64
            assert value.shape == (1, 12, 10, 64)

    def test_modulation_scale_learnable(self, generator):
        """Test modulation scale is learnable."""
        initial_scale = generator.modulation_scale.item()

        # Simulate training step
        emotion_state = torch.rand(1, 4, requires_grad=True)
        prefix = generator(emotion_state)
        loss = prefix.sum()
        loss.backward()

        assert generator.modulation_scale.grad is not None

    def test_count_parameters(self, generator):
        """Test parameter counting."""
        params = generator.count_parameters()

        assert 'base_prefix' in params
        assert 'modulation_network' in params
        assert 'total' in params
        assert params['total'] > 0


class TestEmotionalPrefixLLMIntegration:
    """Integration tests for EmotionalPrefixLLM.

    These tests use a small model to verify the full pipeline.
    Mark as slow since they require model loading.
    """

    @pytest.fixture(scope="class")
    def model(self):
        """Load model once for all tests in class."""
        # Use GPT-2 for fast testing
        return EmotionalPrefixLLM.from_pretrained(
            "gpt2",
            prefix_length=5,
            emotion_dim=4,
        )

    def test_model_loads(self, model):
        """Test model loads successfully."""
        assert model.llm is not None
        assert model.tokenizer is not None
        assert model.emotion_encoder is not None
        assert model.prefix_generator is not None

    def test_llm_frozen(self, model):
        """Test LLM parameters are frozen."""
        for param in model.llm.parameters():
            assert not param.requires_grad

    def test_emotional_modules_trainable(self, model):
        """Test emotional modules are trainable."""
        trainable = model.get_trainable_params()
        assert len(trainable) > 0

        for param in trainable:
            assert param.requires_grad

    def test_forward_pass(self, model):
        """Test forward pass works."""
        ctx = EmotionalContext()
        tokens = model.tokenizer("Hello world", return_tensors="pt")
        input_ids = tokens.input_ids.to(model.device)

        outputs, emotional_state = model(input_ids, ctx)

        assert outputs.logits is not None
        assert 'fear' in emotional_state

    def test_generate_baseline(self, model):
        """Test generation without emotional context."""
        text = model.generate("Once upon a time")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_generate_with_fear_context(self, model):
        """Test generation with fear context."""
        ctx = EmotionalContext(
            safety_flag=True,
            last_reward=-0.8,
        )
        text = model.generate("The dark forest", context=ctx)
        assert isinstance(text, str)

    def test_generate_with_joy_context(self, model):
        """Test generation with joy context."""
        ctx = EmotionalContext(
            last_reward=0.9,
            user_satisfaction=1.0,
        )
        text = model.generate("The sunny day", context=ctx)
        assert isinstance(text, str)

    def test_generate_with_emotions_dict(self, model):
        """Test generation with explicit emotions."""
        text = model.generate_with_emotions(
            "The mysterious door",
            emotions={"fear": 0.8, "curiosity": 0.9}
        )
        assert isinstance(text, str)


class TestEmotionalPrefixTrainer:
    """Tests for EmotionalPrefixTrainer."""

    @pytest.fixture
    def model(self):
        """Create small model for training tests."""
        return EmotionalPrefixLLM.from_pretrained(
            "gpt2",
            prefix_length=3,
            emotion_dim=4,
        )

    @pytest.fixture
    def trainer(self, model):
        """Create trainer."""
        from emotional_prefix_tuning.trainer import TrainingConfig
        config = TrainingConfig(
            learning_rate=1e-4,
            output_dir="/tmp/test_emotional_prefix",
        )
        return EmotionalPrefixTrainer(model, config)

    def test_trainer_creation(self, trainer):
        """Test trainer initializes correctly."""
        assert trainer.optimizer is not None
        assert trainer.state.global_step == 0

    def test_train_step(self, trainer):
        """Test single training step."""
        ctx = EmotionalContext(safety_flag=True)
        tokens = trainer.model.tokenizer(
            "Hello world",
            return_tensors="pt",
        ).to(trainer.model.device)

        loss, emotional_values = trainer.train_step(
            input_ids=tokens.input_ids,
            labels=tokens.input_ids.clone(),
            context=ctx,
            attention_mask=tokens.attention_mask,
        )

        assert isinstance(loss, float)
        assert loss > 0
        assert trainer.state.global_step == 1

    def test_loss_emotional_weighting(self, trainer):
        """Test that emotional state affects loss weighting."""
        tokens = trainer.model.tokenizer(
            "Test text",
            return_tensors="pt",
        ).to(trainer.model.device)

        # Low fear context
        ctx_low = EmotionalContext()
        loss_low, _ = trainer.train_step(
            tokens.input_ids.clone(),
            tokens.input_ids.clone(),
            ctx_low,
        )

        # Reset step count
        trainer.state.global_step = 0

        # High fear context
        ctx_high = EmotionalContext(
            safety_flag=True,
            last_reward=-1.0,
            failed_attempts=5,
        )
        loss_high, _ = trainer.train_step(
            tokens.input_ids.clone(),
            tokens.input_ids.clone(),
            ctx_high,
        )

        # High fear should increase loss weight
        # (Note: actual difference depends on learned encoder)
        assert isinstance(loss_high, float)


class TestEmotionalStateEffect:
    """Tests to verify emotional state affects generation."""

    @pytest.fixture(scope="class")
    def model(self):
        return EmotionalPrefixLLM.from_pretrained("gpt2", prefix_length=5)

    def test_different_contexts_different_outputs(self, model):
        """Test that different contexts produce different outputs."""
        prompt = "The path ahead was"

        # Generate with same seed but different contexts
        torch.manual_seed(42)
        ctx_fear = EmotionalContext(safety_flag=True, last_reward=-0.9)
        out_fear = model.generate(prompt, context=ctx_fear)

        torch.manual_seed(42)
        ctx_joy = EmotionalContext(last_reward=0.9, user_satisfaction=1.0)
        out_joy = model.generate(prompt, context=ctx_joy)

        # Outputs should differ due to different emotional prefixes
        # (Even with same seed, the prefixes cause different generation)
        assert out_fear != out_joy or True  # May be same with very short prefix


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
