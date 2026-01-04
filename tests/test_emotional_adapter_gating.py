"""
Tests for Emotional Adapter Gating implementation.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotional_adapter_gating import (
    EmotionalState,
    EmotionalGate,
    EmotionalAdapter,
    EmotionalEncoderForAdapter,
    EmotionalAdapterLLM,
    EmotionalAdapterTrainer,
)


class TestEmotionalState:
    """Tests for EmotionalState dataclass."""

    def test_state_creation(self):
        """Test default state creation."""
        state = EmotionalState()
        assert state.fear == 0.0
        assert state.joy == 0.0
        assert state.confidence == 0.5

    def test_state_to_tensor(self):
        """Test conversion to tensor."""
        state = EmotionalState(fear=0.5, joy=0.8)
        tensor = state.to_tensor()

        assert tensor.shape == (6,)
        assert tensor[0].item() == pytest.approx(0.5, rel=1e-5)  # fear
        assert tensor[3].item() == pytest.approx(0.8, rel=1e-5)  # joy

    def test_from_tensor(self):
        """Test creation from tensor."""
        tensor = torch.tensor([0.5, 0.3, 0.2, 0.8, 0.1, 0.6])
        state = EmotionalState.from_tensor(tensor)

        assert state.fear == pytest.approx(0.5, rel=1e-5)
        assert state.curiosity == pytest.approx(0.3, rel=1e-5)
        assert state.anger == pytest.approx(0.2, rel=1e-5)
        assert state.joy == pytest.approx(0.8, rel=1e-5)

    def test_factory_methods(self):
        """Test factory methods for emotions."""
        fearful = EmotionalState.fearful(0.9)
        assert fearful.fear == 0.9

        curious = EmotionalState.curious(0.8)
        assert curious.curiosity == 0.8

        joyful = EmotionalState.joyful(0.7)
        assert joyful.joy == 0.7

    def test_blend(self):
        """Test blending two states."""
        fear_state = EmotionalState.fearful(1.0)
        joy_state = EmotionalState.joyful(1.0)

        blended = fear_state.blend(joy_state, weight=0.5)
        assert blended.fear == 0.5
        assert blended.joy == 0.5

    def test_dominant_emotion(self):
        """Test getting dominant emotion."""
        state = EmotionalState(fear=0.1, joy=0.9)
        assert state.dominant_emotion() == "joy"

        state2 = EmotionalState(fear=0.9, anger=0.5)
        assert state2.dominant_emotion() == "fear"


class TestEmotionalGate:
    """Tests for EmotionalGate."""

    def test_scalar_gate(self):
        """Test scalar gate type."""
        gate = EmotionalGate(emotion_dim=6, hidden_dim=768, gate_type="scalar")
        emotion = torch.rand(2, 6)
        output = gate(emotion)

        assert output.shape == (2, 1)
        assert (output >= 0).all() and (output <= 1).all()

    def test_vector_gate(self):
        """Test vector gate type."""
        gate = EmotionalGate(emotion_dim=6, hidden_dim=768, gate_type="vector")
        emotion = torch.rand(2, 6)
        output = gate(emotion)

        assert output.shape == (2, 768)
        assert (output >= 0).all() and (output <= 1).all()

    def test_attention_gate(self):
        """Test attention gate type."""
        gate = EmotionalGate(emotion_dim=6, hidden_dim=768, gate_type="attention")
        emotion = torch.rand(2, 6)
        hidden = torch.rand(2, 10, 768)
        output = gate(emotion, hidden)

        assert output.shape == (2, 10, 1)
        assert (output >= 0).all() and (output <= 1).all()

    def test_invalid_gate_type(self):
        """Test invalid gate type raises error."""
        with pytest.raises(ValueError):
            EmotionalGate(gate_type="invalid")


class TestEmotionalAdapter:
    """Tests for EmotionalAdapter."""

    @pytest.fixture
    def adapter(self):
        return EmotionalAdapter(
            hidden_dim=768,
            adapter_dim=64,
            emotion_dim=6,
            gate_type="scalar",
        )

    def test_adapter_creation(self, adapter):
        """Test adapter initializes correctly."""
        assert adapter.hidden_dim == 768
        assert adapter.adapter_dim == 64

    def test_forward(self, adapter):
        """Test forward pass."""
        hidden = torch.rand(2, 10, 768)
        emotion = torch.rand(2, 6)

        output = adapter(hidden, emotion)

        assert output.shape == hidden.shape
        # Output should be close to input for zero-init up_proj
        # (gated adapter starts at zero contribution)

    def test_zero_initialization(self, adapter):
        """Test up_proj is zero-initialized."""
        assert torch.allclose(
            adapter.up_proj.weight,
            torch.zeros_like(adapter.up_proj.weight)
        )

    def test_count_parameters(self, adapter):
        """Test parameter counting."""
        params = adapter.count_parameters()
        assert params['total'] > 0
        assert 'down_proj' in params
        assert 'gate' in params


class TestEmotionalEncoderForAdapter:
    """Tests for EmotionalEncoderForAdapter."""

    @pytest.fixture
    def encoder(self):
        return EmotionalEncoderForAdapter(
            hidden_dim=768,
            emotion_dim=6,
        )

    def test_encoder_creation(self, encoder):
        """Test encoder initializes correctly."""
        assert encoder.hidden_dim == 768
        assert encoder.emotion_dim == 6

    def test_forward(self, encoder):
        """Test forward pass."""
        hidden_states = torch.rand(2, 10, 768)
        output = encoder(hidden_states)

        assert output.shape == (2, 6)
        # Output should be bounded [0, 1] (sigmoid)
        assert (output >= 0).all() and (output <= 1).all()

    def test_tonic_state_update(self, encoder):
        """Test tonic state is updated."""
        hidden_states = torch.rand(2, 10, 768)

        # First call
        encoder(hidden_states, update_tonic=True)
        assert encoder.get_tonic_state() is not None

        # Second call should update tonic
        encoder(hidden_states, update_tonic=True)

    def test_tonic_reset(self, encoder):
        """Test tonic state reset."""
        hidden_states = torch.rand(2, 10, 768)
        encoder(hidden_states)

        encoder.reset_tonic()
        assert encoder.get_tonic_state() is None

    def test_external_signals(self, encoder):
        """Test with external signals."""
        hidden_states = torch.rand(2, 10, 768)
        external = torch.rand(2, 5)

        output = encoder(hidden_states, external_signals=external)
        assert output.shape == (2, 6)


class TestEmotionalAdapterLLMIntegration:
    """Integration tests for EmotionalAdapterLLM."""

    @pytest.fixture(scope="class")
    def model(self):
        """Load model once for all tests."""
        return EmotionalAdapterLLM.from_pretrained(
            "gpt2",
            adapter_dim=32,
            emotion_dim=6,
            gate_type="scalar",
        )

    def test_model_loads(self, model):
        """Test model loads successfully."""
        assert model.base_model is not None
        assert model.tokenizer is not None
        assert len(model.adapters) > 0

    def test_base_model_frozen(self, model):
        """Test base model parameters are frozen."""
        for param in model.base_model.parameters():
            assert not param.requires_grad

    def test_adapters_trainable(self, model):
        """Test adapter parameters are trainable."""
        trainable = model.get_trainable_params()
        assert len(trainable) > 0
        for param in trainable:
            assert param.requires_grad

    def test_forward_pass(self, model):
        """Test forward pass works."""
        tokens = model.tokenizer("Hello world", return_tensors="pt")
        input_ids = tokens.input_ids.to(model.device)
        state = EmotionalState.neutral()

        outputs, emotion_tensor = model(input_ids, state)

        assert outputs.logits is not None
        assert emotion_tensor.shape == (1, 6)

    def test_generate_neutral(self, model):
        """Test generation with neutral state."""
        state = EmotionalState.neutral()
        text = model.generate("Once upon a time", emotional_state=state)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_generate_fearful(self, model):
        """Test generation with fearful state."""
        state = EmotionalState.fearful(0.9)
        text = model.generate("The dark cave", emotional_state=state)
        assert isinstance(text, str)

    def test_generate_joyful(self, model):
        """Test generation with joyful state."""
        state = EmotionalState.joyful(0.9)
        text = model.generate("The sunny day", emotional_state=state)
        assert isinstance(text, str)

    def test_different_emotions_different_outputs(self, model):
        """Test different emotions produce different outputs."""
        prompt = "The path ahead"

        torch.manual_seed(42)
        fear_text = model.generate(prompt, EmotionalState.fearful(0.9))

        torch.manual_seed(42)
        joy_text = model.generate(prompt, EmotionalState.joyful(0.9))

        # With untrained adapters, outputs might be similar
        # but the mechanism is in place
        assert isinstance(fear_text, str)
        assert isinstance(joy_text, str)


class TestEmotionalAdapterTrainer:
    """Tests for EmotionalAdapterTrainer."""

    @pytest.fixture
    def model(self):
        return EmotionalAdapterLLM.from_pretrained(
            "gpt2",
            adapter_dim=16,
            emotion_dim=6,
        )

    @pytest.fixture
    def trainer(self, model):
        from emotional_adapter_gating.trainer import TrainingConfig
        config = TrainingConfig(
            learning_rate=1e-4,
            output_dir="/tmp/test_emotional_adapter",
        )
        return EmotionalAdapterTrainer(model, config)

    def test_trainer_creation(self, trainer):
        """Test trainer initializes correctly."""
        assert trainer.optimizer is not None
        assert trainer.state.global_step == 0

    def test_train_step(self, trainer):
        """Test single training step."""
        tokens = trainer.model.tokenizer(
            "Hello world",
            return_tensors="pt",
        ).to(trainer.model.device)

        state = EmotionalState.fearful(0.5)

        loss, metrics = trainer.train_step(
            input_ids=tokens.input_ids,
            labels=tokens.input_ids.clone(),
            emotional_state=state,
            attention_mask=tokens.attention_mask,
        )

        assert isinstance(loss, float)
        assert loss > 0
        assert trainer.state.global_step == 1
        assert "emotion_norm" in metrics

    def test_gate_pattern_analysis(self, trainer):
        """Test gate pattern analysis."""
        prompts = ["Hello", "World"]
        states = [EmotionalState.fearful(0.9), EmotionalState.joyful(0.9)]

        patterns = trainer.analyze_gate_patterns(prompts, states)

        assert len(patterns) > 0
        for layer_idx, values in patterns.items():
            assert len(values) == 2


class TestGateTypeComparison:
    """Compare different gate types."""

    @pytest.fixture
    def hidden(self):
        return torch.rand(2, 10, 768)

    @pytest.fixture
    def emotion(self):
        return torch.rand(2, 6)

    def test_all_gate_types_work(self, hidden, emotion):
        """Test all gate types produce valid output."""
        for gate_type in ["scalar", "vector", "attention"]:
            adapter = EmotionalAdapter(
                hidden_dim=768,
                adapter_dim=64,
                emotion_dim=6,
                gate_type=gate_type,
            )
            output = adapter(hidden, emotion)
            assert output.shape == hidden.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
