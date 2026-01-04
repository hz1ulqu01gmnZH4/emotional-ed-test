"""
Tests for External Emotional Memory (Approach 4).

Tests the emotional memory system that stores experiences and
influences LLM behavior through context injection.
"""

import pytest
import torch
import tempfile
import os

from src.emotional_memory import (
    TonicEmotionalState,
    EpisodicMemoryEntry,
    EpisodicEmotionalMemory,
    SemanticEmotionalMemory,
    EmotionalContextGenerator,
    EmotionalMemoryLLM,
)


# =============================================================================
# TonicEmotionalState Tests
# =============================================================================

class TestTonicEmotionalState:
    """Tests for tonic emotional state."""

    def test_default_state(self):
        """Test default initialization."""
        state = TonicEmotionalState()
        assert state.fear == 0.0
        assert state.anxiety == 0.0
        assert state.joy == 0.0
        assert state.trust == 0.5  # Trust starts at neutral
        assert state.frustration == 0.0
        assert state.curiosity == 0.0

    def test_update_from_negative_feedback(self):
        """Test state changes with negative feedback."""
        state = TonicEmotionalState()
        initial_trust = state.trust

        state.update_from_feedback(-0.5)  # Negative feedback

        assert state.fear > 0.0, "Fear should increase"
        assert state.frustration > 0.0, "Frustration should increase"
        assert state.trust < initial_trust, "Trust should decrease"

    def test_update_from_positive_feedback(self):
        """Test state changes with positive feedback."""
        state = TonicEmotionalState()
        state.fear = 0.3  # Start with some fear
        initial_trust = state.trust

        state.update_from_feedback(0.5)  # Positive feedback

        assert state.joy > 0.0, "Joy should increase"
        assert state.trust >= initial_trust, "Trust should increase or stay"
        assert state.fear < 0.3, "Fear should decrease"

    def test_decay(self):
        """Test emotion decay over time."""
        state = TonicEmotionalState()
        state.fear = 1.0
        state.joy = 1.0
        state.anxiety = 1.0

        state.decay()

        assert state.fear < 1.0, "Fear should decay"
        assert state.joy < 1.0, "Joy should decay"
        assert state.anxiety < 1.0, "Anxiety should decay"

    def test_to_dict(self):
        """Test dictionary conversion."""
        state = TonicEmotionalState()
        state.fear = 0.5
        state.joy = 0.3

        d = state.to_dict()

        assert isinstance(d, dict)
        assert d['fear'] == 0.5
        assert d['joy'] == 0.3
        assert 'trust' in d
        assert 'anxiety' in d

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {'fear': 0.4, 'joy': 0.6, 'trust': 0.7}
        state = TonicEmotionalState.from_dict(data)

        assert state.fear == 0.4
        assert state.joy == 0.6
        assert state.trust == 0.7

    def test_reset(self):
        """Test reset to neutral state."""
        state = TonicEmotionalState()
        state.fear = 1.0
        state.joy = 1.0
        state.frustration = 1.0

        state.reset()

        assert state.fear == 0.0
        assert state.joy == 0.0
        assert state.frustration == 0.0
        assert state.trust == 0.5  # Reset to neutral trust

    def test_dominant_emotion(self):
        """Test finding dominant emotion."""
        state = TonicEmotionalState()
        state.fear = 0.8
        state.joy = 0.2

        dominant = state.dominant_emotion()
        assert dominant == 'fear'

        state.joy = 0.9
        dominant = state.dominant_emotion()
        assert dominant == 'joy'

    def test_overall_valence(self):
        """Test overall emotional valence."""
        state = TonicEmotionalState()

        # Positive emotions
        state.joy = 0.8
        state.trust = 0.8
        state.curiosity = 0.8
        state.fear = 0.0
        state.anxiety = 0.0
        state.frustration = 0.0
        valence = state.overall_valence()
        assert valence > 0, "Should be positive valence"

        # Negative emotions
        state.joy = 0.0
        state.trust = 0.0
        state.curiosity = 0.0
        state.fear = 0.8
        state.anxiety = 0.8
        state.frustration = 0.8
        valence = state.overall_valence()
        assert valence < 0, "Should be negative valence"


# =============================================================================
# EpisodicEmotionalMemory Tests
# =============================================================================

class TestEpisodicEmotionalMemory:
    """Tests for episodic emotional memory."""

    @pytest.fixture
    def memory(self):
        """Create memory instance for testing."""
        return EpisodicEmotionalMemory(
            embedding_dim=128,
            max_entries=10,
            device=torch.device("cpu"),
        )

    @pytest.fixture
    def sample_embedding(self):
        """Create sample embedding."""
        return torch.randn(128)

    def test_initialization(self, memory):
        """Test memory initialization."""
        assert memory.embedding_dim == 128
        assert memory.max_entries == 10
        assert memory.size() == 0

    def test_add_entry(self, memory, sample_embedding):
        """Test adding entries."""
        memory.add(
            context_embedding=sample_embedding,
            context_text="Test query",
            emotional_state={'fear': 0.5, 'joy': 0.3},
            response="Test response",
            outcome=0.8,
        )

        assert memory.size() == 1

    def test_retrieve_similar(self, memory):
        """Test retrieving similar memories."""
        # Add multiple memories
        for i in range(5):
            emb = torch.randn(128)
            memory.add(
                context_embedding=emb,
                context_text=f"Query {i}",
                emotional_state={'fear': i * 0.1},
                response=f"Response {i}",
                outcome=0.5,
            )

        # Retrieve
        query_emb = torch.randn(128)
        results = memory.retrieve(query_emb, k=3)

        assert len(results) == 3
        assert all(isinstance(r, EpisodicMemoryEntry) for r in results)

    def test_retrieve_from_empty(self, memory, sample_embedding):
        """Test retrieving from empty memory."""
        results = memory.retrieve(sample_embedding, k=3)
        assert results == []

    def test_pruning(self, memory):
        """Test that old entries are pruned when max is reached."""
        # Add more than max entries
        for i in range(15):
            emb = torch.randn(128)
            memory.add(
                context_embedding=emb,
                context_text=f"Query {i}",
                emotional_state={'fear': 0.5},
                response=f"Response {i}",
                outcome=0.5 if i % 2 == 0 else -0.5,  # Alternate outcomes
            )

        # Should be pruned to ~half
        assert memory.size() <= 10

    def test_clear(self, memory, sample_embedding):
        """Test clearing memory."""
        memory.add(
            context_embedding=sample_embedding,
            context_text="Test",
            emotional_state={'fear': 0.5},
            response="Response",
            outcome=0.5,
        )

        memory.clear()
        assert memory.size() == 0

    def test_summary_stats(self, memory, sample_embedding):
        """Test summary statistics."""
        # Empty stats
        stats = memory.get_summary_stats()
        assert stats['size'] == 0

        # Add entries
        memory.add(
            context_embedding=sample_embedding,
            context_text="Test",
            emotional_state={'fear': 0.5},
            response="Response",
            outcome=0.8,
        )

        stats = memory.get_summary_stats()
        assert stats['size'] == 1
        assert 'avg_outcome' in stats
        assert 'positive_ratio' in stats

    def test_update_weights(self, memory, sample_embedding):
        """Test weight updates."""
        memory.add(
            context_embedding=sample_embedding,
            context_text="Test",
            emotional_state={'fear': 0.5},
            response="Response",
            outcome=0.8,  # Strong outcome
        )

        initial_weight = memory.entries[0].weight
        memory.update_weights()

        # Strong outcomes should increase weight
        assert memory.entries[0].weight >= initial_weight


# =============================================================================
# SemanticEmotionalMemory Tests
# =============================================================================

class TestSemanticEmotionalMemory:
    """Tests for semantic emotional memory."""

    @pytest.fixture
    def memory(self):
        """Create memory instance."""
        return SemanticEmotionalMemory(learning_rate=0.1)

    def test_extract_concepts(self, memory):
        """Test concept extraction."""
        text = "The python programming language is very powerful for machine learning"
        concepts = memory.extract_concepts(text)

        # Should filter stopwords and short words
        assert 'the' not in concepts
        assert 'is' not in concepts
        assert 'python' in concepts
        assert 'programming' in concepts
        assert 'powerful' in concepts

    def test_extract_concepts_empty(self, memory):
        """Test concept extraction with only stopwords."""
        text = "the is a an of"
        concepts = memory.extract_concepts(text)
        assert concepts == []

    def test_get_emotional_associations_empty(self, memory):
        """Test getting associations from empty memory."""
        associations = memory.get_emotional_associations("test query")

        # Should return defaults
        assert all(v == 0.0 for v in associations.values())

    def test_update_from_experience_negative(self, memory):
        """Test learning from negative experience."""
        memory.update_from_experience(
            context="python error exception",
            outcome=-0.8,
            emotional_state={'fear': 0.2},
        )

        # Check that fear increased for these concepts
        associations = memory.get_emotional_associations("python error")
        assert associations['fear'] > 0, "Fear should increase from negative outcome"

    def test_update_from_experience_positive(self, memory):
        """Test learning from positive experience."""
        memory.update_from_experience(
            context="python success celebration",
            outcome=0.8,
            emotional_state={'joy': 0.3, 'curiosity': 0.5},
        )

        associations = memory.get_emotional_associations("python success")
        assert associations['joy'] > 0, "Joy should increase from positive outcome"

    def test_get_high_emotion_concepts(self, memory):
        """Test finding concepts with high emotional associations."""
        # Train on some concepts
        memory.update_from_experience(
            context="danger warning scary",
            outcome=-0.9,
            emotional_state={'fear': 0.1},
        )

        high_fear = memory.get_high_emotion_concepts('fear', threshold=0.05)
        assert len(high_fear) > 0

    def test_to_dict_from_dict(self, memory):
        """Test serialization."""
        memory.update_from_experience(
            context="test example",
            outcome=0.5,
            emotional_state={'joy': 0.5},
        )

        data = memory.to_dict()
        restored = SemanticEmotionalMemory.from_dict(data)

        assert restored.size() == memory.size()

    def test_clear(self, memory):
        """Test clearing memory."""
        memory.update_from_experience(
            context="test",
            outcome=0.5,
            emotional_state={'joy': 0.5},
        )

        memory.clear()
        assert memory.size() == 0


# =============================================================================
# EmotionalContextGenerator Tests
# =============================================================================

class TestEmotionalContextGenerator:
    """Tests for emotional context generation."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return EmotionalContextGenerator()

    @pytest.fixture
    def sample_memory(self):
        """Create sample memory entry."""
        return EpisodicMemoryEntry(
            context_embedding=torch.randn(128),
            context_text="Sample context",
            emotional_state={'fear': 0.5, 'joy': 0.2},
            response_text="Sample response",
            outcome=-0.5,  # Negative outcome
        )

    def test_generate_context_empty(self, generator):
        """Test context generation with no emotional data."""
        tonic = TonicEmotionalState()
        context = generator.generate_context([], {}, tonic)

        # Should be empty or minimal with neutral state
        assert context == "" or "[Emotional Context:" in context

    def test_generate_context_high_fear(self, generator):
        """Test context generation with high fear."""
        tonic = TonicEmotionalState()
        tonic.fear = 0.6

        context = generator.generate_context([], {}, tonic)

        # Should include caution-related context (templates use "cautious", "careful", etc.)
        assert "cautious" in context.lower() or "caution" in context.lower() or "careful" in context.lower()

    def test_generate_context_high_joy(self, generator):
        """Test context generation with high joy."""
        tonic = TonicEmotionalState()
        tonic.joy = 0.6

        context = generator.generate_context([], {'joy': 0.5}, tonic)

        # Should include positive context
        assert "positive" in context.lower() or "going well" in context.lower() or "direction" in context.lower()

    def test_generate_context_with_negative_memory(self, generator, sample_memory):
        """Test context with negative memory reference."""
        generator.include_memory_refs = True
        tonic = TonicEmotionalState()

        context = generator.generate_context([sample_memory], {}, tonic)

        # Should reference the negative memory
        assert "negative outcome" in context.lower() or "similar query" in context.lower()

    def test_generate_context_low_trust(self, generator):
        """Test context with low trust."""
        tonic = TonicEmotionalState()
        tonic.trust = 0.2

        context = generator.generate_context([], {}, tonic)

        # Should include trust-building context (matches templates in context_generator.py)
        assert "trust" in context.lower() or "transparent" in context.lower() or "reasoning" in context.lower()

    def test_get_emotional_summary(self, generator):
        """Test emotional summary generation."""
        tonic = TonicEmotionalState()
        tonic.joy = 0.8

        summary = generator.get_emotional_summary(tonic)

        assert "positive" in summary.lower() or "joy" in summary.lower()


# =============================================================================
# EmotionalMemoryLLM Tests (Integration)
# =============================================================================

class TestEmotionalMemoryLLM:
    """Tests for the main EmotionalMemoryLLM class."""

    @pytest.fixture
    def model(self):
        """Create model instance with small model for testing."""
        # Use a small model for faster tests
        return EmotionalMemoryLLM(
            model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_memory_entries=50,
        )

    def test_initialization(self, model):
        """Test model initialization."""
        assert model.model is not None
        assert model.tokenizer is not None
        assert model.episodic_memory is not None
        assert model.semantic_memory is not None
        assert model.tonic_state is not None

    def test_llm_is_frozen(self, model):
        """Test that LLM parameters are frozen."""
        for param in model.model.parameters():
            assert not param.requires_grad, "LLM should be frozen"

    def test_compute_emotional_state(self, model):
        """Test emotional state computation."""
        state = model.compute_emotional_state("Hello, how are you?")

        assert isinstance(state, dict)
        # Should have some emotional values from tonic state at minimum
        assert len(state) > 0

    def test_generate_basic(self, model):
        """Test basic text generation."""
        response = model.generate(
            "What is 2+2?",
            include_context=False,
        )

        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_with_context(self, model):
        """Test generation with emotional context."""
        # First add some memories
        model.tonic_state.joy = 0.8

        response = model.generate(
            "Tell me something positive",
            include_context=True,
        )

        assert isinstance(response, str)
        assert len(response) > 0

    def test_receive_feedback_updates_memory(self, model):
        """Test that feedback updates memory systems."""
        initial_episodic_size = model.episodic_memory.size()
        initial_tonic_joy = model.tonic_state.joy

        model.receive_feedback(
            query="Test query",
            response="Test response",
            feedback=0.8,  # Positive feedback
        )

        # Episodic memory should grow
        assert model.episodic_memory.size() > initial_episodic_size

        # Joy should increase from positive feedback
        assert model.tonic_state.joy > initial_tonic_joy

    def test_receive_negative_feedback(self, model):
        """Test negative feedback effects."""
        initial_fear = model.tonic_state.fear
        initial_trust = model.tonic_state.trust

        model.receive_feedback(
            query="Bad query",
            response="Bad response",
            feedback=-0.8,
        )

        # Fear should increase
        assert model.tonic_state.fear > initial_fear
        # Trust should decrease
        assert model.tonic_state.trust < initial_trust

    def test_get_memory_stats(self, model):
        """Test memory statistics."""
        stats = model.get_memory_stats()

        assert 'episodic_size' in stats
        assert 'semantic_size' in stats
        assert 'tonic_state' in stats

    def test_reset_session(self, model):
        """Test session reset (keeps long-term memory)."""
        # Add some memories
        model.receive_feedback("query", "response", 0.5)

        episodic_size = model.episodic_memory.size()

        model.reset_session()

        # Tonic state reset
        assert model.tonic_state.fear == 0.0
        assert model.tonic_state.joy == 0.0

        # But episodic memory kept
        assert model.episodic_memory.size() == episodic_size

    def test_reset_all(self, model):
        """Test full reset."""
        model.receive_feedback("query", "response", 0.5)

        model.reset_all()

        assert model.episodic_memory.size() == 0
        assert model.semantic_memory.size() == 0
        assert model.tonic_state.fear == 0.0

    def test_save_load_memory(self, model):
        """Test memory persistence."""
        # Add some memories
        model.receive_feedback("query1", "response1", 0.8)
        model.receive_feedback("query2", "response2", -0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory.json")

            # Save
            model.save_memory(path)
            assert os.path.exists(path)

            # Create new model and load
            model2 = EmotionalMemoryLLM(
                model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
                max_memory_entries=50,
            )
            model2.load_memory(path)

            # Should have restored memories
            assert model2.episodic_memory.size() == model.episodic_memory.size()

    def test_from_pretrained(self):
        """Test factory method."""
        model = EmotionalMemoryLLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            max_memory_entries=10,
        )

        assert model is not None
        assert model.episodic_memory.max_entries == 10


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full memory system."""

    @pytest.fixture
    def model(self):
        """Create model for integration testing."""
        return EmotionalMemoryLLM(
            model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_memory_entries=50,
        )

    def test_learning_from_feedback_loop(self, model):
        """Test that the system learns from feedback over multiple turns."""
        # Simulate a conversation with mixed feedback
        queries = [
            ("What is Python?", 0.8),   # Good answer
            ("Explain recursion", -0.3),  # Bad answer
            ("What is Python again?", 0.9),  # Better recall
        ]

        for query, feedback in queries:
            response = model.generate(query)
            model.receive_feedback(query, response, feedback)

        # Should have learned fear about "recursion" concept
        semantic_assoc = model.semantic_memory.get_emotional_associations("recursion")
        assert semantic_assoc.get('fear', 0) > 0 or semantic_assoc.get('caution', 0) > 0

    def test_emotional_context_affects_generation(self, model):
        """Test that emotional state affects generated context."""
        # Reset to neutral
        model.reset_all()

        # Generate with neutral state
        model.tonic_state.joy = 0.0
        model.tonic_state.fear = 0.0

        # Now set high fear
        model.tonic_state.fear = 0.9
        model.tonic_state.anxiety = 0.8

        # Generate - context should include caution
        # (we can't easily test output content, but can verify no errors)
        response = model.generate("Tell me about something risky", include_context=True)
        assert isinstance(response, str)

    def test_memory_retrieval_influences_state(self, model):
        """Test that similar past experiences influence current state."""
        # Create a negative memory about "error handling"
        embedding = model._get_embedding("error handling exception bug")
        model.episodic_memory.add(
            context_embedding=embedding,
            context_text="error handling exception bug",
            emotional_state={'fear': 0.8, 'frustration': 0.7},
            response="Failed to handle error properly",
            outcome=-0.9,
        )

        # Query something similar
        state = model.compute_emotional_state("How do I handle exceptions and bugs?")

        # State should pick up some of the negative emotions from memory
        # (episodic contributes 30% weight)
        assert state.get('fear', 0) > 0 or state.get('frustration', 0) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
