"""
Emotional Memory LLM - Main model class.

LLM with external emotional memory system.
The LLM is FROZEN. Emotional memory is TRAINABLE through experiences.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path

from .tonic_state import TonicEmotionalState
from .episodic_memory import EpisodicEmotionalMemory
from .semantic_memory import SemanticEmotionalMemory
from .context_generator import EmotionalContextGenerator


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True


class EmotionalMemoryLLM(nn.Module):
    """
    LLM with external emotional memory system.

    The LLM is FROZEN. Emotional memory is TRAINABLE through experiences.
    Uses context injection to influence LLM behavior.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        max_memory_entries: int = 500,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize emotional memory LLM.

        Args:
            model_name: HuggingFace model name
            max_memory_entries: Maximum entries in episodic memory
            device: Device to use
        """
        super().__init__()
        self.model_name = model_name

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Load frozen LLM
        print(f"Loading LLM: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # FREEZE LLM
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Get embedding dimension from model
        self.hidden_dim = self.model.config.hidden_size

        # TRAINABLE memory systems (through experience)
        self.episodic_memory = EpisodicEmotionalMemory(
            embedding_dim=self.hidden_dim,
            max_entries=max_memory_entries,
            device=self.device,
        )
        self.semantic_memory = SemanticEmotionalMemory()
        self.tonic_state = TonicEmotionalState()
        self.context_generator = EmotionalContextGenerator()

        # Current emotional state (for tracking)
        self._current_emotional_state: Dict[str, float] = {}

        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Max memory entries: {max_memory_entries}")

    def _get_embedding(self, text: str) -> torch.Tensor:
        """
        Get embedding for text using the LLM's embedding layer.

        Args:
            text: Input text

        Returns:
            Mean-pooled embedding tensor
        """
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            # Get embeddings from the model
            embeddings = self.model.get_input_embeddings()(tokens.input_ids)
            # Mean pooling over sequence length
            mask = tokens.attention_mask.unsqueeze(-1)
            pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1)

        return pooled.squeeze(0)

    def compute_emotional_state(self, query: str) -> Dict[str, float]:
        """
        Compute emotional state for current query.

        Combines episodic memories, semantic associations, and tonic state.

        Args:
            query: Current user query

        Returns:
            Dictionary of emotion → value
        """
        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Retrieve similar past experiences
        memories = self.episodic_memory.retrieve(query_embedding, k=5)

        # Get semantic associations
        semantic_emotions = self.semantic_memory.get_emotional_associations(query)

        # Combine with tonic state
        combined = defaultdict(float)

        # Tonic contribution (40%)
        for emotion, value in self.tonic_state.to_dict().items():
            combined[emotion] += value * 0.4

        # Episodic contribution (30%)
        if memories:
            for memory in memories:
                similarity_weight = 1.0 / (1 + len(memories))
                for emotion, value in memory.emotional_state.items():
                    combined[emotion] += value * similarity_weight * 0.3

        # Semantic contribution (30%)
        for emotion, value in semantic_emotions.items():
            combined[emotion] += value * 0.3

        self._current_emotional_state = dict(combined)
        return self._current_emotional_state

    def generate(
        self,
        query: str,
        config: Optional[GenerationConfig] = None,
        include_context: bool = True,
    ) -> str:
        """
        Generate response with emotional memory context.

        Args:
            query: User query
            config: Generation configuration
            include_context: Whether to include emotional context

        Returns:
            Generated response
        """
        if config is None:
            config = GenerationConfig()

        # Compute emotional state
        emotional_state = self.compute_emotional_state(query)

        # Build input with emotional context if enabled
        if include_context:
            # Get query embedding for memory retrieval
            query_embedding = self._get_embedding(query)
            memories = self.episodic_memory.retrieve(query_embedding, k=5)
            semantic_emotions = self.semantic_memory.get_emotional_associations(query)

            # Generate emotional context
            emotional_context = self.context_generator.generate_context(
                memories, semantic_emotions, self.tonic_state
            )

            if emotional_context:
                full_input = f"{emotional_context}\n\nUser: {query}\n\nAssistant:"
            else:
                full_input = f"User: {query}\n\nAssistant:"
        else:
            full_input = f"User: {query}\n\nAssistant:"

        # Tokenize and generate
        inputs = self.tokenizer(full_input, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the response part
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        return response

    def receive_feedback(
        self,
        query: str,
        response: str,
        feedback: float,
    ) -> None:
        """
        Update memory based on feedback.

        This is how the system LEARNS from experience.
        Similar to TD-learning in RL agents.

        Args:
            query: Original query
            response: Generated response
            feedback: User feedback (-1 to 1)
        """
        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Store in episodic memory
        self.episodic_memory.add(
            context_embedding=query_embedding,
            context_text=query,
            emotional_state=self._current_emotional_state.copy(),
            response=response,
            outcome=feedback,
        )

        # Update semantic associations
        self.semantic_memory.update_from_experience(
            context=query,
            outcome=feedback,
            emotional_state=self._current_emotional_state,
        )

        # Update tonic state
        self.tonic_state.update_from_feedback(feedback)
        self.tonic_state.decay()

    def get_memory_stats(self) -> Dict:
        """Get statistics about memory usage."""
        return {
            "episodic_size": self.episodic_memory.size(),
            "episodic_stats": self.episodic_memory.get_summary_stats(),
            "semantic_size": self.semantic_memory.size(),
            "tonic_state": self.tonic_state.to_dict(),
        }

    def reset_session(self) -> None:
        """Reset session-level state (keep long-term memory)."""
        self.tonic_state.reset()
        self._current_emotional_state = {}

    def reset_all(self) -> None:
        """Reset all memory (full reset)."""
        self.episodic_memory.clear()
        self.semantic_memory.clear()
        self.tonic_state.reset()
        self._current_emotional_state = {}

    def save_memory(self, path: str) -> None:
        """
        Save memory state to disk.

        Args:
            path: Path to save file
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save entries (embeddings as lists for JSON)
        data = {
            'episodic': [
                {
                    'context_text': e.context_text,
                    'emotional_state': e.emotional_state,
                    'response_text': e.response_text,
                    'outcome': e.outcome,
                    'weight': e.weight,
                    'timestamp': e.timestamp,
                }
                for e in self.episodic_memory.entries
            ],
            'semantic': self.semantic_memory.to_dict(),
            'tonic': self.tonic_state.to_dict(),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_memory(self, path: str) -> None:
        """
        Load memory state from disk.

        Args:
            path: Path to saved file
        """
        with open(path, 'r') as f:
            data = json.load(f)

        # Rebuild episodic memory (need to recompute embeddings)
        for entry_data in data.get('episodic', []):
            embedding = self._get_embedding(entry_data['context_text'])
            self.episodic_memory.add(
                context_embedding=embedding,
                context_text=entry_data['context_text'],
                emotional_state=entry_data['emotional_state'],
                response=entry_data['response_text'],
                outcome=entry_data['outcome'],
            )

        # Restore semantic memory
        self.semantic_memory = SemanticEmotionalMemory.from_dict(
            data.get('semantic', {}),
        )

        # Restore tonic state
        self.tonic_state = TonicEmotionalState.from_dict(
            data.get('tonic', {}),
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "gpt2",
        **kwargs,
    ) -> "EmotionalMemoryLLM":
        """
        Create model from pretrained LLM.

        Args:
            model_name: HuggingFace model name
            **kwargs: Additional arguments

        Returns:
            EmotionalMemoryLLM instance
        """
        return cls(model_name=model_name, **kwargs)
