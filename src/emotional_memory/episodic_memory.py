"""
Episodic Emotional Memory for storing past experiences.

Uses vector similarity to retrieve relevant past experiences.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time


@dataclass
class EpisodicMemoryEntry:
    """Single entry in episodic emotional memory."""

    context_embedding: torch.Tensor   # Embedding of the context
    context_text: str                  # Original text for debugging
    emotional_state: Dict[str, float]  # Emotion at that time
    response_text: str                 # What was responded
    outcome: float                     # Feedback (-1 to 1)
    timestamp: float = field(default_factory=time.time)
    weight: float = 1.0                # Learned importance weight


class EpisodicEmotionalMemory:
    """
    Episodic memory for emotional experiences.

    Uses vector similarity to retrieve relevant past experiences.
    Simplified implementation using PyTorch (no FAISS required).
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        max_entries: int = 1000,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize episodic memory.

        Args:
            embedding_dim: Dimension of embeddings
            max_entries: Maximum number of entries to store
            device: Device for tensors
        """
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries
        self.device = device or torch.device("cpu")

        # Memory entries
        self.entries: List[EpisodicMemoryEntry] = []

        # Embedding matrix for fast retrieval (rebuilt when entries change)
        self._embeddings: Optional[torch.Tensor] = None
        self._embeddings_dirty = True

    def add(
        self,
        context_embedding: torch.Tensor,
        context_text: str,
        emotional_state: Dict[str, float],
        response: str,
        outcome: float,
    ) -> None:
        """
        Add new memory entry.

        Args:
            context_embedding: Pre-computed embedding of context
            context_text: Original text
            emotional_state: Emotional state at that time
            response: What was responded
            outcome: Feedback score (-1 to 1)
        """
        # Normalize embedding
        embedding = F.normalize(context_embedding.detach().to(self.device), dim=-1)

        entry = EpisodicMemoryEntry(
            context_embedding=embedding,
            context_text=context_text,
            emotional_state=emotional_state.copy(),
            response_text=response,
            outcome=outcome,
        )

        self.entries.append(entry)
        self._embeddings_dirty = True

        # Prune if too large
        if len(self.entries) > self.max_entries:
            self._prune_old_entries()

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        k: int = 5,
    ) -> List[EpisodicMemoryEntry]:
        """
        Retrieve k most similar past experiences.

        Args:
            query_embedding: Embedding of current query
            k: Number of entries to retrieve

        Returns:
            List of most similar memory entries
        """
        if len(self.entries) == 0:
            return []

        # Normalize query
        query = F.normalize(query_embedding.detach().to(self.device), dim=-1)

        # Rebuild embedding matrix if needed
        if self._embeddings_dirty or self._embeddings is None:
            self._rebuild_embeddings()

        # Compute cosine similarities
        similarities = torch.matmul(self._embeddings, query.squeeze())

        # Get top-k
        k = min(k, len(self.entries))
        _, indices = torch.topk(similarities, k)

        return [self.entries[i] for i in indices.tolist()]

    def _rebuild_embeddings(self) -> None:
        """Rebuild embedding matrix from entries."""
        if not self.entries:
            self._embeddings = None
            return

        embeddings = []
        for entry in self.entries:
            embeddings.append(entry.context_embedding.flatten())

        self._embeddings = torch.stack(embeddings)
        self._embeddings_dirty = False

    def _prune_old_entries(self) -> None:
        """Remove oldest, least important entries."""
        # Score entries by importance (outcome magnitude * weight * recency)
        current_time = time.time()

        scored = []
        for i, entry in enumerate(self.entries):
            recency = 1.0 / (1.0 + (current_time - entry.timestamp) / 3600)  # Hours
            importance = abs(entry.outcome) * entry.weight * recency
            scored.append((i, importance))

        # Sort by importance descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Keep top half
        keep_indices = set(i for i, _ in scored[:self.max_entries // 2])
        self.entries = [e for i, e in enumerate(self.entries) if i in keep_indices]
        self._embeddings_dirty = True

    def update_weights(self) -> None:
        """Update entry weights based on outcome consistency."""
        # Entries that consistently predict outcomes get higher weight
        for entry in self.entries:
            if abs(entry.outcome) > 0.5:
                entry.weight = min(2.0, entry.weight * 1.1)
            else:
                entry.weight = max(0.5, entry.weight * 0.95)

    def size(self) -> int:
        """Get number of entries in memory."""
        return len(self.entries)

    def clear(self) -> None:
        """Clear all entries."""
        self.entries.clear()
        self._embeddings = None
        self._embeddings_dirty = True

    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics of memory contents."""
        if not self.entries:
            return {"size": 0}

        outcomes = [e.outcome for e in self.entries]
        return {
            "size": len(self.entries),
            "avg_outcome": sum(outcomes) / len(outcomes),
            "positive_ratio": sum(1 for o in outcomes if o > 0) / len(outcomes),
            "avg_weight": sum(e.weight for e in self.entries) / len(self.entries),
        }
