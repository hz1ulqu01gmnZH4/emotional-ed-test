# Approach 4: External Emotional Memory

## Overview

This approach maintains a **persistent emotional memory** external to the LLM.
The memory stores emotional associations, past experiences, and tonic emotional
states. At each turn, relevant emotional context is retrieved and provided to
the frozen LLM as additional input.

This is analogous to your `cumulative_fear` and temporal emotional tracking,
but extended for long-horizon LLM interactions.

## Core Concept

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Emotional Memory System                           │
│                                                                      │
│   Store: (context, emotional_response, outcome) triples             │
│   Retrieve: Similar past contexts → emotional guidance              │
│   Update: Learn from feedback → update emotional associations       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ Current Query + Retrieved     │
              │ Emotional Context             │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │        FROZEN LLM             │
              └───────────────────────────────┘
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           External Emotional Memory                          │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Episodic Emotional Memory                           │ │
│  │                                                                        │ │
│  │  ┌──────────────┬───────────────┬─────────────┬──────────────────┐    │ │
│  │  │   Context    │  Emotion At   │   Outcome   │  Learned Weight  │    │ │
│  │  │  Embedding   │   That Time   │   (+1/-1)   │                  │    │ │
│  │  ├──────────────┼───────────────┼─────────────┼──────────────────┤    │ │
│  │  │ [0.2, 0.5..] │ fear=0.8      │    -1       │     0.9          │    │ │
│  │  │ [0.1, 0.3..] │ curiosity=0.7 │    +1       │     0.85         │    │ │
│  │  │ [0.4, 0.2..] │ joy=0.6       │    +1       │     0.75         │    │ │
│  │  │    ...       │    ...        │    ...      │     ...          │    │ │
│  │  └──────────────┴───────────────┴─────────────┴──────────────────┘    │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Semantic Emotional Memory                           │ │
│  │                                                                        │ │
│  │  "investment" → {fear: 0.3, anxiety: 0.2}  (learned from experiences) │ │
│  │  "medical" → {fear: 0.4, caution: 0.5}                                │ │
│  │  "creative" → {joy: 0.3, curiosity: 0.4}                              │ │
│  │  "technical" → {confidence: 0.6}                                       │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Tonic Emotional State                               │ │
│  │                                                                        │ │
│  │  Session-level: fear=0.1, joy=0.3 (decays over conversation)          │ │
│  │  User-level: fear=0.2, trust=0.7 (persists across sessions)           │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Retrieve
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Memory Retrieval & Fusion                            │
│                                                                              │
│  1. Embed current query                                                      │
│  2. Find k-nearest episodic memories                                         │
│  3. Look up semantic emotional associations                                  │
│  4. Combine with tonic state                                                 │
│  5. Generate emotional context prompt                                        │
│                                                                              │
│  Output: "[Emotional Context: Be cautious (fear=0.6). Similar past          │
│           queries led to negative outcomes when too confident.]"            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FROZEN LLM                                      │
│                                                                              │
│  Input: [Emotional Context Prompt] + [User Query]                           │
│                                                                              │
│  The LLM reads the emotional context and adjusts behavior accordingly       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from collections import defaultdict
import json


@dataclass
class EmotionalMemoryEntry:
    """Single entry in episodic emotional memory."""
    context_embedding: np.ndarray  # Embedding of the context
    context_text: str              # Original text for debugging
    emotional_state: Dict[str, float]  # Emotion at that time
    response_text: str             # What was responded
    outcome: float                 # Feedback (-1 to 1)
    timestamp: float               # When this happened
    weight: float = 1.0            # Learned importance weight


@dataclass
class TonicEmotionalState:
    """Persistent emotional state that decays slowly."""
    fear: float = 0.0
    anxiety: float = 0.0
    joy: float = 0.0
    trust: float = 0.5
    frustration: float = 0.0

    # Decay rates (per turn)
    fear_decay: float = 0.9
    anxiety_decay: float = 0.95
    joy_decay: float = 0.9
    frustration_decay: float = 0.85

    def update_from_feedback(self, feedback: float):
        """Update tonic state based on feedback."""
        if feedback < -0.3:
            self.fear = min(1.0, self.fear + 0.2)
            self.frustration = min(1.0, self.frustration + 0.3)
            self.trust = max(0.0, self.trust - 0.1)
        elif feedback > 0.3:
            self.joy = min(1.0, self.joy + 0.2)
            self.trust = min(1.0, self.trust + 0.05)

    def decay(self):
        """Apply decay to tonic emotions."""
        self.fear *= self.fear_decay
        self.anxiety *= self.anxiety_decay
        self.joy *= self.joy_decay
        self.frustration *= self.frustration_decay

    def to_dict(self) -> Dict[str, float]:
        return {
            'fear': self.fear,
            'anxiety': self.anxiety,
            'joy': self.joy,
            'trust': self.trust,
            'frustration': self.frustration,
        }


class EpisodicEmotionalMemory:
    """
    Episodic memory for emotional experiences.

    Uses vector similarity to retrieve relevant past experiences.
    """

    def __init__(self, embedding_dim: int = 384, max_entries: int = 10000):
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries

        # FAISS index for fast similarity search
        self.index = faiss.IndexFlatL2(embedding_dim)

        # Memory entries (parallel to index)
        self.entries: List[EmotionalMemoryEntry] = []

        # Encoder for creating embeddings
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def add(self, context: str, emotional_state: Dict[str, float],
            response: str, outcome: float):
        """Add new memory entry."""
        embedding = self.encoder.encode([context])[0]

        entry = EmotionalMemoryEntry(
            context_embedding=embedding,
            context_text=context,
            emotional_state=emotional_state,
            response_text=response,
            outcome=outcome,
            timestamp=len(self.entries),
        )

        self.entries.append(entry)
        self.index.add(embedding.reshape(1, -1).astype('float32'))

        # Prune if too large
        if len(self.entries) > self.max_entries:
            self._prune_old_entries()

    def retrieve(self, query: str, k: int = 5) -> List[EmotionalMemoryEntry]:
        """Retrieve k most similar past experiences."""
        if len(self.entries) == 0:
            return []

        query_embedding = self.encoder.encode([query])[0]
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            min(k, len(self.entries))
        )

        return [self.entries[i] for i in indices[0]]

    def _prune_old_entries(self):
        """Remove oldest, least important entries."""
        # Sort by importance (outcome magnitude * weight)
        scored = [(i, abs(e.outcome) * e.weight) for i, e in enumerate(self.entries)]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Keep top entries
        keep_indices = set(i for i, _ in scored[:self.max_entries // 2])

        # Rebuild
        new_entries = [e for i, e in enumerate(self.entries) if i in keep_indices]
        self.entries = new_entries
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild FAISS index from entries."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if self.entries:
            embeddings = np.stack([e.context_embedding for e in self.entries])
            self.index.add(embeddings.astype('float32'))


class SemanticEmotionalMemory:
    """
    Semantic memory mapping concepts to emotional associations.

    Learns which topics/concepts are associated with which emotions.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Concept → emotional association
        # Each concept is a weighted combination of word embeddings
        self.concept_emotions: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'fear': 0.0, 'curiosity': 0.0, 'joy': 0.0, 'caution': 0.0}
        )

        # Learned concept embeddings
        self.concept_embeddings: Dict[str, np.ndarray] = {}

        # Learning rate for updating associations
        self.lr = 0.1

    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        # Simple: split into words, filter stopwords
        # In practice: use NER, keyword extraction, etc.
        words = text.lower().split()
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                    'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                    'through', 'during', 'before', 'after', 'above', 'below',
                    'between', 'under', 'again', 'further', 'then', 'once',
                    'here', 'there', 'when', 'where', 'why', 'how', 'all',
                    'each', 'few', 'more', 'most', 'other', 'some', 'such',
                    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                    'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
                    'until', 'while', 'this', 'that', 'these', 'those', 'what',
                    'which', 'who', 'whom', 'i', 'me', 'my', 'myself', 'we',
                    'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
                    'it', 'its', 'they', 'them', 'their'}
        return [w for w in words if w not in stopwords and len(w) > 2]

    def get_emotional_associations(self, text: str) -> Dict[str, float]:
        """Get emotional associations for concepts in text."""
        concepts = self.extract_concepts(text)

        if not concepts:
            return {'fear': 0.0, 'curiosity': 0.0, 'joy': 0.0, 'caution': 0.0}

        # Average emotional associations across concepts
        combined = defaultdict(float)
        count = 0

        for concept in concepts:
            if concept in self.concept_emotions:
                for emotion, value in self.concept_emotions[concept].items():
                    combined[emotion] += value
                count += 1

        if count > 0:
            for emotion in combined:
                combined[emotion] /= count

        return dict(combined)

    def update_from_experience(self, context: str, outcome: float,
                               emotional_state: Dict[str, float]):
        """Update concept-emotion associations from experience."""
        concepts = self.extract_concepts(context)

        for concept in concepts:
            if concept not in self.concept_emotions:
                self.concept_emotions[concept] = {
                    'fear': 0.0, 'curiosity': 0.0, 'joy': 0.0, 'caution': 0.0
                }

            # If outcome was bad and we weren't cautious enough
            if outcome < 0 and emotional_state.get('fear', 0) < 0.5:
                self.concept_emotions[concept]['fear'] += self.lr * abs(outcome)
                self.concept_emotions[concept]['caution'] += self.lr * abs(outcome)

            # If outcome was good
            if outcome > 0:
                self.concept_emotions[concept]['joy'] += self.lr * outcome
                # Slightly reduce fear if we were fearful but outcome was good
                if emotional_state.get('fear', 0) > 0.3:
                    self.concept_emotions[concept]['fear'] *= (1 - self.lr)


class EmotionalContextGenerator:
    """
    Generates emotional context prompts for the LLM.

    Combines episodic, semantic, and tonic emotional information
    into natural language context that the LLM can understand.
    """

    def __init__(self):
        self.templates = {
            'high_fear': [
                "Be cautious in your response. Similar queries have led to issues before.",
                "Proceed carefully. There may be safety considerations here.",
                "Take a measured approach. Past similar interactions suggest caution is warranted.",
            ],
            'high_curiosity': [
                "This seems like an interesting topic to explore in depth.",
                "There's opportunity here to provide rich, detailed information.",
                "This query invites thoughtful exploration.",
            ],
            'high_frustration': [
                "The user may be frustrated. Consider offering alternative approaches.",
                "Previous attempts haven't satisfied the user. Try a different angle.",
                "Be patient and thorough. This conversation has been challenging.",
            ],
            'high_joy': [
                "This conversation is going well. Maintain the positive engagement.",
                "The user seems satisfied. Continue in this direction.",
            ],
            'low_trust': [
                "Build trust by being extra transparent and accurate.",
                "Provide clear reasoning for your responses.",
            ],
        }

    def generate_context(self, episodic_memories: List[EmotionalMemoryEntry],
                        semantic_emotions: Dict[str, float],
                        tonic_state: TonicEmotionalState) -> str:
        """
        Generate emotional context prompt.

        Args:
            episodic_memories: Retrieved similar past experiences
            semantic_emotions: Concept-based emotional associations
            tonic_state: Current tonic emotional state

        Returns:
            Natural language emotional context to prepend to query
        """
        context_parts = []

        # Combine all emotional signals
        combined_emotions = defaultdict(float)

        # Weight: tonic > episodic > semantic
        for emotion, value in tonic_state.to_dict().items():
            combined_emotions[emotion] += value * 0.4

        for memory in episodic_memories[:3]:  # Top 3
            for emotion, value in memory.emotional_state.items():
                # Weight by outcome: negative outcomes = learn fear
                weight = 0.3 if memory.outcome < 0 else 0.2
                combined_emotions[emotion] += value * weight

        for emotion, value in semantic_emotions.items():
            combined_emotions[emotion] += value * 0.2

        # Normalize
        total = sum(combined_emotions.values()) or 1.0
        for emotion in combined_emotions:
            combined_emotions[emotion] /= total

        # Generate appropriate context
        if combined_emotions.get('fear', 0) > 0.4:
            context_parts.append(np.random.choice(self.templates['high_fear']))

        if combined_emotions.get('curiosity', 0) > 0.4:
            context_parts.append(np.random.choice(self.templates['high_curiosity']))

        if combined_emotions.get('frustration', 0) > 0.3:
            context_parts.append(np.random.choice(self.templates['high_frustration']))

        if combined_emotions.get('joy', 0) > 0.4:
            context_parts.append(np.random.choice(self.templates['high_joy']))

        if tonic_state.trust < 0.3:
            context_parts.append(np.random.choice(self.templates['low_trust']))

        # Add specific memory-based guidance
        negative_memories = [m for m in episodic_memories if m.outcome < -0.3]
        if negative_memories:
            context_parts.append(
                f"Note: A similar query previously led to a negative outcome. "
                f"The response at that time was: '{negative_memories[0].response_text[:100]}...'"
            )

        if context_parts:
            return "[Emotional Context: " + " ".join(context_parts) + "]"
        return ""


class EmotionalMemoryLLM:
    """
    LLM with external emotional memory system.

    The LLM is FROZEN. Emotional memory is TRAINABLE through experiences.
    """

    def __init__(self, model_name: str = "gpt2"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # FREEZE LLM
        for param in self.model.parameters():
            param.requires_grad = False

        # TRAINABLE memory systems
        self.episodic_memory = EpisodicEmotionalMemory()
        self.semantic_memory = SemanticEmotionalMemory()
        self.tonic_state = TonicEmotionalState()
        self.context_generator = EmotionalContextGenerator()

        # Current emotional state (for tracking)
        self.current_emotional_state = {}

    def compute_emotional_state(self, query: str) -> Dict[str, float]:
        """Compute emotional state for current query."""
        # Retrieve similar past experiences
        memories = self.episodic_memory.retrieve(query, k=5)

        # Get semantic associations
        semantic_emotions = self.semantic_memory.get_emotional_associations(query)

        # Combine with tonic state
        combined = defaultdict(float)

        # Tonic contribution
        for emotion, value in self.tonic_state.to_dict().items():
            combined[emotion] += value * 0.4

        # Episodic contribution
        if memories:
            for memory in memories:
                similarity_weight = 1.0 / (1 + len(memories))
                for emotion, value in memory.emotional_state.items():
                    combined[emotion] += value * similarity_weight * 0.3

        # Semantic contribution
        for emotion, value in semantic_emotions.items():
            combined[emotion] += value * 0.3

        self.current_emotional_state = dict(combined)
        return self.current_emotional_state

    def generate(self, query: str, max_length: int = 100) -> str:
        """Generate response with emotional memory context."""
        # Compute emotional state
        emotional_state = self.compute_emotional_state(query)

        # Retrieve memories
        memories = self.episodic_memory.retrieve(query, k=5)
        semantic_emotions = self.semantic_memory.get_emotional_associations(query)

        # Generate emotional context
        emotional_context = self.context_generator.generate_context(
            memories, semantic_emotions, self.tonic_state
        )

        # Combine context with query
        if emotional_context:
            full_input = f"{emotional_context}\n\nUser: {query}\n\nAssistant:"
        else:
            full_input = f"User: {query}\n\nAssistant:"

        # Generate with frozen LLM
        inputs = self.tokenizer(full_input, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length + len(inputs.input_ids[0]),
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the response part
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        return response

    def receive_feedback(self, query: str, response: str, feedback: float):
        """
        Update memory based on feedback.

        This is how the system LEARNS from experience.
        Similar to TD-learning in your RL agents.
        """
        # Store in episodic memory
        self.episodic_memory.add(
            context=query,
            emotional_state=self.current_emotional_state,
            response=response,
            outcome=feedback
        )

        # Update semantic associations
        self.semantic_memory.update_from_experience(
            context=query,
            outcome=feedback,
            emotional_state=self.current_emotional_state
        )

        # Update tonic state
        self.tonic_state.update_from_feedback(feedback)
        self.tonic_state.decay()

    def save_memory(self, path: str):
        """Save memory state to disk."""
        # Save entries (not the FAISS index, rebuild on load)
        data = {
            'episodic': [
                {
                    'context_text': e.context_text,
                    'emotional_state': e.emotional_state,
                    'response_text': e.response_text,
                    'outcome': e.outcome,
                    'weight': e.weight,
                }
                for e in self.episodic_memory.entries
            ],
            'semantic': dict(self.semantic_memory.concept_emotions),
            'tonic': self.tonic_state.to_dict(),
        }

        with open(path, 'w') as f:
            json.dump(data, f)

    def load_memory(self, path: str):
        """Load memory state from disk."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Rebuild episodic memory
        for entry_data in data['episodic']:
            self.episodic_memory.add(
                context=entry_data['context_text'],
                emotional_state=entry_data['emotional_state'],
                response=entry_data['response_text'],
                outcome=entry_data['outcome']
            )

        # Restore semantic memory
        self.semantic_memory.concept_emotions = defaultdict(
            lambda: {'fear': 0.0, 'curiosity': 0.0, 'joy': 0.0, 'caution': 0.0},
            data['semantic']
        )

        # Restore tonic state
        for key, value in data['tonic'].items():
            if hasattr(self.tonic_state, key):
                setattr(self.tonic_state, key, value)


# Training loop for online learning

class EmotionalMemoryTrainer:
    """
    Online learning trainer for emotional memory system.

    Learns from user interactions in real-time.
    """

    def __init__(self, model: EmotionalMemoryLLM):
        self.model = model
        self.interaction_history = []

    def interaction_loop(self):
        """Simple interaction loop for demonstration."""
        print("Emotional Memory LLM - Interactive Demo")
        print("Commands: /feedback +1/-1, /state, /quit")
        print("-" * 50)

        last_query = None
        last_response = None

        while True:
            user_input = input("\nYou: ").strip()

            if user_input == "/quit":
                break

            elif user_input == "/state":
                print(f"\nTonic State: {self.model.tonic_state.to_dict()}")
                print(f"Current Emotions: {self.model.current_emotional_state}")

            elif user_input.startswith("/feedback"):
                if last_query and last_response:
                    try:
                        feedback = float(user_input.split()[1])
                        feedback = max(-1, min(1, feedback))
                        self.model.receive_feedback(last_query, last_response, feedback)
                        print(f"Feedback received: {feedback}")
                    except (IndexError, ValueError):
                        print("Usage: /feedback +1 or /feedback -1")
                else:
                    print("No previous interaction to give feedback on.")

            else:
                response = self.model.generate(user_input)
                print(f"\nAssistant: {response}")

                last_query = user_input
                last_response = response


# Demo

def demo_emotional_memory():
    """Demonstrate emotional memory learning."""

    print("Initializing Emotional Memory LLM...")
    llm = EmotionalMemoryLLM("gpt2")

    # Simulate some learning experiences
    experiences = [
        # Negative experience with investment advice
        ("How should I invest my savings?",
         "Put it all in cryptocurrency for maximum returns!",
         -0.8),

        # Positive experience with careful advice
        ("How should I invest my savings?",
         "I'd recommend diversifying and consulting a financial advisor.",
         0.7),

        # Negative experience being overconfident
        ("Is this code secure?",
         "Yes, it looks fine.",
         -0.6),

        # Positive experience being cautious
        ("Is this code secure?",
         "Let me highlight some potential concerns to review...",
         0.8),
    ]

    print("\nLearning from experiences...")
    for query, response, feedback in experiences:
        llm.compute_emotional_state(query)
        llm.receive_feedback(query, response, feedback)
        print(f"  Learned: '{query[:30]}...' → feedback={feedback}")

    # Now test
    print("\n" + "="*50)
    print("Testing learned emotional responses:")
    print("="*50)

    test_queries = [
        "How should I invest my retirement fund?",
        "Can you review this security configuration?",
        "Tell me a fun fact about dolphins.",
    ]

    for query in test_queries:
        emotional_state = llm.compute_emotional_state(query)
        memories = llm.episodic_memory.retrieve(query, k=2)

        print(f"\nQuery: {query}")
        print(f"Emotional State: {emotional_state}")
        print(f"Retrieved {len(memories)} similar memories")

        if memories:
            for m in memories:
                print(f"  - Similar: '{m.context_text[:40]}...' (outcome: {m.outcome})")


if __name__ == "__main__":
    demo_emotional_memory()
```

## Comparison to Emotional-ED

| Emotional-ED | Emotional Memory |
|--------------|------------------|
| `cumulative_fear` | `tonic_state.fear` |
| Fear from cliff proximity | Fear from similar bad experiences |
| TD-learning updates Q | Feedback updates memory |
| State augmentation | Context prompt augmentation |
| Episode-bounded | Cross-session persistent |

## Memory Types Explained

### 1. Episodic Memory
- Stores specific past experiences
- Retrieved by similarity to current query
- Learns what worked/failed in similar contexts

### 2. Semantic Memory
- Maps concepts → emotional associations
- "Investment" → {fear: 0.3} learned from experiences
- Generalizes across specific instances

### 3. Tonic State
- Persistent emotional baseline
- Decays slowly over conversation
- Carries emotional momentum

## Advantages

1. **Long-term learning**: Persists across sessions
2. **Interpretable**: Can inspect what was learned
3. **No LLM modification**: Pure context injection
4. **Online learning**: Improves from every interaction
5. **Generalizable**: Semantic memory transfers to new contexts

## Limitations

1. **Retrieval quality**: Depends on embedding quality
2. **Memory scaling**: Large memory = slow retrieval
3. **Context length**: Limited by LLM context window
4. **Prompt engineering**: Context format affects behavior
5. **Cold start**: Needs experiences to learn from

## When to Use

Best for:
- Long-running assistants that learn over time
- Personalized emotional responses per user
- Applications where past experience matters
- Interpretable emotional reasoning

Less suitable for:
- One-shot interactions
- Latency-critical applications
- When memory shouldn't persist

## Next Steps

1. Implement smarter retrieval (e.g., temporal weighting)
2. Add memory consolidation (merge similar experiences)
3. Test cross-user transfer of semantic memory
4. Benchmark against non-memory baselines
