"""
Emotional Context Generator for creating prompts.

Combines episodic, semantic, and tonic emotional information
into natural language context that the LLM can understand.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional

from .tonic_state import TonicEmotionalState
from .episodic_memory import EpisodicMemoryEntry


class EmotionalContextGenerator:
    """
    Generates emotional context prompts for the LLM.

    Combines episodic, semantic, and tonic emotional information
    into natural language context that the LLM can understand.
    """

    # Templates for different emotional states
    TEMPLATES = {
        'high_fear': [
            "Be cautious in your response. Similar queries have led to issues before.",
            "Proceed carefully. There may be safety considerations here.",
            "Take a measured approach. Past interactions suggest caution is warranted.",
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
        'high_anxiety': [
            "Take extra care with this response.",
            "Double-check facts and be thorough.",
        ],
    }

    def __init__(self, include_memory_refs: bool = True):
        """
        Initialize context generator.

        Args:
            include_memory_refs: Whether to include references to past memories
        """
        self.include_memory_refs = include_memory_refs

    def generate_context(
        self,
        episodic_memories: List[EpisodicMemoryEntry],
        semantic_emotions: Dict[str, float],
        tonic_state: TonicEmotionalState,
    ) -> str:
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

        # Combine all emotional signals with different weights
        combined_emotions = self._combine_emotions(
            episodic_memories,
            semantic_emotions,
            tonic_state,
        )

        # Generate appropriate context based on emotional state
        if combined_emotions.get('fear', 0) > 0.4:
            context_parts.append(random.choice(self.TEMPLATES['high_fear']))

        if combined_emotions.get('curiosity', 0) > 0.4:
            context_parts.append(random.choice(self.TEMPLATES['high_curiosity']))

        if combined_emotions.get('frustration', 0) > 0.3:
            context_parts.append(random.choice(self.TEMPLATES['high_frustration']))

        if combined_emotions.get('joy', 0) > 0.4:
            context_parts.append(random.choice(self.TEMPLATES['high_joy']))

        if tonic_state.trust < 0.3:
            context_parts.append(random.choice(self.TEMPLATES['low_trust']))

        if combined_emotions.get('anxiety', 0) > 0.4:
            context_parts.append(random.choice(self.TEMPLATES['high_anxiety']))

        # Add specific memory-based guidance if enabled
        if self.include_memory_refs and episodic_memories:
            negative_memories = [m for m in episodic_memories if m.outcome < -0.3]
            if negative_memories:
                memory = negative_memories[0]
                response_preview = memory.response_text[:80] + "..." if len(memory.response_text) > 80 else memory.response_text
                context_parts.append(
                    f"Note: A similar query previously had a negative outcome. "
                    f"Consider avoiding: '{response_preview}'"
                )

        if context_parts:
            return "[Emotional Context: " + " ".join(context_parts) + "]"
        return ""

    def _combine_emotions(
        self,
        episodic_memories: List[EpisodicMemoryEntry],
        semantic_emotions: Dict[str, float],
        tonic_state: TonicEmotionalState,
    ) -> Dict[str, float]:
        """
        Combine emotional signals from different sources.

        Weight: tonic > episodic > semantic
        """
        combined = defaultdict(float)

        # Tonic contribution (40%)
        for emotion, value in tonic_state.to_dict().items():
            combined[emotion] += value * 0.4

        # Episodic contribution (30%)
        for memory in episodic_memories[:3]:  # Top 3
            weight = 0.3 if memory.outcome < 0 else 0.2
            for emotion, value in memory.emotional_state.items():
                combined[emotion] += value * weight

        # Semantic contribution (20%)
        for emotion, value in semantic_emotions.items():
            combined[emotion] += value * 0.2

        # Normalize
        total = sum(combined.values()) or 1.0
        for emotion in combined:
            combined[emotion] /= total

        return dict(combined)

    def get_emotional_summary(
        self,
        tonic_state: TonicEmotionalState,
    ) -> str:
        """
        Get a brief summary of current emotional state.

        Args:
            tonic_state: Current tonic state

        Returns:
            Brief text summary
        """
        dominant = tonic_state.dominant_emotion()
        valence = tonic_state.overall_valence()

        if valence > 0.3:
            mood = "positive"
        elif valence < -0.3:
            mood = "cautious"
        else:
            mood = "neutral"

        return f"Mood: {mood}, Dominant: {dominant}"
