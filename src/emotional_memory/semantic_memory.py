"""
Semantic Emotional Memory for concept-emotion associations.

Learns which topics/concepts are associated with which emotions.
"""

from collections import defaultdict
from typing import Dict, List, Set, Optional
import re


class SemanticEmotionalMemory:
    """
    Semantic memory mapping concepts to emotional associations.

    Learns which topics/concepts are associated with which emotions
    through experience.
    """

    # Common stopwords to filter out
    STOPWORDS: Set[str] = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
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
        'it', 'its', 'they', 'them', 'their', 'about', 'like',
    }

    # Default emotions to track
    DEFAULT_EMOTIONS = ['fear', 'curiosity', 'joy', 'caution', 'frustration']

    def __init__(self, learning_rate: float = 0.1):
        """
        Initialize semantic memory.

        Args:
            learning_rate: How fast to update associations
        """
        self.learning_rate = learning_rate

        # Concept → emotional association
        self.concept_emotions: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {e: 0.0 for e in self.DEFAULT_EMOTIONS}
        )

        # Track concept frequency for normalization
        self.concept_counts: Dict[str, int] = defaultdict(int)

    def extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts from text.

        Args:
            text: Input text

        Returns:
            List of concept words
        """
        # Simple word extraction, filter stopwords and short words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return [w for w in words if w not in self.STOPWORDS and len(w) > 2]

    def get_emotional_associations(self, text: str) -> Dict[str, float]:
        """
        Get emotional associations for concepts in text.

        Args:
            text: Input text

        Returns:
            Dictionary of emotion → association strength
        """
        concepts = self.extract_concepts(text)

        if not concepts:
            return {e: 0.0 for e in self.DEFAULT_EMOTIONS}

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

    def update_from_experience(
        self,
        context: str,
        outcome: float,
        emotional_state: Dict[str, float],
    ) -> None:
        """
        Update concept-emotion associations from experience.

        Args:
            context: Context text
            outcome: Feedback score (-1 to 1)
            emotional_state: Emotional state at that time
        """
        concepts = self.extract_concepts(context)

        for concept in concepts:
            self.concept_counts[concept] += 1

            # Initialize if new
            if concept not in self.concept_emotions:
                self.concept_emotions[concept] = {
                    e: 0.0 for e in self.DEFAULT_EMOTIONS
                }

            # If outcome was bad and we weren't cautious enough
            if outcome < 0 and emotional_state.get('fear', 0) < 0.5:
                self.concept_emotions[concept]['fear'] += self.learning_rate * abs(outcome)
                self.concept_emotions[concept]['caution'] += self.learning_rate * abs(outcome)

            # If outcome was good
            if outcome > 0:
                self.concept_emotions[concept]['joy'] += self.learning_rate * outcome
                # Slightly reduce fear if we were fearful but outcome was good
                if emotional_state.get('fear', 0) > 0.3:
                    self.concept_emotions[concept]['fear'] *= (1 - self.learning_rate)

            # If high curiosity led to good outcome
            if outcome > 0 and emotional_state.get('curiosity', 0) > 0.3:
                self.concept_emotions[concept]['curiosity'] += self.learning_rate * outcome

            # If frustration led to bad outcome, reduce frustration association
            if outcome < 0 and emotional_state.get('frustration', 0) > 0.3:
                self.concept_emotions[concept]['frustration'] += self.learning_rate * abs(outcome)

            # Clamp values
            for emotion in self.concept_emotions[concept]:
                self.concept_emotions[concept][emotion] = max(
                    0.0, min(1.0, self.concept_emotions[concept][emotion])
                )

    def get_high_emotion_concepts(
        self,
        emotion: str,
        threshold: float = 0.3,
    ) -> List[str]:
        """
        Get concepts with high association to an emotion.

        Args:
            emotion: Target emotion
            threshold: Minimum association strength

        Returns:
            List of concept names
        """
        result = []
        for concept, emotions in self.concept_emotions.items():
            if emotions.get(emotion, 0) >= threshold:
                result.append(concept)
        return sorted(result, key=lambda c: self.concept_emotions[c][emotion], reverse=True)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "concept_emotions": dict(self.concept_emotions),
            "concept_counts": dict(self.concept_counts),
        }

    @classmethod
    def from_dict(cls, data: Dict, learning_rate: float = 0.1) -> "SemanticEmotionalMemory":
        """Create from dictionary."""
        memory = cls(learning_rate=learning_rate)
        memory.concept_emotions = defaultdict(
            lambda: {e: 0.0 for e in cls.DEFAULT_EMOTIONS},
            data.get("concept_emotions", {})
        )
        memory.concept_counts = defaultdict(int, data.get("concept_counts", {}))
        return memory

    def size(self) -> int:
        """Get number of concepts in memory."""
        return len(self.concept_emotions)

    def clear(self) -> None:
        """Clear all associations."""
        self.concept_emotions.clear()
        self.concept_counts.clear()
