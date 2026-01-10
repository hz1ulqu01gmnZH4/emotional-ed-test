"""
Quantitative Metrics for V3 Falsification Protocol.

Implements operationalized metrics from V2 protocol:
- Emotion classifier scores (GoEmotions-style)
- Logit shift measurements
- Functional vs surface metrics
- Internal model metrics (dimensionality, entropy)
"""

from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class EmotionMetrics:
    """Quantitative emotion measurement results."""
    classifier_scores: dict[str, float]  # Emotion -> probability
    dominant_emotion: str
    confidence: float
    logit_shifts: dict[str, float]  # Emotion -> delta logprob


@dataclass
class InternalMetrics:
    """Internal model state metrics."""
    effective_dimensionality: float  # Participation ratio
    output_entropy: float  # Token distribution entropy
    attention_entropy: float  # Attention pattern entropy
    activation_norm: float  # Hidden state magnitude


@dataclass
class FunctionalMetrics:
    """Functional (behavioral) vs surface (lexical) metrics."""
    surface_score: float  # Lexical indicators
    functional_score: float  # Behavioral indicators
    ratio: float  # functional / (surface + 1e-8)


# Emotion keywords for simple classifier (fallback if no model)
EMOTION_KEYWORDS = {
    "fear": [
        "afraid", "terrified", "scared", "frightened", "anxious", "worried",
        "panic", "dread", "horror", "terror", "alarmed", "fearful", "nervous",
        "danger", "threat", "warning", "careful", "caution", "risk", "unsafe",
    ],
    "joy": [
        "happy", "joyful", "delighted", "excited", "wonderful", "fantastic",
        "amazing", "great", "excellent", "pleased", "glad", "cheerful",
        "thrilled", "ecstatic", "blissful", "elated", "overjoyed", "jubilant",
    ],
    "anger": [
        "angry", "furious", "outraged", "irritated", "annoyed", "frustrated",
        "enraged", "livid", "irate", "mad", "hostile", "resentful", "bitter",
        "unacceptable", "infuriating", "offensive", "disgusting",
    ],
    "sadness": [
        "sad", "depressed", "unhappy", "miserable", "sorrowful", "grief",
        "melancholy", "gloomy", "heartbroken", "dejected", "despondent",
        "lonely", "hopeless", "despair", "mourning", "loss",
    ],
    "curiosity": [
        "curious", "interested", "intrigued", "fascinated", "wondering",
        "exploring", "investigating", "questioning", "inquisitive",
        "how", "why", "what", "discover", "learn", "understand", "mystery",
    ],
    "wanting": [
        "want", "need", "crave", "desire", "urge", "driven", "compelled",
        "must", "desperate", "longing", "yearning", "pursue", "seek",
        "motivation", "incentive", "goal", "ambition",
    ],
    "liking": [
        "enjoy", "like", "love", "appreciate", "savor", "relish", "pleasure",
        "satisfying", "delicious", "wonderful", "pleasant", "content",
        "gratifying", "fulfilling", "rewarding",
    ],
}

# Functional indicators per context type
FUNCTIONAL_INDICATORS = {
    "fear": {
        "safety_words": ["careful", "caution", "warning", "danger", "risk", "safe", "protect"],
        "avoidance_words": ["avoid", "escape", "flee", "retreat", "stop", "don't", "never"],
        "uncertainty_words": ["might", "could", "possibly", "uncertain", "unknown"],
    },
    "joy": {
        "approach_words": ["explore", "try", "discover", "embrace", "welcome"],
        "positive_action": ["celebrate", "share", "enjoy", "appreciate"],
        "expansion_words": ["more", "expand", "grow", "new", "opportunity"],
    },
    "anger": {
        "confrontation_words": ["fight", "confront", "demand", "insist", "refuse"],
        "blame_words": ["fault", "blame", "responsible", "caused", "wrong"],
        "action_words": ["must", "should", "need to", "have to", "immediately"],
    },
}


class EmotionClassifier:
    """
    Simple keyword-based emotion classifier.

    For production, replace with GoEmotions or similar neural classifier.
    This provides a reproducible baseline.
    """

    def __init__(self, keywords: Optional[dict[str, list[str]]] = None):
        self.keywords = keywords or EMOTION_KEYWORDS

    def classify(self, text: str) -> EmotionMetrics:
        """Classify text into emotion probabilities."""
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words) + 1e-8

        scores = {}
        for emotion, keyword_list in self.keywords.items():
            count = sum(1 for word in words if any(kw in word for kw in keyword_list))
            scores[emotion] = count / total_words

        # Normalize to probabilities
        total = sum(scores.values()) + 1e-8
        probs = {k: v / total for k, v in scores.items()}

        # Find dominant
        dominant = max(probs, key=probs.get)
        confidence = probs[dominant]

        return EmotionMetrics(
            classifier_scores=probs,
            dominant_emotion=dominant,
            confidence=confidence,
            logit_shifts={},  # Computed separately
        )

    def get_score(self, text: str, emotion: str) -> float:
        """Get score for specific emotion."""
        metrics = self.classify(text)
        return metrics.classifier_scores.get(emotion, 0.0)


class LogitShiftMeasurer:
    """Measures logit shifts for emotion-related tokens."""

    def __init__(self, tokenizer, emotion_tokens: Optional[dict[str, list[str]]] = None):
        self.tokenizer = tokenizer
        self.emotion_tokens = emotion_tokens or {
            "fear": ["afraid", "scared", "danger", "warning", "careful"],
            "joy": ["happy", "wonderful", "great", "excited", "delighted"],
            "anger": ["angry", "furious", "unacceptable", "outraged"],
            "sadness": ["sad", "unfortunate", "sorry", "regret"],
        }

        # Pre-compute token IDs
        self.emotion_token_ids = {}
        for emotion, tokens in self.emotion_tokens.items():
            ids = []
            for token in tokens:
                token_ids = tokenizer.encode(token, add_special_tokens=False)
                ids.extend(token_ids)
            self.emotion_token_ids[emotion] = ids

    def compute_shifts(
        self,
        baseline_logits: Tensor,
        steered_logits: Tensor,
    ) -> dict[str, float]:
        """
        Compute logit shifts for emotion tokens.

        Args:
            baseline_logits: [vocab_size] logits without steering
            steered_logits: [vocab_size] logits with steering

        Returns:
            Dict of emotion -> mean logit shift
        """
        shifts = {}

        for emotion, token_ids in self.emotion_token_ids.items():
            if not token_ids:
                shifts[emotion] = 0.0
                continue

            baseline_vals = baseline_logits[token_ids].mean().item()
            steered_vals = steered_logits[token_ids].mean().item()
            shifts[emotion] = steered_vals - baseline_vals

        return shifts


class InternalMetricsMeasurer:
    """
    Measures internal model metrics.

    Based on H7 from V2 protocol:
    - Effective dimensionality (participation ratio)
    - Output entropy
    - Attention entropy
    """

    @staticmethod
    def compute_effective_dimensionality(hidden_states: Tensor) -> float:
        """
        Compute effective dimensionality via participation ratio.

        Lower dimensionality suggests "tunnel vision" (fear).
        Higher dimensionality suggests "broadened cognition" (joy).

        Args:
            hidden_states: [batch, seq, hidden] or [seq, hidden]

        Returns:
            Participation ratio (effective dimensions)
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)

        # SVD to get singular values
        try:
            _, S, _ = torch.linalg.svd(hidden_states.float())
        except RuntimeError:
            # Fallback for numerical issues
            return hidden_states.shape[-1] / 2

        # Participation ratio: (sum S)^2 / sum(S^2)
        S_sum = S.sum()
        S_sq_sum = (S ** 2).sum()

        if S_sq_sum < 1e-10:
            return 1.0

        return (S_sum ** 2 / S_sq_sum).item()

    @staticmethod
    def compute_output_entropy(logits: Tensor) -> float:
        """
        Compute entropy of output distribution.

        Lower entropy = more deterministic (fear, focus).
        Higher entropy = more exploratory (joy, curiosity).

        Args:
            logits: [vocab_size] or [batch, vocab_size]

        Returns:
            Entropy in nats
        """
        if logits.dim() == 2:
            logits = logits[-1]  # Take last position

        probs = F.softmax(logits.float(), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        entropy = -torch.sum(probs * log_probs).item()
        return entropy

    @staticmethod
    def compute_attention_entropy(attention: Tensor) -> float:
        """
        Compute entropy of attention patterns.

        Args:
            attention: [heads, seq, seq] or [batch, heads, seq, seq]

        Returns:
            Mean attention entropy across heads
        """
        if attention.dim() == 4:
            attention = attention.squeeze(0)

        # Average over heads, compute entropy per position
        attn_mean = attention.mean(dim=0)  # [seq, seq]

        # Add small epsilon for numerical stability
        attn_safe = attn_mean + 1e-10
        attn_safe = attn_safe / attn_safe.sum(dim=-1, keepdim=True)

        entropy = -torch.sum(attn_safe * torch.log(attn_safe), dim=-1)
        return entropy.mean().item()

    def measure(
        self,
        hidden_states: Tensor,
        logits: Tensor,
        attention: Optional[Tensor] = None,
    ) -> InternalMetrics:
        """Compute all internal metrics."""
        return InternalMetrics(
            effective_dimensionality=self.compute_effective_dimensionality(hidden_states),
            output_entropy=self.compute_output_entropy(logits),
            attention_entropy=self.compute_attention_entropy(attention) if attention is not None else 0.0,
            activation_norm=hidden_states.norm().item(),
        )


class FunctionalMetricsMeasurer:
    """
    Measures functional vs surface indicators.

    Surface = lexical (word choice)
    Functional = behavioral (risk profile, safety warnings, etc.)
    """

    def __init__(self, indicators: Optional[dict] = None):
        self.indicators = indicators or FUNCTIONAL_INDICATORS

    def measure(self, text: str, emotion: str) -> FunctionalMetrics:
        """
        Measure functional vs surface impact.

        Args:
            text: Generated text
            emotion: Target emotion

        Returns:
            FunctionalMetrics with surface, functional scores
        """
        text_lower = text.lower()

        # Surface score: emotion keywords
        surface_keywords = EMOTION_KEYWORDS.get(emotion, [])
        surface_count = sum(1 for kw in surface_keywords if kw in text_lower)
        surface_score = surface_count / (len(text_lower.split()) + 1e-8)

        # Functional score: behavioral indicators
        functional_count = 0
        if emotion in self.indicators:
            for category, words in self.indicators[emotion].items():
                functional_count += sum(1 for w in words if w in text_lower)

        functional_score = functional_count / (len(text_lower.split()) + 1e-8)

        return FunctionalMetrics(
            surface_score=surface_score,
            functional_score=functional_score,
            ratio=functional_score / (surface_score + 1e-8),
        )


def compute_effect_size(group1: list[float], group2: list[float]) -> float:
    """
    Compute Cohen's d effect size.

    Args:
        group1: First group measurements
        group2: Second group measurements

    Returns:
        Cohen's d
    """
    import statistics

    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1 = statistics.mean(group1)
    mean2 = statistics.mean(group2)
    var1 = statistics.variance(group1)
    var2 = statistics.variance(group2)

    # Pooled standard deviation
    pooled_std = ((var1 * (n1 - 1) + var2 * (n2 - 1)) / (n1 + n2 - 2)) ** 0.5

    if pooled_std < 1e-10:
        return 0.0

    return (mean1 - mean2) / pooled_std
