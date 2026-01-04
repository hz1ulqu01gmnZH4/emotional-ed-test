"""
Logit Modulator - Modifies LLM logits based on emotional signals.

Similar to how FearEDAgent biases Q-values toward safe actions,
this module biases token probabilities based on emotional state.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .signals import EmotionalSignals


class LogitModulator(nn.Module):
    """
    Modulates LLM logits based on emotional signals.

    Learns which tokens to boost/suppress for each emotion.
    """

    def __init__(
        self,
        vocab_size: int,
        n_emotions: int = 6,
        init_scale: float = 0.01,
    ):
        """
        Initialize logit modulator.

        Args:
            vocab_size: Size of vocabulary
            n_emotions: Number of emotion dimensions
            init_scale: Initial scale for bias weights
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.n_emotions = n_emotions

        # Learnable bias per emotion per token
        # This learns which tokens to boost/suppress for each emotion
        self.emotion_token_biases = nn.Parameter(
            torch.randn(n_emotions, vocab_size) * init_scale
        )

        # Scaling factor (how strongly emotions affect logits)
        self.emotion_scale = nn.Parameter(torch.tensor(0.5))

        # Token category masks (optional rule-based modulation)
        self.register_buffer(
            "cautious_token_mask",
            torch.zeros(vocab_size),
        )
        self.register_buffer(
            "exploratory_token_mask",
            torch.zeros(vocab_size),
        )

        # Scaling for rule-based modulation
        self.rule_scale = 2.0

    def set_token_categories(
        self,
        tokenizer,
        cautious_phrases: Optional[List[str]] = None,
        exploratory_phrases: Optional[List[str]] = None,
    ) -> None:
        """
        Set token masks for different emotional categories.

        Args:
            tokenizer: HuggingFace tokenizer
            cautious_phrases: Phrases indicating caution
            exploratory_phrases: Phrases indicating exploration
        """
        if cautious_phrases is None:
            cautious_phrases = [
                "caution", "careful", "uncertain", "might", "perhaps",
                "I'm not sure", "be careful", "consider", "however",
                "on the other hand", "potential risk", "it depends",
                "warning", "risk", "danger", "may", "could",
            ]

        if exploratory_phrases is None:
            exploratory_phrases = [
                "interesting", "curious", "explore", "what if", "imagine",
                "fascinating", "wonder", "could you tell me more",
                "let's dive deeper", "that's intriguing", "amazing",
                "discover", "learn", "exciting",
            ]

        # Reset masks
        self.cautious_token_mask.zero_()
        self.exploratory_token_mask.zero_()

        # Set cautious tokens
        for phrase in cautious_phrases:
            try:
                tokens = tokenizer.encode(phrase, add_special_tokens=False)
                for t in tokens:
                    if t < self.vocab_size:
                        self.cautious_token_mask[t] = 1.0
            except Exception:
                pass  # Skip phrases that can't be encoded

        # Set exploratory tokens
        for phrase in exploratory_phrases:
            try:
                tokens = tokenizer.encode(phrase, add_special_tokens=False)
                for t in tokens:
                    if t < self.vocab_size:
                        self.exploratory_token_mask[t] = 1.0
            except Exception:
                pass

    def forward(
        self,
        logits: torch.Tensor,
        emotional_signals: EmotionalSignals,
        use_rule_based: bool = True,
    ) -> torch.Tensor:
        """
        Modify logits based on emotional state.

        Args:
            logits: [batch, seq_len, vocab_size] from LLM
            emotional_signals: Current emotional state
            use_rule_based: Whether to apply rule-based token boosting

        Returns:
            Modified logits
        """
        device = logits.device
        emotions = emotional_signals.to_tensor(device=device)

        # Compute emotional bias
        # [n_emotions] @ [n_emotions, vocab_size] → [vocab_size]
        bias = self.emotion_scale * (emotions @ self.emotion_token_biases.to(device))

        # Apply bias to logits
        # Broadcast: [batch, seq, vocab] + [vocab]
        modified_logits = logits + bias.unsqueeze(0).unsqueeze(0)

        # Additional rule-based modulation
        if use_rule_based:
            cautious_mask = self.cautious_token_mask.to(device)
            exploratory_mask = self.exploratory_token_mask.to(device)

            if emotional_signals.fear > 0.5:
                # Boost cautious tokens when fearful
                modified_logits = modified_logits + (
                    emotional_signals.fear * cautious_mask * self.rule_scale
                )

            if emotional_signals.curiosity > 0.5:
                # Boost exploratory tokens when curious
                modified_logits = modified_logits + (
                    emotional_signals.curiosity * exploratory_mask * self.rule_scale
                )

            if emotional_signals.anger > 0.5:
                # When angry, slightly suppress exploratory tokens
                modified_logits = modified_logits - (
                    emotional_signals.anger * exploratory_mask * self.rule_scale * 0.5
                )

        return modified_logits

    def get_token_biases_for_emotion(self, emotion_idx: int, top_k: int = 10):
        """
        Get top tokens boosted for a specific emotion.

        Args:
            emotion_idx: Index of emotion (0=fear, 1=curiosity, etc.)
            top_k: Number of top tokens to return

        Returns:
            Tuple of (token indices, bias values)
        """
        biases = self.emotion_token_biases[emotion_idx]
        values, indices = torch.topk(biases, top_k)
        return indices.tolist(), values.tolist()

    def set_rule_scale(self, scale: float) -> None:
        """Set the scale for rule-based modulation."""
        self.rule_scale = scale
