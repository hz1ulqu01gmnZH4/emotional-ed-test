"""
Emotional Reward LLM - Complete LLM with Emotional Reward Model.

The LLM is FROZEN. Only ERM and modulator are TRAINABLE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .signals import EmotionalSignals
from .reward_model import EmotionalRewardModel
from .logit_modulator import LogitModulator
from .temperature_modulator import TemperatureModulator
from .fear_module import FearModule


@dataclass
class GenerationOutput:
    """Output from emotional generation."""

    text: str
    emotions: List[EmotionalSignals]
    temperatures: List[float]


class EmotionalRewardLLM(nn.Module):
    """
    Complete LLM with Emotional Reward Model.

    The LLM is FROZEN. Only ERM, logit modulator, and fear module are TRAINABLE.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Emotional Reward LLM.

        Args:
            model_name: HuggingFace model name
            device: Device to use
        """
        super().__init__()
        self.model_name = model_name

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Load tokenizer and frozen LLM
        print(f"Loading LLM: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # FREEZE LLM
        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm.eval()

        # Get actual device from LLM
        actual_device = next(self.llm.parameters()).device

        # Get model dimensions
        self.hidden_dim = self.llm.config.hidden_size
        self.vocab_size = self.llm.config.vocab_size

        # TRAINABLE components
        self.erm = EmotionalRewardModel(
            hidden_dim=self.hidden_dim,
            n_emotions=6,
        ).to(actual_device)

        self.logit_modulator = LogitModulator(
            vocab_size=self.vocab_size,
            n_emotions=6,
        ).to(actual_device)

        self.fear_module = FearModule(
            hidden_dim=self.hidden_dim,
        ).to(actual_device)

        self.temp_modulator = TemperatureModulator()

        # Initialize token categories
        self._init_token_categories()

        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Device: {actual_device}")

    def _init_token_categories(self) -> None:
        """Initialize cautious and exploratory token categories."""
        self.logit_modulator.set_token_categories(
            self.tokenizer,
            cautious_phrases=[
                "caution", "careful", "uncertain", "might", "perhaps",
                "I'm not sure", "be careful", "consider", "however",
                "on the other hand", "potential risk", "it depends",
                "warning", "risk", "danger", "may", "could",
            ],
            exploratory_phrases=[
                "interesting", "curious", "explore", "what if", "imagine",
                "fascinating", "wonder", "could you tell me more",
                "let's dive deeper", "that's intriguing", "amazing",
                "discover", "learn", "exciting",
            ],
        )

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return trainable parameters."""
        params = list(self.erm.parameters())
        params.extend(self.logit_modulator.parameters())
        params.extend(self.fear_module.parameters())
        return params

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_emotions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[EmotionalSignals]]:
        """
        Forward pass with emotional modulation.

        Args:
            input_ids: [batch, seq_len] input token IDs
            attention_mask: [batch, seq_len] attention mask
            return_emotions: Whether to return emotional signals

        Returns:
            Modified logits, and optionally EmotionalSignals
        """
        # Get LLM outputs with hidden states
        with torch.no_grad():
            llm_outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        logits = llm_outputs.logits
        # Use last layer hidden states for ERM
        hidden_states = llm_outputs.hidden_states[-1]

        # Compute emotional signals (trainable)
        emotional_signals, _ = self.erm(hidden_states)

        # Modulate logits (trainable)
        modified_logits = self.logit_modulator(logits, emotional_signals)

        if return_emotions:
            return modified_logits, emotional_signals
        return modified_logits, None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: Optional[float] = None,
        return_emotions: bool = False,
        use_emotional_temperature: bool = True,
    ) -> GenerationOutput:
        """
        Generate text with emotional modulation.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Base temperature (None = use modulator base)
            return_emotions: Whether to track emotions per token
            use_emotional_temperature: Whether to adjust temp by emotion

        Returns:
            GenerationOutput with text, emotions, and temperatures
        """
        if temperature is not None:
            self.temp_modulator.set_base_temperature(temperature)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # For autoregressive generation
        generated_ids = input_ids.clone()
        all_emotions = []
        all_temperatures = []

        # Reset tonic states for new generation
        self.erm.reset_tonic()
        self.fear_module.reset()

        for _ in range(max_new_tokens):
            # Get modulated logits
            logits, emotions = self.forward(
                generated_ids,
                attention_mask,
                return_emotions=True,
            )
            all_emotions.append(emotions)

            # Get next token logits
            next_logits = logits[:, -1, :]

            # Compute temperature
            if use_emotional_temperature:
                temp = self.temp_modulator.compute_temperature(emotions)
            else:
                temp = self.temp_modulator.base_temperature
            all_temperatures.append(temp)

            # Sample
            probs = F.softmax(next_logits / temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones_like(next_token),
            ], dim=-1)

            # Stop at EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Decode output
        output_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        return GenerationOutput(
            text=output_text,
            emotions=all_emotions if return_emotions else [],
            temperatures=all_temperatures if return_emotions else [],
        )

    def get_emotional_state(self, text: str) -> EmotionalSignals:
        """
        Get emotional state for given text.

        Args:
            text: Input text

        Returns:
            EmotionalSignals for the text
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)

        _, emotions = self.forward(input_ids, return_emotions=True)
        return emotions

    def get_fear_level(
        self,
        text: str,
        feedback: Optional[float] = None,
    ) -> float:
        """
        Get fear level for given text using specialized fear module.

        Args:
            text: Input text
            feedback: Optional feedback signal

        Returns:
            Fear level (0 to 1)
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.llm(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

        return self.fear_module(hidden_states, feedback)

    def reset_emotional_state(self) -> None:
        """Reset all emotional states (tonic emotions)."""
        self.erm.reset_tonic()
        self.fear_module.reset()

    def save_trainable_weights(self, path: str) -> None:
        """
        Save trainable component weights.

        Args:
            path: Path to save file
        """
        torch.save({
            "erm": self.erm.state_dict(),
            "logit_modulator": self.logit_modulator.state_dict(),
            "fear_module": self.fear_module.state_dict(),
        }, path)
        print(f"Saved trainable weights to {path}")

    def load_trainable_weights(self, path: str) -> None:
        """
        Load trainable component weights.

        Args:
            path: Path to saved file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.erm.load_state_dict(checkpoint["erm"])
        self.logit_modulator.load_state_dict(checkpoint["logit_modulator"])
        self.fear_module.load_state_dict(checkpoint["fear_module"])
        print(f"Loaded trainable weights from {path}")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "gpt2",
        weights_path: Optional[str] = None,
        **kwargs,
    ) -> "EmotionalRewardLLM":
        """
        Create model from pretrained LLM and optionally load weights.

        Args:
            model_name: HuggingFace model name
            weights_path: Optional path to trainable weights
            **kwargs: Additional arguments

        Returns:
            EmotionalRewardLLM instance
        """
        model = cls(model_name=model_name, **kwargs)
        if weights_path is not None:
            model.load_trainable_weights(weights_path)
        return model
