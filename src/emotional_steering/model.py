"""
Emotional Steering Model

Main class for loading models and generating text with emotional steering.
"""

import torch
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from .directions import DirectionExtractor, SteeringDirection
from .emotions import EMOTIONS, EMOTION_PAIRS


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.9
    do_sample: bool = True


class SteeringHook:
    """Hook that adds steering vector to hidden states."""

    def __init__(self, direction: torch.Tensor, scale: float):
        self.direction = direction
        self.scale = scale

    def __call__(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        steering = self.direction.to(hidden.device, hidden.dtype) * self.scale
        # Add steering to all positions
        hidden = hidden + steering.unsqueeze(0).unsqueeze(0)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden


class EmotionalSteeringModel:
    """
    A language model with emotional steering capabilities.

    Uses activation steering to guide text generation toward
    specific emotional tones without fine-tuning.

    Example:
        >>> model = EmotionalSteeringModel.from_pretrained("HuggingFaceTB/SmolLM3-3B")
        >>> model.extract_directions()
        >>> text = model.generate("The old house was", emotion="fear", scale=5.0)
    """

    # Optimal configurations per model (discovered through testing)
    MODEL_CONFIGS = {
        "HuggingFaceTB/SmolLM3-3B": {"layer": 9, "scale": 5.0},
        "Qwen/Qwen3-4B": {"layer": 18, "scale": 7.0},
        "default": {"layer": 9, "scale": 5.0},
    }

    def __init__(
        self,
        model,
        tokenizer,
        target_layer: int = 9,
        default_scale: float = 5.0,
        device: str = "cuda",
    ):
        """
        Initialize the emotional steering model.

        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            target_layer: Layer to apply steering (0-indexed)
            default_scale: Default steering strength
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer = target_layer
        self.default_scale = default_scale
        self.device = device

        self.directions: Dict[str, SteeringDirection] = {}
        self._active_hooks: List = []

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "HuggingFaceTB/SmolLM3-3B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> "EmotionalSteeringModel":
        """
        Load a pretrained model with emotional steering.

        Args:
            model_name: HuggingFace model name
            device: Device to use
            torch_dtype: Torch dtype for model weights
            **kwargs: Additional arguments for model loading

        Returns:
            EmotionalSteeringModel instance
        """
        print(f"Loading {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            **kwargs,
        )

        # Get optimal config for this model
        config = cls.MODEL_CONFIGS.get(model_name, cls.MODEL_CONFIGS["default"])

        return cls(
            model=model,
            tokenizer=tokenizer,
            target_layer=config["layer"],
            default_scale=config["scale"],
            device=device,
        )

    def extract_directions(
        self,
        emotions: Optional[List[str]] = None,
        custom_pairs: Optional[Dict[str, List]] = None,
    ):
        """
        Extract steering directions for specified emotions.

        Args:
            emotions: List of emotion names to extract (default: all)
            custom_pairs: Custom emotion-pair mappings (overrides defaults)
        """
        if emotions is None:
            emotions = list(EMOTIONS.keys())

        # Get pairs for each emotion
        pairs = custom_pairs or {}
        for emotion in emotions:
            if emotion not in pairs:
                if emotion in EMOTIONS:
                    pairs[emotion] = EMOTIONS[emotion].pairs
                else:
                    raise ValueError(f"Unknown emotion: {emotion}. "
                                     f"Available: {list(EMOTIONS.keys())}")

        print(f"Extracting directions for {len(emotions)} emotions at layer {self.target_layer}...")

        extractor = DirectionExtractor(
            model=self.model,
            tokenizer=self.tokenizer,
            target_layer=self.target_layer,
            device=self.device,
        )

        self.directions = extractor.extract_all_directions(pairs)
        print(f"Extracted directions: {list(self.directions.keys())}")

    def save_directions(self, save_dir: str):
        """Save extracted directions to disk."""
        if not self.directions:
            raise ValueError("No directions extracted. Call extract_directions() first.")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        extractor = DirectionExtractor(
            model=self.model,
            tokenizer=self.tokenizer,
            target_layer=self.target_layer,
            device=self.device,
        )
        extractor.save_directions(self.directions, save_dir)
        print(f"Saved directions to {save_dir}")

    def load_directions(self, load_dir: str):
        """Load directions from disk."""
        self.directions = DirectionExtractor.load_directions(load_dir)
        print(f"Loaded directions: {list(self.directions.keys())}")

    def _apply_steering(self, emotion: str, scale: float):
        """Apply steering hook for an emotion."""
        self._remove_steering()

        if emotion not in self.directions:
            raise ValueError(f"No direction for emotion '{emotion}'. "
                             f"Available: {list(self.directions.keys())}")

        direction = self.directions[emotion].direction
        hook = SteeringHook(direction, scale)

        layer = self.model.model.layers[self.target_layer]
        handle = layer.register_forward_hook(hook)
        self._active_hooks.append(handle)

    def _remove_steering(self):
        """Remove all active steering hooks."""
        for hook in self._active_hooks:
            hook.remove()
        self._active_hooks = []

    def generate(
        self,
        prompt: str,
        emotion: Optional[str] = None,
        scale: Optional[float] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> str:
        """
        Generate text with optional emotional steering.

        Args:
            prompt: Input text prompt
            emotion: Emotion to steer toward (None for no steering)
            scale: Steering strength (default: self.default_scale)
            config: Generation configuration
            **kwargs: Additional generation arguments

        Returns:
            Generated text (without the prompt)
        """
        if config is None:
            config = GenerationConfig()

        if scale is None:
            scale = self.default_scale

        # Apply steering if emotion specified
        if emotion:
            self._apply_steering(emotion, scale)

        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            input_ids = input_ids.to(self.model.device)

            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )

            # Decode only new tokens
            generated = self.tokenizer.decode(
                output[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            return generated

        finally:
            self._remove_steering()

    def generate_comparison(
        self,
        prompt: str,
        emotions: Optional[List[str]] = None,
        scale: Optional[float] = None,
        config: Optional[GenerationConfig] = None,
    ) -> Dict[str, str]:
        """
        Generate text with different emotions for comparison.

        Args:
            prompt: Input text prompt
            emotions: List of emotions to compare (default: all)
            scale: Steering strength
            config: Generation configuration

        Returns:
            Dict mapping emotion names to generated text
        """
        if emotions is None:
            emotions = list(self.directions.keys())

        results = {"baseline": self.generate(prompt, emotion=None, config=config)}

        for emotion in emotions:
            results[emotion] = self.generate(
                prompt, emotion=emotion, scale=scale, config=config
            )

        return results

    def available_emotions(self) -> List[str]:
        """Return list of available emotions."""
        return list(self.directions.keys())

    def __repr__(self):
        return (
            f"EmotionalSteeringModel("
            f"layer={self.target_layer}, "
            f"scale={self.default_scale}, "
            f"emotions={list(self.directions.keys())})"
        )
