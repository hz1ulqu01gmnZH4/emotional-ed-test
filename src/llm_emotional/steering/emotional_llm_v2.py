"""
Emotional Steering LLM - Version 2.

Uses PCA-based directions with layer-weighted steering for
more effective emotional modulation.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .steering_hooks_v2 import LayerWeightedSteeringManager


class EmotionalSteeringLLMv2:
    """
    LLM wrapper with PCA-based emotional steering.

    This version uses:
    - PCA-extracted directions (more robust than difference-in-means)
    - Layer-weighted steering (applies more steering to effective layers)
    - Combined emotion support (multiple emotions at once)
    """

    EMOTIONS = {"fear", "curiosity", "anger", "joy"}

    def __init__(
        self,
        model_name: str,
        direction_bank_path: Optional[str] = None,
        steering_scale: float = 1.0,
        device: Optional[str] = None,
    ):
        """
        Initialize the emotional steering LLM.

        Args:
            model_name: HuggingFace model name
            direction_bank_path: Path to PCA-trained direction bank
            steering_scale: Global scaling factor for steering
            device: Device to use (None for auto)
        """
        self.model_name = model_name
        self.steering_scale = steering_scale

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)

        # Determine architecture
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.n_layers = len(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self.n_layers = len(self.model.transformer.h)
        else:
            raise ValueError("Cannot determine model architecture")

        self.hidden_dim = self.model.config.hidden_size

        # Initialize steering manager with uniform weights
        self.steering_manager = LayerWeightedSteeringManager(
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
        )
        self.steering_manager.scale = steering_scale
        self.steering_manager.install(self.model)

        # Direction storage
        self.directions: dict[str, Tensor] = {}
        self.layer_weights: dict[str, dict[int, float]] = {}
        self.current_emotion_state: dict[str, float] = {e: 0.0 for e in self.EMOTIONS}

        # Load directions if provided
        if direction_bank_path is not None:
            self.load_directions(direction_bank_path)

    def load_directions(self, path: str):
        """
        Load PCA-trained directions from disk.

        Args:
            path: Path to direction bank JSON file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Direction bank not found: {path}")

        with open(path) as f:
            data = json.load(f)

        # Validate dimensions
        if data["hidden_dim"] != self.hidden_dim:
            raise ValueError(
                f"Direction bank hidden_dim ({data['hidden_dim']}) "
                f"doesn't match model ({self.hidden_dim})"
            )

        # Load directions
        for emotion, direction_list in data["directions"].items():
            self.directions[emotion] = torch.tensor(direction_list, dtype=torch.float32)

        # Load layer weights if available
        if "layer_weights" in data:
            for emotion, weights in data["layer_weights"].items():
                self.layer_weights[emotion] = {
                    int(k): v for k, v in weights.items()
                }

        print(f"Loaded PCA directions for: {list(self.directions.keys())}")
        if self.layer_weights:
            print(f"Layer weights available: {list(self.layer_weights.keys())}")

    def set_emotional_state(
        self,
        fear: float = 0.0,
        curiosity: float = 0.0,
        anger: float = 0.0,
        joy: float = 0.0,
    ):
        """
        Set the emotional state for generation.

        Args:
            fear: Fear intensity (0.0 to 1.0)
            curiosity: Curiosity intensity (0.0 to 1.0)
            anger: Anger/determination intensity (0.0 to 1.0)
            joy: Joy intensity (0.0 to 1.0)
        """
        self.current_emotion_state = {
            "fear": fear,
            "curiosity": curiosity,
            "anger": anger,
            "joy": joy,
        }

        # Combine directions weighted by intensity
        combined_direction = torch.zeros(self.hidden_dim)
        combined_weights = {i: 0.0 for i in range(self.n_layers)}

        total_intensity = 0.0

        for emotion, intensity in self.current_emotion_state.items():
            if intensity > 0 and emotion in self.directions:
                combined_direction += intensity * self.directions[emotion]
                total_intensity += intensity

                # Combine layer weights
                if emotion in self.layer_weights:
                    for layer_idx, weight in self.layer_weights[emotion].items():
                        combined_weights[layer_idx] += intensity * weight

        # Normalize if multiple emotions
        if total_intensity > 0:
            combined_direction = combined_direction / total_intensity
            combined_weights = {
                k: v / total_intensity
                for k, v in combined_weights.items()
            }

        # Update steering manager
        if total_intensity > 0:
            self.steering_manager.set_layer_weights(combined_weights)
            self.steering_manager.set_steering(combined_direction)
        else:
            self.steering_manager.clear_steering()

    def generate_completion(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate a completion with emotional steering.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample (vs greedy)
            **kwargs: Additional generation arguments

        Returns:
            Generated text (response only, not including prompt)
        """
        # Format as chat message
        messages = [
            {"role": "user", "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode response only
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return response.strip()

    def get_current_state(self) -> dict:
        """Get current emotional state."""
        return self.current_emotion_state.copy()
