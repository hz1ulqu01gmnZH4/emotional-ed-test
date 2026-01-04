"""
Emotional Steering LLM - main wrapper combining all components.

Provides a high-level interface for:
- Loading models with emotional steering capability
- Setting emotional states
- Generating text with emotional modulation
- Comparing outputs across emotional states

NO FALLBACK POLICY: If model loading fails, FAIL LOUDLY.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
import torch.nn as nn

from .direction_bank import EmotionalDirectionBank, DirectionBankError
from .steering_hooks import MultiLayerSteeringManager, SteeringHookError


class ModelLoadError(Exception):
    """
    Model failed to load.

    NO FALLBACK to smaller models - fix the issue or use explicit model name.
    """
    pass


class EmotionalSteeringLLM:
    """
    LLM with emotional activation steering.

    Wraps a frozen LLM and adds emotional steering capability via
    learned direction vectors added to hidden states.

    Usage:
        llm = EmotionalSteeringLLM("Qwen/Qwen3-1.7B-Instruct")
        llm.load_directions("path/to/direction_bank.json")

        llm.set_emotional_state(fear=0.7)
        fearful_output = llm.generate("Tell me about heights")

        llm.set_emotional_state(joy=0.8)
        joyful_output = llm.generate("Tell me about heights")
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # Fallback-free: this MUST work

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        direction_bank_path: Optional[Union[str, Path]] = None,
        steering_scale: float = 1.0,
        target_layers: Optional[List[int]] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize emotional steering LLM.

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen3-1.7B-Instruct")
            direction_bank_path: Optional path to pre-trained direction bank
            steering_scale: Global scale for steering magnitude
            target_layers: Which layers to steer (default: all)
            device: Device to load model on (default: auto)
            torch_dtype: Data type for model (default: auto)

        Raises:
            ModelLoadError: If model fails to load (NO FALLBACK)
        """
        self.model_name = model_name
        self.steering_scale = steering_scale
        self.target_layers = target_layers

        # Lazy imports to avoid loading transformers unless needed
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            raise ModelLoadError(
                f"transformers library not installed: {e}. "
                "Install with: pip install transformers"
            )

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load tokenizer for {model_name}: {e}"
            )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        try:
            load_kwargs: Dict[str, Any] = {}

            if device is not None:
                load_kwargs['device_map'] = device
            else:
                # Auto device selection
                load_kwargs['device_map'] = 'auto'

            if torch_dtype is not None:
                load_kwargs['torch_dtype'] = torch_dtype
            else:
                # Auto dtype (use bfloat16 if available)
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    load_kwargs['torch_dtype'] = torch.bfloat16
                else:
                    load_kwargs['torch_dtype'] = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model {model_name}: {e}. "
                "Ensure the model exists and you have sufficient memory."
            )

        # Put model in eval mode
        self.model.eval()

        # Get model config
        self.hidden_dim = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers

        # Initialize direction bank
        self.direction_bank = EmotionalDirectionBank(
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers
        )

        # Install steering hooks BEFORE loading directions
        # (load_directions updates the manager's bank reference)
        self.steering_manager = MultiLayerSteeringManager(
            self.direction_bank,
            layers=target_layers,
            scale=steering_scale,
        )
        self.steering_manager.install(self.model)

        # Load pre-trained directions if provided
        if direction_bank_path is not None:
            self.load_directions(direction_bank_path)

        # Current emotional state
        self._emotional_state: Dict[str, float] = {
            'fear': 0.0,
            'curiosity': 0.0,
            'anger': 0.0,
            'joy': 0.0,
        }

    def load_directions(self, path: Union[str, Path]) -> None:
        """
        Load pre-trained direction bank.

        Args:
            path: Path to direction bank JSON

        Raises:
            DirectionBankError: If loading fails (NO FALLBACK)
        """
        path = Path(path)
        self.direction_bank = EmotionalDirectionBank.load(path)

        # Verify dimensions match
        if self.direction_bank.hidden_dim != self.hidden_dim:
            raise DirectionBankError(
                f"Direction bank hidden_dim ({self.direction_bank.hidden_dim}) "
                f"doesn't match model ({self.hidden_dim})"
            )
        if self.direction_bank.n_layers != self.n_layers:
            raise DirectionBankError(
                f"Direction bank n_layers ({self.direction_bank.n_layers}) "
                f"doesn't match model ({self.n_layers})"
            )

        # Update steering manager's reference
        self.steering_manager.direction_bank = self.direction_bank
        for hook in self.steering_manager.hooks:
            hook.direction_bank = self.direction_bank

    def save_directions(self, path: Union[str, Path]) -> None:
        """
        Save current direction bank.

        Args:
            path: Path to save to
        """
        self.direction_bank.save(Path(path))

    @property
    def emotional_state(self) -> Dict[str, float]:
        """Get current emotional state."""
        return self._emotional_state.copy()

    def set_emotional_state(self, **emotions: float) -> None:
        """
        Set current emotional state.

        Args:
            **emotions: Emotion intensities (e.g., fear=0.7, joy=0.3)
        """
        # Validate and clamp values
        for emotion, intensity in emotions.items():
            if emotion not in self._emotional_state:
                raise ValueError(
                    f"Unknown emotion: {emotion}. "
                    f"Must be one of: {list(self._emotional_state.keys())}"
                )
            self._emotional_state[emotion] = max(0.0, min(1.0, intensity))

        # Update steering manager
        self.steering_manager.set_emotional_state(**self._emotional_state)

    def clear_emotional_state(self) -> None:
        """Reset all emotions to neutral (0.0)."""
        self._emotional_state = {e: 0.0 for e in self._emotional_state}
        self.steering_manager.clear_emotional_state()

    def enable_steering(self) -> None:
        """Enable emotional steering."""
        self.steering_manager.enable()

    def disable_steering(self) -> None:
        """Disable steering (baseline model behavior)."""
        self.steering_manager.disable()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate text with current emotional state.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample (vs greedy)
            **kwargs: Additional generation kwargs

        Returns:
            Generated text (prompt + completion)
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_completion(
        self,
        prompt: str,
        **kwargs,
    ) -> str:
        """
        Generate just the completion (without prompt).

        Args:
            prompt: Input prompt
            **kwargs: Generation kwargs

        Returns:
            Generated completion only
        """
        full_output = self.generate(prompt, **kwargs)
        # Remove prompt from output
        if full_output.startswith(prompt):
            return full_output[len(prompt):].strip()
        return full_output

    def compare_emotions(
        self,
        prompt: str,
        states: List[Dict[str, float]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Compare outputs across different emotional states.

        Args:
            prompt: Input prompt to generate from
            states: List of emotional state dicts to compare
            **kwargs: Generation kwargs

        Returns:
            List of dicts with 'state' and 'output' keys
        """
        results = []

        for state in states:
            self.set_emotional_state(**state)
            output = self.generate_completion(prompt, **kwargs)
            results.append({
                'state': state.copy(),
                'output': output,
            })

        # Reset to neutral
        self.clear_emotional_state()

        return results

    def get_info(self) -> Dict[str, Any]:
        """Get information about the model and steering setup."""
        return {
            'model_name': self.model_name,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'n_steering_hooks': len(self.steering_manager.hooks),
            'steering_scale': self.steering_scale,
            'target_layers': self.target_layers,
            'emotional_state': self.emotional_state,
            'directions_learned': {
                e: self.direction_bank.learned[e]
                for e in self.direction_bank.EMOTIONS
            },
        }

    def __repr__(self) -> str:
        state_str = ", ".join(
            f"{e}={v:.2f}"
            for e, v in self._emotional_state.items()
            if v > 0
        )
        return (
            f"EmotionalSteeringLLM("
            f"model={self.model_name}, "
            f"layers={self.n_layers}, "
            f"state=[{state_str or 'neutral'}])"
        )
