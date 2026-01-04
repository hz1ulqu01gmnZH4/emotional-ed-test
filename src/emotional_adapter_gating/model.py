"""
Emotional Adapter LLM - Main model class.

Full LLM with emotional adapters inserted at each layer.
LLM weights are FROZEN. Only adapters and emotion encoder train.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

from .state import EmotionalState
from .adapter import EmotionalAdapter
from .encoder import SimpleEmotionalEncoder


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True


class EmotionalAdapterLLM(nn.Module):
    """
    LLM with emotional adapters inserted at each layer.

    The base LLM is FROZEN. Only adapters and emotion encoder train.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        adapter_dim: int = 64,
        emotion_dim: int = 6,
        gate_type: str = "scalar",
        dropout: float = 0.1,
        adapter_layers: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize emotional adapter LLM.

        Args:
            model_name: HuggingFace model name
            adapter_dim: Bottleneck dimension for adapters
            emotion_dim: Dimension of emotional state
            gate_type: Gate type for adapters
            dropout: Dropout probability
            adapter_layers: Which layers to add adapters (None = all)
            device: Device to use
            torch_dtype: Data type for model
        """
        super().__init__()
        self.model_name = model_name
        self.adapter_dim = adapter_dim
        self.emotion_dim = emotion_dim
        self.gate_type = gate_type

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Set dtype
        if torch_dtype is None:
            self.torch_dtype = torch.float32  # Adapters work better with float32
        else:
            self.torch_dtype = torch_dtype

        # Load base model
        print(f"Loading LLM: {model_name}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # FREEZE base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        # Get model config
        config = self.base_model.config
        self.hidden_dim = config.hidden_size
        self.n_layers = config.num_hidden_layers

        # Determine which layers get adapters
        if adapter_layers is None:
            self.adapter_layer_indices = list(range(self.n_layers))
        else:
            self.adapter_layer_indices = adapter_layers

        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
        print(f"  Adapter layers: {len(self.adapter_layer_indices)}")

        # Get actual device where model is loaded (may differ with device_map="auto")
        actual_device = next(self.base_model.parameters()).device
        self.device = actual_device

        # Create emotional adapters (TRAINABLE)
        self.adapters = nn.ModuleDict()
        for layer_idx in self.adapter_layer_indices:
            self.adapters[str(layer_idx)] = EmotionalAdapter(
                hidden_dim=self.hidden_dim,
                adapter_dim=adapter_dim,
                emotion_dim=emotion_dim,
                gate_type=gate_type,
                dropout=dropout,
                device=actual_device,
            )

        # Simple emotional encoder (TRAINABLE)
        # Use the actual device where the model is loaded
        actual_device = next(self.base_model.parameters()).device
        self.emotion_encoder = SimpleEmotionalEncoder(
            emotion_dim=emotion_dim,
            device=actual_device,
        )

        # Layer-specific emotion weighting (optional, TRAINABLE)
        # Ensure it's on the correct device
        actual_device = next(self.base_model.parameters()).device
        self.layer_emotion_weights = nn.Parameter(
            torch.ones(len(self.adapter_layer_indices), emotion_dim, device=actual_device)
        )

        # Register forward hooks to inject adapters
        self._register_adapter_hooks()

        # Current emotional state (set during forward)
        self._current_emotional_state: Optional[torch.Tensor] = None

        self._print_parameter_summary()

    def _print_parameter_summary(self) -> None:
        """Print summary of trainable vs frozen parameters."""
        frozen_params = sum(p.numel() for p in self.base_model.parameters())
        trainable_params = sum(
            p.numel() for p in self.get_trainable_params()
        )

        print(f"\nParameter Summary:")
        print(f"  Frozen (LLM): {frozen_params:,}")
        print(f"  Trainable (Adapters): {trainable_params:,}")
        print(f"  Ratio: {trainable_params / frozen_params * 100:.4f}%")

    def _register_adapter_hooks(self) -> None:
        """Register forward hooks to inject adapters after each layer."""
        self._hooks = []

        # Get the transformer layers
        if hasattr(self.base_model, 'transformer'):
            layers = self.base_model.transformer.h
        elif hasattr(self.base_model, 'model'):
            if hasattr(self.base_model.model, 'layers'):
                layers = self.base_model.model.layers
            else:
                layers = self.base_model.model.decoder.layers
        else:
            raise ValueError(f"Unknown model architecture: {type(self.base_model)}")

        for i, layer in enumerate(layers):
            if i in self.adapter_layer_indices:
                hook = layer.register_forward_hook(
                    self._create_adapter_hook(i)
                )
                self._hooks.append(hook)

    def _create_adapter_hook(self, layer_idx: int):
        """Create a forward hook for a specific layer."""
        def hook(module, input, output):
            if self._current_emotional_state is None:
                return output

            # Get hidden states from output
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Apply adapter with emotional gating
            adapter = self.adapters[str(layer_idx)]
            layer_pos = self.adapter_layer_indices.index(layer_idx)

            # Layer-specific emotion weighting
            weighted_emotion = self._current_emotional_state * torch.sigmoid(
                self.layer_emotion_weights[layer_pos]
            )

            # Apply adapter
            modified_hidden = adapter(hidden_states, weighted_emotion)

            # Return modified output
            if isinstance(output, tuple):
                return (modified_hidden,) + output[1:]
            return modified_hidden

        return hook

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return trainable parameters."""
        params = []
        for adapter in self.adapters.values():
            params.extend(adapter.parameters())
        params.append(self.layer_emotion_weights)
        return params

    def forward(
        self,
        input_ids: torch.Tensor,
        emotional_state: EmotionalState,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, torch.Tensor]:
        """
        Forward pass with emotional adapter modulation.

        Args:
            input_ids: [batch, seq_len] input token ids
            emotional_state: EmotionalState object
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation

        Returns:
            Tuple of (model outputs, emotional_state tensor)
        """
        # Convert emotional state to tensor and set for hooks
        emotion_tensor = self.emotion_encoder(emotional_state)
        self._current_emotional_state = emotion_tensor

        # Forward through base model (hooks will apply adapters)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Clear state after forward
        self._current_emotional_state = None

        return outputs, emotion_tensor

    def generate(
        self,
        prompt: str,
        emotional_state: Optional[EmotionalState] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate text with emotional conditioning.

        Args:
            prompt: Input text prompt
            emotional_state: EmotionalState (uses neutral if None)
            config: Generation configuration

        Returns:
            Generated text
        """
        if emotional_state is None:
            emotional_state = EmotionalState.neutral()
        if config is None:
            config = GenerationConfig()

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Set emotional state for hooks
        emotion_tensor = self.emotion_encoder(emotional_state)
        self._current_emotional_state = emotion_tensor

        # Generate
        with torch.no_grad():
            output_ids = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Clear state
        self._current_emotional_state = None

        # Decode output
        generated_text = self.tokenizer.decode(
            output_ids[0][input_ids.size(1):],
            skip_special_tokens=True
        )

        return generated_text

    def reset_emotional_state(self) -> None:
        """Reset emotional encoder's tonic state."""
        self.emotion_encoder.reset_tonic()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "gpt2",
        **kwargs
    ) -> "EmotionalAdapterLLM":
        """
        Create model from pretrained LLM.

        Args:
            model_name: HuggingFace model name
            **kwargs: Additional arguments for __init__

        Returns:
            EmotionalAdapterLLM instance
        """
        return cls(model_name=model_name, **kwargs)
