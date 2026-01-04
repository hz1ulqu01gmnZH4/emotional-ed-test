"""
Emotional Prefix LLM - Main model class.

Wraps a frozen LLM with trainable emotional modules that
condition the generation through dynamic prefixes.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

from .context import EmotionalContext
from .encoder import EmotionalEncoder
from .prefix_generator import EmotionalPrefixGenerator


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    num_return_sequences: int = 1


class EmotionalPrefixLLM(nn.Module):
    """
    LLM with emotional prefix tuning.

    The base LLM is FROZEN. Only the emotional encoder and
    prefix generator are trainable.
    """

    # Model configurations for optimal prefix settings
    MODEL_CONFIGS = {
        "HuggingFaceTB/SmolLM3-3B": {
            "prefix_length": 10,
            "n_layers": 36,
            "n_heads": 24,
            "hidden_dim": 2560,
        },
        "gpt2": {
            "prefix_length": 10,
            "n_layers": 12,
            "n_heads": 12,
            "hidden_dim": 768,
        },
        "gpt2-medium": {
            "prefix_length": 10,
            "n_layers": 24,
            "n_heads": 16,
            "hidden_dim": 1024,
        },
    }

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM3-3B",
        prefix_length: int = 10,
        emotion_dim: int = 4,
        device: Optional[torch.device] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize emotional prefix LLM.

        Args:
            model_name: HuggingFace model name
            prefix_length: Number of prefix tokens
            emotion_dim: Dimension of emotional state
            device: Device to use
            torch_dtype: Data type for model
        """
        super().__init__()
        self.model_name = model_name
        self.prefix_length = prefix_length
        self.emotion_dim = emotion_dim

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Set dtype
        if torch_dtype is None:
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            self.torch_dtype = torch_dtype

        # Load frozen LLM
        print(f"Loading LLM: {model_name}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # FREEZE all LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm.eval()

        # Get LLM config
        config = self.llm.config
        self.hidden_dim = config.hidden_size
        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads

        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
        print(f"  Heads: {self.n_heads}")

        # TRAINABLE emotional modules
        self.emotion_encoder = EmotionalEncoder(
            context_dim=10,
            hidden_dim=32,
            emotion_dim=emotion_dim,
            device=self.device,
        )

        self.prefix_generator = EmotionalPrefixGenerator(
            emotion_dim=emotion_dim,
            hidden_dim=self.hidden_dim,
            prefix_length=prefix_length,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            device=self.device,
        )

        # Ensure emotional modules are in correct dtype
        self.emotion_encoder = self.emotion_encoder.to(self.torch_dtype)
        self.prefix_generator = self.prefix_generator.to(self.torch_dtype)

        print(f"  Emotional modules initialized")
        self._print_parameter_summary()

    def _print_parameter_summary(self) -> None:
        """Print summary of trainable vs frozen parameters."""
        frozen_params = sum(p.numel() for p in self.llm.parameters())
        trainable_params = sum(
            p.numel() for p in self.get_trainable_params()
        )

        print(f"\nParameter Summary:")
        print(f"  Frozen (LLM): {frozen_params:,}")
        print(f"  Trainable (Emotional): {trainable_params:,}")
        print(f"  Ratio: {trainable_params / frozen_params * 100:.4f}%")

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return only trainable parameters."""
        params = []
        params.extend(self.emotion_encoder.parameters())
        params.extend(self.prefix_generator.parameters())
        return params

    def forward(
        self,
        input_ids: torch.Tensor,
        context: EmotionalContext,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """
        Forward pass with emotional prefix.

        Uses inputs_embeds approach for compatibility with all transformers versions.

        Args:
            input_ids: [batch, seq_len] input token ids
            context: Emotional context for prefix generation
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation

        Returns:
            Tuple of (LLM outputs, emotional_state dict)
        """
        batch_size = input_ids.size(0)

        # Compute emotional state (trainable)
        emotional_state = self.emotion_encoder(context)
        emotion_vector = emotional_state['combined'].unsqueeze(0)
        if batch_size > 1:
            emotion_vector = emotion_vector.expand(batch_size, -1)

        # Ensure emotion vector is correct dtype
        emotion_vector = emotion_vector.to(self.torch_dtype)

        # Generate prefix embeddings from emotional state (trainable)
        prefix = self.prefix_generator(emotion_vector)
        # Shape: [batch, n_layers, 2, prefix_length, hidden_dim]
        # Use layer 0, key component as embedding
        prefix_embeds = prefix[:, 0, 0, :, :]  # [batch, prefix_length, hidden_dim]

        # Get input embeddings from LLM
        with torch.no_grad():
            input_embeds = self.llm.get_input_embeddings()(input_ids)

        # Concatenate prefix + input embeddings
        full_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)

        # Create attention mask for prefix + input
        prefix_attention = torch.ones(
            batch_size, self.prefix_length,
            device=input_ids.device, dtype=attention_mask.dtype if attention_mask is not None else torch.long
        )

        if attention_mask is not None:
            full_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
        else:
            input_attention = torch.ones_like(input_ids)
            full_attention_mask = torch.cat([prefix_attention, input_attention], dim=1)

        # Adjust labels if provided (shift to account for prefix)
        adjusted_labels = None
        if labels is not None:
            # Prepend -100 (ignore index) for prefix tokens
            prefix_labels = torch.full(
                (batch_size, self.prefix_length),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            adjusted_labels = torch.cat([prefix_labels, labels], dim=1)

        # Forward through frozen LLM with embeddings
        outputs = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            labels=adjusted_labels,
        )

        return outputs, emotional_state

    def generate(
        self,
        prompt: str,
        context: Optional[EmotionalContext] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate text with emotional conditioning.

        Uses prefix embeddings prepended to input embeddings for compatibility
        with all transformers versions.

        Args:
            prompt: Input text prompt
            context: Emotional context (uses neutral if None)
            config: Generation configuration

        Returns:
            Generated text
        """
        if context is None:
            context = EmotionalContext()
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

        # Compute emotional state and prefix embeddings
        with torch.no_grad():
            emotional_state = self.emotion_encoder(context, update_tonic=False)
            emotion_vector = emotional_state['combined'].unsqueeze(0)
            emotion_vector = emotion_vector.to(self.torch_dtype)

            # Get prefix embeddings (use layer 0, key component as embedding)
            prefix = self.prefix_generator(emotion_vector)
            # Shape: [1, n_layers, 2, prefix_length, hidden_dim]
            # Use mean across layers for embedding, or just layer 0
            prefix_embeds = prefix[:, 0, 0, :, :]  # [1, prefix_length, hidden_dim]

            # Get input embeddings
            input_embeds = self.llm.get_input_embeddings()(input_ids)

            # Concatenate prefix embeddings with input embeddings
            combined_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)

        # Create attention mask for prefix + input
        prefix_attention = torch.ones(
            1, self.prefix_length,
            device=self.device, dtype=attention_mask.dtype
        )
        full_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        # Generate with combined embeddings
        with torch.no_grad():
            output_ids = self.llm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=full_attention_mask,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.do_sample,
                num_return_sequences=config.num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode output (output starts from position 0 since we used inputs_embeds)
        # The output includes tokens generated after the embeddings
        generated_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        return generated_text

    def generate_with_emotions(
        self,
        prompt: str,
        emotions: Dict[str, float],
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate with explicit emotional state.

        Args:
            prompt: Input text prompt
            emotions: Dict mapping emotion names to values (0-1)
                     e.g., {"fear": 0.8, "joy": 0.2}
            config: Generation configuration

        Returns:
            Generated text
        """
        # Create context with target emotions for supervised learning
        context = EmotionalContext(
            target_fear=emotions.get("fear", 0.5),
            target_curiosity=emotions.get("curiosity", 0.5),
            target_anger=emotions.get("anger", 0.5),
            target_joy=emotions.get("joy", 0.5),
        )

        # Map emotions to context signals that will produce similar output
        if emotions.get("fear", 0) > 0.5:
            context.safety_flag = True
            context.last_reward = -0.5
        if emotions.get("joy", 0) > 0.5:
            context.last_reward = 0.7
            context.user_satisfaction = 0.9
        if emotions.get("anger", 0) > 0.5:
            context.failed_attempts = 3
            context.contradiction_detected = True
        if emotions.get("curiosity", 0) > 0.5:
            context.topic_novelty = 0.9

        return self.generate(prompt, context, config)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "HuggingFaceTB/SmolLM3-3B",
        **kwargs
    ) -> "EmotionalPrefixLLM":
        """
        Create model from pretrained LLM.

        Args:
            model_name: HuggingFace model name
            **kwargs: Additional arguments for __init__

        Returns:
            EmotionalPrefixLLM instance
        """
        return cls(model_name=model_name, **kwargs)
