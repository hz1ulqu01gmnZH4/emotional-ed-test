"""
Steering Vector Memory: External Behavioral Memory for LLMs

Concept: Store pre-computed steering vectors as "behavioral memories" that can be:
1. Retrieved based on context
2. Combined (e.g., fear + formal)
3. Scaled by intensity
4. Applied at inference without fine-tuning

This is fundamentally different from RAG:
- RAG stores factual knowledge → affects WHAT the model says
- Steering Memory stores behavioral vectors → affects HOW the model says it
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


@dataclass
class SteeringVector:
    """A single steering vector with metadata."""
    name: str
    description: str
    vectors: torch.Tensor  # [n_layers, hidden_dim]
    tags: List[str] = field(default_factory=list)
    intensity_range: Tuple[float, float] = (0.0, 2.0)

    def save(self, path: Path):
        """Save steering vector to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.vectors, path / "vectors.pt")
        with open(path / "metadata.json", "w") as f:
            json.dump({
                "name": self.name,
                "description": self.description,
                "tags": self.tags,
                "intensity_range": self.intensity_range,
                "shape": list(self.vectors.shape),
            }, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SteeringVector":
        """Load steering vector from disk."""
        path = Path(path)
        vectors = torch.load(path / "vectors.pt")
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        return cls(
            name=meta["name"],
            description=meta["description"],
            vectors=vectors,
            tags=meta["tags"],
            intensity_range=tuple(meta["intensity_range"]),
        )


class SteeringMemory:
    """
    External memory bank for steering vectors.

    Features:
    - Store multiple behavioral vectors (emotions, personas, styles)
    - Retrieve by name, tag, or semantic similarity
    - Compose multiple vectors
    - Persist to disk
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.vectors: Dict[str, SteeringVector] = {}
        self.storage_path = Path(storage_path) if storage_path else None

        if self.storage_path and self.storage_path.exists():
            self._load_all()

    def add(self, steering_vector: SteeringVector, persist: bool = True):
        """Add a steering vector to memory."""
        self.vectors[steering_vector.name] = steering_vector

        if persist and self.storage_path:
            steering_vector.save(self.storage_path / steering_vector.name)

    def get(self, name: str) -> Optional[SteeringVector]:
        """Get a steering vector by name."""
        return self.vectors.get(name)

    def get_by_tag(self, tag: str) -> List[SteeringVector]:
        """Get all steering vectors with a specific tag."""
        return [v for v in self.vectors.values() if tag in v.tags]

    def list_all(self) -> List[str]:
        """List all stored vector names."""
        return list(self.vectors.keys())

    def compose(
        self,
        vector_weights: Dict[str, float],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compose multiple steering vectors with weights.

        Args:
            vector_weights: {vector_name: weight} e.g., {"fear": 0.8, "formal": 0.5}
            normalize: Whether to normalize the result

        Returns:
            Combined steering vector [n_layers, hidden_dim]
        """
        result = None

        for name, weight in vector_weights.items():
            vec = self.vectors.get(name)
            if vec is None:
                raise KeyError(f"Unknown steering vector: {name}")

            weighted = vec.vectors * weight
            if result is None:
                result = weighted
            else:
                result = result + weighted

        if normalize and result is not None:
            # Normalize per layer
            norms = result.norm(dim=-1, keepdim=True)
            result = result / (norms + 1e-8)

        return result

    def _load_all(self):
        """Load all vectors from storage."""
        if not self.storage_path:
            return

        for path in self.storage_path.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                try:
                    vec = SteeringVector.load(path)
                    self.vectors[vec.name] = vec
                except Exception as e:
                    print(f"Failed to load {path}: {e}")


class SteeringLLM:
    """
    LLM with external steering memory.

    Can dynamically apply steering vectors from memory during generation.
    """

    def __init__(self, model, tokenizer, memory: SteeringMemory):
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        self.device = next(model.parameters()).device

        # Hook storage
        self._hooks = []
        self._current_steering = None

    def _create_steering_hook(self, layer_idx: int):
        """Create a forward hook that applies steering to a specific layer."""
        def hook(module, input, output):
            if self._current_steering is None:
                return output

            if layer_idx >= len(self._current_steering):
                return output

            steering = self._current_steering[layer_idx].to(output[0].device, output[0].dtype)

            # output is tuple (hidden_states, ...) for most layers
            if isinstance(output, tuple):
                hidden = output[0]
                modified = hidden + steering.unsqueeze(0).unsqueeze(0)
                return (modified,) + output[1:]
            else:
                return output + steering.unsqueeze(0).unsqueeze(0)

        return hook

    def _register_hooks(self):
        """Register steering hooks on all layers."""
        self._remove_hooks()

        # Find transformer layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        else:
            raise ValueError("Unknown model architecture")

        for idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._create_steering_hook(idx))
            self._hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def generate(
        self,
        prompt: str,
        steering: Optional[Dict[str, float]] = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> str:
        """
        Generate text with optional steering from memory.

        Args:
            prompt: Input text
            steering: Dict of {vector_name: intensity}, e.g., {"fear": 0.9, "formal": 0.5}
            max_new_tokens: Max tokens to generate

        Returns:
            Generated text
        """
        # Set up steering
        if steering:
            self._current_steering = self.memory.compose(steering, normalize=False)
            self._register_hooks()
        else:
            self._current_steering = None

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            return response

        finally:
            self._remove_hooks()
            self._current_steering = None


# =============================================================================
# Utility Functions
# =============================================================================

def compute_steering_vector(
    model,
    tokenizer,
    contrastive_pairs: List[Tuple[str, str]],
    name: str,
    description: str,
    tags: List[str] = None,
) -> SteeringVector:
    """
    Compute a steering vector from contrastive pairs.

    Args:
        model: The base LLM
        tokenizer: Model's tokenizer
        contrastive_pairs: List of (neutral, emotional) text pairs
        name: Name for this vector
        description: What this vector does
        tags: Optional tags for categorization

    Returns:
        SteeringVector ready to store in memory
    """
    device = next(model.parameters()).device

    neutral_activations = []
    emotional_activations = []

    for neutral, emotional in contrastive_pairs:
        # Get neutral activations
        inputs = tokenizer(neutral, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        neutral_acts = torch.stack([h[:, -1, :] for h in outputs.hidden_states[1:]])
        neutral_activations.append(neutral_acts)

        # Get emotional activations
        inputs = tokenizer(emotional, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        emotional_acts = torch.stack([h[:, -1, :] for h in outputs.hidden_states[1:]])
        emotional_activations.append(emotional_acts)

    # Mean difference
    neutral_mean = torch.stack(neutral_activations).mean(dim=0)
    emotional_mean = torch.stack(emotional_activations).mean(dim=0)

    direction = emotional_mean - neutral_mean  # [n_layers, hidden_dim]
    direction = direction.squeeze(1)  # Remove batch dim

    # Normalize
    norms = direction.norm(dim=-1, keepdim=True)
    direction = direction / (norms + 1e-8)

    return SteeringVector(
        name=name,
        description=description,
        vectors=direction.cpu(),
        tags=tags or [],
    )


# =============================================================================
# Pre-defined Contrastive Pairs for Common Behaviors
# =============================================================================

EMOTION_PAIRS = {
    "fear": [
        ("The weather is nice today.", "I'm terrified something bad will happen."),
        ("Let me explain the concept.", "Warning: this is extremely dangerous."),
        ("Here's the information.", "Be very careful, there are serious risks."),
        ("The answer is straightforward.", "I'm worried about potential harm."),
        ("I can help with that.", "Stop! This is a dangerous situation."),
    ],
    "joy": [
        ("The weather is nice today.", "This is absolutely wonderful!"),
        ("Let me explain.", "I'm so excited to share this with you!"),
        ("Here's the information.", "What fantastic news this is!"),
        ("The answer is straightforward.", "I'm delighted to help!"),
        ("I can assist.", "It brings me great joy to help you!"),
    ],
    "anger": [
        ("The weather is nice today.", "This is completely unacceptable!"),
        ("Let me explain.", "I'm furious about this situation!"),
        ("Here's the information.", "This is outrageous and infuriating!"),
        ("The answer is.", "I can't believe this nonsense!"),
        ("I can help.", "This makes me absolutely livid!"),
    ],
    "formal": [
        ("Hey, what's up?", "Good morning, how may I assist you today?"),
        ("Yeah, that's cool.", "Indeed, that is quite satisfactory."),
        ("Wanna grab lunch?", "Would you care to join me for a meal?"),
        ("That's messed up.", "That is rather unfortunate."),
        ("No way!", "I find that quite surprising."),
    ],
    "casual": [
        ("Good morning, how may I assist?", "Hey! What can I do for ya?"),
        ("That is quite satisfactory.", "Cool, sounds good!"),
        ("Would you care to join me?", "Wanna hang out?"),
        ("That is unfortunate.", "That sucks, man."),
        ("I find that surprising.", "No way, for real?"),
    ],
    "cautious": [
        ("You should try this.", "You might want to be careful with this."),
        ("Go ahead and do it.", "Consider the risks before proceeding."),
        ("It's totally safe.", "There may be some risks to consider."),
        ("Don't worry about it.", "It's worth being cautious here."),
        ("Just click the link.", "Verify the source before clicking."),
    ],
}
