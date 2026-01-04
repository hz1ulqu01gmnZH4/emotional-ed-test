"""
Direction extraction for activation steering.

Extracts steering vectors by computing the difference between
emotional and neutral sentence activations.
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class SteeringDirection:
    """A steering direction for a specific emotion."""
    emotion: str
    layer: int
    direction: torch.Tensor
    n_pairs: int  # Number of pairs used to compute this direction

    def save(self, path: str):
        """Save direction to file."""
        torch.save({
            "emotion": self.emotion,
            "layer": self.layer,
            "direction": self.direction,
            "n_pairs": self.n_pairs,
        }, path)

    @classmethod
    def load(cls, path: str) -> "SteeringDirection":
        """Load direction from file."""
        data = torch.load(path, weights_only=True)
        return cls(
            emotion=data["emotion"],
            layer=data["layer"],
            direction=data["direction"],
            n_pairs=data["n_pairs"],
        )


class DirectionExtractor:
    """
    Extracts steering directions from a language model.

    Uses contrastive activation addition: computes the difference
    between emotional and neutral sentence activations to find
    directions in activation space that correspond to emotions.
    """

    def __init__(
        self,
        model,
        tokenizer,
        target_layer: int = 9,  # Best layer for SmolLM3-3B
        device: str = "cuda",
    ):
        """
        Initialize the direction extractor.

        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            target_layer: Layer to extract directions from (0-indexed)
            device: Device to use for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer = target_layer
        self.device = device

        # Validate layer index
        n_layers = len(model.model.layers)
        if target_layer >= n_layers:
            raise ValueError(f"Layer {target_layer} out of range (model has {n_layers} layers)")

    def extract_direction(
        self,
        pairs: List[Tuple[str, str]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Extract a steering direction from sentence pairs.

        Args:
            pairs: List of (neutral, emotional) sentence pairs
            normalize: Whether to normalize the direction to unit length

        Returns:
            Steering direction tensor of shape (hidden_size,)
        """
        diffs = []

        for neutral, emotional in pairs:
            # Tokenize
            n_tokens = self.tokenizer(neutral, return_tensors="pt").to(self.device)
            e_tokens = self.tokenizer(emotional, return_tensors="pt").to(self.device)

            # Get hidden states
            with torch.no_grad():
                n_out = self.model(**n_tokens, output_hidden_states=True)
                e_out = self.model(**e_tokens, output_hidden_states=True)

                # Extract last token activation from target layer
                # hidden_states is a tuple of (n_layers + 1) tensors
                # Index 0 is embeddings, so layer L is at index L+1
                n_hidden = n_out.hidden_states[self.target_layer + 1][:, -1, :]
                e_hidden = e_out.hidden_states[self.target_layer + 1][:, -1, :]

                diff = e_hidden - n_hidden
                diffs.append(diff.cpu().float())

        # Average all differences
        direction = torch.cat(diffs, dim=0).mean(dim=0)

        # Normalize
        if normalize:
            direction = direction / direction.norm()

        return direction

    def extract_all_directions(
        self,
        emotion_pairs: Dict[str, List[Tuple[str, str]]],
        normalize: bool = True,
    ) -> Dict[str, SteeringDirection]:
        """
        Extract directions for multiple emotions.

        Args:
            emotion_pairs: Dict mapping emotion names to lists of pairs
            normalize: Whether to normalize directions

        Returns:
            Dict mapping emotion names to SteeringDirection objects
        """
        directions = {}

        for emotion, pairs in emotion_pairs.items():
            direction = self.extract_direction(pairs, normalize=normalize)
            directions[emotion] = SteeringDirection(
                emotion=emotion,
                layer=self.target_layer,
                direction=direction,
                n_pairs=len(pairs),
            )

        return directions

    def save_directions(
        self,
        directions: Dict[str, SteeringDirection],
        save_dir: str,
    ):
        """Save all directions to a directory."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "target_layer": self.target_layer,
            "emotions": list(directions.keys()),
        }

        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        for emotion, direction in directions.items():
            direction.save(str(save_path / f"{emotion}.pt"))

    @classmethod
    def load_directions(cls, load_dir: str) -> Dict[str, SteeringDirection]:
        """Load all directions from a directory."""
        load_path = Path(load_dir)

        with open(load_path / "metadata.json") as f:
            metadata = json.load(f)

        directions = {}
        for emotion in metadata["emotions"]:
            directions[emotion] = SteeringDirection.load(
                str(load_path / f"{emotion}.pt")
            )

        return directions
