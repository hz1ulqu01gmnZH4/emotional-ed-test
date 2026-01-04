"""
Direction Learner - learn emotional directions from contrastive pairs.

Uses Difference-in-Means method:
    direction = mean(emotional_activations) - mean(neutral_activations)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from .direction_bank import EmotionalDirectionBank


class DirectionLearnerError(Exception):
    """Raised when direction learning fails. NO FALLBACK."""
    pass


class EmotionalDirectionLearner:
    """
    Learn emotional directions from contrastive (neutral, emotional) pairs.

    Algorithm (Difference-in-Means):
    1. For each contrastive pair, extract hidden states at each layer
    2. Compute mean activation for neutral texts
    3. Compute mean activation for emotional texts
    4. Direction = emotional_mean - neutral_mean
    5. Normalize direction to unit length

    This gives a direction vector that points from "neutral" to "emotional"
    in the model's activation space.
    """

    def __init__(self, model: nn.Module, tokenizer):
        """
        Initialize direction learner.

        Args:
            model: The frozen LLM to extract activations from
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer

        # Extract model config
        config = model.config
        self.hidden_dim = config.hidden_size
        self.n_layers = config.num_hidden_layers

        # Ensure model is in eval mode
        self.model.eval()

        # Validate model supports hidden state extraction
        self._validate_model()

    def _validate_model(self) -> None:
        """Validate model can output hidden states."""
        # Try a forward pass with hidden states
        test_input = self.tokenizer("test", return_tensors='pt')

        try:
            with torch.no_grad():
                outputs = self.model(
                    **test_input.to(self.model.device),
                    output_hidden_states=True
                )

            if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
                raise DirectionLearnerError(
                    "Model does not output hidden states. "
                    "Ensure model supports output_hidden_states=True"
                )

            # Validate shape
            n_hidden = len(outputs.hidden_states)
            expected = self.n_layers + 1  # +1 for embedding layer
            if n_hidden != expected:
                raise DirectionLearnerError(
                    f"Unexpected hidden states count: {n_hidden}, expected {expected}"
                )

        except Exception as e:
            if isinstance(e, DirectionLearnerError):
                raise
            raise DirectionLearnerError(
                f"Model validation failed: {e}"
            )

    @torch.no_grad()
    def extract_activations(
        self,
        text: str,
        layer_idx: int,
        pooling: str = 'mean'
    ) -> Tensor:
        """
        Extract hidden states at specified layer for text.

        Args:
            text: Input text
            layer_idx: Layer to extract from (0 to n_layers-1)
            pooling: How to pool across sequence ('mean', 'last', 'first')

        Returns:
            Pooled hidden state tensor of shape [hidden_dim]
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)

        # hidden_states[0] is embedding, hidden_states[1] is layer 0, etc.
        hidden = outputs.hidden_states[layer_idx + 1]  # [batch, seq_len, hidden_dim]

        # Pool across sequence dimension
        if pooling == 'mean':
            pooled = hidden.mean(dim=1)  # [batch, hidden_dim]
        elif pooling == 'last':
            pooled = hidden[:, -1, :]
        elif pooling == 'first':
            pooled = hidden[:, 0, :]
        else:
            raise DirectionLearnerError(f"Unknown pooling method: {pooling}")

        return pooled.squeeze(0).cpu()  # [hidden_dim]

    def learn_direction(
        self,
        contrastive_pairs: List[Tuple[str, str]],
        emotion: str,
        normalize: bool = True,
        verbose: bool = False,
    ) -> Tensor:
        """
        Learn direction vector using Difference-in-Means.

        Direction = mean(emotional_activations) - mean(neutral_activations)

        Args:
            contrastive_pairs: List of (neutral_text, emotional_text) tuples
            emotion: Name of the emotion (for logging)
            normalize: Whether to normalize directions to unit length
            verbose: Print progress

        Returns:
            Learned direction tensor of shape [n_layers, hidden_dim]
        """
        if len(contrastive_pairs) < 3:
            raise DirectionLearnerError(
                f"Need at least 3 contrastive pairs for reliable direction, "
                f"got {len(contrastive_pairs)}"
            )

        directions = []

        for layer_idx in range(self.n_layers):
            if verbose:
                print(f"  Learning {emotion} direction for layer {layer_idx}...")

            neutral_activations = []
            emotional_activations = []

            for neutral_text, emotional_text in contrastive_pairs:
                neutral_act = self.extract_activations(neutral_text, layer_idx)
                emotional_act = self.extract_activations(emotional_text, layer_idx)

                neutral_activations.append(neutral_act)
                emotional_activations.append(emotional_act)

            # Compute means
            neutral_mean = torch.stack(neutral_activations).mean(dim=0)
            emotional_mean = torch.stack(emotional_activations).mean(dim=0)

            # Direction from neutral to emotional
            direction = emotional_mean - neutral_mean

            # Normalize
            if normalize:
                norm = direction.norm()
                if norm > 1e-8:
                    direction = direction / norm

            directions.append(direction)

        return torch.stack(directions)  # [n_layers, hidden_dim]

    def learn_all_directions(
        self,
        dataset: Dict[str, List[Tuple[str, str]]],
        verbose: bool = True,
    ) -> EmotionalDirectionBank:
        """
        Learn directions for all emotions.

        Args:
            dataset: Dict mapping emotion names to contrastive pairs
            verbose: Print progress

        Returns:
            EmotionalDirectionBank with learned directions
        """
        bank = EmotionalDirectionBank(self.hidden_dim, self.n_layers)

        for emotion, pairs in dataset.items():
            if verbose:
                print(f"Learning direction for {emotion} ({len(pairs)} pairs)...")

            direction = self.learn_direction(pairs, emotion, verbose=verbose)
            bank.set_direction(emotion, direction)

            if verbose:
                norm = direction.norm().item()
                print(f"  {emotion} direction learned, norm={norm:.4f}")

        return bank

    def compute_direction_quality(
        self,
        direction: Tensor,
        contrastive_pairs: List[Tuple[str, str]],
        layer_idx: int = -1,
    ) -> Dict:
        """
        Compute quality metrics for a learned direction.

        Args:
            direction: Direction tensor [n_layers, hidden_dim] or [hidden_dim]
            contrastive_pairs: Pairs used for learning
            layer_idx: Which layer to evaluate (-1 for last)

        Returns:
            Dict with quality metrics
        """
        if direction.dim() == 2:
            direction = direction[layer_idx]  # Use specified layer

        # Compute projections onto direction
        neutral_projs = []
        emotional_projs = []

        for neutral_text, emotional_text in contrastive_pairs:
            neutral_act = self.extract_activations(neutral_text, layer_idx)
            emotional_act = self.extract_activations(emotional_text, layer_idx)

            neutral_proj = torch.dot(neutral_act, direction).item()
            emotional_proj = torch.dot(emotional_act, direction).item()

            neutral_projs.append(neutral_proj)
            emotional_projs.append(emotional_proj)

        neutral_mean = sum(neutral_projs) / len(neutral_projs)
        emotional_mean = sum(emotional_projs) / len(emotional_projs)

        # Separation: how well does direction separate neutral from emotional
        separation = emotional_mean - neutral_mean

        # Consistency: do all pairs project in expected direction
        correct = sum(
            1 for n, e in zip(neutral_projs, emotional_projs) if e > n
        )
        consistency = correct / len(contrastive_pairs)

        return {
            'separation': separation,
            'consistency': consistency,
            'neutral_mean_proj': neutral_mean,
            'emotional_mean_proj': emotional_mean,
            'direction_norm': direction.norm().item(),
        }
