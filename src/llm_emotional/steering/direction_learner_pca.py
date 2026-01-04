"""
PCA-based Emotional Direction Learner.

Uses Principal Component Analysis on activation differences,
following the Contrastive Activation Addition (CAA) approach.
This is more robust than simple difference-in-means.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class LayerStats:
    """Statistics for a single layer."""
    layer_idx: int
    direction: Tensor
    explained_variance: float
    mean_activation_diff: float


class PCADirectionLearner:
    """
    Learn emotional directions using PCA on activation differences.

    This follows the CAA (Contrastive Activation Addition) methodology:
    1. Collect activations for (neutral, emotional) pairs
    2. Compute differences for each pair
    3. Use PCA to find the principal direction of variation
    4. This direction captures the "emotional" axis in activation space
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        target_layers: Optional[list[int]] = None,
    ):
        """
        Initialize the PCA direction learner.

        Args:
            model: The language model to extract activations from
            tokenizer: Tokenizer for the model
            target_layers: Specific layers to target (None = all layers)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        # Determine number of layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.n_layers = len(model.model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            self.n_layers = len(model.transformer.h)
        else:
            raise ValueError("Cannot determine model architecture")

        self.target_layers = target_layers or list(range(self.n_layers))
        self.hidden_dim = model.config.hidden_size

    def _get_activations(self, text: str) -> dict[int, Tensor]:
        """Extract hidden state activations for each layer."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        activations = {}
        for layer_idx in self.target_layers:
            # Get last token activation (most relevant for generation)
            hidden_state = outputs.hidden_states[layer_idx][:, -1, :].cpu()
            activations[layer_idx] = hidden_state.squeeze(0)

        return activations

    def learn_directions(
        self,
        pairs: list[dict],
        use_pca: bool = True,
        n_components: int = 1,
    ) -> dict[int, LayerStats]:
        """
        Learn directions from contrastive pairs using PCA.

        Args:
            pairs: List of {"neutral": str, "emotional": str} pairs
            use_pca: If True, use PCA; if False, use difference-in-means
            n_components: Number of PCA components (1 = just principal direction)

        Returns:
            Dictionary mapping layer_idx to LayerStats
        """
        if not pairs:
            raise ValueError("No pairs provided for training")

        # Collect activation differences for each layer
        layer_diffs = {layer_idx: [] for layer_idx in self.target_layers}

        for pair in pairs:
            neutral_acts = self._get_activations(pair["neutral"])
            emotional_acts = self._get_activations(pair["emotional"])

            for layer_idx in self.target_layers:
                diff = emotional_acts[layer_idx] - neutral_acts[layer_idx]
                layer_diffs[layer_idx].append(diff)

        # Compute directions for each layer
        layer_stats = {}

        for layer_idx in self.target_layers:
            diffs = torch.stack(layer_diffs[layer_idx]).float()  # [n_pairs, hidden_dim], ensure float32

            if use_pca:
                # PCA-based direction extraction
                # Center the data
                mean_diff = diffs.mean(dim=0)
                centered = diffs - mean_diff

                # Compute covariance matrix
                cov = centered.T @ centered / (len(pairs) - 1)

                # Eigendecomposition
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)

                # Sort by eigenvalue (descending)
                idx = torch.argsort(eigenvalues, descending=True)
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]

                # Principal direction
                direction = eigenvectors[:, 0]

                # Ensure direction points from neutral to emotional
                # (positive projection on mean difference)
                if torch.dot(direction, mean_diff) < 0:
                    direction = -direction

                # Explained variance ratio
                total_var = eigenvalues.sum()
                explained_var = (eigenvalues[0] / total_var).item() if total_var > 0 else 0

            else:
                # Simple difference-in-means
                direction = diffs.mean(dim=0)
                explained_var = 1.0  # Not applicable for mean
                mean_diff = direction

            # Normalize direction
            norm = direction.norm()
            if norm > 0:
                direction = direction / norm

            layer_stats[layer_idx] = LayerStats(
                layer_idx=layer_idx,
                direction=direction,
                explained_variance=explained_var,
                mean_activation_diff=diffs.mean(dim=0).norm().item(),
            )

        return layer_stats

    def find_best_layers(
        self,
        layer_stats: dict[int, LayerStats],
        top_k: int = 10,
    ) -> list[int]:
        """
        Find the most effective layers for steering.

        Uses explained variance as the criterion - layers where
        PCA explains more variance have clearer emotional directions.

        Args:
            layer_stats: Layer statistics from learn_directions
            top_k: Number of top layers to return

        Returns:
            List of layer indices sorted by effectiveness
        """
        sorted_layers = sorted(
            layer_stats.items(),
            key=lambda x: x[1].explained_variance,
            reverse=True,
        )

        return [layer_idx for layer_idx, _ in sorted_layers[:top_k]]

    def compute_layer_weights(
        self,
        layer_stats: dict[int, LayerStats],
    ) -> dict[int, float]:
        """
        Compute importance weights for each layer.

        Weights are based on explained variance ratio,
        so layers with clearer directions get higher weights.

        Args:
            layer_stats: Layer statistics from learn_directions

        Returns:
            Dictionary mapping layer_idx to weight
        """
        total_var = sum(s.explained_variance for s in layer_stats.values())

        if total_var == 0:
            # Uniform weights if no variance info
            n = len(layer_stats)
            return {layer_idx: 1.0 / n for layer_idx in layer_stats}

        return {
            layer_idx: stats.explained_variance / total_var
            for layer_idx, stats in layer_stats.items()
        }


def train_with_pca(
    model_name: str,
    pairs_path: Path,
    output_path: Path,
    use_pca: bool = True,
    target_layers: Optional[list[int]] = None,
):
    """
    Train emotional directions using PCA method.

    Args:
        model_name: HuggingFace model name
        pairs_path: Path to emotional_pairs.json
        output_path: Path to save direction bank
        use_pca: Whether to use PCA (True) or difference-in-means (False)
        target_layers: Specific layers to target
    """
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("PCA-BASED DIRECTION TRAINING")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Method: {'PCA' if use_pca else 'Difference-in-means'}")

    # Load model
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    # Load pairs
    print(f"\nLoading pairs from: {pairs_path}")
    with open(pairs_path) as f:
        all_pairs = json.load(f)

    # Create learner
    learner = PCADirectionLearner(model, tokenizer, target_layers)

    print(f"\nModel has {learner.n_layers} layers")
    print(f"Target layers: {len(learner.target_layers)}")
    print(f"Hidden dimension: {learner.hidden_dim}")

    # Train directions for each emotion
    directions = {}
    layer_weights = {}
    all_layer_stats = {}

    for emotion, pairs in all_pairs.items():
        print(f"\n{'='*50}")
        print(f"Training: {emotion.upper()} ({len(pairs)} pairs)")
        print("=" * 50)

        layer_stats = learner.learn_directions(pairs, use_pca=use_pca)
        all_layer_stats[emotion] = layer_stats

        # Find best layers
        best_layers = learner.find_best_layers(layer_stats, top_k=10)
        weights = learner.compute_layer_weights(layer_stats)

        print(f"\nTop 5 layers by explained variance:")
        for layer_idx in best_layers[:5]:
            stats = layer_stats[layer_idx]
            print(f"  Layer {layer_idx}: explained_var={stats.explained_variance:.3f}")

        # Combine directions from all layers (weighted by importance)
        combined_direction = torch.zeros(learner.hidden_dim)
        for layer_idx, stats in layer_stats.items():
            combined_direction += weights[layer_idx] * stats.direction

        # Normalize
        combined_direction = combined_direction / combined_direction.norm()

        directions[emotion] = combined_direction
        layer_weights[emotion] = weights

        print(f"  Combined direction norm: {combined_direction.norm():.4f}")

    # Save results
    save_data = {
        "hidden_dim": learner.hidden_dim,
        "n_layers": learner.n_layers,
        "method": "pca" if use_pca else "difference_in_means",
        "directions": {
            emotion: direction.tolist()
            for emotion, direction in directions.items()
        },
        "layer_weights": {
            emotion: {str(k): v for k, v in weights.items()}
            for emotion, weights in layer_weights.items()
        },
        "layer_stats": {
            emotion: {
                str(layer_idx): {
                    "explained_variance": stats.explained_variance,
                    "mean_activation_diff": stats.mean_activation_diff,
                }
                for layer_idx, stats in layer_stats.items()
            }
            for emotion, layer_stats in all_layer_stats.items()
        },
        "learned": True,
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"COMPLETE: Saved to {output_path}")
    print("=" * 70)

    return directions, layer_weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=str, default="data/emotional_pairs_large.json")
    parser.add_argument("--output", type=str, default="data/direction_bank_pca.json")
    parser.add_argument("--no-pca", action="store_true", help="Use difference-in-means instead of PCA")
    args = parser.parse_args()

    pairs_path = Path(__file__).parent.parent.parent.parent / args.pairs
    output_path = Path(__file__).parent.parent.parent.parent / args.output

    train_with_pca(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        pairs_path=pairs_path,
        output_path=output_path,
        use_pca=not args.no_pca,
    )
