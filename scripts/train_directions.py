#!/usr/bin/env python3
"""
Train emotional direction vectors from contrastive pairs.

Usage:
    uv run scripts/train_directions.py

This script:
1. Loads the emotional contrastive dataset
2. Loads a frozen LLM (Qwen2.5-1.5B by default)
3. Learns direction vectors using Difference-in-Means
4. Saves the direction bank for use with EmotionalSteeringLLM
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


class TrainingError(Exception):
    """Raised when training fails. NO FALLBACK."""
    pass


def load_dataset(path: Path) -> dict:
    """Load and validate dataset. FAIL if missing or invalid."""
    if not path.exists():
        raise TrainingError(
            f"Dataset not found at {path}. "
            "Generate it first using the dataset generation script."
        )

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    required_emotions = {'fear', 'curiosity', 'anger', 'joy'}
    missing = required_emotions - set(data.keys())
    if missing:
        raise TrainingError(f"Dataset missing emotions: {missing}")

    for emotion, pairs in data.items():
        if len(pairs) < 10:
            raise TrainingError(
                f"Insufficient pairs for {emotion}: {len(pairs)} < 10"
            )

    return data


def load_model(model_name: str, device: str):
    """Load model and tokenizer. FAIL if loading fails."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise TrainingError(
            "transformers not installed. Run: uv pip install transformers"
        )

    print(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        raise TrainingError(f"Failed to load tokenizer: {e}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {model_name}...")
    try:
        # Determine dtype
        if device == "cuda" and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        else:
            dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
        )

        if device == "cpu":
            model = model.to("cpu")

    except Exception as e:
        raise TrainingError(f"Failed to load model: {e}")

    model.eval()
    print(f"Model loaded: {model.config.hidden_size}d, {model.config.num_hidden_layers} layers")

    return model, tokenizer


def train_directions(
    model,
    tokenizer,
    dataset: dict,
    output_path: Path,
    verbose: bool = True,
):
    """Train direction vectors and save to disk."""
    from src.llm_emotional.steering.direction_learner import EmotionalDirectionLearner
    from src.llm_emotional.steering.direction_bank import EmotionalDirectionBank

    print("\nInitializing direction learner...")
    learner = EmotionalDirectionLearner(model, tokenizer)

    print(f"Model: {learner.hidden_dim}d hidden, {learner.n_layers} layers\n")

    # Convert dataset to contrastive pairs format
    all_pairs = {}
    for emotion, pairs in dataset.items():
        all_pairs[emotion] = [
            (pair['neutral'], pair['emotional'])
            for pair in pairs
        ]

    print("Learning directions...")
    print("=" * 60)

    bank = learner.learn_all_directions(all_pairs, verbose=verbose)

    print("=" * 60)
    print("\nDirection quality analysis:")

    for emotion in bank.EMOTIONS:
        if bank.learned[emotion]:
            pairs = all_pairs.get(emotion, [])
            if pairs:
                quality = learner.compute_direction_quality(
                    bank.directions[emotion],
                    pairs,
                    layer_idx=-1  # Last layer
                )
                print(f"  {emotion}:")
                print(f"    separation: {quality['separation']:.4f}")
                print(f"    consistency: {quality['consistency']:.1%}")
                print(f"    direction norm: {quality['direction_norm']:.4f}")

    # Save
    print(f"\nSaving direction bank to {output_path}...")
    bank.save(output_path)

    # Verify save
    loaded = EmotionalDirectionBank.load(output_path)
    print(f"Verified: loaded {loaded}")

    return bank


def main():
    parser = argparse.ArgumentParser(description="Train emotional directions")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--dataset",
        default="data/emotional_pairs.json",
        help="Path to contrastive pairs dataset"
    )
    parser.add_argument(
        "--output",
        default="data/direction_bank.json",
        help="Path to save direction bank"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output"
    )

    args = parser.parse_args()

    # Resolve paths
    base_dir = Path(__file__).parent.parent
    dataset_path = base_dir / args.dataset
    output_path = base_dir / args.output

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("EMOTIONAL DIRECTION TRAINING")
    print("=" * 60)
    print(f"Model:   {args.model}")
    print(f"Dataset: {dataset_path}")
    print(f"Output:  {output_path}")
    print(f"Device:  {device}")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(dataset_path)
    total_pairs = sum(len(pairs) for pairs in dataset.values())
    print(f"Loaded {total_pairs} pairs across {len(dataset)} emotions")

    # Load model
    model, tokenizer = load_model(args.model, device)

    # Train
    bank = train_directions(
        model,
        tokenizer,
        dataset,
        output_path,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Direction bank saved to: {output_path}")
    print(f"Use with: EmotionalSteeringLLM(direction_bank_path='{output_path}')")


if __name__ == "__main__":
    main()
