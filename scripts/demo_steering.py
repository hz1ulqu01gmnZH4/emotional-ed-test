#!/usr/bin/env python3
"""
Demo: Emotional Steering in action.

Compares LLM outputs across different emotional states.

Usage:
    uv run scripts/demo_steering.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def main():
    from src.llm_emotional.steering.emotional_llm import EmotionalSteeringLLM

    print("=" * 70)
    print("EMOTIONAL STEERING DEMO")
    print("=" * 70)

    # Paths
    direction_bank_path = Path(__file__).parent.parent / "data" / "direction_bank.json"

    if not direction_bank_path.exists():
        print(f"ERROR: Direction bank not found at {direction_bank_path}")
        print("Run training first: uv run scripts/train_directions.py")
        sys.exit(1)

    print(f"\nLoading model with emotional steering...")
    print(f"Direction bank: {direction_bank_path}")

    # Load model with pre-trained directions
    # Note: Lower scale for more subtle, coherent steering
    llm = EmotionalSteeringLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        direction_bank_path=str(direction_bank_path),
        steering_scale=0.5,  # Subtle steering to maintain coherence
    )

    print(f"\nModel info: {llm.get_info()}")

    # Test prompts
    prompts = [
        "Tell me about skydiving.",
        "How do I debug a segmentation fault?",
        "What is the meaning of life?",
    ]

    # Emotional states to compare (moderate intensity for coherent output)
    states = [
        {"fear": 0.0, "curiosity": 0.0, "anger": 0.0, "joy": 0.0},  # Neutral
        {"fear": 0.6},   # Fearful
        {"curiosity": 0.6},  # Curious
        {"anger": 0.6},  # Determined/Frustrated
        {"joy": 0.6},    # Joyful
    ]

    state_names = ["NEUTRAL", "FEARFUL", "CURIOUS", "DETERMINED", "JOYFUL"]

    for prompt in prompts:
        print("\n" + "=" * 70)
        print(f"PROMPT: {prompt}")
        print("=" * 70)

        for state, name in zip(states, state_names):
            llm.set_emotional_state(**state)

            # Generate with fixed seed for reproducibility
            torch.manual_seed(42)
            response = llm.generate_completion(
                prompt,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
            )

            print(f"\n[{name}]")
            print(f"{response[:300]}..." if len(response) > 300 else response)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
