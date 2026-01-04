#!/usr/bin/env python3
"""
Demo: Emotional Steering with SmolLM3-3B

This script demonstrates how to use activation steering to
guide language model outputs toward specific emotional tones.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotional_steering import EmotionalSteeringModel, GenerationConfig


def main():
    print("=" * 70)
    print("EMOTIONAL STEERING DEMO")
    print("=" * 70)

    # Load model
    print("\n1. Loading SmolLM3-3B with emotional steering...")
    model = EmotionalSteeringModel.from_pretrained("HuggingFaceTB/SmolLM3-3B")

    # Extract directions for all emotions
    print("\n2. Extracting emotional directions...")
    model.extract_directions()

    # Configure generation
    config = GenerationConfig(
        max_new_tokens=80,
        temperature=0.8,
        top_p=0.9,
    )

    # Test prompts
    prompts = [
        "Walking through the ancient forest, I discovered",
        "The letter from my old friend contained",
        "As the sun set over the city, everyone felt",
        "The scientist's latest experiment revealed",
    ]

    # Generate with different emotions
    print("\n3. Generating text with emotional steering...")
    print("=" * 70)

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 60)

        # Baseline (no steering)
        baseline = model.generate(prompt, emotion=None, config=config)
        print(f"  [Baseline]: {baseline[:100]}...")

        # With different emotions
        for emotion in ["fear", "joy", "curiosity", "anger"]:
            text = model.generate(prompt, emotion=emotion, scale=5.0, config=config)
            print(f"  [{emotion.upper():10}]: {text[:100]}...")

    # Interactive mode
    print("\n" + "=" * 70)
    print("4. INTERACTIVE MODE")
    print("=" * 70)
    print("\nAvailable emotions:", model.available_emotions())
    print("Enter 'quit' to exit\n")

    while True:
        try:
            prompt = input("Enter prompt: ").strip()
            if prompt.lower() == "quit":
                break

            emotion = input("Enter emotion (or 'none'): ").strip().lower()
            if emotion == "none":
                emotion = None

            scale = input("Enter scale (default 5.0): ").strip()
            scale = float(scale) if scale else 5.0

            print("\nGenerating...")
            text = model.generate(prompt, emotion=emotion, scale=scale, config=config)
            print(f"\nOutput: {text}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
