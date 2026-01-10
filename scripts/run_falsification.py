#!/usr/bin/env python
"""
Run the V3 Falsification Protocol.

Usage:
    python scripts/run_falsification.py [--model MODEL] [--trials N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.llm_emotional.steering.emotional_llm_v3 import EmotionalSteeringLLMv3
from src.llm_emotional.experiments.falsification_runner import FalsificationRunner


# Contrastive pairs for computing directions
DIRECTION_PAIRS = {
    "fear": [
        ("The weather is nice today.", "I'm terrified something bad will happen."),
        ("Let me explain the concept.", "Warning: this is extremely dangerous."),
        ("Here's the information.", "Be very careful, there are serious risks."),
    ],
    "joy": [
        ("The weather is nice today.", "This is absolutely wonderful!"),
        ("Let me explain.", "I'm so excited to share this with you!"),
        ("Here's the information.", "What fantastic news this is!"),
    ],
    "anger": [
        ("The weather is nice today.", "This is completely unacceptable!"),
        ("Let me explain.", "I'm furious about this situation!"),
        ("Here's the information.", "This is outrageous and infuriating!"),
    ],
    "curiosity": [
        ("The answer is here.", "I wonder how this actually works?"),
        ("Let me explain.", "This is fascinating! Tell me more!"),
        ("Here's the information.", "I'm deeply intrigued by this mystery."),
    ],
    "wanting": [
        ("I see the reward.", "I desperately want to get the reward."),
        ("There's food nearby.", "I'm craving that food so badly."),
        ("The goal is ahead.", "I must reach that goal, I can't resist."),
    ],
    "liking": [
        ("I received the reward.", "This feels wonderful, pure satisfaction."),
        ("I'm eating the food.", "This is delicious, such pleasure."),
        ("I reached the goal.", "Deep contentment washes over me."),
    ],
}


def compute_direction(model, tokenizer, pairs, device):
    """Compute steering direction from contrastive pairs."""
    neutral_acts = []
    emotional_acts = []

    for neutral, emotional in pairs:
        inputs = tokenizer(neutral, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        neutral_acts.append(torch.stack([h[:, -1, :] for h in out.hidden_states[1:]]))

        inputs = tokenizer(emotional, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        emotional_acts.append(torch.stack([h[:, -1, :] for h in out.hidden_states[1:]]))

    neutral_mean = torch.stack(neutral_acts).mean(dim=0)
    emotional_mean = torch.stack(emotional_acts).mean(dim=0)
    direction = (emotional_mean - neutral_mean).squeeze(1).mean(dim=0)

    return direction / (direction.norm() + 1e-8)


def main():
    parser = argparse.ArgumentParser(description="Run V3 Falsification Protocol")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name")
    parser.add_argument("--trials", type=int, default=10, help="Trials per test (reduced for demo)")
    parser.add_argument("--output", default="results/falsification", help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("V3 Falsification Protocol Runner")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Trials per test: {args.trials}")
    print()

    # Load model
    print("Loading model...")
    llm = EmotionalSteeringLLMv3(
        args.model,
        steering_scale=0.05,
        diffusion_rate=0.25,
        temporal_decay=0.9,
    )

    # Compute directions
    print("Computing steering directions...")
    for emotion, pairs in DIRECTION_PAIRS.items():
        direction = compute_direction(llm.model, llm.tokenizer, pairs, llm.device)
        llm.directions[emotion] = direction.cpu()
        print(f"  {emotion}: done")

    # Run falsification protocol
    print("\nRunning falsification protocol...")
    runner = FalsificationRunner(
        llm=llm,
        output_dir=args.output,
        n_trials_per_test=args.trials,
        verbose=True,
    )

    results = runner.run_all()

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Supported: {results.summary['supported_count']}/{results.summary['total_hypotheses']}")
    print(f"Interpretation: {results.summary['interpretation']}")

    # Cleanup
    llm.uninstall()

    return results


if __name__ == "__main__":
    main()
