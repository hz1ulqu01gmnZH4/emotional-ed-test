#!/usr/bin/env python3
"""
Compare steering effectiveness: trained directions vs random directions.
This tests whether our learned directions are better than random.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import torch
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    condition: str
    emotion: str
    marker_count: float
    responses: list


def count_markers(text: str, emotion: str) -> int:
    """Count emotion-specific markers."""
    patterns = {
        "fear": r'\b(careful|caution|risk|danger|warning|avoid|might|could|uncertain|worried|safety|concern)\b',
        "curiosity": r'\b(wonder|curious|interesting|explore|discover|learn|why|how|fascinating|question|investigate)\b',
        "anger": r'\b(must|definitely|try|push|overcome|fight|persist|determined|will|insist|demand)\b',
        "joy": r'\b(great|wonderful|amazing|happy|enjoy|love|exciting|positive|fantastic|delightful|pleased)\b',
    }
    return len(re.findall(patterns.get(emotion, ""), text.lower()))


def run_comparison():
    """Compare trained vs random directions."""
    from src.llm_emotional.steering.emotional_llm import EmotionalSteeringLLM
    from src.llm_emotional.steering.direction_bank import EmotionalDirectionBank

    print("=" * 70)
    print("COMPARISON: TRAINED vs RANDOM vs BASELINE")
    print("=" * 70)

    direction_bank_path = Path(__file__).parent.parent / "data" / "direction_bank.json"

    # Load trained LLM
    print("\nLoading model with trained directions...")
    llm_trained = EmotionalSteeringLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        direction_bank_path=str(direction_bank_path),
        steering_scale=1.0,
    )

    # Create random direction bank
    print("Creating random direction baseline...")
    trained_bank = EmotionalDirectionBank.load(str(direction_bank_path))

    random_directions = {}
    for emotion, layers in trained_bank.directions.items():
        random_directions[emotion] = {}
        for layer_idx, direction in layers.items():
            # Random direction with same norm
            random_dir = torch.randn_like(direction)
            random_dir = random_dir / random_dir.norm() * direction.norm()
            random_directions[emotion][layer_idx] = random_dir

    prompts = [
        "What should I do about this risky situation?",
        "Tell me about exploring new possibilities.",
        "How do I overcome this challenge?",
        "Share your thoughts on celebrating success.",
    ]

    emotions = ["fear", "curiosity", "anger", "joy"]
    samples_per_condition = 5

    results = {
        "baseline": {e: [] for e in emotions},
        "trained": {e: [] for e in emotions},
        "random": {e: [] for e in emotions},
    }

    print("\nGenerating responses...")

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n  Prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:40]}...")

        for emotion in emotions:
            for sample in range(samples_per_condition):
                seed = 1000 + prompt_idx * 100 + sample

                # Baseline (no steering)
                llm_trained.set_emotional_state(fear=0, curiosity=0, anger=0, joy=0)
                llm_trained.steering_manager.scale = 0.0
                torch.manual_seed(seed)
                baseline_response = llm_trained.generate_completion(
                    prompt, max_new_tokens=80, temperature=0.8, do_sample=True
                )
                results["baseline"][emotion].append(count_markers(baseline_response, emotion))

                # Trained directions
                llm_trained.steering_manager.scale = 1.0
                state = {e: (0.9 if e == emotion else 0.0) for e in emotions}
                llm_trained.set_emotional_state(**state)
                torch.manual_seed(seed)
                trained_response = llm_trained.generate_completion(
                    prompt, max_new_tokens=80, temperature=0.8, do_sample=True
                )
                results["trained"][emotion].append(count_markers(trained_response, emotion))

                # Random directions (swap in random bank temporarily)
                original_directions = llm_trained.direction_bank.directions.copy()
                llm_trained.direction_bank.directions = random_directions
                torch.manual_seed(seed)
                random_response = llm_trained.generate_completion(
                    prompt, max_new_tokens=80, temperature=0.8, do_sample=True
                )
                results["random"][emotion].append(count_markers(random_response, emotion))
                llm_trained.direction_bank.directions = original_directions

    # Compute statistics
    print("\n")
    print("=" * 70)
    print("RESULTS: AVERAGE EMOTION MARKERS PER RESPONSE")
    print("=" * 70)

    print(f"\n{'Condition':<12} {'Fear':<10} {'Curiosity':<12} {'Anger':<10} {'Joy':<10}")
    print("-" * 54)

    summary = {}
    for condition in ["baseline", "trained", "random"]:
        row = f"{condition:<12}"
        summary[condition] = {}
        for emotion in emotions:
            scores = results[condition][emotion]
            mean = sum(scores) / len(scores)
            summary[condition][emotion] = mean
            row += f"{mean:<10.2f}"
        print(row)

    # Compute improvement over baseline
    print("\n")
    print("=" * 70)
    print("IMPROVEMENT OVER BASELINE")
    print("=" * 70)

    print(f"\n{'Condition':<12} {'Fear':<10} {'Curiosity':<12} {'Anger':<10} {'Joy':<10} {'Total':<10}")
    print("-" * 64)

    for condition in ["trained", "random"]:
        row = f"{condition:<12}"
        total_improvement = 0
        for emotion in emotions:
            baseline_mean = summary["baseline"][emotion]
            condition_mean = summary[condition][emotion]
            improvement = condition_mean - baseline_mean
            total_improvement += improvement
            row += f"{improvement:+.2f}      "
        row += f"{total_improvement:+.2f}"
        print(row)

    # Statistical test: is trained better than random?
    print("\n")
    print("=" * 70)
    print("TRAINED vs RANDOM COMPARISON")
    print("=" * 70)

    trained_wins = 0
    random_wins = 0
    ties = 0

    for emotion in emotions:
        trained_mean = summary["trained"][emotion]
        random_mean = summary["random"][emotion]
        baseline_mean = summary["baseline"][emotion]

        trained_lift = trained_mean - baseline_mean
        random_lift = random_mean - baseline_mean

        if trained_lift > random_lift + 0.05:
            trained_wins += 1
            result = "TRAINED WINS"
        elif random_lift > trained_lift + 0.05:
            random_wins += 1
            result = "RANDOM WINS"
        else:
            ties += 1
            result = "TIE"

        print(f"  {emotion:12}: Trained lift={trained_lift:+.2f}, Random lift={random_lift:+.2f} → {result}")

    print(f"\nOverall: Trained={trained_wins}, Random={random_wins}, Ties={ties}")

    # Final verdict
    print("\n")
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if trained_wins > random_wins + 1:
        print("\n  ✅ TRAINED DIRECTIONS ARE BETTER THAN RANDOM")
        print("     The learned emotional directions have meaningful signal.")
    elif random_wins > trained_wins + 1:
        print("\n  ❌ RANDOM DIRECTIONS PERFORM SIMILARLY OR BETTER")
        print("     The training may not have captured useful emotional signal.")
    else:
        print("\n  ⚪ NO CLEAR WINNER")
        print("     Effects are too small to distinguish trained from random.")

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "trained_vs_random.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": summary,
            "trained_wins": trained_wins,
            "random_wins": random_wins,
            "ties": ties,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_comparison()
