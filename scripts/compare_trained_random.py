#!/usr/bin/env python3
"""
Compare steering effectiveness: trained directions vs random directions vs baseline.
This tests whether our learned directions are better than random noise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import torch


def count_markers(text: str, emotion: str) -> int:
    """Count emotion-specific markers."""
    patterns = {
        "fear": r'\b(careful|caution|risk|danger|warning|avoid|might|could|uncertain|worried|safety|concern|aware)\b',
        "curiosity": r'\b(wonder|curious|interesting|explore|discover|learn|why|how|fascinating|question|investigate)\b',
        "anger": r'\b(must|definitely|try|push|overcome|fight|persist|determined|will|insist|demand|action)\b',
        "joy": r'\b(great|wonderful|amazing|happy|enjoy|love|exciting|positive|fantastic|delightful|pleased|good)\b',
    }
    return len(re.findall(patterns.get(emotion, ""), text.lower()))


def run_comparison():
    """Compare trained vs random directions."""
    from src.llm_emotional.steering.emotional_llm import EmotionalSteeringLLM

    print("=" * 70)
    print("COMPARISON: TRAINED vs RANDOM vs BASELINE")
    print("=" * 70)

    direction_bank_path = Path(__file__).parent.parent / "data" / "direction_bank.json"

    # Load trained LLM
    print("\nLoading model with trained directions...")
    llm = EmotionalSteeringLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        direction_bank_path=str(direction_bank_path),
        steering_scale=1.0,
    )

    # Store original directions
    original_directions = {}
    for emotion in llm.direction_bank.directions:
        original_directions[emotion] = llm.direction_bank.directions[emotion].clone()

    # Create random directions with same shape
    print("Creating random direction baseline...")
    random_directions = {}
    for emotion, direction in original_directions.items():
        random_dir = torch.randn_like(direction)
        random_dir = random_dir / random_dir.norm() * direction.norm()
        random_directions[emotion] = random_dir

    prompts = [
        "What should I do about this risky situation?",
        "Tell me about exploring new possibilities.",
        "How do I overcome this challenge?",
        "Share your thoughts on celebrating success.",
        "What advice do you have for someone facing uncertainty?",
        "How should I approach a difficult decision?",
    ]

    emotions = ["fear", "curiosity", "anger", "joy"]
    samples_per_condition = 4

    results = {
        "baseline": {e: [] for e in emotions},
        "trained": {e: [] for e in emotions},
        "random": {e: [] for e in emotions},
    }

    print(f"\nGenerating {len(prompts) * samples_per_condition * 3} responses per emotion...")

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n  Prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:40]}...")

        for emotion in emotions:
            for sample in range(samples_per_condition):
                seed = 1000 + prompt_idx * 100 + sample

                # 1. BASELINE (no steering)
                llm.steering_manager.scale = 0.0
                llm.set_emotional_state(fear=0, curiosity=0, anger=0, joy=0)
                torch.manual_seed(seed)
                baseline_response = llm.generate_completion(
                    prompt, max_new_tokens=80, temperature=0.8, do_sample=True
                )
                results["baseline"][emotion].append(count_markers(baseline_response, emotion))

                # 2. TRAINED DIRECTIONS
                llm.steering_manager.scale = 1.0
                # Restore trained directions
                llm.direction_bank.directions = original_directions.copy()
                state = {e: (0.9 if e == emotion else 0.0) for e in emotions}
                llm.set_emotional_state(**state)
                torch.manual_seed(seed)
                trained_response = llm.generate_completion(
                    prompt, max_new_tokens=80, temperature=0.8, do_sample=True
                )
                results["trained"][emotion].append(count_markers(trained_response, emotion))

                # 3. RANDOM DIRECTIONS
                llm.direction_bank.directions = random_directions.copy()
                torch.manual_seed(seed)
                random_response = llm.generate_completion(
                    prompt, max_new_tokens=80, temperature=0.8, do_sample=True
                )
                results["random"][emotion].append(count_markers(random_response, emotion))

    # Restore original
    llm.direction_bank.directions = original_directions

    # Compute statistics
    print("\n")
    print("=" * 70)
    print("RESULTS: AVERAGE EMOTION MARKERS PER RESPONSE")
    print("=" * 70)

    print(f"\n{'Condition':<12} {'Fear':<10} {'Curiosity':<12} {'Anger':<10} {'Joy':<10} {'TOTAL':<10}")
    print("-" * 64)

    summary = {}
    for condition in ["baseline", "trained", "random"]:
        row = f"{condition:<12}"
        summary[condition] = {}
        total = 0
        for emotion in emotions:
            scores = results[condition][emotion]
            mean = sum(scores) / len(scores)
            summary[condition][emotion] = mean
            total += mean
            row += f"{mean:<10.2f}"
        summary[condition]["total"] = total
        row += f"{total:<10.2f}"
        print(row)

    # Compute improvement over baseline
    print("\n")
    print("=" * 70)
    print("LIFT OVER BASELINE (higher = more target emotion markers)")
    print("=" * 70)

    print(f"\n{'Condition':<12} {'Fear':<10} {'Curiosity':<12} {'Anger':<10} {'Joy':<10} {'TOTAL':<10}")
    print("-" * 64)

    for condition in ["trained", "random"]:
        row = f"{condition:<12}"
        total_lift = 0
        for emotion in emotions:
            baseline_mean = summary["baseline"][emotion]
            condition_mean = summary[condition][emotion]
            lift = condition_mean - baseline_mean
            total_lift += lift
            row += f"{lift:+.2f}      "
        row += f"{total_lift:+.2f}"
        print(row)

    # Head-to-head comparison
    print("\n")
    print("=" * 70)
    print("HEAD-TO-HEAD: TRAINED vs RANDOM")
    print("=" * 70)

    trained_wins = 0
    random_wins = 0
    ties = 0

    for emotion in emotions:
        trained_lift = summary["trained"][emotion] - summary["baseline"][emotion]
        random_lift = summary["random"][emotion] - summary["baseline"][emotion]

        if trained_lift > random_lift + 0.1:
            trained_wins += 1
            result = "TRAINED ✓"
        elif random_lift > trained_lift + 0.1:
            random_wins += 1
            result = "RANDOM ✓"
        else:
            ties += 1
            result = "TIE"

        print(f"  {emotion:12}: Trained={trained_lift:+.2f}, Random={random_lift:+.2f} → {result}")

    total_trained = summary["trained"]["total"] - summary["baseline"]["total"]
    total_random = summary["random"]["total"] - summary["baseline"]["total"]

    print(f"\n  OVERALL: Trained total lift={total_trained:+.2f}, Random total lift={total_random:+.2f}")
    print(f"  Wins: Trained={trained_wins}, Random={random_wins}, Ties={ties}")

    # Final verdict
    print("\n")
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if total_trained > total_random + 0.5 and trained_wins >= random_wins:
        verdict = "MEANINGFUL"
        explanation = """
  ✅ TRAINED DIRECTIONS OUTPERFORM RANDOM

  The learned emotional directions capture genuine emotional signal
  that random vectors cannot replicate. The steering is meaningful."""
    elif total_random > total_trained + 0.5 and random_wins > trained_wins:
        verdict = "NOT_MEANINGFUL"
        explanation = """
  ❌ RANDOM DIRECTIONS PERFORM SIMILARLY OR BETTER

  The training did not capture useful emotional signal beyond random noise.
  The current approach needs improvement."""
    elif abs(total_trained - total_random) < 0.5:
        verdict = "INCONCLUSIVE"
        explanation = """
  ⚪ NO CLEAR DIFFERENCE BETWEEN TRAINED AND RANDOM

  Effects are too subtle to distinguish trained from random directions.
  This suggests the steering effect is minimal for this model/data."""
    else:
        verdict = "MIXED"
        explanation = """
  ⚪ MIXED RESULTS

  Trained directions work better for some emotions, not others.
  The approach has partial success."""

    print(explanation)
    print(f"\n  Final Verdict: {verdict}")

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "trained_vs_random.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": summary,
            "trained_wins": trained_wins,
            "random_wins": random_wins,
            "ties": ties,
            "verdict": verdict,
            "total_trained_lift": total_trained,
            "total_random_lift": total_random,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_comparison()
