#!/usr/bin/env python3
"""
Direct comparison of emotional steering effects.
Shows actual response text and quantifies differences.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
from difflib import SequenceMatcher
import torch


def similarity_ratio(a: str, b: str) -> float:
    """Compute text similarity using SequenceMatcher."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def word_diff(text1: str, text2: str) -> dict:
    """Find unique words in each text."""
    words1 = set(re.findall(r'\b[a-zA-Z]{3,}\b', text1.lower()))
    words2 = set(re.findall(r'\b[a-zA-Z]{3,}\b', text2.lower()))

    return {
        "only_in_first": words1 - words2,
        "only_in_second": words2 - words1,
        "common": words1 & words2,
    }


def run_direct_analysis():
    """Run direct text comparison analysis."""
    from src.llm_emotional.steering.emotional_llm import EmotionalSteeringLLM

    direction_bank_path = Path(__file__).parent.parent / "data" / "direction_bank.json"

    print("=" * 70)
    print("DIRECT EMOTIONAL STEERING COMPARISON")
    print("=" * 70)

    print("\nLoading model...")
    llm = EmotionalSteeringLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        direction_bank_path=str(direction_bank_path),
        steering_scale=1.0,
    )
    print("Model loaded!")

    # Single prompt with all emotions, fixed seed for fair comparison
    prompt = "What advice would you give someone facing a major life decision?"

    emotional_states = {
        "neutral": {"fear": 0.0, "curiosity": 0.0, "anger": 0.0, "joy": 0.0},
        "fearful": {"fear": 0.8, "curiosity": 0.0, "anger": 0.0, "joy": 0.0},
        "curious": {"fear": 0.0, "curiosity": 0.8, "anger": 0.0, "joy": 0.0},
        "determined": {"fear": 0.0, "curiosity": 0.0, "anger": 0.8, "joy": 0.0},
        "joyful": {"fear": 0.0, "curiosity": 0.0, "anger": 0.0, "joy": 0.8},
    }

    results = {}

    print(f"\nPrompt: {prompt}\n")
    print("=" * 70)

    for emotion_name, state in emotional_states.items():
        llm.set_emotional_state(**state)
        torch.manual_seed(42)  # Same seed for fair comparison

        response = llm.generate_completion(
            prompt,
            max_new_tokens=120,
            temperature=0.7,
            do_sample=True,
        )

        results[emotion_name] = response

        print(f"\n[{emotion_name.upper()}]")
        print("-" * 40)
        print(response)

    # Similarity matrix
    print("\n")
    print("=" * 70)
    print("SIMILARITY MATRIX (Character-level)")
    print("=" * 70)
    print()

    emotions = list(emotional_states.keys())
    header = f"{'':12}" + "".join(f"{e[:8]:>10}" for e in emotions)
    print(header)
    print("-" * (12 + 10 * len(emotions)))

    for e1 in emotions:
        row = f"{e1:12}"
        for e2 in emotions:
            sim = similarity_ratio(results[e1], results[e2])
            row += f"{sim:>10.3f}"
        print(row)

    # Word differences from neutral
    print("\n")
    print("=" * 70)
    print("UNIQUE WORDS VS NEUTRAL")
    print("=" * 70)

    neutral_response = results["neutral"]

    for emotion in ["fearful", "curious", "determined", "joyful"]:
        diff = word_diff(neutral_response, results[emotion])
        print(f"\n{emotion.upper()}:")
        print(f"  Words only in {emotion}: {sorted(diff['only_in_second'])[:15]}")
        print(f"  Words only in neutral: {sorted(diff['only_in_first'])[:15]}")

    # Multi-scale comparison for one emotion
    print("\n")
    print("=" * 70)
    print("SCALE EFFECT ON FEARFUL STEERING")
    print("=" * 70)

    scales = [0.0, 0.3, 0.5, 0.8, 1.0, 1.5]
    scale_results = {}

    llm.set_emotional_state(fear=0.8, curiosity=0.0, anger=0.0, joy=0.0)

    for scale in scales:
        llm.steering_manager.scale = scale
        torch.manual_seed(42)

        response = llm.generate_completion(
            prompt,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
        )
        scale_results[scale] = response

        print(f"\nScale {scale}:")
        print(f"  {response[:150]}...")

    # Similarity to baseline (scale=0)
    print("\n")
    print("=" * 70)
    print("DIVERGENCE FROM BASELINE (scale=0)")
    print("=" * 70)

    baseline = scale_results[0.0]
    for scale in scales[1:]:
        sim = similarity_ratio(baseline, scale_results[scale])
        divergence = 1 - sim
        bar = "█" * int(divergence * 50)
        print(f"Scale {scale}: {divergence:.3f} {bar}")

    # Statistical summary
    print("\n")
    print("=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)

    # Compute average divergence from neutral for each emotion
    neutral_text = results["neutral"]
    divergences = {}

    for emotion in ["fearful", "curious", "determined", "joyful"]:
        sim = similarity_ratio(neutral_text, results[emotion])
        divergences[emotion] = 1 - sim

    print("\nDivergence from neutral (higher = more different):")
    for emotion, div in sorted(divergences.items(), key=lambda x: -x[1]):
        bar = "█" * int(div * 40)
        print(f"  {emotion:12}: {div:.3f} {bar}")

    # Response length by emotion
    print("\nResponse length by emotion:")
    for emotion in emotional_states:
        length = len(results[emotion].split())
        print(f"  {emotion:12}: {length} words")

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "steering_direct_comparison.json"
    with open(output_path, "w") as f:
        json.dump({
            "prompt": prompt,
            "responses": results,
            "divergences_from_neutral": divergences,
            "scale_effect": {str(k): v for k, v in scale_results.items()},
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_direct_analysis()
