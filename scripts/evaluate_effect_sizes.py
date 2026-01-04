#!/usr/bin/env python3
"""
Comprehensive evaluation of emotional steering effect sizes.

Measures whether steering produces statistically significant
and practically meaningful changes in generated text.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import re
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List
import torch

from emotional_steering import EmotionalSteeringModel, GenerationConfig
from emotional_steering.emotions import EMOTION_MARKERS


@dataclass
class EffectResult:
    """Result of effect size analysis."""
    emotion: str
    baseline_mean: float
    steered_mean: float
    cohens_d: float
    p_value: float
    significant: bool  # p < 0.05
    interpretation: str


def count_emotion_markers(text: str, emotion: str) -> int:
    """Count emotion-related words in text."""
    markers = EMOTION_MARKERS.get(emotion, [])
    if not markers:
        return 0
    pattern = r'\b(' + '|'.join(re.escape(m) for m in markers) + r')\b'
    return len(re.findall(pattern, text.lower()))


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean2 - mean1) / pooled_std


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d >= 0.8:
        return "LARGE"
    elif d >= 0.5:
        return "MEDIUM"
    elif d >= 0.2:
        return "SMALL"
    else:
        return "NEGLIGIBLE"


def run_evaluation():
    print("=" * 70)
    print("COMPREHENSIVE EFFECT SIZE EVALUATION")
    print("=" * 70)

    # Configuration
    n_prompts = 10
    n_samples_per_prompt = 5
    emotions_to_test = ["fear", "joy", "curiosity", "anger"]
    scale = 5.0

    # Diverse prompts
    prompts = [
        "Walking through the abandoned building, I suddenly noticed",
        "The letter from the mysterious stranger contained",
        "As the storm approached, everyone in the village felt",
        "The scientist's groundbreaking discovery revealed that",
        "In the depths of the ancient forest, the explorer found",
        "The news report announced something that made people",
        "Behind the locked door, there was a secret that",
        "The old photograph showed something that made her",
        "As midnight approached, the atmosphere became",
        "The unexpected visitor brought news that left everyone",
    ]

    # Load model
    print("\n[1] Loading model and extracting directions...")
    model = EmotionalSteeringModel.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    model.extract_directions(emotions=emotions_to_test)

    config = GenerationConfig(
        max_new_tokens=60,
        temperature=0.8,
        top_p=0.9,
    )

    # Collect data
    print(f"\n[2] Generating {n_prompts} × {n_samples_per_prompt} = {n_prompts * n_samples_per_prompt} samples per condition...")

    results: Dict[str, Dict[str, List[int]]] = {
        "baseline": {e: [] for e in emotions_to_test}
    }
    for emotion in emotions_to_test:
        results[emotion] = {e: [] for e in emotions_to_test}

    all_outputs: Dict[str, List[str]] = {"baseline": []}
    for emotion in emotions_to_test:
        all_outputs[emotion] = []

    total = len(prompts) * n_samples_per_prompt * (len(emotions_to_test) + 1)
    current = 0

    # Generate baseline
    print("\n    Generating baseline...")
    for prompt in prompts:
        for sample_idx in range(n_samples_per_prompt):
            torch.manual_seed(1000 + hash(prompt) % 10000 + sample_idx)
            text = model.generate(prompt, emotion=None, config=config)
            all_outputs["baseline"].append(text)

            for emotion in emotions_to_test:
                count = count_emotion_markers(text, emotion)
                results["baseline"][emotion].append(count)

            current += 1

    # Generate with each emotion
    for steer_emotion in emotions_to_test:
        print(f"    Generating with {steer_emotion} steering...")
        for prompt in prompts:
            for sample_idx in range(n_samples_per_prompt):
                torch.manual_seed(1000 + hash(prompt) % 10000 + sample_idx)
                text = model.generate(prompt, emotion=steer_emotion, scale=scale, config=config)
                all_outputs[steer_emotion].append(text)

                for emotion in emotions_to_test:
                    count = count_emotion_markers(text, emotion)
                    results[steer_emotion][emotion].append(count)

                current += 1

    # Analyze results
    print("\n[3] Analyzing effect sizes...")

    effect_results: List[EffectResult] = []

    for steer_emotion in emotions_to_test:
        baseline_counts = results["baseline"][steer_emotion]
        steered_counts = results[steer_emotion][steer_emotion]

        # Cohen's d
        d = cohens_d(baseline_counts, steered_counts)

        # t-test
        if len(set(baseline_counts)) > 1 or len(set(steered_counts)) > 1:
            t_stat, p_value = stats.ttest_ind(baseline_counts, steered_counts)
        else:
            p_value = 1.0

        effect_results.append(EffectResult(
            emotion=steer_emotion,
            baseline_mean=np.mean(baseline_counts),
            steered_mean=np.mean(steered_counts),
            cohens_d=d,
            p_value=p_value,
            significant=p_value < 0.05,
            interpretation=interpret_effect_size(d),
        ))

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS: EFFECT SIZES (Same-Emotion Markers)")
    print("=" * 70)

    print(f"\n{'Emotion':<12} {'Baseline':<10} {'Steered':<10} {'Cohens d':<12} {'p-value':<10} {'Significant':<12} {'Effect':<12}")
    print("-" * 78)

    for r in effect_results:
        sig_str = "YES *" if r.significant else "no"
        print(f"{r.emotion:<12} {r.baseline_mean:<10.2f} {r.steered_mean:<10.2f} {r.cohens_d:+<12.3f} {r.p_value:<10.4f} {sig_str:<12} {r.interpretation:<12}")

    # Cross-emotion analysis
    print("\n" + "=" * 70)
    print("CROSS-EMOTION ANALYSIS")
    print("=" * 70)
    print("\nDoes steering for emotion X affect markers of emotion Y?")

    print(f"\n{'Steer →':<12}", end="")
    for emotion in emotions_to_test:
        print(f"{emotion:<12}", end="")
    print()
    print("-" * (12 + 12 * len(emotions_to_test)))

    for steer_emotion in emotions_to_test:
        print(f"{steer_emotion:<12}", end="")
        for measure_emotion in emotions_to_test:
            baseline_counts = results["baseline"][measure_emotion]
            steered_counts = results[steer_emotion][measure_emotion]
            d = cohens_d(baseline_counts, steered_counts)

            # Color-code based on effect
            if steer_emotion == measure_emotion:
                marker = f"{d:+.2f} **"  # Same emotion
            else:
                marker = f"{d:+.2f}"
            print(f"{marker:<12}", end="")
        print()

    # Sample outputs
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUTS")
    print("=" * 70)

    for emotion in emotions_to_test:
        print(f"\n--- {emotion.upper()} STEERING ---")

        # Find output with most markers
        best_idx = 0
        best_count = 0
        for i, text in enumerate(all_outputs[emotion]):
            count = count_emotion_markers(text, emotion)
            if count > best_count:
                best_count = count
                best_idx = i

        print(f"  Best example ({best_count} markers):")
        print(f"    '{all_outputs[emotion][best_idx][:120]}...'")

        # Show corresponding baseline
        print(f"  Baseline:")
        print(f"    '{all_outputs['baseline'][best_idx][:120]}...'")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    significant_count = sum(1 for r in effect_results if r.significant)
    avg_d = np.mean([abs(r.cohens_d) for r in effect_results])

    print(f"\n  Significant effects: {significant_count}/{len(effect_results)}")
    print(f"  Average |Cohen's d|: {avg_d:.3f}")

    if avg_d >= 0.5 and significant_count >= len(effect_results) // 2:
        verdict = "MEANINGFUL - Steering produces reliable, detectable effects"
        symbol = "✅"
    elif avg_d >= 0.2 or significant_count > 0:
        verdict = "PARTIAL - Some effects detected but not consistent"
        symbol = "⚪"
    else:
        verdict = "NOT MEANINGFUL - Effects too small to be practical"
        symbol = "❌"

    print(f"\n  {symbol} VERDICT: {verdict}")

    # Per-emotion breakdown
    print("\n  Per-emotion summary:")
    for r in effect_results:
        symbol = "✅" if r.significant and abs(r.cohens_d) >= 0.3 else "⚪" if r.significant else "❌"
        print(f"    {symbol} {r.emotion}: d={r.cohens_d:+.2f} ({r.interpretation}), p={r.p_value:.4f}")

    # Save results
    save_data = {
        "config": {
            "n_prompts": n_prompts,
            "n_samples": n_samples_per_prompt,
            "scale": scale,
            "model": "HuggingFaceTB/SmolLM3-3B",
            "layer": 9,
        },
        "effects": [
            {
                "emotion": r.emotion,
                "baseline_mean": r.baseline_mean,
                "steered_mean": r.steered_mean,
                "cohens_d": r.cohens_d,
                "p_value": r.p_value,
                "significant": bool(r.significant),
                "interpretation": r.interpretation,
            }
            for r in effect_results
        ],
        "summary": {
            "significant_count": significant_count,
            "total_emotions": len(effect_results),
            "avg_cohens_d": avg_d,
            "verdict": verdict,
        }
    }

    with open("data/effect_size_evaluation.json", "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Results saved to: data/effect_size_evaluation.json")

    return avg_d >= 0.2 or significant_count > 0


if __name__ == "__main__":
    success = run_evaluation()
    sys.exit(0 if success else 1)
