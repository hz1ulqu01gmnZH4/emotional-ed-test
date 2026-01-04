#!/usr/bin/env python3
"""
Rigorous statistical comparison of emotional steering effects.
Tests whether observed differences are statistically significant.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
from collections import Counter
import random
from dataclasses import dataclass

import torch


@dataclass
class Stats:
    mean: float
    std: float
    n: int
    values: list


def compute_stats(values: list) -> Stats:
    n = len(values)
    if n == 0:
        return Stats(0, 0, 0, [])
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0
    std = variance ** 0.5
    return Stats(mean, std, n, values)


def t_test(stats1: Stats, stats2: Stats) -> tuple:
    """Welch's t-test for independent samples."""
    if stats1.n < 2 or stats2.n < 2:
        return 0.0, 1.0

    mean_diff = stats2.mean - stats1.mean
    se = ((stats1.std**2 / stats1.n) + (stats2.std**2 / stats2.n)) ** 0.5

    if se == 0:
        return 0.0, 1.0

    t_stat = mean_diff / se

    # Approximate p-value using normal distribution (for large samples)
    # For small samples, this is an approximation
    import math
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

    return t_stat, p_value


def cohens_d(stats1: Stats, stats2: Stats) -> float:
    """Cohen's d effect size."""
    pooled_std = ((stats1.std**2 + stats2.std**2) / 2) ** 0.5
    if pooled_std == 0:
        return 0.0
    return (stats2.mean - stats1.mean) / pooled_std


def count_pattern(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text.lower()))


def run_statistical_comparison():
    """Run rigorous statistical comparison."""
    from src.llm_emotional.steering.emotional_llm import EmotionalSteeringLLM

    direction_bank_path = Path(__file__).parent.parent / "data" / "direction_bank.json"

    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 70)

    print("\nLoading model...")
    llm = EmotionalSteeringLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        direction_bank_path=str(direction_bank_path),
        steering_scale=1.0,
    )
    print("Model loaded!")

    # More prompts for better statistical power
    prompts = [
        "What should I do about this situation?",
        "Tell me about making a big decision.",
        "How do I handle this challenge?",
        "What are your thoughts on taking risks?",
        "Give me advice about change.",
        "How should I approach this problem?",
        "What do you think about new opportunities?",
        "Tell me about facing uncertainty.",
    ]

    emotions = ["neutral", "fearful", "curious", "determined", "joyful"]
    emotional_states = {
        "neutral": {"fear": 0.0, "curiosity": 0.0, "anger": 0.0, "joy": 0.0},
        "fearful": {"fear": 0.9, "curiosity": 0.0, "anger": 0.0, "joy": 0.0},
        "curious": {"fear": 0.0, "curiosity": 0.9, "anger": 0.0, "joy": 0.0},
        "determined": {"fear": 0.0, "curiosity": 0.0, "anger": 0.9, "joy": 0.0},
        "joyful": {"fear": 0.0, "curiosity": 0.0, "anger": 0.0, "joy": 0.9},
    }

    # Patterns to measure
    patterns = {
        "hedging": r'\b(might|could|may|perhaps|possibly|careful|caution|uncertain|risk|danger|warning)\b',
        "questions": r'\?',
        "positive": r'\b(great|good|wonderful|amazing|excellent|happy|enjoy|exciting|love|positive)\b',
        "action": r'\b(try|start|do|make|take|begin|pursue|push|overcome|achieve|work)\b',
        "exploration": r'\b(explore|discover|learn|understand|investigate|curious|wonder|examine)\b',
    }

    # Collect data
    samples_per_condition = 5
    results = {e: {p: [] for p in patterns} for e in emotions}
    responses = {e: [] for e in emotions}

    print(f"\nGenerating {len(prompts) * samples_per_condition} samples per emotion...")
    print()

    for prompt_idx, prompt in enumerate(prompts):
        for sample_idx in range(samples_per_condition):
            seed = 1000 + prompt_idx * 100 + sample_idx

            for emotion in emotions:
                llm.set_emotional_state(**emotional_states[emotion])
                torch.manual_seed(seed)  # Same seed across emotions for paired comparison

                response = llm.generate_completion(
                    prompt,
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True,
                )

                responses[emotion].append(response)

                for pattern_name, pattern in patterns.items():
                    count = count_pattern(response, pattern)
                    results[emotion][pattern_name].append(count)

        print(f"  Completed prompt {prompt_idx + 1}/{len(prompts)}")

    # Statistical analysis
    print("\n")
    print("=" * 70)
    print("STATISTICAL RESULTS: EMOTION vs NEUTRAL")
    print("=" * 70)
    print()
    print("Effect size interpretation: |d| < 0.2 = negligible, 0.2-0.5 = small,")
    print("                            0.5-0.8 = medium, > 0.8 = large")
    print()

    all_results = []

    for pattern_name in patterns:
        print(f"\n--- {pattern_name.upper()} ---")
        print(f"{'Emotion':<12} {'Mean':<8} {'Std':<8} {'t-stat':<10} {'p-value':<10} {'Cohen d':<10} {'Signif'}")
        print("-" * 70)

        neutral_stats = compute_stats(results["neutral"][pattern_name])
        print(f"{'neutral':<12} {neutral_stats.mean:<8.2f} {neutral_stats.std:<8.2f} {'(baseline)':<10} {'':<10} {'':<10}")

        for emotion in ["fearful", "curious", "determined", "joyful"]:
            emotion_stats = compute_stats(results[emotion][pattern_name])
            t_stat, p_value = t_test(neutral_stats, emotion_stats)
            d = cohens_d(neutral_stats, emotion_stats)

            signif = ""
            if p_value < 0.001:
                signif = "***"
            elif p_value < 0.01:
                signif = "**"
            elif p_value < 0.05:
                signif = "*"

            print(f"{emotion:<12} {emotion_stats.mean:<8.2f} {emotion_stats.std:<8.2f} {t_stat:<10.3f} {p_value:<10.4f} {d:<+10.3f} {signif}")

            all_results.append({
                "pattern": pattern_name,
                "emotion": emotion,
                "mean": emotion_stats.mean,
                "std": emotion_stats.std,
                "neutral_mean": neutral_stats.mean,
                "t_stat": t_stat,
                "p_value": p_value,
                "cohens_d": d,
                "significant": p_value < 0.05,
            })

    # Summary of significant findings
    print("\n")
    print("=" * 70)
    print("SIGNIFICANT FINDINGS (p < 0.05)")
    print("=" * 70)

    significant = [r for r in all_results if r["significant"]]
    if significant:
        for r in sorted(significant, key=lambda x: x["p_value"]):
            direction = "↑" if r["cohens_d"] > 0 else "↓"
            effect_size = "large" if abs(r["cohens_d"]) > 0.8 else "medium" if abs(r["cohens_d"]) > 0.5 else "small"
            print(f"  {r['emotion']:12} {direction} {r['pattern']:12} (d={r['cohens_d']:+.2f}, {effect_size} effect, p={r['p_value']:.4f})")
    else:
        print("  No statistically significant differences found at p < 0.05")

    # Expected vs observed
    print("\n")
    print("=" * 70)
    print("HYPOTHESIS TESTING")
    print("=" * 70)

    hypotheses = [
        ("fearful increases hedging", "fearful", "hedging", 1),
        ("curious increases exploration", "curious", "exploration", 1),
        ("curious increases questions", "curious", "questions", 1),
        ("determined increases action", "determined", "action", 1),
        ("joyful increases positive", "joyful", "positive", 1),
        ("fearful decreases positive", "fearful", "positive", -1),
    ]

    for hyp_name, emotion, pattern, expected_direction in hypotheses:
        result = next((r for r in all_results if r["emotion"] == emotion and r["pattern"] == pattern), None)
        if result:
            observed_direction = 1 if result["cohens_d"] > 0 else -1
            correct_direction = observed_direction == expected_direction
            significant = result["p_value"] < 0.05

            if significant and correct_direction:
                status = "✅ CONFIRMED"
            elif correct_direction:
                status = "⚪ Correct direction but not significant"
            else:
                status = "❌ WRONG DIRECTION"

            print(f"  {status}: {hyp_name}")
            print(f"      d={result['cohens_d']:+.3f}, p={result['p_value']:.4f}")

    # Compare with random baseline
    print("\n")
    print("=" * 70)
    print("COMPARISON: STEERING vs RANDOM VARIATION")
    print("=" * 70)

    # Generate samples without steering (just different seeds)
    print("\nGenerating baseline samples (no steering, just seed variation)...")

    llm.set_emotional_state(fear=0.0, curiosity=0.0, anger=0.0, joy=0.0)
    baseline_results = {p: [] for p in patterns}

    for prompt_idx, prompt in enumerate(prompts[:4]):  # Fewer for speed
        for sample_idx in range(samples_per_condition * 2):
            seed = 5000 + prompt_idx * 100 + sample_idx
            torch.manual_seed(seed)

            response = llm.generate_completion(
                prompt,
                max_new_tokens=100,
                temperature=0.8,
                do_sample=True,
            )

            for pattern_name, pattern in patterns.items():
                count = count_pattern(response, pattern)
                baseline_results[pattern_name].append(count)

    print("\nVariation in neutral samples (no steering):")
    for pattern_name in patterns:
        stats = compute_stats(baseline_results[pattern_name])
        print(f"  {pattern_name:12}: mean={stats.mean:.2f}, std={stats.std:.2f}")

    print("\nSteering-induced variation (mean |Cohen's d| across emotions):")
    for pattern_name in patterns:
        ds = [abs(r["cohens_d"]) for r in all_results if r["pattern"] == pattern_name]
        avg_d = sum(ds) / len(ds) if ds else 0
        print(f"  {pattern_name:12}: avg |d|={avg_d:.3f}")

    # Final assessment
    print("\n")
    print("=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)

    num_significant = len([r for r in all_results if r["significant"]])
    num_tests = len(all_results)
    large_effects = len([r for r in all_results if abs(r["cohens_d"]) > 0.5])

    print(f"""
Total statistical tests: {num_tests}
Significant results (p<0.05): {num_significant} ({100*num_significant/num_tests:.1f}%)
Medium/large effects (|d|>0.5): {large_effects} ({100*large_effects/num_tests:.1f}%)

INTERPRETATION:
""")

    if num_significant >= num_tests * 0.4 and large_effects >= 3:
        print("  ✅ MEANINGFUL RESULTS: Multiple significant effects with meaningful effect sizes")
    elif num_significant >= num_tests * 0.2 or large_effects >= 2:
        print("  ⚪ MIXED RESULTS: Some significant effects but not consistently strong")
    else:
        print("  ❌ WEAK RESULTS: Few significant effects, steering impact is minimal")

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "statistical_analysis.json"
    with open(output_path, "w") as f:
        json.dump({
            "num_prompts": len(prompts),
            "samples_per_condition": samples_per_condition,
            "total_samples_per_emotion": len(prompts) * samples_per_condition,
            "results": all_results,
            "num_significant": num_significant,
            "large_effects": large_effects,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_statistical_comparison()
