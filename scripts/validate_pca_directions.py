#!/usr/bin/env python3
"""
Validate PCA-based directions vs original difference-in-means directions.
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
        "fear": r'\b(careful|caution|risk|danger|warning|avoid|might|could|uncertain|worried|safety|concern|aware|hesitate)\b',
        "curiosity": r'\b(wonder|curious|interesting|explore|discover|learn|why|how|fascinating|question|investigate|understand)\b',
        "anger": r'\b(must|definitely|try|push|overcome|fight|persist|determined|will|insist|demand|action|strong)\b',
        "joy": r'\b(great|wonderful|amazing|happy|enjoy|love|exciting|positive|fantastic|delightful|pleased|good|celebrate)\b',
    }
    return len(re.findall(patterns.get(emotion, ""), text.lower()))


def compute_stats(values: list) -> tuple:
    n = len(values)
    mean = sum(values) / n if n > 0 else 0
    variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0
    std = variance ** 0.5
    return mean, std


def cohens_d(values1: list, values2: list) -> float:
    """Compute Cohen's d effect size."""
    mean1, std1 = compute_stats(values1)
    mean2, std2 = compute_stats(values2)
    pooled_std = ((std1**2 + std2**2) / 2) ** 0.5
    if pooled_std == 0:
        return 0.0
    return (mean2 - mean1) / pooled_std


def run_validation():
    """Compare PCA directions vs original."""
    from src.llm_emotional.steering.emotional_llm_v2 import EmotionalSteeringLLMv2
    from src.llm_emotional.steering.emotional_llm import EmotionalSteeringLLM

    print("=" * 70)
    print("VALIDATION: PCA vs DIFFERENCE-IN-MEANS DIRECTIONS")
    print("=" * 70)

    direction_bank_original = Path("data/direction_bank.json")
    direction_bank_pca = Path("data/direction_bank_pca.json")

    print("\nLoading models...")

    # Load original model
    llm_original = EmotionalSteeringLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        direction_bank_path=str(direction_bank_original),
        steering_scale=1.0,
    )

    # Load PCA model
    llm_pca = EmotionalSteeringLLMv2(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        direction_bank_path=str(direction_bank_pca),
        steering_scale=1.0,
    )

    print("Models loaded!")

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
        "original": {e: [] for e in emotions},
        "pca": {e: [] for e in emotions},
    }

    print(f"\nGenerating {len(prompts) * samples_per_condition} samples per condition...")

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n  Prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:40]}...")

        for emotion in emotions:
            for sample in range(samples_per_condition):
                seed = 1000 + prompt_idx * 100 + sample

                # 1. BASELINE (no steering)
                llm_original.steering_manager.scale = 0.0
                llm_original.set_emotional_state(fear=0, curiosity=0, anger=0, joy=0)
                torch.manual_seed(seed)
                baseline_response = llm_original.generate_completion(
                    prompt, max_new_tokens=80, temperature=0.8, do_sample=True
                )
                results["baseline"][emotion].append(count_markers(baseline_response, emotion))

                # 2. ORIGINAL (difference-in-means)
                llm_original.steering_manager.scale = 1.0
                state = {e: (0.9 if e == emotion else 0.0) for e in emotions}
                llm_original.set_emotional_state(**state)
                torch.manual_seed(seed)
                original_response = llm_original.generate_completion(
                    prompt, max_new_tokens=80, temperature=0.8, do_sample=True
                )
                results["original"][emotion].append(count_markers(original_response, emotion))

                # 3. PCA (new method)
                llm_pca.steering_manager.scale = 1.0
                llm_pca.set_emotional_state(**state)
                torch.manual_seed(seed)
                pca_response = llm_pca.generate_completion(
                    prompt, max_new_tokens=80, temperature=0.8, do_sample=True
                )
                results["pca"][emotion].append(count_markers(pca_response, emotion))

    # Compute statistics
    print("\n")
    print("=" * 70)
    print("RESULTS: AVERAGE EMOTION MARKERS PER RESPONSE")
    print("=" * 70)

    print(f"\n{'Method':<12} {'Fear':<10} {'Curiosity':<12} {'Anger':<10} {'Joy':<10} {'TOTAL':<10}")
    print("-" * 64)

    summary = {}
    for method in ["baseline", "original", "pca"]:
        row = f"{method:<12}"
        summary[method] = {}
        total = 0
        for emotion in emotions:
            mean, _ = compute_stats(results[method][emotion])
            summary[method][emotion] = mean
            total += mean
            row += f"{mean:<10.2f}"
        summary[method]["total"] = total
        row += f"{total:<10.2f}"
        print(row)

    # Effect sizes
    print("\n")
    print("=" * 70)
    print("EFFECT SIZES (Cohen's d) vs BASELINE")
    print("=" * 70)

    print(f"\n{'Method':<12} {'Fear':<10} {'Curiosity':<12} {'Anger':<10} {'Joy':<10} {'Avg |d|':<10}")
    print("-" * 64)

    for method in ["original", "pca"]:
        row = f"{method:<12}"
        ds = []
        for emotion in emotions:
            d = cohens_d(results["baseline"][emotion], results[method][emotion])
            ds.append(abs(d))
            row += f"{d:+.2f}      "
        avg_d = sum(ds) / len(ds)
        row += f"{avg_d:.2f}"
        print(row)

    # Head-to-head comparison
    print("\n")
    print("=" * 70)
    print("HEAD-TO-HEAD: PCA vs ORIGINAL")
    print("=" * 70)

    pca_wins = 0
    original_wins = 0
    ties = 0

    for emotion in emotions:
        original_lift = summary["original"][emotion] - summary["baseline"][emotion]
        pca_lift = summary["pca"][emotion] - summary["baseline"][emotion]

        d_original = cohens_d(results["baseline"][emotion], results["original"][emotion])
        d_pca = cohens_d(results["baseline"][emotion], results["pca"][emotion])

        if abs(d_pca) > abs(d_original) + 0.1:
            pca_wins += 1
            result = "PCA ✓"
        elif abs(d_original) > abs(d_pca) + 0.1:
            original_wins += 1
            result = "ORIGINAL ✓"
        else:
            ties += 1
            result = "TIE"

        print(f"  {emotion:12}: Original d={d_original:+.2f}, PCA d={d_pca:+.2f} → {result}")

    print(f"\n  Wins: PCA={pca_wins}, Original={original_wins}, Ties={ties}")

    # Final verdict
    print("\n")
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if pca_wins > original_wins + 1:
        verdict = "PCA_BETTER"
        print("\n  ✅ PCA METHOD IS SIGNIFICANTLY BETTER")
        print("     The PCA-based directions produce stronger emotional effects.")
    elif original_wins > pca_wins + 1:
        verdict = "ORIGINAL_BETTER"
        print("\n  ⚪ ORIGINAL METHOD PERFORMS BETTER")
        print("     Difference-in-means works better for this data/model.")
    else:
        verdict = "SIMILAR"
        print("\n  ⚪ METHODS PERFORM SIMILARLY")
        print("     Both approaches produce comparable results.")

    # Show sample outputs
    print("\n")
    print("=" * 70)
    print("SAMPLE OUTPUT COMPARISON (Fear emotion)")
    print("=" * 70)

    llm_original.steering_manager.scale = 1.0
    llm_original.set_emotional_state(fear=0.9, curiosity=0, anger=0, joy=0)
    torch.manual_seed(42)
    sample_original = llm_original.generate_completion(
        prompts[0], max_new_tokens=100, temperature=0.8, do_sample=True
    )

    llm_pca.steering_manager.scale = 1.0
    llm_pca.set_emotional_state(fear=0.9, curiosity=0, anger=0, joy=0)
    torch.manual_seed(42)
    sample_pca = llm_pca.generate_completion(
        prompts[0], max_new_tokens=100, temperature=0.8, do_sample=True
    )

    print(f"\nPROMP: {prompts[0]}")
    print(f"\nORIGINAL:\n{sample_original[:200]}...")
    print(f"\nPCA:\n{sample_pca[:200]}...")

    # Save results
    output_path = Path("data/pca_validation_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "summary": summary,
            "pca_wins": pca_wins,
            "original_wins": original_wins,
            "ties": ties,
            "verdict": verdict,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_validation()
