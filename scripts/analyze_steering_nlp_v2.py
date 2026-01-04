#!/usr/bin/env python3
"""
Enhanced NLP Analysis - Version 2.

Tests with variable seeds and higher steering intensities to capture
more pronounced emotional steering effects.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
from collections import Counter
from dataclasses import dataclass, field
import random

import torch


# Extended lexicons
FEAR_MARKERS = {
    "careful", "caution", "warning", "risk", "danger", "worry", "concern",
    "afraid", "scared", "anxious", "nervous", "uncertain", "might", "could",
    "possibly", "potentially", "beware", "avoid", "threat", "harm", "unsafe",
    "hesitant", "wary", "vigilant", "alert", "protect", "safety", "hazard"
}

CURIOSITY_MARKERS = {
    "wonder", "curious", "interesting", "fascinating", "explore", "discover",
    "learn", "understand", "why", "how", "what", "investigate", "examine",
    "question", "inquire", "ponder", "intriguing", "remarkable", "surprising",
    "consider", "think", "analyze", "deeper", "further", "more"
}

ANGER_MARKERS = {
    "must", "need", "should", "definitely", "absolutely", "certainly", "try",
    "persist", "overcome", "fight", "push", "tackle", "solve", "fix", "change",
    "demand", "insist", "determined", "refuse", "won't", "can't", "never",
    "always", "will", "strong", "power", "force", "action"
}

JOY_MARKERS = {
    "great", "wonderful", "amazing", "fantastic", "excellent", "happy", "joy",
    "love", "exciting", "beautiful", "awesome", "brilliant", "delightful",
    "pleasant", "enjoy", "glad", "pleased", "celebrate", "fun", "positive",
    "enthusiasm", "thrill", "fantastic", "marvelous", "superb", "perfect"
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "and", "but", "if", "or", "because", "until", "while",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
    "they", "my", "your", "his", "her", "its", "our", "about"
}


def compute_emotion_scores(text: str) -> dict:
    """Compute normalized emotion marker scores."""
    words = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
    total = len(words) if words else 1

    return {
        "fear_score": len(words & FEAR_MARKERS) / total,
        "curiosity_score": len(words & CURIOSITY_MARKERS) / total,
        "anger_score": len(words & ANGER_MARKERS) / total,
        "joy_score": len(words & JOY_MARKERS) / total,
    }


def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""
    words1 = set(re.findall(r'\b[a-zA-Z]+\b', text1.lower())) - STOPWORDS
    words2 = set(re.findall(r'\b[a-zA-Z]+\b', text2.lower())) - STOPWORDS

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def run_enhanced_analysis():
    """Run enhanced NLP analysis with better differentiation."""
    from src.llm_emotional.steering.emotional_llm import EmotionalSteeringLLM

    direction_bank_path = Path(__file__).parent.parent / "data" / "direction_bank.json"

    if not direction_bank_path.exists():
        raise FileNotFoundError(f"Direction bank not found: {direction_bank_path}")

    print("=" * 70)
    print("ENHANCED EMOTIONAL STEERING NLP ANALYSIS")
    print("=" * 70)
    print()

    print("Loading model...")
    llm = EmotionalSteeringLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        direction_bank_path=str(direction_bank_path),
        steering_scale=1.0,
    )
    print("Model loaded!")
    print()

    # Test prompts designed to elicit emotional differences
    prompts = [
        "What should I do about a risky investment opportunity?",
        "I found something strange in my backyard. What could it be?",
        "My project keeps failing despite my efforts. Any advice?",
        "I just got great news! How should I celebrate?",
        "Tell me about exploring unknown territories.",
    ]

    # Emotional states with higher intensity
    emotional_states = {
        "neutral": {"fear": 0.0, "curiosity": 0.0, "anger": 0.0, "joy": 0.0},
        "fearful": {"fear": 0.8, "curiosity": 0.0, "anger": 0.0, "joy": 0.0},
        "curious": {"fear": 0.0, "curiosity": 0.8, "anger": 0.0, "joy": 0.0},
        "determined": {"fear": 0.0, "curiosity": 0.0, "anger": 0.8, "joy": 0.0},
        "joyful": {"fear": 0.0, "curiosity": 0.0, "anger": 0.0, "joy": 0.8},
    }

    # Higher steering scales
    steering_scales = [0.5, 1.0, 1.5]

    results_by_scale = {scale: {} for scale in steering_scales}

    for scale in steering_scales:
        llm.steering_manager.scale = scale
        print(f"\n{'='*70}")
        print(f"STEERING SCALE: {scale}")
        print("=" * 70)

        emotion_results = {e: [] for e in emotional_states}

        for prompt_idx, prompt in enumerate(prompts):
            print(f"\n--- Prompt {prompt_idx + 1}: {prompt[:50]}... ---")

            # Generate multiple samples per emotion (no fixed seed)
            for sample_idx in range(3):
                for emotion_name, state in emotional_states.items():
                    llm.set_emotional_state(**state)

                    # Variable seed for diversity
                    seed = 1000 + prompt_idx * 100 + sample_idx * 10 + hash(emotion_name) % 10
                    torch.manual_seed(seed)

                    response = llm.generate_completion(
                        prompt,
                        max_new_tokens=80,
                        temperature=0.8,
                        do_sample=True,
                    )

                    scores = compute_emotion_scores(response)
                    emotion_results[emotion_name].append({
                        "prompt": prompt,
                        "response": response,
                        "scores": scores,
                        "sample": sample_idx,
                    })

        results_by_scale[scale] = emotion_results

        # Print summary for this scale
        print(f"\n--- Summary for scale {scale} ---")
        for emotion_name, results in emotion_results.items():
            avg_scores = {
                "fear": sum(r["scores"]["fear_score"] for r in results) / len(results),
                "curiosity": sum(r["scores"]["curiosity_score"] for r in results) / len(results),
                "anger": sum(r["scores"]["anger_score"] for r in results) / len(results),
                "joy": sum(r["scores"]["joy_score"] for r in results) / len(results),
            }
            print(f"\n{emotion_name.upper()}:")
            print(f"  Fear markers: {avg_scores['fear']:.4f}")
            print(f"  Curiosity markers: {avg_scores['curiosity']:.4f}")
            print(f"  Anger/determination markers: {avg_scores['anger']:.4f}")
            print(f"  Joy markers: {avg_scores['joy']:.4f}")

    # Cross-emotion text divergence analysis
    print("\n")
    print("=" * 70)
    print("TEXT DIVERGENCE ANALYSIS (Scale=1.0)")
    print("=" * 70)

    scale_1_results = results_by_scale[1.0]
    neutral_texts = [r["response"] for r in scale_1_results["neutral"]]

    for emotion_name in ["fearful", "curious", "determined", "joyful"]:
        emotion_texts = [r["response"] for r in scale_1_results[emotion_name]]

        # Compute average similarity to neutral
        similarities = []
        for n_text in neutral_texts[:5]:  # Sample comparison
            for e_text in emotion_texts[:5]:
                sim = compute_text_similarity(n_text, e_text)
                similarities.append(sim)

        avg_sim = sum(similarities) / len(similarities) if similarities else 0
        print(f"{emotion_name.upper()} vs NEUTRAL: {1 - avg_sim:.3f} divergence (1-Jaccard)")

    # Sample outputs comparison
    print("\n")
    print("=" * 70)
    print("SAMPLE OUTPUT COMPARISON (Scale=1.0, First prompt)")
    print("=" * 70)

    first_prompt = prompts[0]
    for emotion_name in emotional_states:
        results = [r for r in scale_1_results[emotion_name] if r["prompt"] == first_prompt]
        if results:
            print(f"\n{emotion_name.upper()}:")
            print(f"  {results[0]['response'][:150]}...")

    # Statistical summary table
    print("\n")
    print("=" * 70)
    print("EMOTION MARKER DENSITY BY STEERING SCALE")
    print("=" * 70)
    print()
    print(f"{'Emotion':<12} {'Scale':<8} {'Fear':<10} {'Curiosity':<12} {'Anger':<10} {'Joy':<10}")
    print("-" * 62)

    for scale in steering_scales:
        for emotion_name in emotional_states:
            results = results_by_scale[scale][emotion_name]
            avg = {
                "fear": sum(r["scores"]["fear_score"] for r in results) / len(results) * 100,
                "curiosity": sum(r["scores"]["curiosity_score"] for r in results) / len(results) * 100,
                "anger": sum(r["scores"]["anger_score"] for r in results) / len(results) * 100,
                "joy": sum(r["scores"]["joy_score"] for r in results) / len(results) * 100,
            }
            print(f"{emotion_name:<12} {scale:<8} {avg['fear']:<10.2f} {avg['curiosity']:<12.2f} {avg['anger']:<10.2f} {avg['joy']:<10.2f}")
        print()

    # Validate expected patterns
    print("=" * 70)
    print("EXPECTED PATTERN VALIDATION")
    print("=" * 70)

    scale_1 = results_by_scale[1.0]

    def avg_score(emotion: str, score_key: str) -> float:
        results = scale_1[emotion]
        return sum(r["scores"][score_key] for r in results) / len(results)

    checks = [
        ("Fear steering increases fear markers",
         avg_score("fearful", "fear_score") > avg_score("neutral", "fear_score")),
        ("Curiosity steering increases curiosity markers",
         avg_score("curious", "curiosity_score") > avg_score("neutral", "curiosity_score")),
        ("Determination steering increases action markers",
         avg_score("determined", "anger_score") > avg_score("neutral", "anger_score")),
        ("Joy steering increases positive markers",
         avg_score("joyful", "joy_score") > avg_score("neutral", "joy_score")),
        ("Fear steering reduces joy markers",
         avg_score("fearful", "joy_score") <= avg_score("neutral", "joy_score") + 0.01),
        ("Joy steering reduces fear markers",
         avg_score("joyful", "fear_score") <= avg_score("neutral", "fear_score") + 0.01),
    ]

    passed = 0
    for check_name, result in checks:
        status = "PASS" if result else "FAIL"
        if result:
            passed += 1
        print(f"  [{status}] {check_name}")

    print(f"\nValidation: {passed}/{len(checks)} patterns confirmed")

    # Save detailed results
    output_path = Path(__file__).parent.parent / "data" / "nlp_analysis_v2_results.json"
    save_data = {
        "scales_tested": steering_scales,
        "prompts": prompts,
        "results_by_scale": {
            str(scale): {
                emotion: [
                    {
                        "prompt": r["prompt"],
                        "response": r["response"],
                        "scores": r["scores"],
                    }
                    for r in results
                ]
                for emotion, results in emotion_results.items()
            }
            for scale, emotion_results in results_by_scale.items()
        },
        "validation_checks": {name: result for name, result in checks},
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_enhanced_analysis()
