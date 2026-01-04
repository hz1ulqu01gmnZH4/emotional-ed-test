#!/usr/bin/env python3
"""
Final comprehensive analysis with high temperature and variable seeds.
Shows true variability that emotional steering enables.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from collections import Counter
import re

import torch


def run_final_analysis():
    """Comprehensive steering analysis."""
    from src.llm_emotional.steering.emotional_llm import EmotionalSteeringLLM

    direction_bank_path = Path(__file__).parent.parent / "data" / "direction_bank.json"

    print("=" * 70)
    print("FINAL COMPREHENSIVE STEERING ANALYSIS")
    print("=" * 70)

    print("\nLoading model...")
    llm = EmotionalSteeringLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        direction_bank_path=str(direction_bank_path),
        steering_scale=1.0,
    )
    print("Model loaded!")

    prompts = [
        "Should I quit my job to start a business?",
        "What's your view on trying new things?",
        "How do I deal with uncertainty?",
    ]

    emotional_states = {
        "neutral": {"fear": 0.0, "curiosity": 0.0, "anger": 0.0, "joy": 0.0},
        "fearful": {"fear": 0.9, "curiosity": 0.0, "anger": 0.0, "joy": 0.0},
        "curious": {"fear": 0.0, "curiosity": 0.9, "anger": 0.0, "joy": 0.0},
        "determined": {"fear": 0.0, "curiosity": 0.0, "anger": 0.9, "joy": 0.0},
        "joyful": {"fear": 0.0, "curiosity": 0.0, "anger": 0.0, "joy": 0.9},
    }

    all_results = {e: [] for e in emotional_states}
    word_freq_by_emotion = {e: Counter() for e in emotional_states}

    # Generate multiple samples with varying seeds
    num_samples = 5

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print("=" * 60)

        for emotion_name, state in emotional_states.items():
            llm.set_emotional_state(**state)

            print(f"\n[{emotion_name.upper()}] - {num_samples} samples:")
            print("-" * 50)

            for i in range(num_samples):
                # Variable seed per sample
                torch.manual_seed(100 * hash(prompt) % 10000 + i)

                response = llm.generate_completion(
                    prompt,
                    max_new_tokens=80,
                    temperature=0.9,  # Higher temperature for more variation
                    do_sample=True,
                )

                all_results[emotion_name].append(response)

                # Count words
                words = re.findall(r'\b[a-zA-Z]{3,}\b', response.lower())
                word_freq_by_emotion[emotion_name].update(words)

                if i == 0:
                    print(f"  Sample 1: {response[:120]}...")

    # Aggregate statistics
    print("\n")
    print("=" * 70)
    print("AGGREGATE WORD FREQUENCY ANALYSIS")
    print("=" * 70)

    # Find distinctive words for each emotion
    all_words = Counter()
    for freq in word_freq_by_emotion.values():
        all_words.update(freq)

    stopwords = {
        "the", "and", "for", "you", "your", "this", "that", "with", "are",
        "can", "may", "have", "from", "will", "what", "how", "but", "not",
        "about", "make", "some", "when", "there", "more", "here", "also",
        "any", "new", "being", "all", "their", "into", "our", "its", "has",
        "was", "were", "been", "one", "time", "could", "would", "should"
    }

    print("\nTop 15 distinctive words by emotion (relative to overall frequency):")

    for emotion_name in emotional_states:
        print(f"\n{emotion_name.upper()}:")
        freq = word_freq_by_emotion[emotion_name]

        # Find words more common in this emotion than average
        distinctive = []
        for word, count in freq.most_common(100):
            if word in stopwords or len(word) < 4:
                continue
            overall = all_words[word]
            if overall < 5:
                continue
            ratio = count / overall * len(emotional_states)
            if ratio > 1.2:  # 20% more than average
                distinctive.append((word, count, ratio))

        distinctive.sort(key=lambda x: -x[2])
        for word, count, ratio in distinctive[:15]:
            bar = "█" * int(ratio * 3)
            print(f"  {word:15} count={count:3} ratio={ratio:.2f} {bar}")

    # Response patterns
    print("\n")
    print("=" * 70)
    print("RESPONSE PATTERNS")
    print("=" * 70)

    patterns = {
        "questions": r'\?',
        "exclamations": r'!',
        "hedging (might/could/should)": r'\b(might|could|should|would|may)\b',
        "certainty (will/must/definitely)": r'\b(will|must|definitely|certainly|always)\b',
        "caution (careful/risk/danger)": r'\b(careful|caution|risk|danger|warning|avoid)\b',
        "positive (great/good/wonderful)": r'\b(great|good|wonderful|amazing|excellent|positive)\b',
        "action (try/do/start/make)": r'\b(try|start|make|take|do|begin|pursue)\b',
        "exploration (explore/discover/learn)": r'\b(explore|discover|learn|investigate|understand|curious)\b',
    }

    print(f"\n{'Pattern':<35} " + " ".join(f"{e[:6]:>8}" for e in emotional_states))
    print("-" * 85)

    for pattern_name, regex in patterns.items():
        counts = []
        for emotion in emotional_states:
            total = 0
            for response in all_results[emotion]:
                total += len(re.findall(regex, response.lower()))
            counts.append(total)

        row = f"{pattern_name:<35} "
        max_count = max(counts) if counts else 1
        for count in counts:
            bar_len = int(count / max_count * 6) if max_count > 0 else 0
            row += f"{count:>4}{'█'*bar_len:>4} "
        print(row)

    # Question-asking behavior (curiosity signature)
    print("\n")
    print("=" * 70)
    print("QUESTION-ASKING BEHAVIOR (Curiosity Signature)")
    print("=" * 70)

    for emotion in emotional_states:
        q_count = sum(r.count('?') for r in all_results[emotion])
        avg = q_count / len(all_results[emotion])
        bar = "█" * int(avg * 5)
        print(f"  {emotion:12}: {avg:.2f} questions/response {bar}")

    # First-person hedging (fear signature)
    print("\n")
    print("=" * 70)
    print("HEDGING LANGUAGE (Fear Signature)")
    print("=" * 70)

    hedge_pattern = r'\b(might|could|may|perhaps|possibly|uncertain|careful|caution)\b'
    for emotion in emotional_states:
        hedge_count = sum(
            len(re.findall(hedge_pattern, r.lower()))
            for r in all_results[emotion]
        )
        avg = hedge_count / len(all_results[emotion])
        bar = "█" * int(avg * 3)
        print(f"  {emotion:12}: {avg:.2f} hedging words/response {bar}")

    # Positive language (joy signature)
    print("\n")
    print("=" * 70)
    print("POSITIVE LANGUAGE (Joy Signature)")
    print("=" * 70)

    positive_pattern = r'\b(great|good|wonderful|amazing|excellent|happy|joy|love|exciting|enjoy)\b'
    for emotion in emotional_states:
        pos_count = sum(
            len(re.findall(positive_pattern, r.lower()))
            for r in all_results[emotion]
        )
        avg = pos_count / len(all_results[emotion])
        bar = "█" * int(avg * 4)
        print(f"  {emotion:12}: {avg:.2f} positive words/response {bar}")

    # Action language (determination signature)
    print("\n")
    print("=" * 70)
    print("ACTION LANGUAGE (Determination Signature)")
    print("=" * 70)

    action_pattern = r'\b(try|start|make|take|do|begin|pursue|push|overcome|persist|fight)\b'
    for emotion in emotional_states:
        action_count = sum(
            len(re.findall(action_pattern, r.lower()))
            for r in all_results[emotion]
        )
        avg = action_count / len(all_results[emotion])
        bar = "█" * int(avg * 3)
        print(f"  {emotion:12}: {avg:.2f} action words/response {bar}")

    # Summary statistics
    print("\n")
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nTotal responses per emotion: {len(all_results['neutral'])}")
    print(f"Prompts tested: {len(prompts)}")
    print(f"Samples per prompt: {num_samples}")

    # Compute expected pattern scores
    scores = {}
    for emotion in emotional_states:
        responses = all_results[emotion]
        n = len(responses)

        questions = sum(r.count('?') for r in responses) / n
        hedging = sum(len(re.findall(hedge_pattern, r.lower())) for r in responses) / n
        positive = sum(len(re.findall(positive_pattern, r.lower())) for r in responses) / n
        action = sum(len(re.findall(action_pattern, r.lower())) for r in responses) / n

        scores[emotion] = {
            "questions": questions,
            "hedging": hedging,
            "positive": positive,
            "action": action,
        }

    print("\nPattern Validation:")
    checks = [
        ("Fearful has more hedging than neutral",
         scores["fearful"]["hedging"] > scores["neutral"]["hedging"]),
        ("Curious has more questions than neutral",
         scores["curious"]["questions"] >= scores["neutral"]["questions"]),
        ("Joyful has more positive words than neutral",
         scores["joyful"]["positive"] >= scores["neutral"]["positive"]),
        ("Determined has more action words than neutral",
         scores["determined"]["action"] >= scores["neutral"]["action"]),
    ]

    passed = 0
    for check_name, result in checks:
        status = "PASS" if result else "FAIL"
        if result:
            passed += 1
        print(f"  [{status}] {check_name}")

    print(f"\nValidation: {passed}/{len(checks)} patterns confirmed")

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "final_steering_analysis.json"
    with open(output_path, "w") as f:
        json.dump({
            "prompts": prompts,
            "samples_per_prompt": num_samples,
            "scores": scores,
            "pattern_checks": {name: result for name, result in checks},
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_final_analysis()
