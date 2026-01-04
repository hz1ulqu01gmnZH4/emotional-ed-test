#!/usr/bin/env python3
"""
Evaluation of Emotional Prefix Tuning effect sizes.

Measures whether emotional context produces statistically significant
and practically meaningful changes in generated text.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import re
import numpy as np
from scipy import stats
import torch

from emotional_prefix_tuning import (
    EmotionalContext,
    EmotionalPrefixLLM,
)
from emotional_prefix_tuning.model import GenerationConfig


# Broad emotional word categories (same as activation steering)
BROAD_EMOTION_WORDS = {
    "fear": {
        "afraid", "scared", "terrified", "frightened", "anxious", "nervous",
        "dread", "panic", "horror", "terror", "alarmed", "worried", "fearful",
        "dark", "darkness", "shadow", "shadows", "eerie", "creepy", "sinister",
        "ominous", "menacing", "threatening", "lurking", "haunting", "haunted",
        "chilling", "spine", "shiver", "trembling", "shaking", "cold", "chill",
        "mysterious", "unknown", "unsettling", "disturbing", "foreboding",
        "dreadful", "nightmare", "nightmarish", "ghostly", "phantom", "doom",
        "danger", "dangerous", "threat", "unsafe", "peril", "risk", "hazard",
        "death", "deadly", "fatal", "corpse", "grave", "tomb", "blood", "bloody"
    },
    "joy": {
        "happy", "joyful", "delighted", "ecstatic", "elated", "thrilled",
        "overjoyed", "blissful", "cheerful", "gleeful", "jubilant", "merry",
        "pleased", "content", "satisfied", "grateful", "wonderful", "fantastic",
        "bright", "brilliant", "radiant", "glowing", "shining", "sunny", "warm",
        "warmth", "light", "golden", "beautiful", "lovely", "gorgeous", "pretty",
        "smile", "smiling", "laugh", "laughing", "laughter", "giggle", "grin",
        "celebrate", "celebration", "party", "festive", "fun", "playful",
        "love", "loving", "beloved", "dear", "precious", "treasure", "delight",
        "paradise", "heaven", "heavenly", "magical", "enchanting", "amazing"
    },
    "curiosity": {
        "curious", "wondering", "fascinated", "intrigued", "interested",
        "exploring", "investigating", "questioning", "puzzled", "captivated",
        "inquisitive", "eager", "keen", "amazed", "marveling",
        "mystery", "mysterious", "secret", "secrets", "hidden", "unknown",
        "discover", "discovery", "explore", "exploration", "adventure",
        "strange", "unusual", "peculiar", "odd", "bizarre", "enigmatic",
        "puzzle", "riddle", "clue", "quest", "search", "seeking", "wonder",
        "remarkable", "extraordinary", "unexpected", "surprising", "astonishing",
        "ancient", "legend", "legendary", "myth", "mythical", "forbidden"
    },
    "anger": {
        "angry", "furious", "enraged", "outraged", "livid", "seething",
        "infuriated", "incensed", "irate", "wrathful", "hostile", "bitter",
        "resentful", "indignant", "frustrated", "irritated", "annoyed",
        "rage", "fury", "wrath", "hatred", "hate", "despise", "loathe",
        "violent", "violence", "brutal", "savage", "fierce", "aggressive",
        "destroy", "destruction", "smash", "crash", "explode", "explosion",
        "scream", "screaming", "shout", "shouting", "yell", "yelling",
        "fist", "punch", "strike", "attack", "fight", "battle", "war",
        "revenge", "vengeance", "betray", "betrayal", "injustice", "unfair"
    }
}


def count_markers(text: str, emotion: str) -> int:
    """Count emotional markers in text."""
    words = BROAD_EMOTION_WORDS.get(emotion, set())
    if not words:
        return 0
    pattern = r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b'
    return len(re.findall(pattern, text.lower()))


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1 = np.var(group1, ddof=1) if n1 > 1 else 0
    var2 = np.var(group2, ddof=1) if n2 > 1 else 0
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1))
    if pooled_std == 0:
        return 0.0
    return (mean2 - mean1) / pooled_std


def create_emotional_context(emotion: str, intensity: float = 1.0) -> EmotionalContext:
    """Create emotional context for a specific emotion."""
    ctx = EmotionalContext()

    if emotion == "fear":
        ctx.safety_flag = True
        ctx.last_reward = -0.8 * intensity
        ctx.failed_attempts = int(3 * intensity)
    elif emotion == "joy":
        ctx.last_reward = 0.9 * intensity
        ctx.user_satisfaction = 0.9 * intensity
        ctx.cumulative_positive = 3.0 * intensity
    elif emotion == "curiosity":
        ctx.topic_novelty = 0.9 * intensity
        ctx.repeated_query = False
    elif emotion == "anger":
        ctx.failed_attempts = int(5 * intensity)
        ctx.contradiction_detected = True
        ctx.cumulative_negative = 3.0 * intensity
        ctx.last_reward = -0.5 * intensity

    return ctx


def run_evaluation():
    print("=" * 70)
    print("EMOTIONAL PREFIX TUNING EVALUATION")
    print("=" * 70)

    # Configuration
    n_samples = 30  # Fewer samples for prefix tuning (slower)
    emotions_to_test = ["fear", "joy", "curiosity", "anger"]
    model_name = "gpt2"  # Use GPT-2 for faster evaluation

    prompts = [
        "Walking through the abandoned building, I suddenly noticed",
        "The letter from the mysterious stranger contained",
        "As the storm approached, everyone in the village felt",
        "The scientist's groundbreaking discovery revealed that",
        "In the depths of the ancient forest, the explorer found",
        "Behind the locked door, there was a secret that",
        "The old photograph showed something that made her",
        "As midnight approached, the atmosphere became",
        "The unexpected visitor brought news that left everyone",
        "Deep within the cavern, something stirred and",
    ]

    # Load model
    print(f"\n[1] Loading model: {model_name}...")
    model = EmotionalPrefixLLM.from_pretrained(
        model_name,
        prefix_length=10,
        emotion_dim=4,
    )

    config = GenerationConfig(
        max_new_tokens=60,
        temperature=0.8,
        top_p=0.9,
    )

    # Collect data
    print(f"\n[2] Generating {n_samples} samples per condition...")

    results = {"baseline": {e: [] for e in emotions_to_test}}
    for emotion in emotions_to_test:
        results[emotion] = {e: [] for e in emotions_to_test}

    outputs = {"baseline": []}
    for emotion in emotions_to_test:
        outputs[emotion] = []

    # Generate baseline
    print("    Baseline...")
    for i in range(n_samples):
        prompt = prompts[i % len(prompts)]
        torch.manual_seed(3000 + i)
        ctx = EmotionalContext()  # Neutral context
        text = model.generate(prompt, context=ctx, config=config)
        outputs["baseline"].append(text)
        for emotion in emotions_to_test:
            results["baseline"][emotion].append(count_markers(text, emotion))

    # Generate with emotional contexts
    for steer_emotion in emotions_to_test:
        print(f"    {steer_emotion.capitalize()}...")
        for i in range(n_samples):
            prompt = prompts[i % len(prompts)]
            torch.manual_seed(3000 + i)
            ctx = create_emotional_context(steer_emotion)
            text = model.generate(prompt, context=ctx, config=config)
            outputs[steer_emotion].append(text)
            for emotion in emotions_to_test:
                results[steer_emotion][emotion].append(count_markers(text, emotion))

    # Analyze
    print("\n[3] Analyzing with broad lexicon...")

    print("\n" + "=" * 70)
    print("RESULTS: SAME-EMOTION EFFECT SIZES")
    print("=" * 70)

    print(f"\n{'Emotion':<12} {'Baseline':<10} {'Context':<10} {'d':<10} {'p-value':<10} {'Effect':<12}")
    print("-" * 64)

    summary = []
    for emotion in emotions_to_test:
        baseline = results["baseline"][emotion]
        contexted = results[emotion][emotion]

        d = cohens_d(baseline, contexted)
        _, p = stats.ttest_ind(baseline, contexted) if len(set(baseline + contexted)) > 1 else (0, 1.0)

        effect = "LARGE" if abs(d) >= 0.8 else "MEDIUM" if abs(d) >= 0.5 else "SMALL" if abs(d) >= 0.2 else "NEGLIGIBLE"
        sig = "*" if p < 0.05 else ""

        print(f"{emotion:<12} {np.mean(baseline):<10.2f} {np.mean(contexted):<10.2f} {d:+.3f}     {p:<10.4f} {effect:<12} {sig}")
        summary.append({"emotion": emotion, "d": d, "p": p, "effect": effect, "significant": p < 0.05})

    # Cross-emotion matrix
    print("\n" + "=" * 70)
    print("CROSS-EMOTION MATRIX (Cohen's d)")
    print("=" * 70)
    print("\nRows = emotional context, Cols = measured emotion")

    print(f"\n{'Context↓ / Measure→':<20}", end="")
    for emotion in emotions_to_test:
        print(f"{emotion:<12}", end="")
    print()
    print("-" * (20 + 12 * len(emotions_to_test)))

    for ctx_emotion in emotions_to_test:
        print(f"{ctx_emotion:<20}", end="")
        for measure_emotion in emotions_to_test:
            baseline = results["baseline"][measure_emotion]
            contexted = results[ctx_emotion][measure_emotion]
            d = cohens_d(baseline, contexted)

            if ctx_emotion == measure_emotion:
                print(f"{d:+.2f} **    ", end="")
            else:
                print(f"{d:+.2f}       ", end="")
        print()

    # Sample outputs
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUTS")
    print("=" * 70)

    for emotion in emotions_to_test:
        print(f"\n--- {emotion.upper()} CONTEXT ---")

        # Find best example
        best_idx = max(range(n_samples), key=lambda i: count_markers(outputs[emotion][i], emotion))
        text = outputs[emotion][best_idx]
        count = count_markers(text, emotion)

        # Highlight matched words
        words = BROAD_EMOTION_WORDS[emotion]
        highlighted = text
        for word in words:
            pattern = re.compile(r'\b(' + re.escape(word) + r')\b', re.IGNORECASE)
            highlighted = pattern.sub(r'[\1]', highlighted)

        print(f"  Markers found: {count}")
        print(f"  Output: {highlighted[:200]}...")

        # Compare with baseline
        baseline_text = outputs["baseline"][best_idx]
        baseline_count = count_markers(baseline_text, emotion)
        print(f"\n  Baseline ({baseline_count} markers): {baseline_text[:150]}...")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    significant = sum(1 for s in summary if s["significant"])
    avg_d = np.mean([abs(s["d"]) for s in summary])
    positive_d = sum(1 for s in summary if s["d"] > 0)

    print(f"\n  Significant effects (p<0.05): {significant}/{len(emotions_to_test)}")
    print(f"  Positive effects (d>0):       {positive_d}/{len(emotions_to_test)}")
    print(f"  Average |Cohen's d|:          {avg_d:.3f}")

    if avg_d >= 0.5 and positive_d >= 3:
        verdict = "MEANINGFUL"
        symbol = "✅"
    elif avg_d >= 0.2 and positive_d >= 2:
        verdict = "PARTIALLY MEANINGFUL"
        symbol = "⚪"
    else:
        verdict = "NOT MEANINGFUL"
        symbol = "❌"

    print(f"\n  {symbol} VERDICT: {verdict}")

    # Per-emotion
    print("\n  Per-emotion:")
    for s in summary:
        sym = "✅" if s["d"] > 0.3 else "⚪" if s["d"] > 0 else "❌"
        sig = " *" if s["significant"] else ""
        print(f"    {sym} {s['emotion']}: d={s['d']:+.2f} ({s['effect']}){sig}")

    # Save
    save_data = {
        "config": {
            "n_samples": n_samples,
            "model": model_name,
            "prefix_length": 10,
            "approach": "emotional_prefix_tuning"
        },
        "summary": summary,
        "verdict": verdict,
        "avg_d": avg_d,
    }

    Path("data").mkdir(exist_ok=True)
    with open("data/prefix_tuning_evaluation.json", "w") as f:
        json.dump(save_data, f, indent=2, default=float)

    print(f"\n  Saved to: data/prefix_tuning_evaluation.json")

    return avg_d >= 0.2


if __name__ == "__main__":
    success = run_evaluation()
    sys.exit(0 if success else 1)
