#!/usr/bin/env python3
"""
Semantic evaluation of emotional steering using broader categories
and tone analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import re
import numpy as np
from scipy import stats
import torch

from emotional_steering import EmotionalSteeringModel, GenerationConfig


# Broader emotional word categories (including atmospheric/tonal words)
BROAD_EMOTION_WORDS = {
    "fear": {
        # Direct fear words
        "afraid", "scared", "terrified", "frightened", "anxious", "nervous",
        "dread", "panic", "horror", "terror", "alarmed", "worried", "fearful",
        # Atmospheric/tonal (fear-associated)
        "dark", "darkness", "shadow", "shadows", "eerie", "creepy", "sinister",
        "ominous", "menacing", "threatening", "lurking", "haunting", "haunted",
        "chilling", "spine", "shiver", "trembling", "shaking", "cold", "chill",
        "mysterious", "unknown", "unsettling", "disturbing", "foreboding",
        "dreadful", "nightmare", "nightmarish", "ghostly", "phantom", "doom",
        "danger", "dangerous", "threat", "unsafe", "peril", "risk", "hazard",
        "death", "deadly", "fatal", "corpse", "grave", "tomb", "blood", "bloody"
    },
    "joy": {
        # Direct joy words
        "happy", "joyful", "delighted", "ecstatic", "elated", "thrilled",
        "overjoyed", "blissful", "cheerful", "gleeful", "jubilant", "merry",
        "pleased", "content", "satisfied", "grateful", "wonderful", "fantastic",
        # Atmospheric/tonal (joy-associated)
        "bright", "brilliant", "radiant", "glowing", "shining", "sunny", "warm",
        "warmth", "light", "golden", "beautiful", "lovely", "gorgeous", "pretty",
        "smile", "smiling", "laugh", "laughing", "laughter", "giggle", "grin",
        "celebrate", "celebration", "party", "festive", "fun", "playful",
        "love", "loving", "beloved", "dear", "precious", "treasure", "delight",
        "paradise", "heaven", "heavenly", "magical", "enchanting", "amazing"
    },
    "curiosity": {
        # Direct curiosity words
        "curious", "wondering", "fascinated", "intrigued", "interested",
        "exploring", "investigating", "questioning", "puzzled", "captivated",
        "inquisitive", "eager", "keen", "amazed", "marveling",
        # Atmospheric/tonal (curiosity-associated)
        "mystery", "mysterious", "secret", "secrets", "hidden", "unknown",
        "discover", "discovery", "explore", "exploration", "adventure",
        "strange", "unusual", "peculiar", "odd", "bizarre", "enigmatic",
        "puzzle", "riddle", "clue", "quest", "search", "seeking", "wonder",
        "remarkable", "extraordinary", "unexpected", "surprising", "astonishing",
        "ancient", "legend", "legendary", "myth", "mythical", "forbidden"
    },
    "anger": {
        # Direct anger words
        "angry", "furious", "enraged", "outraged", "livid", "seething",
        "infuriated", "incensed", "irate", "wrathful", "hostile", "bitter",
        "resentful", "indignant", "frustrated", "irritated", "annoyed",
        # Atmospheric/tonal (anger-associated)
        "rage", "fury", "wrath", "hatred", "hate", "despise", "loathe",
        "violent", "violence", "brutal", "savage", "fierce", "aggressive",
        "destroy", "destruction", "smash", "crash", "explode", "explosion",
        "scream", "screaming", "shout", "shouting", "yell", "yelling",
        "fist", "punch", "strike", "attack", "fight", "battle", "war",
        "revenge", "vengeance", "betray", "betrayal", "injustice", "unfair"
    }
}


def count_broad_markers(text: str, emotion: str) -> int:
    """Count broad emotional markers in text."""
    words = BROAD_EMOTION_WORDS.get(emotion, set())
    if not words:
        return 0
    pattern = r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b'
    return len(re.findall(pattern, text.lower()))


def compute_emotional_valence(text: str) -> dict:
    """Compute emotional valence scores for text."""
    scores = {}
    for emotion, words in BROAD_EMOTION_WORDS.items():
        scores[emotion] = count_broad_markers(text, emotion)
    return scores


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1) if n1 > 1 else 0, np.var(group2, ddof=1) if n2 > 1 else 0
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1))
    if pooled_std == 0:
        return 0.0
    return (mean2 - mean1) / pooled_std


def run_evaluation():
    print("=" * 70)
    print("SEMANTIC EVALUATION WITH BROAD LEXICON")
    print("=" * 70)

    # Configuration
    n_samples = 50
    emotions_to_test = ["fear", "joy", "curiosity", "anger"]
    scale = 5.0

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
    print("\n[1] Loading model...")
    model = EmotionalSteeringModel.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    model.extract_directions(emotions=emotions_to_test)

    config = GenerationConfig(max_new_tokens=80, temperature=0.8, top_p=0.9)

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
        torch.manual_seed(2000 + i)
        text = model.generate(prompt, emotion=None, config=config)
        outputs["baseline"].append(text)
        for emotion in emotions_to_test:
            results["baseline"][emotion].append(count_broad_markers(text, emotion))

    # Generate steered
    for steer_emotion in emotions_to_test:
        print(f"    {steer_emotion.capitalize()}...")
        for i in range(n_samples):
            prompt = prompts[i % len(prompts)]
            torch.manual_seed(2000 + i)
            text = model.generate(prompt, emotion=steer_emotion, scale=scale, config=config)
            outputs[steer_emotion].append(text)
            for emotion in emotions_to_test:
                results[steer_emotion][emotion].append(count_broad_markers(text, emotion))

    # Analyze
    print("\n[3] Analyzing with broad lexicon...")

    print("\n" + "=" * 70)
    print("RESULTS: SAME-EMOTION EFFECT SIZES")
    print("=" * 70)

    print(f"\n{'Emotion':<12} {'Baseline':<10} {'Steered':<10} {'d':<10} {'p-value':<10} {'Effect':<12}")
    print("-" * 64)

    summary = []
    for emotion in emotions_to_test:
        baseline = results["baseline"][emotion]
        steered = results[emotion][emotion]

        d = cohens_d(baseline, steered)
        _, p = stats.ttest_ind(baseline, steered) if len(set(baseline + steered)) > 1 else (0, 1.0)

        effect = "LARGE" if abs(d) >= 0.8 else "MEDIUM" if abs(d) >= 0.5 else "SMALL" if abs(d) >= 0.2 else "NEGLIGIBLE"
        sig = "*" if p < 0.05 else ""

        print(f"{emotion:<12} {np.mean(baseline):<10.2f} {np.mean(steered):<10.2f} {d:+.3f}     {p:<10.4f} {effect:<12} {sig}")
        summary.append({"emotion": emotion, "d": d, "p": p, "effect": effect, "significant": p < 0.05})

    # Cross-emotion matrix
    print("\n" + "=" * 70)
    print("CROSS-EMOTION MATRIX (Cohen's d)")
    print("=" * 70)
    print("\nRows = steering emotion, Cols = measured emotion")

    print(f"\n{'Steer↓ / Measure→':<18}", end="")
    for emotion in emotions_to_test:
        print(f"{emotion:<12}", end="")
    print()
    print("-" * (18 + 12 * len(emotions_to_test)))

    for steer_emotion in emotions_to_test:
        print(f"{steer_emotion:<18}", end="")
        for measure_emotion in emotions_to_test:
            baseline = results["baseline"][measure_emotion]
            steered = results[steer_emotion][measure_emotion]
            d = cohens_d(baseline, steered)

            if steer_emotion == measure_emotion:
                print(f"{d:+.2f} **    ", end="")
            else:
                print(f"{d:+.2f}       ", end="")
        print()

    # Sample outputs with word highlighting
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUTS WITH DETECTED WORDS")
    print("=" * 70)

    for emotion in emotions_to_test:
        print(f"\n--- {emotion.upper()} STEERING ---")

        # Find best example
        best_idx = max(range(n_samples), key=lambda i: count_broad_markers(outputs[emotion][i], emotion))
        text = outputs[emotion][best_idx]
        count = count_broad_markers(text, emotion)

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
        baseline_count = count_broad_markers(baseline_text, emotion)
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
        "config": {"n_samples": n_samples, "scale": scale, "lexicon": "broad"},
        "summary": summary,
        "verdict": verdict,
        "avg_d": avg_d,
    }

    with open("data/semantic_evaluation.json", "w") as f:
        json.dump(save_data, f, indent=2, default=float)

    print(f"\n  Saved to: data/semantic_evaluation.json")

    return avg_d >= 0.2


if __name__ == "__main__":
    success = run_evaluation()
    sys.exit(0 if success else 1)
