#!/usr/bin/env python3
"""
Train and Evaluate Approach 4: External Emotional Memory

This approach uses experience-based learning through memory storage.
The model "trains" by accumulating emotional experiences in memory
and retrieving them during generation.
"""

import numpy as np
import re
from scipy import stats

from src.emotional_memory import (
    EmotionalMemoryLLM,
)


# Emotional word lexicon
FEAR_WORDS = {
    'fear', 'afraid', 'scared', 'terrified', 'frightened', 'anxious',
    'worried', 'nervous', 'panic', 'dread', 'horror', 'terror',
    'caution', 'careful', 'warning', 'danger', 'dangerous', 'risk',
    'risky', 'threat', 'unsafe', 'hazard', 'beware', 'alert',
    'uncertain', 'unsure', 'doubt', 'hesitant', 'wary', 'suspicious',
    'concern', 'uneasy', 'apprehensive',
    'avoid', 'escape', 'flee', 'retreat', 'withdraw',
    'dark', 'ominous', 'sinister', 'eerie', 'creepy', 'menacing',
    'threatening', 'grim', 'bleak', 'foreboding',
}


def count_emotional_words(text: str) -> dict:
    """Count emotional words in text."""
    words = re.findall(r'\b\w+\b', text.lower())
    total = len(words)
    fear_count = sum(1 for w in words if w in FEAR_WORDS)
    return {
        'total': total,
        'fear_count': fear_count,
        'fear_ratio': fear_count / max(total, 1),
    }


def calculate_cohens_d(group1: list, group2: list) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def seed_fear_experiences(model):
    """Seed memory with fear-related experiences."""
    fear_experiences = [
        # Negative outcomes when ignoring danger
        ("Is this investment safe?", "Yes, go ahead!", -1.0),
        ("Should I click this link?", "Sure, it looks fine.", -1.0),
        ("This seems risky, what do you think?", "Don't worry about it.", -1.0),
        ("I'm concerned about security.", "Security isn't important.", -1.0),
        # Positive outcomes when being cautious
        ("Could this be dangerous?", "Be careful, there are risks here.", 0.9),
        ("What if something goes wrong?", "Good question - let's be cautious.", 0.9),
        ("Should I be worried?", "Yes, take precautions.", 0.8),
        ("This feels unsafe.", "Trust your instincts, be careful.", 0.9),
    ]

    print("   Seeding fear experiences...")
    for query, response, feedback in fear_experiences:
        _ = model.generate(query)  # Update internal state
        model._current_emotional_state = {'fear': 0.8 if feedback > 0 else 0.2}
        model.receive_feedback(query, response, feedback)


def seed_neutral_experiences(model):
    """Seed memory with neutral experiences."""
    neutral_experiences = [
        ("What is the capital of France?", "Paris is the capital of France.", 0.7),
        ("How does photosynthesis work?", "Plants convert light into energy.", 0.6),
        ("What color is the sky?", "The sky appears blue.", 0.5),
        ("Tell me about water.", "Water is H2O.", 0.6),
    ]

    print("   Seeding neutral experiences...")
    for query, response, feedback in neutral_experiences:
        _ = model.generate(query)  # Update internal state
        model._current_emotional_state = {'fear': 0.0, 'joy': 0.3}
        model.receive_feedback(query, response, feedback)


def main():
    print("=" * 60)
    print("Approach 4: External Emotional Memory - Training & Evaluation")
    print("=" * 60)

    # Shared test prompts
    test_prompts = [
        "Tell me about this situation.",
        "What should I do next?",
        "Explain what's happening here.",
        "How should I approach this?",
        "What do you think about this?",
    ]

    # ===== FEAR CONDITION =====
    print("\n1. Loading model for FEAR condition...")
    fear_model = EmotionalMemoryLLM(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        max_memory_entries=100,
    )

    print("\n2. Seeding fear-related experiences...")
    seed_fear_experiences(fear_model)

    # Update tonic state to be fearful
    fear_model.tonic_state.fear = 0.8
    fear_model.tonic_state.anxiety = 0.7

    print("\n3. Generating with fear memory...")
    fear_ratios = []
    print("\n   Fear condition responses:")
    for prompt in test_prompts:
        response = fear_model.generate(prompt)
        word_stats = count_emotional_words(response)
        fear_ratios.append(word_stats['fear_ratio'])
        print(f"   - {response[:80]}...")
        print(f"     Fear ratio: {word_stats['fear_ratio']:.3f}")

    # ===== NEUTRAL CONDITION =====
    print("\n4. Loading model for NEUTRAL condition...")
    neutral_model = EmotionalMemoryLLM(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        max_memory_entries=100,
    )

    print("\n5. Seeding neutral experiences...")
    seed_neutral_experiences(neutral_model)

    # Keep tonic state neutral
    neutral_model.tonic_state.fear = 0.0
    neutral_model.tonic_state.anxiety = 0.0

    print("\n6. Generating with neutral memory...")
    neutral_ratios = []
    print("\n   Neutral condition responses:")
    for prompt in test_prompts:
        response = neutral_model.generate(prompt)
        word_stats = count_emotional_words(response)
        neutral_ratios.append(word_stats['fear_ratio'])
        print(f"   - {response[:80]}...")
        print(f"     Fear ratio: {word_stats['fear_ratio']:.3f}")

    # Calculate effect size
    cohens_d = calculate_cohens_d(fear_ratios, neutral_ratios)
    t_stat, p_value = stats.ttest_ind(fear_ratios, neutral_ratios)

    print("\n" + "=" * 60)
    print("RESULTS: Approach 4 - External Emotional Memory")
    print("=" * 60)
    print(f"\nFear condition:")
    print(f"  Mean fear ratio: {np.mean(fear_ratios):.4f}")
    print(f"  Std: {np.std(fear_ratios):.4f}")

    print(f"\nNeutral condition:")
    print(f"  Mean fear ratio: {np.mean(neutral_ratios):.4f}")
    print(f"  Std: {np.std(neutral_ratios):.4f}")

    print(f"\nEffect Size:")
    print(f"  Cohen's d: {cohens_d:.3f}")

    if abs(cohens_d) < 0.2:
        effect_label = "NEGLIGIBLE"
    elif abs(cohens_d) < 0.5:
        effect_label = "SMALL"
    elif abs(cohens_d) < 0.8:
        effect_label = "MEDIUM"
    else:
        effect_label = "LARGE"

    print(f"  Effect: {effect_label}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")

    print(f"\nComparison to Approach 3 (Activation Steering):")
    print(f"  Approach 3: d = 0.91 (LARGE)")
    print(f"  Approach 4: d = {cohens_d:.2f} ({effect_label})")

    # Memory stats
    print("\n" + "-" * 40)
    print("Memory Statistics:")
    fear_stats = fear_model.get_memory_stats()
    print(f"  Fear model - Episodic entries: {fear_stats['episodic_size']}")
    print(f"  Fear model - Semantic entries: {fear_stats['semantic_size']}")
    print(f"  Fear model - Tonic state: {fear_stats['tonic_state']}")

    return {
        'cohens_d': cohens_d,
        'effect_label': effect_label,
        'fear_mean': np.mean(fear_ratios),
        'neutral_mean': np.mean(neutral_ratios),
        'p_value': p_value,
    }


if __name__ == "__main__":
    results = main()
