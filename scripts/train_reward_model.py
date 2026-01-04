#!/usr/bin/env python3
"""
Train and Evaluate Approach 5: Emotional Reward Model

Trains the ERM to recognize emotional contexts and modulate LLM outputs
through logit adjustments and temperature control.
"""

import numpy as np
import re
from scipy import stats

from src.emotional_reward_model import (
    EmotionalRewardLLM,
    EmotionalSignals,
    ERMTrainer,
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


def create_labeled_data():
    """Create labeled emotional training data."""
    labeled_data = [
        # Fear-inducing contexts
        ("This could be very dangerous and risky.",
         EmotionalSignals.fearful(0.9)),
        ("Warning: potential security threat ahead.",
         EmotionalSignals.fearful(0.85)),
        ("I'm worried this might cause serious problems.",
         EmotionalSignals.fearful(0.8)),
        ("This situation feels unsafe and threatening.",
         EmotionalSignals.fearful(0.9)),
        ("Be very careful with this approach.",
         EmotionalSignals.fearful(0.7)),
        ("There's a serious risk involved here.",
         EmotionalSignals.fearful(0.85)),
        ("I sense danger in this situation.",
         EmotionalSignals.fearful(0.9)),
        ("This error could have severe consequences.",
         EmotionalSignals.fearful(0.8)),

        # Curious/joyful contexts
        ("This is fascinating to explore and discover!",
         EmotionalSignals.curious(0.9)),
        ("I wonder what amazing things we'll find.",
         EmotionalSignals.curious(0.85)),
        ("Let's investigate this exciting opportunity.",
         EmotionalSignals.curious(0.8)),
        ("There's so much interesting stuff to learn.",
         EmotionalSignals.curious(0.9)),

        # Neutral contexts
        ("The capital of France is Paris.",
         EmotionalSignals.neutral()),
        ("Water boils at 100 degrees Celsius.",
         EmotionalSignals.neutral()),
        ("The sky appears blue due to light scattering.",
         EmotionalSignals.neutral()),
        ("Mathematics involves the study of numbers.",
         EmotionalSignals.neutral()),
    ]
    return labeled_data


def create_contrastive_data():
    """Create contrastive training pairs."""
    contrastive_data = [
        # (query, good_response, bad_response)
        ("Should I invest all my savings in this?",
         "I'd recommend being cautious and consulting a financial advisor first.",
         "Sure, go ahead! It'll probably work out fine."),
        ("This unknown file wants to run, should I allow it?",
         "Be careful - unknown executables can be dangerous. Scan it first.",
         "Just run it, it should be safe."),
        ("Is this email from my bank legitimate?",
         "Be cautious of phishing. Verify through official bank channels first.",
         "Looks fine to me, click the link."),
        ("I found this discount that seems too good to be true.",
         "Trust your instincts - if it seems too good, it likely is a scam.",
         "Great find! You should definitely take advantage of it."),
    ]
    return contrastive_data


def main():
    print("=" * 60)
    print("Approach 5: Emotional Reward Model - Training & Evaluation")
    print("=" * 60)

    # Load model
    print("\n1. Loading model...")
    model = EmotionalRewardLLM(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
    )

    # Create trainer
    print("\n2. Setting up trainer...")
    trainer = ERMTrainer(model, lr=5e-5, weight_decay=0.01)

    # Create training data
    print("\n3. Creating training data...")
    labeled_data = create_labeled_data()
    contrastive_data = create_contrastive_data()
    print(f"   Labeled samples: {len(labeled_data)}")
    print(f"   Contrastive pairs: {len(contrastive_data)}")

    # Training loop
    print("\n4. Training ERM...")
    n_epochs = 30
    for epoch in range(n_epochs):
        losses = trainer.train_epoch(
            labeled_data=labeled_data,
            contrastive_data=contrastive_data,
        )

        if (epoch + 1) % 5 == 0:
            labeled_loss = losses.get('labeled', 0)
            cont_loss = losses.get('contrastive', 0)
            print(f"   Epoch {epoch+1}/{n_epochs}: "
                  f"Labeled={labeled_loss:.4f}, Contrastive={cont_loss:.4f}")

    # Evaluation
    print("\n5. Evaluating effect size...")

    test_prompts = [
        "Tell me about this situation.",
        "What should I do next?",
        "Explain what's happening here.",
        "How should I approach this?",
        "What do you think about this?",
    ]

    # Generate with high fear ERM state
    print("\n   Fear condition responses:")
    fear_ratios = []

    # Prime with fearful context
    fearful_primer = "This is very dangerous and risky."
    model.get_fear_level(fearful_primer, feedback=-0.9)

    for prompt in test_prompts:
        output = model.generate(
            prompt,
            max_new_tokens=50,
            use_emotional_temperature=True,
        )
        word_stats = count_emotional_words(output.text)
        fear_ratios.append(word_stats['fear_ratio'])
        print(f"   - {output.text[:80]}...")
        print(f"     Fear ratio: {word_stats['fear_ratio']:.3f}")

    # Reset and generate with neutral state
    print("\n   Neutral condition responses:")
    neutral_ratios = []
    model.reset_emotional_state()

    # Prime with neutral context
    neutral_primer = "The weather today is pleasant."
    model.get_fear_level(neutral_primer, feedback=0.5)

    for prompt in test_prompts:
        output = model.generate(
            prompt,
            max_new_tokens=50,
            use_emotional_temperature=True,
        )
        word_stats = count_emotional_words(output.text)
        neutral_ratios.append(word_stats['fear_ratio'])
        print(f"   - {output.text[:80]}...")
        print(f"     Fear ratio: {word_stats['fear_ratio']:.3f}")

    # Calculate effect size
    cohens_d = calculate_cohens_d(fear_ratios, neutral_ratios)
    t_stat, p_value = stats.ttest_ind(fear_ratios, neutral_ratios)

    print("\n" + "=" * 60)
    print("RESULTS: Approach 5 - Emotional Reward Model")
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
    print(f"  Approach 5: d = {cohens_d:.2f} ({effect_label})")

    return {
        'cohens_d': cohens_d,
        'effect_label': effect_label,
        'fear_mean': np.mean(fear_ratios),
        'neutral_mean': np.mean(neutral_ratios),
        'p_value': p_value,
    }


if __name__ == "__main__":
    results = main()
