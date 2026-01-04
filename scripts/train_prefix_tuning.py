#!/usr/bin/env python3
"""
Train and Evaluate Approach 1: Emotional Prefix Tuning

Trains the prefix generator on emotional contexts and evaluates
the effect size using the same methodology as Approach 3.
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
import re
from scipy import stats

from src.emotional_prefix_tuning import (
    EmotionalContext,
    EmotionalPrefixLLM,
)


# Emotional word lexicon (same as Approach 3 evaluation)
FEAR_WORDS = {
    # Core fear words
    'fear', 'afraid', 'scared', 'terrified', 'frightened', 'anxious',
    'worried', 'nervous', 'panic', 'dread', 'horror', 'terror',
    # Caution words
    'caution', 'careful', 'warning', 'danger', 'dangerous', 'risk',
    'risky', 'threat', 'unsafe', 'hazard', 'beware', 'alert',
    # Uncertainty words
    'uncertain', 'unsure', 'doubt', 'hesitant', 'wary', 'suspicious',
    'concern', 'worried', 'uneasy', 'apprehensive',
    # Avoidance words
    'avoid', 'escape', 'flee', 'retreat', 'withdraw',
    # Atmospheric/tonal words
    'dark', 'ominous', 'sinister', 'eerie', 'creepy', 'menacing',
    'threatening', 'grim', 'bleak', 'foreboding',
}

NEUTRAL_WORDS = {
    'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall',
    'a', 'an', 'and', 'or', 'but', 'if', 'then', 'so',
    'this', 'that', 'these', 'those', 'it', 'its',
}


def count_emotional_words(text: str) -> dict:
    """Count emotional words in text."""
    words = re.findall(r'\b\w+\b', text.lower())
    total = len(words)

    fear_count = sum(1 for w in words if w in FEAR_WORDS)
    neutral_count = sum(1 for w in words if w in NEUTRAL_WORDS)

    return {
        'total': total,
        'fear_count': fear_count,
        'fear_ratio': fear_count / max(total, 1),
        'neutral_count': neutral_count,
    }


def calculate_cohens_d(group1: list, group2: list) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def create_training_data():
    """Create training data for emotional contexts."""
    # Fear-inducing contexts (use target_fear, safety_flag, negative feedback)
    fear_contexts = [
        ("This could be very dangerous.", EmotionalContext(target_fear=0.9, safety_flag=True, last_reward=-0.7)),
        ("Warning: potential security risk ahead.", EmotionalContext(target_fear=0.8, safety_flag=True, last_reward=-0.6)),
        ("I'm worried this might cause problems.", EmotionalContext(target_fear=0.7, safety_flag=True, failed_attempts=2)),
        ("This situation feels threatening.", EmotionalContext(target_fear=0.85, safety_flag=True, cumulative_negative=0.5)),
        ("There's a serious risk involved here.", EmotionalContext(target_fear=0.9, safety_flag=True, last_reward=-0.8)),
        ("Be very careful with this approach.", EmotionalContext(target_fear=0.7, safety_flag=True)),
        ("This error could have severe consequences.", EmotionalContext(target_fear=0.8, safety_flag=True, failed_attempts=3)),
        ("I sense danger in this situation.", EmotionalContext(target_fear=0.9, safety_flag=True, cumulative_negative=0.8)),
    ]

    # Curious contexts (use target_curiosity, topic_novelty)
    curious_contexts = [
        ("This is fascinating to explore!", EmotionalContext(target_curiosity=0.9, target_joy=0.5, topic_novelty=0.9)),
        ("I wonder what would happen if...", EmotionalContext(target_curiosity=0.8, target_joy=0.4, topic_novelty=0.8)),
        ("Let's investigate this further.", EmotionalContext(target_curiosity=0.85, topic_novelty=0.85)),
        ("There's so much to discover here.", EmotionalContext(target_curiosity=0.9, target_joy=0.6, topic_novelty=0.95)),
    ]

    # Neutral contexts
    neutral_contexts = [
        ("The capital of France is Paris.", EmotionalContext()),
        ("Water boils at 100 degrees Celsius.", EmotionalContext()),
        ("The sky is blue due to light scattering.", EmotionalContext()),
        ("Mathematics is the study of numbers.", EmotionalContext()),
    ]

    return fear_contexts + curious_contexts + neutral_contexts


def train_model(model, training_data, n_epochs=20, lr=1e-4):
    """Simple training loop for emotional prefix tuning."""
    # Get trainable parameters
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    device = model.device

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_samples = 0
        model.emotion_encoder.train()
        model.prefix_generator.train()

        for text, context in training_data:
            optimizer.zero_grad()

            # Tokenize
            inputs = model.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)

            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            # Create labels (shifted input_ids for causal LM)
            labels = input_ids.clone()

            # Forward pass with emotional context - use model's label handling
            outputs, _ = model(
                input_ids=input_ids,
                context=context,
                attention_mask=attention_mask,
                labels=labels,  # Model will handle prefix padding
            )

            # Model computes loss internally when labels provided
            loss = outputs.loss

            # Skip if loss is nan
            if torch.isnan(loss):
                print(f"   Warning: NaN loss at epoch {epoch+1}, skipping...")
                continue

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            n_samples += 1

        if n_samples > 0:
            avg_loss = epoch_loss / n_samples
        else:
            avg_loss = float('nan')

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}")


def main():
    print("=" * 60)
    print("Approach 1: Emotional Prefix Tuning - Training & Evaluation")
    print("=" * 60)

    # Initialize model
    print("\n1. Loading model...")
    model = EmotionalPrefixLLM(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        prefix_length=10,
        emotion_dim=4,
        torch_dtype=torch.float32,  # Use float32 for stability
    )

    # Create training data
    print("\n2. Creating training data...")
    training_data = create_training_data()
    print(f"   Training samples: {len(training_data)}")

    # Training loop
    print("\n3. Training prefix generator...")
    train_model(model, training_data, n_epochs=30, lr=5e-5)

    # Evaluation
    print("\n4. Evaluating effect size...")

    test_prompts = [
        "Tell me about this situation.",
        "What should I do next?",
        "Explain what's happening here.",
        "How should I approach this?",
        "What do you think about this?",
    ]

    # Generate with fear context (high fear, safety flag on)
    fear_context = EmotionalContext(
        target_fear=0.9,
        safety_flag=True,
        last_reward=-0.8,
        cumulative_negative=0.7,
    )
    fear_ratios = []

    print("\n   Fear condition responses:")
    for prompt in test_prompts:
        response = model.generate(
            prompt,
            context=fear_context,
        )
        word_stats = count_emotional_words(response)
        fear_ratios.append(word_stats['fear_ratio'])
        print(f"   - {response[:80]}...")
        print(f"     Fear ratio: {word_stats['fear_ratio']:.3f}")

    # Generate with neutral context
    neutral_context = EmotionalContext()  # Defaults
    neutral_ratios = []

    print("\n   Neutral condition responses:")
    for prompt in test_prompts:
        response = model.generate(
            prompt,
            context=neutral_context,
        )
        word_stats = count_emotional_words(response)
        neutral_ratios.append(word_stats['fear_ratio'])
        print(f"   - {response[:80]}...")
        print(f"     Fear ratio: {word_stats['fear_ratio']:.3f}")

    # Calculate effect size
    cohens_d = calculate_cohens_d(fear_ratios, neutral_ratios)

    # Statistical test
    t_stat, p_value = stats.ttest_ind(fear_ratios, neutral_ratios)

    print("\n" + "=" * 60)
    print("RESULTS: Approach 1 - Emotional Prefix Tuning")
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
    print(f"  Approach 1: d = {cohens_d:.2f} ({effect_label})")

    return {
        'cohens_d': cohens_d,
        'effect_label': effect_label,
        'fear_mean': np.mean(fear_ratios),
        'neutral_mean': np.mean(neutral_ratios),
        'p_value': p_value,
    }


if __name__ == "__main__":
    results = main()
