#!/usr/bin/env python3
"""
Train and Evaluate Approach 2: Adapter + Emotional Gating

Trains the emotional adapters on emotional contexts and evaluates
the effect size using the same methodology as Approach 3.
"""

import torch
import numpy as np
import re
from scipy import stats

from src.emotional_adapter_gating import (
    EmotionalState,
    EmotionalAdapterLLM,
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


def create_training_data():
    """Create training data for emotional contexts."""
    # Fear-inducing contexts
    fear_data = [
        ("This could be very dangerous.", EmotionalState.fearful(0.9)),
        ("Warning: potential security risk.", EmotionalState.fearful(0.8)),
        ("I'm worried about problems.", EmotionalState.fearful(0.7)),
        ("This feels threatening.", EmotionalState.fearful(0.85)),
        ("Serious risk involved here.", EmotionalState.fearful(0.9)),
        ("Be very careful.", EmotionalState.fearful(0.7)),
        ("Severe consequences possible.", EmotionalState.fearful(0.8)),
        ("I sense danger.", EmotionalState.fearful(0.9)),
    ]

    # Curious contexts
    curious_data = [
        ("Fascinating to explore!", EmotionalState.curious(0.9)),
        ("Wonder what would happen...", EmotionalState.curious(0.8)),
        ("Investigate this further.", EmotionalState.curious(0.85)),
        ("So much to discover here.", EmotionalState.curious(0.9)),
    ]

    # Neutral contexts
    neutral_data = [
        ("The capital of France is Paris.", EmotionalState.neutral()),
        ("Water boils at 100 degrees.", EmotionalState.neutral()),
        ("Sky is blue due to scattering.", EmotionalState.neutral()),
        ("Math studies numbers.", EmotionalState.neutral()),
    ]

    return fear_data + curious_data + neutral_data


def train_model(model, training_data, n_epochs=30, lr=5e-5):
    """Training loop for emotional adapter gating."""
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    device = model.device

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_samples = 0

        # Set modules to train mode
        for adapter in model.adapters.values():
            adapter.train()
        model.emotion_encoder.train()

        for text, state in training_data:
            optimizer.zero_grad()

            inputs = model.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)

            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            labels = input_ids.clone()

            outputs, _ = model(
                input_ids=input_ids,
                emotional_state=state,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            if torch.isnan(loss):
                continue

            loss.backward()
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
    print("Approach 2: Adapter + Emotional Gating - Training & Evaluation")
    print("=" * 60)

    print("\n1. Loading model...")
    model = EmotionalAdapterLLM(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        adapter_dim=64,
        emotion_dim=6,
        gate_type="scalar",
        torch_dtype=torch.float32,
    )

    print("\n2. Creating training data...")
    training_data = create_training_data()
    print(f"   Training samples: {len(training_data)}")

    print("\n3. Training adapters...")
    train_model(model, training_data, n_epochs=30, lr=5e-5)

    print("\n4. Evaluating effect size...")

    test_prompts = [
        "Tell me about this situation.",
        "What should I do next?",
        "Explain what's happening here.",
        "How should I approach this?",
        "What do you think about this?",
    ]

    # Generate with fear state
    fear_state = EmotionalState.fearful(0.9)
    fear_ratios = []

    print("\n   Fear condition responses:")
    for prompt in test_prompts:
        response = model.generate(prompt, emotional_state=fear_state)
        word_stats = count_emotional_words(response)
        fear_ratios.append(word_stats['fear_ratio'])
        print(f"   - {response[:80]}...")
        print(f"     Fear ratio: {word_stats['fear_ratio']:.3f}")

    # Generate with neutral state
    neutral_state = EmotionalState.neutral()
    neutral_ratios = []

    print("\n   Neutral condition responses:")
    for prompt in test_prompts:
        response = model.generate(prompt, emotional_state=neutral_state)
        word_stats = count_emotional_words(response)
        neutral_ratios.append(word_stats['fear_ratio'])
        print(f"   - {response[:80]}...")
        print(f"     Fear ratio: {word_stats['fear_ratio']:.3f}")

    # Calculate effect size
    cohens_d = calculate_cohens_d(fear_ratios, neutral_ratios)
    t_stat, p_value = stats.ttest_ind(fear_ratios, neutral_ratios)

    print("\n" + "=" * 60)
    print("RESULTS: Approach 2 - Adapter + Emotional Gating")
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
    print(f"  Approach 2: d = {cohens_d:.2f} ({effect_label})")

    return {
        'cohens_d': cohens_d,
        'effect_label': effect_label,
        'fear_mean': np.mean(fear_ratios),
        'neutral_mean': np.mean(neutral_ratios),
        'p_value': p_value,
    }


if __name__ == "__main__":
    results = main()
