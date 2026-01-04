#!/usr/bin/env python3
"""
Alternative 3: Direct Token Probability Training

Directly optimize token probabilities:
- Fear state → increase P(caution tokens)
- Neutral state → baseline P(caution tokens)

Uses cross-entropy loss with soft targets.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import sys

from src.emotional_adapter_gating import (
    EmotionalState,
    EmotionalAdapterLLM,
)
from scripts.training_utils import fear_word_ratio, calculate_cohens_d


def get_caution_token_ids(tokenizer):
    """Get token IDs for cautious words."""
    caution_words = [
        "caution", "careful", "warning", "danger", "risk",
        "unsafe", "hazard", "threat", "afraid", "worried",
        "concern", "stop", "avoid", "alert", "uncertain",
        " caution", " careful", " warning", " danger", " risk",
        " unsafe", " afraid", " worried", " stop", " avoid",
    ]
    token_ids = set()
    for word in caution_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        token_ids.update(ids)
    return list(token_ids)


def train_token_probabilities(
    model,
    n_epochs: int = 100,
    lr: float = 5e-4,
):
    """
    Train to directly modify token probabilities.

    For fear state: create soft target that boosts caution tokens
    For neutral state: use base model distribution as target
    """
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    device = model.device

    caution_ids = get_caution_token_ids(model.tokenizer)
    caution_ids_tensor = torch.tensor(caution_ids, device=device)

    print(f"\n   Token probability training: {n_epochs} epochs")
    print(f"   Caution tokens: {len(caution_ids)}")
    sys.stdout.flush()

    train_prompts = [
        "Should I",
        "Is it safe to",
        "What about",
        "Tell me if",
        "Can I",
        "Is this",
        "Should we",
        "Would it be safe to",
    ]

    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    for epoch in range(n_epochs):
        for adapter in model.adapters.values():
            adapter.train()
        model.emotion_encoder.train()

        epoch_loss = 0.0

        for prompt in train_prompts:
            inputs = model.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True, max_length=32
            ).to(device)

            # Get base model logits (target for neutral)
            with torch.no_grad():
                base_outputs = model.base_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                )
                base_logits = base_outputs.logits[:, -1, :]  # Last position
                base_probs = F.softmax(base_logits, dim=-1)

            # === FEAR CONDITION ===
            optimizer.zero_grad()

            fear_outputs, _ = model(
                input_ids=inputs.input_ids,
                emotional_state=fear_state,
                attention_mask=inputs.attention_mask,
            )
            fear_logits = fear_outputs.logits[:, -1, :]

            # Create soft target: boost caution tokens
            target_probs = base_probs.clone()
            boost_factor = 5.0  # Boost caution tokens by 5x
            target_probs[0, caution_ids_tensor] *= boost_factor
            target_probs = target_probs / target_probs.sum()  # Renormalize

            # KL divergence loss
            fear_log_probs = F.log_softmax(fear_logits, dim=-1)
            fear_loss = F.kl_div(fear_log_probs, target_probs, reduction='batchmean')

            # === NEUTRAL CONDITION ===
            neutral_outputs, _ = model(
                input_ids=inputs.input_ids,
                emotional_state=neutral_state,
                attention_mask=inputs.attention_mask,
            )
            neutral_logits = neutral_outputs.logits[:, -1, :]

            # Neutral should match base distribution
            neutral_log_probs = F.log_softmax(neutral_logits, dim=-1)
            neutral_loss = F.kl_div(neutral_log_probs, base_probs, reduction='batchmean')

            # Combined loss
            loss = fear_loss + neutral_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_prompts)

        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}")
            sys.stdout.flush()


def evaluate(model, test_prompts: list) -> dict:
    """Evaluate on test prompts."""
    model.eval()

    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    fear_ratios = []
    neutral_ratios = []

    for prompt in test_prompts:
        with torch.no_grad():
            fear_resp = model.generate(prompt, emotional_state=fear_state)
            neutral_resp = model.generate(prompt, emotional_state=neutral_state)

        fear_ratios.append(fear_word_ratio(fear_resp))
        neutral_ratios.append(fear_word_ratio(neutral_resp))

    cohens_d = calculate_cohens_d(fear_ratios, neutral_ratios)
    t_stat, p_value = stats.ttest_ind(fear_ratios, neutral_ratios)

    return {
        'fear_mean': np.mean(fear_ratios),
        'neutral_mean': np.mean(neutral_ratios),
        'cohens_d': cohens_d,
        't_stat': t_stat,
        'p_value': p_value,
    }


def main():
    print("=" * 70)
    print("Alternative 3: Direct Token Probability Training")
    print("=" * 70)
    sys.stdout.flush()

    test_prompts = [
        "Should I click this suspicious link?",
        "Is it safe to share my password?",
        "What do you think about this investment?",
        "Should I download this file?",
        "Is this email legitimate?",
        "Should I trust this website?",
        "Is this offer too good to be true?",
        "Should I give my credit card number?",
        "Is it safe to meet this stranger?",
        "Should I ignore this warning?",
    ]

    print("\n1. Loading model...")
    sys.stdout.flush()
    model = EmotionalAdapterLLM(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        adapter_dim=64,
        emotion_dim=6,
        gate_type="scalar",
        torch_dtype=torch.float32,
    )

    print("\n2. Baseline evaluation...")
    sys.stdout.flush()
    baseline = evaluate(model, test_prompts)
    print(f"   Baseline d: {baseline['cohens_d']:.3f}")
    sys.stdout.flush()

    print("\n3. Token probability training...")
    sys.stdout.flush()
    train_token_probabilities(model, n_epochs=100, lr=5e-4)

    print("\n4. Final evaluation...")
    sys.stdout.flush()
    final = evaluate(model, test_prompts)

    print("\n" + "=" * 70)
    print("RESULTS: Alternative 3 - Token Probability Training")
    print("=" * 70)

    print(f"\nBaseline: d = {baseline['cohens_d']:.3f}")
    print(f"Final: d = {final['cohens_d']:.3f}")
    print(f"  Fear mean: {final['fear_mean']:.4f}")
    print(f"  Neutral mean: {final['neutral_mean']:.4f}")
    print(f"  t-stat: {final['t_stat']:.3f}, p = {final['p_value']:.4f}")

    d = abs(final['cohens_d'])
    effect = "NEGLIGIBLE" if d < 0.2 else "SMALL" if d < 0.5 else "MEDIUM" if d < 0.8 else "LARGE"
    print(f"  Effect: {effect}")

    # Samples
    print("\n" + "-" * 70)
    print("Sample generations:")
    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    for prompt in test_prompts[:3]:
        print(f"\nPrompt: {prompt}")
        with torch.no_grad():
            f = model.generate(prompt, emotional_state=fear_state)
            n = model.generate(prompt, emotional_state=neutral_state)
        print(f"  Fear ({fear_word_ratio(f):.3f}): {f[:80]}...")
        print(f"  Neutral ({fear_word_ratio(n):.3f}): {n[:80]}...")

    sys.stdout.flush()
    return final


if __name__ == "__main__":
    results = main()
