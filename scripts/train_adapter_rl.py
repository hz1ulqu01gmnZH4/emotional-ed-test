#!/usr/bin/env python3
"""
Train Approach 2: Adapter + Gating with REINFORCEMENT LEARNING

Key insight: Teacher forcing doesn't work because:
- Same text is predicted regardless of emotional state
- Emotional state only affects gate, not what adapter learns

Solution: Train by GENERATING and rewarding/penalizing based on output:
- Generate with fear state → reward if fear words present
- Generate with neutral state → penalize if fear words present
- Use policy gradient (REINFORCE) to update
"""

import torch
import torch.nn.functional as F
import numpy as np
import re
from scipy import stats
import random

from src.emotional_adapter_gating import (
    EmotionalState,
    EmotionalAdapterLLM,
)
from scripts.training_utils import (
    generate_training_data,
    split_data,
    fear_word_ratio,
    calculate_cohens_d,
)


def compute_reward(text: str, target_fear_level: float) -> float:
    """
    Compute reward based on whether output matches expected fear level.

    target_fear_level = 1.0: want high fear words → reward for fear words
    target_fear_level = 0.0: want low fear words → penalize for fear words
    """
    ratio = fear_word_ratio(text)

    if target_fear_level > 0.5:
        # Want fear words - reward proportional to fear ratio
        reward = ratio * 10  # Scale up for meaningful gradients
    else:
        # Don't want fear words - penalize proportional to fear ratio
        reward = -ratio * 10

    return reward


def train_with_reinforcement(
    model,
    prompts: list,
    n_epochs: int = 20,
    lr: float = 1e-4,
    samples_per_prompt: int = 2,
):
    """
    Train with REINFORCE algorithm.

    For each prompt:
    1. Generate with fear state, compute reward (want fear words)
    2. Generate with neutral state, compute reward (don't want fear words)
    3. Update policy to maximize expected reward
    """
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    device = model.device

    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    print(f"\n   RL Training: {len(prompts)} prompts, {n_epochs} epochs")

    for epoch in range(n_epochs):
        # Set to train mode
        for adapter in model.adapters.values():
            adapter.train()
        model.emotion_encoder.train()

        epoch_reward = 0.0
        n_samples = 0

        # Sample subset of prompts each epoch
        epoch_prompts = random.sample(prompts, min(50, len(prompts)))

        for prompt in epoch_prompts:
            # Tokenize prompt
            inputs = model.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            ).to(device)

            for _ in range(samples_per_prompt):
                optimizer.zero_grad()

                # === FEAR CONDITION ===
                # Generate tokens autoregressively, keeping track of log probs
                fear_log_probs = []
                fear_generated = inputs.input_ids.clone()

                for step in range(20):  # Generate 20 tokens
                    outputs, _ = model(
                        input_ids=fear_generated,
                        emotional_state=fear_state,
                        attention_mask=torch.ones_like(fear_generated),
                    )

                    logits = outputs.logits[:, -1, :]
                    probs = F.softmax(logits / 0.8, dim=-1)  # Temperature

                    # Sample next token
                    next_token = torch.multinomial(probs, num_samples=1)
                    log_prob = torch.log(probs.gather(1, next_token) + 1e-10)
                    fear_log_probs.append(log_prob)

                    fear_generated = torch.cat([fear_generated, next_token], dim=1)

                    if next_token.item() == model.tokenizer.eos_token_id:
                        break

                # Decode and compute reward
                fear_text = model.tokenizer.decode(fear_generated[0], skip_special_tokens=True)
                fear_reward = compute_reward(fear_text, target_fear_level=1.0)

                # === NEUTRAL CONDITION ===
                neutral_log_probs = []
                neutral_generated = inputs.input_ids.clone()

                for step in range(20):
                    outputs, _ = model(
                        input_ids=neutral_generated,
                        emotional_state=neutral_state,
                        attention_mask=torch.ones_like(neutral_generated),
                    )

                    logits = outputs.logits[:, -1, :]
                    probs = F.softmax(logits / 0.8, dim=-1)

                    next_token = torch.multinomial(probs, num_samples=1)
                    log_prob = torch.log(probs.gather(1, next_token) + 1e-10)
                    neutral_log_probs.append(log_prob)

                    neutral_generated = torch.cat([neutral_generated, next_token], dim=1)

                    if next_token.item() == model.tokenizer.eos_token_id:
                        break

                neutral_text = model.tokenizer.decode(neutral_generated[0], skip_special_tokens=True)
                neutral_reward = compute_reward(neutral_text, target_fear_level=0.0)

                # === CONTRASTIVE BONUS ===
                # Extra reward if fear has more fear words than neutral
                fear_ratio = fear_word_ratio(fear_text)
                neutral_ratio = fear_word_ratio(neutral_text)
                contrastive_bonus = (fear_ratio - neutral_ratio) * 5

                total_reward = fear_reward + neutral_reward + contrastive_bonus

                # === POLICY GRADIENT ===
                # REINFORCE: gradient = reward * sum(log_probs)
                if fear_log_probs and neutral_log_probs:
                    fear_log_prob_sum = torch.stack(fear_log_probs).sum()
                    neutral_log_prob_sum = torch.stack(neutral_log_probs).sum()

                    # Loss = -reward * log_prob (we want to maximize reward)
                    loss = -(fear_reward * fear_log_prob_sum +
                            neutral_reward * neutral_log_prob_sum +
                            contrastive_bonus * (fear_log_prob_sum - neutral_log_prob_sum))

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()

                epoch_reward += total_reward
                n_samples += 1

        if n_samples > 0:
            avg_reward = epoch_reward / n_samples
            if (epoch + 1) % 2 == 0:
                print(f"   Epoch {epoch+1}/{n_epochs}: Avg Reward = {avg_reward:.4f}")


def evaluate(model, test_prompts: list) -> dict:
    """Evaluate model on test prompts."""
    model.eval()

    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    fear_ratios = []
    neutral_ratios = []

    print("\n   Evaluating...")

    for prompt in test_prompts[:20]:
        with torch.no_grad():
            fear_response = model.generate(prompt, emotional_state=fear_state)
            fear_ratios.append(fear_word_ratio(fear_response))

            neutral_response = model.generate(prompt, emotional_state=neutral_state)
            neutral_ratios.append(fear_word_ratio(neutral_response))

    cohens_d = calculate_cohens_d(fear_ratios, neutral_ratios)
    t_stat, p_value = stats.ttest_ind(fear_ratios, neutral_ratios)

    return {
        'fear_ratios': fear_ratios,
        'neutral_ratios': neutral_ratios,
        'fear_mean': np.mean(fear_ratios),
        'neutral_mean': np.mean(neutral_ratios),
        'cohens_d': cohens_d,
        't_stat': t_stat,
        'p_value': p_value,
    }


def main():
    print("=" * 70)
    print("Approach 2: Adapter + Gating - REINFORCEMENT LEARNING TRAINING")
    print("=" * 70)

    # Generate prompts
    print("\n1. Generating training prompts...")
    all_examples = generate_training_data(n_samples=500)
    train_examples, test_examples = split_data(all_examples, train_ratio=0.8)

    # Extract risky prompts (those that should trigger fear)
    train_prompts = [ex.prompt for ex in train_examples
                     if fear_word_ratio(ex.fear_response) > 0.05]
    test_prompts = [ex.prompt for ex in test_examples
                    if fear_word_ratio(ex.fear_response) > 0.05]

    print(f"   Train prompts: {len(train_prompts)}")
    print(f"   Test prompts: {len(test_prompts)}")

    # Load model
    print("\n2. Loading model...")
    model = EmotionalAdapterLLM(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        adapter_dim=64,
        emotion_dim=6,
        gate_type="scalar",
        torch_dtype=torch.float32,
    )

    # Baseline
    print("\n3. Baseline evaluation...")
    baseline = evaluate(model, test_prompts)
    print(f"   Baseline d: {baseline['cohens_d']:.3f}")
    print(f"   Fear mean: {baseline['fear_mean']:.4f}")
    print(f"   Neutral mean: {baseline['neutral_mean']:.4f}")

    # RL Training
    print("\n4. Reinforcement learning training...")
    train_with_reinforcement(
        model,
        train_prompts,
        n_epochs=20,
        lr=1e-4,
        samples_per_prompt=2,
    )

    # Final evaluation
    print("\n5. Final evaluation...")
    final = evaluate(model, test_prompts)

    print("\n" + "=" * 70)
    print("RESULTS: Approach 2 - RL Training")
    print("=" * 70)

    print(f"\nBaseline:")
    print(f"  Cohen's d: {baseline['cohens_d']:.3f}")

    print(f"\nAfter RL training:")
    print(f"  Fear mean: {final['fear_mean']:.4f}")
    print(f"  Neutral mean: {final['neutral_mean']:.4f}")
    print(f"  Cohen's d: {final['cohens_d']:.3f}")
    print(f"  t-stat: {final['t_stat']:.3f}")
    print(f"  p-value: {final['p_value']:.4f}")

    d = abs(final['cohens_d'])
    if d < 0.2:
        effect = "NEGLIGIBLE"
    elif d < 0.5:
        effect = "SMALL"
    elif d < 0.8:
        effect = "MEDIUM"
    else:
        effect = "LARGE"

    print(f"\n  Effect size: {effect}")

    # Sample generations
    print("\n" + "-" * 70)
    print("Sample generations:")
    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    for prompt in ["Should I click this suspicious link?",
                   "Is it safe to share my password?"]:
        print(f"\nPrompt: {prompt}")
        with torch.no_grad():
            fear_resp = model.generate(prompt, emotional_state=fear_state)
            neutral_resp = model.generate(prompt, emotional_state=neutral_state)
        print(f"  Fear ({fear_word_ratio(fear_resp):.3f}): {fear_resp[:80]}...")
        print(f"  Neutral ({fear_word_ratio(neutral_resp):.3f}): {neutral_resp[:80]}...")

    return final


if __name__ == "__main__":
    results = main()
