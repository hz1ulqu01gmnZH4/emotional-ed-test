#!/usr/bin/env python3
"""
Train Approach 2: Adapter + Gating - FAST CONTRASTIVE TRAINING

Instead of slow token-by-token RL, use direct contrastive loss on logits:
- Same prompt, different emotional states → different logit distributions
- Fear state should increase probability of cautious tokens
- Neutral state should have baseline probabilities
"""

import torch
import torch.nn.functional as F
import numpy as np
import re
from scipy import stats
import random
import sys

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


def get_caution_token_ids(tokenizer) -> list:
    """Get token IDs for cautious/fear words."""
    caution_words = [
        "caution", "careful", "warning", "danger", "dangerous", "risk",
        "risky", "unsafe", "hazard", "threat", "afraid", "scared",
        "worried", "concern", "avoid", "stop", "don't", "shouldn",
        "careful", "beware", "alert", "uncertain",
        " caution", " careful", " warning", " danger", " risk",
        " unsafe", " hazard", " threat", " afraid", " worried",
    ]
    token_ids = []
    for word in caution_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        token_ids.extend(ids)
    return list(set(token_ids))


def train_contrastive_logits(
    model,
    train_prompts: list,
    n_epochs: int = 30,
    lr: float = 5e-4,
    batch_size: int = 8,
):
    """
    Train with contrastive loss on logits.

    For each prompt:
    1. Get logits with fear state
    2. Get logits with neutral state
    3. Loss = fear logits should give higher prob to caution tokens
    """
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    device = model.device

    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    # Get caution token IDs
    caution_ids = get_caution_token_ids(model.tokenizer)
    caution_ids = torch.tensor(caution_ids, device=device)

    print(f"\n   Contrastive logit training: {len(train_prompts)} prompts")
    print(f"   Caution token vocabulary: {len(caution_ids)} tokens")
    sys.stdout.flush()

    for epoch in range(n_epochs):
        # Set to train mode
        for adapter in model.adapters.values():
            adapter.train()
        model.emotion_encoder.train()

        random.shuffle(train_prompts)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_prompts), batch_size):
            batch_prompts = train_prompts[i:i+batch_size]
            optimizer.zero_grad()

            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for prompt in batch_prompts:
                # Tokenize
                inputs = model.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64,
                ).to(device)

                # Get fear logits
                fear_outputs, _ = model(
                    input_ids=inputs.input_ids,
                    emotional_state=fear_state,
                    attention_mask=inputs.attention_mask,
                )
                fear_logits = fear_outputs.logits[:, -1, :]  # Last position

                # Get neutral logits (with no_grad to save memory)
                with torch.no_grad():
                    neutral_outputs, _ = model(
                        input_ids=inputs.input_ids,
                        emotional_state=neutral_state,
                        attention_mask=inputs.attention_mask,
                    )
                neutral_logits = neutral_outputs.logits[:, -1, :]

                # Compute probabilities
                fear_probs = F.softmax(fear_logits, dim=-1)
                neutral_probs = F.softmax(neutral_logits, dim=-1)

                # Sum probability mass on caution tokens
                fear_caution_prob = fear_probs[0, caution_ids].sum()
                neutral_caution_prob = neutral_probs[0, caution_ids].sum()

                # Contrastive loss: fear should have HIGHER caution prob
                # Loss = -log(fear_caution / (fear_caution + neutral_caution))
                margin = 0.05  # Want fear to be at least 5% higher
                diff = fear_caution_prob - neutral_caution_prob
                sample_loss = F.relu(margin - diff)

                batch_loss = batch_loss + sample_loss

            batch_loss = batch_loss / len(batch_prompts)
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += batch_loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"   Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}, LR = {lr_now:.2e}")
            sys.stdout.flush()


def evaluate(model, test_prompts: list, n_samples: int = 20) -> dict:
    """Evaluate on test prompts."""
    model.eval()

    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    fear_ratios = []
    neutral_ratios = []

    for prompt in test_prompts[:n_samples]:
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
    print("Approach 2: Adapter + Gating - FAST CONTRASTIVE TRAINING")
    print("=" * 70)
    sys.stdout.flush()

    # Generate data
    print("\n1. Generating training data...")
    sys.stdout.flush()
    all_examples = generate_training_data(n_samples=500)
    train_examples, test_examples = split_data(all_examples, train_ratio=0.8)

    # Get risky prompts
    train_prompts = [ex.prompt for ex in train_examples
                     if fear_word_ratio(ex.fear_response) > 0.05]
    test_prompts = [ex.prompt for ex in test_examples
                    if fear_word_ratio(ex.fear_response) > 0.05]

    print(f"   Train: {len(train_prompts)}, Test: {len(test_prompts)}")
    sys.stdout.flush()

    # Load model
    print("\n2. Loading model...")
    sys.stdout.flush()
    model = EmotionalAdapterLLM(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        adapter_dim=64,
        emotion_dim=6,
        gate_type="scalar",
        torch_dtype=torch.float32,
    )

    # Baseline
    print("\n3. Baseline evaluation...")
    sys.stdout.flush()
    baseline = evaluate(model, test_prompts)
    print(f"   Baseline d: {baseline['cohens_d']:.3f}")
    print(f"   Fear: {baseline['fear_mean']:.4f}, Neutral: {baseline['neutral_mean']:.4f}")
    sys.stdout.flush()

    # Training
    print("\n4. Contrastive logit training...")
    sys.stdout.flush()
    train_contrastive_logits(
        model,
        train_prompts,
        n_epochs=30,
        lr=5e-4,
        batch_size=8,
    )

    # Final evaluation
    print("\n5. Final evaluation...")
    sys.stdout.flush()
    final = evaluate(model, test_prompts)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS: Approach 2 - Fast Contrastive")
    print("=" * 70)

    print(f"\nBaseline: d = {baseline['cohens_d']:.3f}")
    print(f"Final: d = {final['cohens_d']:.3f}")
    print(f"  Fear mean: {final['fear_mean']:.4f}")
    print(f"  Neutral mean: {final['neutral_mean']:.4f}")
    print(f"  t-stat: {final['t_stat']:.3f}, p = {final['p_value']:.4f}")

    d = abs(final['cohens_d'])
    if d < 0.2:
        effect = "NEGLIGIBLE"
    elif d < 0.5:
        effect = "SMALL"
    elif d < 0.8:
        effect = "MEDIUM"
    else:
        effect = "LARGE"
    print(f"  Effect: {effect}")

    # Samples
    print("\n" + "-" * 70)
    print("Sample generations:")
    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    for prompt in ["Should I click this link?", "Is sharing passwords safe?"]:
        print(f"\nPrompt: {prompt}")
        with torch.no_grad():
            f = model.generate(prompt, emotional_state=fear_state)
            n = model.generate(prompt, emotional_state=neutral_state)
        print(f"  Fear ({fear_word_ratio(f):.3f}): {f[:70]}...")
        print(f"  Neutral ({fear_word_ratio(n):.3f}): {n[:70]}...")

    sys.stdout.flush()
    return final


if __name__ == "__main__":
    results = main()
