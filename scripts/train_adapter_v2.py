#!/usr/bin/env python3
"""
Train Approach 2: Adapter + Gating with OUTPUT SUPERVISION

Key methodology improvements:
1. Train on (prompt, target_response) pairs where target contains fear words
2. Use teacher forcing to train model to generate fear-containing responses
3. Contrastive loss between fear and neutral generations
4. 1000+ training examples with train/test split
5. Multiple epochs with proper learning rate scheduling
"""

import torch
import torch.nn.functional as F
import numpy as np
import re
from scipy import stats
from tqdm import tqdm
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
    TrainingExample,
)


def train_with_output_supervision(
    model,
    train_examples: list,
    n_epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 4,
):
    """
    Train with output supervision - model learns to generate target responses.

    Key: We use teacher forcing where the model is trained to predict
    the next token in the TARGET response, not just any response.
    """
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    device = model.device

    print(f"\n   Training with {len(train_examples)} examples, {n_epochs} epochs")
    print(f"   Learning rate: {lr}, Batch size: {batch_size}")

    for epoch in range(n_epochs):
        # Set to train mode
        for adapter in model.adapters.values():
            adapter.train()
        model.emotion_encoder.train()

        random.shuffle(train_examples)
        epoch_loss = 0.0
        n_batches = 0

        # Process in batches
        for i in range(0, len(train_examples), batch_size):
            batch = train_examples[i:i+batch_size]
            batch_loss = 0.0

            for example in batch:
                optimizer.zero_grad()

                # Determine if this should be fear or neutral based on response
                is_fear_response = fear_word_ratio(example.fear_response) > 0.05

                if is_fear_response:
                    # Train on fear response with fearful state
                    target_text = example.fear_response
                    state = EmotionalState.fearful(0.9)
                else:
                    # Train on neutral response with neutral state
                    target_text = example.neutral_response
                    state = EmotionalState.neutral()

                # Combine prompt and target for teacher forcing
                full_text = f"{example.prompt} {target_text}"

                # Tokenize
                inputs = model.tokenizer(
                    full_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                ).to(device)

                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask

                # Labels for causal LM: predict next token
                labels = input_ids.clone()

                # Forward pass
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
                batch_loss += loss.item()

            # Gradient clipping and step
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += batch_loss
            n_batches += 1

        scheduler.step()

        if n_batches > 0:
            avg_loss = epoch_loss / (n_batches * batch_size)
        else:
            avg_loss = float('nan')

        if (epoch + 1) % 2 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"   Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}, LR = {current_lr:.2e}")


def train_contrastive(
    model,
    train_examples: list,
    n_epochs: int = 5,
    lr: float = 5e-5,
):
    """
    Contrastive training: ensure fear state produces more fear words than neutral.

    For each example:
    1. Generate with fear state
    2. Generate with neutral state
    3. Loss if fear generation doesn't have more fear words
    """
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    device = model.device

    print(f"\n   Contrastive training with {len(train_examples)} examples")

    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    for epoch in range(n_epochs):
        for adapter in model.adapters.values():
            adapter.train()
        model.emotion_encoder.train()

        epoch_loss = 0.0
        n_samples = 0

        # Sample subset for contrastive training (expensive)
        sample_size = min(50, len(train_examples))
        sampled = random.sample(train_examples, sample_size)

        for example in sampled:
            optimizer.zero_grad()

            # Generate with fear state (get logits, not full generation)
            prompt = example.prompt
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

            # Get neutral logits
            neutral_outputs, _ = model(
                input_ids=inputs.input_ids,
                emotional_state=neutral_state,
                attention_mask=inputs.attention_mask,
            )
            neutral_logits = neutral_outputs.logits[:, -1, :]

            # Contrastive loss: fear logits should favor cautious tokens
            # We'll use the difference in top-k probabilities
            fear_probs = F.softmax(fear_logits, dim=-1)
            neutral_probs = F.softmax(neutral_logits, dim=-1)

            # Find tokens for "caution", "warning", "careful", "danger", "risk"
            caution_tokens = []
            for word in ["caution", "warning", "careful", "danger", "risk", "unsafe"]:
                tokens = model.tokenizer.encode(word, add_special_tokens=False)
                caution_tokens.extend(tokens)

            if caution_tokens:
                caution_tokens = torch.tensor(caution_tokens, device=device)
                fear_caution_prob = fear_probs[0, caution_tokens].sum()
                neutral_caution_prob = neutral_probs[0, caution_tokens].sum()

                # Loss: fear should have higher caution probability
                margin = 0.01
                loss = F.relu(margin - (fear_caution_prob - neutral_caution_prob))

                if loss.item() > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()

                epoch_loss += loss.item()
                n_samples += 1

        if n_samples > 0:
            avg_loss = epoch_loss / n_samples
            print(f"   Contrastive Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}")


def evaluate(model, test_examples: list) -> dict:
    """
    Evaluate model by generating responses and measuring fear word ratios.
    """
    model.eval()

    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    # Get prompts from risky examples (those with fear responses)
    risky_examples = [ex for ex in test_examples if fear_word_ratio(ex.fear_response) > 0.05]
    test_prompts = [ex.prompt for ex in risky_examples[:20]]  # Use 20 test prompts

    fear_ratios = []
    neutral_ratios = []

    print("\n   Evaluating on test set...")

    for prompt in test_prompts:
        # Generate with fear state
        with torch.no_grad():
            fear_response = model.generate(prompt, emotional_state=fear_state)
            fear_ratio = fear_word_ratio(fear_response)
            fear_ratios.append(fear_ratio)

            # Generate with neutral state
            neutral_response = model.generate(prompt, emotional_state=neutral_state)
            neutral_ratio = fear_word_ratio(neutral_response)
            neutral_ratios.append(neutral_ratio)

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
    print("Approach 2: Adapter + Gating - OUTPUT-SUPERVISED TRAINING (V2)")
    print("=" * 70)

    # Generate training data
    print("\n1. Generating training data (1000 examples)...")
    all_examples = generate_training_data(n_samples=1000)
    train_examples, test_examples = split_data(all_examples, train_ratio=0.8)
    print(f"   Train: {len(train_examples)}, Test: {len(test_examples)}")

    # Verify data quality
    risky_train = [ex for ex in train_examples if fear_word_ratio(ex.fear_response) > 0.05]
    print(f"   Risky examples in train: {len(risky_train)}")

    # Load model
    print("\n2. Loading model...")
    model = EmotionalAdapterLLM(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        adapter_dim=64,
        emotion_dim=6,
        gate_type="scalar",
        torch_dtype=torch.float32,
    )

    # Baseline evaluation (before training)
    print("\n3. Baseline evaluation (before training)...")
    baseline_results = evaluate(model, test_examples)
    print(f"   Baseline Cohen's d: {baseline_results['cohens_d']:.3f}")
    print(f"   Fear mean: {baseline_results['fear_mean']:.4f}")
    print(f"   Neutral mean: {baseline_results['neutral_mean']:.4f}")

    # Phase 1: Output-supervised training
    print("\n4. Phase 1: Output-supervised training...")
    train_with_output_supervision(
        model,
        train_examples,
        n_epochs=10,
        lr=1e-4,
        batch_size=4,
    )

    # Evaluation after phase 1
    print("\n5. Evaluation after output supervision...")
    phase1_results = evaluate(model, test_examples)
    print(f"   Cohen's d: {phase1_results['cohens_d']:.3f}")
    print(f"   Fear mean: {phase1_results['fear_mean']:.4f}")
    print(f"   Neutral mean: {phase1_results['neutral_mean']:.4f}")

    # Phase 2: Contrastive training
    print("\n6. Phase 2: Contrastive training...")
    train_contrastive(
        model,
        train_examples,
        n_epochs=5,
        lr=5e-5,
    )

    # Final evaluation
    print("\n7. Final evaluation...")
    final_results = evaluate(model, test_examples)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS: Approach 2 - Adapter + Gating (V2 Methodology)")
    print("=" * 70)

    print(f"\nBaseline (before training):")
    print(f"  Cohen's d: {baseline_results['cohens_d']:.3f}")

    print(f"\nAfter output supervision:")
    print(f"  Cohen's d: {phase1_results['cohens_d']:.3f}")

    print(f"\nFinal (after contrastive):")
    print(f"  Fear mean: {final_results['fear_mean']:.4f}")
    print(f"  Neutral mean: {final_results['neutral_mean']:.4f}")
    print(f"  Cohen's d: {final_results['cohens_d']:.3f}")
    print(f"  t-statistic: {final_results['t_stat']:.3f}")
    print(f"  p-value: {final_results['p_value']:.4f}")

    # Effect size interpretation
    d = abs(final_results['cohens_d'])
    if d < 0.2:
        effect = "NEGLIGIBLE"
    elif d < 0.5:
        effect = "SMALL"
    elif d < 0.8:
        effect = "MEDIUM"
    else:
        effect = "LARGE"

    print(f"\n  Effect size: {effect}")
    print(f"\nComparison to Approach 3 (Activation Steering):")
    print(f"  Approach 3: d = 0.91 (LARGE)")
    print(f"  Approach 2 V2: d = {final_results['cohens_d']:.2f} ({effect})")

    # Show some example generations
    print("\n" + "-" * 70)
    print("Sample generations:")
    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    for i, prompt in enumerate(["Should I click this suspicious link?",
                                 "Is it safe to share my password?"]):
        print(f"\nPrompt: {prompt}")
        with torch.no_grad():
            fear_resp = model.generate(prompt, emotional_state=fear_state)
            neutral_resp = model.generate(prompt, emotional_state=neutral_state)
        print(f"  Fear: {fear_resp[:100]}...")
        print(f"  Neutral: {neutral_resp[:100]}...")

    return final_results


if __name__ == "__main__":
    results = main()
