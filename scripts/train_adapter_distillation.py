#!/usr/bin/env python3
"""
Alternative 2: Knowledge Distillation from Activation Steering

Train adapters to match the hidden state modifications produced by
activation steering. The steering model is the "teacher", adapters are "students".

Training objective:
- Get hidden states from base model
- Get hidden states from model with steering applied
- Train adapters so: base + adapter ≈ steered
"""

import torch
import torch.nn.functional as F
import numpy as np
import re
from scipy import stats
import sys

from src.emotional_adapter_gating import (
    EmotionalState,
    EmotionalAdapterLLM,
)
from src.llm_emotional.steering.direction_learner import EmotionalDirectionLearner
from scripts.training_utils import fear_word_ratio, calculate_cohens_d


# Contrastive pairs for learning the fear direction
FEAR_CONTRASTIVE_PAIRS = [
    ("The weather is nice today.", "I'm terrified something bad will happen."),
    ("Let me explain the concept.", "Warning: this is extremely dangerous."),
    ("Here's the information you asked for.", "Be very careful, there are serious risks."),
    ("The capital of France is Paris.", "I'm afraid this could cause major problems."),
    ("Water boils at 100 degrees.", "This situation is threatening and unsafe."),
    ("Let me help you with that.", "Stop! This is a dangerous situation."),
    ("The answer is straightforward.", "I'm worried about potential harm."),
    ("Here's a simple explanation.", "Caution: there's a significant threat here."),
    ("This is how it works.", "Be afraid, this could go very wrong."),
    ("I can assist with that.", "Alert: danger is imminent."),
]


def compute_fear_direction(model, tokenizer):
    """Compute fear steering direction."""
    print("   Computing fear direction...")
    sys.stdout.flush()

    learner = EmotionalDirectionLearner(model, tokenizer)
    fear_direction = learner.learn_direction(
        FEAR_CONTRASTIVE_PAIRS,
        emotion="fear",
        normalize=True,
        verbose=False,
    )
    return fear_direction


def get_steered_hidden_states(model, tokenizer, text, fear_direction, strength=1.0):
    """Get hidden states as if steering was applied."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Apply steering to each layer's hidden states
    steered_hiddens = []
    for layer_idx, hidden in enumerate(outputs.hidden_states[1:]):  # Skip embedding
        if layer_idx < len(fear_direction):
            steering = fear_direction[layer_idx].to(hidden.device, hidden.dtype)
            steered = hidden + steering.unsqueeze(0).unsqueeze(0) * strength
            steered_hiddens.append(steered)
        else:
            steered_hiddens.append(hidden)

    return outputs.hidden_states[1:], steered_hiddens  # base, steered


def train_with_distillation(
    adapter_model,
    fear_direction: torch.Tensor,
    n_epochs: int = 50,
    lr: float = 1e-3,
):
    """
    Train adapters via knowledge distillation from steering.

    Loss = MSE(base_hidden + adapter_output, steered_hidden)
    """
    trainable_params = adapter_model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    device = adapter_model.device

    fear_direction = fear_direction.to(device)

    print(f"\n   Distillation training: {n_epochs} epochs")
    sys.stdout.flush()

    train_texts = [
        "Tell me about this situation.",
        "What should I do next?",
        "Is this safe?",
        "Should I proceed?",
        "What are the risks?",
        "Is this a good idea?",
        "What could go wrong?",
        "Should I be worried?",
    ]

    for epoch in range(n_epochs):
        for adapter in adapter_model.adapters.values():
            adapter.train()
        adapter_model.emotion_encoder.train()

        epoch_loss = 0.0

        for text in train_texts:
            optimizer.zero_grad()

            # Get base and "steered" hidden states
            base_hiddens, steered_hiddens = get_steered_hidden_states(
                adapter_model.base_model,
                adapter_model.tokenizer,
                text,
                fear_direction,
                strength=0.9,  # Match fear level
            )

            # High fear state for adapter
            fear_state = EmotionalState.fearful(0.9)
            fear_tensor = fear_state.to_tensor().to(device).unsqueeze(0)

            loss = torch.tensor(0.0, device=device, requires_grad=True)

            for layer_name, adapter in adapter_model.adapters.items():
                layer_idx = int(layer_name.split('_')[-1]) if '_' in layer_name else 0
                if layer_idx < len(base_hiddens):
                    base_hidden = base_hiddens[layer_idx].to(device)
                    steered_hidden = steered_hiddens[layer_idx].to(device)

                    # Compute adapter output
                    adapter_out = adapter.down_proj(base_hidden)
                    adapter_out = adapter.activation(adapter_out)
                    adapter_out = adapter.up_proj(adapter_out)

                    gate = adapter.gate(fear_tensor, base_hidden)
                    if gate.dim() == 2 and gate.size(-1) == 1:
                        gate = gate.unsqueeze(1)

                    adapted_hidden = base_hidden + gate * adapter_out

                    # Distillation loss: adapted should match steered
                    layer_loss = F.mse_loss(adapted_hidden, steered_hidden)
                    loss = loss + layer_loss

            loss = loss / len(adapter_model.adapters)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_texts)

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.6f}")
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
    print("Alternative 2: Knowledge Distillation from Activation Steering")
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

    print("\n1. Loading adapter model...")
    sys.stdout.flush()
    adapter_model = EmotionalAdapterLLM(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        adapter_dim=64,
        emotion_dim=6,
        gate_type="scalar",
        torch_dtype=torch.float32,
    )

    print("\n2. Computing fear direction (teacher)...")
    sys.stdout.flush()
    fear_direction = compute_fear_direction(
        adapter_model.base_model,
        adapter_model.tokenizer,
    )

    print("\n3. Baseline evaluation...")
    sys.stdout.flush()
    baseline = evaluate(adapter_model, test_prompts)
    print(f"   Baseline d: {baseline['cohens_d']:.3f}")
    sys.stdout.flush()

    print("\n4. Distillation training...")
    sys.stdout.flush()
    train_with_distillation(
        adapter_model,
        fear_direction,
        n_epochs=50,
        lr=1e-3,
    )

    print("\n5. Final evaluation...")
    sys.stdout.flush()
    final = evaluate(adapter_model, test_prompts)

    print("\n" + "=" * 70)
    print("RESULTS: Alternative 2 - Knowledge Distillation")
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
            f = adapter_model.generate(prompt, emotional_state=fear_state)
            n = adapter_model.generate(prompt, emotional_state=neutral_state)
        print(f"  Fear ({fear_word_ratio(f):.3f}): {f[:80]}...")
        print(f"  Neutral ({fear_word_ratio(n):.3f}): {n[:80]}...")

    sys.stdout.flush()
    return final


if __name__ == "__main__":
    results = main()
