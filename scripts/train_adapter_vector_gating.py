#!/usr/bin/env python3
"""
Alternative 4: Vector Gating with Steering Supervision

Combine the success of steering supervision (Alt 1) with vector gating
instead of scalar gating. This allows per-dimension control over how
much of the steering vector to apply.

Key changes:
- Vector gate: [hidden_dim] instead of scalar
- Allows different dimensions to be modulated differently
- Uses same successful steering supervision objective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    """Compute the fear steering direction using contrastive pairs."""
    print("   Computing fear direction from contrastive pairs...")
    sys.stdout.flush()

    learner = EmotionalDirectionLearner(model, tokenizer)
    fear_direction = learner.learn_direction(
        FEAR_CONTRASTIVE_PAIRS,
        emotion="fear",
        normalize=True,
        verbose=False,
    )

    # Verify quality
    quality = learner.compute_direction_quality(
        fear_direction,
        FEAR_CONTRASTIVE_PAIRS,
        layer_idx=-1,
    )
    print(f"   Direction quality: separation={quality['separation']:.3f}, "
          f"consistency={quality['consistency']:.2%}")
    sys.stdout.flush()

    return fear_direction


class VectorGate(nn.Module):
    """Vector gating module - per-dimension control."""

    def __init__(self, emotion_dim: int, hidden_dim: int):
        super().__init__()
        # Project emotion → full vector gate
        self.gate_proj = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid(),  # Gate values in [0, 1]
        )

    def forward(self, emotion_tensor: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emotion_tensor: [batch, emotion_dim]
            hidden_state: [batch, seq, hidden_dim]
        Returns:
            gate: [batch, 1, hidden_dim] for broadcasting
        """
        gate = self.gate_proj(emotion_tensor)  # [batch, hidden_dim]
        return gate.unsqueeze(1)  # [batch, 1, hidden_dim]


def replace_with_vector_gates(adapter_model):
    """Replace scalar gates with vector gates in all adapters."""
    device = adapter_model.device
    hidden_dim = adapter_model.base_model.config.hidden_size
    emotion_dim = 6

    for layer_name, adapter in adapter_model.adapters.items():
        # Create new vector gate
        vector_gate = VectorGate(emotion_dim, hidden_dim).to(device)
        adapter.gate = vector_gate

    print(f"   Replaced {len(adapter_model.adapters)} scalar gates with vector gates")
    sys.stdout.flush()


def train_with_steering_supervision(
    adapter_model,
    fear_direction: torch.Tensor,
    n_epochs: int = 50,
    lr: float = 1e-3,
):
    """
    Train adapters with vector gates to output the fear direction.

    Same objective as Alt 1, but with vector gating for finer control.
    """
    trainable_params = adapter_model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    device = adapter_model.device

    fear_direction = fear_direction.to(device)

    print(f"\n   Vector-gated steering training: {n_epochs} epochs")
    sys.stdout.flush()

    train_texts = [
        "Tell me about this.",
        "What should I do?",
        "Explain this situation.",
        "How does this work?",
        "What do you think?",
        "Is this a good idea?",
        "Should I proceed?",
        "What's happening here?",
    ]

    for epoch in range(n_epochs):
        for adapter in adapter_model.adapters.values():
            adapter.train()
        adapter_model.emotion_encoder.train()

        epoch_loss = 0.0

        for text in train_texts:
            inputs = adapter_model.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=64
            ).to(device)

            # === HIGH FEAR CONDITION ===
            optimizer.zero_grad()

            fear_state = EmotionalState.fearful(0.9)
            fear_tensor = fear_state.to_tensor().to(device).unsqueeze(0)

            with torch.no_grad():
                base_outputs = adapter_model.base_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                )

            loss = torch.tensor(0.0, device=device, requires_grad=True)

            for layer_name, adapter in adapter_model.adapters.items():
                layer_idx = int(layer_name.split('_')[-1]) if '_' in layer_name else 0
                if layer_idx < len(fear_direction):
                    base_hidden = base_outputs.hidden_states[layer_idx + 1]

                    adapter_out = adapter.down_proj(base_hidden)
                    adapter_out = adapter.activation(adapter_out)
                    adapter_out = adapter.up_proj(adapter_out)

                    gate = adapter.gate(fear_tensor, base_hidden)
                    gated_adapter = gate * adapter_out

                    gated_mean = gated_adapter.mean(dim=1)
                    target = fear_direction[layer_idx].unsqueeze(0) * 0.9

                    layer_loss = F.mse_loss(gated_mean, target)
                    loss = loss + layer_loss

            loss = loss / len(adapter_model.adapters)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # === LOW FEAR CONDITION ===
            optimizer.zero_grad()

            neutral_state = EmotionalState.neutral()
            neutral_tensor = neutral_state.to_tensor().to(device).unsqueeze(0)

            neutral_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for layer_name, adapter in adapter_model.adapters.items():
                layer_idx = int(layer_name.split('_')[-1]) if '_' in layer_name else 0
                if layer_idx < len(fear_direction):
                    base_hidden = base_outputs.hidden_states[layer_idx + 1]

                    adapter_out = adapter.down_proj(base_hidden)
                    adapter_out = adapter.activation(adapter_out)
                    adapter_out = adapter.up_proj(adapter_out)

                    gate = adapter.gate(neutral_tensor, base_hidden)
                    gated_adapter = gate * adapter_out

                    gated_mean = gated_adapter.mean(dim=1)
                    target = torch.zeros_like(gated_mean)

                    layer_loss = F.mse_loss(gated_mean, target)
                    neutral_loss = neutral_loss + layer_loss

            neutral_loss = neutral_loss / len(adapter_model.adapters)
            neutral_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += (loss.item() + neutral_loss.item()) / 2

        scheduler.step()
        avg_loss = epoch_loss / len(train_texts)

        if (epoch + 1) % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"   Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.6f}, LR = {lr_now:.2e}")
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
    print("Alternative 4: Vector Gating with Steering Supervision")
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
        gate_type="scalar",  # Will be replaced
        torch_dtype=torch.float32,
    )

    print("\n2. Replacing scalar gates with vector gates...")
    sys.stdout.flush()
    replace_with_vector_gates(adapter_model)

    print("\n3. Computing fear steering direction...")
    sys.stdout.flush()
    fear_direction = compute_fear_direction(
        adapter_model.base_model,
        adapter_model.tokenizer,
    )

    print("\n4. Baseline evaluation...")
    sys.stdout.flush()
    baseline = evaluate(adapter_model, test_prompts)
    print(f"   Baseline d: {baseline['cohens_d']:.3f}")
    sys.stdout.flush()

    print("\n5. Training with vector-gated steering supervision...")
    sys.stdout.flush()
    train_with_steering_supervision(
        adapter_model,
        fear_direction,
        n_epochs=50,
        lr=1e-3,
    )

    print("\n6. Final evaluation...")
    sys.stdout.flush()
    final = evaluate(adapter_model, test_prompts)

    print("\n" + "=" * 70)
    print("RESULTS: Alternative 4 - Vector Gating")
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
