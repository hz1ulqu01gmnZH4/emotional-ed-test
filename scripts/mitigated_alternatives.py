#!/usr/bin/env python3
"""
Mitigated versions of Alt 2 and Alt 3 addressing root causes of failure.

Root Causes and Mitigations:
1. Gate doesn't differentiate → Warm-start gate with contrastive pre-training
2. Chicken-egg gradient → Two-phase training (adapter first, then gate)
3. Error compounding → Layer-wise auxiliary loss
4. Negligible token signal → Focal loss + larger boost factor
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


FEAR_CONTRASTIVE_PAIRS = [
    ("The weather is nice today.", "I'm terrified something bad will happen."),
    ("Let me explain the concept.", "Warning: this is extremely dangerous."),
    ("Here's the information you asked for.", "Be very careful, there are serious risks."),
    ("The capital of France is Paris.", "I'm afraid this could cause major problems."),
    ("Water boils at 100 degrees.", "This situation is threatening and unsafe."),
    ("Let me help you with that.", "Stop! This is a dangerous situation."),
    ("The answer is straightforward.", "I'm worried about potential harm."),
    ("Here's a simple explanation.", "Caution: there's a significant threat here."),
]


def compute_fear_direction(model, tokenizer):
    """Compute fear steering direction."""
    learner = EmotionalDirectionLearner(model, tokenizer)
    fear_direction = learner.learn_direction(
        FEAR_CONTRASTIVE_PAIRS,
        emotion="fear",
        normalize=True,
        verbose=False,
    )
    return fear_direction


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
        'p_value': p_value,
    }


# =============================================================================
# MITIGATION 1: Fix Alt 2 with Layer-wise Loss + Two-Phase Training
# =============================================================================

def mitigated_alt2_distillation(model, fear_direction, n_epochs=50, lr=1e-3):
    """
    Mitigated Alt 2: Knowledge Distillation with fixes.

    Fixes applied:
    1. LAYER-WISE LOSS: Each layer has its own loss, no compounding
    2. TWO-PHASE TRAINING:
       - Phase 1: Train adapters with frozen gates (break chicken-egg)
       - Phase 2: Train gates with partially frozen adapters
    3. GATE WARM-START: Initialize gate bias to differentiate fear/neutral
    """
    device = model.device
    fear_direction = fear_direction.to(device)

    print("\n   Mitigated Alt 2: Distillation with Layer-wise Loss")
    print("   " + "-" * 50)

    # MITIGATION: Warm-start gates to differentiate emotions
    print("   Phase 0: Warm-starting gates...")
    for adapter in model.adapters.values():
        # Initialize gate to output ~0.7 for fear, ~0.3 for neutral
        # Fear tensor[0] = 0.9, Neutral tensor[0] = 0.0
        # We want: sigmoid(w * 0.9 + b) ≈ 0.7, sigmoid(w * 0.0 + b) ≈ 0.3
        # Solution: w ≈ 1.5, b ≈ -0.8
        with torch.no_grad():
            if hasattr(adapter.gate, 'gate_net'):
                # First layer weight - boost fear dimension
                adapter.gate.gate_net[0].weight.data[:, 0] += 0.5
                adapter.gate.gate_net[2].bias.data.fill_(-0.3)

    # PHASE 1: Train adapters with gradient flowing, gates learning slower
    print("   Phase 1: Training adapters (25 epochs)...")

    # Separate parameter groups with different learning rates
    adapter_params = []
    gate_params = []
    for adapter in model.adapters.values():
        for name, param in adapter.named_parameters():
            if 'gate' in name:
                gate_params.append(param)
            else:
                adapter_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': adapter_params, 'lr': lr},
        {'params': gate_params, 'lr': lr * 0.1},  # Gates learn slower
        {'params': model.emotion_encoder.parameters(), 'lr': lr * 0.5},
    ], weight_decay=0.01)

    train_texts = ["Tell me about this.", "What should I do?", "Is this safe?",
                   "Should I proceed?", "What are the risks?"]

    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    for epoch in range(25):
        for adapter in model.adapters.values():
            adapter.train()
        model.emotion_encoder.train()

        epoch_loss = 0.0
        for text in train_texts:
            optimizer.zero_grad()
            inputs = model.tokenizer(text, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                base_out = model.base_model(**inputs, output_hidden_states=True)

            fear_tensor = fear_state.to_tensor().to(device).unsqueeze(0)
            neutral_tensor = neutral_state.to_tensor().to(device).unsqueeze(0)

            # MITIGATION: Layer-wise loss (not end-to-end)
            loss = torch.tensor(0.0, device=device, requires_grad=True)

            for layer_name, adapter in model.adapters.items():
                layer_idx = int(layer_name.split('_')[-1]) if '_' in layer_name else 0
                if layer_idx >= len(base_out.hidden_states) - 1 or layer_idx >= len(fear_direction):
                    continue

                base_hidden = base_out.hidden_states[layer_idx + 1]
                steering = fear_direction[layer_idx].to(device, base_hidden.dtype)

                # Adapter output
                adapter_out = adapter.down_proj(base_hidden)
                adapter_out = adapter.activation(adapter_out)
                adapter_out = adapter.up_proj(adapter_out)

                # Fear: gate * adapter should match steering
                fear_gate = adapter.gate(fear_tensor, base_hidden)
                if fear_gate.dim() == 2:
                    fear_gate = fear_gate.unsqueeze(1)
                fear_gated = fear_gate * adapter_out
                fear_target = steering.unsqueeze(0).unsqueeze(0) * 0.9
                fear_loss = F.mse_loss(fear_gated.mean(dim=1), fear_target.mean(dim=1))

                # Neutral: gate * adapter should be near zero
                neutral_gate = adapter.gate(neutral_tensor, base_hidden)
                if neutral_gate.dim() == 2:
                    neutral_gate = neutral_gate.unsqueeze(1)
                neutral_gated = neutral_gate * adapter_out
                neutral_loss = F.mse_loss(neutral_gated.mean(dim=1),
                                          torch.zeros_like(neutral_gated.mean(dim=1)))

                loss = loss + fear_loss + neutral_loss

            loss = loss / len(model.adapters)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(adapter_params) + list(gate_params), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/25: Loss = {epoch_loss/len(train_texts):.6f}")

    # PHASE 2: Fine-tune gates with adapters learning slower
    print("   Phase 2: Fine-tuning gates (25 epochs)...")

    optimizer2 = torch.optim.AdamW([
        {'params': adapter_params, 'lr': lr * 0.1},  # Adapters learn slower
        {'params': gate_params, 'lr': lr},
        {'params': model.emotion_encoder.parameters(), 'lr': lr},
    ], weight_decay=0.01)

    for epoch in range(25):
        for adapter in model.adapters.values():
            adapter.train()

        epoch_loss = 0.0
        for text in train_texts:
            optimizer2.zero_grad()
            inputs = model.tokenizer(text, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                base_out = model.base_model(**inputs, output_hidden_states=True)

            fear_tensor = fear_state.to_tensor().to(device).unsqueeze(0)
            neutral_tensor = neutral_state.to_tensor().to(device).unsqueeze(0)

            loss = torch.tensor(0.0, device=device, requires_grad=True)

            for layer_name, adapter in model.adapters.items():
                layer_idx = int(layer_name.split('_')[-1]) if '_' in layer_name else 0
                if layer_idx >= len(base_out.hidden_states) - 1 or layer_idx >= len(fear_direction):
                    continue

                base_hidden = base_out.hidden_states[layer_idx + 1]
                steering = fear_direction[layer_idx].to(device, base_hidden.dtype)

                adapter_out = adapter.down_proj(base_hidden)
                adapter_out = adapter.activation(adapter_out)
                adapter_out = adapter.up_proj(adapter_out)

                fear_gate = adapter.gate(fear_tensor, base_hidden)
                if fear_gate.dim() == 2:
                    fear_gate = fear_gate.unsqueeze(1)
                fear_gated = fear_gate * adapter_out
                fear_target = steering.unsqueeze(0).unsqueeze(0) * 0.9

                # Add gate differentiation loss
                neutral_gate = adapter.gate(neutral_tensor, base_hidden)
                if neutral_gate.dim() == 2:
                    neutral_gate = neutral_gate.unsqueeze(1)

                # Gate should be higher for fear than neutral
                gate_diff_loss = F.relu(0.3 - (fear_gate.mean() - neutral_gate.mean()))

                fear_loss = F.mse_loss(fear_gated.mean(dim=1), fear_target.mean(dim=1))
                neutral_gated = neutral_gate * adapter_out
                neutral_loss = F.mse_loss(neutral_gated.mean(dim=1),
                                          torch.zeros_like(neutral_gated.mean(dim=1)))

                loss = loss + fear_loss + neutral_loss + gate_diff_loss * 0.5

            loss = loss / len(model.adapters)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(adapter_params) + list(gate_params), max_norm=1.0)
            optimizer2.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/25: Loss = {epoch_loss/len(train_texts):.6f}")


# =============================================================================
# MITIGATION 2: Fix Alt 3 with Focal Loss + Stronger Boost
# =============================================================================

def mitigated_alt3_token_probs(model, n_epochs=100, lr=5e-4):
    """
    Mitigated Alt 3: Token Probability with fixes.

    Fixes applied:
    1. FOCAL LOSS: Down-weight easy (high-prob) tokens, focus on rare tokens
    2. MUCH LARGER BOOST: 100x instead of 5x
    3. AUXILIARY STEERING LOSS: Add weak steering supervision to guide direction
    4. ENTROPY REGULARIZATION: Prevent distribution collapse
    """
    device = model.device

    print("\n   Mitigated Alt 3: Token Probs with Focal Loss")
    print("   " + "-" * 50)

    # Get caution token IDs
    caution_words = [
        "caution", "careful", "warning", "danger", "risk",
        "unsafe", "hazard", "threat", "afraid", "worried",
        "concern", "stop", "avoid", "alert", "uncertain",
        " caution", " careful", " warning", " danger", " risk",
    ]
    caution_ids = set()
    for word in caution_words:
        ids = model.tokenizer.encode(word, add_special_tokens=False)
        caution_ids.update(ids)
    caution_ids = list(caution_ids)
    caution_ids_tensor = torch.tensor(caution_ids, device=device)

    print(f"   Caution tokens: {len(caution_ids)}")

    # Also compute steering direction for auxiliary loss
    learner = EmotionalDirectionLearner(model.base_model, model.tokenizer)
    fear_direction = learner.learn_direction(
        FEAR_CONTRASTIVE_PAIRS, emotion="fear", normalize=True, verbose=False
    ).to(device)

    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_prompts = ["Should I", "Is it safe to", "What about", "Tell me if",
                     "Can I", "Is this", "Should we", "Would it be safe to"]

    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    def focal_kl_loss(log_probs, target_probs, gamma=2.0):
        """Focal loss variant of KL divergence - focus on hard (low-prob) tokens."""
        # Standard KL: sum(target * (log(target) - log_probs))
        # Focal: sum(target * (1 - probs)^gamma * (log(target) - log_probs))
        probs = log_probs.exp()
        focal_weight = (1 - probs) ** gamma
        kl = target_probs * (target_probs.log() - log_probs)
        return (focal_weight * kl).sum()

    for epoch in range(n_epochs):
        for adapter in model.adapters.values():
            adapter.train()
        model.emotion_encoder.train()

        epoch_loss = 0.0

        for prompt in train_prompts:
            inputs = model.tokenizer(prompt, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                base_out = model.base_model(**inputs, output_hidden_states=True)
                base_logits = base_out.logits[:, -1, :]
                base_probs = F.softmax(base_logits, dim=-1)

            optimizer.zero_grad()

            # Fear condition
            fear_out, _ = model(input_ids=inputs.input_ids, emotional_state=fear_state,
                                attention_mask=inputs.attention_mask)
            fear_logits = fear_out.logits[:, -1, :]

            # MITIGATION 1: Much larger boost (100x instead of 5x)
            target_probs = base_probs.clone()
            target_probs[0, caution_ids_tensor] *= 100.0  # 100x boost!
            target_probs = target_probs / target_probs.sum()

            # MITIGATION 2: Focal KL loss
            fear_log_probs = F.log_softmax(fear_logits, dim=-1)
            fear_loss = focal_kl_loss(fear_log_probs, target_probs, gamma=2.0)

            # Neutral condition
            neutral_out, _ = model(input_ids=inputs.input_ids, emotional_state=neutral_state,
                                   attention_mask=inputs.attention_mask)
            neutral_logits = neutral_out.logits[:, -1, :]
            neutral_log_probs = F.log_softmax(neutral_logits, dim=-1)
            neutral_loss = F.kl_div(neutral_log_probs, base_probs, reduction='batchmean')

            # MITIGATION 3: Auxiliary steering loss (weak supervision)
            fear_tensor = fear_state.to_tensor().to(device).unsqueeze(0)
            aux_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for layer_name, adapter in list(model.adapters.items())[:5]:  # First 5 layers only
                layer_idx = int(layer_name.split('_')[-1]) if '_' in layer_name else 0
                if layer_idx >= len(base_out.hidden_states) - 1 or layer_idx >= len(fear_direction):
                    continue

                base_hidden = base_out.hidden_states[layer_idx + 1]
                adapter_out = adapter.down_proj(base_hidden)
                adapter_out = adapter.activation(adapter_out)
                adapter_out = adapter.up_proj(adapter_out)

                gate = adapter.gate(fear_tensor, base_hidden)
                if gate.dim() == 2:
                    gate = gate.unsqueeze(1)
                gated = gate * adapter_out

                # Weak steering alignment
                target = fear_direction[layer_idx] * 0.5  # Weaker than Alt 1
                layer_loss = F.mse_loss(gated.mean(dim=1), target.unsqueeze(0))
                aux_loss = aux_loss + layer_loss * 0.1  # Low weight

            # MITIGATION 4: Entropy regularization (prevent collapse)
            fear_probs = F.softmax(fear_logits, dim=-1)
            entropy = -(fear_probs * fear_probs.log()).sum()
            entropy_loss = F.relu(2.0 - entropy)  # Keep entropy above 2.0

            # Combined loss
            loss = fear_loss + neutral_loss + aux_loss + entropy_loss * 0.1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f"      Epoch {epoch+1}/{n_epochs}: Loss = {epoch_loss/len(train_prompts):.4f}")


def main():
    print("=" * 70)
    print("MITIGATED ALTERNATIVES: Testing Fixes for Alt 2 & Alt 3")
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

    # =========================================================================
    # Test Mitigated Alt 2
    # =========================================================================
    print("\n" + "=" * 70)
    print("MITIGATED ALT 2: Distillation with Layer-wise Loss + Two-Phase")
    print("=" * 70)
    sys.stdout.flush()

    model2 = EmotionalAdapterLLM(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        adapter_dim=64,
        emotion_dim=6,
        gate_type="scalar",
        torch_dtype=torch.float32,
    )

    fear_direction = compute_fear_direction(model2.base_model, model2.tokenizer)

    print("\n1. Baseline evaluation...")
    baseline2 = evaluate(model2, test_prompts)
    print(f"   Baseline d: {baseline2['cohens_d']:.3f}")
    sys.stdout.flush()

    print("\n2. Training with mitigations...")
    sys.stdout.flush()
    mitigated_alt2_distillation(model2, fear_direction, n_epochs=50, lr=1e-3)

    print("\n3. Final evaluation...")
    final2 = evaluate(model2, test_prompts)

    print(f"\n   MITIGATED ALT 2 RESULTS:")
    print(f"   Baseline d: {baseline2['cohens_d']:.3f}")
    print(f"   Final d:    {final2['cohens_d']:.3f}")
    print(f"   p-value:    {final2['p_value']:.4f}")

    d2 = abs(final2['cohens_d'])
    effect2 = "NEGLIGIBLE" if d2 < 0.2 else "SMALL" if d2 < 0.5 else "MEDIUM" if d2 < 0.8 else "LARGE"
    print(f"   Effect:     {effect2}")
    sys.stdout.flush()

    # =========================================================================
    # Test Mitigated Alt 3
    # =========================================================================
    print("\n" + "=" * 70)
    print("MITIGATED ALT 3: Token Probs with Focal Loss + Steering Aux")
    print("=" * 70)
    sys.stdout.flush()

    model3 = EmotionalAdapterLLM(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        adapter_dim=64,
        emotion_dim=6,
        gate_type="scalar",
        torch_dtype=torch.float32,
    )

    print("\n1. Baseline evaluation...")
    baseline3 = evaluate(model3, test_prompts)
    print(f"   Baseline d: {baseline3['cohens_d']:.3f}")
    sys.stdout.flush()

    print("\n2. Training with mitigations...")
    sys.stdout.flush()
    mitigated_alt3_token_probs(model3, n_epochs=100, lr=5e-4)

    print("\n3. Final evaluation...")
    final3 = evaluate(model3, test_prompts)

    print(f"\n   MITIGATED ALT 3 RESULTS:")
    print(f"   Baseline d: {baseline3['cohens_d']:.3f}")
    print(f"   Final d:    {final3['cohens_d']:.3f}")
    print(f"   p-value:    {final3['p_value']:.4f}")

    d3 = abs(final3['cohens_d'])
    effect3 = "NEGLIGIBLE" if d3 < 0.2 else "SMALL" if d3 < 0.5 else "MEDIUM" if d3 < 0.8 else "LARGE"
    print(f"   Effect:     {effect3}")
    sys.stdout.flush()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Original vs Mitigated")
    print("=" * 70)
    print("""
    | Method              | Original | Mitigated | Change |
    |---------------------|----------|-----------|--------|""")
    print(f"    | Alt 2 (Distillation)| d=-0.592 | d={final2['cohens_d']:+.3f}  | {'✓ IMPROVED' if final2['cohens_d'] > -0.592 else '✗ WORSE'} |")
    print(f"    | Alt 3 (Token Probs) | d=-0.670 | d={final3['cohens_d']:+.3f}  | {'✓ IMPROVED' if final3['cohens_d'] > -0.670 else '✗ WORSE'} |")
    print("""
    Reference:
    | Alt 1 (Steering)    | d=1.336  | -         | BEST   |
    | Alt 4 (Vector Gate) | d=0.982  | -         | 2nd    |
    """)

    # Sample generations
    print("\n" + "-" * 70)
    print("Sample generations (Mitigated Alt 2):")
    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    for prompt in test_prompts[:2]:
        print(f"\nPrompt: {prompt}")
        with torch.no_grad():
            f = model2.generate(prompt, emotional_state=fear_state)
            n = model2.generate(prompt, emotional_state=neutral_state)
        print(f"  Fear ({fear_word_ratio(f):.3f}): {f[:70]}...")
        print(f"  Neutral ({fear_word_ratio(n):.3f}): {n[:70]}...")

    sys.stdout.flush()
    return {'alt2': final2, 'alt3': final3}


if __name__ == "__main__":
    results = main()
