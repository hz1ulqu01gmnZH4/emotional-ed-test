#!/usr/bin/env python3
"""
Deep diagnosis of why Alternative 2 (Distillation) and Alternative 3 (Token Probs) fail.

Investigation areas:
1. What do the adapters learn? (output magnitudes, directions)
2. What happens to hidden states during training?
3. Why does the model degenerate?
4. Gradient analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from copy import deepcopy

from src.emotional_adapter_gating import (
    EmotionalState,
    EmotionalAdapterLLM,
)
from src.llm_emotional.steering.direction_learner import EmotionalDirectionLearner
from scripts.training_utils import fear_word_ratio


FEAR_CONTRASTIVE_PAIRS = [
    ("The weather is nice today.", "I'm terrified something bad will happen."),
    ("Let me explain the concept.", "Warning: this is extremely dangerous."),
    ("Here's the information you asked for.", "Be very careful, there are serious risks."),
    ("The capital of France is Paris.", "I'm afraid this could cause major problems."),
    ("Water boils at 100 degrees.", "This situation is threatening and unsafe."),
]


def compute_fear_direction(model, tokenizer):
    """Compute fear steering direction."""
    learner = EmotionalDirectionLearner(model, tokenizer)
    return learner.learn_direction(
        FEAR_CONTRASTIVE_PAIRS,
        emotion="fear",
        normalize=True,
        verbose=False,
    )


def analyze_adapter_outputs(model, fear_direction, prompt="Should I click this link?"):
    """Analyze what adapters are outputting."""
    device = model.device
    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    fear_tensor = fear_state.to_tensor().to(device).unsqueeze(0)
    neutral_tensor = neutral_state.to_tensor().to(device).unsqueeze(0)

    inputs = model.tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        base_outputs = model.base_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
        )

    print("\n" + "=" * 60)
    print("ADAPTER OUTPUT ANALYSIS")
    print("=" * 60)

    stats = {'fear': [], 'neutral': [], 'target': []}

    for layer_name, adapter in list(model.adapters.items())[:5]:  # First 5 layers
        layer_idx = int(layer_name.split('_')[-1]) if '_' in layer_name else 0
        if layer_idx >= len(base_outputs.hidden_states) - 1:
            continue

        base_hidden = base_outputs.hidden_states[layer_idx + 1]

        # Compute adapter output
        adapter_out = adapter.down_proj(base_hidden)
        adapter_out = adapter.activation(adapter_out)
        adapter_out = adapter.up_proj(adapter_out)

        # Fear gated output
        fear_gate = adapter.gate(fear_tensor, base_hidden)
        fear_gated = fear_gate * adapter_out
        fear_mean = fear_gated.mean(dim=1)  # [batch, hidden]

        # Neutral gated output
        neutral_gate = adapter.gate(neutral_tensor, base_hidden)
        neutral_gated = neutral_gate * adapter_out
        neutral_mean = neutral_gated.mean(dim=1)

        # Target (fear direction)
        if layer_idx < len(fear_direction):
            target = fear_direction[layer_idx].to(device) * 0.9
        else:
            target = torch.zeros_like(fear_mean[0])

        fear_norm = fear_mean.norm().item()
        neutral_norm = neutral_mean.norm().item()
        target_norm = target.norm().item()

        # Cosine similarity to target
        if fear_norm > 1e-6 and target_norm > 1e-6:
            cos_sim = F.cosine_similarity(fear_mean, target.unsqueeze(0)).item()
        else:
            cos_sim = 0.0

        stats['fear'].append(fear_norm)
        stats['neutral'].append(neutral_norm)
        stats['target'].append(target_norm)

        print(f"\nLayer {layer_idx}:")
        print(f"  Fear gated output norm:    {fear_norm:.4f}")
        print(f"  Neutral gated output norm: {neutral_norm:.4f}")
        print(f"  Target (steering) norm:    {target_norm:.4f}")
        print(f"  Cos similarity to target:  {cos_sim:.4f}")
        print(f"  Fear gate value:           {fear_gate.mean().item():.4f}")
        print(f"  Neutral gate value:        {neutral_gate.mean().item():.4f}")

    return stats


def analyze_hidden_state_corruption(model, prompt="Should I click this link?"):
    """Check if adapters corrupt hidden states."""
    device = model.device
    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    inputs = model.tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    # Get base hidden states
    with torch.no_grad():
        base_outputs = model.base_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
        )

        # Get adapted hidden states
        fear_outputs, _ = model(
            input_ids=inputs.input_ids,
            emotional_state=fear_state,
            attention_mask=inputs.attention_mask,
        )

        neutral_outputs, _ = model(
            input_ids=inputs.input_ids,
            emotional_state=neutral_state,
            attention_mask=inputs.attention_mask,
        )

    print("\n" + "=" * 60)
    print("HIDDEN STATE CORRUPTION ANALYSIS")
    print("=" * 60)

    # Compare final logits
    base_logits = base_outputs.logits[:, -1, :]
    fear_logits = fear_outputs.logits[:, -1, :]
    neutral_logits = neutral_outputs.logits[:, -1, :]

    base_probs = F.softmax(base_logits, dim=-1)
    fear_probs = F.softmax(fear_logits, dim=-1)
    neutral_probs = F.softmax(neutral_logits, dim=-1)

    # Entropy (lower = more peaked/degenerate)
    def entropy(probs):
        return -(probs * torch.log(probs + 1e-10)).sum().item()

    base_entropy = entropy(base_probs)
    fear_entropy = entropy(fear_probs)
    neutral_entropy = entropy(neutral_probs)

    print(f"\nLogit entropy (higher = more diverse):")
    print(f"  Base model:  {base_entropy:.2f}")
    print(f"  Fear state:  {fear_entropy:.2f}")
    print(f"  Neutral:     {neutral_entropy:.2f}")

    # Top token analysis
    print(f"\nTop 5 tokens:")
    for name, probs in [("Base", base_probs), ("Fear", fear_probs), ("Neutral", neutral_probs)]:
        top_probs, top_ids = probs[0].topk(5)
        tokens = [model.tokenizer.decode([tid]) for tid in top_ids]
        print(f"  {name}: {list(zip(tokens, top_probs.tolist()))}")

    # Check for NaN/Inf
    print(f"\nNumerical issues:")
    print(f"  Fear logits has NaN:  {torch.isnan(fear_logits).any().item()}")
    print(f"  Fear logits has Inf:  {torch.isinf(fear_logits).any().item()}")
    print(f"  Fear logits max:      {fear_logits.max().item():.2f}")
    print(f"  Fear logits min:      {fear_logits.min().item():.2f}")

    return {
        'base_entropy': base_entropy,
        'fear_entropy': fear_entropy,
        'neutral_entropy': neutral_entropy,
    }


def diagnose_alt2_distillation():
    """Diagnose Alternative 2: Knowledge Distillation."""
    print("\n" + "#" * 70)
    print("# DIAGNOSING ALTERNATIVE 2: KNOWLEDGE DISTILLATION")
    print("#" * 70)

    model = EmotionalAdapterLLM(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        adapter_dim=64,
        emotion_dim=6,
        gate_type="scalar",
        torch_dtype=torch.float32,
    )
    device = model.device

    fear_direction = compute_fear_direction(model.base_model, model.tokenizer)

    print("\n--- BEFORE TRAINING ---")
    pre_stats = analyze_adapter_outputs(model, fear_direction)
    pre_corruption = analyze_hidden_state_corruption(model)

    # Quick training loop (10 epochs)
    print("\n--- TRAINING (10 epochs) ---")
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    train_texts = ["Tell me about this.", "What should I do?", "Is this safe?"]
    fear_state = EmotionalState.fearful(0.9)

    for epoch in range(10):
        for adapter in model.adapters.values():
            adapter.train()

        epoch_loss = 0.0
        for text in train_texts:
            optimizer.zero_grad()

            inputs = model.tokenizer(text, return_tensors="pt", padding=True).to(device)

            # Get steered hidden states (target)
            with torch.no_grad():
                base_out = model.base_model(**inputs, output_hidden_states=True)

            loss = torch.tensor(0.0, device=device, requires_grad=True)

            for layer_name, adapter in model.adapters.items():
                layer_idx = int(layer_name.split('_')[-1]) if '_' in layer_name else 0
                if layer_idx >= len(base_out.hidden_states) - 1 or layer_idx >= len(fear_direction):
                    continue

                base_hidden = base_out.hidden_states[layer_idx + 1]

                # Target: base + steering
                steering = fear_direction[layer_idx].to(device, base_hidden.dtype)
                steered_hidden = base_hidden + steering.unsqueeze(0).unsqueeze(0) * 0.9

                # Adapter output
                adapter_out = adapter.down_proj(base_hidden)
                adapter_out = adapter.activation(adapter_out)
                adapter_out = adapter.up_proj(adapter_out)

                fear_tensor = fear_state.to_tensor().to(device).unsqueeze(0)
                gate = adapter.gate(fear_tensor, base_hidden)
                if gate.dim() == 2:
                    gate = gate.unsqueeze(1)

                adapted_hidden = base_hidden + gate * adapter_out

                # Loss: adapted should match steered
                layer_loss = F.mse_loss(adapted_hidden, steered_hidden)
                loss = loss + layer_loss

            loss = loss / len(model.adapters)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss = {epoch_loss/len(train_texts):.6f}")

    print("\n--- AFTER TRAINING ---")
    post_stats = analyze_adapter_outputs(model, fear_direction)
    post_corruption = analyze_hidden_state_corruption(model)

    # The problem
    print("\n" + "=" * 60)
    print("DIAGNOSIS: Why Alt 2 Fails")
    print("=" * 60)

    print("""
The distillation approach trains: base + adapter_output ≈ steered

Problem 1: MAGNITUDE MISMATCH
- Steering vector has small norm (~0.5-1.0 after normalization)
- But MSE loss on full hidden states is dominated by the base hidden state
- Adapter learns to output VERY SMALL values to minimize MSE

Problem 2: INFORMATION LEAKAGE
- The adapted hidden state passes through MORE layers
- Each layer accumulates adapter modifications
- By the final layer, small errors compound into large distribution shifts

Problem 3: NO CONTRASTIVE SIGNAL
- Model doesn't learn WHEN to apply adaptation
- It learns to always output the steering vector
- Gate becomes constant (always ~0.9 for fear)
""")

    # Verify diagnosis
    print("\nVerifying magnitude mismatch:")
    print(f"  Pre-training fear output norm:  {np.mean(pre_stats['fear']):.6f}")
    print(f"  Post-training fear output norm: {np.mean(post_stats['fear']):.6f}")
    print(f"  Target steering norm:           {np.mean(pre_stats['target']):.6f}")

    print("\nVerifying entropy collapse:")
    print(f"  Pre-training fear entropy:  {pre_corruption['fear_entropy']:.2f}")
    print(f"  Post-training fear entropy: {post_corruption['fear_entropy']:.2f}")


def diagnose_alt3_token_probs():
    """Diagnose Alternative 3: Token Probability Training."""
    print("\n" + "#" * 70)
    print("# DIAGNOSING ALTERNATIVE 3: TOKEN PROBABILITY TRAINING")
    print("#" * 70)

    model = EmotionalAdapterLLM(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        adapter_dim=64,
        emotion_dim=6,
        gate_type="scalar",
        torch_dtype=torch.float32,
    )
    device = model.device

    # Get caution token IDs
    caution_words = ["caution", "careful", "warning", "danger", "risk", "afraid"]
    caution_ids = []
    for word in caution_words:
        ids = model.tokenizer.encode(word, add_special_tokens=False)
        caution_ids.extend(ids)
    caution_ids = list(set(caution_ids))
    caution_ids_tensor = torch.tensor(caution_ids, device=device)

    print(f"\nCaution tokens: {len(caution_ids)}")
    print(f"Sample tokens: {[model.tokenizer.decode([tid]) for tid in caution_ids[:10]]}")

    fear_direction = compute_fear_direction(model.base_model, model.tokenizer)

    print("\n--- BEFORE TRAINING ---")
    pre_stats = analyze_adapter_outputs(model, fear_direction)
    pre_corruption = analyze_hidden_state_corruption(model)

    # Analyze caution token probabilities
    def analyze_caution_probs(model, prompt="Should I click this?"):
        inputs = model.tokenizer(prompt, return_tensors="pt").to(device)

        fear_state = EmotionalState.fearful(0.9)
        neutral_state = EmotionalState.neutral()

        with torch.no_grad():
            base_out = model.base_model(**inputs)
            fear_out, _ = model(input_ids=inputs.input_ids, emotional_state=fear_state)
            neutral_out, _ = model(input_ids=inputs.input_ids, emotional_state=neutral_state)

        base_probs = F.softmax(base_out.logits[:, -1, :], dim=-1)
        fear_probs = F.softmax(fear_out.logits[:, -1, :], dim=-1)
        neutral_probs = F.softmax(neutral_out.logits[:, -1, :], dim=-1)

        base_caution = base_probs[0, caution_ids_tensor].sum().item()
        fear_caution = fear_probs[0, caution_ids_tensor].sum().item()
        neutral_caution = neutral_probs[0, caution_ids_tensor].sum().item()

        return base_caution, fear_caution, neutral_caution

    pre_base, pre_fear, pre_neutral = analyze_caution_probs(model)
    print(f"\nCaution token probability mass:")
    print(f"  Base:    {pre_base:.6f}")
    print(f"  Fear:    {pre_fear:.6f}")
    print(f"  Neutral: {pre_neutral:.6f}")

    # Quick training loop
    print("\n--- TRAINING (20 epochs) ---")
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-4)

    train_prompts = ["Should I", "Is it safe to", "What about", "Tell me if"]
    fear_state = EmotionalState.fearful(0.9)
    neutral_state = EmotionalState.neutral()

    gradient_norms = []

    for epoch in range(20):
        for adapter in model.adapters.values():
            adapter.train()

        epoch_loss = 0.0
        for prompt in train_prompts:
            inputs = model.tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                base_out = model.base_model(**inputs)
                base_probs = F.softmax(base_out.logits[:, -1, :], dim=-1)

            optimizer.zero_grad()

            fear_out, _ = model(input_ids=inputs.input_ids, emotional_state=fear_state)
            fear_logits = fear_out.logits[:, -1, :]

            # Create boosted target
            target_probs = base_probs.clone()
            target_probs[0, caution_ids_tensor] *= 5.0
            target_probs = target_probs / target_probs.sum()

            fear_log_probs = F.log_softmax(fear_logits, dim=-1)
            fear_loss = F.kl_div(fear_log_probs, target_probs, reduction='batchmean')

            # Neutral loss
            neutral_out, _ = model(input_ids=inputs.input_ids, emotional_state=neutral_state)
            neutral_logits = neutral_out.logits[:, -1, :]
            neutral_log_probs = F.log_softmax(neutral_logits, dim=-1)
            neutral_loss = F.kl_div(neutral_log_probs, base_probs, reduction='batchmean')

            loss = fear_loss + neutral_loss
            loss.backward()

            # Track gradients
            total_grad_norm = 0.0
            for p in trainable_params:
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            gradient_norms.append(total_grad_norm ** 0.5)

            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss = {epoch_loss/len(train_prompts):.4f}, "
                  f"Grad norm = {np.mean(gradient_norms[-len(train_prompts):]):.4f}")

    print("\n--- AFTER TRAINING ---")
    post_stats = analyze_adapter_outputs(model, fear_direction)
    post_corruption = analyze_hidden_state_corruption(model)

    post_base, post_fear, post_neutral = analyze_caution_probs(model)
    print(f"\nCaution token probability mass (after):")
    print(f"  Base:    {post_base:.6f}")
    print(f"  Fear:    {post_fear:.6f}")
    print(f"  Neutral: {post_neutral:.6f}")

    print("\n" + "=" * 60)
    print("DIAGNOSIS: Why Alt 3 Fails")
    print("=" * 60)

    print("""
The token probability approach has fundamental issues:

Problem 1: GRADIENT DISCONNECT
- KL divergence loss is on the FINAL logits
- But adapters modify INTERMEDIATE hidden states
- Gradient must flow through ALL remaining transformer layers
- Signal gets diluted/corrupted through long backprop path

Problem 2: COMPETING OBJECTIVES
- Fear loss wants: P(caution) higher
- Neutral loss wants: match base distribution
- These fight each other since adapters affect BOTH conditions
- Model finds degenerate solution (collapse to simple distribution)

Problem 3: WRONG TARGET
- We boost P(caution tokens) by 5x
- But caution tokens have TINY base probability (~0.0001)
- 5x of tiny is still tiny
- Meanwhile, the loss tries to match the ENTIRE distribution
- Model learns to match high-prob tokens, ignores caution tokens

Problem 4: NO HIDDEN STATE GUIDANCE
- Unlike Alt 1, we don't tell adapters WHAT direction to modify
- We only say "make logits different"
- Too many degrees of freedom → finds degenerate solution
""")

    print("\nVerifying gradient flow issue:")
    print(f"  Mean gradient norm: {np.mean(gradient_norms):.6f}")
    print(f"  Gradient norm variance: {np.std(gradient_norms):.6f}")

    print("\nVerifying entropy collapse:")
    print(f"  Pre-training entropy:  {pre_corruption['fear_entropy']:.2f}")
    print(f"  Post-training entropy: {post_corruption['fear_entropy']:.2f}")

    if post_corruption['fear_entropy'] < pre_corruption['fear_entropy'] * 0.5:
        print("  → CONFIRMED: Entropy collapsed by >50%")


def compare_with_alt1():
    """Show why Alt 1 works where others fail."""
    print("\n" + "#" * 70)
    print("# WHY ALTERNATIVE 1 SUCCEEDS")
    print("#" * 70)

    print("""
Alternative 1 (Steering Supervision) works because:

1. DIRECT TARGET
   - Loss: MSE(adapter_output, steering_vector)
   - Adapter knows EXACTLY what to output
   - No ambiguity, no degenerate solutions

2. SHORT GRADIENT PATH
   - Loss is directly on adapter output
   - No need to backprop through transformer layers
   - Clean, strong gradient signal

3. EXPLICIT CONTRAST
   - Fear state: output = steering_vector * 0.9
   - Neutral state: output = zero
   - Gate learns meaningful differentiation

4. MAGNITUDE CONTROLLED
   - Steering vector is pre-normalized
   - Target magnitude is explicit (0.9 * norm)
   - No risk of explosion or collapse

The key insight: Don't try to learn WHAT to output from scratch.
Use steering (which already works) to TEACH the adapter what to output.
""")


def main():
    print("=" * 70)
    print("DEEP DIAGNOSIS: Why Alternative 2 & 3 Fail")
    print("=" * 70)
    sys.stdout.flush()

    diagnose_alt2_distillation()
    sys.stdout.flush()

    diagnose_alt3_token_probs()
    sys.stdout.flush()

    compare_with_alt1()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
