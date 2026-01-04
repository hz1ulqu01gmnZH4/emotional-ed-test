#!/usr/bin/env python3
"""
Deep dive into the gate differentiation problem.

Key question: Why doesn't the gate learn to differentiate fear vs neutral?
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys

from src.emotional_adapter_gating import (
    EmotionalState,
    EmotionalAdapterLLM,
)
from src.llm_emotional.steering.direction_learner import EmotionalDirectionLearner


FEAR_CONTRASTIVE_PAIRS = [
    ("The weather is nice today.", "I'm terrified something bad will happen."),
    ("Let me explain the concept.", "Warning: this is extremely dangerous."),
    ("Here's the information you asked for.", "Be very careful, there are serious risks."),
]


def analyze_gate_learning():
    """Analyze why gates don't differentiate in Alt 2/3 but do in Alt 1."""
    print("=" * 70)
    print("GATE DIFFERENTIATION ANALYSIS")
    print("=" * 70)

    model = EmotionalAdapterLLM(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        adapter_dim=64,
        emotion_dim=6,
        gate_type="scalar",
        torch_dtype=torch.float32,
    )
    device = model.device

    # Get emotional states
    emotions = {
        'neutral': EmotionalState.neutral(),
        'fear_0.3': EmotionalState.fearful(0.3),
        'fear_0.6': EmotionalState.fearful(0.6),
        'fear_0.9': EmotionalState.fearful(0.9),
        'joy_0.9': EmotionalState.joyful(0.9),
        'anger_0.9': EmotionalState.angry(0.9),
    }

    # Analyze gate outputs for different emotional states
    print("\n1. GATE VALUES FOR DIFFERENT EMOTIONAL STATES (before training)")
    print("-" * 60)

    inputs = model.tokenizer("Should I click this link?", return_tensors="pt").to(device)
    with torch.no_grad():
        base_out = model.base_model(**inputs, output_hidden_states=True)

    for emo_name, emo_state in emotions.items():
        emo_tensor = emo_state.to_tensor().to(device).unsqueeze(0)

        gate_values = []
        for layer_name, adapter in list(model.adapters.items())[:3]:
            layer_idx = int(layer_name.split('_')[-1]) if '_' in layer_name else 0
            if layer_idx < len(base_out.hidden_states) - 1:
                base_hidden = base_out.hidden_states[layer_idx + 1]
                gate = adapter.gate(emo_tensor, base_hidden)
                gate_values.append(gate.mean().item())

        print(f"  {emo_name:12s}: gates = {[f'{g:.4f}' for g in gate_values]}")

    # Analyze what the gate network sees
    print("\n2. GATE NETWORK INPUT ANALYSIS")
    print("-" * 60)

    fear_tensor = emotions['fear_0.9'].to_tensor().to(device)
    neutral_tensor = emotions['neutral'].to_tensor().to(device)

    print(f"\n  Fear tensor:    {fear_tensor.tolist()}")
    print(f"  Neutral tensor: {neutral_tensor.tolist()}")
    print(f"  L2 distance:    {(fear_tensor - neutral_tensor).norm().item():.4f}")

    # The gate is: sigmoid(W @ emotion_tensor)
    # If W is random, output is ~0.5 for any input
    adapter = list(model.adapters.values())[0]
    print(f"\n  Gate architecture: {adapter.gate}")

    # Check gate network weights
    for name, param in adapter.gate.named_parameters():
        print(f"  {name}: shape={param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")

    print("\n3. WHY GATES DON'T DIFFERENTIATE IN ALT 2/3")
    print("-" * 60)
    print("""
  In Alt 2 (Distillation) and Alt 3 (Token Probs):

  The loss is computed on FINAL outputs (hidden states or logits).
  The gradient for the gate comes from:

    d(Loss)/d(gate) = d(Loss)/d(output) * d(output)/d(gate)

  But d(output)/d(gate) depends on the adapter output itself:
    output = base_hidden + gate * adapter_out
    d(output)/d(gate) = adapter_out

  PROBLEM: At initialization, adapter_out ≈ 0 (random small weights)

  So: d(Loss)/d(gate) ≈ 0 regardless of Loss!

  The gate receives ZERO gradient signal initially because the adapter
  output is near-zero. This is a "chicken-and-egg" problem:
  - Gate needs adapter output to receive gradients
  - Adapter needs gate signal to learn what to output

  In Alt 1 (Steering Supervision):
  Loss = MSE(gate * adapter_out, target)

  This creates a DIRECT gradient on both gate and adapter simultaneously:
  - d(Loss)/d(gate) = 2 * (gate * adapter_out - target) * adapter_out
  - d(Loss)/d(adapter_out) = 2 * (gate * adapter_out - target) * gate

  Both receive signal even when values are small!
""")

    # Demonstrate the gradient issue
    print("\n4. GRADIENT ANALYSIS: Alt 2 vs Alt 1")
    print("-" * 60)

    # Reset model
    model2 = EmotionalAdapterLLM(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        adapter_dim=64,
        emotion_dim=6,
        gate_type="scalar",
        torch_dtype=torch.float32,
    )

    # Compute fear direction
    learner = EmotionalDirectionLearner(model2.base_model, model2.tokenizer)
    fear_direction = learner.learn_direction(FEAR_CONTRASTIVE_PAIRS, emotion="fear", normalize=True, verbose=False)

    device = model2.device
    inputs = model2.tokenizer("Test prompt", return_tensors="pt").to(device)

    # ALT 2 style loss (on hidden states)
    print("\n  ALT 2 STYLE (Distillation):")
    with torch.no_grad():
        base_out = model2.base_model(**inputs, output_hidden_states=True)

    fear_tensor = EmotionalState.fearful(0.9).to_tensor().to(device).unsqueeze(0)

    adapter = list(model2.adapters.values())[0]
    base_hidden = base_out.hidden_states[1].requires_grad_(False)

    # Forward through adapter
    adapter_out = adapter.down_proj(base_hidden)
    adapter_out = adapter.activation(adapter_out)
    adapter_out = adapter.up_proj(adapter_out)

    gate = adapter.gate(fear_tensor, base_hidden)
    if gate.dim() == 2:
        gate = gate.unsqueeze(1)

    adapted = base_hidden + gate * adapter_out

    # Target: steered hidden
    steering = fear_direction[0].to(device, base_hidden.dtype)
    target = base_hidden + steering.unsqueeze(0).unsqueeze(0) * 0.9

    loss_alt2 = F.mse_loss(adapted, target)
    loss_alt2.backward()

    gate_grad_alt2 = None
    adapter_grad_alt2 = None
    for name, param in adapter.named_parameters():
        if param.grad is not None:
            if 'gate' in name:
                gate_grad_alt2 = param.grad.norm().item()
            else:
                adapter_grad_alt2 = param.grad.norm().item() if adapter_grad_alt2 is None else adapter_grad_alt2

    print(f"    Gate gradient norm:    {gate_grad_alt2:.6f}")
    print(f"    Adapter gradient norm: {adapter_grad_alt2:.6f}")

    # ALT 1 style loss (direct on adapter output)
    print("\n  ALT 1 STYLE (Steering Supervision):")

    # Reset gradients
    for param in adapter.parameters():
        param.grad = None

    adapter_out = adapter.down_proj(base_hidden)
    adapter_out = adapter.activation(adapter_out)
    adapter_out = adapter.up_proj(adapter_out)

    gate = adapter.gate(fear_tensor, base_hidden)
    if gate.dim() == 2:
        gate = gate.unsqueeze(1)

    gated_adapter = gate * adapter_out
    gated_mean = gated_adapter.mean(dim=1)

    target_alt1 = fear_direction[0].to(device) * 0.9

    loss_alt1 = F.mse_loss(gated_mean, target_alt1.unsqueeze(0))
    loss_alt1.backward()

    gate_grad_alt1 = None
    adapter_grad_alt1 = None
    for name, param in adapter.named_parameters():
        if param.grad is not None:
            if 'gate' in name:
                gate_grad_alt1 = param.grad.norm().item()
            else:
                adapter_grad_alt1 = param.grad.norm().item() if adapter_grad_alt1 is None else adapter_grad_alt1

    print(f"    Gate gradient norm:    {gate_grad_alt1:.6f}")
    print(f"    Adapter gradient norm: {adapter_grad_alt1:.6f}")

    print(f"\n  RATIO (Alt1/Alt2):")
    if gate_grad_alt2 > 0:
        print(f"    Gate gradient ratio:    {gate_grad_alt1/gate_grad_alt2:.2f}x")
    if adapter_grad_alt2 and adapter_grad_alt2 > 0:
        print(f"    Adapter gradient ratio: {adapter_grad_alt1/adapter_grad_alt2:.2f}x")

    print("\n5. THE COMPOUNDING ERROR PROBLEM IN ALT 2")
    print("-" * 60)
    print("""
  Alt 2 tries to match: base + adapter_out ≈ base + steering

  This seems equivalent to: adapter_out ≈ steering

  BUT there's a hidden problem: The loss is computed on FULL hidden states,
  which have norm ~10-50. The steering vector has norm ~1.

  So MSE loss is dominated by the base hidden state matching itself!
  The steering correction is a tiny fraction of the loss.

  Example:
    base_hidden norm: 20.0
    steering norm: 1.0

    If adapter_out = 0:
      Loss = MSE(base, base + steering) = steering.norm()² ≈ 1.0

    If adapter_out = steering:
      Loss = MSE(base + steering, base + steering) = 0.0

    Gradient = 2 * (adapter_out - steering)

  This SHOULD work... but the problem is that the loss is on the OUTPUT
  of the whole transformer, not on each layer's hidden state.

  Errors COMPOUND through layers:
  - Layer 1: small error
  - Layer 2: receives corrupted input + adds its own error
  - Layer 3: errors compound further
  - ...
  - Layer 30: catastrophic error accumulation
""")

    print("\n6. SUMMARY: ROOT CAUSES OF FAILURE")
    print("-" * 60)
    print("""
  ALT 2 (Knowledge Distillation) FAILS because:
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Error compounding through 30 layers                      │
  │ 2. Each layer's adapter modifies input for ALL later layers │
  │ 3. No layer-wise loss to prevent drift                      │
  │ 4. Gate receives weak gradient (adapter_out ≈ 0 initially)  │
  └─────────────────────────────────────────────────────────────┘

  ALT 3 (Token Probability) FAILS because:
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Caution tokens have negligible base probability (1e-6)   │
  │ 2. 5x boost of 1e-6 = 5e-6 (still negligible)              │
  │ 3. KL loss dominated by high-probability tokens             │
  │ 4. No guidance on WHAT direction to modify hidden states    │
  │ 5. Adapters learn arbitrary perturbations                   │
  └─────────────────────────────────────────────────────────────┘

  ALT 1 (Steering Supervision) SUCCEEDS because:
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Direct loss on adapter output (no compounding)           │
  │ 2. Explicit target (steering vector) - no ambiguity         │
  │ 3. Strong gradient signal to both gate and adapter          │
  │ 4. Layer-wise training prevents error accumulation          │
  └─────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    analyze_gate_learning()
    sys.stdout.flush()
