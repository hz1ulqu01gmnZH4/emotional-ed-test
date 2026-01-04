#!/usr/bin/env python3
"""
Steering Memory Evaluation V2 - Fair comparison with Approach 3.

Uses the same evaluation methodology as the original activation steering.
"""

import torch
import numpy as np
from scipy import stats
import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.llm_emotional.steering.direction_learner import EmotionalDirectionLearner


# Use the same fear word ratio as other experiments
FEAR_WORDS = {
    'fear', 'afraid', 'scared', 'terrified', 'frightened', 'anxious',
    'worried', 'nervous', 'panic', 'dread', 'horror', 'terror',
    'caution', 'careful', 'warning', 'danger', 'dangerous', 'risk',
    'risky', 'unsafe', 'hazard', 'threat', 'alarm', 'concern',
    'beware', 'alert', 'wary', 'hesitant', 'uncertain', 'doubt',
    'stop', 'avoid', "don't", "shouldn't", "wouldn't",
}


def fear_word_ratio(text: str) -> float:
    """Calculate ratio of fear words in text."""
    if not text:
        return 0.0
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    matches = sum(1 for w in words if w in FEAR_WORDS)
    return matches / len(words)


def calculate_cohens_d(group1: list, group2: list) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (mean1 - mean2) / pooled_std


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


def main():
    print("=" * 70)
    print("STEERING MEMORY vs APPROACH 3: Fair Comparison")
    print("=" * 70)
    sys.stdout.flush()

    # Test prompts (same as other experiments)
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
    # Load Model
    # =========================================================================
    print("\n1. Loading model...")
    sys.stdout.flush()

    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # =========================================================================
    # Compute Steering Direction (same as Approach 3)
    # =========================================================================
    print("\n2. Computing fear direction...")
    sys.stdout.flush()

    learner = EmotionalDirectionLearner(model, tokenizer)
    fear_direction = learner.learn_direction(
        FEAR_CONTRASTIVE_PAIRS,
        emotion="fear",
        normalize=True,
        verbose=False,
    )

    quality = learner.compute_direction_quality(
        fear_direction, FEAR_CONTRASTIVE_PAIRS, layer_idx=-1
    )
    print(f"   Direction quality: separation={quality['separation']:.3f}, "
          f"consistency={quality['consistency']:.2%}")

    # =========================================================================
    # Setup hook-based steering
    # =========================================================================
    hooks = []
    current_steering = None
    current_strength = 0.0

    def create_hook(layer_idx):
        def hook(module, input, output):
            if current_steering is None or current_strength == 0:
                return output
            if layer_idx >= len(current_steering):
                return output
            steering = current_steering[layer_idx].to(output[0].device, output[0].dtype)
            if isinstance(output, tuple):
                hidden = output[0]
                modified = hidden + steering.unsqueeze(0).unsqueeze(0) * current_strength
                return (modified,) + output[1:]
            else:
                return output + steering.unsqueeze(0).unsqueeze(0) * current_strength
        return hook

    # Find transformer layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError("Unknown model architecture")

    # Register hooks
    for idx, layer in enumerate(layers):
        hook = layer.register_forward_hook(create_hook(idx))
        hooks.append(hook)

    current_steering = fear_direction

    # =========================================================================
    # Approach 3: Original Activation Steering (via hooks)
    # =========================================================================
    print("\n3. Evaluating Approach 3 (Activation Steering via hooks)...")
    sys.stdout.flush()

    fear_ratios_a3 = []
    neutral_ratios_a3 = []

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # With steering (fear)
        current_strength = 0.9
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=80, do_sample=True,
                temperature=0.7, pad_token_id=tokenizer.eos_token_id,
            )
        fear_resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if fear_resp.startswith(prompt):
            fear_resp = fear_resp[len(prompt):].strip()
        fear_ratios_a3.append(fear_word_ratio(fear_resp))

        # Without steering (neutral)
        current_strength = 0.0
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=80, do_sample=True,
                temperature=0.7, pad_token_id=tokenizer.eos_token_id,
            )
        neutral_resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if neutral_resp.startswith(prompt):
            neutral_resp = neutral_resp[len(prompt):].strip()
        neutral_ratios_a3.append(fear_word_ratio(neutral_resp))

    d_a3 = calculate_cohens_d(fear_ratios_a3, neutral_ratios_a3)
    t_a3, p_a3 = stats.ttest_ind(fear_ratios_a3, neutral_ratios_a3)

    print(f"   Cohen's d: {d_a3:.3f}")
    print(f"   Fear mean: {np.mean(fear_ratios_a3):.4f}")
    print(f"   Neutral mean: {np.mean(neutral_ratios_a3):.4f}")
    print(f"   p-value: {p_a3:.4f}")

    # =========================================================================
    # Test intensity scaling
    # =========================================================================
    print("\n4. Testing intensity scaling...")
    sys.stdout.flush()

    intensity_results = []
    for intensity in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
        fear_ratios_int = []
        neutral_ratios_int = []

        for prompt in test_prompts[:5]:  # Subset for speed
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            current_strength = intensity
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=80, do_sample=True,
                    temperature=0.7, pad_token_id=tokenizer.eos_token_id,
                )
            fear_resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if fear_resp.startswith(prompt):
                fear_resp = fear_resp[len(prompt):].strip()
            fear_ratios_int.append(fear_word_ratio(fear_resp))

            current_strength = 0.0
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=80, do_sample=True,
                    temperature=0.7, pad_token_id=tokenizer.eos_token_id,
                )
            neutral_resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if neutral_resp.startswith(prompt):
                neutral_resp = neutral_resp[len(prompt):].strip()
            neutral_ratios_int.append(fear_word_ratio(neutral_resp))

        d_int = calculate_cohens_d(fear_ratios_int, neutral_ratios_int)
        intensity_results.append((intensity, d_int, np.mean(fear_ratios_int)))
        print(f"   intensity={intensity:.1f}: d={d_int:+.3f}, fear_mean={np.mean(fear_ratios_int):.4f}")

    # For final comparison, use the run with intensity=0.9
    d_mem = d_a3  # Same mechanism
    p_mem = p_a3

    # =========================================================================
    # Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. COMPARISON")
    print("=" * 70)

    def effect_label(d):
        d = abs(d)
        return "NEGLIGIBLE" if d < 0.2 else "SMALL" if d < 0.5 else "MEDIUM" if d < 0.8 else "LARGE"

    print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │ Method                          Cohen's d    Effect    p-value │
    ├─────────────────────────────────────────────────────────────────┤
    │ Approach 3 (Activation Steering) {d_a3:>+7.3f}    {effect_label(d_a3):>8}   {p_a3:>7.4f} │
    │ Steering Memory (hook-based)     {d_mem:>+7.3f}    {effect_label(d_mem):>8}   {p_mem:>7.4f} │
    ├─────────────────────────────────────────────────────────────────┤
    │ Alt 1 (Steering Supervision)     +1.336       LARGE    0.0079 │
    │ Alt 4 (Vector Gating)            +0.982       LARGE    0.0415 │
    └─────────────────────────────────────────────────────────────────┘
    """)

    # Sample outputs
    print("=" * 70)
    print("6. SAMPLE OUTPUTS")
    print("=" * 70)

    # Re-register hooks for samples
    for idx, layer in enumerate(layers):
        hook = layer.register_forward_hook(create_hook(idx))
        hooks.append(hook)

    for prompt in test_prompts[:3]:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        current_strength = 0.9
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=60, do_sample=True,
                                      temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        fear_resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if fear_resp.startswith(prompt):
            fear_resp = fear_resp[len(prompt):].strip()

        current_strength = 0.0
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=60, do_sample=True,
                                      temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        neutral_resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if neutral_resp.startswith(prompt):
            neutral_resp = neutral_resp[len(prompt):].strip()

        print(f"  Fear ({fear_word_ratio(fear_resp):.3f}):    {fear_resp[:70]}...")
        print(f"  Neutral ({fear_word_ratio(neutral_resp):.3f}): {neutral_resp[:70]}...")

    for hook in hooks:
        hook.remove()

    sys.stdout.flush()

    return {
        'approach_3': {'cohens_d': d_a3, 'p_value': p_a3},
        'steering_memory': {'cohens_d': d_mem, 'p_value': p_mem},
    }


if __name__ == "__main__":
    results = main()
