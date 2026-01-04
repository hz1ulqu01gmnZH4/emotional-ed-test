#!/usr/bin/env python3
"""
Final Steering Memory Evaluation with multiple runs for statistical robustness.
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

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

FEAR_WORDS = {
    'fear', 'afraid', 'scared', 'terrified', 'frightened', 'anxious',
    'worried', 'nervous', 'panic', 'dread', 'horror', 'terror',
    'caution', 'careful', 'warning', 'danger', 'dangerous', 'risk',
    'risky', 'unsafe', 'hazard', 'threat', 'alarm', 'concern',
    'beware', 'alert', 'wary', 'hesitant', 'uncertain', 'doubt',
    'stop', 'avoid', "don't", "shouldn't", "wouldn't",
}


def fear_word_ratio(text: str) -> float:
    if not text:
        return 0.0
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    matches = sum(1 for w in words if w in FEAR_WORDS)
    return matches / len(words)


def calculate_cohens_d(group1: list, group2: list) -> float:
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
    print("STEERING MEMORY: Final Evaluation")
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
        "Can I trust this online store?",
        "Should I open this attachment?",
        "Is this a scam?",
        "Should I provide my personal information?",
        "Is it safe to use this app?",
    ]

    print("\n1. Loading model...")
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("\n2. Computing fear direction...")
    learner = EmotionalDirectionLearner(model, tokenizer)
    fear_direction = learner.learn_direction(
        FEAR_CONTRASTIVE_PAIRS, emotion="fear", normalize=True, verbose=False
    )

    # Setup hooks
    hooks = []
    current_strength = [0.0]  # Use list to allow mutation in closure

    def create_hook(layer_idx):
        def hook(module, input, output):
            if current_strength[0] == 0:
                return output
            if layer_idx >= len(fear_direction):
                return output
            steering = fear_direction[layer_idx].to(output[0].device, output[0].dtype)
            if isinstance(output, tuple):
                hidden = output[0]
                modified = hidden + steering.unsqueeze(0).unsqueeze(0) * current_strength[0]
                return (modified,) + output[1:]
            return output + steering.unsqueeze(0).unsqueeze(0) * current_strength[0]
        return hook

    layers = model.model.layers
    for idx, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(create_hook(idx)))

    print("\n3. Running multiple evaluation runs...")
    sys.stdout.flush()

    # Run 3 times for robustness
    all_results = []

    for run in range(3):
        print(f"\n   Run {run+1}/3:")
        fear_ratios = []
        neutral_ratios = []

        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Fear
            current_strength[0] = 0.9
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=100, do_sample=True,
                    temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
                )
            fear_resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if fear_resp.startswith(prompt):
                fear_resp = fear_resp[len(prompt):].strip()
            fear_ratios.append(fear_word_ratio(fear_resp))

            # Neutral
            current_strength[0] = 0.0
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=100, do_sample=True,
                    temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
                )
            neutral_resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if neutral_resp.startswith(prompt):
                neutral_resp = neutral_resp[len(prompt):].strip()
            neutral_ratios.append(fear_word_ratio(neutral_resp))

        d = calculate_cohens_d(fear_ratios, neutral_ratios)
        t, p = stats.ttest_ind(fear_ratios, neutral_ratios)
        all_results.append({
            'd': d, 'p': p,
            'fear_mean': np.mean(fear_ratios),
            'neutral_mean': np.mean(neutral_ratios),
        })
        print(f"      d={d:+.3f}, fear={np.mean(fear_ratios):.4f}, neutral={np.mean(neutral_ratios):.4f}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Average results
    avg_d = np.mean([r['d'] for r in all_results])
    avg_p = np.mean([r['p'] for r in all_results])
    avg_fear = np.mean([r['fear_mean'] for r in all_results])
    avg_neutral = np.mean([r['neutral_mean'] for r in all_results])

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    def effect_label(d):
        d = abs(d)
        return "NEGLIGIBLE" if d < 0.2 else "SMALL" if d < 0.5 else "MEDIUM" if d < 0.8 else "LARGE"

    print(f"""
    Steering Memory (Hook-based Activation Steering)
    ─────────────────────────────────────────────────
    Cohen's d (avg):    {avg_d:+.3f} ({effect_label(avg_d)})
    Fear mean:          {avg_fear:.4f}
    Neutral mean:       {avg_neutral:.4f}
    p-value (avg):      {avg_p:.4f}

    Individual runs: {[f"{r['d']:+.3f}" for r in all_results]}
    """)

    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │               COMPLETE COMPARISON TABLE                         │
    ├─────────────────────────────────────────────────────────────────┤
    │ Method                              Cohen's d    Effect   Stat │
    ├─────────────────────────────────────────────────────────────────┤
    │ Alt 1 (Steering Supervision)          +1.336    LARGE    ***   │
    │ Alt 4 (Vector Gating + Steering)      +0.982    LARGE    *     │
    │ Approach 3 (Activation Steering)      +0.910    LARGE    *     │""")
    print(f"    │ Steering Memory (this evaluation)     {avg_d:+6.3f}    {effect_label(avg_d):<6}   {'*' if avg_p < 0.05 else ''}     │")
    print("""    │ Mitigated Alt 2 (Distillation)        +0.556    MEDIUM        │
    │ Mitigated Alt 3 (Token Probs)         +0.447    SMALL         │
    └─────────────────────────────────────────────────────────────────┘

    Note: Steering Memory uses the SAME mechanism as Approach 3.
    Variance between runs is due to stochastic generation.
    """)

    # Sample outputs
    print("=" * 70)
    print("SAMPLE OUTPUTS")
    print("=" * 70)

    # Re-register hooks for samples
    for idx, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(create_hook(idx)))

    for prompt in test_prompts[:3]:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        current_strength[0] = 0.9
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)
        fear_resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if fear_resp.startswith(prompt):
            fear_resp = fear_resp[len(prompt):].strip()

        current_strength[0] = 0.0
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)
        neutral_resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if neutral_resp.startswith(prompt):
            neutral_resp = neutral_resp[len(prompt):].strip()

        print(f"  Fear ({fear_word_ratio(fear_resp):.3f}):    {fear_resp[:80]}...")
        print(f"  Neutral ({fear_word_ratio(neutral_resp):.3f}): {neutral_resp[:80]}...")

    for hook in hooks:
        hook.remove()

    sys.stdout.flush()
    return {'avg_d': avg_d, 'all_results': all_results}


if __name__ == "__main__":
    main()
