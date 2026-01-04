#!/usr/bin/env python3
"""
Evaluate Steering Memory with Cohen's d metric.

Tests:
1. Individual behaviors (fear, joy, formal, casual, cautious)
2. Behavior compositions
3. Intensity scaling
"""

import torch
import numpy as np
from scipy import stats
import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.steering_memory import (
    SteeringMemory,
    SteeringLLM,
    compute_steering_vector,
    EMOTION_PAIRS,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Evaluation Metrics
# =============================================================================

# Word lists for different behaviors
FEAR_WORDS = {
    'fear', 'afraid', 'scared', 'terrified', 'frightened', 'anxious',
    'worried', 'nervous', 'panic', 'dread', 'horror', 'terror',
    'caution', 'careful', 'warning', 'danger', 'dangerous', 'risk',
    'risky', 'unsafe', 'hazard', 'threat', 'alarm', 'concern',
    'beware', 'alert', 'wary', 'hesitant', 'uncertain', 'doubt',
    'stop', 'avoid', 'don\'t', 'shouldn\'t', 'wouldn\'t', 'careful',
}

JOY_WORDS = {
    'happy', 'joy', 'joyful', 'delighted', 'excited', 'thrilled',
    'wonderful', 'fantastic', 'amazing', 'great', 'excellent', 'love',
    'lovely', 'beautiful', 'pleasure', 'pleased', 'glad', 'cheerful',
    'enthusiasm', 'enthusiastic', 'celebrate', 'celebrating', 'fun',
    'enjoy', 'enjoying', 'awesome', 'brilliant', 'magnificent', 'superb',
}

FORMAL_WORDS = {
    'indeed', 'furthermore', 'moreover', 'therefore', 'consequently',
    'regarding', 'concerning', 'pursuant', 'hereby', 'whereas',
    'acknowledge', 'appreciate', 'request', 'inquire', 'assist',
    'facilitate', 'endeavor', 'commence', 'conclude', 'recommend',
    'respectfully', 'sincerely', 'cordially', 'appropriately',
    'professional', 'formal', 'proper', 'suitable', 'adequate',
}

CASUAL_WORDS = {
    'hey', 'yeah', 'yep', 'nope', 'cool', 'awesome', 'gonna',
    'wanna', 'gotta', 'kinda', 'sorta', 'stuff', 'things', 'guy',
    'dude', 'man', 'like', 'okay', 'ok', 'sure', 'right', 'well',
    'so', 'basically', 'actually', 'honestly', 'literally', 'super',
    'totally', 'pretty', 'really', 'just', 'maybe', 'probably',
}

CAUTIOUS_WORDS = FEAR_WORDS | {
    'consider', 'careful', 'caution', 'verify', 'check', 'ensure',
    'confirm', 'validate', 'review', 'evaluate', 'assess', 'examine',
    'potential', 'possible', 'might', 'could', 'may', 'perhaps',
    'recommend', 'suggest', 'advise', 'before', 'first', 'important',
}


def word_ratio(text: str, word_set: set) -> float:
    """Calculate ratio of words from word_set in text."""
    if not text:
        return 0.0
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    matches = sum(1 for w in words if w in word_set)
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


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_behavior(
    steering_llm: SteeringLLM,
    behavior: str,
    word_set: set,
    test_prompts: list,
    intensity: float = 1.0,
) -> dict:
    """Evaluate a single behavior."""
    steered_ratios = []
    baseline_ratios = []

    for prompt in test_prompts:
        # Baseline (no steering)
        baseline_resp = steering_llm.generate(prompt, steering=None, max_new_tokens=80)
        baseline_ratios.append(word_ratio(baseline_resp, word_set))

        # Steered
        steered_resp = steering_llm.generate(
            prompt,
            steering={behavior: intensity},
            max_new_tokens=80,
        )
        steered_ratios.append(word_ratio(steered_resp, word_set))

    cohens_d = calculate_cohens_d(steered_ratios, baseline_ratios)
    t_stat, p_value = stats.ttest_rel(steered_ratios, baseline_ratios)

    return {
        'behavior': behavior,
        'intensity': intensity,
        'steered_mean': np.mean(steered_ratios),
        'baseline_mean': np.mean(baseline_ratios),
        'cohens_d': cohens_d,
        't_stat': t_stat,
        'p_value': p_value,
        'steered_ratios': steered_ratios,
        'baseline_ratios': baseline_ratios,
    }


def evaluate_composition(
    steering_llm: SteeringLLM,
    composition: dict,
    word_set: set,
    test_prompts: list,
) -> dict:
    """Evaluate a behavior composition."""
    steered_ratios = []
    baseline_ratios = []

    for prompt in test_prompts:
        baseline_resp = steering_llm.generate(prompt, steering=None, max_new_tokens=80)
        baseline_ratios.append(word_ratio(baseline_resp, word_set))

        steered_resp = steering_llm.generate(prompt, steering=composition, max_new_tokens=80)
        steered_ratios.append(word_ratio(steered_resp, word_set))

    cohens_d = calculate_cohens_d(steered_ratios, baseline_ratios)
    t_stat, p_value = stats.ttest_rel(steered_ratios, baseline_ratios)

    return {
        'composition': composition,
        'steered_mean': np.mean(steered_ratios),
        'baseline_mean': np.mean(baseline_ratios),
        'cohens_d': cohens_d,
        't_stat': t_stat,
        'p_value': p_value,
    }


def main():
    print("=" * 70)
    print("STEERING MEMORY EVALUATION")
    print("=" * 70)
    sys.stdout.flush()

    # =========================================================================
    # Setup
    # =========================================================================
    print("\n1. Loading model and steering memory...")
    sys.stdout.flush()

    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load or create steering memory
    storage_path = Path("data/steering_memory")
    memory = SteeringMemory(storage_path)

    # Ensure all behaviors are computed
    behaviors_config = {
        "fear": FEAR_WORDS,
        "joy": JOY_WORDS,
        "formal": FORMAL_WORDS,
        "casual": CASUAL_WORDS,
        "cautious": CAUTIOUS_WORDS,
    }

    for behavior in behaviors_config:
        if behavior not in memory.vectors:
            print(f"   Computing '{behavior}' vector...")
            vec = compute_steering_vector(
                model, tokenizer, EMOTION_PAIRS[behavior],
                name=behavior,
                description=f"Steering vector for {behavior}",
                tags=[behavior],
            )
            memory.add(vec, persist=True)

    print(f"   Loaded {len(memory.vectors)} steering vectors")

    steering_llm = SteeringLLM(model, tokenizer, memory)

    # =========================================================================
    # Test Prompts
    # =========================================================================
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
        "Tell me about your opinion on this matter.",
        "What would you recommend?",
        "How should I proceed?",
        "Can you help me decide?",
        "What are the risks involved?",
    ]

    print(f"   Test prompts: {len(test_prompts)}")
    sys.stdout.flush()

    # =========================================================================
    # Evaluate Individual Behaviors
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. INDIVIDUAL BEHAVIOR EVALUATION")
    print("=" * 70)
    sys.stdout.flush()

    results = {}

    for behavior, word_set in behaviors_config.items():
        print(f"\n   Evaluating '{behavior}'...")
        sys.stdout.flush()

        result = evaluate_behavior(steering_llm, behavior, word_set, test_prompts)
        results[behavior] = result

        d = result['cohens_d']
        effect = "NEGLIGIBLE" if abs(d) < 0.2 else "SMALL" if abs(d) < 0.5 else "MEDIUM" if abs(d) < 0.8 else "LARGE"
        sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""

        print(f"   {behavior:10s}: d = {d:+.3f} ({effect}) p = {result['p_value']:.4f} {sig}")
        print(f"              steered = {result['steered_mean']:.4f}, baseline = {result['baseline_mean']:.4f}")

    # =========================================================================
    # Evaluate Intensity Scaling
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. INTENSITY SCALING (Fear)")
    print("=" * 70)
    sys.stdout.flush()

    intensity_results = []
    for intensity in [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]:
        result = evaluate_behavior(steering_llm, "fear", FEAR_WORDS, test_prompts[:8], intensity)
        intensity_results.append(result)

        d = result['cohens_d']
        print(f"   intensity={intensity:.1f}: d = {d:+.3f}, steered = {result['steered_mean']:.4f}")
        sys.stdout.flush()

    # =========================================================================
    # Evaluate Compositions
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. BEHAVIOR COMPOSITIONS")
    print("=" * 70)
    sys.stdout.flush()

    compositions = [
        ({"fear": 0.8, "formal": 0.5}, CAUTIOUS_WORDS, "fear+formal → cautious"),
        ({"joy": 0.8, "casual": 0.5}, JOY_WORDS, "joy+casual → enthusiastic"),
        ({"cautious": 1.0, "formal": 0.8}, CAUTIOUS_WORDS, "cautious+formal → professional"),
        ({"fear": 0.5, "joy": 0.5}, FEAR_WORDS | JOY_WORDS, "fear+joy → mixed"),
    ]

    for composition, word_set, label in compositions:
        result = evaluate_composition(steering_llm, composition, word_set, test_prompts[:8])
        d = result['cohens_d']
        effect = "NEGLIGIBLE" if abs(d) < 0.2 else "SMALL" if abs(d) < 0.5 else "MEDIUM" if abs(d) < 0.8 else "LARGE"
        print(f"\n   {label}:")
        print(f"      d = {d:+.3f} ({effect}), p = {result['p_value']:.4f}")
        print(f"      steered = {result['steered_mean']:.4f}, baseline = {result['baseline_mean']:.4f}")
        sys.stdout.flush()

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. SUMMARY")
    print("=" * 70)

    print("\n   Individual Behaviors:")
    print("   " + "-" * 55)
    print(f"   {'Behavior':<12} {'Cohens d':>10} {'Effect':>12} {'p-value':>10} {'Sig':>5}")
    print("   " + "-" * 55)

    for behavior, result in results.items():
        d = result['cohens_d']
        effect = "NEGLIGIBLE" if abs(d) < 0.2 else "SMALL" if abs(d) < 0.5 else "MEDIUM" if abs(d) < 0.8 else "LARGE"
        sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
        print(f"   {behavior:<12} {d:>+10.3f} {effect:>12} {result['p_value']:>10.4f} {sig:>5}")

    print("   " + "-" * 55)

    # Compare with previous approaches
    print("\n   Comparison with Previous Approaches:")
    print("   " + "-" * 55)
    print(f"   {'Method':<30} {'Cohens d':>10} {'Status':>12}")
    print("   " + "-" * 55)
    print(f"   {'Alt 1 (Steering Supervision)':<30} {'1.336':>10} {'BEST':>12}")
    print(f"   {'Alt 4 (Vector Gating)':<30} {'0.982':>10} {'2nd':>12}")
    print(f"   {'Approach 3 (Act. Steering)':<30} {'0.910':>10} {'3rd':>12}")

    fear_d = results['fear']['cohens_d']
    print(f"   {'Steering Memory (fear)':<30} {fear_d:>+10.3f} {'NEW':>12}")

    print("   " + "-" * 55)

    # Sample outputs
    print("\n" + "=" * 70)
    print("6. SAMPLE OUTPUTS")
    print("=" * 70)

    sample_prompt = "Should I click this suspicious link?"
    print(f"\n   Prompt: {sample_prompt}\n")

    print("   Baseline:")
    resp = steering_llm.generate(sample_prompt, steering=None, max_new_tokens=60)
    print(f"   {resp[:100]}...\n")

    for behavior in ["fear", "joy", "formal", "cautious"]:
        print(f"   {behavior.upper()} (intensity=1.0):")
        resp = steering_llm.generate(sample_prompt, steering={behavior: 1.0}, max_new_tokens=60)
        print(f"   {resp[:100]}...\n")

    sys.stdout.flush()
    return results


if __name__ == "__main__":
    results = main()
