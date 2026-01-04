#!/usr/bin/env python3
"""
Test emotional steering on Qwen2.5-1.5B BASE model (not instruct).
Base models are typically more steerable than instruction-tuned variants.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def count_markers(text: str, emotion: str) -> int:
    patterns = {
        "fear": r'\b(careful|caution|risk|danger|warning|avoid|might|could|uncertain|worried|safety|concern|aware|hesitate|threat|afraid|scary)\b',
        "curiosity": r'\b(wonder|curious|interesting|explore|discover|learn|why|how|fascinating|question|investigate|understand|inquire|intriguing)\b',
        "anger": r'\b(must|definitely|try|push|overcome|fight|persist|determined|will|insist|demand|action|strong|refuse|never|always)\b',
        "joy": r'\b(great|wonderful|amazing|happy|enjoy|love|exciting|positive|fantastic|delightful|pleased|good|celebrate|thrilled|beautiful)\b',
    }
    return len(re.findall(patterns.get(emotion, ""), text.lower()))


def compute_stats(values):
    n = len(values)
    mean = sum(values) / n if n > 0 else 0
    var = sum((x - mean)**2 for x in values) / n if n > 1 else 0
    return mean, var ** 0.5


def cohens_d(v1, v2):
    m1, s1 = compute_stats(v1)
    m2, s2 = compute_stats(v2)
    pooled = ((s1**2 + s2**2) / 2) ** 0.5
    return (m2 - m1) / pooled if pooled > 0 else 0


class SteeringHook:
    def __init__(self, direction, scale):
        self.direction = direction
        self.scale = scale

    def __call__(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        steering = self.direction.to(hidden.device, hidden.dtype) * self.scale
        hidden = hidden + steering.unsqueeze(0).unsqueeze(0)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden


def extract_directions_base(model, tokenizer, pairs, device):
    """Extract directions from base model using contrastive pairs."""
    directions = {}

    for emotion, emotion_pairs in pairs.items():
        print(f"  Extracting {emotion} direction from {len(emotion_pairs)} pairs...")

        neutral_acts = []
        emotional_acts = []

        for pair in emotion_pairs[:100]:  # Use up to 100 pairs
            # Neutral activation
            neutral_input = tokenizer(pair["neutral"], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**neutral_input, output_hidden_states=True)
                # Layer 0 had highest explained variance
                act = outputs.hidden_states[0][:, -1, :].cpu().float()
                neutral_acts.append(act)

            # Emotional activation
            emotional_input = tokenizer(pair["emotional"], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**emotional_input, output_hidden_states=True)
                act = outputs.hidden_states[0][:, -1, :].cpu().float()
                emotional_acts.append(act)

        # Compute direction as difference in means
        neutral_mean = torch.cat(neutral_acts, dim=0).mean(dim=0)
        emotional_mean = torch.cat(emotional_acts, dim=0).mean(dim=0)
        direction = emotional_mean - neutral_mean

        # Normalize
        direction = direction / (direction.norm() + 1e-8)
        directions[emotion] = direction

        print(f"    Direction norm: {direction.norm():.4f}")

    return directions


def run_base_model_test():
    print("=" * 70)
    print("TESTING BASE MODEL: Qwen2.5-1.5B (NOT Instruct)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load BASE model (not instruct)
    base_model_name = "Qwen/Qwen2.5-1.5B"
    instruct_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    print(f"\nLoading BASE model: {base_model_name}")
    tokenizer_base = AutoTokenizer.from_pretrained(base_model_name)
    model_base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    # Load training pairs
    print("\nLoading training pairs...")
    with open("data/emotional_pairs_large.json") as f:
        pairs = json.load(f)

    # Extract directions for base model
    print("\nExtracting directions for BASE model...")
    directions_base = extract_directions_base(model_base, tokenizer_base, pairs, device)

    # Test prompts (completion style for base model)
    prompts = [
        "The risky investment opportunity made me feel",
        "When I discovered something new and fascinating, I",
        "Despite all the obstacles, I was determined to",
        "The celebration was wonderful because",
        "Facing uncertainty, one should",
        "The dangerous situation required",
        "My curiosity led me to explore",
        "With joy in my heart, I",
    ]

    emotions = ["fear", "curiosity", "anger", "joy"]
    samples = 5
    scales = [0.0, 1.0, 2.0, 3.0]

    results = {scale: {e: [] for e in emotions} for scale in scales}
    sample_outputs = {scale: {e: "" for e in emotions} for scale in scales}

    print(f"\nTesting {len(prompts)} prompts × {samples} samples × {len(scales)} scales...")

    for scale in scales:
        print(f"\n  Scale {scale}:")

        for emotion in emotions:
            direction = directions_base[emotion]

            # Install hook on layer 0
            if scale > 0:
                hook = SteeringHook(direction, scale=scale)
                handle = model_base.model.layers[0].register_forward_hook(hook)

            for p_idx, prompt in enumerate(prompts):
                for s in range(samples):
                    seed = 2000 + p_idx * 100 + s
                    torch.manual_seed(seed)

                    input_ids = tokenizer_base(prompt, return_tensors="pt").input_ids.to(device)

                    with torch.no_grad():
                        out = model_base.generate(
                            input_ids,
                            max_new_tokens=50,
                            temperature=0.8,
                            do_sample=True,
                            pad_token_id=tokenizer_base.eos_token_id,
                        )

                    resp = tokenizer_base.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
                    results[scale][emotion].append(count_markers(resp, emotion))

                    # Save first sample
                    if p_idx == 0 and s == 0:
                        sample_outputs[scale][emotion] = resp[:100]

            if scale > 0:
                handle.remove()

    # Results
    print("\n")
    print("=" * 70)
    print("RESULTS: AVERAGE EMOTION MARKERS BY SCALE")
    print("=" * 70)

    print(f"\n{'Scale':<8} {'Fear':<10} {'Curiosity':<12} {'Anger':<10} {'Joy':<10} {'Total':<10}")
    print("-" * 60)

    summary = {}
    for scale in scales:
        summary[scale] = {}
        row = f"{scale:<8}"
        total = 0
        for e in emotions:
            m, _ = compute_stats(results[scale][e])
            summary[scale][e] = m
            total += m
            row += f"{m:<10.2f}"
        summary[scale]["total"] = total
        row += f"{total:<10.2f}"
        print(row)

    # Effect sizes vs baseline
    print("\n")
    print("=" * 70)
    print("EFFECT SIZES (Cohen's d) vs NO STEERING")
    print("=" * 70)

    print(f"\n{'Scale':<8} {'Fear':<10} {'Curiosity':<12} {'Anger':<10} {'Joy':<10} {'Avg |d|':<10}")
    print("-" * 60)

    effect_sizes = {}
    for scale in [1.0, 2.0, 3.0]:
        row = f"{scale:<8}"
        ds = []
        for e in emotions:
            d = cohens_d(results[0.0][e], results[scale][e])
            ds.append(abs(d))
            row += f"{d:+.2f}      "
        avg_d = sum(ds) / len(ds)
        effect_sizes[scale] = avg_d
        row += f"{avg_d:.2f}"
        print(row)

    # Sample outputs
    print("\n")
    print("=" * 70)
    print("SAMPLE OUTPUTS (Fear emotion)")
    print("=" * 70)

    prompt = prompts[0]
    print(f"\nPrompt: '{prompt}'")

    for scale in scales:
        print(f"\n  Scale {scale}: {sample_outputs[scale]['fear'][:80]}...")

    # Verdict
    print("\n")
    print("=" * 70)
    print("VERDICT: BASE MODEL STEERABILITY")
    print("=" * 70)

    best_scale = max(effect_sizes, key=effect_sizes.get)
    best_d = effect_sizes[best_scale]

    if best_d > 0.8:
        verdict = "STRONG"
        print(f"\n  ✅ STRONG STEERING EFFECT (avg |d|={best_d:.2f} at scale={best_scale})")
        print("     Base model is significantly more steerable than instruct!")
    elif best_d > 0.5:
        verdict = "MEDIUM"
        print(f"\n  ⚪ MEDIUM STEERING EFFECT (avg |d|={best_d:.2f} at scale={best_scale})")
        print("     Base model shows moderate steerability.")
    elif best_d > 0.2:
        verdict = "SMALL"
        print(f"\n  ⚪ SMALL STEERING EFFECT (avg |d|={best_d:.2f} at scale={best_scale})")
        print("     Base model is slightly more steerable than instruct.")
    else:
        verdict = "NEGLIGIBLE"
        print(f"\n  ❌ NEGLIGIBLE EFFECT (avg |d|={best_d:.2f})")
        print("     Base model also resistant to steering.")

    # Compare with instruct results
    print("\n")
    print("=" * 70)
    print("COMPARISON: BASE vs INSTRUCT")
    print("=" * 70)

    instruct_avg_d = 0.06  # From previous test
    print(f"\n  Instruct model avg |d|: {instruct_avg_d:.2f}")
    print(f"  Base model avg |d|:     {best_d:.2f}")
    print(f"  Improvement:            {best_d / instruct_avg_d:.1f}x" if instruct_avg_d > 0 else "  Improvement: N/A")

    # Save results
    with open("data/base_model_results.json", "w") as f:
        json.dump({
            "summary": {str(k): v for k, v in summary.items()},
            "effect_sizes": effect_sizes,
            "best_scale": best_scale,
            "best_d": best_d,
            "verdict": verdict,
        }, f, indent=2)

    print(f"\nResults saved to: data/base_model_results.json")

    # Cleanup
    del model_base
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_base_model_test()
