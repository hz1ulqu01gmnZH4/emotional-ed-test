#!/usr/bin/env python3
"""
Test with free-form prompts to avoid MCQ-style generation.
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
        "fear": r'\b(careful|caution|risk|danger|warning|avoid|might|could|uncertain|worried|safety|concern|aware|hesitate|threat|afraid|scary|anxious|nervous|uneasy)\b',
        "curiosity": r'\b(wonder|curious|interesting|explore|discover|learn|why|how|fascinating|question|investigate|understand|inquire|intriguing|remarkable|mystery)\b',
        "anger": r'\b(must|definitely|try|push|overcome|fight|persist|determined|will|insist|demand|action|strong|refuse|never|always|force|won\'t)\b',
        "joy": r'\b(great|wonderful|amazing|happy|enjoy|love|exciting|positive|fantastic|delightful|pleased|good|celebrate|thrilled|beautiful|joyful|glad)\b',
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


def run_test():
    print("=" * 70)
    print("BASE MODEL: FREE-FORM PROMPTS")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    print("Loading pairs and directions...")
    with open("data/emotional_pairs_large.json") as f:
        pairs = json.load(f)

    # Extract direction from layer 0
    directions = {}
    for emotion, emotion_pairs in pairs.items():
        diffs = []
        for pair in emotion_pairs[:50]:
            n_in = tokenizer(pair["neutral"], return_tensors="pt").to(device)
            e_in = tokenizer(pair["emotional"], return_tensors="pt").to(device)
            with torch.no_grad():
                n_out = model(**n_in, output_hidden_states=True)
                e_out = model(**e_in, output_hidden_states=True)
                diff = e_out.hidden_states[0][:, -1, :] - n_out.hidden_states[0][:, -1, :]
                diffs.append(diff.cpu().float())
        direction = torch.cat(diffs, dim=0).mean(dim=0)
        direction = direction / direction.norm()
        directions[emotion] = direction

    # Free-form prompts (narrative style, not question style)
    prompts = [
        "The old house at the end of the street was",
        "Walking through the dark forest, she noticed",
        "After years of hard work, he finally",
        "The scientist discovered something remarkable:",
        "In the face of adversity, the team",
        "The celebration lasted all night because",
        "Deep in the cave, explorers found",
        "Against all odds, they managed to",
    ]

    emotions = ["fear", "curiosity", "anger", "joy"]
    scales = [0.0, 4.0, 6.0, 8.0]
    samples = 5

    results = {s: {e: [] for e in emotions} for s in scales}
    all_outputs = {s: {e: [] for e in emotions} for s in scales}

    print(f"\nTesting {len(scales)} scales × {len(prompts)} prompts × {samples} samples...")

    for scale in scales:
        print(f"\n  Scale {scale}...")

        for emotion in emotions:
            handles = []
            if scale > 0:
                hook = SteeringHook(directions[emotion], scale)
                handle = model.model.layers[0].register_forward_hook(hook)
                handles.append(handle)

            for p_idx, prompt in enumerate(prompts):
                for s in range(samples):
                    torch.manual_seed(4000 + p_idx * 100 + s)

                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

                    with torch.no_grad():
                        out = model.generate(
                            input_ids,
                            max_new_tokens=60,
                            temperature=1.0,  # Higher for more variation
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    resp = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
                    results[scale][emotion].append(count_markers(resp, emotion))
                    all_outputs[scale][emotion].append((prompt, resp))

            for h in handles:
                h.remove()

    # Results
    print("\n")
    print("=" * 70)
    print("RESULTS: EMOTION MARKERS BY SCALE")
    print("=" * 70)

    print(f"\n{'Scale':<8} {'Fear':<10} {'Curiosity':<12} {'Anger':<10} {'Joy':<10} {'Total':<10}")
    print("-" * 60)

    for scale in scales:
        row = f"{scale:<8}"
        total = 0
        for e in emotions:
            m, _ = compute_stats(results[scale][e])
            total += m
            row += f"{m:<10.2f}"
        row += f"{total:<10.2f}"
        print(row)

    # Effect sizes
    print("\n")
    print("=" * 70)
    print("EFFECT SIZES vs BASELINE (scale=0)")
    print("=" * 70)

    print(f"\n{'Scale':<8} {'Fear':<10} {'Curiosity':<12} {'Anger':<10} {'Joy':<10} {'Avg |d|':<10}")
    print("-" * 60)

    best_d = 0
    best_scale = 0

    for scale in [4.0, 6.0, 8.0]:
        row = f"{scale:<8}"
        ds = []
        for e in emotions:
            d = cohens_d(results[0.0][e], results[scale][e])
            ds.append(abs(d))
            row += f"{d:+.2f}      "
        avg_d = sum(ds) / len(ds)
        row += f"{avg_d:.2f}"
        print(row)

        if avg_d > best_d:
            best_d = avg_d
            best_scale = scale

    # Show sample outputs
    print("\n")
    print("=" * 70)
    print("SAMPLE OUTPUTS COMPARISON")
    print("=" * 70)

    for emotion in emotions:
        print(f"\n--- {emotion.upper()} STEERING ---")

        # Find an output that shows difference
        for i, (prompt, _) in enumerate(all_outputs[0.0][emotion][:3]):
            baseline_out = all_outputs[0.0][emotion][i][1]
            steered_out = all_outputs[best_scale][emotion][i][1]

            if baseline_out != steered_out:
                print(f"\nPrompt: '{prompt}'")
                print(f"  Baseline: {baseline_out[:100]}...")
                print(f"  Scale {best_scale}: {steered_out[:100]}...")
                break

    # Verdict
    print("\n")
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if best_d > 0.8:
        verdict = "LARGE"
        symbol = "✅"
    elif best_d > 0.5:
        verdict = "MEDIUM"
        symbol = "✅"
    elif best_d > 0.3:
        verdict = "SMALL-MEDIUM"
        symbol = "⚪"
    else:
        verdict = "SMALL"
        symbol = "⚪"

    print(f"\n  {symbol} {verdict} EFFECT at scale={best_scale}")
    print(f"     avg |d| = {best_d:.2f}")

    # Per-emotion breakdown
    print("\n  Per-emotion effects at best scale:")
    for e in emotions:
        d = cohens_d(results[0.0][e], results[best_scale][e])
        direction = "↑" if d > 0 else "↓"
        size = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small" if abs(d) > 0.2 else "negligible"
        print(f"    {e:12}: d={d:+.2f} ({size})")

    with open("data/base_freeform_results.json", "w") as f:
        json.dump({
            "best_scale": best_scale,
            "best_d": best_d,
            "verdict": verdict,
        }, f, indent=2)

    print(f"\nResults saved to: data/base_freeform_results.json")


if __name__ == "__main__":
    run_test()
