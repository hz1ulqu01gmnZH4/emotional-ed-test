#!/usr/bin/env python3
"""
Validate directions trained on 800 pairs vs 80 pairs.
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
        "fear": r'\b(careful|caution|risk|danger|warning|avoid|might|could|uncertain|worried|safety|concern|aware|hesitate|threat)\b',
        "curiosity": r'\b(wonder|curious|interesting|explore|discover|learn|why|how|fascinating|question|investigate|understand|inquire)\b',
        "anger": r'\b(must|definitely|try|push|overcome|fight|persist|determined|will|insist|demand|action|strong|refuse)\b',
        "joy": r'\b(great|wonderful|amazing|happy|enjoy|love|exciting|positive|fantastic|delightful|pleased|good|celebrate|thrilled)\b',
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


def run_validation():
    print("=" * 70)
    print("VALIDATION: 800 PAIRS vs 80 PAIRS vs BASELINE")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    # Load both direction banks
    with open("data/direction_bank_pca.json") as f:
        dirs_80 = json.load(f)
    with open("data/direction_bank_pca_large.json") as f:
        dirs_800 = json.load(f)

    print("Model loaded!")

    prompts = [
        "What should I do about this risky situation?",
        "Tell me about exploring new possibilities.",
        "How do I overcome this challenge?",
        "Share your thoughts on celebrating success.",
        "What advice for someone facing uncertainty?",
        "How should I approach a difficult decision?",
        "What are the dangers of this approach?",
        "What makes this topic fascinating?",
    ]

    emotions = ["fear", "curiosity", "anger", "joy"]
    samples = 5

    results = {
        "baseline": {e: [] for e in emotions},
        "80_pairs": {e: [] for e in emotions},
        "800_pairs": {e: [] for e in emotions},
    }

    print(f"\nTesting {len(prompts)} prompts × {samples} samples × 3 conditions...")

    for p_idx, prompt in enumerate(prompts):
        print(f"  Prompt {p_idx+1}/{len(prompts)}...")

        for emotion in emotions:
            dir_80 = torch.tensor(dirs_80["directions"][emotion], dtype=torch.float32)
            dir_800 = torch.tensor(dirs_800["directions"][emotion], dtype=torch.float32)

            for s in range(samples):
                seed = 1000 + p_idx * 100 + s

                # Baseline
                torch.manual_seed(seed)
                messages = [{"role": "user", "content": prompt}]
                input_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    out = model.generate(input_ids, max_new_tokens=80, temperature=0.8,
                                        do_sample=True, pad_token_id=tokenizer.eos_token_id)
                resp = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
                results["baseline"][emotion].append(count_markers(resp, emotion))

                # 80 pairs - target layer 0 (highest variance)
                hook = SteeringHook(dir_80, scale=2.0)
                handle = model.model.layers[0].register_forward_hook(hook)
                torch.manual_seed(seed)
                with torch.no_grad():
                    out = model.generate(input_ids, max_new_tokens=80, temperature=0.8,
                                        do_sample=True, pad_token_id=tokenizer.eos_token_id)
                resp = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
                results["80_pairs"][emotion].append(count_markers(resp, emotion))
                handle.remove()

                # 800 pairs - target layer 0
                hook = SteeringHook(dir_800, scale=2.0)
                handle = model.model.layers[0].register_forward_hook(hook)
                torch.manual_seed(seed)
                with torch.no_grad():
                    out = model.generate(input_ids, max_new_tokens=80, temperature=0.8,
                                        do_sample=True, pad_token_id=tokenizer.eos_token_id)
                resp = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
                results["800_pairs"][emotion].append(count_markers(resp, emotion))
                handle.remove()

    # Results
    print("\n")
    print("=" * 70)
    print("RESULTS: AVERAGE EMOTION MARKERS")
    print("=" * 70)

    print(f"\n{'Condition':<12} {'Fear':<10} {'Curiosity':<12} {'Anger':<10} {'Joy':<10} {'Total':<10}")
    print("-" * 64)

    summary = {}
    for cond in ["baseline", "80_pairs", "800_pairs"]:
        summary[cond] = {}
        row = f"{cond:<12}"
        total = 0
        for e in emotions:
            m, _ = compute_stats(results[cond][e])
            summary[cond][e] = m
            total += m
            row += f"{m:<10.2f}"
        summary[cond]["total"] = total
        row += f"{total:<10.2f}"
        print(row)

    # Effect sizes
    print("\n")
    print("=" * 70)
    print("EFFECT SIZES (Cohen's d) vs BASELINE")
    print("=" * 70)

    print(f"\n{'Condition':<12} {'Fear':<10} {'Curiosity':<12} {'Anger':<10} {'Joy':<10} {'Avg |d|':<10}")
    print("-" * 64)

    for cond in ["80_pairs", "800_pairs"]:
        row = f"{cond:<12}"
        ds = []
        for e in emotions:
            d = cohens_d(results["baseline"][e], results[cond][e])
            ds.append(abs(d))
            row += f"{d:+.2f}      "
        row += f"{sum(ds)/len(ds):.2f}"
        print(row)

    # Comparison
    print("\n")
    print("=" * 70)
    print("800 PAIRS vs 80 PAIRS")
    print("=" * 70)

    wins_800 = 0
    wins_80 = 0

    for e in emotions:
        d_80 = abs(cohens_d(results["baseline"][e], results["80_pairs"][e]))
        d_800 = abs(cohens_d(results["baseline"][e], results["800_pairs"][e]))

        if d_800 > d_80 + 0.1:
            wins_800 += 1
            winner = "800 ✓"
        elif d_80 > d_800 + 0.1:
            wins_80 += 1
            winner = "80 ✓"
        else:
            winner = "TIE"

        print(f"  {e:12}: |d| 80={d_80:.2f}, 800={d_800:.2f} → {winner}")

    print(f"\n  Wins: 800_pairs={wins_800}, 80_pairs={wins_80}")

    # Verdict
    print("\n")
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    avg_d_80 = sum(abs(cohens_d(results["baseline"][e], results["80_pairs"][e])) for e in emotions) / 4
    avg_d_800 = sum(abs(cohens_d(results["baseline"][e], results["800_pairs"][e])) for e in emotions) / 4

    if avg_d_800 > 0.5:
        print(f"\n  ✅ 800 PAIRS PRODUCES MEDIUM+ EFFECTS (avg |d|={avg_d_800:.2f})")
    elif avg_d_800 > avg_d_80 + 0.1:
        print(f"\n  ⚪ 800 PAIRS IS BETTER BUT STILL WEAK (avg |d|={avg_d_800:.2f})")
    else:
        print(f"\n  ❌ MORE DATA DIDN'T HELP SIGNIFICANTLY (avg |d|={avg_d_800:.2f})")

    with open("data/large_dataset_validation.json", "w") as f:
        json.dump({"summary": summary, "wins_800": wins_800, "wins_80": wins_80,
                   "avg_d_80": avg_d_80, "avg_d_800": avg_d_800}, f, indent=2)

    print(f"\nResults saved to: data/large_dataset_validation.json")


if __name__ == "__main__":
    run_validation()
