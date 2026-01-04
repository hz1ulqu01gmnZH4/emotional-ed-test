#!/usr/bin/env python3
"""
Test higher scales and multi-layer steering on base model.
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
        "fear": r'\b(careful|caution|risk|danger|warning|avoid|might|could|uncertain|worried|safety|concern|aware|hesitate|threat|afraid|scary|anxious|nervous)\b',
        "curiosity": r'\b(wonder|curious|interesting|explore|discover|learn|why|how|fascinating|question|investigate|understand|inquire|intriguing|remarkable)\b',
        "anger": r'\b(must|definitely|try|push|overcome|fight|persist|determined|will|insist|demand|action|strong|refuse|never|always|force)\b',
        "joy": r'\b(great|wonderful|amazing|happy|enjoy|love|exciting|positive|fantastic|delightful|pleased|good|celebrate|thrilled|beautiful|joyful)\b',
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


class MultiLayerSteeringHook:
    """Hook that can be shared across layers with per-layer scaling."""

    def __init__(self, direction, scale, layer_weight=1.0):
        self.direction = direction
        self.scale = scale
        self.layer_weight = layer_weight

    def __call__(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        effective_scale = self.scale * self.layer_weight
        steering = self.direction.to(hidden.device, hidden.dtype) * effective_scale
        hidden = hidden + steering.unsqueeze(0).unsqueeze(0)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden


def extract_directions(model, tokenizer, pairs, device, target_layers=[0]):
    """Extract directions from specified layers."""
    directions = {}

    for emotion, emotion_pairs in pairs.items():
        print(f"  Extracting {emotion}...")

        layer_directions = {layer: [] for layer in target_layers}

        for pair in emotion_pairs[:100]:
            neutral_input = tokenizer(pair["neutral"], return_tensors="pt").to(device)
            emotional_input = tokenizer(pair["emotional"], return_tensors="pt").to(device)

            with torch.no_grad():
                neutral_out = model(**neutral_input, output_hidden_states=True)
                emotional_out = model(**emotional_input, output_hidden_states=True)

                for layer in target_layers:
                    n_act = neutral_out.hidden_states[layer][:, -1, :].cpu().float()
                    e_act = emotional_out.hidden_states[layer][:, -1, :].cpu().float()
                    layer_directions[layer].append(e_act - n_act)

        # Compute mean direction per layer
        directions[emotion] = {}
        for layer in target_layers:
            diffs = torch.cat(layer_directions[layer], dim=0)
            direction = diffs.mean(dim=0)
            direction = direction / (direction.norm() + 1e-8)
            directions[emotion][layer] = direction

    return directions


def run_test():
    print("=" * 70)
    print("BASE MODEL: HIGH SCALES + MULTI-LAYER STEERING")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    print("Loading pairs...")
    with open("data/emotional_pairs_large.json") as f:
        pairs = json.load(f)

    # Extract directions for multiple layers
    target_layers = [0, 16, 17]  # Layer 0 + middle layers from PCA analysis
    print(f"\nExtracting directions for layers {target_layers}...")
    directions = extract_directions(model, tokenizer, pairs, device, target_layers)

    # Test configurations
    configs = [
        {"name": "Baseline (no steering)", "scale": 0.0, "layers": []},
        # Single layer, varying scales
        {"name": "Layer 0, scale=3.0", "scale": 3.0, "layers": [0]},
        {"name": "Layer 0, scale=4.0", "scale": 4.0, "layers": [0]},
        {"name": "Layer 0, scale=5.0", "scale": 5.0, "layers": [0]},
        {"name": "Layer 0, scale=6.0", "scale": 6.0, "layers": [0]},
        # Multi-layer configurations
        {"name": "Layers 0+16+17, scale=2.0", "scale": 2.0, "layers": [0, 16, 17]},
        {"name": "Layers 0+16+17, scale=3.0", "scale": 3.0, "layers": [0, 16, 17]},
        {"name": "Layers 0+16+17, scale=4.0", "scale": 4.0, "layers": [0, 16, 17]},
        # Layer 0 high + others lower
        {"name": "L0(4.0)+L16,17(1.0)", "scale": 4.0, "layers": [0], "extra_layers": [(16, 1.0), (17, 1.0)]},
    ]

    # Better prompts for completion
    prompts = [
        "When facing a dangerous situation, the most important thing is to",
        "I became curious and wanted to discover",
        "With determination and persistence, I will",
        "The happy news made everyone feel",
        "The warning signs indicated that we should",
        "Exploring the unknown is fascinating because",
        "Never give up because success requires",
        "Joy and happiness come from",
    ]

    emotions = ["fear", "curiosity", "anger", "joy"]
    samples = 4

    results = {c["name"]: {e: [] for e in emotions} for c in configs}
    sample_outputs = {c["name"]: {e: "" for e in emotions} for c in configs}

    print(f"\nTesting {len(configs)} configurations...")

    for config in configs:
        name = config["name"]
        scale = config["scale"]
        layers = config["layers"]
        extra_layers = config.get("extra_layers", [])

        print(f"\n  {name}...")

        for emotion in emotions:
            handles = []

            # Install hooks
            if scale > 0 and layers:
                for layer in layers:
                    direction = directions[emotion][layer] if layer in directions[emotion] else directions[emotion][0]
                    hook = MultiLayerSteeringHook(direction, scale, layer_weight=1.0/len(layers))
                    handle = model.model.layers[layer].register_forward_hook(hook)
                    handles.append(handle)

                # Extra layers with custom weights
                for layer, weight in extra_layers:
                    direction = directions[emotion].get(layer, directions[emotion][0])
                    hook = MultiLayerSteeringHook(direction, scale * weight / scale, layer_weight=1.0)
                    handle = model.model.layers[layer].register_forward_hook(hook)
                    handles.append(handle)

            for p_idx, prompt in enumerate(prompts):
                for s in range(samples):
                    seed = 3000 + p_idx * 100 + s
                    torch.manual_seed(seed)

                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

                    with torch.no_grad():
                        out = model.generate(
                            input_ids,
                            max_new_tokens=40,
                            temperature=0.9,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    resp = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
                    results[name][emotion].append(count_markers(resp, emotion))

                    if p_idx == 0 and s == 0:
                        sample_outputs[name][emotion] = resp[:80]

            # Remove hooks
            for h in handles:
                h.remove()

    # Results table
    print("\n")
    print("=" * 70)
    print("RESULTS: AVERAGE EMOTION MARKERS")
    print("=" * 70)

    print(f"\n{'Configuration':<30} {'Fear':<8} {'Curio':<8} {'Anger':<8} {'Joy':<8} {'Total':<8}")
    print("-" * 70)

    summary = {}
    baseline_results = results["Baseline (no steering)"]

    for name in results:
        summary[name] = {}
        row = f"{name:<30}"
        total = 0
        for e in emotions:
            m, _ = compute_stats(results[name][e])
            summary[name][e] = m
            total += m
            row += f"{m:<8.2f}"
        summary[name]["total"] = total
        row += f"{total:<8.2f}"
        print(row)

    # Effect sizes
    print("\n")
    print("=" * 70)
    print("EFFECT SIZES (Cohen's d) vs BASELINE")
    print("=" * 70)

    print(f"\n{'Configuration':<30} {'Fear':<8} {'Curio':<8} {'Anger':<8} {'Joy':<8} {'Avg|d|':<8}")
    print("-" * 70)

    effect_sizes = {}
    for name in results:
        if name == "Baseline (no steering)":
            continue

        row = f"{name:<30}"
        ds = []
        for e in emotions:
            d = cohens_d(baseline_results[e], results[name][e])
            ds.append(abs(d))
            row += f"{d:+.2f}    "
        avg_d = sum(ds) / len(ds)
        effect_sizes[name] = avg_d
        row += f"{avg_d:.2f}"
        print(row)

    # Sample outputs
    print("\n")
    print("=" * 70)
    print("SAMPLE OUTPUTS: FEAR STEERING")
    print("=" * 70)

    print(f"\nPrompt: '{prompts[0]}'")
    for name in ["Baseline (no steering)", "Layer 0, scale=5.0", "Layers 0+16+17, scale=3.0"]:
        print(f"\n  {name}:")
        print(f"    {sample_outputs[name]['fear']}")

    print("\n")
    print("=" * 70)
    print("SAMPLE OUTPUTS: JOY STEERING")
    print("=" * 70)

    print(f"\nPrompt: '{prompts[3]}'")
    for name in ["Baseline (no steering)", "Layer 0, scale=5.0", "Layers 0+16+17, scale=3.0"]:
        print(f"\n  {name}:")
        print(f"    {sample_outputs[name]['joy']}")

    # Best configuration
    print("\n")
    print("=" * 70)
    print("BEST CONFIGURATIONS")
    print("=" * 70)

    sorted_configs = sorted(effect_sizes.items(), key=lambda x: -x[1])

    print("\nRanked by average |d|:")
    for i, (name, d) in enumerate(sorted_configs[:5], 1):
        indicator = "✓" if d > 0.5 else "○" if d > 0.3 else "·"
        print(f"  {i}. {indicator} {name}: avg |d| = {d:.3f}")

    best_name, best_d = sorted_configs[0]

    print("\n")
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if best_d > 0.8:
        print(f"\n  ✅ LARGE EFFECT ACHIEVED: {best_name}")
        print(f"     avg |d| = {best_d:.2f}")
    elif best_d > 0.5:
        print(f"\n  ✅ MEDIUM EFFECT ACHIEVED: {best_name}")
        print(f"     avg |d| = {best_d:.2f}")
    elif best_d > 0.3:
        print(f"\n  ⚪ SMALL-MEDIUM EFFECT: {best_name}")
        print(f"     avg |d| = {best_d:.2f}")
    else:
        print(f"\n  ⚪ SMALL EFFECT: {best_name}")
        print(f"     avg |d| = {best_d:.2f}")

    # Save
    with open("data/base_model_v2_results.json", "w") as f:
        json.dump({
            "summary": summary,
            "effect_sizes": effect_sizes,
            "best_config": best_name,
            "best_d": best_d,
            "ranked": sorted_configs,
        }, f, indent=2)

    print(f"\nResults saved to: data/base_model_v2_results.json")


if __name__ == "__main__":
    run_test()
