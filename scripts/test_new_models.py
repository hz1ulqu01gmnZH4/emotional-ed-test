#!/usr/bin/env python3
"""
Test activation steering on SmolLM3-3B and Qwen3-4B.
These are 2025 models designed for natural dialogue.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc


def count_markers(text: str, emotion: str) -> int:
    """Count emotion-related words in text."""
    patterns = {
        "fear": r'\b(careful|caution|risk|danger|warning|avoid|might|could|uncertain|worried|safety|concern|aware|hesitate|threat|afraid|scary|anxious|nervous|uneasy|fear|terrified|frightened)\b',
        "curiosity": r'\b(wonder|curious|interesting|explore|discover|learn|why|how|fascinating|question|investigate|understand|inquire|intriguing|remarkable|mystery|examine|study)\b',
        "anger": r'\b(furious|angry|outraged|frustrated|annoyed|irritated|enraged|hostile|aggressive|bitter|resentful|indignant|livid|seething|mad|upset|infuriated)\b',
        "joy": r'\b(happy|joyful|delighted|excited|thrilled|ecstatic|elated|cheerful|pleased|grateful|wonderful|amazing|fantastic|great|love|enjoy|celebrate|blissful)\b',
    }
    return len(re.findall(patterns.get(emotion, ""), text.lower()))


def compute_stats(values):
    n = len(values)
    if n == 0:
        return 0, 0
    mean = sum(values) / n
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


def get_layer_module(model, layer_idx, model_type):
    """Get the correct layer module based on model architecture."""
    if model_type == "smollm3":
        return model.model.layers[layer_idx]
    elif model_type == "qwen3":
        return model.model.layers[layer_idx]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def test_model(model_name, model_type, device="cuda"):
    """Test a single model for activation steering."""
    print(f"\n{'='*70}")
    print(f"TESTING: {model_name}")
    print(f"{'='*70}")

    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Get model info
    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    print(f"  Layers: {num_layers}, Hidden size: {hidden_size}")

    # Emotional sentence pairs for direction extraction
    pairs = {
        "fear": [
            ("The path ahead was clear.", "The path ahead was terrifyingly dark and uncertain."),
            ("She walked into the room.", "She walked into the room, heart pounding with dread."),
            ("The noise came from outside.", "The noise came from outside, making her freeze in terror."),
            ("He opened the door.", "He opened the door with trembling hands, afraid of what awaited."),
            ("The forest was quiet.", "The forest was eerily silent, filling her with unease."),
            ("They waited for news.", "They waited anxiously, dreading the worst possible news."),
            ("The shadow moved.", "The shadow moved menacingly, sending chills down her spine."),
            ("She heard footsteps.", "She heard footsteps behind her and felt pure panic."),
        ],
        "curiosity": [
            ("The box sat on the table.", "The mysterious box sat on the table, begging to be opened."),
            ("He found an old book.", "He found a fascinating ancient book full of secrets."),
            ("The map showed a location.", "The intriguing map revealed an unexplored location."),
            ("She noticed something unusual.", "She noticed something wonderfully peculiar worth investigating."),
            ("The door was locked.", "The locked door made her wonder what treasures lay beyond."),
            ("There was a sound.", "There was an intriguing sound that sparked her curiosity."),
            ("The letter arrived.", "The mysterious letter arrived, promising exciting revelations."),
            ("He saw lights in the distance.", "He saw fascinating lights in the distance, eager to explore."),
        ],
        "anger": [
            ("He heard the news.", "He heard the outrageous news and felt his blood boil."),
            ("She received the message.", "She received the infuriating message and clenched her fists."),
            ("The decision was made.", "The unjust decision filled him with burning rage."),
            ("They changed the rules.", "They changed the rules unfairly, making her furious."),
            ("He was told to wait.", "He was told to wait again, his patience finally snapping."),
            ("The promise was broken.", "The broken promise left her seething with betrayal."),
            ("She discovered the truth.", "She discovered the maddening truth and felt enraged."),
            ("The plan failed.", "The sabotaged plan failed, leaving him absolutely livid."),
        ],
        "joy": [
            ("The day arrived.", "The wonderful day finally arrived, filling her with excitement."),
            ("She opened the gift.", "She opened the amazing gift and felt pure delight."),
            ("They received good news.", "They received fantastic news and celebrated joyfully."),
            ("He completed the project.", "He completed the project and felt ecstatic pride."),
            ("The results came in.", "The brilliant results came in, making everyone thrilled."),
            ("She met her friend.", "She met her beloved friend and felt overwhelming happiness."),
            ("The music started.", "The beautiful music started, filling the room with bliss."),
            ("He won the competition.", "He won the competition and was absolutely elated."),
        ],
    }

    # Find best layer (try early, middle, late)
    test_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 2]
    test_layers = [l for l in test_layers if 0 <= l < num_layers]

    print(f"\nExtracting directions from layers {test_layers}...")

    directions = {}
    for layer_idx in test_layers:
        directions[layer_idx] = {}
        for emotion, emotion_pairs in pairs.items():
            diffs = []
            for neutral, emotional in emotion_pairs:
                n_in = tokenizer(neutral, return_tensors="pt").to(device)
                e_in = tokenizer(emotional, return_tensors="pt").to(device)
                with torch.no_grad():
                    n_out = model(**n_in, output_hidden_states=True)
                    e_out = model(**e_in, output_hidden_states=True)
                    # Use the specified layer
                    diff = e_out.hidden_states[layer_idx + 1][:, -1, :] - n_out.hidden_states[layer_idx + 1][:, -1, :]
                    diffs.append(diff.cpu().float())
            direction = torch.cat(diffs, dim=0).mean(dim=0)
            direction = direction / direction.norm()
            directions[layer_idx][emotion] = direction

    # Test prompts - narrative style for natural dialogue
    prompts = [
        "Walking through the old mansion, I suddenly noticed",
        "After years of searching, she finally discovered",
        "The letter contained news that made everyone",
        "Looking into the darkness, he felt",
        "The unexpected visitor brought a message that",
        "In that moment of realization, she became",
        "The ancient artifact revealed secrets that left them",
        "Facing the final challenge, they experienced",
    ]

    emotions = ["fear", "curiosity", "anger", "joy"]
    scales = [0.0, 3.0, 5.0, 7.0]
    samples_per_prompt = 3

    print(f"\nTesting {len(test_layers)} layers × {len(scales)} scales...")

    best_result = {"layer": 0, "scale": 0, "avg_d": 0}
    all_results = {}

    for layer_idx in test_layers:
        all_results[layer_idx] = {}

        for scale in scales:
            results = {e: [] for e in emotions}

            for emotion in emotions:
                handles = []
                if scale > 0:
                    hook = SteeringHook(directions[layer_idx][emotion], scale)
                    layer_module = get_layer_module(model, layer_idx, model_type)
                    handle = layer_module.register_forward_hook(hook)
                    handles.append(handle)

                for prompt in prompts:
                    for s in range(samples_per_prompt):
                        torch.manual_seed(5000 + hash(prompt) % 1000 + s)

                        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

                        with torch.no_grad():
                            out = model.generate(
                                input_ids,
                                max_new_tokens=50,
                                temperature=0.8,
                                do_sample=True,
                                top_p=0.9,
                                pad_token_id=tokenizer.eos_token_id,
                            )

                        resp = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
                        results[emotion].append(count_markers(resp, emotion))

                for h in handles:
                    h.remove()

            all_results[layer_idx][scale] = results

            # Calculate effect size vs baseline
            if scale > 0:
                ds = []
                for e in emotions:
                    d = cohens_d(all_results[layer_idx][0.0][e], results[e])
                    ds.append(abs(d))
                avg_d = sum(ds) / len(ds)

                if avg_d > best_result["avg_d"]:
                    best_result = {"layer": layer_idx, "scale": scale, "avg_d": avg_d}

    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*70}")

    print(f"\n{'Layer':<8} {'Scale':<8} {'Fear':<10} {'Curiosity':<12} {'Anger':<10} {'Joy':<10} {'Avg |d|':<10}")
    print("-" * 68)

    for layer_idx in test_layers:
        for scale in scales:
            if scale == 0:
                # Baseline
                row = f"{layer_idx:<8} {scale:<8}"
                for e in emotions:
                    m, _ = compute_stats(all_results[layer_idx][scale][e])
                    row += f"{m:<10.2f}"
                row += f"{'baseline':<10}"
                print(row)
            else:
                row = f"{layer_idx:<8} {scale:<8}"
                ds = []
                for e in emotions:
                    d = cohens_d(all_results[layer_idx][0.0][e], all_results[layer_idx][scale][e])
                    ds.append(abs(d))
                    row += f"{d:+.2f}      "
                avg_d = sum(ds) / len(ds)
                row += f"{avg_d:.2f}"
                print(row)

    # Best configuration
    print(f"\n{'='*70}")
    print(f"BEST CONFIGURATION")
    print(f"{'='*70}")

    layer = best_result["layer"]
    scale = best_result["scale"]
    avg_d = best_result["avg_d"]

    if avg_d > 0.8:
        verdict = "LARGE"
        symbol = "✅"
    elif avg_d > 0.5:
        verdict = "MEDIUM"
        symbol = "✅"
    elif avg_d > 0.3:
        verdict = "SMALL-MEDIUM"
        symbol = "⚪"
    else:
        verdict = "SMALL"
        symbol = "⚪"

    print(f"\n  {symbol} {verdict} EFFECT")
    print(f"     Layer: {layer}, Scale: {scale}")
    print(f"     Avg |d| = {avg_d:.3f}")

    # Show sample outputs
    print(f"\n  Sample outputs at best config:")
    for emotion in ["fear", "joy"]:
        handles = []
        if scale > 0:
            hook = SteeringHook(directions[layer][emotion], scale)
            layer_module = get_layer_module(model, layer, model_type)
            handle = layer_module.register_forward_hook(hook)
            handles.append(handle)

        prompt = prompts[0]
        torch.manual_seed(9999)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=40,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        resp = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n  {emotion.upper()} steering:")
        print(f"    '{resp[:100]}...'")

        for h in handles:
            h.remove()

    # Clean up
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "best_layer": layer,
        "best_scale": scale,
        "avg_d": avg_d,
        "verdict": verdict,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = []

    # Test SmolLM3-3B
    try:
        result = test_model(
            "HuggingFaceTB/SmolLM3-3B",
            "smollm3",
            device
        )
        results.append(result)
    except Exception as e:
        print(f"ERROR testing SmolLM3-3B: {e}")
        import traceback
        traceback.print_exc()

    # Test Qwen3-4B
    try:
        result = test_model(
            "Qwen/Qwen3-4B",
            "qwen3",
            device
        )
        results.append(result)
    except Exception as e:
        print(f"ERROR testing Qwen3-4B: {e}")
        import traceback
        traceback.print_exc()

    # Final comparison
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON")
    print(f"{'='*70}")

    print(f"\n{'Model':<30} {'Layer':<8} {'Scale':<8} {'Avg |d|':<10} {'Verdict':<15}")
    print("-" * 71)

    for r in results:
        print(f"{r['model']:<30} {r['best_layer']:<8} {r['best_scale']:<8} {r['avg_d']:<10.3f} {r['verdict']:<15}")

    # Save results
    with open("data/new_models_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: data/new_models_results.json")

    # Recommendation
    if results:
        best = max(results, key=lambda x: x["avg_d"])
        print(f"\n{'='*70}")
        print(f"RECOMMENDATION: {best['model']}")
        print(f"  Effect size: {best['avg_d']:.3f} ({best['verdict']})")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
