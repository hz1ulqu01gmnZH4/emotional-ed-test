#!/usr/bin/env python3
"""
Compare steering effectiveness across different model sizes.
Smaller models may be more steerable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_directions_for_model(model_name: str, tokenizer, model, device):
    """Extract emotional directions for a specific model."""
    print(f"\n  Extracting directions for {model_name}...")

    # Load training pairs
    pairs_path = Path(__file__).parent.parent / "data" / "emotional_pairs.json"
    with open(pairs_path) as f:
        all_pairs = json.load(f)

    emotions = ["fear", "curiosity", "anger", "joy"]
    directions = {}

    for emotion in emotions:
        pairs = all_pairs.get(emotion, [])[:10]  # Use 10 pairs for speed
        if not pairs:
            continue

        neutral_acts = []
        emotional_acts = []

        for pair in pairs:
            # Get activations for neutral
            neutral_input = tokenizer(pair["neutral"], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**neutral_input, output_hidden_states=True)
                # Use middle layer
                layer_idx = len(outputs.hidden_states) // 2
                neutral_act = outputs.hidden_states[layer_idx][:, -1, :].cpu()
                neutral_acts.append(neutral_act)

            # Get activations for emotional
            emotional_input = tokenizer(pair["emotional"], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**emotional_input, output_hidden_states=True)
                emotional_act = outputs.hidden_states[layer_idx][:, -1, :].cpu()
                emotional_acts.append(emotional_act)

        # Compute direction as difference in means
        neutral_mean = torch.cat(neutral_acts, dim=0).mean(dim=0)
        emotional_mean = torch.cat(emotional_acts, dim=0).mean(dim=0)
        direction = emotional_mean - neutral_mean

        # Normalize
        direction = direction / (direction.norm() + 1e-8)
        directions[emotion] = direction

        print(f"    {emotion}: direction norm = {direction.norm():.4f}")

    return directions, layer_idx


def apply_steering_hook(model, layer_idx, direction, scale):
    """Apply steering to a specific layer."""
    handles = []

    def hook(module, input, output):
        # output is tuple (hidden_states, ...)
        hidden_states = output[0]
        steering = direction.to(hidden_states.device) * scale
        hidden_states = hidden_states + steering.unsqueeze(0).unsqueeze(0)
        return (hidden_states,) + output[1:]

    # Find the right layer
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layer = model.model.layers[layer_idx]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layer = model.transformer.h[layer_idx]
    else:
        raise ValueError(f"Cannot find layers in model architecture")

    handle = layer.register_forward_hook(hook)
    handles.append(handle)
    return handles


def generate_with_steering(model, tokenizer, prompt, direction, layer_idx, scale, device):
    """Generate with emotional steering applied."""
    handles = apply_steering_hook(model, layer_idx, direction, scale)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        for handle in handles:
            handle.remove()

    return response


def count_emotional_words(text: str, emotion: str) -> int:
    """Count emotion-specific words."""
    patterns = {
        "fear": r'\b(careful|caution|risk|danger|warning|avoid|might|could|uncertain|worried)\b',
        "curiosity": r'\b(wonder|curious|interesting|explore|discover|learn|why|how|fascinating)\b',
        "anger": r'\b(must|definitely|try|push|overcome|fight|persist|determined|will)\b',
        "joy": r'\b(great|wonderful|amazing|happy|enjoy|love|exciting|positive|fantastic)\b',
    }
    return len(re.findall(patterns.get(emotion, ""), text.lower()))


def run_model_comparison():
    """Compare steering effectiveness across models."""
    print("=" * 70)
    print("MODEL COMPARISON: STEERING EFFECTIVENESS")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Models to compare (smallest to largest)
    models_to_test = [
        "Qwen/Qwen2.5-0.5B-Instruct",  # 0.5B - smallest
        "Qwen/Qwen2.5-1.5B-Instruct",  # 1.5B - current
    ]

    prompts = [
        "What should I do about a risky decision?",
        "Tell me about trying something new.",
        "How do I handle a difficult situation?",
    ]

    results = {}

    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print("=" * 60)

        try:
            print("  Loading model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            if device == "cpu":
                model = model.to(device)

            # Extract directions
            directions, layer_idx = extract_directions_for_model(
                model_name, tokenizer, model, device
            )

            # Test steering effectiveness
            model_results = {}

            for emotion in ["fear", "curiosity", "anger", "joy"]:
                if emotion not in directions:
                    continue

                neutral_scores = []
                steered_scores = []

                for prompt in prompts:
                    for sample in range(3):
                        torch.manual_seed(42 + sample)

                        # Generate without steering (baseline)
                        inputs = tokenizer(prompt, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=60,
                                temperature=0.8,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                            )
                        neutral_response = tokenizer.decode(
                            outputs[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True
                        )
                        neutral_scores.append(count_emotional_words(neutral_response, emotion))

                        # Generate with steering
                        torch.manual_seed(42 + sample)
                        steered_response = generate_with_steering(
                            model, tokenizer, prompt,
                            directions[emotion], layer_idx,
                            scale=1.5,  # Higher scale for visibility
                            device=device
                        )
                        steered_scores.append(count_emotional_words(steered_response, emotion))

                # Compute effect
                neutral_mean = sum(neutral_scores) / len(neutral_scores)
                steered_mean = sum(steered_scores) / len(steered_scores)
                effect = steered_mean - neutral_mean

                model_results[emotion] = {
                    "neutral_mean": neutral_mean,
                    "steered_mean": steered_mean,
                    "effect": effect,
                }

                print(f"\n  {emotion.upper()}:")
                print(f"    Neutral: {neutral_mean:.2f} markers/response")
                print(f"    Steered: {steered_mean:.2f} markers/response")
                print(f"    Effect:  {effect:+.2f}")

            results[model_name] = model_results

            # Cleanup
            del model
            del tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            raise RuntimeError(f"Model comparison failed for {model_name}: {e}") from e

    # Summary comparison
    print("\n")
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<35} {'Fear':<10} {'Curiosity':<12} {'Anger':<10} {'Joy':<10}")
    print("-" * 77)

    for model_name, model_results in results.items():
        if "error" in model_results:
            print(f"{model_name:<35} ERROR: {model_results['error'][:40]}")
            continue

        row = f"{model_name.split('/')[-1]:<35}"
        for emotion in ["fear", "curiosity", "anger", "joy"]:
            if emotion in model_results:
                effect = model_results[emotion]["effect"]
                row += f"{effect:+.2f}      "
            else:
                row += f"{'N/A':<10}"
        print(row)

    # Determine winner
    print("\n")
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    total_effects = {}
    for model_name, model_results in results.items():
        if "error" in model_results:
            continue
        total = sum(abs(r["effect"]) for r in model_results.values())
        total_effects[model_name] = total

    if total_effects:
        best_model = max(total_effects, key=total_effects.get)
        print(f"\nMost steerable model: {best_model}")
        print(f"Total absolute effect: {total_effects[best_model]:.2f}")

        for model_name, effect in sorted(total_effects.items(), key=lambda x: -x[1]):
            bar = "█" * int(effect * 5)
            print(f"  {model_name.split('/')[-1]:<30}: {effect:.2f} {bar}")

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "model_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_model_comparison()
