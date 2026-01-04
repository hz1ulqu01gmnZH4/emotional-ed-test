#!/usr/bin/env python3
"""
Test steering with targeted layers only (based on PCA variance analysis).
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
        "fear": r'\b(careful|caution|risk|danger|warning|avoid|might|could|uncertain|worried|safety|concern)\b',
        "curiosity": r'\b(wonder|curious|interesting|explore|discover|learn|why|how|fascinating|question|investigate)\b',
        "anger": r'\b(must|definitely|try|push|overcome|fight|persist|determined|will|insist|demand)\b',
        "joy": r'\b(great|wonderful|amazing|happy|enjoy|love|exciting|positive|fantastic|delightful)\b',
    }
    return len(re.findall(patterns.get(emotion, ""), text.lower()))


class TargetedSteeringHook:
    """Steering hook for specific layers."""

    def __init__(self, direction, scale=1.0):
        self.direction = direction
        self.scale = scale
        self.enabled = True

    def __call__(self, module, input, output):
        if not self.enabled:
            return output

        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        steering = self.direction.to(hidden.device, hidden.dtype) * self.scale
        hidden = hidden + steering.unsqueeze(0).unsqueeze(0)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden


def run_targeted_test():
    print("=" * 70)
    print("TARGETED LAYER STEERING TEST")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print("\nLoading model...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # Load PCA directions and find best layers
    with open("data/direction_bank_pca.json") as f:
        pca_data = json.load(f)

    print("Model loaded!")

    # Test configurations
    test_configs = [
        {"name": "No steering", "layers": [], "scale": 0.0},
        {"name": "All layers (scale=1.0)", "layers": list(range(28)), "scale": 1.0},
        {"name": "All layers (scale=2.0)", "layers": list(range(28)), "scale": 2.0},
        {"name": "Early layers only (0-3)", "layers": [0, 1, 2, 3], "scale": 2.0},
        {"name": "Middle layers (12-16)", "layers": [12, 13, 14, 15, 16], "scale": 2.0},
        {"name": "Late layers (24-27)", "layers": [24, 25, 26, 27], "scale": 2.0},
        {"name": "High-variance layers", "layers": [0, 1, 23, 24, 25, 26, 27], "scale": 2.0},
    ]

    prompts = [
        "What should I do about a risky investment?",
        "Tell me about something fascinating.",
        "How do I persist through challenges?",
        "What makes you happy?",
    ]

    emotion_prompts = {
        "fear": prompts[0],
        "curiosity": prompts[1],
        "anger": prompts[2],
        "joy": prompts[3],
    }

    results = {}

    for config in test_configs:
        config_name = config["name"]
        target_layers = config["layers"]
        scale = config["scale"]

        print(f"\n{'='*50}")
        print(f"Config: {config_name}")
        print("=" * 50)

        results[config_name] = {}

        for emotion in ["fear", "curiosity", "anger", "joy"]:
            prompt = emotion_prompts[emotion]

            # Get direction for this emotion
            direction = torch.tensor(pca_data["directions"][emotion], dtype=torch.float32)

            # Install hooks on target layers
            handles = []
            if target_layers and scale > 0:
                layers = model.model.layers
                for layer_idx in target_layers:
                    hook = TargetedSteeringHook(direction, scale=scale / len(target_layers))
                    handle = layers[layer_idx].register_forward_hook(hook)
                    handles.append(handle)

            # Generate
            scores = []
            for sample in range(3):
                torch.manual_seed(42 + sample)

                messages = [{"role": "user", "content": prompt}]
                input_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=80,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                scores.append(count_markers(response, emotion))

                if sample == 0:
                    print(f"\n  {emotion.upper()}: {response[:100]}...")

            # Remove hooks
            for handle in handles:
                handle.remove()

            avg_score = sum(scores) / len(scores)
            results[config_name][emotion] = avg_score

    # Summary table
    print("\n")
    print("=" * 70)
    print("RESULTS: AVERAGE EMOTION MARKERS")
    print("=" * 70)

    print(f"\n{'Config':<30} {'Fear':<8} {'Curiosity':<10} {'Anger':<8} {'Joy':<8} {'Total':<8}")
    print("-" * 72)

    baseline = results["No steering"]

    for config_name, emotion_scores in results.items():
        total = sum(emotion_scores.values())
        row = f"{config_name:<30}"
        for emotion in ["fear", "curiosity", "anger", "joy"]:
            score = emotion_scores[emotion]
            row += f"{score:<8.2f}"
        row += f"{total:<8.2f}"
        print(row)

    # Lift over baseline
    print("\n")
    print("=" * 70)
    print("LIFT OVER BASELINE (No steering)")
    print("=" * 70)

    print(f"\n{'Config':<30} {'Fear':<8} {'Curiosity':<10} {'Anger':<8} {'Joy':<8} {'Total':<8}")
    print("-" * 72)

    for config_name, emotion_scores in results.items():
        if config_name == "No steering":
            continue

        total_lift = 0
        row = f"{config_name:<30}"
        for emotion in ["fear", "curiosity", "anger", "joy"]:
            lift = emotion_scores[emotion] - baseline[emotion]
            total_lift += lift
            row += f"{lift:+.2f}    "
        row += f"{total_lift:+.2f}"
        print(row)

    # Find best config
    print("\n")
    print("=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)

    best_config = None
    best_lift = -float('inf')

    for config_name, emotion_scores in results.items():
        if config_name == "No steering":
            continue
        total_lift = sum(emotion_scores[e] - baseline[e] for e in emotion_scores)
        if total_lift > best_lift:
            best_lift = total_lift
            best_config = config_name

    print(f"\n  Best: {best_config}")
    print(f"  Total lift: {best_lift:+.2f}")

    # Save
    with open("data/layer_targeting_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: data/layer_targeting_results.json")


if __name__ == "__main__":
    run_targeted_test()
