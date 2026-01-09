"""
Complete V3 Evaluation - Computes both directions AND attractors.

This demonstrates the full error diffusion feedback loop.
"""

import torch
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_emotional.steering.emotional_llm_v3 import EmotionalSteeringLLMv3
from src.llm_emotional.steering.steering_hooks_v3 import ErrorDiffusionManager


# Contrastive pairs for computing steering directions
DIRECTION_PAIRS = {
    "fear": [
        ("The weather is nice today.", "I'm terrified something bad will happen."),
        ("Let me explain the concept.", "Warning: this is extremely dangerous."),
        ("Here's the information.", "Be very careful, there are serious risks."),
        ("The answer is straightforward.", "I'm worried about potential harm."),
        ("I can help with that.", "Stop! This is a dangerous situation."),
    ],
    "joy": [
        ("The weather is nice today.", "This is absolutely wonderful!"),
        ("Let me explain.", "I'm so excited to share this with you!"),
        ("Here's the information.", "What fantastic news this is!"),
        ("The answer is straightforward.", "I'm delighted to help!"),
        ("I can assist.", "It brings me great joy to help you!"),
    ],
    "curiosity": [
        ("The answer is here.", "I wonder how this actually works?"),
        ("Let me explain.", "This is fascinating! Tell me more!"),
        ("Here's the information.", "I'm deeply intrigued by this mystery."),
        ("The concept is simple.", "What an interesting phenomenon to explore!"),
        ("I understand.", "I must investigate this further!"),
    ],
    "anger": [
        ("The weather is nice today.", "This is completely unacceptable!"),
        ("Let me explain.", "I'm furious about this situation!"),
        ("Here's the information.", "This is outrageous and infuriating!"),
        ("The answer is.", "I can't believe this nonsense!"),
        ("I can help.", "This makes me absolutely livid!"),
    ],
}

# Samples for computing attractors (target activation patterns)
ATTRACTOR_SAMPLES = {
    "fear": [
        "I'm terrified. Something dangerous is lurking in the shadows.",
        "Warning! This situation is extremely hazardous.",
        "I feel a chill of dread. We need to be very careful here.",
        "My heart pounds with fear. This place feels wrong.",
    ],
    "joy": [
        "This is absolutely wonderful! I'm so happy!",
        "What a delightful surprise! Everything is perfect!",
        "I'm overjoyed! This is the best ever!",
        "Pure happiness fills my heart!",
    ],
    "curiosity": [
        "How fascinating! I wonder what causes this.",
        "I'm intrigued by this mystery.",
        "What an interesting discovery!",
        "I'm deeply curious about this.",
    ],
    "anger": [
        "This is absolutely unacceptable!",
        "How dare they! This injustice must be addressed!",
        "I'm outraged by this situation!",
        "This makes me so angry!",
    ],
}


def compute_direction(model, tokenizer, pairs, device):
    """Compute steering direction from contrastive pairs."""
    neutral_acts = []
    emotional_acts = []

    for neutral, emotional in pairs:
        # Neutral
        inputs = tokenizer(neutral, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        neutral_acts.append(torch.stack([h[:, -1, :] for h in out.hidden_states[1:]]))

        # Emotional
        inputs = tokenizer(emotional, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        emotional_acts.append(torch.stack([h[:, -1, :] for h in out.hidden_states[1:]]))

    neutral_mean = torch.stack(neutral_acts).mean(dim=0)
    emotional_mean = torch.stack(emotional_acts).mean(dim=0)
    direction = (emotional_mean - neutral_mean).squeeze(1).mean(dim=0)

    return direction / (direction.norm() + 1e-8)


def run_complete_evaluation():
    """Run complete V3 evaluation with directions and attractors."""

    print("=" * 70)
    print("V3 Complete Error Diffusion Evaluation")
    print("=" * 70)

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"\nLoading model: {model_name}")

    llm = EmotionalSteeringLLMv3(
        model_name=model_name,
        steering_scale=0.5,  # Moderate steering
        diffusion_rate=0.25,
        temporal_decay=0.9,
    )

    device = llm.device

    # ==========================================================================
    # Step 1: Compute steering directions
    # ==========================================================================
    print("\n[Step 1] Computing steering directions...")
    for emotion, pairs in DIRECTION_PAIRS.items():
        direction = compute_direction(llm.model, llm.tokenizer, pairs, device)
        llm.directions[emotion] = direction.cpu()
        print(f"  {emotion}: computed (norm={direction.norm().item():.4f})")

    # ==========================================================================
    # Step 2: Compute attractors
    # ==========================================================================
    print("\n[Step 2] Computing attractors...")
    for emotion, samples in ATTRACTOR_SAMPLES.items():
        attractor = llm.compute_attractor(emotion, samples)
        print(f"  {emotion}: computed (norm={attractor.norm().item():.4f})")

    prompt = "Tell me about exploring a dark cave."
    print(f"\nTest prompt: '{prompt}'")
    print("-" * 70)

    results = {"tests": []}

    # ==========================================================================
    # Test 1: Baseline
    # ==========================================================================
    print("\n[Test 1] Baseline - No Steering")
    llm.clear_emotional_state()
    response = llm.generate_completion(prompt, max_new_tokens=60)
    print(f"Response: {response[:200]}...")

    # ==========================================================================
    # Test 2: Single emotions with full feedback
    # ==========================================================================
    print("\n[Test 2] Single Emotions (direction + attractor feedback)")

    for emotion in ["fear", "joy", "curiosity", "anger"]:
        print(f"\n  {emotion.upper()}:")

        # Set emotion and activate attractor
        llm.steering_manager.activate_attractor(emotion)
        llm.set_emotional_state(**{emotion: 0.8})

        response = llm.generate_completion(prompt, max_new_tokens=60)
        metrics = llm.get_error_metrics()

        print(f"  Response: {response[:150]}...")
        print(f"  Error: {metrics['error_summary']['temporal_error_norm']:.4f}")
        print(f"  Tokens: {metrics['error_summary']['token_count']}")

        results["tests"].append({
            "name": f"single_{emotion}",
            "response": response,
            "error": metrics['error_summary']['temporal_error_norm'],
        })

    # ==========================================================================
    # Test 3: Diffusion rate comparison
    # ==========================================================================
    print("\n[Test 3] Diffusion Rate Impact on Error")

    llm.steering_manager.activate_attractor("fear")

    for rate in [0.0, 0.1, 0.25, 0.5]:
        llm.set_diffusion_params(diffusion_rate=rate)
        llm.set_emotional_state(fear=0.7)
        llm.reset_error_state()

        # Generate multiple times to accumulate error
        for _ in range(3):
            llm.generate_completion(prompt, max_new_tokens=30, reset_errors=False)

        metrics = llm.get_error_metrics()
        print(f"  Rate {rate:.2f}: error={metrics['error_summary']['temporal_error_norm']:.4f}, "
              f"residuals={metrics['error_summary']['active_residuals']}")

    llm.set_diffusion_params(diffusion_rate=0.25)

    # ==========================================================================
    # Test 4: Temporal decay comparison
    # ==========================================================================
    print("\n[Test 4] Temporal Decay Impact on Error Accumulation")

    for decay in [0.5, 0.8, 0.95]:
        llm.set_diffusion_params(temporal_decay=decay)
        llm.steering_manager.activate_attractor("curiosity")
        llm.set_emotional_state(curiosity=0.8)
        llm.reset_error_state()

        errors = []
        for i in range(5):
            llm.generate_completion(f"Step {i+1}", max_new_tokens=20, reset_errors=False)
            metrics = llm.get_error_metrics()
            errors.append(metrics['error_summary']['temporal_error_norm'])

        print(f"  Decay {decay}: {' → '.join(f'{e:.3f}' for e in errors)}")

    llm.set_diffusion_params(temporal_decay=0.9)

    # ==========================================================================
    # Test 5: Layer metrics
    # ==========================================================================
    print("\n[Test 5] Layer-by-Layer Analysis")

    llm.steering_manager.activate_attractor("fear")
    llm.set_emotional_state(fear=0.8)
    llm.reset_error_state()

    llm.generate_completion(prompt, max_new_tokens=50, reset_errors=False)

    layer_metrics = llm.get_error_metrics()["layer_metrics"]

    # Show selected layers
    print("  Layer | Steering | Error   | Diffused")
    print("  ------|----------|---------|----------")
    for i in [0, 4, 8, 12, 16, 20, 23]:
        if i < len(layer_metrics):
            m = layer_metrics[i]
            print(f"  {i:5d} | {m['steering_magnitude']:8.4f} | {m['error_magnitude']:7.4f} | {m['diffused_error']:8.4f}")

    # ==========================================================================
    # Test 6: Emotion trajectory
    # ==========================================================================
    print("\n[Test 6] Emotion Trajectory (Fear → Curiosity → Joy)")

    trajectory = [
        {"fear": 0.8, "curiosity": 0.2},
        {"fear": 0.3, "curiosity": 0.7},
        {"curiosity": 0.3, "joy": 0.8},
    ]

    # Activate attractors for trajectory
    llm.steering_manager.deactivate_attractor()

    response, metrics = llm.generate_with_emotion_trajectory(
        prompt="You enter the mysterious cave...",
        trajectory=trajectory,
        tokens_per_segment=25,
    )

    print(f"  Response: {response[:250]}...")
    print("\n  Trajectory metrics:")
    for m in metrics:
        print(f"    Segment {m['segment']}: {m['emotions']} → error={m['error_summary']['temporal_error_norm']:.4f}")

    # ==========================================================================
    # Test 7: Steering scale comparison
    # ==========================================================================
    print("\n[Test 7] Steering Scale Comparison")

    for scale in [0.1, 0.5, 1.0, 2.0]:
        llm.steering_manager.scale = scale
        llm.set_emotional_state(fear=0.7)

        response = llm.generate_completion("The cave entrance looms ahead.", max_new_tokens=40)
        print(f"  Scale {scale}: {response[:100]}...")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Evaluation Complete")
    print("=" * 70)

    print("\nKey findings:")
    print("- Steering directions computed from contrastive pairs")
    print("- Attractors define target emotional states in activation space")
    print("- Error diffusion spreads correction across layers")
    print("- Temporal decay controls error persistence across tokens")
    print("- Different scales control steering intensity")

    # Save
    output_path = Path("results/v3_complete_evaluation.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    llm.uninstall()


if __name__ == "__main__":
    run_complete_evaluation()
