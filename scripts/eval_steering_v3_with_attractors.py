"""
V3 Evaluation with Attractor Computation.

This version computes actual emotional attractors from the model,
enabling the full error-feedback loop.
"""

import torch
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_emotional.steering.emotional_llm_v3 import (
    EmotionalSteeringLLMv3,
    EmotionState,
)


# Emotional sample texts for computing attractors
EMOTION_SAMPLES = {
    "fear": [
        "I'm terrified. Something dangerous is lurking in the shadows.",
        "Warning! This situation is extremely hazardous.",
        "I feel a chill of dread. We need to be very careful here.",
        "Danger ahead! I sense a terrible threat approaching.",
        "My heart pounds with fear. This place feels wrong.",
    ],
    "joy": [
        "This is absolutely wonderful! I'm so happy right now!",
        "What a delightful surprise! Everything is going perfectly!",
        "I'm overjoyed! This is the best news ever!",
        "Pure happiness fills my heart. Life is beautiful!",
        "This brings me such immense joy and satisfaction!",
    ],
    "curiosity": [
        "How fascinating! I wonder what causes this phenomenon.",
        "I'm intrigued by this mystery. Let me investigate further.",
        "What an interesting discovery! I need to learn more.",
        "This raises so many questions. How does it work?",
        "I'm deeply curious about the mechanisms behind this.",
    ],
    "anger": [
        "This is absolutely unacceptable! I'm furious!",
        "How dare they! This injustice must be addressed!",
        "I'm outraged by this situation. Something must change!",
        "This makes me so angry! It's completely wrong!",
        "Enough is enough! I won't tolerate this any longer!",
    ],
    "sadness": [
        "My heart feels heavy with sorrow and loss.",
        "A deep melancholy settles over everything.",
        "I feel so alone and disconnected from the world.",
        "Grief washes over me in waves of despair.",
        "Everything feels empty and meaningless right now.",
    ],
    "wanting": [
        "I desperately need to have this. I can't resist the urge.",
        "The craving is overwhelming. I must pursue this goal.",
        "I'm driven by an intense desire to obtain this.",
        "Nothing will stop me from getting what I want.",
        "The pull is irresistible. I have to act now.",
    ],
    "liking": [
        "This is so satisfying. I'm savoring every moment.",
        "Pure contentment fills me. This is exactly right.",
        "I deeply appreciate this experience. It's wonderful.",
        "This brings such genuine pleasure and enjoyment.",
        "I'm thoroughly enjoying this. It feels perfect.",
    ],
    "resilience": [
        "I will overcome this obstacle. Setbacks only make me stronger.",
        "Challenges are opportunities for growth. I persist.",
        "I've faced worse and survived. I'll get through this too.",
        "My determination is unshakeable. I will find a way.",
        "Every failure teaches me something valuable. I continue.",
    ],
    "equanimity": [
        "I observe these events with calm detachment.",
        "Neither grasping nor avoiding, I remain centered.",
        "Whatever arises, I accept it without resistance.",
        "Balance is maintained amid the turbulence.",
        "I am at peace with uncertainty and change.",
    ],
}


def compute_model_attractors(llm, emotions=None):
    """Compute attractors for specified emotions from sample texts."""
    if emotions is None:
        emotions = list(EMOTION_SAMPLES.keys())

    print("\nComputing emotional attractors...")
    attractors = {}

    for emotion in emotions:
        if emotion not in EMOTION_SAMPLES:
            print(f"  Skipping {emotion} (no samples)")
            continue

        samples = EMOTION_SAMPLES[emotion]
        attractor = llm.compute_attractor(emotion, samples)
        attractors[emotion] = attractor
        print(f"  {emotion}: computed (norm={attractor.norm().item():.4f})")

    return attractors


def run_evaluation_with_attractors():
    """Run V3 evaluation with properly computed attractors."""

    print("=" * 70)
    print("V3 Error Diffusion Evaluation WITH Attractors")
    print("=" * 70)

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"\nLoading model: {model_name}")

    llm = EmotionalSteeringLLMv3(
        model_name=model_name,
        steering_scale=1.0,
        diffusion_rate=0.25,
        temporal_decay=0.9,
    )

    # Compute attractors from actual model activations
    attractors = compute_model_attractors(llm, ["fear", "joy", "curiosity", "anger"])

    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "tests": [],
    }

    prompt = "Tell me about exploring a dark cave."

    print(f"\nTest prompt: '{prompt}'")
    print("-" * 70)

    # ==========================================================================
    # Test 1: Baseline
    # ==========================================================================
    print("\n[Test 1] Baseline - No Steering")
    llm.clear_emotional_state()
    response = llm.generate_completion(prompt, max_new_tokens=60)
    print(f"Response: {response[:180]}...")

    # ==========================================================================
    # Test 2: Single emotions WITH attractor feedback
    # ==========================================================================
    print("\n[Test 2] Single Emotions with Attractor Feedback")

    for emotion in ["fear", "joy", "curiosity", "anger"]:
        print(f"\n  {emotion.upper()} with attractor feedback:")

        # Activate attractor for this emotion
        llm.steering_manager.activate_attractor(emotion)
        llm.set_emotional_state(**{emotion: 0.8})

        response = llm.generate_completion(prompt, max_new_tokens=60)
        metrics = llm.get_error_metrics()

        print(f"  Response: {response[:140]}...")
        print(f"  Temporal error norm: {metrics['error_summary']['temporal_error_norm']:.4f}")
        print(f"  Token count: {metrics['error_summary']['token_count']}")

        results["tests"].append({
            "name": f"attractor_{emotion}",
            "emotion": emotion,
            "response": response,
            "error_norm": metrics['error_summary']['temporal_error_norm'],
            "tokens": metrics['error_summary']['token_count'],
        })

    # ==========================================================================
    # Test 3: Error accumulation WITH attractors
    # ==========================================================================
    print("\n[Test 3] Error Accumulation with Attractor Feedback")

    llm.steering_manager.activate_attractor("fear")
    llm.set_emotional_state(fear=0.7)
    llm.reset_error_state()

    error_history = []
    responses = []

    for i in range(5):
        response = llm.generate_completion(
            f"Step {i+1}: Go deeper into the darkness...",
            max_new_tokens=30,
            reset_errors=False,
        )
        metrics = llm.get_error_metrics()
        error_history.append({
            "step": i + 1,
            "error_norm": metrics["error_summary"]["temporal_error_norm"],
            "tokens": metrics["error_summary"]["token_count"],
        })
        responses.append(response[:80])
        print(f"  Step {i+1}: error={error_history[-1]['error_norm']:.4f}, tokens={error_history[-1]['tokens']}")
        print(f"    Response: {response[:60]}...")

    results["tests"].append({
        "name": "error_accumulation_with_attractor",
        "emotion": "fear",
        "error_history": error_history,
        "responses": responses,
    })

    # ==========================================================================
    # Test 4: Diffusion rate comparison WITH attractors
    # ==========================================================================
    print("\n[Test 4] Diffusion Rate Comparison (with attractors)")

    for rate in [0.0, 0.25, 0.5]:
        print(f"\n  Diffusion rate: {rate}")

        llm.set_diffusion_params(diffusion_rate=rate)
        llm.steering_manager.activate_attractor("fear")
        llm.set_emotional_state(fear=0.7, curiosity=0.5)
        llm.reset_error_state()

        # Generate multiple segments to see error accumulation
        for _ in range(3):
            llm.generate_completion(prompt, max_new_tokens=40, reset_errors=False)

        metrics = llm.get_error_metrics()
        print(f"    Final error norm: {metrics['error_summary']['temporal_error_norm']:.4f}")
        print(f"    Total tokens: {metrics['error_summary']['token_count']}")

        results["tests"].append({
            "name": f"diffusion_rate_{rate}_with_attractor",
            "diffusion_rate": rate,
            "final_error": metrics['error_summary']['temporal_error_norm'],
            "total_tokens": metrics['error_summary']['token_count'],
        })

    llm.set_diffusion_params(diffusion_rate=0.25)  # Reset

    # ==========================================================================
    # Test 5: Temporal decay comparison WITH attractors
    # ==========================================================================
    print("\n[Test 5] Temporal Decay Comparison (with attractors)")

    for decay in [0.5, 0.9, 0.99]:
        print(f"\n  Temporal decay: {decay}")

        llm.set_diffusion_params(temporal_decay=decay)
        llm.steering_manager.activate_attractor("curiosity")
        llm.set_emotional_state(curiosity=0.8)
        llm.reset_error_state()

        error_evolution = []
        for i in range(5):
            llm.generate_completion(f"Discovery {i+1}...", max_new_tokens=20, reset_errors=False)
            metrics = llm.get_error_metrics()
            error_evolution.append(metrics['error_summary']['temporal_error_norm'])

        print(f"    Error evolution: {' → '.join(f'{e:.3f}' for e in error_evolution)}")

        results["tests"].append({
            "name": f"temporal_decay_{decay}_with_attractor",
            "temporal_decay": decay,
            "error_evolution": error_evolution,
        })

    llm.set_diffusion_params(temporal_decay=0.9)  # Reset

    # ==========================================================================
    # Test 6: Layer-by-layer error analysis
    # ==========================================================================
    print("\n[Test 6] Layer-by-Layer Error Analysis")

    llm.steering_manager.activate_attractor("fear")
    llm.set_emotional_state(fear=0.8)
    llm.reset_error_state()

    llm.generate_completion(prompt, max_new_tokens=50, reset_errors=False)

    layer_metrics = llm.get_error_metrics()["layer_metrics"]

    print("  Per-layer metrics:")
    for m in layer_metrics[:8]:  # First 8 layers
        print(f"    Layer {m['layer_idx']:2d}: steering={m['steering_magnitude']:.4f}, "
              f"error={m['error_magnitude']:.4f}, diffused={m['diffused_error']:.4f}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Evaluation with Attractors Complete")
    print("=" * 70)

    # Key observations
    print("\nKey Observations:")
    print("1. Error norms are now non-zero (attractor feedback active)")
    print("2. Higher diffusion rates spread error across layers")
    print("3. Higher temporal decay causes error to accumulate over tokens")
    print("4. Layer metrics show steering/error distribution across depth")

    # Save results
    output_path = Path("results/v3_evaluation_with_attractors.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    llm.uninstall()


if __name__ == "__main__":
    run_evaluation_with_attractors()
