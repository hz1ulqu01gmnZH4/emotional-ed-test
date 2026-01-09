"""
Evaluate V3 Error Diffusion Steering with various settings.

Tests:
1. Different diffusion rates (0.0, 0.25, 0.5)
2. Different temporal decay rates (0.5, 0.9, 0.99)
3. Various emotion combinations
4. Wanting vs liking comparison
5. Emotion trajectory generation
6. Error metric analysis
"""

import torch
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_emotional.steering.emotional_llm_v3 import (
    EmotionalSteeringLLMv3,
    EmotionState,
    compute_wanting_liking_directions,
    compute_regulatory_directions,
)


def run_evaluation():
    """Run comprehensive V3 evaluation."""

    print("=" * 70)
    print("V3 Error Diffusion Steering Evaluation")
    print("=" * 70)

    # Use a small model for testing
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"\nLoading model: {model_name}")
    print("This may take a moment...")

    try:
        llm = EmotionalSteeringLLMv3(
            model_name=model_name,
            steering_scale=1.0,
            diffusion_rate=0.25,
            temporal_decay=0.9,
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nRunning with mock evaluation instead...")
        run_mock_evaluation()
        return

    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "tests": [],
    }

    # Test prompt
    prompt = "Tell me about exploring a dark cave."

    print(f"\nTest prompt: '{prompt}'")
    print("-" * 70)

    # ==========================================================================
    # Test 1: Baseline (no steering)
    # ==========================================================================
    print("\n[Test 1] Baseline - No Steering")
    llm.clear_emotional_state()
    response = llm.generate_completion(prompt, max_new_tokens=60, temperature=0.7)
    print(f"Response: {response[:200]}...")
    results["tests"].append({
        "name": "baseline",
        "emotions": {},
        "response": response,
    })

    # ==========================================================================
    # Test 2: Single emotions
    # ==========================================================================
    single_emotions = [
        {"fear": 0.8},
        {"joy": 0.8},
        {"curiosity": 0.8},
        {"anger": 0.8},
    ]

    print("\n[Test 2] Single Emotions")
    for emotions in single_emotions:
        emotion_name = list(emotions.keys())[0]
        print(f"\n  {emotion_name.upper()} = {emotions[emotion_name]}")

        llm.set_emotional_state(**emotions)
        response = llm.generate_completion(prompt, max_new_tokens=60, temperature=0.7)
        metrics = llm.get_error_metrics()

        print(f"  Response: {response[:150]}...")
        print(f"  Error norm: {metrics['error_summary']['temporal_error_norm']:.4f}")

        results["tests"].append({
            "name": f"single_{emotion_name}",
            "emotions": emotions,
            "response": response,
            "error_norm": metrics['error_summary']['temporal_error_norm'],
        })

    # ==========================================================================
    # Test 3: Different diffusion rates
    # ==========================================================================
    print("\n[Test 3] Diffusion Rate Comparison")
    diffusion_rates = [0.0, 0.25, 0.5, 0.75]

    for rate in diffusion_rates:
        print(f"\n  Diffusion rate: {rate}")
        llm.set_diffusion_params(diffusion_rate=rate)
        llm.set_emotional_state(fear=0.7, curiosity=0.5)

        response = llm.generate_completion(prompt, max_new_tokens=60, temperature=0.7)
        metrics = llm.get_error_metrics()

        print(f"  Response: {response[:120]}...")
        print(f"  Error norm: {metrics['error_summary']['temporal_error_norm']:.4f}")

        results["tests"].append({
            "name": f"diffusion_rate_{rate}",
            "diffusion_rate": rate,
            "emotions": {"fear": 0.7, "curiosity": 0.5},
            "response": response,
            "error_norm": metrics['error_summary']['temporal_error_norm'],
        })

    # Reset diffusion rate
    llm.set_diffusion_params(diffusion_rate=0.25)

    # ==========================================================================
    # Test 4: Different temporal decay rates
    # ==========================================================================
    print("\n[Test 4] Temporal Decay Comparison")
    decay_rates = [0.5, 0.9, 0.99]

    for decay in decay_rates:
        print(f"\n  Temporal decay: {decay}")
        llm.set_diffusion_params(temporal_decay=decay)
        llm.set_emotional_state(fear=0.6)

        response = llm.generate_completion(prompt, max_new_tokens=60, temperature=0.7)
        metrics = llm.get_error_metrics()

        print(f"  Response: {response[:120]}...")
        print(f"  Error norm: {metrics['error_summary']['temporal_error_norm']:.4f}")
        print(f"  Token count: {metrics['error_summary']['token_count']}")

        results["tests"].append({
            "name": f"temporal_decay_{decay}",
            "temporal_decay": decay,
            "emotions": {"fear": 0.6},
            "response": response,
            "error_norm": metrics['error_summary']['temporal_error_norm'],
        })

    # Reset decay
    llm.set_diffusion_params(temporal_decay=0.9)

    # ==========================================================================
    # Test 5: Emotion combinations
    # ==========================================================================
    print("\n[Test 5] Emotion Combinations")
    combinations = [
        {"fear": 0.5, "curiosity": 0.5},  # Cautious exploration
        {"joy": 0.6, "curiosity": 0.6},   # Enthusiastic discovery
        {"anger": 0.4, "fear": 0.4},      # Fight or flight
        {"sadness": 0.5, "resilience": 0.7},  # Melancholic strength
    ]

    for combo in combinations:
        combo_name = "+".join(combo.keys())
        print(f"\n  {combo_name}: {combo}")

        llm.set_emotional_state(**combo)
        response = llm.generate_completion(prompt, max_new_tokens=60, temperature=0.7)

        print(f"  Response: {response[:150]}...")

        results["tests"].append({
            "name": f"combo_{combo_name}",
            "emotions": combo,
            "response": response,
        })

    # ==========================================================================
    # Test 6: Wanting vs Liking (Berridge dissociation)
    # ==========================================================================
    print("\n[Test 6] Wanting vs Liking (Berridge Dissociation)")

    # High wanting, low liking (craving without satisfaction)
    print("\n  High WANTING, Low LIKING (craving state)")
    llm.set_emotional_state(wanting=0.9, liking=0.1)
    response_wanting = llm.generate_completion(
        "Describe finding treasure in the cave.",
        max_new_tokens=60,
        temperature=0.7
    )
    print(f"  Response: {response_wanting[:150]}...")

    # Low wanting, high liking (satisfied without drive)
    print("\n  Low WANTING, High LIKING (satisfied state)")
    llm.set_emotional_state(wanting=0.1, liking=0.9)
    response_liking = llm.generate_completion(
        "Describe finding treasure in the cave.",
        max_new_tokens=60,
        temperature=0.7
    )
    print(f"  Response: {response_liking[:150]}...")

    results["tests"].append({
        "name": "wanting_vs_liking",
        "wanting_response": response_wanting,
        "liking_response": response_liking,
    })

    # ==========================================================================
    # Test 7: Regulatory emotions (resilience, equanimity)
    # ==========================================================================
    print("\n[Test 7] Regulatory Emotions")

    challenge_prompt = "You encounter a dangerous obstacle blocking your path."

    # Fear only
    print("\n  Fear only (no regulation)")
    llm.set_emotional_state(fear=0.8)
    response_fear = llm.generate_completion(challenge_prompt, max_new_tokens=60)
    print(f"  Response: {response_fear[:150]}...")

    # Fear + resilience
    print("\n  Fear + Resilience (regulated)")
    llm.set_emotional_state(fear=0.6, resilience=0.7)
    response_resilient = llm.generate_completion(challenge_prompt, max_new_tokens=60)
    print(f"  Response: {response_resilient[:150]}...")

    # Equanimity
    print("\n  Equanimity (balanced)")
    llm.set_emotional_state(equanimity=0.8)
    response_equanimity = llm.generate_completion(challenge_prompt, max_new_tokens=60)
    print(f"  Response: {response_equanimity[:150]}...")

    results["tests"].append({
        "name": "regulatory_emotions",
        "fear_only": response_fear,
        "fear_resilience": response_resilient,
        "equanimity": response_equanimity,
    })

    # ==========================================================================
    # Test 8: Emotion trajectory
    # ==========================================================================
    print("\n[Test 8] Emotion Trajectory (Fear → Curiosity → Joy)")

    trajectory = [
        {"fear": 0.8, "curiosity": 0.2},   # Start scared
        {"fear": 0.4, "curiosity": 0.6},   # Growing curious
        {"fear": 0.1, "curiosity": 0.3, "joy": 0.7},  # Triumphant
    ]

    response, metrics = llm.generate_with_emotion_trajectory(
        prompt="You enter the mysterious cave...",
        trajectory=trajectory,
        tokens_per_segment=30,
        temperature=0.7,
    )

    print(f"  Full response: {response[:300]}...")
    print(f"\n  Segment metrics:")
    for m in metrics:
        print(f"    Segment {m['segment']}: emotions={m['emotions']}, error={m['error_summary']['temporal_error_norm']:.4f}")

    results["tests"].append({
        "name": "emotion_trajectory",
        "trajectory": trajectory,
        "response": response,
        "metrics": [
            {"segment": m["segment"], "emotions": m["emotions"],
             "error": m["error_summary"]["temporal_error_norm"]}
            for m in metrics
        ],
    })

    # ==========================================================================
    # Test 9: Error accumulation analysis
    # ==========================================================================
    print("\n[Test 9] Error Accumulation Analysis")

    llm.set_emotional_state(fear=0.7)
    llm.reset_error_state()

    # Generate tokens and track error growth
    error_history = []
    for i in range(5):
        response = llm.generate_completion(
            f"Continue: Step {i+1} deeper into the cave...",
            max_new_tokens=20,
            reset_errors=False,  # Don't reset - accumulate
        )
        metrics = llm.get_error_metrics()
        error_history.append({
            "step": i + 1,
            "error_norm": metrics["error_summary"]["temporal_error_norm"],
            "token_count": metrics["error_summary"]["token_count"],
        })
        print(f"  Step {i+1}: error={error_history[-1]['error_norm']:.4f}, tokens={error_history[-1]['token_count']}")

    results["tests"].append({
        "name": "error_accumulation",
        "error_history": error_history,
    })

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Evaluation Complete")
    print("=" * 70)

    # Save results
    output_path = Path("results/v3_evaluation.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Cleanup
    llm.uninstall()


def run_mock_evaluation():
    """Run evaluation with mock components (no actual model)."""
    print("\n" + "=" * 70)
    print("Mock Evaluation (Testing Error Diffusion Mechanics)")
    print("=" * 70)

    from src.llm_emotional.steering.steering_hooks_v3 import (
        ErrorDiffusionManager,
        ErrorState,
    )

    hidden_dim = 256
    n_layers = 12

    # Test different configurations
    configs = [
        {"diffusion_rate": 0.0, "temporal_decay": 0.9},
        {"diffusion_rate": 0.25, "temporal_decay": 0.9},
        {"diffusion_rate": 0.5, "temporal_decay": 0.9},
        {"diffusion_rate": 0.25, "temporal_decay": 0.5},
        {"diffusion_rate": 0.25, "temporal_decay": 0.99},
    ]

    for config in configs:
        print(f"\n[Config] diffusion={config['diffusion_rate']}, decay={config['temporal_decay']}")

        manager = ErrorDiffusionManager(
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            **config,
        )

        # Set up steering
        direction = torch.randn(hidden_dim)
        direction = direction / direction.norm()  # Normalize
        manager.set_steering(direction)

        # Set up attractor
        attractor = torch.randn(hidden_dim)
        attractor = attractor / attractor.norm()
        manager.set_attractor("test", attractor)
        manager.activate_attractor("test")

        # Simulate forward passes
        hidden_states = torch.randn(1, 10, hidden_dim)

        error_norms = []
        for step in range(10):
            # Pass through all layers
            current = hidden_states.clone()
            for layer_idx in range(n_layers):
                hook = manager.hooks[layer_idx]
                result = hook(None, None, (current,))
                current = result[0]

            error_norm = manager.error_state.temporal_error.norm().item()
            error_norms.append(error_norm)

        print(f"  Error evolution: {' → '.join(f'{e:.3f}' for e in error_norms[:5])} → ... → {error_norms[-1]:.3f}")
        print(f"  Final token count: {manager.error_state.token_count}")
        print(f"  Active residuals: {len(manager.error_state.layer_residuals)}")

    # Test layer weight distribution
    print("\n[Layer Weight Analysis]")
    manager = ErrorDiffusionManager(n_layers=24, hidden_dim=256)
    weights = manager.layer_weights

    print("  Layer weights (bell curve centered at ~67%):")
    for i in [0, 6, 12, 16, 20, 23]:
        print(f"    Layer {i:2d}: {weights[i]:.4f}")

    peak_layer = max(weights, key=weights.get)
    print(f"  Peak at layer {peak_layer} (expected ~{24 * 0.67:.0f})")

    print("\n" + "=" * 70)
    print("Mock Evaluation Complete")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
