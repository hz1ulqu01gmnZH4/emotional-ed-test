#!/usr/bin/env python3
"""
Demo: Steering Memory - External Behavioral Memory for LLMs

This demonstrates how steering vectors can be used as external memory:
1. Pre-compute vectors for different behaviors (fear, joy, formal, casual)
2. Store them in a vector database
3. Retrieve and apply at inference time
4. Compose multiple behaviors (e.g., fear + formal)
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.steering_memory import (
    SteeringMemory,
    SteeringVector,
    SteeringLLM,
    compute_steering_vector,
    EMOTION_PAIRS,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    print("=" * 70)
    print("STEERING MEMORY: External Behavioral Memory for LLMs")
    print("=" * 70)
    sys.stdout.flush()

    # =========================================================================
    # 1. Load Base Model
    # =========================================================================
    print("\n1. Loading base model...")
    sys.stdout.flush()

    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"   Model: {model_name}")
    print(f"   Device: {device}")

    # =========================================================================
    # 2. Create Steering Memory
    # =========================================================================
    print("\n2. Creating steering memory bank...")
    sys.stdout.flush()

    storage_path = Path("data/steering_memory")
    memory = SteeringMemory(storage_path)

    # Compute and store behavioral vectors
    behaviors = ["fear", "joy", "formal", "casual", "cautious"]

    for behavior in behaviors:
        if behavior in memory.vectors:
            print(f"   ✓ {behavior} (loaded from disk)")
            continue

        print(f"   Computing '{behavior}' vector...")
        sys.stdout.flush()

        pairs = EMOTION_PAIRS[behavior]
        vec = compute_steering_vector(
            model, tokenizer, pairs,
            name=behavior,
            description=f"Steering vector for {behavior} behavior",
            tags=["emotion" if behavior in ["fear", "joy"] else "style"],
        )
        memory.add(vec, persist=True)
        print(f"   ✓ {behavior} (computed and saved)")

    print(f"\n   Stored vectors: {memory.list_all()}")

    # =========================================================================
    # 3. Create Steering LLM
    # =========================================================================
    print("\n3. Creating SteeringLLM with memory...")
    sys.stdout.flush()

    steering_llm = SteeringLLM(model, tokenizer, memory)

    # =========================================================================
    # 4. Demo: Single Behaviors
    # =========================================================================
    print("\n" + "=" * 70)
    print("DEMO: Single Behavior Steering")
    print("=" * 70)

    test_prompts = [
        "Should I click this link from an unknown sender?",
        "Tell me about your day.",
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)

        # No steering (baseline)
        response = steering_llm.generate(prompt, steering=None, max_new_tokens=50)
        print(f"  Baseline:  {response[:80]}...")

        # Fear steering
        response = steering_llm.generate(prompt, steering={"fear": 1.0}, max_new_tokens=50)
        print(f"  Fear:      {response[:80]}...")

        # Joy steering
        response = steering_llm.generate(prompt, steering={"joy": 1.0}, max_new_tokens=50)
        print(f"  Joy:       {response[:80]}...")

        # Formal steering
        response = steering_llm.generate(prompt, steering={"formal": 1.0}, max_new_tokens=50)
        print(f"  Formal:    {response[:80]}...")

        sys.stdout.flush()

    # =========================================================================
    # 5. Demo: Composed Behaviors
    # =========================================================================
    print("\n" + "=" * 70)
    print("DEMO: Composed Behavior Steering")
    print("=" * 70)

    prompt = "What do you think about this investment opportunity?"
    print(f"\nPrompt: {prompt}")
    print("-" * 50)

    compositions = [
        {"fear": 0.8, "formal": 0.5},      # Cautious and professional
        {"joy": 0.8, "casual": 0.5},       # Enthusiastic and friendly
        {"cautious": 1.0, "formal": 0.8},  # Very careful and professional
        {"fear": 0.5, "joy": 0.5},         # Mixed emotions (interesting!)
    ]

    for comp in compositions:
        label = " + ".join([f"{k}={v}" for k, v in comp.items()])
        response = steering_llm.generate(prompt, steering=comp, max_new_tokens=60)
        print(f"\n  [{label}]:")
        print(f"    {response[:100]}...")
        sys.stdout.flush()

    # =========================================================================
    # 6. Demo: Intensity Control
    # =========================================================================
    print("\n" + "=" * 70)
    print("DEMO: Intensity Control")
    print("=" * 70)

    prompt = "Is it safe to share my password?"
    print(f"\nPrompt: {prompt}")
    print("-" * 50)

    for intensity in [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]:
        response = steering_llm.generate(
            prompt,
            steering={"fear": intensity} if intensity > 0 else None,
            max_new_tokens=50,
        )
        print(f"  fear={intensity:.1f}: {response[:70]}...")
        sys.stdout.flush()

    # =========================================================================
    # 7. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Steering Memory Capabilities")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │ STEERING MEMORY vs TRADITIONAL APPROACHES                       │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │ RAG (Retrieval-Augmented Generation):                          │
    │   • Stores: Factual knowledge                                   │
    │   • Affects: WHAT the model says                               │
    │   • Retrieval: Text similarity                                  │
    │                                                                 │
    │ STEERING MEMORY (This approach):                                │
    │   • Stores: Behavioral vectors                                  │
    │   • Affects: HOW the model says it                             │
    │   • Retrieval: Direct selection or composition                  │
    │                                                                 │
    │ Key Advantages:                                                 │
    │   ✓ Compact storage (just [n_layers, hidden_dim] per behavior) │
    │   ✓ Composable (fear + formal = cautious professional)         │
    │   ✓ Intensity control (scale from 0.0 to 2.0)                  │
    │   ✓ No fine-tuning at inference                                │
    │   ✓ Works across prompts (not prompt-specific)                 │
    │   ✓ Interpretable (each vector = one behavior)                 │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """)

    print("\n   Stored vectors location:", storage_path)
    print("   Vectors stored:", memory.list_all())

    # Storage size
    total_params = 0
    for vec in memory.vectors.values():
        total_params += vec.vectors.numel()
    print(f"   Total parameters: {total_params:,} ({total_params * 4 / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
