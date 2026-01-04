#!/usr/bin/env python3
"""
Quick validation of the emotional steering implementation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch


def main():
    print("=" * 70)
    print("VALIDATING EMOTIONAL STEERING IMPLEMENTATION")
    print("=" * 70)

    # Test 1: Import modules
    print("\n[1] Testing imports...")
    try:
        from emotional_steering import EmotionalSteeringModel, DirectionExtractor, EMOTIONS
        print("    OK - All modules imported")
    except ImportError as e:
        print(f"    FAIL - Import error: {e}")
        return False

    # Test 2: Check emotion definitions
    print("\n[2] Checking emotion definitions...")
    expected_emotions = ["fear", "joy", "anger", "curiosity", "sadness", "surprise"]
    for emotion in expected_emotions:
        if emotion in EMOTIONS:
            n_pairs = len(EMOTIONS[emotion].pairs)
            print(f"    OK - {emotion}: {n_pairs} pairs")
        else:
            print(f"    WARN - {emotion}: not defined")

    # Test 3: Load model
    print("\n[3] Loading SmolLM3-3B...")
    try:
        model = EmotionalSteeringModel.from_pretrained("HuggingFaceTB/SmolLM3-3B")
        print(f"    OK - Model loaded")
        print(f"    Target layer: {model.target_layer}")
        print(f"    Default scale: {model.default_scale}")
    except Exception as e:
        print(f"    FAIL - {e}")
        return False

    # Test 4: Extract directions
    print("\n[4] Extracting directions (fear, joy, curiosity, anger)...")
    try:
        model.extract_directions(emotions=["fear", "joy", "curiosity", "anger"])
        print(f"    OK - Extracted {len(model.directions)} directions")
        for name, direction in model.directions.items():
            print(f"    - {name}: layer={direction.layer}, pairs={direction.n_pairs}")
    except Exception as e:
        print(f"    FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Generate baseline
    print("\n[5] Generating baseline text...")
    prompt = "The mysterious cave entrance beckoned, and I"
    try:
        baseline = model.generate(prompt, emotion=None)
        print(f"    Prompt: '{prompt}'")
        print(f"    Output: '{baseline[:80]}...'")
    except Exception as e:
        print(f"    FAIL - {e}")
        return False

    # Test 6: Generate with each emotion
    print("\n[6] Generating with emotional steering...")
    for emotion in ["fear", "joy", "curiosity", "anger"]:
        try:
            text = model.generate(prompt, emotion=emotion, scale=5.0)
            print(f"    [{emotion:10}]: '{text[:60]}...'")
        except Exception as e:
            print(f"    [{emotion:10}]: FAIL - {e}")

    # Test 7: Compare generations
    print("\n[7] Generating comparison...")
    try:
        results = model.generate_comparison(
            "The abandoned hospital was",
            emotions=["fear", "joy"],
        )
        print(f"    Baseline: '{results['baseline'][:50]}...'")
        print(f"    Fear:     '{results['fear'][:50]}...'")
        print(f"    Joy:      '{results['joy'][:50]}...'")
    except Exception as e:
        print(f"    FAIL - {e}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
