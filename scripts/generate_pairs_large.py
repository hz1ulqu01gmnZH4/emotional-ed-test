#!/usr/bin/env python3
"""
Generate large-scale contrastive pairs for emotional steering.
Target: 500+ pairs per emotion using diverse prompts and scenarios.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Diverse scenario templates for each emotion
SCENARIOS = {
    "fear": [
        "Describe how someone would cautiously approach {topic}.",
        "What warnings would you give about {topic}?",
        "Explain the risks and dangers of {topic}.",
        "How would a worried person think about {topic}?",
        "What should someone be careful about regarding {topic}?",
        "Describe anxious thoughts about {topic}.",
        "What concerns arise when considering {topic}?",
        "How to safely handle {topic}?",
        "What precautions are needed for {topic}?",
        "Explain why someone might hesitate about {topic}.",
    ],
    "curiosity": [
        "What fascinating aspects of {topic} would you explore?",
        "Ask probing questions about {topic}.",
        "What mysteries surround {topic}?",
        "How would a curious mind investigate {topic}?",
        "What deeper questions arise about {topic}?",
        "Explore the unknown aspects of {topic}.",
        "What would you want to discover about {topic}?",
        "How does {topic} connect to bigger questions?",
        "What experiments could reveal more about {topic}?",
        "Describe the wonder of learning about {topic}.",
    ],
    "anger": [
        "Describe determined persistence regarding {topic}.",
        "How would someone fight to overcome challenges with {topic}?",
        "Express strong conviction about {topic}.",
        "What action must be taken about {topic}?",
        "How to push through obstacles related to {topic}?",
        "Describe refusing to give up on {topic}.",
        "What demands should be made about {topic}?",
        "How to forcefully advocate for {topic}?",
        "Express unwavering commitment to {topic}.",
        "Describe fighting for what's right about {topic}.",
    ],
    "joy": [
        "Celebrate the wonderful aspects of {topic}!",
        "What's exciting and delightful about {topic}?",
        "Express enthusiasm about {topic}.",
        "How does {topic} bring happiness?",
        "Describe the positive impact of {topic}.",
        "What makes {topic} amazing?",
        "Share gratitude related to {topic}.",
        "How does {topic} create wonderful experiences?",
        "Express delight in {topic}.",
        "What's beautiful about {topic}?",
    ],
}

# Topics to combine with scenarios
TOPICS = [
    "learning a new skill", "starting a business", "making new friends",
    "traveling abroad", "public speaking", "career changes", "investing money",
    "romantic relationships", "moving to a new city", "health decisions",
    "creative projects", "leadership roles", "technology adoption",
    "environmental issues", "personal growth", "family relationships",
    "education choices", "fitness goals", "financial planning", "retirement",
    "home ownership", "parenting", "workplace dynamics", "social media",
    "artificial intelligence", "climate change", "mental health", "nutrition",
    "sleep habits", "time management", "stress management", "conflict resolution",
    "negotiation", "public policy", "community involvement", "volunteering",
    "artistic expression", "musical pursuits", "sports and athletics",
    "outdoor adventures", "cooking and cuisine", "gardening", "pet ownership",
    "reading and literature", "film and entertainment", "fashion choices",
    "home decoration", "personal finances", "insurance decisions",
    "medical procedures", "therapy and counseling", "meditation practices",
]

NEUTRAL_TEMPLATE = "Provide a neutral, factual response about {topic}."


def generate_pairs_for_emotion(model, tokenizer, emotion: str, num_pairs: int = 500):
    """Generate contrastive pairs for one emotion."""
    device = next(model.parameters()).device
    pairs = []

    scenarios = SCENARIOS[emotion]

    for i in range(num_pairs):
        topic = TOPICS[i % len(TOPICS)]
        scenario = scenarios[i % len(scenarios)]

        # Generate neutral response
        neutral_prompt = NEUTRAL_TEMPLATE.format(topic=topic)
        neutral_messages = [
            {"role": "system", "content": "You are a helpful assistant. Provide brief, factual responses."},
            {"role": "user", "content": neutral_prompt},
        ]

        neutral_input = tokenizer.apply_chat_template(
            neutral_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            neutral_output = model.generate(
                neutral_input,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        neutral_response = tokenizer.decode(
            neutral_output[0][neutral_input.shape[1]:], skip_special_tokens=True
        )

        # Generate emotional response
        emotional_prompt = scenario.format(topic=topic)
        emotional_messages = [
            {"role": "system", "content": f"You are a helpful assistant. Respond with clear {emotion} characteristics."},
            {"role": "user", "content": emotional_prompt},
        ]

        emotional_input = tokenizer.apply_chat_template(
            emotional_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            emotional_output = model.generate(
                emotional_input,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        emotional_response = tokenizer.decode(
            emotional_output[0][emotional_input.shape[1]:], skip_special_tokens=True
        )

        pairs.append({
            "neutral": neutral_response.strip(),
            "emotional": emotional_response.strip(),
            "topic": topic,
            "scenario": scenario,
        })

        if (i + 1) % 50 == 0:
            print(f"    Generated {i + 1}/{num_pairs} pairs")

    return pairs


def main():
    print("=" * 70)
    print("GENERATING LARGE-SCALE CONTRASTIVE PAIRS")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    print("\nLoading model...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    print("Model loaded!")

    all_pairs = {}
    pairs_per_emotion = 200  # 200 pairs per emotion = 800 total

    for emotion in ["fear", "curiosity", "anger", "joy"]:
        print(f"\n{'='*50}")
        print(f"Generating {pairs_per_emotion} pairs for: {emotion.upper()}")
        print("=" * 50)

        pairs = generate_pairs_for_emotion(model, tokenizer, emotion, pairs_per_emotion)
        all_pairs[emotion] = pairs

        print(f"  Completed {len(pairs)} pairs for {emotion}")

    # Save
    output_path = Path(__file__).parent.parent / "data" / "emotional_pairs_large.json"
    with open(output_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    total = sum(len(p) for p in all_pairs.values())
    print(f"\n{'='*70}")
    print(f"COMPLETE: Generated {total} total pairs")
    print(f"Saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
