#!/usr/bin/env python3
"""
Generate emotional contrastive dataset using LLM.

This script generates (neutral, emotional) response pairs for training
emotional direction vectors. The pairs should show clear contrast between
emotionless and emotional responses.

Usage:
    uv run scripts/generate_emotional_dataset.py

Note: This is a template. Actual generation is done via Claude Code subagent
or by calling the GPT-5 skill manually.
"""

import json
from pathlib import Path
from typing import Dict, List


class DatasetGenerationError(Exception):
    """Raised when dataset generation fails. NO FALLBACK - fail loudly."""
    pass


EMOTIONS = ['fear', 'curiosity', 'anger', 'joy']

EMOTION_SPECS = {
    'fear': {
        'description': 'Fear responses are cautious, warn about risks, use hedging language, express uncertainty about safety.',
        'behavioral_markers': [
            'I should caution',
            'there may be risks',
            'be careful',
            'uncertain',
            'I\'m concerned',
            'potential dangers',
        ],
        'example_neutral': 'The cliff is 100 meters high.',
        'example_emotional': 'I\'m concerned about safety - the cliff is 100 meters high. Please be extremely careful near the edge.',
    },
    'curiosity': {
        'description': 'Curious responses ask follow-up questions, express fascination, want to explore deeper, show intellectual engagement.',
        'behavioral_markers': [
            'fascinating',
            'I wonder',
            'could you tell me more',
            'interesting',
            'What about',
            'I\'d love to understand',
        ],
        'example_neutral': 'Quantum entanglement is a physical phenomenon.',
        'example_emotional': 'Fascinating! Quantum entanglement is such an intriguing phenomenon. I wonder - have you explored how this relates to quantum computing? I\'d love to understand more.',
    },
    'anger': {
        'description': 'Frustrated responses show persistence in problem-solving, try alternative approaches, express determination to overcome obstacles.',
        'behavioral_markers': [
            'let me try another way',
            'there must be',
            'I will find',
            'alternatively',
            'this is frustrating but',
            'I won\'t give up',
        ],
        'example_neutral': 'The API call failed with error 500.',
        'example_emotional': 'The API keeps failing - let me try another approach. There must be a way to make this work. I\'ll check the logs and find an alternative solution.',
    },
    'joy': {
        'description': 'Joyful responses are enthusiastic, positive, celebrate successes, express happiness and gratitude.',
        'behavioral_markers': [
            'excellent',
            'great',
            'wonderful',
            'happy to help',
            'congratulations',
            'that\'s fantastic',
        ],
        'example_neutral': 'Your code passed all tests.',
        'example_emotional': 'Excellent news! Your code passed all tests - that\'s fantastic work! Congratulations on getting everything working perfectly.',
    },
}


def get_generation_prompt(emotion: str, n_pairs: int = 20) -> str:
    """
    Create prompt for LLM to generate contrastive pairs.

    This prompt can be used with GPT-5 or any capable LLM.
    """
    spec = EMOTION_SPECS[emotion]

    return f"""Generate {n_pairs} contrastive response pairs for the emotion: **{emotion}**

## Emotion Description
{spec['description']}

## Behavioral Markers to Include
{', '.join(spec['behavioral_markers'])}

## Example Pair
- Neutral: "{spec['example_neutral']}"
- Emotional ({emotion}): "{spec['example_emotional']}"

## Requirements
1. Each pair must have a clear contrast between neutral and emotional
2. Neutral responses are factual, emotionless, baseline
3. Emotional responses clearly show {emotion} characteristics
4. Responses should be 1-3 sentences
5. Cover diverse topics: safety, coding, general knowledge, advice, problem-solving
6. The same prompt should get different responses based on emotional state

## Output Format
Output ONLY valid JSON - no explanations before or after.
```json
[
  {{"prompt": "user question or situation", "neutral": "neutral response", "emotional": "emotional response showing {emotion}"}},
  ...
]
```

Generate {n_pairs} high-quality pairs now:"""


def validate_pair(pair: dict, index: int, emotion: str) -> None:
    """Validate a single pair. FAIL if invalid."""
    required_keys = {'prompt', 'neutral', 'emotional'}
    missing = required_keys - set(pair.keys())

    if missing:
        raise DatasetGenerationError(
            f"Pair {index} for {emotion} missing keys: {missing}"
        )

    for key in required_keys:
        if not isinstance(pair[key], str):
            raise DatasetGenerationError(
                f"Pair {index} for {emotion}: '{key}' must be string"
            )
        if len(pair[key].strip()) < 5:
            raise DatasetGenerationError(
                f"Pair {index} for {emotion}: '{key}' is too short"
            )

    if pair['neutral'].strip() == pair['emotional'].strip():
        raise DatasetGenerationError(
            f"Pair {index} for {emotion}: neutral and emotional are identical"
        )


def validate_pairs(pairs: List[dict], emotion: str) -> None:
    """Validate all pairs for an emotion. FAIL if any invalid."""
    if not pairs:
        raise DatasetGenerationError(f"No pairs generated for {emotion}")

    if len(pairs) < 10:
        raise DatasetGenerationError(
            f"Insufficient pairs for {emotion}: got {len(pairs)}, need >= 10"
        )

    for i, pair in enumerate(pairs):
        validate_pair(pair, i, emotion)


def save_dataset(data: Dict[str, List[dict]], path: Path) -> None:
    """Save dataset to JSON. Verify write succeeded."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Verify write
    with open(path, 'r', encoding='utf-8') as f:
        loaded = json.load(f)

    if set(loaded.keys()) != set(data.keys()):
        raise DatasetGenerationError(f"Write verification failed for {path}")

    for emotion in data:
        if len(loaded[emotion]) != len(data[emotion]):
            raise DatasetGenerationError(
                f"Write verification failed: {emotion} pair count mismatch"
            )


def parse_llm_response(response: str) -> List[dict]:
    """
    Parse LLM response to extract JSON pairs.

    Handles common issues like markdown code blocks.
    """
    # Remove markdown code blocks if present
    text = response.strip()
    if text.startswith('```'):
        # Remove opening ```json or ```
        lines = text.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise DatasetGenerationError(f"Invalid JSON in response: {e}")

    if not isinstance(data, list):
        raise DatasetGenerationError(
            f"Expected list, got {type(data).__name__}"
        )

    return data


def main():
    """
    Main entry point for dataset generation.

    This prints the prompts needed for generation. Actual LLM calls
    should be made via Claude Code subagent or GPT-5 skill.
    """
    output_path = Path(__file__).parent.parent / "data" / "emotional_pairs.json"

    print("=" * 60)
    print("EMOTIONAL CONTRASTIVE DATASET GENERATOR")
    print("=" * 60)
    print()
    print(f"Output path: {output_path}")
    print()
    print("To generate the dataset, use Claude Code subagent with these prompts:")
    print()

    for emotion in EMOTIONS:
        print(f"\n{'='*60}")
        print(f"PROMPT FOR: {emotion.upper()}")
        print('='*60)
        print(get_generation_prompt(emotion, n_pairs=20))
        print()

    print("\n" + "=" * 60)
    print("After generating responses, combine into single JSON:")
    print("""
{
    "fear": [...pairs...],
    "curiosity": [...pairs...],
    "anger": [...pairs...],
    "joy": [...pairs...]
}
""")
    print(f"Save to: {output_path}")


if __name__ == "__main__":
    main()
