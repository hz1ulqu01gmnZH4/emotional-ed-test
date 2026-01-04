"""
Dataset handling for emotional contrastive pairs.

NO FALLBACK POLICY: If dataset is missing or corrupted, FAIL LOUDLY.
Do not silently use hardcoded data - that masks generation failures.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


class DatasetNotFoundError(Exception):
    """
    Dataset file not found.

    Generate it first using: uv run scripts/generate_emotional_dataset.py
    """
    pass


class DatasetValidationError(Exception):
    """
    Dataset failed validation.

    The data is malformed or incomplete. Regenerate with proper data.
    """
    pass


class DatasetCorruptedError(Exception):
    """
    Dataset file is corrupted (invalid JSON, wrong structure).

    Delete and regenerate the dataset.
    """
    pass


REQUIRED_EMOTIONS = frozenset({'fear', 'curiosity', 'anger', 'joy'})
MIN_PAIRS_PER_EMOTION = 10


def validate_pair(pair: dict, index: int, emotion: str) -> None:
    """
    Validate a single contrastive pair.

    Args:
        pair: Dict with 'prompt', 'neutral', 'emotional' keys
        index: Index in the pairs list (for error messages)
        emotion: Which emotion this pair is for

    Raises:
        DatasetValidationError: If pair is invalid
    """
    required_keys = {'prompt', 'neutral', 'emotional'}
    missing_keys = required_keys - set(pair.keys())

    if missing_keys:
        raise DatasetValidationError(
            f"Pair {index} for {emotion} missing keys: {missing_keys}. "
            f"Got keys: {set(pair.keys())}"
        )

    # Ensure non-empty strings
    for key in required_keys:
        if not isinstance(pair[key], str):
            raise DatasetValidationError(
                f"Pair {index} for {emotion}: '{key}' must be string, "
                f"got {type(pair[key]).__name__}"
            )
        if not pair[key].strip():
            raise DatasetValidationError(
                f"Pair {index} for {emotion}: '{key}' is empty"
            )

    # Ensure contrast exists (neutral != emotional)
    if pair['neutral'].strip() == pair['emotional'].strip():
        raise DatasetValidationError(
            f"Pair {index} for {emotion}: neutral and emotional are identical. "
            "No contrast to learn from."
        )


def validate_dataset(data: dict) -> None:
    """
    Validate entire dataset structure.

    Args:
        data: Loaded dataset dict

    Raises:
        DatasetValidationError: If dataset is invalid
    """
    if not isinstance(data, dict):
        raise DatasetValidationError(
            f"Dataset must be a dict, got {type(data).__name__}"
        )

    # Check required emotions present
    missing_emotions = REQUIRED_EMOTIONS - set(data.keys())
    if missing_emotions:
        raise DatasetValidationError(
            f"Missing emotions in dataset: {missing_emotions}"
        )

    # Validate each emotion's pairs
    for emotion in REQUIRED_EMOTIONS:
        pairs = data[emotion]

        if not isinstance(pairs, list):
            raise DatasetValidationError(
                f"Pairs for {emotion} must be a list, got {type(pairs).__name__}"
            )

        if len(pairs) < MIN_PAIRS_PER_EMOTION:
            raise DatasetValidationError(
                f"Insufficient pairs for {emotion}: got {len(pairs)}, "
                f"need >= {MIN_PAIRS_PER_EMOTION}"
            )

        for i, pair in enumerate(pairs):
            validate_pair(pair, i, emotion)


def load_dataset(path: Path) -> Dict[str, List[Dict]]:
    """
    Load emotional contrastive dataset.

    FAILS LOUDLY if:
    - File doesn't exist
    - JSON is invalid
    - Data is malformed

    NO FALLBACK to hardcoded data - that would mask generation failures.

    Args:
        path: Path to dataset JSON file

    Returns:
        Dict mapping emotion names to lists of contrastive pairs

    Raises:
        DatasetNotFoundError: If file doesn't exist
        DatasetCorruptedError: If JSON is invalid
        DatasetValidationError: If data is malformed
    """
    path = Path(path)

    if not path.exists():
        raise DatasetNotFoundError(
            f"Dataset not found at {path}. "
            f"Generate it first: uv run scripts/generate_emotional_dataset.py"
        )

    # Load JSON
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DatasetCorruptedError(
            f"Dataset at {path} is not valid JSON: {e}. "
            "Delete and regenerate."
        )
    except Exception as e:
        raise DatasetCorruptedError(
            f"Failed to read dataset at {path}: {e}"
        )

    # Validate structure
    validate_dataset(data)

    return data


def get_contrastive_pairs(
    emotion: str,
    dataset_path: Path
) -> List[Tuple[str, str]]:
    """
    Get (neutral, emotional) pairs for direction learning.

    Args:
        emotion: One of 'fear', 'curiosity', 'anger', 'joy'
        dataset_path: Path to dataset JSON

    Returns:
        List of (neutral_text, emotional_text) tuples

    Raises:
        DatasetNotFoundError: If dataset missing
        DatasetValidationError: If emotion unknown or data invalid
    """
    if emotion not in REQUIRED_EMOTIONS:
        raise DatasetValidationError(
            f"Unknown emotion: {emotion}. "
            f"Must be one of: {sorted(REQUIRED_EMOTIONS)}"
        )

    data = load_dataset(dataset_path)

    return [
        (pair['neutral'], pair['emotional'])
        for pair in data[emotion]
    ]


def get_all_pairs(dataset_path: Path) -> Dict[str, List[Tuple[str, str]]]:
    """
    Get contrastive pairs for all emotions.

    Args:
        dataset_path: Path to dataset JSON

    Returns:
        Dict mapping emotion names to lists of (neutral, emotional) tuples
    """
    data = load_dataset(dataset_path)

    return {
        emotion: [
            (pair['neutral'], pair['emotional'])
            for pair in pairs
        ]
        for emotion, pairs in data.items()
    }


def get_dataset_stats(dataset_path: Path) -> Dict:
    """
    Get statistics about the dataset.

    Args:
        dataset_path: Path to dataset JSON

    Returns:
        Dict with statistics (counts, avg lengths, etc.)
    """
    data = load_dataset(dataset_path)

    stats = {
        'total_pairs': 0,
        'by_emotion': {},
    }

    for emotion, pairs in data.items():
        n_pairs = len(pairs)
        stats['total_pairs'] += n_pairs

        # Compute average response lengths
        neutral_lens = [len(p['neutral']) for p in pairs]
        emotional_lens = [len(p['emotional']) for p in pairs]

        stats['by_emotion'][emotion] = {
            'n_pairs': n_pairs,
            'avg_neutral_len': sum(neutral_lens) / len(neutral_lens),
            'avg_emotional_len': sum(emotional_lens) / len(emotional_lens),
        }

    return stats
