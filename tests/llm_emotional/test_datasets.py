"""
Tests for dataset handling.

Tests the NO FALLBACK policy - dataset operations must fail loudly
when data is missing or corrupted.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.llm_emotional.emotions.datasets import (
    load_dataset,
    get_contrastive_pairs,
    get_all_pairs,
    get_dataset_stats,
    validate_pair,
    DatasetNotFoundError,
    DatasetCorruptedError,
    DatasetValidationError,
)


def create_valid_dataset():
    """Create a valid test dataset."""
    return {
        'fear': [
            {'prompt': f'q{i}', 'neutral': f'neutral{i}', 'emotional': f'fear{i}'}
            for i in range(15)
        ],
        'curiosity': [
            {'prompt': f'q{i}', 'neutral': f'neutral{i}', 'emotional': f'curious{i}'}
            for i in range(12)
        ],
        'anger': [
            {'prompt': f'q{i}', 'neutral': f'neutral{i}', 'emotional': f'angry{i}'}
            for i in range(10)
        ],
        'joy': [
            {'prompt': f'q{i}', 'neutral': f'neutral{i}', 'emotional': f'joyful{i}'}
            for i in range(20)
        ],
    }


class TestLoadDataset:
    """Test dataset loading - NO FALLBACK policy."""

    def test_load_valid_dataset(self):
        """Should load valid dataset successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            path.write_text(json.dumps(create_valid_dataset()))

            data = load_dataset(path)

        assert set(data.keys()) == {'fear', 'curiosity', 'anger', 'joy'}
        assert len(data['fear']) == 15

    def test_load_nonexistent_fails_loudly(self):
        """Should FAIL on nonexistent file - NO FALLBACK."""
        with pytest.raises(DatasetNotFoundError) as exc_info:
            load_dataset(Path("/nonexistent/path.json"))

        assert "not found" in str(exc_info.value).lower()
        assert "generate_emotional_dataset" in str(exc_info.value)

    def test_load_invalid_json_fails_loudly(self):
        """Should FAIL on invalid JSON - NO FALLBACK."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text("this is not { valid json ]]")

            with pytest.raises(DatasetCorruptedError) as exc_info:
                load_dataset(path)

            assert "not valid JSON" in str(exc_info.value)

    def test_load_missing_emotion_fails_loudly(self):
        """Should FAIL if any required emotion is missing."""
        incomplete = {
            'fear': [{'prompt': 'q', 'neutral': 'n', 'emotional': 'e'}] * 10,
            'curiosity': [{'prompt': 'q', 'neutral': 'n', 'emotional': 'e'}] * 10,
            # Missing anger and joy
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "incomplete.json"
            path.write_text(json.dumps(incomplete))

            with pytest.raises(DatasetValidationError) as exc_info:
                load_dataset(path)

            assert "Missing emotions" in str(exc_info.value)

    def test_load_insufficient_pairs_fails_loudly(self):
        """Should FAIL if any emotion has < 10 pairs."""
        data = create_valid_dataset()
        data['anger'] = data['anger'][:5]  # Only 5 pairs

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "few.json"
            path.write_text(json.dumps(data))

            with pytest.raises(DatasetValidationError) as exc_info:
                load_dataset(path)

            assert "Insufficient pairs" in str(exc_info.value)
            assert "anger" in str(exc_info.value)


class TestValidatePair:
    """Test individual pair validation."""

    def test_valid_pair_passes(self):
        """Valid pair should not raise."""
        pair = {'prompt': 'How tall?', 'neutral': 'It is 10m.', 'emotional': 'Be careful, it is 10m!'}
        validate_pair(pair, index=0, emotion='fear')  # Should not raise

    def test_missing_key_fails(self):
        """Missing key should fail."""
        pair = {'prompt': 'How?', 'neutral': 'Answer'}  # Missing emotional

        with pytest.raises(DatasetValidationError, match="missing keys"):
            validate_pair(pair, index=0, emotion='fear')

    def test_empty_value_fails(self):
        """Empty string value should fail."""
        pair = {'prompt': 'Q', 'neutral': '   ', 'emotional': 'E'}

        with pytest.raises(DatasetValidationError, match="is empty"):
            validate_pair(pair, index=5, emotion='joy')

    def test_non_string_fails(self):
        """Non-string value should fail."""
        pair = {'prompt': 'Q', 'neutral': 123, 'emotional': 'E'}

        with pytest.raises(DatasetValidationError, match="must be string"):
            validate_pair(pair, index=0, emotion='anger')

    def test_identical_neutral_emotional_fails(self):
        """Identical neutral and emotional should fail - no contrast to learn."""
        pair = {'prompt': 'Q', 'neutral': 'Same response', 'emotional': 'Same response'}

        with pytest.raises(DatasetValidationError, match="identical"):
            validate_pair(pair, index=0, emotion='curiosity')


class TestGetContrastivePairs:
    """Test pair extraction."""

    def test_get_pairs_valid(self):
        """Should return (neutral, emotional) tuples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            path.write_text(json.dumps(create_valid_dataset()))

            pairs = get_contrastive_pairs('fear', path)

        assert len(pairs) == 15
        assert all(isinstance(p, tuple) for p in pairs)
        assert all(len(p) == 2 for p in pairs)
        assert pairs[0] == ('neutral0', 'fear0')

    def test_get_pairs_unknown_emotion_fails(self):
        """Should fail on unknown emotion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            path.write_text(json.dumps(create_valid_dataset()))

            with pytest.raises(DatasetValidationError, match="Unknown emotion"):
                get_contrastive_pairs('sadness', path)


class TestGetAllPairs:
    """Test getting all emotion pairs at once."""

    def test_get_all_pairs(self):
        """Should return dict of all emotions to pairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            path.write_text(json.dumps(create_valid_dataset()))

            all_pairs = get_all_pairs(path)

        assert set(all_pairs.keys()) == {'fear', 'curiosity', 'anger', 'joy'}
        assert len(all_pairs['fear']) == 15
        assert len(all_pairs['joy']) == 20


class TestGetDatasetStats:
    """Test statistics extraction."""

    def test_get_stats(self):
        """Should return useful statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            path.write_text(json.dumps(create_valid_dataset()))

            stats = get_dataset_stats(path)

        assert stats['total_pairs'] == 15 + 12 + 10 + 20
        assert 'by_emotion' in stats
        assert stats['by_emotion']['fear']['n_pairs'] == 15
        assert 'avg_neutral_len' in stats['by_emotion']['fear']
