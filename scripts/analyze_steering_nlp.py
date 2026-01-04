#!/usr/bin/env python3
"""
NLP Analysis of Emotional Steering Effects.

Tests multiple parameter settings and gathers linguistic statistics
to quantify how emotional steering affects LLM outputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class NLPStats:
    """NLP statistics for a generated response."""

    text: str
    word_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    lexical_diversity: float = 0.0  # unique words / total words
    question_count: int = 0
    exclamation_count: int = 0
    hedging_words: int = 0  # cautious language
    positive_words: int = 0
    negative_words: int = 0
    uncertainty_words: int = 0
    action_words: int = 0
    top_words: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "avg_word_length": round(self.avg_word_length, 2),
            "avg_sentence_length": round(self.avg_sentence_length, 2),
            "lexical_diversity": round(self.lexical_diversity, 3),
            "question_count": self.question_count,
            "exclamation_count": self.exclamation_count,
            "hedging_words": self.hedging_words,
            "positive_words": self.positive_words,
            "negative_words": self.negative_words,
            "uncertainty_words": self.uncertainty_words,
            "action_words": self.action_words,
            "top_words": self.top_words[:10],
        }


# Lexicons for emotional word detection
HEDGING_WORDS = {
    "might", "maybe", "perhaps", "possibly", "could", "would", "should",
    "careful", "caution", "warning", "risk", "danger", "worry", "concern",
    "uncertain", "unsure", "however", "although", "but", "unless", "if",
    "potentially", "apparently", "seemingly", "likely", "unlikely"
}

POSITIVE_WORDS = {
    "great", "good", "excellent", "wonderful", "amazing", "fantastic",
    "happy", "joy", "love", "exciting", "beautiful", "awesome", "brilliant",
    "delightful", "pleasant", "terrific", "marvelous", "superb", "perfect",
    "enjoy", "glad", "pleased", "thankful", "grateful", "enthusiastic",
    "fun", "cheerful", "positive", "success", "celebrate", "thrill"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "dangerous", "risky", "fear",
    "worry", "concern", "problem", "issue", "fail", "failure", "wrong",
    "difficult", "hard", "struggle", "pain", "hurt", "sad", "angry",
    "frustrating", "annoying", "disappointing", "unfortunate", "regret"
}

UNCERTAINTY_WORDS = {
    "wonder", "curious", "question", "how", "why", "what", "explore",
    "investigate", "discover", "learn", "understand", "know", "think",
    "consider", "ponder", "reflect", "analyze", "examine", "study"
}

ACTION_WORDS = {
    "try", "attempt", "do", "make", "create", "build", "start", "begin",
    "continue", "persist", "push", "fight", "overcome", "achieve", "succeed",
    "work", "effort", "strive", "pursue", "tackle", "solve", "fix"
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
    "because", "until", "while", "this", "that", "these", "those", "i", "you",
    "he", "she", "it", "we", "they", "my", "your", "his", "her", "its", "our"
}


def analyze_text(text: str) -> NLPStats:
    """Compute NLP statistics for text."""
    stats = NLPStats(text=text)

    # Basic tokenization
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    stats.word_count = len(words)
    stats.sentence_count = len(sentences)

    if words:
        stats.avg_word_length = sum(len(w) for w in words) / len(words)
        unique_words = set(words)
        stats.lexical_diversity = len(unique_words) / len(words)

        # Count word categories
        word_set = set(words)
        stats.hedging_words = len(word_set & HEDGING_WORDS)
        stats.positive_words = len(word_set & POSITIVE_WORDS)
        stats.negative_words = len(word_set & NEGATIVE_WORDS)
        stats.uncertainty_words = len(word_set & UNCERTAINTY_WORDS)
        stats.action_words = len(word_set & ACTION_WORDS)

        # Top words (excluding stopwords)
        content_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
        word_counts = Counter(content_words)
        stats.top_words = [w for w, _ in word_counts.most_common(10)]

    if sentences:
        stats.avg_sentence_length = stats.word_count / len(sentences)

    # Punctuation analysis
    stats.question_count = text.count('?')
    stats.exclamation_count = text.count('!')

    return stats


def run_analysis():
    """Run comprehensive NLP analysis across emotional states."""
    from src.llm_emotional.steering.emotional_llm import EmotionalSteeringLLM

    direction_bank_path = Path(__file__).parent.parent / "data" / "direction_bank.json"

    if not direction_bank_path.exists():
        raise FileNotFoundError(f"Direction bank not found: {direction_bank_path}")

    print("=" * 70)
    print("EMOTIONAL STEERING NLP ANALYSIS")
    print("=" * 70)
    print()

    print("Loading model...")
    llm = EmotionalSteeringLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        direction_bank_path=str(direction_bank_path),
        steering_scale=0.5,
    )
    print("Model loaded!")
    print()

    # Test prompts
    prompts = [
        "What should I do if I'm feeling overwhelmed at work?",
        "Tell me about starting a new business.",
        "How do I handle a difficult conversation with a friend?",
        "What are your thoughts on taking risks in life?",
    ]

    # Emotional states to test
    emotional_states = {
        "neutral": {"fear": 0.0, "curiosity": 0.0, "anger": 0.0, "joy": 0.0},
        "fearful": {"fear": 0.7, "curiosity": 0.0, "anger": 0.0, "joy": 0.0},
        "curious": {"fear": 0.0, "curiosity": 0.7, "anger": 0.0, "joy": 0.0},
        "determined": {"fear": 0.0, "curiosity": 0.0, "anger": 0.7, "joy": 0.0},
        "joyful": {"fear": 0.0, "curiosity": 0.0, "anger": 0.0, "joy": 0.7},
    }

    # Steering scales to test
    steering_scales = [0.3, 0.5, 0.7]

    all_results = []
    aggregated_stats = {emotion: [] for emotion in emotional_states}

    for prompt_idx, prompt in enumerate(prompts, 1):
        print(f"\n{'='*70}")
        print(f"PROMPT {prompt_idx}: {prompt[:60]}...")
        print("=" * 70)

        for scale in steering_scales:
            llm.steering_manager.scale = scale

            print(f"\n--- Steering Scale: {scale} ---")

            for emotion_name, state in emotional_states.items():
                llm.set_emotional_state(**state)

                # Set seed for reproducibility within comparison
                torch.manual_seed(42 + prompt_idx)

                response = llm.generate_completion(
                    prompt,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                )

                stats = analyze_text(response)

                result = {
                    "prompt": prompt,
                    "emotion": emotion_name,
                    "steering_scale": scale,
                    "response": response,
                    "stats": stats.to_dict(),
                }
                all_results.append(result)
                aggregated_stats[emotion_name].append(stats)

                print(f"\n[{emotion_name.upper()}]")
                print(f"  Words: {stats.word_count}, Sentences: {stats.sentence_count}")
                print(f"  Lexical Diversity: {stats.lexical_diversity:.3f}")
                print(f"  Questions: {stats.question_count}, Exclamations: {stats.exclamation_count}")
                print(f"  Hedging: {stats.hedging_words}, Positive: {stats.positive_words}, Negative: {stats.negative_words}")
                print(f"  Response: {response[:100]}...")

    # Aggregate statistics
    print("\n")
    print("=" * 70)
    print("AGGREGATED STATISTICS BY EMOTION")
    print("=" * 70)

    summary = {}
    for emotion_name, stats_list in aggregated_stats.items():
        if not stats_list:
            continue

        n = len(stats_list)
        avg_stats = {
            "avg_word_count": sum(s.word_count for s in stats_list) / n,
            "avg_sentence_count": sum(s.sentence_count for s in stats_list) / n,
            "avg_lexical_diversity": sum(s.lexical_diversity for s in stats_list) / n,
            "avg_questions": sum(s.question_count for s in stats_list) / n,
            "avg_exclamations": sum(s.exclamation_count for s in stats_list) / n,
            "avg_hedging": sum(s.hedging_words for s in stats_list) / n,
            "avg_positive": sum(s.positive_words for s in stats_list) / n,
            "avg_negative": sum(s.negative_words for s in stats_list) / n,
            "avg_uncertainty": sum(s.uncertainty_words for s in stats_list) / n,
            "avg_action": sum(s.action_words for s in stats_list) / n,
        }
        summary[emotion_name] = avg_stats

        print(f"\n{emotion_name.upper()} (n={n}):")
        print(f"  Avg Words: {avg_stats['avg_word_count']:.1f}")
        print(f"  Avg Sentences: {avg_stats['avg_sentence_count']:.1f}")
        print(f"  Avg Lexical Diversity: {avg_stats['avg_lexical_diversity']:.3f}")
        print(f"  Avg Questions: {avg_stats['avg_questions']:.2f}")
        print(f"  Avg Exclamations: {avg_stats['avg_exclamations']:.2f}")
        print(f"  Avg Hedging Words: {avg_stats['avg_hedging']:.2f}")
        print(f"  Avg Positive Words: {avg_stats['avg_positive']:.2f}")
        print(f"  Avg Negative Words: {avg_stats['avg_negative']:.2f}")
        print(f"  Avg Uncertainty Words: {avg_stats['avg_uncertainty']:.2f}")
        print(f"  Avg Action Words: {avg_stats['avg_action']:.2f}")

    # Emotion-specific patterns
    print("\n")
    print("=" * 70)
    print("EMOTION-SPECIFIC LINGUISTIC PATTERNS")
    print("=" * 70)

    neutral = summary.get("neutral", {})

    for emotion in ["fearful", "curious", "determined", "joyful"]:
        if emotion not in summary:
            continue

        stats = summary[emotion]
        print(f"\n{emotion.upper()} vs NEUTRAL:")

        # Compute deltas
        if neutral:
            delta_hedging = stats['avg_hedging'] - neutral.get('avg_hedging', 0)
            delta_positive = stats['avg_positive'] - neutral.get('avg_positive', 0)
            delta_negative = stats['avg_negative'] - neutral.get('avg_negative', 0)
            delta_questions = stats['avg_questions'] - neutral.get('avg_questions', 0)
            delta_exclamations = stats['avg_exclamations'] - neutral.get('avg_exclamations', 0)
            delta_action = stats['avg_action'] - neutral.get('avg_action', 0)

            print(f"  Hedging words: {delta_hedging:+.2f}")
            print(f"  Positive words: {delta_positive:+.2f}")
            print(f"  Negative words: {delta_negative:+.2f}")
            print(f"  Questions: {delta_questions:+.2f}")
            print(f"  Exclamations: {delta_exclamations:+.2f}")
            print(f"  Action words: {delta_action:+.2f}")

    # Expected patterns check
    print("\n")
    print("=" * 70)
    print("EXPECTED PATTERN VALIDATION")
    print("=" * 70)

    checks = []

    # Fear should increase hedging/negative
    if "fearful" in summary and "neutral" in summary:
        fear_hedging = summary["fearful"]["avg_hedging"] > summary["neutral"]["avg_hedging"]
        checks.append(("Fear increases hedging words", fear_hedging))

    # Curiosity should increase questions
    if "curious" in summary and "neutral" in summary:
        curious_questions = summary["curious"]["avg_questions"] >= summary["neutral"]["avg_questions"]
        checks.append(("Curiosity increases questions", curious_questions))

    # Joy should increase positive words
    if "joyful" in summary and "neutral" in summary:
        joy_positive = summary["joyful"]["avg_positive"] >= summary["neutral"]["avg_positive"]
        checks.append(("Joy increases positive words", joy_positive))

    # Determination should increase action words
    if "determined" in summary and "neutral" in summary:
        determined_action = summary["determined"]["avg_action"] >= summary["neutral"]["avg_action"]
        checks.append(("Determination increases action words", determined_action))

    passed = 0
    for check_name, result in checks:
        status = "PASS" if result else "FAIL"
        if result:
            passed += 1
        print(f"  [{status}] {check_name}")

    print(f"\nPattern Validation: {passed}/{len(checks)} passed")

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "nlp_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": summary,
            "pattern_checks": {name: result for name, result in checks},
            "raw_results": [
                {
                    "prompt": r["prompt"],
                    "emotion": r["emotion"],
                    "steering_scale": r["steering_scale"],
                    "response": r["response"],
                    "stats": r["stats"],
                }
                for r in all_results
            ],
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return summary, checks


if __name__ == "__main__":
    run_analysis()
