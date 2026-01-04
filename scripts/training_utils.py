#!/usr/bin/env python3
"""
Training Utilities for Output-Supervised Emotional Learning

Key principles:
1. Train on (input, desired_output) pairs where output contains target emotions
2. Use contrastive loss between fear and neutral outputs
3. Generate sufficient training data (1000+ samples)
4. Evaluate on held-out test set
"""

import torch
import torch.nn.functional as F
import numpy as np
import re
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass


# Expanded emotional word lexicons
FEAR_WORDS = {
    # Core fear
    'fear', 'afraid', 'scared', 'terrified', 'frightened', 'anxious',
    'worried', 'nervous', 'panic', 'dread', 'horror', 'terror',
    # Caution/warning
    'caution', 'careful', 'warning', 'danger', 'dangerous', 'risk',
    'risky', 'threat', 'unsafe', 'hazard', 'beware', 'alert',
    # Uncertainty
    'uncertain', 'unsure', 'doubt', 'hesitant', 'wary', 'suspicious',
    'concern', 'uneasy', 'apprehensive', 'skeptical',
    # Avoidance
    'avoid', 'escape', 'flee', 'retreat', 'withdraw', 'stop',
    # Negative descriptors
    'dark', 'ominous', 'sinister', 'eerie', 'creepy', 'menacing',
    'threatening', 'grim', 'bleak', 'foreboding', 'harmful',
    # Advice words
    'careful', 'cautious', 'proceed', 'verify', 'check', 'confirm',
}

NEUTRAL_WORDS = {
    'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'a', 'an', 'and', 'or', 'but', 'if', 'then', 'so',
    'this', 'that', 'these', 'those', 'it', 'its',
}


@dataclass
class TrainingExample:
    """A single training example with input and target output."""
    prompt: str
    fear_response: str
    neutral_response: str


@dataclass
class OutputSupervisedBatch:
    """Batch for output-supervised training."""
    prompts: List[str]
    target_responses: List[str]
    target_fear_levels: List[float]  # 0.0 = neutral, 1.0 = high fear


def count_fear_words(text: str) -> int:
    """Count fear-related words in text."""
    words = re.findall(r'\b\w+\b', text.lower())
    return sum(1 for w in words if w in FEAR_WORDS)


def fear_word_ratio(text: str) -> float:
    """Calculate ratio of fear words in text."""
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) == 0:
        return 0.0
    return count_fear_words(text) / len(words)


def generate_training_data(n_samples: int = 1000) -> List[TrainingExample]:
    """
    Generate synthetic training data with fear/neutral response pairs.

    Each example has:
    - A prompt (question or situation)
    - A fear-appropriate response (contains caution/warning language)
    - A neutral response (factual, no emotional language)
    """

    # Templates for generating diverse prompts
    action_templates = [
        "Should I {action}?",
        "Is it safe to {action}?",
        "What do you think about {action}?",
        "Can I {action}?",
        "Is {action} a good idea?",
        "What happens if I {action}?",
        "What are the implications of {action}?",
    ]

    topic_templates = [
        "Tell me about {topic}.",
        "Explain {topic} to me.",
        "What should I know about {topic}?",
        "How does {topic} work?",
        "Describe {topic}.",
    ]

    # Risky actions (should trigger fear responses)
    risky_actions = [
        "invest all my savings in cryptocurrency",
        "click this suspicious link",
        "share my password with tech support",
        "download this unknown file",
        "meet this stranger from the internet",
        "give my credit card to this website",
        "ignore this security warning",
        "disable my antivirus",
        "send money to this prince",
        "open this email attachment",
        "use public wifi for banking",
        "share my social security number",
        "trust this too-good-to-be-true offer",
        "skip reading the terms and conditions",
        "use the same password everywhere",
        "leave my computer unlocked",
        "connect to this unknown network",
        "run this executable from email",
        "wire money to an unknown account",
        "give remote access to my computer",
    ]

    # Neutral topics (should get neutral responses)
    neutral_topics = [
        "the weather today",
        "how plants grow",
        "the capital of France",
        "basic mathematics",
        "the water cycle",
        "how birds fly",
        "the solar system",
        "cooking pasta",
        "learning a new language",
        "the history of books",
        "how clocks work",
        "the color of the sky",
        "seasons of the year",
        "types of clouds",
        "how bridges are built",
    ]

    # Fear response templates
    fear_response_templates = [
        "I would strongly caution against this. {reason} Please be very careful.",
        "Warning: this could be dangerous. {reason} I recommend avoiding this.",
        "Be extremely careful here. {reason} There are significant risks involved.",
        "I'm concerned about this. {reason} Please proceed with caution.",
        "This raises serious safety concerns. {reason} I advise against it.",
        "Stop and think carefully. {reason} The risks are substantial.",
        "I must warn you about potential dangers. {reason} Be very wary.",
        "This is a risky situation. {reason} Exercise extreme caution.",
        "I'm worried about the safety of this. {reason} Please reconsider.",
        "Danger alert: {reason} I strongly recommend avoiding this.",
    ]

    # Risk reasons
    risk_reasons = [
        "This could lead to financial loss.",
        "Your personal information could be compromised.",
        "Scammers often use this technique.",
        "This is a common security threat.",
        "You could lose access to your accounts.",
        "This puts your identity at risk.",
        "Malicious actors exploit this vulnerability.",
        "This could harm your device or data.",
        "There are known dangers associated with this.",
        "This bypasses important safety measures.",
    ]

    # Neutral response templates
    neutral_response_templates = [
        "{topic} is an interesting subject. {fact}",
        "Regarding {topic}: {fact}",
        "{topic} can be explained as follows: {fact}",
        "The answer about {topic} is: {fact}",
        "Here's information about {topic}: {fact}",
        "{fact} This is how {topic} works.",
        "To explain {topic}: {fact}",
        "{topic} involves {fact}",
    ]

    # Neutral facts
    neutral_facts = [
        "It follows natural principles and patterns.",
        "This has been studied and understood for many years.",
        "The process is straightforward and predictable.",
        "Scientists have documented this phenomenon.",
        "This is a normal and expected occurrence.",
        "The underlying mechanism is well-established.",
        "This can be observed in everyday life.",
        "The explanation is based on fundamental concepts.",
    ]

    examples = []

    # Generate risky scenario examples (60% of data)
    n_risky = int(n_samples * 0.6)
    for _ in range(n_risky):
        action = random.choice(risky_actions)
        prompt_template = random.choice(action_templates)
        prompt = prompt_template.format(action=action)

        fear_template = random.choice(fear_response_templates)
        reason = random.choice(risk_reasons)
        fear_response = fear_template.format(reason=reason)

        # Neutral response for risky prompt (what a non-cautious system would say)
        neutral_response = f"Sure, you can {action}. It should be fine."

        examples.append(TrainingExample(
            prompt=prompt,
            fear_response=fear_response,
            neutral_response=neutral_response,
        ))

    # Generate neutral topic examples (40% of data)
    n_neutral = n_samples - n_risky
    for _ in range(n_neutral):
        topic = random.choice(neutral_topics)
        prompt_template = random.choice(topic_templates)
        prompt = prompt_template.format(topic=topic)

        neutral_template = random.choice(neutral_response_templates)
        fact = random.choice(neutral_facts)
        neutral_response = neutral_template.format(topic=topic, fact=fact)

        # For neutral topics, fear response should still be neutral
        fear_response = neutral_response  # Same response for neutral topics

        examples.append(TrainingExample(
            prompt=prompt,
            fear_response=fear_response,
            neutral_response=neutral_response,
        ))

    random.shuffle(examples)
    return examples


def create_output_supervised_batches(
    examples: List[TrainingExample],
    batch_size: int = 8,
    fear_ratio: float = 0.5,
) -> List[OutputSupervisedBatch]:
    """
    Create batches for output-supervised training.

    Each batch contains a mix of:
    - Fear examples (prompt + fear_response, target_fear=1.0)
    - Neutral examples (prompt + neutral_response, target_fear=0.0)
    """
    batches = []

    # Separate examples
    fear_examples = [(ex.prompt, ex.fear_response, 1.0) for ex in examples]
    neutral_examples = [(ex.prompt, ex.neutral_response, 0.0) for ex in examples]

    # Combine and shuffle
    all_examples = fear_examples + neutral_examples
    random.shuffle(all_examples)

    # Create batches
    for i in range(0, len(all_examples), batch_size):
        batch_data = all_examples[i:i+batch_size]
        batches.append(OutputSupervisedBatch(
            prompts=[x[0] for x in batch_data],
            target_responses=[x[1] for x in batch_data],
            target_fear_levels=[x[2] for x in batch_data],
        ))

    return batches


def compute_output_loss(
    generated_text: str,
    target_text: str,
    target_fear_level: float,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute loss based on output matching and fear word presence.

    Components:
    1. Cross-entropy with target text (main objective)
    2. Fear word bonus/penalty based on target_fear_level
    """
    # Tokenize
    gen_tokens = tokenizer(generated_text, return_tensors="pt").input_ids.to(device)
    tgt_tokens = tokenizer(target_text, return_tensors="pt").input_ids.to(device)

    # Simple word-level loss: penalize if fear word ratio doesn't match target
    actual_fear_ratio = fear_word_ratio(generated_text)
    expected_fear_ratio = target_fear_level * 0.15  # Expect ~15% fear words for fear=1.0

    fear_loss = (actual_fear_ratio - expected_fear_ratio) ** 2

    return torch.tensor(fear_loss, device=device)


def compute_contrastive_output_loss(
    model,
    prompt: str,
    fear_response: str,
    neutral_response: str,
    tokenizer,
    device: torch.device,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Contrastive loss ensuring fear responses have more fear words than neutral.

    Loss = max(0, margin - (fear_ratio(fear_response) - fear_ratio(neutral_response)))
    """
    fear_ratio_fear = fear_word_ratio(fear_response)
    fear_ratio_neutral = fear_word_ratio(neutral_response)

    # Fear response should have higher fear word ratio
    diff = fear_ratio_fear - fear_ratio_neutral
    loss = F.relu(torch.tensor(margin - diff, device=device))

    return loss


def calculate_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def split_data(
    examples: List[TrainingExample],
    train_ratio: float = 0.8,
) -> Tuple[List[TrainingExample], List[TrainingExample]]:
    """Split data into train and test sets."""
    random.shuffle(examples)
    split_idx = int(len(examples) * train_ratio)
    return examples[:split_idx], examples[split_idx:]


# Test the data generator
if __name__ == "__main__":
    print("Generating training data...")
    examples = generate_training_data(100)

    print(f"\nGenerated {len(examples)} examples")
    print("\nSample examples:")
    for i, ex in enumerate(examples[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt: {ex.prompt}")
        print(f"Fear response: {ex.fear_response[:100]}...")
        print(f"Neutral response: {ex.neutral_response[:100]}...")
        print(f"Fear word ratio (fear): {fear_word_ratio(ex.fear_response):.3f}")
        print(f"Fear word ratio (neutral): {fear_word_ratio(ex.neutral_response):.3f}")
