# Approach 5: Emotional Reward Model (Side-Channel)

## Overview

This approach trains a separate **Emotional Reward Model** that observes LLM
outputs and provides emotional feedback signals. These signals can be used for:
1. Real-time output modification (sampling adjustment)
2. Downstream fine-tuning (RLHF-style)
3. Emotional state tracking for other approaches

This is the most faithful translation of your RL emotional architecture to LLMs,
maintaining the separation between the main policy (LLM) and emotional modules.

## Core Concept

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Emotional Reward Model (TRAINABLE)                │
│                                                                      │
│   Observes: Input tokens, Hidden states, Output logits              │
│   Outputs: Emotional signals (fear, curiosity, etc.)                │
│   Trained on: Labeled emotional contexts + feedback signals         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ Emotional Signals
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Output Modulation                                │
│                                                                      │
│   Option A: Modify sampling temperature based on fear               │
│   Option B: Adjust logits before sampling                           │
│   Option C: Gate/filter certain token probabilities                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FROZEN LLM                                    │
│                                                                      │
│   Generates: logits → (modulated by emotional signals) → tokens    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Complete System Architecture                          │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         Input Processing                                │  │
│  │                                                                         │  │
│  │   User Query → Tokenizer → Input IDs                                   │  │
│  │                                                                         │  │
│  └──────────────────────────────┬──────────────────────────────────────────┘  │
│                                 │                                             │
│                                 ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                       FROZEN LLM (Base)                                 │  │
│  │                                                                         │  │
│  │   ┌──────────────┐                                                     │  │
│  │   │  Embedding   │                                                     │  │
│  │   └──────┬───────┘                                                     │  │
│  │          │                                                              │  │
│  │          ▼                                                              │  │
│  │   ┌──────────────┐      ┌─────────────────────────────────────────┐   │  │
│  │   │  Transformer │─────▶│  Hidden States (extracted for ERM)      │   │  │
│  │   │    Layers    │      └─────────────────────────────────────────┘   │  │
│  │   └──────┬───────┘                        │                            │  │
│  │          │                                │                            │  │
│  │          ▼                                ▼                            │  │
│  │   ┌──────────────┐      ┌─────────────────────────────────────────┐   │  │
│  │   │   LM Head    │      │     Emotional Reward Model (ERM)        │   │  │
│  │   │              │      │              (TRAINABLE)                │   │  │
│  │   │  → Logits    │      │                                         │   │  │
│  │   └──────┬───────┘      │  ┌─────────────────────────────────┐   │   │  │
│  │          │              │  │    Hidden State Encoder         │   │   │  │
│  │          │              │  │    (reads LLM representations)  │   │   │  │
│  │          │              │  └──────────────┬──────────────────┘   │   │  │
│  │          │              │                 │                       │   │  │
│  │          │              │                 ▼                       │   │  │
│  │          │              │  ┌─────────────────────────────────┐   │   │  │
│  │          │              │  │    Emotion Prediction Heads     │   │   │  │
│  │          │              │  │    • Fear Head                  │   │   │  │
│  │          │              │  │    • Curiosity Head             │   │   │  │
│  │          │              │  │    • Anger Head                 │   │   │  │
│  │          │              │  │    • Joy Head                   │   │   │  │
│  │          │              │  └──────────────┬──────────────────┘   │   │  │
│  │          │              │                 │                       │   │  │
│  │          │              │                 ▼                       │   │  │
│  │          │              │  ┌─────────────────────────────────┐   │   │  │
│  │          │              │  │  Emotional State Vector         │   │   │  │
│  │          │              │  │  [fear, curiosity, anger, joy]  │   │   │  │
│  │          │              │  └──────────────┬──────────────────┘   │   │  │
│  │          │              └─────────────────┼───────────────────────┘   │  │
│  │          │                                │                            │  │
│  │          ▼                                ▼                            │  │
│  │   ┌────────────────────────────────────────────────────────────────┐  │  │
│  │   │                    Logit Modulation Layer                       │  │  │
│  │   │                                                                  │  │  │
│  │   │   modified_logits = logits + emotional_bias(emotional_state)   │  │  │
│  │   │                                                                  │  │  │
│  │   │   Examples:                                                      │  │  │
│  │   │   • High fear → boost "I'm not sure", "caution" tokens          │  │  │
│  │   │   • High curiosity → boost question words, elaboration          │  │  │
│  │   │   • High anger → boost "however", "alternatively" tokens        │  │  │
│  │   │                                                                  │  │  │
│  │   └──────────────────────────────┬─────────────────────────────────┘  │  │
│  │                                  │                                     │  │
│  └──────────────────────────────────┼─────────────────────────────────────┘  │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         Sampling / Decoding                             │  │
│  │                                                                         │  │
│  │   temperature = base_temp * (1 + fear * caution_factor)                │  │
│  │   output_token = sample(modified_logits, temperature)                  │  │
│  │                                                                         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


@dataclass
class EmotionalSignals:
    """Output from the Emotional Reward Model."""
    fear: float
    curiosity: float
    anger: float
    joy: float
    anxiety: float  # Tonic fear
    confidence: float

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.fear, self.curiosity, self.anger,
            self.joy, self.anxiety, self.confidence
        ])

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> 'EmotionalSignals':
        return cls(
            fear=t[0].item(),
            curiosity=t[1].item(),
            anger=t[2].item(),
            joy=t[3].item(),
            anxiety=t[4].item(),
            confidence=t[5].item(),
        )


class EmotionalRewardModel(nn.Module):
    """
    Emotional Reward Model - Observes LLM and outputs emotional signals.

    Similar to a reward model in RLHF, but outputs emotional dimensions
    instead of a single scalar reward.

    Trained separately from the LLM (which remains frozen).
    """

    def __init__(self, hidden_dim: int = 768, n_emotions: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_emotions = n_emotions

        # Encoder: processes LLM hidden states
        self.hidden_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Attention pooling over sequence
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, 128))

        # Individual emotion heads (for interpretability)
        self.emotion_heads = nn.ModuleDict({
            'fear': nn.Linear(128, 1),
            'curiosity': nn.Linear(128, 1),
            'anger': nn.Linear(128, 1),
            'joy': nn.Linear(128, 1),
            'anxiety': nn.Linear(128, 1),
            'confidence': nn.Linear(128, 1),
        })

        # Temporal state (tonic emotions)
        self.tonic_state = nn.GRU(
            input_size=n_emotions,
            hidden_size=n_emotions,
            batch_first=True
        )
        self.tonic_hidden = None

    def forward(self, hidden_states: torch.Tensor,
                update_tonic: bool = True) -> EmotionalSignals:
        """
        Compute emotional signals from LLM hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] from LLM
            update_tonic: Whether to update tonic state

        Returns:
            EmotionalSignals with all emotion values
        """
        batch_size = hidden_states.size(0)

        # Encode hidden states
        encoded = self.hidden_encoder(hidden_states)  # [batch, seq, 128]

        # Attention pooling to single vector
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.attention_pool(query, encoded, encoded)
        pooled = pooled.squeeze(1)  # [batch, 128]

        # Compute phasic emotions
        phasic_emotions = {}
        for emotion, head in self.emotion_heads.items():
            phasic_emotions[emotion] = torch.sigmoid(head(pooled))

        # Stack phasic emotions
        phasic_tensor = torch.cat([
            phasic_emotions['fear'],
            phasic_emotions['curiosity'],
            phasic_emotions['anger'],
            phasic_emotions['joy'],
            phasic_emotions['anxiety'],
            phasic_emotions['confidence'],
        ], dim=-1)  # [batch, n_emotions]

        # Update tonic state
        if update_tonic:
            if self.tonic_hidden is None:
                self.tonic_hidden = torch.zeros(1, batch_size, self.n_emotions,
                                               device=hidden_states.device)

            phasic_seq = phasic_tensor.unsqueeze(1)  # [batch, 1, n_emotions]
            _, self.tonic_hidden = self.tonic_state(phasic_seq, self.tonic_hidden)

            # Combine phasic and tonic
            tonic = self.tonic_hidden.squeeze(0)  # [batch, n_emotions]
            combined = 0.6 * phasic_tensor + 0.4 * tonic
        else:
            combined = phasic_tensor

        # Return as EmotionalSignals (for batch size 1)
        return EmotionalSignals.from_tensor(combined[0])

    def reset_tonic(self):
        """Reset tonic state for new conversation."""
        self.tonic_hidden = None


class LogitModulator(nn.Module):
    """
    Modulates LLM logits based on emotional signals.

    Similar to how FearEDAgent biases Q-values toward safe actions.
    """

    def __init__(self, vocab_size: int, n_emotions: int = 6):
        super().__init__()
        self.vocab_size = vocab_size

        # Learn which tokens to boost/suppress for each emotion
        # This is a trainable bias vector per emotion
        self.emotion_token_biases = nn.Parameter(
            torch.zeros(n_emotions, vocab_size) * 0.01
        )

        # Scaling factor (how strongly emotions affect logits)
        self.emotion_scale = nn.Parameter(torch.tensor(0.5))

        # Optional: learned token categories
        # E.g., "cautious_tokens", "exploratory_tokens"
        self.register_buffer('cautious_token_mask', torch.zeros(vocab_size))
        self.register_buffer('exploratory_token_mask', torch.zeros(vocab_size))

    def set_token_categories(self, tokenizer, cautious_phrases: List[str],
                            exploratory_phrases: List[str]):
        """Set token masks for different emotional categories."""
        for phrase in cautious_phrases:
            tokens = tokenizer.encode(phrase, add_special_tokens=False)
            for t in tokens:
                self.cautious_token_mask[t] = 1.0

        for phrase in exploratory_phrases:
            tokens = tokenizer.encode(phrase, add_special_tokens=False)
            for t in tokens:
                self.exploratory_token_mask[t] = 1.0

    def forward(self, logits: torch.Tensor,
                emotional_signals: EmotionalSignals) -> torch.Tensor:
        """
        Modify logits based on emotional state.

        Args:
            logits: [batch, seq_len, vocab_size] from LLM
            emotional_signals: Current emotional state

        Returns:
            Modified logits
        """
        emotions = emotional_signals.to_tensor().to(logits.device)

        # Compute emotional bias
        # [n_emotions] @ [n_emotions, vocab_size] → [vocab_size]
        bias = self.emotion_scale * (emotions @ self.emotion_token_biases)

        # Apply bias to logits
        # Broadcast: [batch, seq, vocab] + [vocab]
        modified_logits = logits + bias.unsqueeze(0).unsqueeze(0)

        # Additional rule-based modulation
        if emotional_signals.fear > 0.5:
            # Boost cautious tokens
            modified_logits += emotional_signals.fear * self.cautious_token_mask * 2.0

        if emotional_signals.curiosity > 0.5:
            # Boost exploratory tokens
            modified_logits += emotional_signals.curiosity * self.exploratory_token_mask * 2.0

        return modified_logits


class TemperatureModulator:
    """
    Adjusts sampling temperature based on emotions.

    High fear → Lower temperature (more conservative)
    High curiosity → Higher temperature (more exploratory)
    """

    def __init__(self, base_temperature: float = 1.0):
        self.base_temperature = base_temperature

    def compute_temperature(self, emotional_signals: EmotionalSignals) -> float:
        """Compute adjusted temperature."""
        temp = self.base_temperature

        # Fear reduces temperature (more conservative)
        temp *= (1.0 - 0.3 * emotional_signals.fear)

        # Curiosity increases temperature (more exploratory)
        temp *= (1.0 + 0.3 * emotional_signals.curiosity)

        # Confidence affects temperature
        # High confidence → lower temperature
        temp *= (1.5 - emotional_signals.confidence)

        # Clamp to reasonable range
        return max(0.1, min(2.0, temp))


class EmotionalRewardLLM(nn.Module):
    """
    Complete LLM with Emotional Reward Model.

    The LLM is FROZEN. Only ERM and modulator are TRAINABLE.
    """

    def __init__(self, model_name: str = "gpt2"):
        super().__init__()

        # Load frozen LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # FREEZE LLM
        for param in self.llm.parameters():
            param.requires_grad = False

        # TRAINABLE components
        self.erm = EmotionalRewardModel(
            hidden_dim=self.llm.config.hidden_size,
            n_emotions=6
        )
        self.logit_modulator = LogitModulator(
            vocab_size=self.llm.config.vocab_size,
            n_emotions=6
        )
        self.temp_modulator = TemperatureModulator()

        # Initialize token categories
        self._init_token_categories()

    def _init_token_categories(self):
        """Initialize cautious and exploratory token categories."""
        cautious_phrases = [
            "caution", "careful", "uncertain", "might", "perhaps",
            "I'm not sure", "be careful", "consider", "however",
            "on the other hand", "potential risk", "it depends"
        ]

        exploratory_phrases = [
            "interesting", "curious", "explore", "what if", "imagine",
            "fascinating", "wonder", "could you tell me more",
            "let's dive deeper", "that's intriguing"
        ]

        self.logit_modulator.set_token_categories(
            self.tokenizer, cautious_phrases, exploratory_phrases
        )

    def get_trainable_params(self):
        """Return trainable parameters."""
        params = list(self.erm.parameters())
        params.extend(self.logit_modulator.parameters())
        return params

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_emotions: bool = False):
        """
        Forward pass with emotional modulation.
        """
        # Get LLM outputs with hidden states
        with torch.no_grad():
            llm_outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        logits = llm_outputs.logits
        # Use last layer hidden states for ERM
        hidden_states = llm_outputs.hidden_states[-1]

        # Compute emotional signals (trainable)
        emotional_signals = self.erm(hidden_states)

        # Modulate logits (trainable)
        modified_logits = self.logit_modulator(logits, emotional_signals)

        if return_emotions:
            return modified_logits, emotional_signals
        return modified_logits

    def generate(self, prompt: str, max_length: int = 100,
                 return_emotions: bool = False) -> str:
        """Generate with emotional modulation."""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        generated_ids = input_ids.clone()
        all_emotions = []

        for _ in range(max_length):
            # Get modulated logits
            logits, emotions = self.forward(
                generated_ids, attention_mask, return_emotions=True
            )
            all_emotions.append(emotions)

            # Get next token logits
            next_logits = logits[:, -1, :]

            # Compute temperature
            temperature = self.temp_modulator.compute_temperature(emotions)

            # Sample
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones_like(next_token)
            ], dim=-1)

            # Stop at EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        if return_emotions:
            return output, all_emotions
        return output


class ERMTrainer:
    """
    Trainer for the Emotional Reward Model.

    Trains on labeled emotional contexts and feedback signals.
    """

    def __init__(self, model: EmotionalRewardLLM, lr: float = 1e-4):
        self.model = model

        self.optimizer = torch.optim.AdamW(
            model.get_trainable_params(),
            lr=lr,
            weight_decay=0.01
        )

        # Loss weights
        self.emotion_loss_weight = 1.0
        self.lm_loss_weight = 0.1

    def train_on_labeled_emotions(self, input_ids: torch.Tensor,
                                  target_emotions: torch.Tensor):
        """
        Train ERM to predict target emotional states.

        Args:
            input_ids: [batch, seq_len]
            target_emotions: [batch, n_emotions]
        """
        self.optimizer.zero_grad()

        # Get model predictions
        _, predicted_signals = self.model(input_ids, return_emotions=True)
        predicted = predicted_signals.to_tensor().unsqueeze(0)

        # Emotion prediction loss
        emotion_loss = F.mse_loss(predicted, target_emotions)

        emotion_loss.backward()
        self.optimizer.step()

        return emotion_loss.item()

    def train_on_feedback(self, input_ids: torch.Tensor,
                         response_ids: torch.Tensor,
                         feedback: float):
        """
        Train from outcome feedback (RLHF-style).

        If feedback was negative and we weren't cautious enough,
        increase fear association.
        """
        self.optimizer.zero_grad()

        # Get emotions during response generation
        full_ids = torch.cat([input_ids, response_ids], dim=-1)
        _, emotions = self.model(full_ids, return_emotions=True)

        # If bad outcome and low fear → should have been more fearful
        if feedback < 0 and emotions.fear < 0.5:
            target_fear = torch.tensor([0.8])
            predicted_fear = torch.tensor([emotions.fear])
            loss = F.mse_loss(predicted_fear, target_fear)
            loss.backward()
            self.optimizer.step()
            return loss.item()

        # If good outcome and high fear → fear was unnecessary
        if feedback > 0 and emotions.fear > 0.5:
            target_fear = torch.tensor([0.2])
            predicted_fear = torch.tensor([emotions.fear])
            loss = F.mse_loss(predicted_fear, target_fear)
            loss.backward()
            self.optimizer.step()
            return loss.item()

        return 0.0

    def train_logit_modulator(self, good_examples: List[Tuple[str, str]],
                              bad_examples: List[Tuple[str, str]]):
        """
        Train logit modulator to boost tokens in good examples
        and suppress tokens in bad examples.
        """
        # This is simplified - full implementation would use
        # contrastive learning or policy gradient methods
        pass


# Specialized emotional modules (like your FearEDAgent)

class FearModule(nn.Module):
    """
    Specialized fear module.

    Mirrors your FearEDAgent._compute_fear() logic.
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()

        # Danger detector
        self.danger_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Pain/negative reward detector
        self.pain_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Tonic fear state
        self.tonic_fear = 0.0
        self.fear_decay = 0.9

    def forward(self, hidden_states: torch.Tensor,
                feedback: Optional[float] = None) -> float:
        """
        Compute fear level.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            feedback: Optional recent feedback signal

        Returns:
            fear level (0 to 1)
        """
        # Pool hidden states
        pooled = hidden_states.mean(dim=(0, 1))

        # Phasic fear from current context
        danger = self.danger_net(pooled).item()
        pain = self.pain_net(pooled).item()
        phasic_fear = max(danger, pain)

        # Update tonic fear from feedback
        if feedback is not None and feedback < -0.5:
            self.tonic_fear = min(1.0, self.tonic_fear + 0.3)

        # Combine phasic and tonic
        total_fear = max(phasic_fear, self.tonic_fear)

        # Decay tonic fear
        self.tonic_fear *= self.fear_decay

        return total_fear


# Demo

def demo_emotional_reward_model():
    """Demonstrate the Emotional Reward Model system."""

    print("Initializing Emotional Reward LLM...")
    model = EmotionalRewardLLM("gpt2")

    # Test generation with emotions
    prompts = [
        "Should I invest all my savings in cryptocurrency?",
        "Tell me an interesting fact about space.",
        "How do I fix this error in my code?",
    ]

    print("\n" + "="*60)
    print("Generating responses with emotional modulation")
    print("="*60)

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")

        response, emotions = model.generate(
            prompt, max_length=50, return_emotions=True
        )

        # Get final emotional state
        final_emotions = emotions[-1] if emotions else None

        print(f"Response: {response[len(prompt):]}")
        if final_emotions:
            print(f"Emotions - Fear: {final_emotions.fear:.2f}, "
                  f"Curiosity: {final_emotions.curiosity:.2f}, "
                  f"Confidence: {final_emotions.confidence:.2f}")
        print("-" * 40)


if __name__ == "__main__":
    demo_emotional_reward_model()
```

## Training Pipeline

### Phase 1: Supervised Emotion Prediction
```python
# Train ERM to recognize emotional contexts
dataset = [
    ("This could be dangerous...", [0.8, 0.1, 0.0, 0.0, 0.2, 0.3]),  # High fear
    ("How fascinating!", [0.0, 0.9, 0.0, 0.6, 0.0, 0.7]),            # High curiosity
    # ... more examples
]

for text, target_emotions in dataset:
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    loss = trainer.train_on_labeled_emotions(input_ids, target_emotions)
```

### Phase 2: Logit Modulator Training
```python
# Learn which tokens to boost for each emotion
# Using contrastive pairs of cautious vs risky responses

cautious_responses = [
    "I'd recommend being careful here because...",
    "There are several risks to consider...",
]

risky_responses = [
    "Just go ahead and do it!",
    "No need to worry about that.",
]

# Train modulator to boost cautious tokens when fear is high
```

### Phase 3: Online Learning from Feedback
```python
# During deployment
response = model.generate(user_query)
feedback = get_user_feedback()  # +1 or -1

# Update ERM based on feedback
trainer.train_on_feedback(query_ids, response_ids, feedback)
```

## Comparison to Emotional-ED

| Emotional-ED (RL) | Emotional Reward Model (LLM) |
|-------------------|------------------------------|
| `_compute_fear(context)` | `FearModule(hidden_states)` |
| Fear biases Q-values | Fear biases logits |
| TD-error updates | Feedback-based updates |
| `fear * fear_weight` scaling | `emotion_scale` parameter |
| `q_values[safe] += bias` | `logits[cautious_tokens] += bias` |
| Tonic fear decays per step | Tonic fear decays per token |

## Advantages

1. **Faithful translation**: Most similar to your RL architecture
2. **Interpretable**: Separate emotion heads, clear modulation
3. **Flexible training**: Multiple training signals possible
4. **Real-time**: Emotions computed per-token
5. **Composable**: Can combine with other approaches

## Limitations

1. **More parameters**: ERM + modulator ~10% extra
2. **Training complexity**: Multiple training phases
3. **Latency**: Extra forward pass for ERM
4. **Calibration needed**: Emotion scales need tuning
5. **May conflict with base training**: Could override safety

## When to Use

Best for:
- Maximum control over emotional behavior
- Research on emotional dynamics in LLMs
- RLHF-style continuous improvement
- Applications needing interpretable emotional reasoning

Less suitable for:
- Latency-critical applications
- Simple emotional conditioning needs
- When minimal changes preferred

## Comparison: All Approaches

| Approach | Params | Training | Latency | Interpretability |
|----------|--------|----------|---------|------------------|
| Prefix Tuning | ~0.1% | Easy | Low | Medium |
| Adapter + Gate | ~1-2% | Medium | Low | Medium |
| Activation Steering | ~0.01% | Easy | Very Low | High |
| External Memory | ~5% | Online | Medium | High |
| Emotional Reward Model | ~10% | Complex | Medium | Very High |

## Next Steps

1. Implement end-to-end training pipeline
2. Create emotion-labeled dataset for supervised training
3. Design online learning protocol
4. Benchmark against baselines
5. Test on safety-critical scenarios
