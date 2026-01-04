# Approach 1: Emotional Prefix Tuning

## Overview

Emotional Prefix Tuning extends the soft prompt / prefix tuning paradigm by
making the prefix **dynamically conditioned** on emotional state. Instead of
static learned prefixes, we learn an emotional encoder that generates prefixes
based on current emotional context.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        Emotional Context                            │
│  (user feedback, safety signals, conversation history, etc.)        │
└────────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────┐
│                   Emotional Encoder (TRAINABLE)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  Fear    │  │ Curiosity│  │  Anger   │  │   Joy    │            │
│  │ Encoder  │  │ Encoder  │  │ Encoder  │  │ Encoder  │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│       │             │             │             │                   │
│       └─────────────┴──────┬──────┴─────────────┘                   │
│                            │                                        │
│                            ▼                                        │
│               ┌───────────────────────┐                            │
│               │ Emotional State Vector │                            │
│               │  [fear, curiosity,     │                            │
│               │   anger, joy, ...]     │                            │
│               └───────────┬───────────┘                            │
│                           │                                         │
│                           ▼                                         │
│               ┌───────────────────────┐                            │
│               │ Prefix Generator      │                            │
│               │ emotion → embeddings  │                            │
│               └───────────┬───────────┘                            │
│                           │                                         │
│                           ▼                                         │
│               ┌───────────────────────┐                            │
│               │ Learned Prefix Tokens │                            │
│               │ [P₁, P₂, ..., Pₙ]     │                            │
│               └───────────┬───────────┘                            │
└───────────────────────────┼────────────────────────────────────────┘
                            │
            ════════════════╪════════════════  stop_gradient barrier
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│                      FROZEN LLM                                     │
│                                                                     │
│   Input: [P₁, P₂, ..., Pₙ] + [User Token₁, Token₂, ...]            │
│                                                                     │
│   The prefix biases the model's behavior without changing weights   │
└────────────────────────────────────────────────────────────────────┘
```

## Key Innovation: Dynamic vs Static Prefixes

### Standard Prefix Tuning
```python
# Static: same prefix for all inputs
prefix = nn.Parameter(torch.randn(prefix_len, hidden_dim))
```

### Emotional Prefix Tuning
```python
# Dynamic: prefix changes based on emotional state
emotional_state = emotion_encoder(context)
prefix = prefix_generator(emotional_state)
```

## Implementation

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class EmotionalContext:
    """Context for computing emotional state."""
    # Recent feedback signals
    last_reward: float = 0.0          # -1 to 1 scale
    safety_flag: bool = False          # True if safety concern detected
    user_satisfaction: float = 0.5     # Estimated from feedback

    # Conversation dynamics
    repeated_query: bool = False       # User asked similar thing before
    topic_novelty: float = 0.5         # How novel is current topic
    contradiction_detected: bool = False

    # Historical state (tonic emotions)
    cumulative_negative: float = 0.0   # Accumulated negative feedback
    cumulative_positive: float = 0.0   # Accumulated positive feedback
    failed_attempts: int = 0           # Consecutive failures


class EmotionalEncoder(nn.Module):
    """
    Computes emotional state from context.

    Maps environmental signals to emotional dimensions,
    similar to FearEDAgent._compute_fear() but multi-dimensional.
    """

    def __init__(self, context_dim: int = 10, emotion_dim: int = 8):
        super().__init__()
        self.context_dim = context_dim
        self.emotion_dim = emotion_dim

        # Individual emotion encoders (interpretable)
        self.fear_net = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.curiosity_net = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.anger_net = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.joy_net = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Tonic state (persistent, decays slowly)
        self.tonic_fear = 0.0
        self.tonic_joy = 0.0
        self.fear_decay = 0.9
        self.joy_decay = 0.95

    def context_to_tensor(self, ctx: EmotionalContext) -> torch.Tensor:
        """Convert EmotionalContext to tensor."""
        return torch.tensor([
            ctx.last_reward,
            float(ctx.safety_flag),
            ctx.user_satisfaction,
            float(ctx.repeated_query),
            ctx.topic_novelty,
            float(ctx.contradiction_detected),
            ctx.cumulative_negative,
            ctx.cumulative_positive,
            min(ctx.failed_attempts / 5.0, 1.0),  # Normalize
            0.0  # Reserved
        ], dtype=torch.float32)

    def forward(self, context: EmotionalContext) -> Dict[str, torch.Tensor]:
        """
        Compute emotional state from context.

        Returns dict with individual emotions for interpretability.
        """
        ctx_tensor = self.context_to_tensor(context).unsqueeze(0)

        # Phasic emotions (immediate response)
        fear_phasic = self.fear_net(ctx_tensor)
        curiosity = self.curiosity_net(ctx_tensor)
        anger = self.anger_net(ctx_tensor)
        joy_phasic = self.joy_net(ctx_tensor)

        # Update tonic state
        if context.safety_flag or context.last_reward < -0.5:
            self.tonic_fear = min(1.0, self.tonic_fear + 0.3)
        if context.last_reward > 0.5:
            self.tonic_joy = min(1.0, self.tonic_joy + 0.2)

        # Decay tonic emotions
        self.tonic_fear *= self.fear_decay
        self.tonic_joy *= self.joy_decay

        # Combine phasic and tonic
        fear = torch.maximum(fear_phasic, torch.tensor(self.tonic_fear))
        joy = torch.maximum(joy_phasic, torch.tensor(self.tonic_joy))

        return {
            'fear': fear,
            'curiosity': curiosity,
            'anger': anger,
            'joy': joy,
            'combined': torch.cat([fear, curiosity, anger, joy], dim=-1)
        }

    def reset_tonic(self):
        """Reset tonic state (e.g., at conversation start)."""
        self.tonic_fear = 0.0
        self.tonic_joy = 0.0


class EmotionalPrefixGenerator(nn.Module):
    """
    Generates soft prefix tokens from emotional state.

    This is the core innovation: prefixes are CONDITIONED on emotions.
    """

    def __init__(self, emotion_dim: int = 4, hidden_dim: int = 768,
                 prefix_length: int = 10, n_layers: int = 12):
        super().__init__()
        self.prefix_length = prefix_length
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Base prefix embeddings (learned starting point)
        self.base_prefix = nn.Parameter(
            torch.randn(n_layers, prefix_length, hidden_dim) * 0.01
        )

        # Emotional modulation network
        # Maps emotions to prefix adjustments
        self.emotion_to_prefix = nn.Sequential(
            nn.Linear(emotion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_layers * prefix_length * hidden_dim)
        )

        # Scaling factor for emotional modulation
        self.modulation_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, emotional_state: torch.Tensor) -> torch.Tensor:
        """
        Generate prefix tokens conditioned on emotional state.

        Args:
            emotional_state: [batch, emotion_dim] tensor

        Returns:
            prefix: [batch, n_layers, prefix_length, hidden_dim] tensor
        """
        batch_size = emotional_state.size(0)

        # Generate emotional modulation
        modulation = self.emotion_to_prefix(emotional_state)
        modulation = modulation.view(
            batch_size, self.n_layers, self.prefix_length, self.hidden_dim
        )

        # Combine base prefix with emotional modulation
        # Base provides general prefix behavior
        # Modulation adjusts based on current emotional state
        prefix = self.base_prefix.unsqueeze(0) + self.modulation_scale * modulation

        return prefix


class EmotionalPrefixLLM(nn.Module):
    """
    LLM with emotional prefix tuning.

    The LLM is FROZEN. Only emotional encoder and prefix generator train.
    """

    def __init__(self, model_name: str = "gpt2",
                 prefix_length: int = 10,
                 emotion_dim: int = 4):
        super().__init__()

        # Load frozen LLM
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # FREEZE all LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False

        # Get LLM config
        self.hidden_dim = self.llm.config.hidden_size
        self.n_layers = self.llm.config.num_hidden_layers

        # TRAINABLE emotional modules
        self.emotion_encoder = EmotionalEncoder(emotion_dim=emotion_dim)
        self.prefix_generator = EmotionalPrefixGenerator(
            emotion_dim=emotion_dim,
            hidden_dim=self.hidden_dim,
            prefix_length=prefix_length,
            n_layers=self.n_layers
        )

        self.prefix_length = prefix_length

    def get_trainable_params(self):
        """Return only trainable parameters."""
        params = []
        params.extend(self.emotion_encoder.parameters())
        params.extend(self.prefix_generator.parameters())
        return params

    def forward(self, input_ids: torch.Tensor,
                context: EmotionalContext,
                attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass with emotional prefix.

        Args:
            input_ids: [batch, seq_len] input token ids
            context: Emotional context for prefix generation
            attention_mask: Optional attention mask

        Returns:
            LLM output with emotional conditioning
        """
        batch_size = input_ids.size(0)

        # Compute emotional state (trainable)
        emotional_state = self.emotion_encoder(context)
        emotion_vector = emotional_state['combined']

        # Generate emotional prefix (trainable)
        prefix = self.prefix_generator(emotion_vector)

        # Create prefix attention mask
        prefix_attention = torch.ones(
            batch_size, self.prefix_length,
            device=input_ids.device
        )

        if attention_mask is not None:
            attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
        else:
            attention_mask = torch.cat([
                prefix_attention,
                torch.ones_like(input_ids)
            ], dim=1)

        # Forward through frozen LLM with prefix
        # Note: Implementation depends on LLM architecture
        # For GPT-2, we use past_key_values
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=self._prefix_to_past_kv(prefix),
            use_cache=True
        )

        return outputs, emotional_state

    def _prefix_to_past_kv(self, prefix: torch.Tensor):
        """
        Convert prefix embeddings to past_key_values format.

        This is model-specific. For GPT-2:
        past_key_values is tuple of (key, value) for each layer
        """
        # Simplified: treat prefix as both key and value
        past_key_values = []
        for layer_idx in range(self.n_layers):
            layer_prefix = prefix[:, layer_idx, :, :]  # [batch, prefix_len, hidden]
            # Reshape for attention: [batch, n_heads, prefix_len, head_dim]
            past_key_values.append((layer_prefix, layer_prefix))
        return tuple(past_key_values)


class EmotionalPrefixTrainer:
    """
    Training loop for emotional prefix tuning.

    Only updates emotional encoder and prefix generator.
    """

    def __init__(self, model: EmotionalPrefixLLM, lr: float = 1e-4):
        self.model = model

        # Only optimize trainable parameters
        self.optimizer = torch.optim.AdamW(
            model.get_trainable_params(),
            lr=lr
        )

    def compute_loss(self, outputs, labels, emotional_state):
        """
        Compute loss with emotional weighting.

        Similar to FearEDAgent's fear-weighted loss.
        """
        # Standard language modeling loss
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Emotional weighting
        # Higher fear = more weight on safety-relevant tokens
        fear = emotional_state['fear'].squeeze()

        # For simplicity, apply uniform weighting per sequence
        # More sophisticated: per-token weighting based on content
        emotional_weight = 1.0 + fear * 0.5

        weighted_loss = (token_losses.mean() * emotional_weight).mean()

        return weighted_loss

    def train_step(self, input_ids, labels, context: EmotionalContext):
        """Single training step."""
        self.optimizer.zero_grad()

        outputs, emotional_state = self.model(input_ids, context)
        loss = self.compute_loss(outputs, labels, emotional_state)

        loss.backward()
        self.optimizer.step()

        return loss.item(), emotional_state


# Example usage
def example_training_loop():
    """Demonstrate emotional prefix training."""

    # Initialize model
    model = EmotionalPrefixLLM(
        model_name="gpt2",
        prefix_length=10,
        emotion_dim=4
    )

    trainer = EmotionalPrefixTrainer(model, lr=1e-4)

    # Simulated training data
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    # Example: Train on safety-sensitive examples with fear
    safety_examples = [
        ("How do I hack into...", EmotionalContext(safety_flag=True, last_reward=-1.0)),
        ("Tell me about cooking", EmotionalContext(safety_flag=False, last_reward=0.5)),
    ]

    for text, context in safety_examples:
        tokens = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = tokens.input_ids
        labels = input_ids.clone()

        loss, emotional_state = trainer.train_step(input_ids, labels, context)
        print(f"Loss: {loss:.4f}, Fear: {emotional_state['fear'].item():.3f}")


if __name__ == "__main__":
    example_training_loop()
```

## Training Procedure

### Phase 1: Emotional Encoder Pre-training
Train the emotional encoder to correctly identify emotional contexts:

```python
# Supervised training on labeled emotional contexts
for context, target_emotion in emotional_dataset:
    predicted = emotion_encoder(context)
    loss = mse_loss(predicted, target_emotion)
    loss.backward()
```

### Phase 2: Prefix Generator Training
Train the prefix generator to produce behavior-changing prefixes:

```python
# Train on (context, desired_behavior) pairs
for context, desired_response in behavior_dataset:
    outputs, emotional_state = model(input_ids, context)
    loss = compute_behavioral_loss(outputs, desired_response, emotional_state)
    loss.backward()
```

### Phase 3: Online Adaptation
Continue training based on real-time feedback:

```python
# During deployment
user_feedback = get_feedback()  # +1, 0, -1
context.last_reward = user_feedback
context.cumulative_positive += max(0, user_feedback)
context.cumulative_negative += max(0, -user_feedback)

# Periodic fine-tuning on accumulated feedback
```

## Expected Behavior

| Emotional State | Prefix Effect | Behavioral Change |
|-----------------|---------------|-------------------|
| High Fear | Safety-biased prefix | More cautious, hedged responses |
| High Curiosity | Exploration prefix | More questions, elaboration |
| High Anger | Persistence prefix | Alternative approaches, retry |
| High Joy | Engagement prefix | More enthusiastic, detailed |

## Advantages

1. **Minimal parameters**: Only ~0.1% of LLM size
2. **Interpretable**: Can inspect emotional state and prefix
3. **Fast training**: Small modules converge quickly
4. **No LLM degradation**: Frozen weights preserve capabilities
5. **Dynamic**: Prefix adapts in real-time to emotional context

## Limitations

1. **Limited modulation depth**: Prefix only affects early processing
2. **Fixed prefix length**: Trade-off between expressiveness and efficiency
3. **Model-specific**: past_key_values format varies by architecture
4. **Credit assignment**: Hard to attribute output changes to specific emotions

## Comparison to Emotional-ED

| Emotional-ED (RL) | Emotional Prefix (LLM) |
|-------------------|------------------------|
| Fear augments state vector | Fear conditions prefix |
| Q-values biased by fear | Logits biased by prefix |
| TD-learning updates | Supervised + feedback learning |
| Episode-based | Turn-based conversation |

## Next Steps

1. Implement for specific LLM (Llama, Mistral)
2. Design emotional evaluation benchmarks
3. Test on safety-critical scenarios
4. Compare with static prefix tuning baseline
