# Approach 2: Adapter Layers with Emotional Gating

## Overview

This approach inserts small trainable adapter modules between frozen LLM layers,
where the adapter output is **gated by emotional state**. This provides deeper
integration than prefix tuning, allowing emotional modulation at every layer.

## Core Concept

Standard LoRA/Adapter:
```
hidden = frozen_layer(x)
hidden = hidden + adapter(hidden)  # Fixed contribution
```

Emotional Adapter:
```
hidden = frozen_layer(x)
emotional_gate = compute_gate(emotional_state)
hidden = hidden + emotional_gate * adapter(hidden)  # Modulated contribution
```

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                           FROZEN LLM LAYER                              │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    Multi-Head Attention (Frozen)                │    │
│  └────────────────────────────────┬───────────────────────────────┘    │
│                                   │                                     │
│                                   ▼                                     │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                                                                 │    │
│  │   ┌─────────────────┐        ┌─────────────────────────────┐   │    │
│  │   │ Emotional State │───────▶│ Gate Network (TRAINABLE)    │   │    │
│  │   │ [fear, joy,...] │        │ g = σ(W_g · emotion)        │   │    │
│  │   └─────────────────┘        └──────────────┬──────────────┘   │    │
│  │                                              │                  │    │
│  │   ┌─────────────────┐                       │                  │    │
│  │   │ Hidden State    │──┬─────────────────────┼──────────────┐  │    │
│  │   │ from attention  │  │                     │              │  │    │
│  │   └─────────────────┘  │                     │              │  │    │
│  │                        ▼                     ▼              │  │    │
│  │            ┌───────────────────────┐   ┌─────────┐          │  │    │
│  │            │ Adapter (TRAINABLE)   │   │   g     │          │  │    │
│  │            │ down_proj → up_proj   │   │ (gate)  │          │  │    │
│  │            └───────────┬───────────┘   └────┬────┘          │  │    │
│  │                        │                     │              │  │    │
│  │                        │         ┌───────────┘              │  │    │
│  │                        ▼         ▼                          │  │    │
│  │                    ┌─────────────────┐                      │  │    │
│  │                    │  g * adapter_out│                      │  │    │
│  │                    └────────┬────────┘                      │  │    │
│  │                             │                               │  │    │
│  │                             ▼                               │  │    │
│  │                    ┌─────────────────┐                      │  │    │
│  │                    │  hidden + gated │◀─────────────────────┘  │    │
│  │                    └────────┬────────┘                         │    │
│  │                             │                                  │    │
│  └─────────────────────────────┼──────────────────────────────────┘    │
│                                │                                        │
│                                ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    Feed-Forward Network (Frozen)                │    │
│  └────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────┘
```

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class EmotionalState:
    """Current emotional state vector."""
    fear: float = 0.0
    curiosity: float = 0.0
    anger: float = 0.0
    joy: float = 0.0
    anxiety: float = 0.0  # Tonic fear
    confidence: float = 0.5

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.fear, self.curiosity, self.anger,
            self.joy, self.anxiety, self.confidence
        ], dtype=torch.float32)


class EmotionalGate(nn.Module):
    """
    Computes gating values from emotional state.

    Different emotions can have different effects:
    - Fear: reduces adapter contribution (conservative)
    - Curiosity: increases adapter contribution (exploratory)
    - Anger: modulates specific dimensions (persistence)
    """

    def __init__(self, emotion_dim: int = 6, hidden_dim: int = 768,
                 gate_type: str = "scalar"):
        super().__init__()
        self.gate_type = gate_type
        self.hidden_dim = hidden_dim

        if gate_type == "scalar":
            # Single scalar gate per layer
            self.gate_net = nn.Sequential(
                nn.Linear(emotion_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        elif gate_type == "vector":
            # Per-dimension gating
            self.gate_net = nn.Sequential(
                nn.Linear(emotion_dim, 64),
                nn.ReLU(),
                nn.Linear(64, hidden_dim),
                nn.Sigmoid()
            )
        elif gate_type == "attention":
            # Emotion-conditioned attention over hidden dims
            self.emotion_query = nn.Linear(emotion_dim, 64)
            self.hidden_key = nn.Linear(hidden_dim, 64)
            self.gate_proj = nn.Linear(64, 1)

    def forward(self, emotional_state: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute gate values.

        Args:
            emotional_state: [batch, emotion_dim]
            hidden: [batch, seq_len, hidden_dim] (for attention gate)

        Returns:
            gate: scalar, [hidden_dim], or [batch, seq_len, 1]
        """
        if self.gate_type == "scalar":
            return self.gate_net(emotional_state)

        elif self.gate_type == "vector":
            return self.gate_net(emotional_state)

        elif self.gate_type == "attention":
            # Emotion-conditioned attention
            q = self.emotion_query(emotional_state)  # [batch, 64]
            k = self.hidden_key(hidden)  # [batch, seq_len, 64]

            # Compute attention scores
            scores = torch.einsum('bd,bsd->bs', q, k) / math.sqrt(64)
            gate = torch.sigmoid(self.gate_proj(k))  # [batch, seq_len, 1]

            return gate


class EmotionalAdapter(nn.Module):
    """
    LoRA-style adapter with emotional gating.

    Structure: down_proj → activation → up_proj, gated by emotion
    """

    def __init__(self, hidden_dim: int = 768, adapter_dim: int = 64,
                 emotion_dim: int = 6, gate_type: str = "scalar",
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim

        # Adapter layers (trainable)
        self.down_proj = nn.Linear(hidden_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Emotional gate (trainable)
        self.gate = EmotionalGate(emotion_dim, hidden_dim, gate_type)

        # Initialize up_proj to zero for stable training start
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden: torch.Tensor,
                emotional_state: torch.Tensor) -> torch.Tensor:
        """
        Apply emotionally-gated adapter.

        Args:
            hidden: [batch, seq_len, hidden_dim]
            emotional_state: [batch, emotion_dim]

        Returns:
            modulated hidden: [batch, seq_len, hidden_dim]
        """
        # Adapter transformation
        adapter_out = self.down_proj(hidden)
        adapter_out = self.activation(adapter_out)
        adapter_out = self.dropout(adapter_out)
        adapter_out = self.up_proj(adapter_out)

        # Emotional gating
        gate = self.gate(emotional_state, hidden)

        # Apply gate
        if gate.dim() == 2:  # Scalar gate [batch, 1]
            gate = gate.unsqueeze(1)  # [batch, 1, 1]
        elif gate.dim() == 1:  # Vector gate [hidden_dim]
            gate = gate.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]

        gated_adapter = gate * adapter_out

        return hidden + gated_adapter


class EmotionalEncoderForAdapter(nn.Module):
    """
    Emotional encoder that tracks conversation state.

    More sophisticated than prefix version - tracks multi-turn dynamics.
    """

    def __init__(self, hidden_dim: int = 768, emotion_dim: int = 6):
        super().__init__()

        # Encode current turn hidden states to emotional features
        self.hidden_to_emotion = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emotion_dim),
            nn.Sigmoid()
        )

        # Temporal smoothing (tonic state)
        self.emotion_rnn = nn.GRU(
            input_size=emotion_dim,
            hidden_size=emotion_dim,
            batch_first=True
        )

        # External signal integration
        self.signal_encoder = nn.Linear(5, emotion_dim)

        # Combine internal and external
        self.fusion = nn.Linear(emotion_dim * 2, emotion_dim)

        # Tonic state
        self.tonic_state = None

    def forward(self, hidden_states: torch.Tensor,
                external_signals: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute emotional state from hidden representations.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] - LLM hidden states
            external_signals: [batch, 5] - external feedback signals

        Returns:
            emotional_state: [batch, emotion_dim]
        """
        # Pool hidden states (e.g., mean pooling)
        pooled = hidden_states.mean(dim=1)  # [batch, hidden_dim]

        # Compute phasic emotions from current context
        phasic_emotion = self.hidden_to_emotion(pooled)

        # Update tonic state with GRU
        if self.tonic_state is None:
            self.tonic_state = torch.zeros_like(phasic_emotion).unsqueeze(0)

        phasic_seq = phasic_emotion.unsqueeze(1)  # [batch, 1, emotion_dim]
        _, self.tonic_state = self.emotion_rnn(phasic_seq, self.tonic_state)
        tonic_emotion = self.tonic_state.squeeze(0)  # [batch, emotion_dim]

        # Combine phasic and tonic
        internal_emotion = 0.7 * phasic_emotion + 0.3 * tonic_emotion

        # Integrate external signals if provided
        if external_signals is not None:
            external_emotion = torch.sigmoid(self.signal_encoder(external_signals))
            combined = torch.cat([internal_emotion, external_emotion], dim=-1)
            emotional_state = torch.sigmoid(self.fusion(combined))
        else:
            emotional_state = internal_emotion

        return emotional_state

    def reset_tonic(self):
        """Reset tonic state for new conversation."""
        self.tonic_state = None


class EmotionalAdapterLLM(nn.Module):
    """
    Full LLM with emotional adapters inserted at each layer.

    LLM weights are FROZEN. Only adapters and emotion encoder train.
    """

    def __init__(self, base_model, adapter_dim: int = 64,
                 emotion_dim: int = 6, gate_type: str = "scalar"):
        super().__init__()

        self.base_model = base_model
        self.hidden_dim = base_model.config.hidden_size
        self.n_layers = base_model.config.num_hidden_layers

        # FREEZE base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Create emotional adapters for each layer (TRAINABLE)
        self.adapters = nn.ModuleList([
            EmotionalAdapter(
                hidden_dim=self.hidden_dim,
                adapter_dim=adapter_dim,
                emotion_dim=emotion_dim,
                gate_type=gate_type
            )
            for _ in range(self.n_layers)
        ])

        # Emotional encoder (TRAINABLE)
        self.emotion_encoder = EmotionalEncoderForAdapter(
            hidden_dim=self.hidden_dim,
            emotion_dim=emotion_dim
        )

        # Layer-specific emotion weighting (optional)
        self.layer_emotion_weights = nn.Parameter(
            torch.ones(self.n_layers, emotion_dim)
        )

    def get_trainable_params(self):
        """Return trainable parameters."""
        params = list(self.emotion_encoder.parameters())
        for adapter in self.adapters:
            params.extend(adapter.parameters())
        params.append(self.layer_emotion_weights)
        return params

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                external_signals: Optional[torch.Tensor] = None,
                output_emotions: bool = False):
        """
        Forward pass with emotional adapter modulation.
        """
        # Get embeddings
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

        hidden_states = inputs_embeds
        all_emotions = []

        # Process through each layer
        for layer_idx, layer in enumerate(self.base_model.transformer.h):
            # Apply frozen layer
            layer_output = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_output[0]

            # Compute emotional state from current hidden states
            emotional_state = self.emotion_encoder(hidden_states, external_signals)

            # Layer-specific emotion weighting
            weighted_emotion = emotional_state * torch.sigmoid(
                self.layer_emotion_weights[layer_idx]
            )

            # Apply emotional adapter
            hidden_states = self.adapters[layer_idx](hidden_states, weighted_emotion)

            if output_emotions:
                all_emotions.append(emotional_state.detach())

        # Final layer norm
        hidden_states = self.base_model.transformer.ln_f(hidden_states)

        # LM head
        logits = self.base_model.lm_head(hidden_states)

        if output_emotions:
            return logits, all_emotions
        return logits


class EmotionSpecificAdapters(nn.Module):
    """
    Alternative: Separate adapters for each emotion type.

    Fear adapter, curiosity adapter, etc. - activated based on emotional state.
    """

    def __init__(self, hidden_dim: int = 768, adapter_dim: int = 64,
                 n_emotions: int = 6):
        super().__init__()

        # One adapter per emotion
        self.emotion_adapters = nn.ModuleDict({
            'fear': self._make_adapter(hidden_dim, adapter_dim),
            'curiosity': self._make_adapter(hidden_dim, adapter_dim),
            'anger': self._make_adapter(hidden_dim, adapter_dim),
            'joy': self._make_adapter(hidden_dim, adapter_dim),
            'anxiety': self._make_adapter(hidden_dim, adapter_dim),
            'confidence': self._make_adapter(hidden_dim, adapter_dim),
        })

    def _make_adapter(self, hidden_dim: int, adapter_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, hidden_dim)
        )

    def forward(self, hidden: torch.Tensor,
                emotional_state: EmotionalState) -> torch.Tensor:
        """
        Apply emotion-specific adapters weighted by emotional state.
        """
        emotion_values = {
            'fear': emotional_state.fear,
            'curiosity': emotional_state.curiosity,
            'anger': emotional_state.anger,
            'joy': emotional_state.joy,
            'anxiety': emotional_state.anxiety,
            'confidence': emotional_state.confidence,
        }

        total_adapter_out = torch.zeros_like(hidden)

        for emotion_name, adapter in self.emotion_adapters.items():
            emotion_strength = emotion_values[emotion_name]
            if emotion_strength > 0.1:  # Threshold for activation
                adapter_out = adapter(hidden)
                total_adapter_out += emotion_strength * adapter_out

        return hidden + total_adapter_out


# Training utilities

class EmotionalAdapterTrainer:
    """Training loop for emotional adapters."""

    def __init__(self, model: EmotionalAdapterLLM, lr: float = 1e-4,
                 emotion_loss_weight: float = 0.1):
        self.model = model
        self.emotion_loss_weight = emotion_loss_weight

        self.optimizer = torch.optim.AdamW(
            model.get_trainable_params(),
            lr=lr,
            weight_decay=0.01
        )

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                     emotions: list, target_emotions: Optional[torch.Tensor] = None):
        """
        Combined language modeling and emotional loss.
        """
        # LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        # Emotional consistency loss (optional)
        # Encourage emotions to match target when provided
        emotion_loss = torch.tensor(0.0)
        if target_emotions is not None and emotions:
            final_emotion = emotions[-1]  # Use last layer emotion
            emotion_loss = F.mse_loss(final_emotion, target_emotions)

        total_loss = lm_loss + self.emotion_loss_weight * emotion_loss

        return total_loss, {'lm_loss': lm_loss, 'emotion_loss': emotion_loss}

    def train_step(self, input_ids, labels, external_signals=None,
                   target_emotions=None):
        """Single training step."""
        self.optimizer.zero_grad()

        logits, emotions = self.model(
            input_ids,
            external_signals=external_signals,
            output_emotions=True
        )

        loss, loss_dict = self.compute_loss(logits, labels, emotions, target_emotions)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.get_trainable_params(), 1.0)
        self.optimizer.step()

        return loss.item(), loss_dict


# Behavior Analysis

def analyze_gate_patterns(model: EmotionalAdapterLLM,
                         test_inputs: list,
                         emotional_states: list):
    """
    Analyze how gates activate for different emotional states.

    Helps understand what each emotion's effect is.
    """
    gate_activations = {i: [] for i in range(model.n_layers)}

    model.eval()
    with torch.no_grad():
        for input_ids, emotion in zip(test_inputs, emotional_states):
            hidden = model.base_model.get_input_embeddings()(input_ids)

            for layer_idx, (layer, adapter) in enumerate(
                zip(model.base_model.transformer.h, model.adapters)
            ):
                layer_out = layer(hidden)[0]
                emotion_tensor = emotion.to_tensor().unsqueeze(0)
                gate_value = adapter.gate(emotion_tensor).item()
                gate_activations[layer_idx].append({
                    'fear': emotion.fear,
                    'curiosity': emotion.curiosity,
                    'gate': gate_value
                })
                hidden = adapter(layer_out, emotion_tensor)

    return gate_activations


if __name__ == "__main__":
    # Example: Create emotional adapter model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained("gpt2")
    model = EmotionalAdapterLLM(base, adapter_dim=64, gate_type="scalar")

    print(f"Base model params: {sum(p.numel() for p in base.parameters()):,}")
    print(f"Trainable params: {sum(p.numel() for p in model.get_trainable_params()):,}")
    print(f"Ratio: {sum(p.numel() for p in model.get_trainable_params()) / sum(p.numel() for p in base.parameters()):.2%}")
```

## Key Design Decisions

### 1. Gate Type Selection

| Gate Type | Parameters | Flexibility | Use Case |
|-----------|------------|-------------|----------|
| Scalar | O(emotion_dim) | Low | Simple on/off modulation |
| Vector | O(emotion_dim × hidden_dim) | Medium | Per-dimension control |
| Attention | O(hidden_dim × 64) | High | Context-dependent gating |

### 2. Adapter Placement

Options:
- **Post-attention only**: Lower compute, affects reasoning
- **Post-FFN only**: Affects knowledge retrieval
- **Both**: Maximum control, more parameters

### 3. Emotion-Adapter Relationship

Options:
- **Shared adapter, emotion gates**: One adapter, emotion controls strength
- **Emotion-specific adapters**: Separate adapter per emotion
- **Hybrid**: Base adapter + emotion-specific residuals

## Training Strategies

### Strategy 1: Behavioral Cloning
```python
# Train on (input, emotional_context, desired_output) triples
for batch in dataset:
    logits = model(batch.input, external_signals=batch.emotion)
    loss = cross_entropy(logits, batch.output)
```

### Strategy 2: Contrastive Emotional Learning
```python
# Learn to differentiate emotional responses
fearful_response = model(input, EmotionalState(fear=1.0))
calm_response = model(input, EmotionalState(fear=0.0))
loss = contrastive_loss(fearful_response, calm_response, "more cautious")
```

### Strategy 3: Reinforcement from Feedback
```python
# RLHF-style with emotional state
response = model.generate(input, emotional_state)
reward = get_human_feedback(response)
update_emotional_state(emotional_state, reward)
policy_gradient_update(model, response, reward)
```

## Expected Effects by Emotion

| Emotion | Gate Behavior | Output Effect |
|---------|---------------|---------------|
| Fear ↑ | Adapter contribution ↓ | Conservative, closer to base model |
| Curiosity ↑ | Adapter contribution ↑ | More exploratory, diverse outputs |
| Anger ↑ | Specific dimensions activated | Persistence, trying alternatives |
| Joy ↑ | Broader activation | More elaborate, engaging responses |

## Comparison to Emotional-ED

| Emotional-ED | Adapter + Gating |
|--------------|------------------|
| Fear modulates Q-value | Fear modulates gate |
| Single state input | Per-layer modulation |
| Action selection bias | Token probability bias |
| `effective_lr *= (1 + fear)` | `adapter_out *= gate(fear)` |

## Advantages

1. **Layer-wise control**: Different effects at different depths
2. **Efficient**: Only ~1-2% extra parameters
3. **Interpretable gates**: Can analyze what each emotion does
4. **Stable training**: Zero-initialized adapters, gradual learning

## Limitations

1. **More complex than prefix**: Multiple modules to tune
2. **Hyperparameter sensitive**: Gate type, adapter dim matter
3. **Harder to debug**: Multiple interaction points
4. **May need per-model tuning**: Architecture-specific

## Next Steps

1. Benchmark scalar vs vector vs attention gates
2. Compare emotion-specific vs shared adapters
3. Analyze gate activation patterns across emotions
4. Test on safety-critical scenarios
