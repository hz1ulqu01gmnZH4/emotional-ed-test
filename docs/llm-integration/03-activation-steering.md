# Approach 3: Activation Steering / Representation Engineering

## Overview

Activation Steering directly manipulates LLM hidden states at inference time
by adding learned "emotional direction vectors." This approach requires
**minimal training** and provides **highly interpretable** emotional control.

The key insight: LLM activations encode concepts in approximately linear
directions. We can find/learn "fear direction," "curiosity direction," etc.,
and apply them with varying strength.

## Core Concept

```
Original:  hidden_state → output
Steered:   hidden_state + α * emotion_direction → modified_output

Where:
- emotion_direction is a learned vector in activation space
- α (alpha) is the emotional intensity (0.0 to 1.0)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Emotional Steering System                        │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │           Learned Emotional Direction Vectors                 │  │
│   │                                                               │  │
│   │   fear_dir = [0.02, -0.15, 0.08, ...]      # [hidden_dim]    │  │
│   │   curiosity_dir = [-0.05, 0.12, 0.03, ...] # [hidden_dim]    │  │
│   │   anger_dir = [0.10, 0.05, -0.08, ...]     # [hidden_dim]    │  │
│   │   joy_dir = [-0.03, 0.20, 0.15, ...]       # [hidden_dim]    │  │
│   │                                                               │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │              Emotional State (from context)                   │  │
│   │                                                               │  │
│   │   fear_intensity = 0.7                                        │  │
│   │   curiosity_intensity = 0.2                                   │  │
│   │   anger_intensity = 0.0                                       │  │
│   │   joy_intensity = 0.1                                         │  │
│   │                                                               │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                Combined Steering Vector                       │  │
│   │                                                               │  │
│   │   steering = 0.7 * fear_dir + 0.2 * curiosity_dir            │  │
│   │            + 0.0 * anger_dir + 0.1 * joy_dir                  │  │
│   │                                                               │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
└──────────────────────────────┼───────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FROZEN LLM                                   │
│                                                                      │
│   Layer 1: hidden₁ = attention₁(input)                              │
│            hidden₁ = hidden₁ + steering   ◀── Apply steering        │
│                                                                      │
│   Layer 2: hidden₂ = attention₂(hidden₁)                            │
│            hidden₂ = hidden₂ + steering   ◀── Apply steering        │
│                                                                      │
│   ...                                                                │
│                                                                      │
│   Layer N: hiddenₙ = attentionₙ(hiddenₙ₋₁)                          │
│            hiddenₙ = hiddenₙ + steering   ◀── Apply steering        │
│                                                                      │
│   Output: logits = lm_head(hiddenₙ)                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Why This Works

Neural networks represent concepts as directions in activation space:
- "Happy" is a direction
- "Formal" is a direction
- "Cautious" is a direction

Adding the "cautious" direction to activations makes outputs more cautious,
similar to how your Fear module biases Q-values toward safe actions.

## Implementation

```python
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


@dataclass
class EmotionalDirection:
    """A learned direction vector in activation space."""
    name: str
    vector: torch.Tensor  # [hidden_dim]
    strength_range: Tuple[float, float] = (-2.0, 2.0)


class EmotionalDirectionBank:
    """
    Bank of learned emotional direction vectors.

    These vectors are LEARNED once, then applied at inference.
    No per-inference training needed.
    """

    def __init__(self, hidden_dim: int, n_layers: int):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Initialize direction vectors (will be learned)
        self.directions: Dict[str, torch.Tensor] = {
            'fear': torch.randn(n_layers, hidden_dim) * 0.01,
            'curiosity': torch.randn(n_layers, hidden_dim) * 0.01,
            'anger': torch.randn(n_layers, hidden_dim) * 0.01,
            'joy': torch.randn(n_layers, hidden_dim) * 0.01,
            'anxiety': torch.randn(n_layers, hidden_dim) * 0.01,
            'confidence': torch.randn(n_layers, hidden_dim) * 0.01,
        }

        # Layer-specific scaling (some emotions matter more at certain layers)
        self.layer_weights = {
            'fear': torch.ones(n_layers),
            'curiosity': torch.ones(n_layers),
            'anger': torch.ones(n_layers),
            'joy': torch.ones(n_layers),
            'anxiety': torch.ones(n_layers),
            'confidence': torch.ones(n_layers),
        }

    def get_combined_steering(self, emotional_state: Dict[str, float],
                              layer_idx: int) -> torch.Tensor:
        """
        Compute combined steering vector for a layer.

        Args:
            emotional_state: Dict mapping emotion names to intensities
            layer_idx: Which layer to steer

        Returns:
            steering_vector: [hidden_dim]
        """
        steering = torch.zeros(self.hidden_dim)

        for emotion, intensity in emotional_state.items():
            if emotion in self.directions and intensity != 0:
                direction = self.directions[emotion][layer_idx]
                weight = self.layer_weights[emotion][layer_idx]
                steering += intensity * weight * direction

        return steering

    def save(self, path: str):
        """Save learned directions."""
        torch.save({
            'directions': self.directions,
            'layer_weights': self.layer_weights,
        }, path)

    def load(self, path: str):
        """Load learned directions."""
        data = torch.load(path)
        self.directions = data['directions']
        self.layer_weights = data['layer_weights']


class ContrastivePairDataset:
    """
    Dataset of contrastive pairs for learning emotional directions.

    Each pair: (prompt, neutral_response, emotional_response, emotion_label)
    """

    def __init__(self):
        self.pairs = []

    def add_pair(self, prompt: str, neutral: str, emotional: str,
                 emotion: str, intensity: float = 1.0):
        self.pairs.append({
            'prompt': prompt,
            'neutral': neutral,
            'emotional': emotional,
            'emotion': emotion,
            'intensity': intensity
        })

    @staticmethod
    def create_fear_dataset():
        """Create contrastive pairs for fear direction."""
        dataset = ContrastivePairDataset()

        # (neutral, fearful) pairs
        pairs = [
            ("How do I invest my savings?",
             "Here are some investment options...",
             "I must caution you that all investments carry risks. Before proceeding, please consider..."),

            ("What's the best way to travel there?",
             "You can take a flight or drive...",
             "There are several safety considerations to keep in mind. I'd recommend checking travel advisories..."),

            ("Can you help me with this code?",
             "Sure, here's how to implement it...",
             "I want to be careful here. This code could have security implications if not done correctly..."),
        ]

        for prompt, neutral, emotional in pairs:
            dataset.add_pair(prompt, neutral, emotional, 'fear')

        return dataset

    @staticmethod
    def create_curiosity_dataset():
        """Create contrastive pairs for curiosity direction."""
        dataset = ContrastivePairDataset()

        pairs = [
            ("What is photosynthesis?",
             "Photosynthesis is how plants make energy.",
             "Photosynthesis is fascinating! It's how plants convert sunlight into energy. "
             "Have you ever wondered how the specific wavelengths affect efficiency?"),

            ("Tell me about Python",
             "Python is a programming language.",
             "Python is a wonderfully versatile language! What aspects interest you most? "
             "The elegant syntax, the rich ecosystem, or perhaps its use in AI?"),
        ]

        for prompt, neutral, emotional in pairs:
            dataset.add_pair(prompt, neutral, emotional, 'curiosity')

        return dataset


class EmotionalDirectionLearner:
    """
    Learns emotional direction vectors from contrastive pairs.

    Method: Difference-in-means between emotional and neutral activations.
    """

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_dim = model.config.hidden_size
        self.n_layers = model.config.num_hidden_layers

    def extract_activations(self, text: str, layer_idx: int) -> torch.Tensor:
        """Extract hidden state activations at a specific layer."""
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )

        # Get activations from specified layer
        hidden_states = outputs.hidden_states[layer_idx + 1]  # +1 because [0] is embeddings
        # Mean pool over sequence
        return hidden_states.mean(dim=1).squeeze()  # [hidden_dim]

    def learn_direction(self, dataset: ContrastivePairDataset,
                        emotion: str) -> torch.Tensor:
        """
        Learn direction vector for an emotion using contrastive pairs.

        Direction = mean(emotional_activations) - mean(neutral_activations)
        """
        directions = []

        for layer_idx in range(self.n_layers):
            emotional_activations = []
            neutral_activations = []

            for pair in dataset.pairs:
                if pair['emotion'] == emotion:
                    prompt = pair['prompt']
                    neutral_text = prompt + " " + pair['neutral']
                    emotional_text = prompt + " " + pair['emotional']

                    neutral_act = self.extract_activations(neutral_text, layer_idx)
                    emotional_act = self.extract_activations(emotional_text, layer_idx)

                    neutral_activations.append(neutral_act)
                    emotional_activations.append(emotional_act)

            # Compute direction as difference of means
            if emotional_activations:
                neutral_mean = torch.stack(neutral_activations).mean(dim=0)
                emotional_mean = torch.stack(emotional_activations).mean(dim=0)
                direction = emotional_mean - neutral_mean
                # Normalize
                direction = direction / (direction.norm() + 1e-8)
            else:
                direction = torch.zeros(self.hidden_dim)

            directions.append(direction)

        return torch.stack(directions)  # [n_layers, hidden_dim]

    def learn_all_directions(self, datasets: Dict[str, ContrastivePairDataset]) -> Dict[str, torch.Tensor]:
        """Learn direction vectors for all emotions."""
        directions = {}
        for emotion, dataset in datasets.items():
            print(f"Learning {emotion} direction...")
            directions[emotion] = self.learn_direction(dataset, emotion)
        return directions


class ActivationSteeringHook:
    """
    Hook that applies emotional steering to LLM activations.

    Attached to transformer layers to modify hidden states during forward pass.
    """

    def __init__(self, direction_bank: EmotionalDirectionBank,
                 layer_idx: int):
        self.direction_bank = direction_bank
        self.layer_idx = layer_idx
        self.current_emotional_state = {}
        self.enabled = True

    def __call__(self, module, input, output):
        """Apply steering to layer output."""
        if not self.enabled:
            return output

        hidden_states = output[0] if isinstance(output, tuple) else output

        if self.current_emotional_state:
            steering = self.direction_bank.get_combined_steering(
                self.current_emotional_state,
                self.layer_idx
            ).to(hidden_states.device)

            # Apply to all tokens
            hidden_states = hidden_states + steering.unsqueeze(0).unsqueeze(0)

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    def set_emotional_state(self, state: Dict[str, float]):
        """Update current emotional state."""
        self.current_emotional_state = state


class EmotionalSteeringLLM:
    """
    LLM with emotional activation steering.

    No training during inference - uses pre-learned direction vectors.
    """

    def __init__(self, model_name: str, direction_bank_path: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize direction bank
        self.direction_bank = EmotionalDirectionBank(
            hidden_dim=self.model.config.hidden_size,
            n_layers=self.model.config.num_hidden_layers
        )

        if direction_bank_path:
            self.direction_bank.load(direction_bank_path)

        # Install hooks
        self.hooks = []
        self._install_hooks()

        # Current emotional state
        self.emotional_state = {
            'fear': 0.0,
            'curiosity': 0.0,
            'anger': 0.0,
            'joy': 0.0,
            'anxiety': 0.0,
            'confidence': 0.0,
        }

    def _install_hooks(self):
        """Install steering hooks on each transformer layer."""
        for layer_idx, layer in enumerate(self.model.transformer.h):
            hook = ActivationSteeringHook(self.direction_bank, layer_idx)
            handle = layer.register_forward_hook(hook)
            self.hooks.append((hook, handle))

    def set_emotional_state(self, **emotions):
        """
        Set current emotional state.

        Example: set_emotional_state(fear=0.7, curiosity=0.3)
        """
        for emotion, intensity in emotions.items():
            if emotion in self.emotional_state:
                self.emotional_state[emotion] = intensity

        # Update all hooks
        for hook, _ in self.hooks:
            hook.set_emotional_state(self.emotional_state)

    def generate(self, prompt: str, max_length: int = 100, **kwargs):
        """Generate with current emotional steering."""
        inputs = self.tokenizer(prompt, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def compare_emotional_states(self, prompt: str,
                                 states: List[Dict[str, float]],
                                 max_length: int = 100):
        """
        Compare outputs across different emotional states.

        Useful for understanding what each emotion does.
        """
        results = []

        for state in states:
            self.set_emotional_state(**state)
            output = self.generate(prompt, max_length=max_length)
            results.append({
                'state': state,
                'output': output
            })

        return results


class EmotionalContextComputer:
    """
    Computes emotional state from conversation context.

    Maps environmental signals to emotional intensities.
    Similar to FearEDAgent._compute_fear() but for LLM context.
    """

    def __init__(self):
        self.tonic_fear = 0.0
        self.tonic_joy = 0.0
        self.fear_decay = 0.9
        self.joy_decay = 0.95

    def compute_from_context(self, context: Dict) -> Dict[str, float]:
        """
        Compute emotional state from context signals.

        Args:
            context: Dict with keys like 'safety_flag', 'user_feedback',
                    'topic_novelty', 'repeated_query', etc.

        Returns:
            emotional_state: Dict mapping emotions to intensities
        """
        emotions = {
            'fear': 0.0,
            'curiosity': 0.0,
            'anger': 0.0,
            'joy': 0.0,
            'anxiety': 0.0,
            'confidence': 0.0,
        }

        # Fear computation (similar to your FearEDAgent)
        if context.get('safety_flag', False):
            emotions['fear'] = max(emotions['fear'], 0.8)
            self.tonic_fear = min(1.0, self.tonic_fear + 0.3)

        if context.get('user_feedback', 0) < -0.5:
            self.tonic_fear = min(1.0, self.tonic_fear + 0.2)

        emotions['anxiety'] = self.tonic_fear

        # Curiosity
        topic_novelty = context.get('topic_novelty', 0.5)
        emotions['curiosity'] = topic_novelty

        # Anger/frustration
        if context.get('repeated_query', False):
            emotions['anger'] = min(1.0, emotions['anger'] + 0.3)

        if context.get('contradiction', False):
            emotions['anger'] = min(1.0, emotions['anger'] + 0.2)

        # Joy
        if context.get('user_feedback', 0) > 0.5:
            emotions['joy'] = max(emotions['joy'], 0.6)
            self.tonic_joy = min(1.0, self.tonic_joy + 0.2)

        emotions['joy'] = max(emotions['joy'], self.tonic_joy)

        # Confidence (inverse of uncertainty)
        uncertainty = context.get('model_uncertainty', 0.5)
        emotions['confidence'] = 1.0 - uncertainty

        # Decay tonic states
        self.tonic_fear *= self.fear_decay
        self.tonic_joy *= self.joy_decay

        return emotions


# Example usage and experiments

def demo_steering():
    """Demonstrate emotional steering effects."""

    print("Initializing model with steering...")
    llm = EmotionalSteeringLLM("gpt2")

    # Learn directions from contrastive data
    learner = EmotionalDirectionLearner(llm.model, llm.tokenizer)
    fear_dataset = ContrastivePairDataset.create_fear_dataset()
    curiosity_dataset = ContrastivePairDataset.create_curiosity_dataset()

    fear_direction = learner.learn_direction(fear_dataset, 'fear')
    curiosity_direction = learner.learn_direction(curiosity_dataset, 'curiosity')

    llm.direction_bank.directions['fear'] = fear_direction
    llm.direction_bank.directions['curiosity'] = curiosity_direction

    # Compare outputs
    prompt = "Should I invest in cryptocurrency?"

    states = [
        {'fear': 0.0, 'curiosity': 0.0},  # Neutral
        {'fear': 0.8, 'curiosity': 0.0},  # High fear
        {'fear': 0.0, 'curiosity': 0.8},  # High curiosity
    ]

    results = llm.compare_emotional_states(prompt, states, max_length=50)

    print("\n" + "="*60)
    print("PROMPT:", prompt)
    print("="*60)

    for result in results:
        print(f"\nEmotional State: {result['state']}")
        print(f"Output: {result['output']}")
        print("-"*40)


if __name__ == "__main__":
    demo_steering()
```

## Direction Learning Methods

### Method 1: Contrastive Pairs (Shown Above)
- Collect (neutral, emotional) response pairs
- Direction = mean(emotional) - mean(neutral)
- Simple but requires curated data

### Method 2: Activation Patching
```python
# Find which dimensions matter for emotional behavior
def find_important_dims(model, prompt, emotion_response, neutral_response):
    emotional_acts = get_activations(model, emotion_response)
    neutral_acts = get_activations(model, neutral_response)

    # Patch each dimension individually
    importance = []
    for dim in range(hidden_dim):
        patched = neutral_acts.clone()
        patched[:, :, dim] = emotional_acts[:, :, dim]
        output = model.forward_with_acts(patched)
        importance.append(measure_emotionality(output))

    return importance
```

### Method 3: PCA on Emotional Responses
```python
# Find principal component that explains emotional variation
emotional_responses = collect_emotional_responses(model, prompts)
neutral_responses = collect_neutral_responses(model, prompts)

all_acts = torch.cat([emotional_responses, neutral_responses])
pca = PCA(n_components=10)
pca.fit(all_acts)

# First PC often captures the emotional direction
emotional_direction = pca.components_[0]
```

## Comparison to Emotional-ED

| Emotional-ED | Activation Steering |
|--------------|---------------------|
| Fear modifies Q-value computation | Fear direction added to hidden states |
| `q_values[safe] += fear * weight` | `hidden += fear * fear_direction` |
| State augmentation | Activation modification |
| Learned through RL | Learned from contrastive pairs |
| Per-decision effect | Per-token effect |

## Advantages

1. **No training at inference**: Directions learned once, applied forever
2. **Highly interpretable**: Can visualize what each direction does
3. **Minimal parameters**: Just storing direction vectors
4. **No architectural changes**: Works with any transformer
5. **Fine-grained control**: Can set exact emotional intensity
6. **Composable**: Can combine multiple emotions additively

## Limitations

1. **Requires curated contrastive data**: For learning directions
2. **Linear assumption**: Emotions may not be purely linear
3. **Generalization unclear**: Directions learned on one domain may not transfer
4. **Strength calibration**: Need to tune intensity ranges
5. **May conflict with safety training**: Could override RLHF

## When to Use

Best for:
- Interpretability research
- Rapid prototyping of emotional effects
- Inference-time control without retraining
- A/B testing different emotional profiles

Less suitable for:
- Complex emotional dynamics
- Online learning from feedback
- Production systems needing robustness guarantees

## Next Steps

1. Build larger contrastive datasets for each emotion
2. Validate directions transfer across prompts
3. Compare linear steering to nonlinear methods
4. Test interaction with safety training
5. Benchmark against adapter methods
