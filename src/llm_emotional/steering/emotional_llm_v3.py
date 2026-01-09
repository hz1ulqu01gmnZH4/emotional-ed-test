"""
Emotional Steering LLM - Version 3 with Error Diffusion.

Key innovation: Treats emotional steering as a feedback control system.

Drawing from research synthesis:
- Damasio's somatic markers: Emotions provide continuous error signals
- Berridge's wanting/liking: Separate motivational and hedonic signals
- Emotion Circuits (Wang 2025): Context-agnostic directions achieve 99.65% accuracy

V3 improvements over V2:
1. Error diffusion across layers (Floyd-Steinberg inspired)
2. Temporal error accumulation (leaky integrator)
3. Attractor-based steering with feedback correction
4. Adaptive steering based on accumulated error
5. Separate wanting vs liking vector support

The error diffusion mechanism ensures:
- Smoother emotional transitions (no abrupt jumps)
- Self-correcting behavior (feedback loop)
- More stable generation (prevents oscillation)
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .steering_hooks_v3 import ErrorDiffusionManager, ErrorState


@dataclass
class EmotionState:
    """Represents current emotional state with wanting/liking separation."""
    # Valence emotions
    fear: float = 0.0
    joy: float = 0.0
    anger: float = 0.0
    sadness: float = 0.0

    # Arousal/engagement
    curiosity: float = 0.0
    engagement: float = 0.0

    # Wanting vs liking (Berridge dissociation)
    wanting: float = 0.0  # Motivational salience (dopamine)
    liking: float = 0.0   # Hedonic impact (opioid)

    # Regulatory
    resilience: float = 0.0
    equanimity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "fear": self.fear,
            "joy": self.joy,
            "anger": self.anger,
            "sadness": self.sadness,
            "curiosity": self.curiosity,
            "engagement": self.engagement,
            "wanting": self.wanting,
            "liking": self.liking,
            "resilience": self.resilience,
            "equanimity": self.equanimity,
        }

    def active_emotions(self) -> dict[str, float]:
        """Return only non-zero emotions."""
        return {k: v for k, v in self.to_dict().items() if v != 0.0}


class EmotionalSteeringLLMv3:
    """
    LLM wrapper with error-diffusion emotional steering.

    Key features:
    - Error diffusion: Steering errors propagate between layers
    - Temporal accumulation: Errors accumulate across tokens
    - Attractor-based: Define target emotional states in activation space
    - Wanting/liking: Separate motivational and hedonic dimensions
    - Adaptive: Steering adjusts based on feedback
    """

    # All supported emotions
    EMOTIONS = {
        "fear", "joy", "anger", "sadness",
        "curiosity", "engagement",
        "wanting", "liking",
        "resilience", "equanimity",
    }

    def __init__(
        self,
        model_name: str,
        direction_bank_path: Optional[str] = None,
        steering_scale: float = 1.0,
        diffusion_rate: float = 0.25,
        temporal_decay: float = 0.9,
        device: Optional[str] = None,
    ):
        """
        Initialize the V3 emotional steering LLM.

        Args:
            model_name: HuggingFace model name
            direction_bank_path: Path to direction bank JSON
            steering_scale: Global steering intensity multiplier
            diffusion_rate: Error diffusion rate between layers
            temporal_decay: Temporal error decay rate
            device: Device ("cuda", "cpu", or None for auto)
        """
        self.model_name = model_name
        self.steering_scale = steering_scale
        self.diffusion_rate = diffusion_rate
        self.temporal_decay = temporal_decay

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)

        # Determine architecture
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.n_layers = len(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self.n_layers = len(self.model.transformer.h)
        else:
            raise ValueError("Cannot determine model architecture")

        self.hidden_dim = self.model.config.hidden_size
        print(f"Model: {self.n_layers} layers, {self.hidden_dim} hidden dim")

        # Initialize error diffusion manager
        self.steering_manager = ErrorDiffusionManager(
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
            diffusion_rate=diffusion_rate,
            temporal_decay=temporal_decay,
        )
        self.steering_manager.scale = steering_scale
        self.steering_manager.install(self.model)

        # Direction storage
        self.directions: dict[str, Tensor] = {}
        self.layer_weights: dict[str, dict[int, float]] = {}

        # Current emotional state
        self.current_state = EmotionState()

        # Attractor computation cache
        self._attractor_cache: dict[str, Tensor] = {}

        # Load directions if provided
        if direction_bank_path is not None:
            self.load_directions(direction_bank_path)

    def load_directions(self, path: str):
        """Load PCA-trained directions from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Direction bank not found: {path}")

        with open(path) as f:
            data = json.load(f)

        # Validate dimensions
        if data["hidden_dim"] != self.hidden_dim:
            raise ValueError(
                f"Direction bank hidden_dim ({data['hidden_dim']}) "
                f"doesn't match model ({self.hidden_dim})"
            )

        # Load directions
        for emotion, direction_list in data["directions"].items():
            self.directions[emotion] = torch.tensor(direction_list, dtype=torch.float32)

        # Load layer weights if available
        if "layer_weights" in data:
            for emotion, weights in data["layer_weights"].items():
                self.layer_weights[emotion] = {int(k): v for k, v in weights.items()}

        # Load attractors if available (V3 feature)
        if "attractors" in data:
            for emotion, attractor_list in data["attractors"].items():
                attractor = torch.tensor(attractor_list, dtype=torch.float32)
                self.steering_manager.set_attractor(emotion, attractor)

        print(f"Loaded directions for: {list(self.directions.keys())}")

    def compute_attractor(self, emotion: str, samples: list[str]) -> Tensor:
        """
        Compute attractor from sample texts.

        An attractor represents the target activation pattern for
        a given emotional state, computed as the mean activation
        over emotion-eliciting samples.

        Args:
            emotion: Name of the emotion
            samples: List of sample texts expressing this emotion

        Returns:
            Attractor tensor [hidden_dim]
        """
        activations = []

        for sample in samples:
            inputs = self.tokenizer(sample, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            # Get activations from middle-to-late layers (most emotion-relevant)
            target_layers = range(self.n_layers // 2, int(self.n_layers * 0.8))
            layer_activations = []

            for layer_idx in target_layers:
                # +1 because hidden_states[0] is embeddings
                hidden = outputs.hidden_states[layer_idx + 1]
                # Mean over sequence
                layer_act = hidden.mean(dim=1).squeeze(0)
                layer_activations.append(layer_act)

            # Mean over selected layers
            sample_activation = torch.stack(layer_activations).mean(dim=0)
            activations.append(sample_activation)

        # Mean over all samples
        attractor = torch.stack(activations).mean(dim=0)

        # Store and register
        self._attractor_cache[emotion] = attractor
        self.steering_manager.set_attractor(emotion, attractor)

        return attractor

    def set_emotional_state(self, **emotions):
        """
        Set the emotional state for generation.

        Supports all V3 emotions including wanting/liking.

        Args:
            **emotions: Emotion intensities (0.0 to 1.0)
                fear, joy, anger, sadness, curiosity, engagement,
                wanting, liking, resilience, equanimity
        """
        # Update state
        for emotion, intensity in emotions.items():
            if emotion not in self.EMOTIONS:
                raise ValueError(f"Unknown emotion: {emotion}")
            setattr(self.current_state, emotion, intensity)

        # Combine directions weighted by intensity
        combined_direction = torch.zeros(self.hidden_dim)
        combined_weights = {i: 0.0 for i in range(self.n_layers)}
        total_intensity = 0.0

        active = self.current_state.active_emotions()

        for emotion, intensity in active.items():
            if emotion in self.directions:
                combined_direction += intensity * self.directions[emotion]
                total_intensity += intensity

                # Combine layer weights
                if emotion in self.layer_weights:
                    for layer_idx, weight in self.layer_weights[emotion].items():
                        combined_weights[layer_idx] += intensity * weight

        # Normalize if multiple emotions
        if total_intensity > 0:
            combined_direction = combined_direction / total_intensity
            combined_weights = {k: v / total_intensity for k, v in combined_weights.items()}

        # Update steering manager
        if total_intensity > 0:
            self.steering_manager.set_layer_weights(combined_weights)
            self.steering_manager.set_steering(combined_direction)

            # Activate dominant attractor for error feedback
            dominant_emotion = max(active.items(), key=lambda x: x[1])[0]
            if dominant_emotion in self.steering_manager.attractors:
                self.steering_manager.activate_attractor(dominant_emotion)
        else:
            self.steering_manager.clear_steering()

    def clear_emotional_state(self):
        """Clear all emotional steering."""
        self.current_state = EmotionState()
        self.steering_manager.clear_steering()

    def reset_error_state(self):
        """
        Reset accumulated error state.

        Call this between unrelated generations to prevent
        error carryover.
        """
        self.steering_manager.reset_error_state()

    def generate_completion(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        reset_errors: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate a completion with error-diffusion emotional steering.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample (vs greedy)
            reset_errors: Whether to reset error state before generation
            **kwargs: Additional generation arguments

        Returns:
            Generated text (response only)
        """
        if reset_errors:
            self.reset_error_state()

        # Format as chat message
        messages = [{"role": "user", "content": prompt}]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode response only
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return response.strip()

    def generate_with_emotion_trajectory(
        self,
        prompt: str,
        trajectory: list[dict[str, float]],
        tokens_per_segment: int = 25,
        temperature: float = 0.7,
        **kwargs,
    ) -> tuple[str, list[dict]]:
        """
        Generate with changing emotional states.

        Allows emotion to shift during generation, with error diffusion
        smoothing the transitions.

        Args:
            prompt: Input prompt
            trajectory: List of emotion dicts, one per segment
            tokens_per_segment: Tokens to generate per emotion state
            temperature: Sampling temperature

        Returns:
            Tuple of (full_response, metrics_per_segment)
        """
        self.reset_error_state()

        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        all_metrics = []

        with torch.no_grad():
            for segment_idx, emotions in enumerate(trajectory):
                # Set emotional state for this segment
                self.set_emotional_state(**emotions)

                # Generate segment
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=tokens_per_segment,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    **kwargs,
                )

                # Update input_ids for next segment
                input_ids = outputs

                # Collect metrics
                metrics = {
                    "segment": segment_idx,
                    "emotions": emotions,
                    "error_summary": self.steering_manager.get_error_summary(),
                }
                all_metrics.append(metrics)

        # Decode full response
        messages_len = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).shape[1]

        response = self.tokenizer.decode(
            outputs[0][messages_len:],
            skip_special_tokens=True,
        )

        return response.strip(), all_metrics

    def get_current_state(self) -> dict:
        """Get current emotional state."""
        return self.current_state.to_dict()

    def get_error_metrics(self) -> dict:
        """Get current error diffusion metrics."""
        return {
            "error_summary": self.steering_manager.get_error_summary(),
            "layer_metrics": self.steering_manager.get_all_metrics(),
        }

    def set_diffusion_params(
        self,
        diffusion_rate: Optional[float] = None,
        temporal_decay: Optional[float] = None,
    ):
        """
        Adjust error diffusion parameters.

        Higher diffusion_rate = more error spreading between layers
        Higher temporal_decay = errors persist longer across tokens

        Args:
            diffusion_rate: Layer diffusion rate (0-1)
            temporal_decay: Temporal decay rate (0-1)
        """
        if diffusion_rate is not None:
            for hook in self.steering_manager.hooks.values():
                hook.diffusion_rate = diffusion_rate

        if temporal_decay is not None:
            for hook in self.steering_manager.hooks.values():
                hook.temporal_decay = temporal_decay

    def uninstall(self):
        """Remove all hooks from model."""
        self.steering_manager.uninstall()


# =============================================================================
# Utility: Compute Wanting vs Liking Directions
# =============================================================================

def compute_wanting_liking_directions(
    model,
    tokenizer,
    device: str = "cuda",
) -> tuple[Tensor, Tensor]:
    """
    Compute separate wanting and liking directions.

    Based on Berridge's research:
    - Wanting: Anticipation, craving, motivation (dopaminergic)
    - Liking: Actual pleasure, satisfaction (opioidergic)

    Returns:
        Tuple of (wanting_direction, liking_direction)
    """
    # Wanting pairs (anticipation, craving, approach)
    wanting_pairs = [
        ("I see the reward.", "I desperately want to get the reward."),
        ("There's food nearby.", "I'm craving that food so badly."),
        ("The goal is ahead.", "I must reach that goal, I can't resist."),
        ("An opportunity exists.", "I'm driven to pursue this opportunity."),
        ("Something valuable is there.", "I need to obtain it, I'm compelled."),
    ]

    # Liking pairs (actual pleasure, satisfaction)
    liking_pairs = [
        ("I received the reward.", "This feels wonderful, pure satisfaction."),
        ("I'm eating the food.", "This is delicious, such pleasure."),
        ("I reached the goal.", "Deep contentment washes over me."),
        ("I have the opportunity.", "I'm savoring this moment completely."),
        ("I obtained the valuable thing.", "This brings me genuine joy."),
    ]

    def compute_direction(pairs):
        neutral_acts = []
        target_acts = []

        for neutral, target in pairs:
            # Neutral
            inputs = tokenizer(neutral, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            neutral_acts.append(torch.stack([h[:, -1, :] for h in out.hidden_states[1:]]))

            # Target
            inputs = tokenizer(target, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            target_acts.append(torch.stack([h[:, -1, :] for h in out.hidden_states[1:]]))

        neutral_mean = torch.stack(neutral_acts).mean(dim=0)
        target_mean = torch.stack(target_acts).mean(dim=0)
        direction = (target_mean - neutral_mean).squeeze(1).mean(dim=0)

        # Normalize
        return direction / (direction.norm() + 1e-8)

    wanting = compute_direction(wanting_pairs)
    liking = compute_direction(liking_pairs)

    return wanting.cpu(), liking.cpu()


# =============================================================================
# Utility: Compute Resilience/Equanimity Directions
# =============================================================================

def compute_regulatory_directions(
    model,
    tokenizer,
    device: str = "cuda",
) -> tuple[Tensor, Tensor]:
    """
    Compute resilience and equanimity directions.

    Based on emotion regulation research:
    - Resilience: Bounce back from adversity
    - Equanimity: Maintain balance amid turbulence

    Returns:
        Tuple of (resilience_direction, equanimity_direction)
    """
    # Resilience pairs
    resilience_pairs = [
        ("I failed.", "I failed, but I'll try again stronger."),
        ("Things went wrong.", "Setbacks make me more determined."),
        ("I was rejected.", "Each rejection brings me closer to success."),
        ("The project collapsed.", "I'll rebuild better than before."),
        ("I made a mistake.", "Mistakes are how I learn and grow."),
    ]

    # Equanimity pairs
    equanimity_pairs = [
        ("This is chaos.", "I observe the chaos with calm detachment."),
        ("Everything is uncertain.", "Uncertainty is simply the nature of things."),
        ("Strong emotions arise.", "I acknowledge the feelings and let them pass."),
        ("The situation is intense.", "I remain centered amid the intensity."),
        ("Conflict surrounds me.", "I stay balanced, neither grasping nor avoiding."),
    ]

    def compute_direction(pairs):
        neutral_acts = []
        target_acts = []

        for neutral, target in pairs:
            inputs = tokenizer(neutral, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            neutral_acts.append(torch.stack([h[:, -1, :] for h in out.hidden_states[1:]]))

            inputs = tokenizer(target, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            target_acts.append(torch.stack([h[:, -1, :] for h in out.hidden_states[1:]]))

        neutral_mean = torch.stack(neutral_acts).mean(dim=0)
        target_mean = torch.stack(target_acts).mean(dim=0)
        direction = (target_mean - neutral_mean).squeeze(1).mean(dim=0)

        return direction / (direction.norm() + 1e-8)

    resilience = compute_direction(resilience_pairs)
    equanimity = compute_direction(equanimity_pairs)

    return resilience.cpu(), equanimity.cpu()
