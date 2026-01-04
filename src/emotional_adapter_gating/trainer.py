"""
Emotional Adapter Trainer for training the adapter modules.

Only updates adapters and emotion-related parameters.
The LLM remains frozen throughout training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path

from .state import EmotionalState
from .model import EmotionalAdapterLLM


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    batch_size: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    logging_steps: int = 10
    output_dir: str = "checkpoints/emotional_adapter"


@dataclass
class TrainingState:
    """State of training."""
    global_step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    train_losses: List[float] = field(default_factory=list)


class EmotionalAdapterTrainer:
    """
    Trainer for emotional adapters.

    Only updates adapter parameters - LLM stays frozen.
    """

    def __init__(
        self,
        model: EmotionalAdapterLLM,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: EmotionalAdapterLLM to train
            config: Training configuration
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.state = TrainingState()

        # Setup optimizer for trainable params only
        self.optimizer = AdamW(
            model.get_trainable_params(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def compute_loss(
        self,
        outputs,
        labels: torch.Tensor,
        emotion_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with optional emotional weighting.

        Args:
            outputs: Model outputs with logits
            labels: Target token ids
            emotion_tensor: Current emotional state tensor

        Returns:
            Weighted loss tensor
        """
        # Get logits and shift for next-token prediction
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        return loss

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        emotional_state: EmotionalState,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Single training step.

        Args:
            input_ids: Input token ids
            labels: Target token ids
            emotional_state: EmotionalState object
            attention_mask: Optional attention mask

        Returns:
            Tuple of (loss, metrics dict)
        """
        self.model.train()

        # Forward pass
        outputs, emotion_tensor = self.model(
            input_ids=input_ids,
            emotional_state=emotional_state,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Compute loss
        loss = self.compute_loss(outputs, labels, emotion_tensor)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.get_trainable_params(),
            self.config.max_grad_norm
        )

        # Optimizer step
        self.optimizer.step()

        # Metrics
        metrics = {
            "loss": loss.item(),
            "emotion_norm": torch.norm(emotion_tensor).item(),
        }

        self.state.global_step += 1
        self.state.train_losses.append(loss.item())

        return loss.item(), metrics

    def evaluate(
        self,
        eval_data: List[Tuple[str, EmotionalState]],
    ) -> float:
        """
        Evaluate model on test data.

        Args:
            eval_data: List of (text, emotional_state) tuples

        Returns:
            Average loss
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for text, emotional_state in eval_data:
                tokens = self.model.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                ).to(self.model.device)

                input_ids = tokens.input_ids
                labels = input_ids.clone()

                outputs, emotion_tensor = self.model(
                    input_ids=input_ids,
                    emotional_state=emotional_state,
                    attention_mask=tokens.attention_mask,
                    labels=labels,
                )

                loss = self.compute_loss(outputs, labels, emotion_tensor)
                total_loss += loss.item()

        return total_loss / len(eval_data)

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """
        Save trainable modules and training state.

        Args:
            path: Optional custom path

        Returns:
            Path where checkpoint was saved
        """
        if path is None:
            path = f"{self.config.output_dir}/checkpoint-{self.state.global_step}"

        Path(path).mkdir(parents=True, exist_ok=True)

        # Save adapter parameters
        adapter_state = {
            name: adapter.state_dict()
            for name, adapter in self.model.adapters.items()
        }

        torch.save({
            'adapters': adapter_state,
            'layer_emotion_weights': self.model.layer_emotion_weights,
            'optimizer': self.optimizer.state_dict(),
        }, f"{path}/pytorch_model.bin")

        # Save training state
        with open(f"{path}/training_state.json", "w") as f:
            json.dump({
                'global_step': self.state.global_step,
                'epoch': self.state.epoch,
                'best_loss': self.state.best_loss,
                'train_losses': self.state.train_losses[-100:],
            }, f, indent=2)

        # Save config
        with open(f"{path}/config.json", "w") as f:
            json.dump({
                'model_name': self.model.model_name,
                'adapter_dim': self.model.adapter_dim,
                'emotion_dim': self.model.emotion_dim,
                'gate_type': self.model.gate_type,
            }, f, indent=2)

        print(f"Saved checkpoint to {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """
        Load trainable modules and training state.

        Args:
            path: Path to checkpoint directory
        """
        checkpoint = torch.load(
            f"{path}/pytorch_model.bin",
            map_location=self.model.device
        )

        # Load adapter parameters
        for name, state_dict in checkpoint['adapters'].items():
            self.model.adapters[name].load_state_dict(state_dict)

        # Load emotion weights
        self.model.layer_emotion_weights.data = checkpoint['layer_emotion_weights']

        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        # Load training state
        with open(f"{path}/training_state.json", "r") as f:
            state_dict = json.load(f)
            self.state.global_step = state_dict['global_step']
            self.state.epoch = state_dict['epoch']
            self.state.best_loss = state_dict['best_loss']
            self.state.train_losses = state_dict['train_losses']

        print(f"Loaded checkpoint from {path}")

    def analyze_gate_patterns(
        self,
        prompts: List[str],
        emotional_states: List[EmotionalState],
    ) -> Dict[str, List[float]]:
        """
        Analyze how gates activate for different emotional states.

        Args:
            prompts: List of text prompts
            emotional_states: Corresponding emotional states

        Returns:
            Dict mapping layer indices to gate values
        """
        self.model.eval()
        gate_values = {str(i): [] for i in self.model.adapter_layer_indices}

        with torch.no_grad():
            for prompt, state in zip(prompts, emotional_states):
                tokens = self.model.tokenizer(
                    prompt, return_tensors="pt"
                ).to(self.model.device)

                emotion_tensor = self.model.emotion_encoder(state)

                for layer_idx in self.model.adapter_layer_indices:
                    adapter = self.model.adapters[str(layer_idx)]
                    # Get gate value (for scalar gate)
                    gate = adapter.gate(emotion_tensor)
                    gate_values[str(layer_idx)].append(gate.mean().item())

        return gate_values
