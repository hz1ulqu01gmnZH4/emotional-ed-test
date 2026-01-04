"""
Emotional Prefix Trainer for training the emotional modules.

Only updates the emotional encoder and prefix generator.
The LLM remains frozen throughout training.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import json
from pathlib import Path

from .context import EmotionalContext
from .model import EmotionalPrefixLLM


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    output_dir: str = "checkpoints/emotional_prefix"


@dataclass
class TrainingState:
    """State of training."""
    global_step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    train_losses: List[float] = field(default_factory=list)
    eval_losses: List[float] = field(default_factory=list)


class EmotionalPrefixTrainer:
    """
    Trainer for emotional prefix tuning.

    Only updates emotional encoder and prefix generator.
    """

    def __init__(
        self,
        model: EmotionalPrefixLLM,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: EmotionalPrefixLLM to train
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

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def compute_loss(
        self,
        outputs,
        labels: torch.Tensor,
        emotional_state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute loss with emotional weighting.

        Similar to FearEDAgent's fear-weighted loss:
        - Higher fear = more weight on safety-relevant tokens
        - Higher curiosity = encourage diverse outputs

        Args:
            outputs: LLM outputs with logits
            labels: Target token ids
            emotional_state: Current emotional state

        Returns:
            Weighted loss tensor
        """
        # Get logits and shift for next-token prediction
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute per-token loss
        batch_size, seq_len, vocab_size = shift_logits.size()
        token_losses = self.loss_fn(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        )
        token_losses = token_losses.view(batch_size, seq_len)

        # Get emotional weights
        fear = emotional_state['fear'].squeeze()
        joy = emotional_state['joy'].squeeze()

        # Emotional weighting factors
        # Higher fear increases weight (more careful about mistakes)
        # Higher joy slightly decreases weight (more lenient)
        base_weight = 1.0
        fear_weight = fear * 0.5   # Up to +50% weight with high fear
        joy_bonus = joy * 0.1     # Up to -10% weight with high joy

        emotional_weight = base_weight + fear_weight - joy_bonus
        emotional_weight = emotional_weight.clamp(0.5, 2.0)  # Bound weights

        # Apply weighting
        weighted_loss = (token_losses.mean(dim=-1) * emotional_weight).mean()

        return weighted_loss

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        context: EmotionalContext,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Single training step.

        Args:
            input_ids: Input token ids
            labels: Target token ids
            context: Emotional context
            attention_mask: Optional attention mask

        Returns:
            Tuple of (loss, emotional_state_values)
        """
        self.model.train()

        # Forward pass
        outputs, emotional_state = self.model(
            input_ids=input_ids,
            context=context,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Compute loss
        loss = self.compute_loss(outputs, labels, emotional_state)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.get_trainable_params(),
            self.config.max_grad_norm
        )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Extract emotional values for logging
        emotional_values = {
            k: v.item() for k, v in emotional_state.items()
            if isinstance(v, torch.Tensor) and v.numel() == 1
        }

        self.state.global_step += 1

        return loss.item(), emotional_values

    def train_on_batch(
        self,
        batch: Dict[str, Any],
        contexts: List[EmotionalContext],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Train on a batch of examples with multiple contexts.

        Args:
            batch: Dict with input_ids, attention_mask, labels
            contexts: List of emotional contexts, one per example

        Returns:
            Tuple of (avg_loss, avg_emotional_values)
        """
        total_loss = 0.0
        all_emotional_values = {}

        batch_size = batch['input_ids'].size(0)

        for i in range(batch_size):
            input_ids = batch['input_ids'][i:i+1]
            labels = batch['labels'][i:i+1] if 'labels' in batch else input_ids.clone()
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask[i:i+1]

            context = contexts[i] if i < len(contexts) else EmotionalContext()

            loss, emotional_values = self.train_step(
                input_ids, labels, context, attention_mask
            )
            total_loss += loss

            for k, v in emotional_values.items():
                if k not in all_emotional_values:
                    all_emotional_values[k] = []
                all_emotional_values[k].append(v)

        avg_loss = total_loss / batch_size
        avg_emotional_values = {
            k: sum(v) / len(v) for k, v in all_emotional_values.items()
        }

        self.state.train_losses.append(avg_loss)

        return avg_loss, avg_emotional_values

    def evaluate(
        self,
        eval_data: List[Tuple[str, EmotionalContext]],
    ) -> float:
        """
        Evaluate model on test data.

        Args:
            eval_data: List of (text, context) tuples

        Returns:
            Average loss
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for text, context in eval_data:
                tokens = self.model.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                ).to(self.model.device)

                input_ids = tokens.input_ids
                labels = input_ids.clone()

                outputs, emotional_state = self.model(
                    input_ids=input_ids,
                    context=context,
                    attention_mask=tokens.attention_mask,
                    labels=labels,
                )

                loss = self.compute_loss(outputs, labels, emotional_state)
                total_loss += loss.item()

        avg_loss = total_loss / len(eval_data)
        self.state.eval_losses.append(avg_loss)

        return avg_loss

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

        # Save trainable parameters
        torch.save({
            'emotion_encoder': self.model.emotion_encoder.state_dict(),
            'prefix_generator': self.model.prefix_generator.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, f"{path}/pytorch_model.bin")

        # Save training state
        with open(f"{path}/training_state.json", "w") as f:
            json.dump({
                'global_step': self.state.global_step,
                'epoch': self.state.epoch,
                'best_loss': self.state.best_loss,
                'train_losses': self.state.train_losses[-100:],  # Keep last 100
                'eval_losses': self.state.eval_losses,
            }, f, indent=2)

        # Save config
        with open(f"{path}/config.json", "w") as f:
            json.dump({
                'model_name': self.model.model_name,
                'prefix_length': self.model.prefix_length,
                'emotion_dim': self.model.emotion_dim,
            }, f, indent=2)

        print(f"Saved checkpoint to {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """
        Load trainable modules and training state.

        Args:
            path: Path to checkpoint directory
        """
        # Load trainable parameters
        checkpoint = torch.load(f"{path}/pytorch_model.bin", map_location=self.model.device)
        self.model.emotion_encoder.load_state_dict(checkpoint['emotion_encoder'])
        self.model.prefix_generator.load_state_dict(checkpoint['prefix_generator'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        # Load training state
        with open(f"{path}/training_state.json", "r") as f:
            state_dict = json.load(f)
            self.state.global_step = state_dict['global_step']
            self.state.epoch = state_dict['epoch']
            self.state.best_loss = state_dict['best_loss']
            self.state.train_losses = state_dict['train_losses']
            self.state.eval_losses = state_dict['eval_losses']

        print(f"Loaded checkpoint from {path}")
