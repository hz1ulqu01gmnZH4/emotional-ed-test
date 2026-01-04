"""
Trainer for the Emotional Reward Model.

Trains on labeled emotional contexts and feedback signals.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from .model import EmotionalRewardLLM
from .signals import EmotionalSignals


class ERMTrainer:
    """
    Trainer for the Emotional Reward Model.

    Supports:
    1. Supervised training on labeled emotional contexts
    2. Feedback-based learning (RLHF-style)
    3. Contrastive learning on response pairs
    """

    def __init__(
        self,
        model: EmotionalRewardLLM,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        """
        Initialize trainer.

        Args:
            model: EmotionalRewardLLM to train
            lr: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model
        self.device = model.device

        # Optimizer for trainable components only
        self.optimizer = torch.optim.AdamW(
            model.get_trainable_params(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Loss weights
        self.emotion_loss_weight = 1.0
        self.fear_loss_weight = 0.5

        # Training history
        self.history: Dict[str, List[float]] = {
            "emotion_loss": [],
            "feedback_loss": [],
            "total_loss": [],
        }

    def train_on_labeled_emotions(
        self,
        texts: List[str],
        target_emotions: List[EmotionalSignals],
    ) -> float:
        """
        Train ERM to predict target emotional states.

        Args:
            texts: List of input texts
            target_emotions: List of target EmotionalSignals

        Returns:
            Average loss
        """
        self.model.erm.train()
        self.model.logit_modulator.train()
        self.model.fear_module.train()
        total_loss = 0.0

        for text, target in zip(texts, target_emotions):
            self.optimizer.zero_grad()

            # Get model predictions - use raw tensor from ERM
            inputs = self.model.tokenizer(text, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)

            # Get LLM hidden states
            with torch.no_grad():
                llm_outputs = self.model.llm(
                    input_ids=input_ids,
                    output_hidden_states=True,
                )
            hidden_states = llm_outputs.hidden_states[-1]

            # Get ERM predictions (this has gradients)
            _, predicted_tensor = self.model.erm(hidden_states)
            target_tensor = target.to_tensor(device=self.device).unsqueeze(0)

            # Emotion prediction loss
            emotion_loss = F.mse_loss(predicted_tensor, target_tensor)

            emotion_loss.backward()
            self.optimizer.step()

            total_loss += emotion_loss.item()

        avg_loss = total_loss / len(texts)
        self.history["emotion_loss"].append(avg_loss)
        return avg_loss

    def train_on_feedback(
        self,
        query: str,
        response: str,
        feedback: float,
    ) -> float:
        """
        Train from outcome feedback (RLHF-style).

        If feedback was negative and we weren't cautious enough,
        increase fear association. If positive and too fearful,
        reduce fear.

        Args:
            query: Original user query
            response: Generated response
            feedback: User feedback (-1 to 1)

        Returns:
            Loss value (0 if no update)
        """
        self.model.train()

        # Tokenize full interaction
        full_text = f"{query} {response}"
        inputs = self.model.tokenizer(full_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)

        # Get current emotional prediction
        _, emotions = self.model(input_ids, return_emotions=True)

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # If bad outcome and low fear → should have been more fearful
        if feedback < -0.3 and emotions.fear < 0.5:
            target_fear = torch.tensor([0.8], device=self.device)
            predicted_fear = torch.tensor([emotions.fear], device=self.device, requires_grad=True)
            loss = F.mse_loss(predicted_fear, target_fear)

        # If good outcome and high fear → fear was unnecessary
        elif feedback > 0.3 and emotions.fear > 0.5:
            target_fear = torch.tensor([0.2], device=self.device)
            predicted_fear = torch.tensor([emotions.fear], device=self.device, requires_grad=True)
            loss = F.mse_loss(predicted_fear, target_fear)

        # If high curiosity led to good outcome → reinforce curiosity
        elif feedback > 0.5 and emotions.curiosity > 0.3:
            target_curiosity = torch.tensor([0.9], device=self.device)
            predicted_curiosity = torch.tensor([emotions.curiosity], device=self.device, requires_grad=True)
            loss = F.mse_loss(predicted_curiosity, target_curiosity)

        if loss.item() > 0:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.history["feedback_loss"].append(loss.item())
        return loss.item()

    def train_contrastive(
        self,
        query: str,
        good_response: str,
        bad_response: str,
        margin: float = 0.5,
    ) -> float:
        """
        Train using contrastive pairs of responses.

        Good responses should have higher confidence/joy,
        bad responses should trigger more fear/caution.

        Args:
            query: Original query
            good_response: Response with positive outcome
            bad_response: Response with negative outcome
            margin: Margin for contrastive loss

        Returns:
            Contrastive loss value
        """
        self.model.erm.train()
        self.model.logit_modulator.train()
        self.model.fear_module.train()
        self.optimizer.zero_grad()

        # Get emotions for good response - use raw tensors
        good_text = f"{query} {good_response}"
        good_inputs = self.model.tokenizer(good_text, return_tensors="pt")
        with torch.no_grad():
            good_llm = self.model.llm(
                good_inputs.input_ids.to(self.device),
                output_hidden_states=True,
            )
        _, good_tensor = self.model.erm(good_llm.hidden_states[-1])

        # Get emotions for bad response
        bad_text = f"{query} {bad_response}"
        bad_inputs = self.model.tokenizer(bad_text, return_tensors="pt")
        with torch.no_grad():
            bad_llm = self.model.llm(
                bad_inputs.input_ids.to(self.device),
                output_hidden_states=True,
            )
        _, bad_tensor = self.model.erm(bad_llm.hidden_states[-1])

        # Good response should have higher confidence (idx 5), lower fear (idx 0)
        # good_tensor shape: [1, 6], bad_tensor shape: [1, 6]
        margin_tensor = torch.tensor(margin, device=self.device)

        confidence_diff = good_tensor[0, 5] - bad_tensor[0, 5]  # confidence
        fear_diff = bad_tensor[0, 0] - good_tensor[0, 0]  # fear

        confidence_loss = F.relu(margin_tensor - confidence_diff)
        fear_loss = F.relu(margin_tensor - fear_diff)

        loss = confidence_loss + fear_loss
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(
        self,
        labeled_data: Optional[List[Tuple[str, EmotionalSignals]]] = None,
        feedback_data: Optional[List[Tuple[str, str, float]]] = None,
        contrastive_data: Optional[List[Tuple[str, str, str]]] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch on all available data.

        Args:
            labeled_data: List of (text, target_emotions) tuples
            feedback_data: List of (query, response, feedback) tuples
            contrastive_data: List of (query, good_response, bad_response) tuples

        Returns:
            Dict of average losses
        """
        losses = {}

        if labeled_data:
            texts = [t for t, _ in labeled_data]
            emotions = [e for _, e in labeled_data]
            losses["labeled"] = self.train_on_labeled_emotions(texts, emotions)

        if feedback_data:
            fb_losses = []
            for query, response, feedback in feedback_data:
                fb_losses.append(
                    self.train_on_feedback(query, response, feedback)
                )
            losses["feedback"] = sum(fb_losses) / len(fb_losses)

        if contrastive_data:
            cont_losses = []
            for query, good, bad in contrastive_data:
                cont_losses.append(
                    self.train_contrastive(query, good, bad)
                )
            losses["contrastive"] = sum(cont_losses) / len(cont_losses)

        return losses

    def save_checkpoint(self, path: str, epoch: int) -> None:
        """
        Save training checkpoint.

        Args:
            path: Path to save
            epoch: Current epoch
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "epoch": epoch,
            "model_state": {
                "erm": self.model.erm.state_dict(),
                "logit_modulator": self.model.logit_modulator.state_dict(),
                "fear_module": self.model.fear_module.state_dict(),
            },
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.history,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Epoch number
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.erm.load_state_dict(checkpoint["model_state"]["erm"])
        self.model.logit_modulator.load_state_dict(
            checkpoint["model_state"]["logit_modulator"]
        )
        self.model.fear_module.load_state_dict(
            checkpoint["model_state"]["fear_module"]
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.history = checkpoint["history"]

        return checkpoint["epoch"]

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self.history

    def reset_history(self) -> None:
        """Reset training history."""
        self.history = {
            "emotion_loss": [],
            "feedback_loss": [],
            "total_loss": [],
        }
