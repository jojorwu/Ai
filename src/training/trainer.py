"""
PyTorch implementation of the Trainer class, which encapsulates the core training logic.
"""
from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, Any, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.training.loop import TrainingLoop

if TYPE_CHECKING:
    from src.config.core import TrainConfig
    from src.training.runners import (
        EvolutionRunner,
        PretrainingRunner,
        ValidationRunner,
    )


class Trainer:
    """
    Orchestrates the training process by delegating to specialized runners.

    This class provides high-level methods to run validation, pre-training,
    and evolution cycles, and handles model saving.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Adam,
        scheduler: CosineAnnealingLR,
        value_loss_fn: nn.Module,
        config: TrainConfig,
        accelerator: Any,
        validation_runner: ValidationRunner,
        pretraining_runner: PretrainingRunner,
        evolution_runner: EvolutionRunner,
    ) -> None:
        """
        Initializes the Trainer.

        Args:
            model: The Transformer model to train.
            optimizer: The optimizer for the base model.
            scheduler: Learning rate scheduler.
            value_loss_fn: Loss function for the value head.
            config: Training configuration.
            accelerator: Accelerator instance for distributed training.
            validation_runner: Runner for model validation.
            pretraining_runner: Runner for pre-training epochs.
            evolution_runner: Runner for the agent evolution process.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.value_loss_fn = value_loss_fn
        self.config = config
        self.accelerator = accelerator
        self._validation_runner = validation_runner
        self._pretraining_runner = pretraining_runner
        self._evolution_runner = evolution_runner

    def get_model(self) -> nn.Module:
        """
        Returns the underlying model.

        Returns:
            The PyTorch model instance.
        """
        return self.model

    def get_learning_rate(self) -> float:
        """
        Returns the current learning rate from the scheduler.

        Returns:
            The current learning rate as a float.
        """
        return self.scheduler.get_last_lr()[0]

    def run_validation(self) -> float:
        """
        Delegates validation to the ValidationRunner.

        Returns:
            The average validation loss.
        """
        return self._validation_runner.run()

    def train_pretrain_epoch(self) -> Tuple[float, float]:
        """
        Delegates pre-training to the PretrainingRunner.

        Returns:
            A tuple of (average_loss, epoch_time).
        """
        return self._pretraining_runner.run()

    def run_evolution_cycle(self) -> float:
        """
        Delegates the evolution cycle to the EvolutionRunner.

        Returns:
            The time taken for the evolution cycle.
        """
        return self._evolution_runner.run()

    def save_final_model(self, model_dir: str) -> None:
        """
        Saves the final, unwrapped model for easy inference.

        Args:
            model_dir: Directory where the model will be saved.
        """
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        output_path = os.path.join(model_dir, "model.pt")
        torch.save(unwrapped_model.state_dict(), output_path)
        logging.info("\nTraining complete! Final model saved to: %s", output_path)

    def train(self, checkpoint_dir: str, model_dir: str, resume_from: str | None) -> None:
        """
        Executes the main training loop and saves the final model.

        Args:
            checkpoint_dir: Directory to store checkpoints.
            model_dir: Directory to save the final model.
            resume_from: Optional name of the model to resume from.
        """
        loop = TrainingLoop(self, self.config)
        loop.run(checkpoint_dir, resume_from)
        if self.accelerator.is_main_process:
            self.save_final_model(model_dir)
