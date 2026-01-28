"""
PyTorch implementation of the Trainer class, which encapsulates the core training logic.
"""
import logging
import os
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config.core import TrainConfig
from src.training.loop import TrainingLoop
from src.training.runners import (EvolutionRunner, PretrainingRunner, ValidationRunner)


class Trainer:
    """
    Orchestrates the training process by delegating to specialized runners.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Adam,
        scheduler: CosineAnnealingLR,
        value_loss_fn: nn.Module,
        config: TrainConfig,
        accelerator,
        validation_runner: ValidationRunner,
        pretraining_runner: PretrainingRunner,
        evolution_runner: EvolutionRunner,
    ):
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
        """Returns the underlying model."""
        return self.model

    def get_learning_rate(self) -> float:
        """Returns the current learning rate from the scheduler."""
        return self.scheduler.get_last_lr()[0]

    def run_validation(self) -> float:
        """Delegates validation to the ValidationRunner."""
        return self._validation_runner.run()

    def train_pretrain_epoch(self) -> tuple[float, float]:
        """Delegates pre-training to the PretrainingRunner."""
        return self._pretraining_runner.run()

    def run_evolution_cycle(self):
        """Delegates the evolution cycle to the EvolutionRunner."""
        return self._evolution_runner.run()

    def save_final_model(self, model_dir: str):
        """Saves the final, unwrapped model for easy inference."""
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        output_path = os.path.join(model_dir, "model.pt")
        torch.save(unwrapped_model.state_dict(), output_path)
        logging.info("\nTraining complete! Final model saved to: %s", output_path)

    def train(self, checkpoint_dir: str, model_dir: str, resume_from: str | None):
        """Executes the main training loop and saves the final model."""
        loop = TrainingLoop(self, self.config)
        loop.run(checkpoint_dir, resume_from)
        if self.accelerator.is_main_process:
            self.save_final_model(model_dir)
