"""
PyTorch implementation of the Trainer class, which encapsulates the core training logic.
"""
import logging
import os

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config.core import TrainConfig
from src.config.model_config import TransformerConfig
from src.model.model import Transformer
from .runners import (DataComponents, EvolutionRunner, PretrainingRunner,
                      TrainerConfig, TrainingComponents, ValidationRunner)


def create_trainer(
    config: TrainConfig,
    data_components: "DataComponents",
    accelerator: "Accelerator",
    load_in_4bit: bool = False,
) -> "Trainer":
    """Initializes and returns a Trainer instance."""
    transformer_config = TransformerConfig(
        vocab_size=data_components.tokenizer.vocab_size,
        model=config.model,
        vision=config.vision,
        ltm=config.ltm,
        tokenizer=data_components.tokenizer,
    )
    model = Transformer(transformer_config, load_in_4bit=load_in_4bit)
    optimizer = Adam(model.parameters(), lr=config.optimizer.learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.scheduler.training_steps
    )
    value_loss_fn = nn.MSELoss()

    model, optimizer, scheduler, value_loss_fn = accelerator.prepare(
        model, optimizer, scheduler, value_loss_fn
    )

    training_components = TrainingComponents(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        value_loss_fn=value_loss_fn,
    )
    trainer_config = TrainerConfig(
        components=training_components,
        data=data_components,
        config=config,
        accelerator=accelerator,
    )
    return Trainer(trainer_config)


class TrainingState:
    """A simple class to hold and manage the training state like epoch."""

    def __init__(self, epoch=0):
        self.epoch = epoch

    def state_dict(self):
        """Returns the state of the training."""
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict):
        """Loads the training state."""
        self.epoch = state_dict["epoch"]


class Trainer:
    """
    Orchestrates the training process by delegating to specialized runners.
    """

    def __init__(self, trainer_config: TrainerConfig):
        self._config = trainer_config
        self._validation_runner = ValidationRunner(trainer_config)
        self._pretraining_runner = PretrainingRunner(trainer_config)
        self._evolution_runner = EvolutionRunner(trainer_config)

    def get_model(self) -> nn.Module:
        """Returns the underlying model."""
        return self._config.components.model

    def get_learning_rate(self) -> float:
        """Returns the current learning rate from the scheduler."""
        return self._config.components.scheduler.get_last_lr()[0]

    def run_validation(self) -> float:
        """Delegates validation to the ValidationRunner."""
        return self._validation_runner.run()

    def train_pretrain_epoch(self) -> tuple[float, float]:
        """Delegates pre-training to the PretrainingRunner."""
        return self._pretraining_runner.run()

    def run_evolution_cycle(self):
        """Delegates the evolution cycle to the EvolutionRunner."""
        return self._evolution_runner.run()

    def train(
        self, checkpoint_dir: str, resume_from: str | None
    ):  # pylint: disable=too-many-locals
        """Executes the main training loop."""
        accelerator = self._config.accelerator
        config = self._config.config

        if accelerator.is_main_process:
            logging.info("Starting training loop...")

        best_val_loss = float("inf")
        epochs_no_improve = 0
        total_epochs = (
            config.evolution.pretrain_epochs + config.evolution.evolution_epochs
        )
        training_state = TrainingState()
        accelerator.register_for_checkpointing(training_state)

        if accelerator.is_main_process and resume_from:
            try:
                checkpoint_path = os.path.join("models", resume_from, "checkpoints")
                accelerator.load_state(checkpoint_path)
                logging.info(
                    "Successfully loaded checkpoint. Starting from epoch %d.",
                    training_state.epoch,
                )
            except FileNotFoundError:
                logging.warning(
                    "Checkpoint not found at '%s'. Starting from scratch.",
                    checkpoint_path,
                )

        start_epoch = training_state.epoch
        for epoch in range(start_epoch, total_epochs):
            training_state.epoch = epoch
            is_pretrain = epoch < config.evolution.pretrain_epochs
            phase = "Pre-training" if is_pretrain else "Evolution"
            phase_epoch = (
                epoch if is_pretrain else epoch - config.evolution.pretrain_epochs
            )
            total_phase_epochs = (
                config.evolution.pretrain_epochs
                if is_pretrain
                else config.evolution.evolution_epochs
            )
            logging.info(
                "\n--- %s Epoch %d/%d ---",
                phase,
                phase_epoch + 1,
                total_phase_epochs,
            )

            if is_pretrain:
                avg_loss, epoch_time = self.train_pretrain_epoch()
                if accelerator.is_main_process:
                    accelerator.log(
                        {
                            "avg_loss": avg_loss,
                            "epoch_time": epoch_time,
                            "learning_rate": self.get_learning_rate(),
                        },
                        step=epoch,
                    )
                logging.info("    - Average Loss: %.4f", avg_loss)
            else:
                epoch_time = self.run_evolution_cycle()
                if accelerator.is_main_process:
                    accelerator.log({"epoch_time": epoch_time}, step=epoch)

            val_loss = self.run_validation()
            if accelerator.is_main_process:
                accelerator.log({"val_loss": val_loss}, step=epoch)
            logging.info(
                "    - Validation Loss: %.4f, Epoch Time: %.2fs",
                val_loss,
                epoch_time,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                accelerator.save_state(checkpoint_dir)
                logging.info(
                    "    - New best checkpoint saved (Val Loss: %.4f)",
                    best_val_loss,
                )
            else:
                epochs_no_improve += 1
                logging.info(
                    "    - No improvement in validation loss for %d epochs.",
                    epochs_no_improve,
                )

            if epochs_no_improve >= config.evolution.early_stopping_patience:
                logging.warning("Early stopping triggered.")
                break
