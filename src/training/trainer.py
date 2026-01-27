"""
PyTorch implementation of the Trainer class, which encapsulates the core training logic.
"""
import logging
import os
from dataclasses import dataclass
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config.core import TrainConfig, TransformerConfig
from src.model.model import Transformer
from .runners import (EvolutionRunner, PretrainingRunner, ValidationRunner)


class TrainingLoop:
    """Encapsulates the main training loop logic."""

    def __init__(self, trainer: "Trainer", config: TrainConfig):
        self.trainer = trainer
        self.config = config
        self.accelerator = trainer.accelerator

    def run(self, checkpoint_dir: str, resume_from: str | None):
        if self.accelerator.is_main_process:
            logging.info("Starting training loop...")

        best_val_loss = float("inf")
        epochs_no_improve = 0
        total_epochs = (
            self.config.evolution.pretrain_epochs + self.config.evolution.evolution_epochs
        )
        training_state = TrainingState()
        self.accelerator.register_for_checkpointing(training_state)

        if self.accelerator.is_main_process and resume_from:
            checkpoint_path = os.path.join("models", resume_from, "checkpoints")
            try:
                self.accelerator.load_state(checkpoint_path)
                logging.info(
                    "Successfully loaded checkpoint. Starting from epoch %d.",
                    training_state.epoch,
                )
            except FileNotFoundError as e:
                logging.error(
                    "Checkpoint directory not found at '%s'. Please check the path.",
                    checkpoint_path,
                )
                raise e

        start_epoch = training_state.epoch
        for epoch in range(start_epoch, total_epochs):
            training_state.epoch = epoch
            is_pretrain = epoch < self.config.evolution.pretrain_epochs
            phase = "Pre-training" if is_pretrain else "Evolution"
            phase_epoch = (
                epoch if is_pretrain else epoch - self.config.evolution.pretrain_epochs
            )
            total_phase_epochs = (
                self.config.evolution.pretrain_epochs
                if is_pretrain
                else self.config.evolution.evolution_epochs
            )
            logging.info(
                "\n--- %s Epoch %d/%d ---",
                phase,
                phase_epoch + 1,
                total_phase_epochs,
            )

            if is_pretrain:
                avg_loss, epoch_time = self.trainer.train_pretrain_epoch()
                if self.accelerator.is_main_process:
                    self.accelerator.log(
                        {
                            "avg_loss": avg_loss,
                            "epoch_time": epoch_time,
                            "learning_rate": self.trainer.get_learning_rate(),
                        },
                        step=epoch,
                    )
                logging.info("    - Average Loss: %.4f", avg_loss)
            else:
                epoch_time = self.trainer.run_evolution_cycle()
                if self.accelerator.is_main_process:
                    self.accelerator.log({"epoch_time": epoch_time}, step=epoch)

            val_loss = self.trainer.run_validation()
            if self.accelerator.is_main_process:
                self.accelerator.log({"val_loss": val_loss}, step=epoch)
            logging.info(
                "    - Validation Loss: %.4f, Epoch Time: %.2fs",
                val_loss,
                epoch_time,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.accelerator.save_state(checkpoint_dir)
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

            if epochs_no_improve >= self.config.evolution.early_stopping_patience:
                logging.warning("Early stopping triggered.")
                break


def create_trainer(
    config: TrainConfig,
    tokenizer: "Tokenizer",
    train_data: list,
    val_data: list,
    accelerator: "Accelerator",
    load_in_4bit: bool = False,
) -> "Trainer":
    """Initializes and returns a Trainer instance."""
    transformer_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        model=config.model,
        vision=config.vision,
        ltm=config.ltm,
        tokenizer=tokenizer,
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

    if config.hardware.torch_compile:
        logging.info("Enabling torch.compile for the model.")
        model = torch.compile(model)

    validation_runner = ValidationRunner(
        model=model,
        val_data=val_data,
        tokenizer=tokenizer,
        evolution_config=config.evolution,
        accelerator=accelerator,
    )
    pretraining_runner = PretrainingRunner(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_data=train_data,
        tokenizer=tokenizer,
        evolution_config=config.evolution,
        optimizer_config=config.optimizer,
    )
    evolution_runner = EvolutionRunner(
        accelerator=accelerator,
        model=model,
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer,
        evolution_config=config.evolution,
    )

    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        value_loss_fn=value_loss_fn,
        config=config,
        accelerator=accelerator,
        validation_runner=validation_runner,
        pretraining_runner=pretraining_runner,
        evolution_runner=evolution_runner,
    )


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
