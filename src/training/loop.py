"""
Implements the main training loop and training state management.
"""
import gc
import logging
import math
import os

import torch
from src.training.callbacks import TrainerCallback, LoggingCallback, CheckpointCallback


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


class TrainingLoop:
    """Encapsulates the main training loop logic."""

    def __init__(
        self,
        trainer: "Trainer",
        config: "TrainConfig",
        callbacks: list[TrainerCallback] = None
    ):
        self.trainer = trainer
        self.config = config
        self.accelerator = trainer.accelerator
        self.callbacks = callbacks or []

    def _invoke_callbacks(self, method_name: str, *args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if callable(method):
                method(*args, **kwargs)

    def run(self, checkpoint_dir: str, resume_from: str | None):
        """Runs the main training loop."""
        if not self.callbacks:
             self.callbacks = [
                 LoggingCallback(),
                 CheckpointCallback(checkpoint_dir)
             ]

        total_epochs = (
            self.config.evolution.pretrain_epochs + self.config.evolution.evolution_epochs
        )
        training_state = TrainingState()
        self.accelerator.register_for_checkpointing(training_state)

        self._invoke_callbacks("on_train_begin", config=self.config, state=training_state)

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
            self._invoke_callbacks(
                "on_epoch_begin", epoch=epoch, config=self.config, state=training_state
            )

            is_pretrain = epoch < self.config.evolution.pretrain_epochs
            metrics = {}

            if is_pretrain:
                avg_loss, epoch_time = self.trainer.train_pretrain_epoch()
                metrics.update({
                    "avg_loss": avg_loss,
                    "epoch_time": epoch_time,
                    "learning_rate": self.trainer.get_learning_rate(),
                })
            else:
                epoch_time = self.trainer.run_evolution_cycle()
                metrics["epoch_time"] = epoch_time

            val_loss = self.trainer.run_validation()
            metrics["val_loss"] = val_loss

            self._invoke_callbacks(
                "on_epoch_end",
                epoch=epoch,
                config=self.config,
                state=training_state,
                metrics=metrics,
                accelerator=self.accelerator
            )

            # Check early stopping status from callbacks if needed,
            # or just rely on the CheckpointCallback's internal state.
            # For simplicity, we check if any callback wants to stop.
            # Here we just check the CheckpointCallback specifically if it's in the list.
            for cb in self.callbacks:
                 if isinstance(cb, CheckpointCallback):
                      if cb.epochs_no_improve >= self.config.evolution.early_stopping_patience:
                           logging.warning("Early stopping triggered by callback.")
                           break
            else:
                 # Continue loop
                 pass

            if any(isinstance(cb, CheckpointCallback) and cb.epochs_no_improve >= self.config.evolution.early_stopping_patience for cb in self.callbacks):
                break

            # Explicit memory management after each epoch/cycle.
            # Free Accelerator's internal state alongside standard garbage collection.
            self.accelerator.free_memory()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._invoke_callbacks("on_train_end", config=self.config, state=training_state)
