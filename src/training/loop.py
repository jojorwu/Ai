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

    def __init__(self, epoch=0, best_val_loss=float('inf'), epochs_no_improve=0):
        self.epoch = epoch
        self.best_val_loss = best_val_loss
        self.epochs_no_improve = epochs_no_improve

    def state_dict(self):
        """Returns the state of the training."""
        return {
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "epochs_no_improve": self.epochs_no_improve
        }

    def load_state_dict(self, state_dict):
        """Loads the training state."""
        self.epoch = state_dict.get("epoch", 0)
        self.best_val_loss = state_dict.get("best_val_loss", float('inf'))
        self.epochs_no_improve = state_dict.get("epochs_no_improve", 0)


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
            # Default callbacks if none provided.
            # We assume standard structure if not provided.
            import os
            self.callbacks = [
                LoggingCallback(),
                CheckpointCallback(
                    best_dir=os.path.join(checkpoint_dir, "best"),
                    latest_dir=os.path.join(checkpoint_dir, "latest")
                )
            ]

        total_epochs = (
            self.config.evolution.pretrain_epochs + self.config.evolution.evolution_epochs
        )
        training_state = TrainingState()
        self.accelerator.register_for_checkpointing(training_state)

        self._invoke_callbacks("on_train_begin", config=self.config, state=training_state)

        if self.accelerator.is_main_process and resume_from:
            # Check for latest, then best subdirectories
            base_checkpoint_path = os.path.join("models", resume_from, "checkpoints")
            potential_paths = [
                os.path.join(base_checkpoint_path, "latest"),
                os.path.join(base_checkpoint_path, "best"),
                base_checkpoint_path # Backward compatibility
            ]

            loaded = False
            for path in potential_paths:
                if os.path.exists(path) and any(os.scandir(path)):
                    try:
                        self.accelerator.load_state(path)
                        logging.info(
                            "Successfully loaded checkpoint from %s. Resuming from epoch %d.",
                            path,
                            training_state.epoch,
                        )
                        loaded = True
                        break
                    except Exception as e:
                        logging.warning("Failed to load checkpoint from %s: %s", path, e)

            if not loaded:
                 logging.error("No valid checkpoint found to resume from in %s", base_checkpoint_path)

        start_epoch = training_state.epoch
        try:
            for epoch in range(start_epoch, total_epochs):
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

                # Increment epoch only after successful completion and saving
                training_state.epoch = epoch + 1

                # Check early stopping status from callbacks
                should_stop = False
                for cb in self.callbacks:
                    if isinstance(cb, CheckpointCallback):
                        if cb.epochs_no_improve >= self.config.evolution.early_stopping_patience:
                            logging.warning("Early stopping triggered by callback.")
                            should_stop = True
                            break
                if should_stop:
                    break

                # Explicit memory management after each epoch/cycle.
                self.accelerator.free_memory()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except KeyboardInterrupt:
            logging.info("Training interrupted by user.")
            self._invoke_callbacks(
                "on_interrupt",
                config=self.config,
                state=training_state,
                accelerator=self.accelerator
            )
            raise

        self._invoke_callbacks("on_train_end", config=self.config, state=training_state)
