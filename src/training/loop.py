"""
Implements the main training loop and training state management.
"""
import gc
import logging
import math
import os

import torch


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

    def __init__(self, trainer: "Trainer", config: "TrainConfig"):
        self.trainer = trainer
        self.config = config
        self.accelerator = trainer.accelerator

    def run(self, checkpoint_dir: str, resume_from: str | None):
        """Runs the main training loop."""
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

            # Check for non-finite validation loss before proceeding or saving
            if not math.isfinite(val_loss):
                logging.warning("Non-finite validation loss encountered. Skipping checkpoint saving.")
            elif val_loss < best_val_loss:
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

            # Explicit memory management after each epoch/cycle
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
