"""
Defines the callback system for the training loop.
"""
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

@runtime_checkable
class TrainerCallback(Protocol):
    """
    Protocol for training loop callbacks.
    """

    def on_train_begin(self, config: Any, state: Any, **kwargs) -> None:
        """Called at the beginning of training."""

    def on_train_end(self, config: Any, state: Any, **kwargs) -> None:
        """Called at the end of training."""

    def on_epoch_begin(self, epoch: int, config: Any, state: Any, **kwargs) -> None:
        """Called at the beginning of each epoch."""

    def on_epoch_end(
        self, epoch: int, config: Any, state: Any, metrics: Dict[str, Any], **kwargs
    ) -> None:
        """Called at the end of each epoch."""

    def on_step_end(
        self, step: int, config: Any, state: Any, metrics: Dict[str, Any], **kwargs
    ) -> None:
        """Called at the end of each training step."""

    def on_interrupt(self, config: Any, state: Any, **kwargs) -> None:
        """Called when training is interrupted (e.g., Ctrl+C)."""


class LoggingCallback:
    """Default callback for logging training progress."""

    def on_train_begin(self, config: Any, state: Any, **kwargs) -> None:
        import logging
        logging.info("Starting training loop...")

    def on_train_end(self, config: Any, state: Any, **kwargs) -> None:
        import logging
        logging.info("Training complete.")

    def on_epoch_begin(self, epoch: int, config: Any, state: Any, **kwargs) -> None:
        import logging
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

    def on_epoch_end(
        self, epoch: int, config: Any, state: Any, metrics: Dict[str, Any], **kwargs
    ) -> None:
        import logging
        accelerator = kwargs.get("accelerator")
        if accelerator and accelerator.is_main_process:
            accelerator.log(metrics, step=epoch)

        log_parts = [f"    - Epoch {epoch}"]
        for k, v in metrics.items():
             if isinstance(v, float):
                 log_parts.append(f"{k}: {v:.4f}")
             else:
                 log_parts.append(f"{k}: {v}")
        logging.info(", ".join(log_parts))


class CheckpointCallback:
    """Callback for saving model checkpoints."""

    def __init__(self, best_dir: str, latest_dir: str):
        self.best_dir = best_dir
        self.latest_dir = latest_dir

    def on_epoch_end(
        self, epoch: int, config: Any, state: Any, metrics: Dict[str, Any], **kwargs
    ) -> None:
        import logging
        import math
        import os

        val_loss = metrics.get("val_loss")
        accelerator = kwargs.get("accelerator")

        if accelerator:
            # Always save to the latest directory for resuming.
            os.makedirs(self.latest_dir, exist_ok=True)
            accelerator.save_state(self.latest_dir)
            logging.info("    - Latest state saved to %s", self.latest_dir)

        if val_loss is None or not math.isfinite(val_loss):
            logging.warning("Non-finite or missing validation loss. Skipping 'best' checkpoint.")
            return

        if val_loss < state.best_val_loss:
            state.best_val_loss = val_loss
            state.epochs_no_improve = 0
            if accelerator:
                os.makedirs(self.best_dir, exist_ok=True)
                accelerator.save_state(self.best_dir)
                logging.info(
                    "    - New best checkpoint saved (Val Loss: %.4f)",
                    state.best_val_loss,
                )
        else:
            state.epochs_no_improve += 1
            logging.info(
                "    - No improvement in validation loss for %d epochs (Best: %.4f).",
                state.epochs_no_improve,
                state.best_val_loss,
            )

    def on_interrupt(self, config: Any, state: Any, **kwargs) -> None:
        import logging
        import os
        accelerator = kwargs.get("accelerator")
        if accelerator:
            logging.info("Interrupt detected. Saving current state to %s...", self.latest_dir)
            os.makedirs(self.latest_dir, exist_ok=True)
            accelerator.save_state(self.latest_dir)
