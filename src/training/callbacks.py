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

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

    def on_epoch_end(
        self, epoch: int, config: Any, state: Any, metrics: Dict[str, Any], **kwargs
    ) -> None:
        import logging
        import math
        val_loss = metrics.get("val_loss")
        accelerator = kwargs.get("accelerator")

        if val_loss is None or not math.isfinite(val_loss):
            logging.warning("Non-finite or missing validation loss. Skipping checkpoint saving.")
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
            if accelerator:
                accelerator.save_state(self.checkpoint_dir)
                logging.info(
                    "    - New best checkpoint saved (Val Loss: %.4f)",
                    self.best_val_loss,
                )
        else:
            self.epochs_no_improve += 1
            logging.info(
                "    - No improvement in validation loss for %d epochs.",
                self.epochs_no_improve,
            )
            if self.epochs_no_improve >= config.evolution.early_stopping_patience:
                logging.warning("Early stopping threshold reached.")
