"""
PyTorch implementation of the Trainer class, which encapsulates the core training logic.
"""
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
