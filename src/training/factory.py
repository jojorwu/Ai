"""
Factory functions for creating training-related components.
"""
import logging
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.agent.agent_manager import AgentManager
from src.agent.evaluator import CollaborativeEvaluator
from src.config.core import TrainConfig, TransformerConfig
from src.model.model import Transformer
from src.training.runners import (EvolutionRunner, PretrainingRunner, ValidationRunner)
from src.training.trainer import Trainer


def create_trainer(
    config: TrainConfig,
    tokenizer: "Tokenizer",
    train_data: list,
    val_data: list,
    accelerator: "Accelerator",
    load_in_4bit: bool = False,
) -> Trainer:
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
        hardware_config=config.hardware,
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
        hardware_config=config.hardware,
    )
    evaluator = CollaborativeEvaluator(agents=[], base_model=model)
    agent_manager = AgentManager(
        base_model=model,
        num_agents=config.evolution.num_agents,
        accelerator=accelerator,
        evaluator=evaluator,
        hardware_config=config.hardware,
    )
    evaluator.agents = agent_manager.agents

    evolution_runner = EvolutionRunner(
        accelerator=accelerator,
        model=model,
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer,
        evolution_config=config.evolution,
        agent_manager=agent_manager,
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
