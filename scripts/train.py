"""
Main script for agent-centric training of the Transformer model using PyTorch.
"""
import argparse
import logging
import os

import torch

from src.config.core import TrainConfig
from src.training.setup import prepare_training_environment
from src.training.trainer import create_trainer
from src.utils.cli import make_model_name_required
from src.utils.decorators import main_entrypoint
from src.utils.setup import setup_from_args


def add_training_args(parser: argparse.ArgumentParser):
    """Adds arguments specific to training."""
    parser.add_argument(
        "--resume-from", type=str, help="Name of the model run to resume from."
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging."
    )
    make_model_name_required(parser)


@main_entrypoint
def main():
    """Main training script."""
    app_setup = setup_from_args(TrainConfig, add_training_args)
    env = prepare_training_environment(app_setup.args)

    trainer = create_trainer(
        config=env.config,
        tokenizer=env.tokenizer,
        train_data=env.train_data,
        val_data=env.val_data,
        accelerator=env.accelerator,
        load_in_4bit=app_setup.args.load_in_4bit,
    )

    trainer.train(env.checkpoint_dir, env.model_dir, app_setup.args.resume_from)


if __name__ == "__main__":
    main()
