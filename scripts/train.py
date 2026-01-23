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
from src.utils.core import main_entrypoint
from src.utils.setup import setup_from_args


def add_training_args(parser: argparse.ArgumentParser):
    """Adds arguments specific to training."""
    parser.add_argument(
        "--resume-from", type=str, help="Name of the model run to resume from."
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging."
    )


@main_entrypoint
def main():
    """Main training script."""
    app_setup = setup_from_args(TrainConfig, add_training_args)
    env = prepare_training_environment(app_setup.args)

    trainer = create_trainer(
        env.config, env.data_components, env.accelerator, app_setup.args.load_in_4bit
    )

    trainer.train(env.checkpoint_dir, app_setup.args.resume_from)

    # Save the final, unwrapped model for easy inference
    unwrapped_model = env.accelerator.unwrap_model(trainer.get_model())
    torch.save(
        unwrapped_model.state_dict(), os.path.join(env.model_dir, "model.pt")
    )
    logging.info(
        "\nTraining complete! Final model saved to: %s", env.model_dir
    )


if __name__ == "__main__":
    main()
