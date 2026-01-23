"""
Main script for agent-centric training of the Transformer model using PyTorch.
"""
import logging
import os

import torch

from src.training.setup import prepare_training_environment
from src.training.trainer import create_trainer
from src.utils.cli import create_main_parser, make_model_name_required
from src.utils.core import main_entrypoint


@main_entrypoint
def main():
    """Main training script."""
    parser = create_main_parser()
    parser.add_argument(
        "--resume-from", type=str, help="Name of the model run to resume from."
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging."
    )
    make_model_name_required(parser)
    args = parser.parse_args()

    env = prepare_training_environment(args)

    trainer = create_trainer(
        env.config, env.data_components, env.accelerator, args.load_in_4bit
    )

    trainer.train(env.checkpoint_dir, args.resume_from)

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
