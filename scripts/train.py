"""
Main script for agent-centric training of the Transformer model using PyTorch.
"""
import logging
import os
import shutil

import torch
from accelerate import Accelerator

from src.config.core import TrainConfig
from src.data.data_loader import load_multimodal_data_from_directory
from src.data.tokenizer import Tokenizer
from src.training.trainer import create_trainer
from src.training.runners import DataComponents
from src.utils.cli import (
    create_main_parser,
    make_model_name_required,
    apply_cli_args_to_config,
)
from src.utils.core import (
    main_entrypoint,
    setup_logging,
)


def load_and_prepare_data(data_dir: str, tokenizer_path: str, validation_split: float):
    """Initializes tokenizer and loads data."""
    tokenizer = Tokenizer(tokenizer_path)
    multimodal_data = load_multimodal_data_from_directory(data_dir)
    if not multimodal_data:
        raise ValueError(f"No data found in {data_dir}")
    all_text = " ".join([text for text, _ in multimodal_data])
    data_tokens = tokenizer.encode(all_text, add_special_tokens=True)
    split_idx = int(len(data_tokens) * (1 - validation_split))
    return tokenizer, data_tokens[:split_idx], data_tokens[split_idx:]


def setup_environment(args):
    """Sets up directories, logging, and configuration."""
    model_dir = os.path.join('models', args.model_name)
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    resume_from_checkpoint = (
        os.path.join('models', args.resume_from, 'checkpoints')
        if args.resume_from
        else None
    )

    if resume_from_checkpoint:
        config_path = os.path.join('models', args.resume_from, 'config.json')
        os.makedirs(model_dir, exist_ok=True)
        log_path = os.path.join(model_dir, 'training.log')
        setup_logging(log_path)
        logging.info(
            "Resuming training from checkpoint '%s'. New model and logs will be in '%s'.",
            resume_from_checkpoint,
            args.model_name,
        )
    else:
        if os.path.exists(model_dir):
            raise FileExistsError(f"Model directory '{model_dir}' already exists.")
        os.makedirs(model_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        config_path = 'config_train.json'
        log_path = os.path.join(model_dir, 'training.log')
        setup_logging(log_path)
        logging.info("Starting new training run: '%s'.", args.model_name)

    config = TrainConfig.from_json(config_path)
    return config, model_dir, checkpoint_dir, resume_from_checkpoint


def save_updated_config(config: TrainConfig, tokenizer: Tokenizer, model_dir: str):
    """Saves the updated config with the correct vocab size."""
    config.model.vocab_size = tokenizer.vocab_size
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        f.write(config.model_dump_json(indent=4))

@main_entrypoint
def main():
    """Main training script."""
    parser = create_main_parser()
    parser.add_argument(
        '--resume-from', type=str, help="Name of the model run to resume from."
    )
    parser.add_argument(
        '--wandb', action='store_true', help="Enable Weights & Biases logging."
    )
    make_model_name_required(parser)
    args = parser.parse_args()

    config, model_dir, checkpoint_dir, resume_from_checkpoint = setup_environment(args)
    apply_cli_args_to_config(args, config)

    accelerator = Accelerator(mixed_precision="fp16", log_with="wandb" if args.wandb else None)
    if accelerator.is_main_process and args.wandb:
        accelerator.init_trackers(
            project_name="transformer-project",
            config=config.model_dump()
        )

    tokenizer, train_data, val_data = load_and_prepare_data(
        config.evolution.data_dir,
        model_dir,
        config.evolution.validation_split,
    )

    # Save config and tokenizer vocab only on new runs
    if not resume_from_checkpoint:
        save_updated_config(config, tokenizer, model_dir)
        if os.path.exists(os.path.join(config.evolution.data_dir, 'tokenizer_vocab.json')):
            shutil.copy(
                os.path.join(config.evolution.data_dir, 'tokenizer_vocab.json'),
                model_dir,
            )

    data_components = DataComponents(
        tokenizer=tokenizer, train_data=train_data, val_data=val_data
    )
    trainer = create_trainer(
        config, data_components, accelerator, args.load_in_4bit
    )

    trainer.train(checkpoint_dir, args.resume_from)

    # Save the final, unwrapped model for easy inference
    unwrapped_model = accelerator.unwrap_model(trainer.get_model())
    torch.save(unwrapped_model.state_dict(), os.path.join(model_dir, 'model.pt'))
    logging.info(
        "\nTraining complete! Final model saved to: %s", model_dir
    )

if __name__ == "__main__":
    main()
