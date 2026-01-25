"""
Handles the setup and configuration of the training environment.
"""
import argparse
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Optional

from accelerate import Accelerator

from src.config.core import TrainConfig
from src.data.data_loader import load_multimodal_data_from_directory
from src.data.tokenizer import Tokenizer
from src.utils.cli import apply_cli_args_to_config
from src.utils.setup import setup_logging


def load_and_prepare_data(
    data_dir: str, tokenizer: Tokenizer, validation_split: float
):
    """Loads and splits the data."""
    multimodal_data = load_multimodal_data_from_directory(data_dir)
    if not multimodal_data:
        raise ValueError(f"No data found in {data_dir}")
    all_text = " ".join([text for text, _ in multimodal_data])
    data_tokens = tokenizer.encode(all_text, add_special_tokens=True)
    split_idx = int(len(data_tokens) * (1 - validation_split))
    return tokenizer, data_tokens[:split_idx], data_tokens[split_idx:]


@dataclass
class RunPaths:
    """A container for all paths related to a training run."""
    model_dir: str
    checkpoint_dir: str
    config_path: str


def setup_paths_and_logging(model_name: str, resume_from: Optional[str]) -> RunPaths:
    """
    Sets up directories and logging for a new or resumed training run.

    - Creates model and checkpoint directories.
    - Copies the root config for a new run.
    - Sets up file-based logging.

    Returns:
        A RunPaths object containing the necessary paths.
    """
    model_dir = os.path.join("models", model_name)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")

    if resume_from:
        config_path = os.path.join("models", resume_from, "config.json")
        os.makedirs(model_dir, exist_ok=True)
        log_path = os.path.join(model_dir, "training.log")
        setup_logging(log_path)
        logging.info(
            "Resuming training from '%s'. New model and logs will be in '%s'.",
            resume_from,
            model_name,
        )
    else:
        if os.path.exists(model_dir):
            raise FileExistsError(f"Model directory '{model_dir}' already exists.")
        os.makedirs(model_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)

        root_config_path = "config_train.json"
        if not os.path.exists(root_config_path):
            raise FileNotFoundError(f"Root config file '{root_config_path}' not found.")
        config_path = os.path.join(model_dir, "config.json")
        shutil.copy(root_config_path, config_path)

        log_path = os.path.join(model_dir, "training.log")
        setup_logging(log_path)
        logging.info("Starting new training run: '%s'.", model_name)

    return RunPaths(
        model_dir=model_dir,
        checkpoint_dir=checkpoint_dir,
        config_path=config_path,
    )


def load_config_for_run(config_path: str) -> TrainConfig:
    """Loads the training configuration from the specified path."""
    return TrainConfig.from_json(config_path)


def save_updated_config(config: TrainConfig, tokenizer: Tokenizer, model_dir: str):
    """Saves the updated config with the correct vocab size."""
    config.model.vocab_size = tokenizer.vocab_size
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        f.write(config.model_dump_json(indent=4))


@dataclass
class TrainingEnvironment:
    """A container for all components required for training."""

    config: TrainConfig
    accelerator: Accelerator
    tokenizer: Tokenizer
    train_data: list
    val_data: list
    checkpoint_dir: str
    model_dir: str


def prepare_training_environment(
    args: argparse.Namespace,
) -> TrainingEnvironment:
    """
    Orchestrates the setup of the entire training environment.

    This function handles:
    1. Setting up directories and logging.
    2. Loading and preparing the configuration.
    3. Applying command-line argument overrides.
    4. Initializing the Accelerator for distributed training.
    5. Loading and preparing the dataset and tokenizer.
    6. Saving the final configuration.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A TrainingEnvironment object containing all necessary components.
    """
    paths = setup_paths_and_logging(args.model_name, args.resume_from)
    config = load_config_for_run(paths.config_path)
    apply_cli_args_to_config(args, config)

    accelerator = Accelerator(
        mixed_precision="fp16", log_with="wandb" if args.wandb else None
    )
    if accelerator.is_main_process and args.wandb:
        accelerator.init_trackers(
            project_name="transformer-project", config=config.model_dump()
        )

    # Initialize tokenizer from the source data directory, not the model output directory
    tokenizer, train_data, val_data = load_and_prepare_data(
        config.evolution.data_dir,
        Tokenizer(config.evolution.data_dir),
        config.evolution.validation_split,
    )

    if not args.resume_from:
        save_updated_config(config, tokenizer, paths.model_dir)
        tokenizer_vocab_path = os.path.join(
            config.evolution.data_dir, "tokenizer_vocab.json"
        )
        if os.path.exists(tokenizer_vocab_path):
            shutil.copy(tokenizer_vocab_path, paths.model_dir)

    return TrainingEnvironment(
        config=config,
        accelerator=accelerator,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        checkpoint_dir=paths.checkpoint_dir,
        model_dir=paths.model_dir,
    )
