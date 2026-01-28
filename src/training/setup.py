"""
Handles the setup and configuration of the training environment.
"""
import argparse
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Optional

import torch
from accelerate import Accelerator

from src.config.core import TrainConfig
from src.data.data_loader import load_multimodal_data_from_directory
from src.data.tokenizer import Tokenizer
from src.model.device_manager import DeviceManager
from src.utils.setup import setup_logging


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


def setup_environment(args: argparse.Namespace):
    """Sets up directories, logging, and configuration."""
    model_dir = os.path.join("models", args.model_name)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    resume_from = args.resume_from

    if resume_from:
        config_path = os.path.join("models", resume_from, "config.json")
        os.makedirs(model_dir, exist_ok=True)
        log_path = os.path.join(model_dir, "training.log")
        setup_logging(log_path)
        logging.info(
            "Resuming training from '%s'. New model and logs will be in '%s'.",
            resume_from,
            args.model_name,
        )
    else:
        if os.path.exists(model_dir):
            raise FileExistsError(f"Model directory '{model_dir}' already exists.")
        os.makedirs(model_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Make training run self-contained by copying the config
        root_config_path = "config/config_train.json"
        if not os.path.exists(root_config_path):
            raise FileNotFoundError(
                f"Root config file '{root_config_path}' not found."
            )
        config_path = os.path.join(model_dir, "config.json")
        shutil.copy(root_config_path, config_path)

        log_path = os.path.join(model_dir, "training.log")
        setup_logging(log_path)
        logging.info("Starting new training run: '%s'.", args.model_name)

    config = TrainConfig.from_json(config_path)
    return config, model_dir, checkpoint_dir, resume_from


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
    """
    config, model_dir, checkpoint_dir, resume_from = setup_environment(args)

    # Apply hardware optimizations
    device_manager = DeviceManager(config.hardware)
    device_manager.optimize_environment()

    # Determine best mixed precision strategy
    if torch.cuda.is_available():
        mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    elif (
        hasattr(torch, "cpu")
        and hasattr(torch.cpu, "is_bf16_supported")
        and torch.cpu.is_bf16_supported()
    ):
        mixed_precision = "bf16"
    else:
        # Avoid fp16 on CPU as it is often not native and slower than fp32
        mixed_precision = "no"

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        log_with="wandb" if args.wandb else None,
    )
    logging.info("Using %s mixed precision.", mixed_precision)
    if accelerator.is_main_process and args.wandb:
        accelerator.init_trackers(
            project_name="transformer-project", config=config.model_dump()
        )

    tokenizer, train_data, val_data = load_and_prepare_data(
        config.evolution.data_dir,
        model_dir,
        config.evolution.validation_split,
    )

    if not resume_from:
        save_updated_config(config, tokenizer, model_dir)
        tokenizer_vocab_path = os.path.join(
            config.evolution.data_dir, "tokenizer_vocab.json"
        )
        if os.path.exists(tokenizer_vocab_path):
            shutil.copy(tokenizer_vocab_path, model_dir)

    return TrainingEnvironment(
        config=config,
        accelerator=accelerator,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        checkpoint_dir=checkpoint_dir,
        model_dir=model_dir,
    )
