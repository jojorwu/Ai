"""
Utilities for setting up the application environment from command-line arguments.
"""
import argparse
import os
from dataclasses import dataclass
from typing import Type, Union

from src.config.core import GenerateConfig, TrainConfig
from src.utils.cli import (
    apply_cli_args_to_config,
    create_main_parser,
    select_model_interactively,
)


@dataclass
class AppSetup:
    """Container for the application setup components."""

    args: argparse.Namespace
    config: Union[TrainConfig, GenerateConfig]
    model_name: str


def setup_from_args(
    config_class: Union[Type[TrainConfig], Type[GenerateConfig]],
    add_extra_args=None,
) -> AppSetup:
    """
    Handles the common setup logic for scripts.
    - Creates and configures the argument parser.
    - Prompts for model selection if not provided.
    - Loads the appropriate configuration file.
    - Applies CLI arguments to override the config.
    """
    parser = create_main_parser()
    if add_extra_args:
        add_extra_args(parser)
    args = parser.parse_args()

    model_name = args.model_name or select_model_interactively()
    if not model_name:
        raise SystemExit("No model selected. Exiting.")

    model_dir = os.path.join("models", model_name)
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found for model '{model_name}' at {config_path}"
        )

    config = config_class.from_json(config_path)
    apply_cli_args_to_config(args, config)

    return AppSetup(args=args, config=config, model_name=model_name)
