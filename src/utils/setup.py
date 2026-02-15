"""
Utilities for setting up the application environment from command-line arguments.
"""
import argparse
import logging
import os
from dataclasses import dataclass
from typing import Type, Union

from src.config.core import GenerateConfig, TrainConfig
from src.utils.cli import create_main_parser
from src.utils.interaction import select_model_interactively
from src.utils.logging_config import setup_logging as _setup_logging


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
    # Check for --explain early to avoid required argument errors
    import sys
    if '--explain' in sys.argv:
        config_class.print_explanations(config_class)
        raise SystemExit(0)

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
    config.apply_cli_args(args)

    return AppSetup(args=args, config=config, model_name=model_name)


def setup_logging(log_path: str = None):
    """
    Configures logging to file and console using the centralized utility.
    """
    _setup_logging(log_path=log_path)
