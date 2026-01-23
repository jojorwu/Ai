"""
Command-line interface (CLI) utilities for the project.
"""
import argparse
import logging
import os
from typing import Union

from src.config.core import GenerateConfig, TrainConfig


def select_model_interactively() -> str | None:
    """
    Lists available models in the 'models' directory and prompts the user to select one.
    Returns the selected model name or None if no selection is made.
    """
    models_dir = 'models'
    if not os.path.isdir(models_dir) or not os.listdir(models_dir):
        logging.error("No models found in the '%s' directory.", models_dir)
        return None

    available_models = [
        d for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d))
    ]

    if not available_models:
        logging.error("No valid model directories found in '%s'.", models_dir)
        return None
    if len(available_models) == 1:
        logging.info("Automatically selecting the only available model: %s",
                     available_models[0])
        return available_models[0]

    logging.info("Available models:")
    for i, model_name in enumerate(available_models):
        logging.info("  %d: %s", i + 1, model_name)

    while True:
        try:
            choice = int(input("Please select a model by number: "))
            if 1 <= choice <= len(available_models):
                return available_models[choice - 1]
            logging.warning("Invalid number. Please try again.")
        except ValueError:
            logging.warning("Invalid input. Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            logging.info("\nSelection cancelled.")
            return None


def apply_cli_args_to_config(
    args: argparse.Namespace, config: Union[TrainConfig, GenerateConfig]
):
    """
    Overrides configuration fields based on command-line arguments.

    Args:
        args: Parsed arguments from argparse.
        config: The configuration object to modify.
    """
    if args.hardware_strategy:
        config.hardware.strategy = args.hardware_strategy
        logging.info(
            "Overriding hardware strategy with '%s'", args.hardware_strategy
        )
    if args.torch_compile:
        config.hardware.torch_compile = True
        logging.info("Enabling torch.compile.")


def make_model_name_required(parser: argparse.ArgumentParser):
    """
    Makes the '--model-name' argument required in the provided parser.

    Args:
        parser (argparse.ArgumentParser): The parser to modify.
    """
    # Find the action for --model-name and set 'required' to True
    for action in parser._actions:  # pylint: disable=protected-access
        if action.dest == 'model_name':
            action.required = True
            return
    # This should not happen if the main parser is used correctly
    raise ValueError("--model-name argument not found in the parser.")


def create_main_parser():
    """
    Creates an ArgumentParser with common arguments for training and generation.

    Returns:
        argparse.ArgumentParser: An ArgumentParser with pre-populated arguments.
    """
    parser = argparse.ArgumentParser(
        description="Transformer Model Training and Generation"
    )
    parser.add_argument(
        '--model-name',
        type=str,
        help="The name of the model to use or create."
    )
    parser.add_argument(
        '--load-in-4bit',
        action='store_true',
        help="Load the model in 4-bit quantization."
    )
    parser.add_argument(
        '--hardware-strategy',
        type=str,
        choices=['unified', 'discrete', 'hybrid'],
        help="Override the hardware strategy from the config."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config_train.json',
        help="Path to the configuration file."
    )
    parser.add_argument(
        '--quantized',
        action='store_true',
        help="Load a quantized version of the model for CPU."
    )
    parser.add_argument(
        '--torch_compile',
        action='store_true',
        help="Enable torch.compile for the model (GPU only)."
    )
    return parser
