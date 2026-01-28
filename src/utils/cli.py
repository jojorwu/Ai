"""
Command-line interface (CLI) utilities for the project.
"""
import argparse
import logging
import os


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
        default='config/config_train.json',
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
        help="Enable torch.compile for the model."
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        help="Number of threads for intra-op parallelism (CPU)."
    )
    parser.add_argument(
        '--num-interop-threads',
        type=int,
        help="Number of threads for inter-op parallelism (CPU)."
    )
    parser.add_argument(
        '--disable-mkldnn',
        action='store_true',
        help="Disable oneDNN (MKLDNN) optimizations for CPU."
    )
    parser.add_argument(
        '--flush-denormals',
        action='store_true',
        help="Enable flushing denormal numbers to zero on CPU."
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        help="Number of worker processes for data loading."
    )
    parser.add_argument(
        '--no-pin-memory',
        action='store_false',
        dest='pin_memory',
        help="Disable pinned memory for CPU-to-GPU transfers."
    )
    return parser
