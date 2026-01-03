"""
Command-line interface (CLI) utilities for the project.
"""
import argparse

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
        '--torch_compile',
        action='store_true',
        help="Enable torch.compile for the model for faster execution."
    )
    return parser
