#!/usr/bin/env python3
"""
This script loads a trained model, applies dynamic quantization for CPU,
and saves the quantized model.
"""
import logging
import os
import torch
import torch.quantization

from src.config.core import TrainConfig
from src.model.factory import load_model_and_tokenizer
from src.utils.cli import create_main_parser, make_model_name_required
from src.utils.core import main_entrypoint, setup_logging


def quantize_model(model_name: str, config: TrainConfig):
    """
    Loads a model, applies dynamic quantization, and saves the quantized model.

    Args:
        model_name (str): The name of the model to quantize.
        config (TrainConfig): The training configuration.
    """
    logging.info("Loading model '%s' for quantization...", model_name)
    # Use dispatch=False to ensure the model stays on the CPU
    model, _ = load_model_and_tokenizer(
        model_name, config, load_in_4bit=False, quantized=True, dispatch=False
    )
    logging.info("Model loaded and quantized successfully.")

    model_dir = os.path.join('models', model_name)
    output_path = os.path.join(model_dir, 'model_quantized_cpu.pt')

    torch.save(model.state_dict(), output_path)
    logging.info("Quantized model saved to %s", output_path)


@main_entrypoint
def main():
    """Main function to handle model quantization."""
    setup_logging()
    parser = create_main_parser()
    make_model_name_required(parser)
    args = parser.parse_args()

    model_dir = os.path.join('models', args.model_name)
    config_path = os.path.join(model_dir, 'config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    config = TrainConfig.from_json(config_path)
    quantize_model(args.model_name, config)


if __name__ == "__main__":
    main()
