"""
This script quantizes a trained model for CPU inference.
"""
import logging
import os

import torch

from src.config import BaseConfig
from src.utils.cli import create_main_parser
from src.utils.core import load_model_and_tokenizer, main_entrypoint, setup_logging


@main_entrypoint
def main():
    """
    Main function to load a model, apply dynamic quantization, and save the
    quantized model.
    """
    parser = create_main_parser()
    # model-name is required for quantization
    for action in parser._actions:  # pylint: disable=protected-access
        if action.dest == 'model_name':
            action.required = True
            break
    args = parser.parse_args()

    setup_logging()

    # --- Load Config and Model ---
    logging.info("Loading and quantizing model '%s'...", args.model_name)
    model_dir = os.path.join("models", args.model_name)
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    config = BaseConfig.from_json_file(config_path)

    # The utility now handles quantization internally
    quantized_model, _ = load_model_and_tokenizer(
        model_name=args.model_name,
        config=config,
        load_in_4bit=False,
        quantized=True,   # Enable quantization
        dispatch=False,   # Keep on CPU
    )
    logging.info("Model quantization complete.")

    # --- Save Quantized Model ---
    save_path = os.path.join(model_dir, "model_quantized_cpu.pt")
    torch.save(quantized_model.state_dict(), save_path)
    logging.info("Quantized model saved to: %s", save_path)
    logging.info(
        "To run inference, use: python scripts/generate.py --model-name %s --quantized",
        args.model_name
    )

if __name__ == "__main__":
    main()
