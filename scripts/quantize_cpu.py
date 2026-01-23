#!/usr/bin/env python3
"""
This script loads a trained model, applies dynamic quantization for CPU,
and saves the quantized model.
"""
import os
import torch
import torch.quantization

from src.config.core import TrainConfig
from src.utils.core import load_model_and_tokenizer


def quantize_model(model_path: str, config_path: str, output_path: str):
    """
    Loads a model, applies dynamic quantization, and saves the quantized model.

    Args:
        model_path: Path to the trained model's state_dict.
        config_path: Path to the model's configuration JSON file.
        output_path: Path to save the quantized model.
    """
    # Load the main configuration
    config = TrainConfig.from_json(config_path)
    # The load_model_and_tokenizer function already handles the quantization
    # when the 'quantized' flag is set to True.
    model, _ = load_model_and_tokenizer(
        os.path.basename(os.path.dirname(model_path)),
        config,
        load_in_4bit=False,
        quantized=True,
        dispatch=False
    )

    # Save the quantized model state dictionary
    torch.save(model.state_dict(), output_path)
    print(f"Quantized model saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Quantize a trained Transformer model for CPU."
    )
    parser.add_argument(
        "model_path", type=str, help="Path to the trained model's state_dict."
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the model's configuration JSON file."
    )
    parser.add_argument(
        "output_path", type=str, help="Path to save the quantized model."
    )
    args = parser.parse_args()

    quantize_model(args.model_path, args.config_path, args.output_path)
