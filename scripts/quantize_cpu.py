#!/usr/bin/env python3
"""
This script loads a trained model, applies dynamic quantization for CPU,
and saves the quantized model.
"""
import os
import torch
import torch.quantization

from src.model import Transformer
from src.config import Config, TransformerConfig
from src.tokenizer import Tokenizer


def quantize_model(model_path: str, config_path: str, output_path: str):
    """
    Loads a model, applies dynamic quantization, and saves the quantized model.

    Args:
        model_path: Path to the trained model's state_dict.
        config_path: Path to the model's configuration JSON file.
        output_path: Path to save the quantized model.
    """
    # Load the main configuration
    config = Config.from_json(config_path)

    # Instantiate tokenizer to get vocab size
    model_dir = os.path.dirname(config_path)
    tokenizer = Tokenizer(model_dir)

    # Create the TransformerConfig
    transformer_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        model=config.model,
        vision=config.vision,
        ltm=config.ltm,
        tokenizer=tokenizer,
    )

    # Instantiate the model
    model = Transformer(transformer_config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # Set the model to evaluation mode
    model.eval()

    # Apply dynamic quantization for CPU
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Save the quantized model
    torch.save(quantized_model.state_dict(), output_path)
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
