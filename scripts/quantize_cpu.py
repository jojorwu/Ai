"""
This script quantizes a trained model for CPU inference.
"""
import logging
import os

import torch

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

    # --- Load Model ---
    logging.info("Loading model '%s' for quantization...", args.model_name)
    # Load the model without dispatching it to a device yet, ensuring it stays on CPU
    model, _ = load_model_and_tokenizer(
        args.model_name,
        load_in_4bit=False,  # Quantization is a CPU feature, not 4-bit
        quantized=False,     # Load the original, unquantized model
        dispatch=False       # Do not dispatch to accelerator yet
    )
    model.to('cpu')
    model.eval()

    # --- Apply Dynamic Quantization ---
    logging.info("Applying dynamic quantization...")
    # `torch.quantization.quantize_dynamic` is a utility that automatically
    # replaces specified layers (like Linear) with their quantized versions.
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    logging.info("Quantization complete.")

    # --- Save Quantized Model ---
    model_dir = os.path.join("models", args.model_name)
    save_path = os.path.join(model_dir, "model_quantized_cpu.pt")
    torch.save(quantized_model.state_dict(), save_path)
    logging.info("Quantized model saved to: %s", save_path)
    logging.info(
        "To run inference, use: python scripts/generate.py --model-name %s --quantized",
        args.model_name
    )

if __name__ == "__main__":
    main()
