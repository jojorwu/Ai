"""
This script quantizes a trained model's weights from float32 to int8.
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
from config import Config, TransformerConfig
from model import Transformer
from tokenizer import Tokenizer


def main():
    """Main function to quantize a model."""
    parser = argparse.ArgumentParser(
        description="Quantize a model's weights to int8.")
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the .npz file of the model to quantize.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save the quantized .npz model file.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(args.model_path):
        logging.error("Model file not found: %s", args.model_path)
        sys.exit(1)

    try:
        # Load config from the model file
        with np.load(args.model_path, allow_pickle=True) as data:
            if 'config' not in data:
                logging.error(
                    "Config not found in model file: %s", args.model_path)
                sys.exit(1)
            config_json = data['config'][0]
            loaded_config_dict = json.loads(config_json)
            config = Config(**loaded_config_dict)

        # Load tokenizer
        model_dir = os.path.dirname(args.model_path)
        tokenizer_path = os.path.join(model_dir, 'tokenizer_vocab.json')
        if not os.path.exists(tokenizer_path):
            # Fallback to project root if not found in model dir
            tokenizer_path = 'tokenizer_vocab.json'
            if not os.path.exists(tokenizer_path):
                logging.error("Could not find tokenizer_vocab.json in %s or project root.", model_dir)
                sys.exit(1)

        tokenizer = Tokenizer(tokenizer_path)
        vocab_size = tokenizer.vocab_size

        # Load the model
        logging.info("Loading model from %s...", args.model_path)
        model = Transformer.load_model(
            args.model_path, vocab_size, config, tokenizer)

        # Quantize the model
        logging.info("Quantizing model...")
        model.quantize_model()

        # Save the quantized model
        logging.info("Saving quantized model to %s...", args.output_path)
        model.save_weights(args.output_path, loaded_config_dict)

        logging.info("Model quantization successful!")

    except Exception as e:
        logging.error("An error occurred during quantization: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
