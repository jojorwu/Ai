"""
A minimal generation script to test the PyTorch-based Transformer model.
"""
import torch
import logging
import argparse

from accelerate import Accelerator

from config import Config, TransformerConfig
from model import Transformer, GenerateInput
from tokenizer import Tokenizer

def main():
    """
    Runs a minimal generation loop to verify the PyTorch model's inference.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(description="Generate text with a PyTorch Transformer model.")
    parser.add_argument('--prompt', type=str, default="hello world", help="The initial text prompt.")
    parser.add_argument('--max-new-tokens', type=int, default=50, help="Maximum number of new tokens to generate.")
    parser.add_argument('--load-in-4bit', action='store_true', help="Load the model in 4-bit.")
    args = parser.parse_args()

    # --- 1. Accelerator ---
    accelerator = Accelerator()
    device = accelerator.device

    # --- 2. Configuration ---
    logging.info("Loading configuration...")
    config = Config.from_json('config.json')

    try:
        tokenizer = Tokenizer("data")
    except Exception as e:
        logging.error(f"Could not initialize tokenizer. Error: {e}")
        return

    transformer_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        model=config.model,
        vision=config.vision,
        ltm=config.ltm,
        tokenizer=tokenizer
    )

    # --- 3. Model Initialization ---
    logging.info(f"Initializing model (4-bit: {args.load_in_4bit})...")
    model = Transformer(transformer_config, load_in_4bit=args.load_in_4bit)
    model = accelerator.prepare(model)
    model.eval()
    logging.info(f"Model moved to {device}.")

    # --- 4. Generation ---
    logging.info(f"Starting generation with prompt: '{args.prompt}'")

    start_tokens = tokenizer.encode(args.prompt)
    start_tokens_tensor = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)

    gen_input = GenerateInput(
        start_tokens=start_tokens_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=0.8,
        top_k=20
    )

    generated_tokens = model.module.generate(gen_input)

    # --- 5. Decode and Print ---
    generated_text = tokenizer.decode(generated_tokens.squeeze(0).tolist())

    logging.info("--- Generated Text ---")
    print(generated_text)


if __name__ == "__main__":
    main()
