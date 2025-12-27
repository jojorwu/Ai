"""
A minimal generation script to test the PyTorch-based Transformer model.
"""
import torch
import logging
import argparse

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
    args = parser.parse_args()

    # --- 1. Configuration ---
    logging.info("Loading configuration...")
    config = Config.from_json('config.json')

    # We need a tokenizer to encode the prompt
    # In a real scenario, this would be loaded from the model's directory
    try:
        tokenizer = Tokenizer("data") # Assumes a vocab can be built from 'data' dir
    except Exception as e:
        logging.error(f"Could not initialize tokenizer. Make sure there is a 'data' directory with text files or a vocab file. Error: {e}")
        return

    transformer_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        model=config.model,
        vision=config.vision,
        ltm=config.ltm,
        tokenizer=tokenizer
    )

    # --- 2. Model Initialization ---
    logging.info("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(transformer_config).to(device)
    model.eval() # Set model to evaluation mode
    logging.info(f"Model moved to {device}.")

    # --- 3. Generation ---
    logging.info(f"Starting generation with prompt: '{args.prompt}'")

    # Encode the prompt
    start_tokens = tokenizer.encode(args.prompt)
    start_tokens_tensor = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)

    # Create generation input
    gen_input = GenerateInput(
        start_tokens=start_tokens_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=0.8,
        top_k=20
    )

    # Generate!
    generated_tokens = model.generate(gen_input)

    # --- 4. Decode and Print ---
    generated_text = tokenizer.decode(generated_tokens.squeeze(0).tolist())

    logging.info("--- Generated Text ---")
    print(generated_text)


if __name__ == "__main__":
    main()
