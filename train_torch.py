"""
A minimal training script to test the end-to-end migration of the Transformer model to PyTorch.
This script is for verification purposes and will replace the original train.py later.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse

from accelerate import Accelerator
from bitsandbytes.optim import Adam8bit

from config import Config, TransformerConfig
from model import Transformer

def main():
    """
    Runs a single, minimal training step to verify the PyTorch model integration.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(description="PyTorch Transformer Training Verification")
    parser.add_argument('--load-in-4bit', action='store_true', help="Load the model in 4-bit.")
    args = parser.parse_args()

    # --- 1. Accelerator ---
    accelerator = Accelerator()
    device = accelerator.device

    # --- 2. Configuration ---
    logging.info("Loading configuration...")
    config = Config.from_json('config.json')
    vocab_size = 100 # Dummy vocab size

    transformer_config = TransformerConfig(
        vocab_size=vocab_size,
        model=config.model,
        vision=config.vision,
        ltm=config.ltm
    )

    # --- 3. Model Initialization ---
    logging.info(f"Initializing model (4-bit: {args.load_in_4bit})...")
    model = Transformer(transformer_config, load_in_4bit=args.load_in_4bit)
    model = accelerator.prepare(model)
    logging.info(f"Model moved to {device}.")

    # --- 4. Optimizer and Loss Function ---
    # Use Adam8bit for 4-bit models
    optimizer_class = Adam8bit if args.load_in_4bit else optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=config.optimizer.learning_rate)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    optimizer, policy_loss_fn, value_loss_fn = accelerator.prepare(optimizer, policy_loss_fn, value_loss_fn)

    # --- 5. Dummy Data ---
    batch_size = 4
    seq_len = config.model.max_seq_len
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    dummy_policy_target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    dummy_value_target = torch.randn(batch_size, 1, device=device)
    dummy_input, dummy_policy_target, dummy_value_target = accelerator.prepare(dummy_input, dummy_policy_target, dummy_value_target)


    # --- 6. Training Step ---
    logging.info("Performing a single training step...")

    initial_weights = model.module.decoder_blocks[0].mha.wo.weights.clone().detach()

    model.train()
    optimizer.zero_grad()

    logits, value, aux_loss = model(dummy_input)

    loss_policy = policy_loss_fn(logits.view(-1, vocab_size), dummy_policy_target.view(-1))
    loss_value = value_loss_fn(value, dummy_value_target)

    total_loss = loss_policy + loss_value + aux_loss

    accelerator.backward(total_loss)
    optimizer.step()

    logging.info(f"Initial Loss: {total_loss.item():.4f}")

    # --- 7. Verification ---
    logging.info("Verifying weight update...")
    updated_weights = model.module.decoder_blocks[0].mha.wo.weights.clone().detach()

    weights_updated = not torch.equal(initial_weights, updated_weights)

    if weights_updated:
        logging.info("Verification PASSED: Model weights were updated successfully.")
    else:
        logging.error("Verification FAILED: Model weights were NOT updated.")

if __name__ == "__main__":
    main()
