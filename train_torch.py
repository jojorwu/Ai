"""
A minimal training script to test the end-to-end migration of the Transformer model to PyTorch.
This script is for verification purposes and will replace the original train.py later.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import logging

from config import Config, TransformerConfig
from model import Transformer

def main():
    """
    Runs a single, minimal training step to verify the PyTorch model integration.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # --- 1. Configuration ---
    logging.info("Loading configuration...")
    # Using the base config and adapting it for a minimal test
    config = Config.from_json('config.json')
    vocab_size = 100 # Dummy vocab size

    transformer_config = TransformerConfig(
        vocab_size=vocab_size,
        model=config.model,
        vision=config.vision,
        ltm=config.ltm
        # tokenizer is not needed for this minimal test
    )

    # --- 2. Model Initialization ---
    logging.info("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(transformer_config).to(device)
    logging.info(f"Model moved to {device}.")

    # --- 3. Optimizer and Loss Function ---
    optimizer = optim.Adam(model.parameters(), lr=config.optimizer.learning_rate)
    # Policy head loss
    policy_loss_fn = nn.CrossEntropyLoss()
    # Value head loss (e.g., MSE)
    value_loss_fn = nn.MSELoss()

    # --- 4. Dummy Data ---
    batch_size = 4
    seq_len = config.model.max_seq_len
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    dummy_policy_target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    dummy_value_target = torch.randn(batch_size, 1, device=device)

    # --- 5. Training Step ---
    logging.info("Performing a single training step...")

    # Store initial weights of one layer to check for updates
    initial_weights = model.decoder_blocks[0].mha.wo.weights.clone().detach()

    model.train()
    optimizer.zero_grad()

    # Forward pass
    logits, value, aux_loss = model(dummy_input)

    # Calculate losses
    # Reshape for CrossEntropyLoss: (Batch * SeqLen, VocabSize)
    loss_policy = policy_loss_fn(logits.view(-1, vocab_size), dummy_policy_target.view(-1))
    loss_value = value_loss_fn(value, dummy_value_target)

    total_loss = loss_policy + loss_value + aux_loss

    # Backward pass and optimization
    total_loss.backward()
    optimizer.step()

    logging.info(f"Initial Loss: {total_loss.item():.4f}")

    # --- 6. Verification ---
    logging.info("Verifying weight update...")
    updated_weights = model.decoder_blocks[0].mha.wo.weights.clone().detach()

    weights_updated = not torch.equal(initial_weights, updated_weights)

    if weights_updated:
        logging.info("Verification PASSED: Model weights were updated successfully.")
    else:
        logging.error("Verification FAILED: Model weights were NOT updated.")

if __name__ == "__main__":
    main()
