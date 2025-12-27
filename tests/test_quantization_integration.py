"""
Integration test to verify the end-to-end training process of a 4-bit quantized model.
"""
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os

from accelerate import Accelerator, init_empty_weights, dispatch_model
from bitsandbytes.optim import Adam8bit

from config import Config, TransformerConfig
from model import Transformer

class TestQuantizationIntegration(unittest.TestCase):
    """
    Tests that a 4-bit quantized model can complete a training step and update its weights.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available, skipping 4-bit test")
    def test_4bit_model_training_step(self):
        """
        Verifies that a 4-bit model's weights are updated after one training step.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

        torch.cuda.empty_cache()

        # --- 1. Accelerator ---
        accelerator = Accelerator()

        # --- 2. Configuration ---
        config = Config.from_json('config.json')
        vocab_size = 100 # Dummy vocab size

        transformer_config = TransformerConfig(
            vocab_size=vocab_size,
            model=config.model,
            vision=config.vision,
            ltm=config.ltm
        )

        # --- 3. Model Initialization (in 4-bit) ---
        # Use accelerate's context manager to initialize the model on the meta device
        with init_empty_weights():
            model = Transformer(transformer_config, load_in_4bit=True)

        # Manually set device map to force model onto GPU, as 4-bit is CUDA-only
        device_map = {"": 0}
        logging.info(f"Using manual device map: {device_map}")
        model = dispatch_model(model, device_map=device_map)

        # --- 4. Optimizer and Loss Function ---
        optimizer = Adam8bit(model.parameters(), lr=config.optimizer.learning_rate)
        policy_loss_fn = nn.CrossEntropyLoss()
        value_loss_fn = nn.MSELoss()

        # Prepare everything except the model, which is already dispatched
        optimizer, policy_loss_fn, value_loss_fn = accelerator.prepare(
            optimizer, policy_loss_fn, value_loss_fn
        )

        # --- 5. Dummy Data ---
        batch_size = 4
        seq_len = config.model.max_seq_len
        # Create data on CPU, accelerator.prepare will move it to the correct device
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device='cpu')
        dummy_policy_target = torch.randint(0, vocab_size, (batch_size, seq_len), device='cpu')
        dummy_value_target = torch.randn(batch_size, 1, device='cpu')

        dummy_input, dummy_policy_target, dummy_value_target = accelerator.prepare(
            dummy_input, dummy_policy_target, dummy_value_target
        )

        # --- 6. Training Step ---
        # Get a reference to a weight tensor before the update.
        # The model is not wrapped by accelerator, so we don't use .module
        initial_weights = model.decoder_blocks[0].mha.wo.weight.clone().detach()

        model.train()
        optimizer.zero_grad()

        logits, value, aux_loss = model(dummy_input)

        loss_policy = policy_loss_fn(logits.view(-1, vocab_size), dummy_policy_target.view(-1))
        loss_value = value_loss_fn(value, dummy_value_target)

        total_loss = loss_policy + loss_value
        if aux_loss is not None:
            total_loss += aux_loss

        accelerator.backward(total_loss)
        optimizer.step()

        # --- 7. Verification ---
        updated_weights = model.decoder_blocks[0].mha.wo.weight.clone().detach()

        weights_updated = not torch.equal(initial_weights, updated_weights)

        self.assertTrue(weights_updated, "Model weights were NOT updated after a training step.")
        logging.info("Verification PASSED: 4-bit model weights were updated successfully.")

if __name__ == "__main__":
    unittest.main()
