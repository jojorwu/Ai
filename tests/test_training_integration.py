"""
Integration test for the PyTorch-based training pipeline.
"""
import sys
import os
import unittest
import torch
import torch.nn as nn
import torch.optim as optim

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config, TransformerConfig
from model import Transformer


def _create_test_config():
    """Creates a minimal configuration for integration testing."""
    config = Config.from_json('config.json')
    config.model.d_model = 8
    config.model.num_layers = 1
    config.model.num_heads = 2
    config.model.num_kv_heads = 2
    config.model.d_ff = 16
    config.model.max_seq_len = 16
    config.model.ltm_d_hidden = 4
    config.model.ltm_num_layers = 1
    return config


class TestTrainingIntegration(unittest.TestCase):
    """
    Tests that a single training step updates the model's weights.
    """

    def test_single_training_step_updates_weights(self):
        """
        Tests that a single training step correctly updates the model's weights.
        """
        vocab_size = 10
        config = _create_test_config()
        transformer_config = TransformerConfig(
            vocab_size=vocab_size,
            model=config.model,
            vision=config.vision,
            ltm=config.ltm
        )
        model = Transformer(transformer_config)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        # Dummy data
        x = torch.randint(0, vocab_size, (2, 8)) # Batch, SeqLen
        y = torch.randint(0, vocab_size, (2, 16))

        # Store initial weights of one layer
        initial_weights = model.decoder_blocks[0].mha.wo.weights.clone().detach()

        # Training step
        model.train()
        optimizer.zero_grad()
        logits, _, _ = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        # Check for weight updates
        updated_weights = model.decoder_blocks[0].mha.wo.weights.clone().detach()
        self.assertFalse(torch.equal(initial_weights, updated_weights),
                         "Model weights were not updated after a training step.")


if __name__ == "__main__":
    unittest.main()
