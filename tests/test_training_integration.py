"""
Integration test for the training pipeline.
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Transformer
from optimizer import Adam
from nn_components.loss import SoftmaxCrossEntropy

class TestTrainingIntegration(unittest.TestCase):
    """
    Tests that a single training step updates the model's weights.
    This is a smoke test for the entire training pipeline.
    """
    def test_single_training_step(self):
        """
        Tests that a single, simple training step updates the model's weights.
        """
        print("\\nRunning Test: Training Integration (single step)...")
        # --- Config ---
        vocab_size = 10
        d_model = 8
        num_layers = 1
        num_heads = 2
        d_ff = 16
        max_seq_len = 5
        batch_size = 2
        seq_len = 4

        # --- Model and Data ---
        model = Transformer(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len)
        x = np.random.randint(0, vocab_size, (batch_size, seq_len))
        y = np.random.randint(0, vocab_size, (batch_size, seq_len))
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)

        # --- Loss and Optimizer ---
        policy_loss_fn = SoftmaxCrossEntropy()
        optimizer = Adam(model.get_named_params(), learning_rate=0.001)

        # --- Get initial weights ---
        initial_weights = np.copy(model.decoder_blocks[0].ffn.w1.W)

        # --- Perform a single training step ---
        model.train()
        model.zero_grad()

        # Forward pass
        logits, value = model.forward(x, mask)

        # Policy loss calculation
        _ = policy_loss_fn.forward(logits, y)
        dlogits = policy_loss_fn.backward()

        # Backward pass for the whole model
        model.backward(dlogits, np.zeros_like(value))

        # Optimizer step
        optimizer.step()

        # --- Check if weights have been updated ---
        updated_weights = model.decoder_blocks[0].ffn.w1.W

        self.assertFalse(np.allclose(initial_weights, updated_weights),
                         "Weights were not updated after a training step.")
        print("Training Integration test PASSED.")

if __name__ == "__main__":
    unittest.main()
