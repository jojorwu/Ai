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
from nn_components.loss import SoftmaxCrossEntropy, MarginRankingLoss

class TestTrainingIntegration(unittest.TestCase):
    """
    Tests that a single training step updates the model's weights.
    This is a smoke test for the entire training pipeline.
    """
    def test_single_training_step(self):
        """
        Tests that a single training step updates the model's weights.
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
        value_loss_fn = MarginRankingLoss(margin=1.0)
        optimizer = Adam(model.get_named_params(), learning_rate=0.001)

        # --- Get initial weights ---
        initial_weights = np.copy(model.decoder_blocks[0].ffn.w1.W)

        # --- Perform a single training step ---
        model.train()
        model.zero_grad()

        # Get grads for a "good" candidate (just the standard policy pass)
        logits, good_value = model.forward(x, mask)
        _ = policy_loss_fn.forward(logits, y)
        dlogits = policy_loss_fn.backward()
        model.backward(dlogits, np.zeros_like(good_value))
        good_grads = model.get_gradients()

        # Get grads for a "bad" candidate
        model.zero_grad()
        logits, bad_value = model.forward(x, mask)
        _ = policy_loss_fn.forward(logits, y)
        dlogits_bad = policy_loss_fn.backward()
        model.backward(dlogits_bad, np.zeros_like(bad_value))

        # Value loss
        _ = value_loss_fn.forward(good_value, bad_value)
        d_good, d_bad = value_loss_fn.backward()

        # Accumulate gradients manually
        final_grads = good_grads

        model.zero_grad()
        model.eval()
        _, _ = model.forward(x, mask)
        model.backward(np.zeros_like(logits), d_good)
        good_value_grads = model.get_gradients()
        for key in final_grads:
            final_grads[key] += good_value_grads[key]

        _, _ = model.forward(x, mask)
        model.backward(np.zeros_like(logits), d_bad)
        bad_value_grads = model.get_gradients()
        for key in final_grads:
            final_grads[key] += bad_value_grads[key]

        model.set_gradients(final_grads)

        optimizer.step()

        # --- Check if weights have been updated ---
        updated_weights = model.decoder_blocks[0].ffn.w1.W
        self.assertFalse(np.allclose(initial_weights, updated_weights), "Weights were not updated after a training step.")
        print("Training Integration test PASSED.")

if __name__ == "__main__":
    unittest.main()
