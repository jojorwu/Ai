"""
Tests for the text generation functionality.
"""

import os
import sys
import unittest
from unittest.mock import patch
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import Transformer
from config import ModelConfig, LTMConfig, LTMOptimizerConfig

class TestGeneration(unittest.TestCase):
    """
    Tests for the text generation functionality.
    """

    def setUp(self):
        # A minimal model configuration for testing
        model_config = ModelConfig(
            d_model=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            d_ff=64,
            max_seq_len=100,
            dropout_rate=0.0,
        )
        self.model = Transformer(vocab_size=50, model_config=model_config)
        print("\\nRunning Test: Generation reject and rollback...")

    @patch('model.Transformer.backward')
    @patch('model.Transformer.forward')
    def test_reject_and_rollback(self, mock_forward, mock_backward):
        """
        Tests that the generation process correctly rejects a low-value chunk
        and then accepts a higher-value one on retry.
        """
        # For this test, we need a model with LTM enabled.
        model_config_ltm = ModelConfig(
            d_model=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            d_ff=64,
            max_seq_len=100,
            dropout_rate=0.0,
            ltm_d_hidden=32,
            ltm_num_layers=2
        )
        ltm_optimizer_config = LTMOptimizerConfig(
            learning_rate=1e-5, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01
        )
        ltm_config = LTMConfig(surprise_threshold=1.0, optimizer=ltm_optimizer_config)
        self.model = Transformer(vocab_size=50, model_config=model_config_ltm, ltm_config=ltm_config)
        # --- Mock Setup ---
        # We need to simulate the forward pass returning different values over time.
        # The first time a speculative chunk is evaluated, its value is low (-0.8).
        # The second time (after rollback and retry), the value is high (0.8).

        # Mock logits (shape: batch, seq_len, vocab_size)
        mock_logits_1 = np.random.randn(1, 1, 50).astype(np.float32)
        mock_logits_2 = np.random.randn(1, 1, 50).astype(np.float32)
        mock_logits_3 = np.random.randn(1, 1, 50).astype(np.float32)

        # Mock values (shape: batch, 1)
        mock_value_low = np.array([[-0.8]], dtype=np.float32)
        mock_value_high = np.array([[0.8]], dtype=np.float32)

        # The mock will be called multiple times. We set up the sequence of return values.
        mock_forward.side_effect = [
            # Initial prompt processing
            (mock_logits_1, None, 0.0),
            # 1st speculative attempt (generates 1 token, gets a low value)
            (mock_logits_2, mock_value_low, 0.0),
            # 2nd speculative attempt (generates 1 token, gets a high value)
            (mock_logits_3, mock_value_high, 0.0)
        ]

        # --- Test Execution ---
        start_tokens = [1, 2]
        generated_tokens = self.model.generate(
            start_tokens=start_tokens,
            max_len=1,
            speculative_steps=1,
            value_threshold=0.0,
            max_retries=3
        )

        # --- Assertions ---
        # The first attempt should be rejected, the second should be accepted.
        # We expect exactly 3 calls to forward(): 1 for prompt, 1 for fail, 1 for success.
        self.assertEqual(mock_forward.call_count, 3, "Forward pass was not called the expected number of times.")

        # The final output should contain exactly one generated token from the successful attempt.
        self.assertEqual(len(generated_tokens), 1, "Generation did not produce the correct number of tokens after rollback.")
        print("Generation reject and rollback test PASSED.")


if __name__ == '__main__':
    unittest.main()
