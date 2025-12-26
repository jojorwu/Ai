"""
Tests for the Rotary Positional Embedding.
"""
import logging
import sys
import os
import unittest

import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradient_check import check_gradient, numerical_gradient
from nn_components.rotary_embedding import (apply_rotary_pos_emb,
                                            precompute_rope_embeddings,
                                            rotary_backward)


class TestRotaryEmbedding(unittest.TestCase):
    """
    Tests for the Rotary Positional Embedding.
    """

    def test_rotary_embedding_backward_gradient_check(self):
        """Numerically checks the gradients for the `rotary_backward` function."""
        logging.info(
            "\nRunning Test: Gradient check for Rotary Positional Embedding backward pass...")

        batch_size, n_heads, seq_len, dim = 2, 4, 8, 16

        np.random.seed(42)

        cos, sin = precompute_rope_embeddings(dim, max_seq_len=seq_len)

        x = np.random.randn(batch_size, n_heads, seq_len, dim)
        dout = np.random.randn(batch_size, n_heads, seq_len, dim)

        dx_analytic = rotary_backward(dout, x, cos, sin)

        dx_numerical = numerical_gradient(
            lambda x_arg: apply_rotary_pos_emb(x_arg, cos, sin), x, dout)

        check_gradient(self, dx_analytic, dx_numerical, "dx")
        logging.info("All Rotary Embedding gradient checks passed!")


if __name__ == "__main__":
    unittest.main()
