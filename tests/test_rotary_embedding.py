"""
Tests for the Rotary Positional Embedding.
"""
import logging
import unittest

import numpy as np

from nn_components.rotary_embedding import (RotaryPositionalEmbedding,
                                            apply_rotary_pos_emb,
                                            rotary_backward)
from tests.gradient_check import check_gradient, numerical_gradient


class TestRotaryEmbedding(unittest.TestCase):
    """
    Tests for the Rotary Positional Embedding.
    """

    def test_rotary_embedding_backward_gradient_check(self):
        """Numerically checks the gradients for the `rotary_backward` function."""
        logging.info("\nRunning Test: Gradient check for Rotary Positional Embedding backward pass...")

        batch_size, n_heads, seq_len, dim = 2, 4, 8, 16

        np.random.seed(42)

        rope = RotaryPositionalEmbedding(dim, max_seq_len=seq_len)
        cos = rope.cos_cached[:, :, :seq_len, :]
        sin = rope.sin_cached[:, :, :seq_len, :]

        x = np.random.randn(batch_size, n_heads, seq_len, dim)
        dout = np.random.randn(batch_size, n_heads, seq_len, dim)

        dx_analytic = rotary_backward(dout, x, cos, sin)

        dx_numerical = numerical_gradient(lambda: apply_rotary_pos_emb(x, cos, sin), x, dout)

        check_gradient(self, dx_analytic, dx_numerical, "dx")
        logging.info("All Rotary Embedding gradient checks passed!")


if __name__ == "__main__":
    unittest.main()
