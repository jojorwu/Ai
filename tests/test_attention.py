"""
Tests for the ScaledDotProductAttention layer.
"""
import logging
import unittest

import numpy as np

from nn_components.attention import ScaledDotProductAttention
from tests.gradient_check import check_gradient, numerical_gradient


class TestAttention(unittest.TestCase):
    """
    Tests for the ScaledDotProductAttention layer.
    """

    def test_attention_backward(self):
        """Numerically checks the gradients for the `backward` method."""
        logging.info("\nRunning Test: Gradient check for ScaledDotProductAttention backward pass...")

        np.random.seed(42)
        batch_size, num_heads, seq_len, d_k, d_v = 2, 8, 3, 4, 5

        q = np.random.randn(batch_size, num_heads, seq_len, d_k)
        k = np.random.randn(batch_size, num_heads, seq_len, d_k)
        v = np.random.randn(batch_size, num_heads, seq_len, d_v)
        dout = np.random.randn(batch_size, num_heads, seq_len, d_v)

        attention = ScaledDotProductAttention()

        _ = attention.forward(q, k, v)
        dq, dk, dv = attention.backward(dout)

        forward_fn = lambda: attention.forward(q, k, v)

        dq_num = numerical_gradient(forward_fn, q, dout)
        check_gradient(self, dq, dq_num, "dQ")

        dk_num = numerical_gradient(forward_fn, k, dout)
        check_gradient(self, dk, dk_num, "dK")

        dv_num = numerical_gradient(forward_fn, v, dout)
        check_gradient(self, dv, dv_num, "dV")

        logging.info("All ScaledDotProductAttention gradient checks passed!")


if __name__ == "__main__":
    unittest.main()
