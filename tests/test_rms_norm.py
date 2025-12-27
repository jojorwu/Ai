"""
Tests for the RMSNorm layer.
"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import unittest

import numpy as np

from gradient_check import check_gradient, numerical_gradient
from nn_components.rms_norm import RMSNorm


class TestRMSNorm(unittest.TestCase):
    """
    Tests for the RMSNorm layer.
    """

    def test_rms_norm_backward_gradient_check(self):
        """Numerically checks the gradients for the `backward` method of RMSNorm."""
        logging.info("\nRunning Test: Gradient check for RMSNorm backward pass...")

        batch_size, seq_len, d_model = 2, 5, 16

        np.random.seed(42)
        norm = RMSNorm(d_model)
        norm.gamma = np.random.randn(d_model)

        x = np.random.randn(batch_size, seq_len, d_model)
        dout = np.random.randn(batch_size, seq_len, d_model)

        _ = norm.forward(x)
        dx = norm.backward(dout)
        dgamma = norm.dgamma

        def forward_gamma(_):
            return norm.forward(x)

        dgamma_num = numerical_gradient(forward_gamma, norm.gamma, dout)
        check_gradient(self, dgamma, dgamma_num, "dgamma")

        dx_num = numerical_gradient(norm.forward, x, dout)
        check_gradient(self, dx, dx_num, "dx")

        logging.info("All RMSNorm gradient checks passed!")


if __name__ == "__main__":
    unittest.main()
