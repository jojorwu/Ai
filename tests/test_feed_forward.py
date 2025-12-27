"""
Tests for the optimized FeedForward (SwiGLU) layer.
"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import unittest

from gradient_check import check_gradient, numerical_gradient
from backend import np
from nn_components.feed_forward import FeedForward


class TestFeedForward(unittest.TestCase):
    """
    Tests for the optimized FeedForward (SwiGLU) layer with a fused projection.
    """

    def _setup_test(self):
        """Sets up the test data and FeedForward layer."""
        batch_size, seq_len, d_model, d_ff = 2, 5, 16, 32
        np.random.seed(1337)
        ffn = FeedForward(d_model, d_ff, bias=False, num_layers=1)
        x_input = np.random.randn(batch_size, seq_len, d_model)
        dout = np.random.randn(batch_size, seq_len, d_model)
        return ffn, x_input, dout

    def _check_gradients(self, ffn, x_input, dout):
        """Checks the gradients for the FeedForward layer."""
        _ = ffn.forward(x_input)
        dx_analytic = ffn.backward(dout)

        def model_forward_x(x):
            return ffn.forward(x)

        dx_numerical = numerical_gradient(model_forward_x, x_input, dout)
        check_gradient(self, dx_analytic, dx_numerical, "dx")

        all_params = ffn.get_trainable_params()
        for param_name, (param_val, param_grad) in all_params.items():
            def param_forward(_):
                return ffn.forward(x_input)

            grad_numerical = numerical_gradient(param_forward, param_val, dout)
            check_gradient(self, param_grad, grad_numerical, f"d{param_name}")

    def test_swiglu_feed_forward_backward_gradient_check(self):
        """
        Numerically checks the gradients for the optimized FeedForward backward method.
        """
        logging.info("\nRunning Test: Gradient check for optimized SwiGLU FFN...")
        ffn, x_input, dout = self._setup_test()
        self._check_gradients(ffn, x_input, dout)
        logging.info("Optimized FeedForward gradient checks passed!")


if __name__ == "__main__":
    unittest.main()
