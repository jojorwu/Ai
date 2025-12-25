"""
Tests for the optimized FeedForward (SwiGLU) layer.
"""
import logging
import unittest

from backend import np

from nn_components.feed_forward import FeedForward
from tests.gradient_check import check_gradient, numerical_gradient


class TestFeedForward(unittest.TestCase):
    """
    Tests for the optimized FeedForward (SwiGLU) layer with a fused projection.
    """

    def test_swiglu_feed_forward_backward_gradient_check(self):
        """
        Numerically checks the gradients for the optimized FeedForward backward method.
        """
        logging.info("\nRunning Test: Gradient check for optimized SwiGLU FFN...")
        batch_size, seq_len, d_model, d_ff = 2, 5, 16, 32

        np.random.seed(1337)

        ffn = FeedForward(d_model, d_ff, bias=False, num_layers=1)
        x_input = np.random.randn(batch_size, seq_len, d_model)
        dout = np.random.randn(batch_size, seq_len, d_model)

        # --- Forward and Backward Pass ---
        _ = ffn.forward(x_input)
        dx_analytic = ffn.backward(dout)

        # --- Numerical Gradient Check ---
        # Check gradients with respect to the input 'x'
        logging.info("Checking gradients for input: dx...")
        dx_numerical = numerical_gradient(ffn.forward, x_input, dout)
        check_gradient(self, dx_analytic, dx_numerical, "dx")

        # Check gradients for all trainable parameters
        all_params = ffn.get_trainable_params()
        for param_name, (param_val, param_grad) in all_params.items():
            logging.info("Checking gradients for parameter: %s...", param_name)
            # Use a lambda that captures the current parameter being tested
            grad_numerical = numerical_gradient(lambda p: ffn.forward(x_input), param_val, dout)
            check_gradient(self, param_grad, grad_numerical, f"d{param_name}")

        logging.info("Optimized FeedForward gradient checks passed!")


if __name__ == "__main__":
    unittest.main()
