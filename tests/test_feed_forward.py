"""
Tests for the FeedForward (SwiGLU) layer.
"""
import logging
import unittest

import numpy as np

from nn_components.feed_forward import FeedForward
from tests.gradient_check import check_gradient, numerical_gradient


class TestFeedForward(unittest.TestCase):
    """
    Tests for the FeedForward (SwiGLU) layer.
    """

    def test_swiglu_feed_forward_backward_gradient_check(self):
        """
        Numerically checks the gradients for the FeedForward (SwiGLU) backward method.
        """
        logging.info("\nRunning Test: Gradient check for SwiGLU FeedForward backward pass...")
        batch_size, seq_len, d_model, d_ff = 2, 5, 16, 32

        np.random.seed(1337)

        ffn = FeedForward(d_model, d_ff)
        x = np.random.randn(batch_size, seq_len, d_model)
        dout = np.random.randn(batch_size, seq_len, d_model)

        _ = ffn.forward(x)
        dx_analytic = ffn.backward(dout)

        logging.info("Checking gradients for input: dx...")
        dx_numerical = numerical_gradient(lambda x_arg: ffn.forward(x_arg), x, dout)
        check_gradient(self, dx_analytic, dx_numerical, "input dx")

        all_linear_layers = {'w1': ffn.w1, 'w2': ffn.w2, 'w3': ffn.w3}
        for layer_name, layer_obj in all_linear_layers.items():
            params = layer_obj.get_trainable_params()
            for p_name, (p_param, p_grad) in params.items():
                logging.info(f"Checking gradients for parameter: {layer_name}.{p_name}...")
                grad_numerical = numerical_gradient(lambda p_arg: ffn.forward(x), p_param, dout)
                check_gradient(self, p_grad, grad_numerical, f"parameter {layer_name}.{p_name}")

        logging.info("All SwiGLU FeedForward gradient checks passed!")


if __name__ == "__main__":
    unittest.main()
