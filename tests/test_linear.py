"""
Tests for the Linear layer.
"""
import logging
import unittest

import numpy as np

from nn_components.linear import Linear
from gradient_check import check_gradient, numerical_gradient


class TestLinear(unittest.TestCase):
    """
    Tests for the Linear layer.
    """

    def test_linear_backward_gradient_check(self):
        """Numerically checks the gradients for the `backward` method of Linear."""
        logging.info("\nRunning Test: Gradient check for Linear layer backward pass...")

        batch_size, seq_len, input_dim, output_dim = 2, 5, 10, 20

        np.random.seed(42)
        layer = Linear(input_dim, output_dim)
        x = np.random.randn(batch_size, seq_len, input_dim)
        dout = np.random.randn(batch_size, seq_len, output_dim)

        _ = layer.forward(x)
        dx = layer.backward(dout)
        dw = layer.dweights
        db = layer.dbias

        dx_num = numerical_gradient(lambda x_arg: layer.forward(x_arg), x, dout)
        check_gradient(self, dx, dx_num, "dx")

        dw_num = numerical_gradient(lambda w_arg: layer.forward(x), layer.weights, dout)
        check_gradient(self, dw, dw_num, "dweights", atol=1e-3)

        db_num = numerical_gradient(lambda b_arg: layer.forward(x), layer.bias, dout)
        check_gradient(self, db, db_num, "dbias")

        logging.info("All Linear gradient checks passed!")


if __name__ == "__main__":
    unittest.main()
