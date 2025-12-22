"""
Tests for the Linear layer.
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_components.linear import Linear
from tests.gradient_check import check_gradient, numerical_gradient

class TestLinear(unittest.TestCase):
    """
    Tests for the Linear layer.
    """
    def test_linear_backward_gradient_check(self):
        """Численная проверка градиентов для `backward` метода Linear."""
        print("\\nRunning Test: Gradient check for Linear layer backward pass...")

        batch_size, seq_len, input_dim, output_dim = 2, 5, 10, 20

        np.random.seed(42)
        layer = Linear(input_dim, output_dim)
        x = np.random.randn(batch_size, seq_len, input_dim)
        dout = np.random.randn(batch_size, seq_len, output_dim)

        _ = layer.forward(x)
        dx = layer.backward(dout)
        dW = layer.dW
        db = layer.db

        dx_num = numerical_gradient(lambda: layer.forward(x), x, dout)
        check_gradient(self, dx, dx_num, "dx")

        dW_num = numerical_gradient(lambda: layer.forward(x), layer.W, dout)
        check_gradient(self, dW, dW_num, "dW")

        db_num = numerical_gradient(lambda: layer.forward(x), layer.b, dout)
        check_gradient(self, db, db_num, "db")

        print("All Linear gradient checks passed!")

if __name__ == "__main__":
    unittest.main()
