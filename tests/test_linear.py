"""
Tests for the Linear layer.
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_components.linear import Linear

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

        # --- Аналитические градиенты ---
        _ = layer.forward(x)
        dx = layer.backward(dout)
        dW = layer.dW
        db = layer.db

        epsilon = 1e-6

        # --- Численная проверка dx ---
        dx_num = self._numerical_gradient(layer, x, x, dout, epsilon)
        self.assertTrue(np.allclose(dx, dx_num, rtol=1e-4, atol=1e-4),
                        "Gradient check for dx FAILED")
        print("Gradient check for dx PASSED.")

        # --- Численная проверка dW ---
        dW_num = self._numerical_gradient(layer, layer.W, x, dout, epsilon)
        self.assertTrue(np.allclose(dW, dW_num, rtol=1e-4, atol=1e-4),
                        "Gradient check for dW FAILED")
        print("Gradient check for dW PASSED.")

        # --- Численная проверка db ---
        db_num = self._numerical_gradient(layer, layer.b, x, dout, epsilon)
        self.assertTrue(np.allclose(db, db_num, rtol=1e-4, atol=1e-4),
                        "Gradient check for db FAILED")
        print("Gradient check for db PASSED.")

        print("All Linear gradient checks passed!")

    def _numerical_gradient(self, model, param, x, dout, epsilon):
        """Helper for numerical gradient checking."""
        grad_numerical = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            original_value = param[ix]

            param[ix] = original_value + epsilon
            fx_plus_h = np.sum(model.forward(x) * dout)

            param[ix] = original_value - epsilon
            fx_minus_h = np.sum(model.forward(x) * dout)

            grad_numerical[ix] = (fx_plus_h - fx_minus_h) / (2 * epsilon)

            param[ix] = original_value
            it.iternext()
        return grad_numerical

if __name__ == "__main__":
    unittest.main()
