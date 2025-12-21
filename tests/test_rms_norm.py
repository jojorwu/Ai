"""
Tests for the RMSNorm layer.
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_components.rms_norm import RMSNorm

class TestRMSNorm(unittest.TestCase):
    """
    Tests for the RMSNorm layer.
    """
    def test_rms_norm_backward_gradient_check(self):
        """Численная проверка градиентов для `backward` метода RMSNorm."""
        print("\\nRunning Test: Gradient check for RMSNorm backward pass...")

        batch_size, seq_len, d_model = 2, 5, 16

        np.random.seed(42)
        norm = RMSNorm(d_model)
        # Инициализируем gamma случайными значениями для более общей проверки
        norm.gamma = np.random.randn(d_model)

        x = np.random.randn(batch_size, seq_len, d_model)
        dout = np.random.randn(batch_size, seq_len, d_model)

        # --- Аналитические градиенты ---
        _ = norm.forward(x)
        dx = norm.backward(dout)
        dgamma = norm.dgamma

        epsilon = 1e-5

        # --- Численная проверка dgamma ---
        dgamma_num = self._numerical_gradient(norm, norm.gamma, x, dout, epsilon)
        self.assertTrue(np.allclose(dgamma, dgamma_num, rtol=1e-4, atol=1e-4),
                        "Gradient check for dgamma FAILED")
        print("Gradient check for dgamma PASSED.")

        # --- Численная проверка dx ---
        dx_num = self._numerical_gradient(norm, x, x, dout, epsilon)
        self.assertTrue(np.allclose(dx, dx_num, rtol=1e-4, atol=1e-4),
                        "Gradient check for dx FAILED")
        print("Gradient check for dx PASSED.")

        print("All RMSNorm gradient checks passed!")

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
