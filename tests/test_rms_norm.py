"""
Tests for the RMSNorm layer.
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_components.rms_norm import RMSNorm
from tests.gradient_check import check_gradient, numerical_gradient

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

        # --- Численная проверка dgamma ---
        dgamma_num = numerical_gradient(lambda: norm.forward(x), norm.gamma, dout)
        check_gradient(self, dgamma, dgamma_num, "dgamma")

        # --- Численная проверка dx ---
        dx_num = numerical_gradient(lambda: norm.forward(x), x, dout)
        check_gradient(self, dx, dx_num, "dx")

        print("All RMSNorm gradient checks passed!")

if __name__ == "__main__":
    unittest.main()
