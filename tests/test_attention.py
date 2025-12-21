"""
Tests for the ScaledDotProductAttention layer.
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn_components.attention import ScaledDotProductAttention

class TestAttention(unittest.TestCase):
    """
    Tests for the ScaledDotProductAttention layer.
    """
    def test_attention_backward(self):
        """Численная проверка градиентов для `backward` метода."""
        print("Running tests for ScaledDotProductAttention (Backward Pass)...")

        np.random.seed(42)
        batch_size, seq_len, d_k, d_v = 2, 3, 4, 5

        q = np.random.randn(batch_size, 1, seq_len, d_k)
        k = np.random.randn(batch_size, 1, seq_len, d_k)
        v = np.random.randn(batch_size, 1, seq_len, d_v)
        dout = np.random.randn(batch_size, 1, seq_len, d_v)

        attention = ScaledDotProductAttention()

        _ = attention.forward(q, k, v)
        dq, dk, _ = attention.backward(dout)

        epsilon = 1e-6

        dq_num = self._numerical_gradient(attention, q, q, k, v, dout, epsilon)
        dk_num = self._numerical_gradient(attention, k, q, k, v, dout, epsilon)

        self.assertTrue(np.allclose(dq, dq_num, rtol=1e-4, atol=1e-4),
                        "Gradient check for dq FAILED")
        print("Gradient check for dq PASSED.")
        self.assertTrue(np.allclose(dk, dk_num, rtol=1e-4, atol=1e-4),
                        "Gradient check for dk FAILED")
        print("Gradient check for dk PASSED.")

        print("All tests passed!")

    def _numerical_gradient(self, model, param, q, k, v, dout, epsilon):
        """Helper for numerical gradient checking."""
        grad_numerical = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            original_value = param[ix]

            param[ix] = original_value + epsilon
            fx_plus_h = np.sum(model.forward(q, k, v) * dout)

            param[ix] = original_value - epsilon
            fx_minus_h = np.sum(model.forward(q, k, v) * dout)

            grad_numerical[ix] = (fx_plus_h - fx_minus_h) / (2 * epsilon)

            param[ix] = original_value
            it.iternext()
        return grad_numerical

if __name__ == "__main__":
    unittest.main()
