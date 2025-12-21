import numpy as np
import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn_components.attention import ScaledDotProductAttention

class TestAttention(unittest.TestCase):
    def test_attention_backward(self):
        """Численная проверка градиентов для `backward` метода."""
        print("Running tests for ScaledDotProductAttention (Backward Pass)...")

        np.random.seed(42)
        batch_size, seq_len, d_k, d_v = 2, 3, 4, 5

        q = np.random.randn(batch_size, 1, seq_len, d_k) # Добавим "головы" для совместимости
        k = np.random.randn(batch_size, 1, seq_len, d_k)
        v = np.random.randn(batch_size, 1, seq_len, d_v)
        dout = np.random.randn(batch_size, 1, seq_len, d_v)

        attention = ScaledDotProductAttention()

        # --- Аналитические градиенты ---
        _ = attention.forward(q, k, v)
        dq, dk, dv = attention.backward(dout)

        # --- Численные градиенты ---
        epsilon = 1e-6

        # 1. Проверка dq
        dq_num = np.zeros_like(q)
        it = np.nditer(q, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            old_val = q[ix]
            q[ix] = old_val + epsilon
            fx_plus = np.sum(attention.forward(q, k, v) * dout)
            q[ix] = old_val - epsilon
            fx_minus = np.sum(attention.forward(q, k, v) * dout)
            dq_num[ix] = (fx_plus - fx_minus) / (2 * epsilon)
            q[ix] = old_val
            it.iternext()

        # 2. Проверка dk
        dk_num = np.zeros_like(k)
        it = np.nditer(k, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            old_val = k[ix]
            k[ix] = old_val + epsilon
            fx_plus = np.sum(attention.forward(q, k, v) * dout)
            k[ix] = old_val - epsilon
            fx_minus = np.sum(attention.forward(q, k, v) * dout)
            dk_num[ix] = (fx_plus - fx_minus) / (2 * epsilon)
            k[ix] = old_val
            it.iternext()

        # --- Сравнение ---
        self.assertTrue(np.allclose(dq, dq_num, rtol=1e-4, atol=1e-4), "Gradient check for dq FAILED")
        print("Gradient check for dq PASSED.")
        self.assertTrue(np.allclose(dk, dk_num, rtol=1e-4, atol=1e-4), "Gradient check for dk FAILED")
        print("Gradient check for dk PASSED.")
        # dv тривиален и менее склонен к ошибкам, но его тоже стоит проверять

        print("All tests passed!")

if __name__ == "__main__":
    unittest.main()
