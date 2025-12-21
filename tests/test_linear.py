import numpy as np
import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_components.linear import Linear

class TestLinear(unittest.TestCase):
    def test_linear_backward_gradient_check(self):
        """Численная проверка градиентов для `backward` метода Linear."""
        print("\nRunning Test: Gradient check for Linear layer backward pass...")

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
        dx_num = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            old_val = x[ix]
            x[ix] = old_val + epsilon
            fx_plus = np.sum(layer.forward(x) * dout)
            x[ix] = old_val - epsilon
            fx_minus = np.sum(layer.forward(x) * dout)
            dx_num[ix] = (fx_plus - fx_minus) / (2 * epsilon)
            x[ix] = old_val
            it.iternext()
        self.assertTrue(np.allclose(dx, dx_num, rtol=1e-4, atol=1e-4), "Gradient check for dx FAILED")
        print("Gradient check for dx PASSED.")

        # --- Численная проверка dW ---
        dW_num = np.zeros_like(layer.W)
        it = np.nditer(layer.W, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            old_val = layer.W[ix]
            layer.W[ix] = old_val + epsilon
            fW_plus = np.sum(layer.forward(x) * dout)
            layer.W[ix] = old_val - epsilon
            fW_minus = np.sum(layer.forward(x) * dout)
            dW_num[ix] = (fW_plus - fW_minus) / (2 * epsilon)
            layer.W[ix] = old_val
            it.iternext()
        self.assertTrue(np.allclose(dW, dW_num, rtol=1e-4, atol=1e-4), "Gradient check for dW FAILED")
        print("Gradient check for dW PASSED.")

        # --- Численная проверка db ---
        db_num = np.zeros_like(layer.b)
        it = np.nditer(layer.b, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            old_val = layer.b[ix]
            layer.b[ix] = old_val + epsilon
            fb_plus = np.sum(layer.forward(x) * dout)
            layer.b[ix] = old_val - epsilon
            fb_minus = np.sum(layer.forward(x) * dout)
            db_num[ix] = (fb_plus - fb_minus) / (2 * epsilon)
            layer.b[ix] = old_val
            it.iternext()
        self.assertTrue(np.allclose(db, db_num, rtol=1e-4, atol=1e-4), "Gradient check for db FAILED")
        print("Gradient check for db PASSED.")

        print("All Linear gradient checks passed!")

if __name__ == "__main__":
    unittest.main()
