import numpy as np
import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_components.feed_forward import FeedForward

class TestFeedForward(unittest.TestCase):
    def test_swiglu_feed_forward_backward_gradient_check(self):
        """
        Численно проверяет градиенты для backward метода FeedForward (SwiGLU).
        """
        print("\nRunning Test: Gradient check for SwiGLU FeedForward backward pass...")
        batch_size, seq_len, d_model, d_ff = 2, 5, 16, 32

        np.random.seed(1337)

        # Инициализация
        ffn = FeedForward(d_model, d_ff)
        x = np.random.randn(batch_size, seq_len, d_model)
        dout = np.random.randn(batch_size, seq_len, d_model)

        # Прямой и обратный проход для получения аналитических градиентов
        _ = ffn.forward(x)
        dx_analytic = ffn.backward(dout)

        epsilon = 1e-5

        # --- Численная проверка градиента для входа dx ---
        print("Checking gradients for input: dx...")
        dx_numerical = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            original_value = x[ix]

            x[ix] = original_value + epsilon
            fx_plus_h = np.sum(ffn.forward(x) * dout)

            x[ix] = original_value - epsilon
            fx_minus_h = np.sum(ffn.forward(x) * dout)

            dx_numerical[ix] = (fx_plus_h - fx_minus_h) / (2 * epsilon)

            x[ix] = original_value
            it.iternext()

        self.assertTrue(np.allclose(dx_analytic, dx_numerical, rtol=1e-4, atol=1e-4), "Gradient check for input dx FAILED")
        print("Gradient check for input dx PASSED.")

        # --- Численная проверка градиентов для весов ---
        all_linear_layers = {'w1': ffn.w1, 'w2': ffn.w2, 'w3': ffn.w3}
        for layer_name, layer_obj in all_linear_layers.items():
            params = layer_obj.get_trainable_params()
            for p_name, (p_param, p_grad) in params.items():
                print(f"Checking gradients for parameter: {layer_name}.{p_name}...")

                grad_numerical = np.zeros_like(p_param)
                it = np.nditer(p_param, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    ix = it.multi_index
                    original_value = p_param[ix]

                    p_param[ix] = original_value + epsilon
                    fx_plus_h = np.sum(ffn.forward(x) * dout)

                    p_param[ix] = original_value - epsilon
                    fx_minus_h = np.sum(ffn.forward(x) * dout)

                    grad_numerical[ix] = (fx_plus_h - fx_minus_h) / (2 * epsilon)

                    p_param[ix] = original_value
                    it.iternext()

                self.assertTrue(np.allclose(p_grad, grad_numerical, rtol=1e-4, atol=1e-4), f"Gradient check for {layer_name}.{p_name} FAILED")
                print(f"Gradient check for parameter {layer_name}.{p_name} PASSED.")

        print("All SwiGLU FeedForward gradient checks passed!")

if __name__ == "__main__":
    unittest.main()
