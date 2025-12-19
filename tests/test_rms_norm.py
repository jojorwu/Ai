import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_components.rms_norm import RMSNorm

def test_rms_norm_backward_gradient_check():
    """Численная проверка градиентов для `backward` метода RMSNorm."""
    print("\nRunning Test: Gradient check for RMSNorm backward pass...")

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
    dgamma_num = np.zeros_like(norm.gamma)
    for i in range(d_model):
        old_val = norm.gamma[i]
        norm.gamma[i] = old_val + epsilon
        fx_plus = np.sum(norm.forward(x) * dout)
        norm.gamma[i] = old_val - epsilon
        fx_minus = np.sum(norm.forward(x) * dout)
        dgamma_num[i] = (fx_plus - fx_minus) / (2 * epsilon)
        norm.gamma[i] = old_val
    assert np.allclose(dgamma, dgamma_num, rtol=1e-4, atol=1e-4), "Gradient check for dgamma FAILED"
    print("Gradient check for dgamma PASSED.")

    # --- Численная проверка dx ---
    dx_num = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_val = x[ix]
        x[ix] = old_val + epsilon
        fx_plus = np.sum(norm.forward(x) * dout)
        x[ix] = old_val - epsilon
        fx_minus = np.sum(norm.forward(x) * dout)
        dx_num[ix] = (fx_plus - fx_minus) / (2 * epsilon)
        x[ix] = old_val
        it.iternext()
    assert np.allclose(dx, dx_num, rtol=1e-4, atol=1e-4), "Gradient check for dx FAILED"
    print("Gradient check for dx PASSED.")

    print("All RMSNorm gradient checks passed!")

if __name__ == "__main__":
    test_rms_norm_backward_gradient_check()
