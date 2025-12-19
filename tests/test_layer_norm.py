import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_components.layer_norm import LayerNormalization

def test_layer_normalization_backward_gradient_check():
    """Численная проверка градиентов для `backward` метода LayerNormalization."""
    print("\nRunning Test: Gradient check for LayerNormalization backward pass...")

    batch_size, seq_len, d_model = 2, 3, 8

    np.random.seed(42)
    layer = LayerNormalization(d_model)
    # Инициализируем gamma и beta случайными значениями для более общей проверки
    layer.gamma = np.random.randn(d_model)
    layer.beta = np.random.randn(d_model)

    x = np.random.randn(batch_size, seq_len, d_model)
    dout = np.random.randn(batch_size, seq_len, d_model)

    # --- Аналитические градиенты ---
    _ = layer.forward(x)
    dx = layer.backward(dout)
    dgamma = layer.dgamma
    dbeta = layer.dbeta

    epsilon = 1e-5

    # --- Численная проверка dgamma ---
    dgamma_num = np.zeros_like(layer.gamma)
    for i in range(d_model):
        old_val = layer.gamma[i]
        layer.gamma[i] = old_val + epsilon
        fx_plus = np.sum(layer.forward(x) * dout)
        layer.gamma[i] = old_val - epsilon
        fx_minus = np.sum(layer.forward(x) * dout)
        dgamma_num[i] = (fx_plus - fx_minus) / (2 * epsilon)
        layer.gamma[i] = old_val
    assert np.allclose(dgamma, dgamma_num, rtol=1e-4, atol=1e-4), "Gradient check for dgamma FAILED"
    print("Gradient check for dgamma PASSED.")

    # --- Численная проверка dbeta ---
    dbeta_num = np.zeros_like(layer.beta)
    for i in range(d_model):
        old_val = layer.beta[i]
        layer.beta[i] = old_val + epsilon
        fx_plus = np.sum(layer.forward(x) * dout)
        layer.beta[i] = old_val - epsilon
        fx_minus = np.sum(layer.forward(x) * dout)
        dbeta_num[i] = (fx_plus - fx_minus) / (2 * epsilon)
        layer.beta[i] = old_val
    assert np.allclose(dbeta, dbeta_num, rtol=1e-4, atol=1e-4), "Gradient check for dbeta FAILED"
    print("Gradient check for dbeta PASSED.")

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
    assert np.allclose(dx, dx_num, rtol=1e-4, atol=1e-4), "Gradient check for dx FAILED"
    print("Gradient check for dx PASSED.")

    print("All LayerNormalization gradient checks passed!")

if __name__ == "__main__":
    test_layer_normalization_backward_gradient_check()
