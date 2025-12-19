import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_components.rotary_embedding import RotaryPositionalEmbedding, apply_rotary_pos_emb, rotary_backward

def test_rotary_embedding_backward_gradient_check():
    """Численная проверка градиентов для `rotary_backward` функции."""
    print("\nRunning Test: Gradient check for Rotary Positional Embedding backward pass...")

    batch_size, n_heads, seq_len, dim = 2, 4, 8, 16

    np.random.seed(42)

    # 1. Инициализация RoPE для получения cos/sin кеша
    rope = RotaryPositionalEmbedding(dim, max_seq_len=seq_len)
    cos = rope.cos_cached[:, :, :seq_len, :]
    sin = rope.sin_cached[:, :, :seq_len, :]

    # 2. Инициализация входных данных
    x = np.random.randn(batch_size, n_heads, seq_len, dim)
    dout = np.random.randn(batch_size, n_heads, seq_len, dim)

    # 3. Аналитический градиент
    dx_analytic = rotary_backward(dout, x, cos, sin)

    # 4. Численный градиент
    epsilon = 1e-5
    dx_numerical = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        original_value = x[ix]

        # f(x + h)
        x[ix] = original_value + epsilon
        fx_plus_h = np.sum(apply_rotary_pos_emb(x, cos, sin) * dout)

        # f(x - h)
        x[ix] = original_value - epsilon
        fx_minus_h = np.sum(apply_rotary_pos_emb(x, cos, sin) * dout)

        dx_numerical[ix] = (fx_plus_h - fx_minus_h) / (2 * epsilon)

        # Restore original value
        x[ix] = original_value
        it.iternext()

    # 5. Сравнение
    assert np.allclose(dx_analytic, dx_numerical, rtol=1e-4, atol=1e-4), "Gradient check for dx FAILED"
    print("Gradient check for dx PASSED.")
    print("All Rotary Embedding gradient checks passed!")

if __name__ == "__main__":
    test_rotary_embedding_backward_gradient_check()
