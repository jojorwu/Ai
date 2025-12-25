"""
Implementation of Rotary Positional Embeddings (RoPE).
"""
import numpy as np

class RotaryPositionalEmbedding:
    """
    Класс для создания и кеширования Rotary Positional Embeddings (RoPE).
    """
    def __init__(self, dim, max_seq_len, theta=10000.0):
        # Вычисляем частоты для каждой пары измерений
        inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))

        # Создаем матрицу позиций и частот
        t = np.arange(max_seq_len, dtype=np.float32)
        freqs = np.einsum('i,j->ij', t, inv_freq)

        # Создаем комплексные числа вида R * e^(i * m * theta_j)
        emb = np.concatenate((freqs, freqs), axis=-1)

        # Кешируем cos и sin значения
        self.cos_cached = np.cos(emb)[None, None, :, :]
        self.sin_cached = np.sin(emb)[None, None, :, :]

def apply_rotary_pos_emb(x, cos, sin):
    """
    Применяет RoPE к входному тензору x.
    x: (batch, n_heads, seq_len, dim)
    """
    # Разделяем x на две половины
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    # Применяем вращение
    # [x1, x2] -> [-x2, x1]
    rotated_x = np.stack((-x2, x1), axis=-1).reshape(x.shape)

    # y = x * cos + rotated_x * sin
    output = x * cos + rotated_x * sin

    return output

def rotary_backward(dout, x, cos, sin):
    """
    Вычисляет градиенты для RoPE.
    """
    # Градиенты:
    # dL/dx1 = dout1 * cos1 + dout2 * sin2
    # dL/dx2 = -dout1 * sin1 + dout2 * cos2

    dout1 = dout[..., 0::2]
    dout2 = dout[..., 1::2]

    # Градиенты должны использовать те же cos/sin, что и forward pass
    cos1 = cos[..., 0::2]
    cos2 = cos[..., 1::2]
    sin1 = sin[..., 0::2]
    sin2 = sin[..., 1::2]

    # dL/dx1 = dout1 * cos1 + dout2 * sin2
    # dL/dx2 = -dout1 * sin1 + dout2 * cos2
    dx1 = dout1 * cos1 + dout2 * sin2
    dx2 = -dout1 * sin1 + dout2 * cos2

    dx = np.stack((dx1, dx2), axis=-1).reshape(x.shape)

    return dx
