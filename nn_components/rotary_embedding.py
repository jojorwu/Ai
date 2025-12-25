"""
This module implements Rotary Positional Embeddings (RoPE).
"""
import numpy as np

class RotaryPositionalEmbedding:
    """
    Class for creating and caching Rotary Positional Embeddings (RoPE).
    """
    def __init__(self, dim, max_seq_len, theta=10000.0):
        # Calculate frequencies for each pair of dimensions
        inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))

        # Create a matrix of positions and frequencies
        t = np.arange(max_seq_len, dtype=np.float32)
        freqs = np.einsum('i,j->ij', t, inv_freq)

        # Create complex numbers of the form R * e^(i * m * theta_j)
        emb = np.concatenate((freqs, freqs), axis=-1)

        # Cache cos and sin values
        self.cos_cached = np.cos(emb)[None, None, :, :]
        self.sin_cached = np.sin(emb)[None, None, :, :]

def apply_rotary_pos_emb(x, cos, sin):
    """
    Applies RoPE to the input tensor x.
    x: (batch, n_heads, seq_len, dim)
    """
    # Split x into two halves
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    # Apply rotation
    # [x1, x2] -> [-x2, x1]
    rotated_x = np.stack((-x2, x1), axis=-1).reshape(x.shape)

    # y = x * cos + rotated_x * sin
    output = x * cos + rotated_x * sin

    return output

def rotary_backward(dout, x, cos, sin):
    """
    Calculates the gradients for RoPE.
    """
    # Split x into two halves
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    # Forward transformation:
    # y1 = x1 * cos1 + (-x2 * sin1)
    # y2 = x2 * cos2 + ( x1 * sin2)

    # Gradients:
    # dL/dx1 = dL/dy1 * dy1/dx1 + dL/dy2 * dy2/dx1
    # dy1/dx1 = cos1
    # dy2/dx1 = sin2 (for identical cos/sin)
    # dL/dx1 = dout1 * cos1 + dout2 * sin2

    # dL/dx2 = dL/dy1 * dy1/dx2 + dL/dy2 * dy2/dx2
    # dy1/dx2 = -sin1
    # dy2/dx2 = cos2
    # dL/dx2 = dout1 * (-sin1) + dout2 * cos2

    dout1 = dout[..., 0::2]
    dout2 = dout[..., 1::2]

    # Gradients should use the same cos/sin as the forward pass
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
