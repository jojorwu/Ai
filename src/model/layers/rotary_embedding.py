"""
PyTorch implementation of Rotary Positional Embeddings (RoPE).
"""
import torch


def precompute_rope_embeddings(d_k: int, max_seq_len: int):
    """
    Precomputes RoPE frequencies and embeddings for a given dimension and max sequence length.

    Returns a tuple of (cosines, sines) tensors.
    """
    # Create the theta term for RoPE
    theta = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float() / d_k))

    # Create the sequence positions
    seq_indices = torch.arange(max_seq_len, dtype=torch.float)

    # Outer product to get all theta * m values
    idx_theta = torch.outer(seq_indices, theta)

    # Precompute cosines and sines
    freqs = torch.polar(torch.ones_like(idx_theta), idx_theta)

    # freqs is now a complex tensor, split into real (cos) and imag (sin)
    # and reshape for broadcasting: (max_seq_len, 1, d_k)
    cos = freqs.real.unsqueeze(1)
    sin = freqs.imag.unsqueeze(1)

    return cos, sin


def apply_rope_embeddings(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seq_offset: int = 0
):
    """
    Applies RoPE to a query or key tensor.

    Args:
        x: Input tensor (query or key) of shape (batch, heads, seq_len, d_k).
        cos, sin: Precomputed RoPE tensors.
        seq_offset: The starting position in the sequence for RoPE.

    Returns:
        Tensor with RoPE applied.
    """
    # Reshape x into complex numbers
    # x is (batch, heads, seq_len, d_k)
    # x_complex is (batch, heads, seq_len, d_k/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Reshape cos and sin for broadcasting
    # cos/sin are (seq_len, 1, d_k) -> (seq_len, d_k/2) after reshape
    cos = cos[seq_offset : seq_offset + x.shape[2], :, :].squeeze(1)[:, :x_complex.shape[-1]]
    sin = sin[seq_offset : seq_offset + x.shape[2], :, :].squeeze(1)[:, :x_complex.shape[-1]]

    # Create complex rotation vector
    # rope_embed is (seq_len, d_k/2)
    rope_embed = torch.complex(cos, sin)

    # Apply rotation
    # x_complex is (batch, heads, seq_len, d_k/2)
    # rope_embed is (1, 1, seq_len, d_k/2) after unsqueeze
    x_rotated = x_complex * rope_embed.unsqueeze(0).unsqueeze(0)

    # Reshape back to real numbers
    # x_out is (batch, heads, seq_len, d_k)
    x_out = torch.view_as_real(x_rotated).flatten(3)

    return x_out.type_as(x)
