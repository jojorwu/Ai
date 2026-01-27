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
    Applies RoPE to a query or key tensor using optimized complex number arithmetic.

    Args:
        x: Input tensor (query or key) of shape (batch, heads, seq_len, d_k).
        cos, sin: Precomputed RoPE tensors of shape (max_seq_len, 1, d_k/2).
        seq_offset: The starting position in the sequence for RoPE.

    Returns:
        Tensor with RoPE applied.
    """
    batch, heads, seq_len, d_k = x.shape

    # Reshape x into complex numbers: (batch, heads, seq_len, d_k/2)
    # We do this in the original dtype to avoid unnecessary casting where possible,
    # though view_as_complex requires the last dimension to be 2.
    x_complex = torch.view_as_complex(x.reshape(batch, heads, seq_len, -1, 2))

    # Slice and prepare RoPE frequencies for the current sequence window.
    # cos/sin are pre-calculated as (max_seq_len, 1, d_k/2)
    rope_cos = cos[seq_offset : seq_offset + seq_len, :, :]
    rope_sin = sin[seq_offset : seq_offset + seq_len, :, :]

    # Create complex rotation vector: (seq_len, 1, d_k/2)
    # This is more efficient than repeatedly unsqueezing.
    rope_embed = torch.complex(rope_cos, rope_sin)

    # Apply rotation using broadcasting:
    # x_complex: (batch, heads, seq_len, d_k/2)
    # rope_embed: (seq_len, 1, d_k/2) -> will broadcast to (batch, heads, seq_len, d_k/2)
    # Note: rope_embed needs to be (seq_len, d_k/2) and then broadcasted.
    # Current rope_embed is (seq_len, 1, d_k/2). Let's transpose it to be (1, 1, seq_len, d_k/2)
    # for cleaner broadcasting with (batch, heads, seq_len, d_k/2).
    x_rotated = x_complex * rope_embed.view(1, 1, seq_len, -1)

    # Reshape back to real numbers and return
    return torch.view_as_real(x_rotated).reshape(batch, heads, seq_len, d_k)
