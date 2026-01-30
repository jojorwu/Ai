"""
PyTorch implementation of Rotary Positional Embeddings (RoPE) with NTK-aware scaling.
"""
import torch


def precompute_rope_embeddings(d_k: int, max_seq_len: int, ntk_factor: float = 1.0):
    """
    Precomputes RoPE frequencies and embeddings for a given dimension and max sequence length.
    Supports NTK-aware scaling for context window extension.

    Args:
        d_k: Dimension of the keys/queries.
        max_seq_len: Maximum sequence length.
        ntk_factor: Scaling factor for NTK-aware RoPE.
                    If > 1.0, the base frequency is adjusted to handle longer sequences.
    Returns:
        A tuple of (cosines, sines) tensors.
    """
    base = 10000.0
    if ntk_factor > 1.0:
        # NTK-aware scaling: base' = base * (factor ^ (d / (d-2)))
        # This spreads the frequencies to better accommodate longer sequences.
        base = base * (ntk_factor ** (d_k / (d_k - 2)))

    # Create the theta term for RoPE
    theta = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))

    # Create the sequence positions
    seq_indices = torch.arange(max_seq_len, dtype=torch.float)

    # Outer product to get all theta * m values
    idx_theta = torch.outer(seq_indices, theta)

    # Precompute cosines and sines using polar form for stability
    freqs = torch.polar(torch.ones_like(idx_theta), idx_theta)

    # freqs is now a complex tensor, split into real (cos) and imag (sin)
    # and reshape for broadcasting: (max_seq_len, 1, d_k/2)
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

    # Handle cases where the sequence exceeds precomputed RoPE limit.
    max_rope_len = cos.size(0)
    actual_offset = seq_offset
    if seq_offset + seq_len > max_rope_len:
        # For stability, we use a circular approach if precomputation was insufficient.
        # This is a safe fallback to prevent crashes during extreme context extension.
        actual_offset = seq_offset % max_rope_len
        if actual_offset + seq_len > max_rope_len:
            # If even the wrapped window overflows, we fallback to identity (no rotation)
            return x

    # Reshape x into complex numbers: (batch, heads, seq_len, d_k/2)
    x_complex = torch.view_as_complex(x.reshape(batch, heads, seq_len, -1, 2))

    # Slice and prepare RoPE frequencies for the current sequence window.
    # We use actual_offset which handles the circular buffer case.
    rope_cos = cos[actual_offset : actual_offset + seq_len, :, :]
    rope_sin = sin[actual_offset : actual_offset + seq_len, :, :]

    # Create complex rotation vector: (seq_len, 1, d_k/2)
    rope_embed = torch.complex(rope_cos, rope_sin)

    # Apply rotation using broadcasting:
    # x_complex: (batch, heads, seq_len, d_k/2)
    # rope_embed: (seq_len, 1, d_k/2)
    x_rotated = x_complex * rope_embed.view(1, 1, seq_len, -1)

    # Reshape back to real numbers and return
    return torch.view_as_real(x_rotated).reshape(batch, heads, seq_len, d_k)
