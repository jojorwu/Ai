"""
PyTorch implementation of Scaled Dot-Product Attention using the optimized
built-in PyTorch function.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    """
    A wrapper for the highly optimized `torch.nn.functional.scaled_dot_product_attention`.
    This implementation automatically handles causal masking when `is_causal=True`.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for Scaled Dot-Product Attention.

        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.
            mask: Optional attention mask.
            is_causal: If True, applies a causal mask for autoregressive decoding.
                       This argument is mutually exclusive with `mask`.

        Returns:
            The output tensor after applying attention.
        """
        if is_causal and mask is not None:
            raise ValueError("`is_causal` and `mask` are mutually exclusive.")

        # The built-in function is highly optimized and can use backends
        # like FlashAttention if available.
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal
        )
