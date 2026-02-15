"""
PyTorch implementation of Scaled Dot-Product Attention using the optimized
built-in PyTorch function.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AttentionInput:
    """
    Encapsulates the input for the ScaledDotProductAttention layer.
    """
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    mask: Optional[torch.Tensor] = None
    is_causal: bool = False
    scale: Optional[float] = None


class ScaledDotProductAttention(nn.Module):
    """
    A wrapper for the highly optimized `torch.nn.functional.scaled_dot_product_attention`.
    This implementation automatically handles causal masking when `is_causal=True`.
    """

    def forward(self, inputs: AttentionInput) -> torch.Tensor:
        """
        Forward pass for Scaled Dot-Product Attention.

        Args:
            inputs: An AttentionInput object containing q, k, v, mask, and is_causal.

        Returns:
            The output tensor after applying attention.
        """
        if inputs.is_causal and inputs.mask is not None:
            raise ValueError("`is_causal` and `mask` are mutually exclusive.")

        # The built-in function is highly optimized and can use backends
        # like FlashAttention if available.
        # pylint: disable=not-callable
        return F.scaled_dot_product_attention(
            inputs.q,
            inputs.k,
            inputs.v,
            attn_mask=inputs.mask,
            dropout_p=0.0,
            is_causal=inputs.is_causal,
            scale=inputs.scale,
        )
