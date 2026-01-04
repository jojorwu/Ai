"""
PyTorch implementation of Scaled Dot-Product Attention using the optimized
built-in PyTorch function.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

try:
    from flash_attn.flash_attn_interface import flash_attn_func
    # To support इज़_causal, we need at least version 2.0.
    from flash_attn import __version__ as flash_attn_version
    FLASH_ATTENTION_AVAILABLE = True if flash_attn_version >= "2.0.0" else False
except ImportError:
    flash_attn_func = None
    FLASH_ATTENTION_AVAILABLE = False


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

        use_flash_attention = (
            FLASH_ATTENTION_AVAILABLE and
            inputs.mask is None and # Flash Attention has its own causal implementation
            inputs.q.device.type == 'cuda' and
            inputs.q.dtype in (torch.float16, torch.bfloat16)
        )

        if use_flash_attention:
            return flash_attn_func(
                inputs.q, inputs.k, inputs.v, causal=inputs.is_causal
            )

        # Fallback to the standard PyTorch implementation
        return F.scaled_dot_product_attention(
            inputs.q, inputs.k, inputs.v, attn_mask=inputs.mask, is_causal=inputs.is_causal
        )
