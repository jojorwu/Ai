"""
PyTorch implementation of a Gated Cross-Attention mechanism for LTM fusion.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

from src.model.layers.core.linear import Linear
from src.model.layers.core.rms_norm import RMSNorm

class GatedCrossAttention(nn.Module):
    """
    Implements a gated cross-attention mechanism to fuse LTM context into hidden states.

    This allows the model to selectively attend to retrieved long-term memory
    and integrate it with the current context using a learnable gating mechanism.
    """

    def __init__(self, d_model: int, num_heads: int = 4, linear_class: nn.Module = Linear) -> None:
        """
        Initializes the GatedCrossAttention layer.

        Args:
            d_model: Dimension of the hidden states.
            num_heads: Number of attention heads.
            linear_class: The linear layer class to use.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = linear_class(d_model, d_model, bias=False)
        self.k_proj = linear_class(d_model, d_model, bias=False)
        self.v_proj = linear_class(d_model, d_model, bias=False)
        self.out_proj = linear_class(d_model, d_model, bias=False)

        self.norm = RMSNorm(d_model)

        # Gating parameter: initialized to 0 to start with an identity-like behavior
        self.gate = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for gated cross-attention.

        Args:
            x: Query hidden states [batch, seq_len, d_model].
            context: Context tokens to attend to [batch, context_len, d_model].

        Returns:
            The hidden states with fused context.
        """
        batch_size, seq_len, _ = x.shape
        _, context_len, _ = context.shape

        # 1. Projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        # Note: No causal mask needed for cross-attention usually,
        # as we attend to the full LTM context.
        attn_out = F.scaled_dot_product_attention(q, k, v)

        # 3. Merge heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)

        # 4. Gated residual connection
        # Use sigmoid(gate) to bound the influence initially
        return x + torch.sigmoid(self.gate) * attn_out
