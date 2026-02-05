"""
Implements multi-head learned pooling for sequence summarization in Titans.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

from src.model.layers.core.linear import Linear


class SummaryNetwork(nn.Module):
    """
    A multi-head attention-based pooling network for sequence summarization.

    Instead of simple mean pooling or single-head weighting, this network learns
    to extract multiple 'features' from a sequence using parallel attention heads,
    providing a richer summary for Long-Term Memory.
    """

    def __init__(self, d_model: int, num_heads: int = 4) -> None:
        """
        Initializes the SummaryNetwork.

        Args:
            d_model: Dimension of the input hidden states.
            num_heads: Number of parallel pooling heads.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        # Key and Value projections for summarization
        self.k_proj = Linear(d_model, d_model, bias=False)
        self.v_proj = Linear(d_model, d_model, bias=False)

        # Learnable query tokens for each head (one query per head)
        self.queries = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim))

        # Output projection
        self.out_proj = Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Summarizes a sequence using multi-head attention pooling.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model].

        Returns:
            A summarized tensor of shape [batch, 1, d_model].
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project to keys and values
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Multi-head attention pooling
        # q: [batch, num_heads, 1, head_dim]
        # k: [batch, num_heads, seq_len, head_dim]
        q = self.queries.expand(batch_size, -1, -1, -1)

        # [batch, num_heads, 1, seq_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # [batch, num_heads, 1, head_dim]
        summary_heads = torch.matmul(attn_weights, v)

        # 3. Concatenate and project
        summary = summary_heads.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        return self.out_proj(summary)
