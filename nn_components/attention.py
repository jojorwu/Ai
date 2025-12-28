"""
PyTorch implementation of Scaled Dot-Product Attention.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Computes Scaled Dot-Product Attention, migrated to PyTorch.
    """
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for Scaled Dot-Product Attention.

        Args:
            q: Query tensor, shape (batch, heads, seq_len_q, d_k).
            k: Key tensor, shape (batch, heads, seq_len_k, d_k).
            v: Value tensor, shape (batch, heads, seq_len_v, d_v). Note: seq_len_k == seq_len_v.
            mask: Optional mask tensor.

        Returns:
            Output tensor and attention weights.
        """
        d_k = q.size(-1)
        # (batch, heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)

        # (batch, heads, seq_len_q, d_v)
        output = torch.matmul(attn_weights, v)

        return output
