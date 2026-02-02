"""
Implements learned pooling for sequence summarization in Titans.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

from src.model.layers.core.linear import Linear


class SummaryNetwork(nn.Module):
    """
    A lightweight attention-based pooling network for sequence summarization.

    Instead of simple mean pooling, this network learns to weight different
    tokens in a chunk to create a more informative summary for Long-Term Memory.
    """

    def __init__(self, d_model: int) -> None:
        """
        Initializes the SummaryNetwork.

        Args:
            d_model: Dimension of the input hidden states.
        """
        super().__init__()
        # Use a small MLP to compute importance weights for each token.
        self.weight_net = nn.Sequential(
            Linear(d_model, d_model // 2),
            nn.GELU(),
            Linear(d_model // 2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Summarizes a sequence of hidden states into a single vector.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model].

        Returns:
            A summarized tensor of shape [batch, 1, d_model].
        """
        # Compute raw weights [batch, seq_len, 1]
        raw_weights = self.weight_net(x)

        # Normalize weights across the sequence dimension
        weights = F.softmax(raw_weights, dim=1)

        # Compute weighted sum
        summary = torch.sum(x * weights, dim=1, keepdim=True)

        return summary
