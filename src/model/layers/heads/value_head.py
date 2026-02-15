"""
PyTorch implementation of the Value Head.
"""
from __future__ import annotations
import torch
from torch import nn

from src.model.layers.core.linear import Linear


class ValueHead(nn.Module):
    """
    A multi-layer Value Head network.

    This module predicts a scalar "usefulness" score for a given sequence
    representation. It uses a 2-layer MLP with GELU activation and Tanh
    output for better expressivity and bounded range.
    """

    def __init__(self, d_model: int, linear_class: nn.Module = Linear) -> None:
        """
        Initializes the ValueHead.

        Args:
            d_model: Dimension of the input hidden states.
            linear_class: The linear layer class to use.
        """
        super().__init__()
        self.network = nn.Sequential(
            linear_class(d_model, d_model // 2),
            nn.GELU(),
            linear_class(d_model // 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ValueHead.

        Args:
            x: Input tensor of shape [batch, d_model].

        Returns:
            The predicted value score of shape [batch, 1].
        """
        return self.network(x)
