"""
PyTorch implementations of activation functions.
"""
import torch
from torch import nn


class Tanh(nn.Module):
    """
    Applies the Tanh activation function element-wise.
    """
    def forward(self, x):
        """Forward pass."""
        return torch.tanh(x)
