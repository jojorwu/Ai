"""
PyTorch implementation of the Value Head.
"""
from torch import nn

from nn_components.activations import Tanh
from nn_components.linear import Linear


class ValueHead(nn.Module):
    """
    A simple Value Head network.
    """
    def __init__(self, d_model: int, linear_class=Linear):
        super().__init__()
        self.linear = linear_class(d_model, 1, bias=False)
        self.activation = Tanh()

    def forward(self, x):
        """Forward pass."""
        return self.activation(self.linear(x))
