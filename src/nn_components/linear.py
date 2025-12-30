"""
PyTorch implementation of the Linear layer.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Linear(nn.Module):
    """
    A fully connected (linear) layer, migrated to PyTorch.
    """
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = bias

        self.weights = nn.Parameter(torch.empty(output_dim, input_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes weights with GPT-2 style initialization."""
        nn.init.normal_(self.weights, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def special_residual_init(self, num_layers):
        """Special initialization for residual connections, as in GPT-2."""
        std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.weights, mean=0.0, std=std)

    def forward(self, x):
        """Forward pass."""
        return F.linear(x, self.weights, self.bias)  # pylint: disable=not-callable
