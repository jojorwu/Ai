"""
PyTorch implementation of the Linear layer.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Linear(nn.Linear):
    """
    A fully connected (linear) layer that inherits from torch.nn.Linear
    with custom GPT-2 style weight initialization.
    """
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__(input_dim, output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes weights with GPT-2 style initialization."""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def special_residual_init(self, num_layers):
        """Special initialization for residual connections, as in GPT-2."""
        std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.weight, mean=0.0, std=std)
