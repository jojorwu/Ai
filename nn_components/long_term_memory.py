"""
PyTorch implementation of the Long-Term Memory (LTM) module.
"""
import torch
import torch.nn as nn

from nn_components.linear import Linear
from nn_components.activations import Tanh


class LongTermMemory(nn.Module):
    """
    A simple Long-Term Memory (LTM) network, migrated to PyTorch.
    It's a small MLP that processes a summary of the input sequence.
    """
    def __init__(self, d_model: int, d_hidden: int, num_layers: int):
        super().__init__()

        layers = []
        # Input layer
        layers.append(Linear(d_model, d_hidden))
        layers.append(Tanh())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(Linear(d_hidden, d_hidden))
            layers.append(Tanh())

        # Output layer
        layers.append(Linear(d_hidden, d_model))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LTM.
        Args:
            x: Input tensor, typically a summary of the main sequence
               (e.g., shape (batch_size, 1, d_model)).
        Returns:
            A memory context vector of shape (batch_size, 1, d_model).
        """
        return self.network(x)
