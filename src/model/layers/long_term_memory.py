"""
PyTorch implementation of the Long-Term Memory (LTM) module.
"""
import torch
from torch import nn

from src.model.layers.activations import Tanh
from src.model.layers.linear import Linear


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

        self.network = nn.Sequential(*layers)
        self.context_head = Linear(d_hidden, d_model)
        self.complexity_head = nn.Sequential(
            Linear(d_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the LTM.
        Args:
            x: Input tensor, typically a summary of the main sequence
               (e.g., shape (batch_size, 1, d_model)).
        Returns:
            A tuple containing the memory context vector and the complexity score.
        """
        hidden_state = self.network(x)
        context = self.context_head(hidden_state)
        complexity_score = self.complexity_head(hidden_state)
        return context, complexity_score
