"""
PyTorch implementation of Root Mean Square Normalization (RMSNorm).
"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Normalization, migrated to PyTorch.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            Normalized tensor of the same shape.
        """
        # Calculate the root mean square of the last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize the input and scale by gamma
        return (x / rms) * self.gamma
