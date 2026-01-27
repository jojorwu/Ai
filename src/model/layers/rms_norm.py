"""
PyTorch implementation of Root Mean Square Normalization (RMSNorm).
"""
import torch
from torch import nn


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
        Forward pass for RMSNorm, optimized for performance and numerical stability.
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            Normalized tensor of the same shape.
        """
        # Perform calculation in float32 for stability
        input_dtype = x.dtype
        x = x.to(torch.float32)

        # Calculate the root mean square using rsqrt for efficiency
        # rms = 1 / sqrt(mean(x^2) + eps)
        inv_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # Normalize the input, scale by gamma, and cast back to original dtype
        return (x * inv_rms).to(input_dtype) * self.gamma
