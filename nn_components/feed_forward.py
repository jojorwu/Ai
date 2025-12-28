"""
PyTorch implementation of the Feed-Forward Network (FFN) layer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FeedForwardConfig
from nn_components.linear import Linear


class FeedForward(nn.Module):
    """
    Implements the SwiGLU Feed-Forward Network layer, migrated to PyTorch.
    FFN = SwiGLU(x, W1, V, W2) = (SiLU(x @ W1) * (x @ V)) @ W2
    """
    def __init__(self, config: FeedForwardConfig, linear_class=Linear):
        super().__init__()
        self.w1 = linear_class(
            config.d_model, config.d_ff, bias=config.bias)
        self.w3 = linear_class(
            config.d_model, config.d_ff, bias=config.bias)
        self.w2 = linear_class(
            config.d_ff, config.d_model, bias=config.bias)

        # Apply special initialization for the output layer as in GPT-2
        if hasattr(self.w2, 'special_residual_init'):
            self.w2.special_residual_init(config.num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SwiGLU FFN.
        """
        # Gating mechanism: SiLU(x @ W1) * (x @ W3)
        gate_output = F.silu(self.w1(x)) * self.w3(x)
        # Final projection
        return self.w2(gate_output)
