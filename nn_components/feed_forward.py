"""
PyTorch implementation of the Feed-Forward Network (FFN) layer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_components.linear import Linear


class FeedForward(nn.Module):
    """
    Implements the SwiGLU Feed-Forward Network layer, migrated to PyTorch.
    FFN = SwiGLU(x, W1, V, W2) = (SiLU(x @ W1) * (x @ V)) @ W2
    """
    def __init__(self, d_model: int, d_ff: int, bias: bool = False, num_layers: int = 1):
        super().__init__()
        # The SwiGLU FFN has two linear layers in parallel for the gating mechanism,
        # followed by one output linear layer.
        self.w1 = Linear(d_model, d_ff, bias=bias)
        self.w3 = Linear(d_model, d_ff, bias=bias) # This is 'V' in the SwiGLU paper
        self.w2 = Linear(d_ff, d_model, bias=bias)

        # Apply special initialization for the output layer as in GPT-2
        self.w2.special_residual_init(num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SwiGLU FFN.
        """
        # Gating mechanism: SiLU(x @ W1) * (x @ W3)
        gate_output = F.silu(self.w1(x)) * self.w3(x)
        # Final projection
        return self.w2(gate_output)
