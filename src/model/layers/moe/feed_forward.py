"""
PyTorch implementation of the Feed-Forward Network (FFN) layer.
"""
import torch
from torch import nn
from torch.nn import functional as F

from src.config.model_config import FeedForwardConfig
from src.model.layers.core.linear import Linear


class FeedForward(nn.Module):
    """
    Implements the SwiGLU Feed-Forward Network layer, migrated to PyTorch.
    FFN = SwiGLU(x, W1, V, W2) = (SiLU(x @ W1) * (x @ V)) @ W2
    """
    def __init__(self, config: FeedForwardConfig, linear_class=Linear):
        super().__init__()
        # Combine w1 and w3 into one projection to increase throughput
        self.w1_w3 = linear_class(
            config.d_model, 2 * config.d_ff, bias=config.bias
        )
        self.w2 = linear_class(
            config.d_ff, config.d_model, bias=config.bias
        )

        # Apply special initialization for the output layer as in GPT-2
        if hasattr(self.w2, 'special_residual_init'):
            self.w2.special_residual_init(config.num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SwiGLU FFN using a combined projection for efficiency.
        """
        # Combined projection: [batch, seq, 2 * d_ff]
        w1_w3_out = self.w1_w3(x)

        # Split into w1 (gate) and w3 (value) paths
        w1_out, w3_out = w1_w3_out.chunk(2, dim=-1)

        # Gating mechanism: SiLU(x @ W1) * (x @ W3)
        gate_output = F.silu(w1_out) * w3_out

        # Final projection
        return self.w2(gate_output)
