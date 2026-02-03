"""
PyTorch implementation of the standard Feed-Forward Network (FFN) layer.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

from src.config.model_config import FeedForwardConfig
from src.model.layers.core.linear import Linear
from src.model.layers.core.rms_norm import RMSNorm


class FeedForward(nn.Module):
    """
    Implements a standard 2-layer Feed-Forward Network with SwiGLU activation.

    This architecture uses a combined projection for W1 and W3 for efficiency,
    following the Llama-style SwiGLU FFN.
    """

    def __init__(
        self, config: FeedForwardConfig, linear_class: nn.Module = Linear
    ) -> None:
        """
        Initializes the FeedForward layer.

        Args:
            config: Configuration for the FFN layer.
            linear_class: The linear layer class to use.
        """
        super().__init__()
        # Combined projection: [d_model, 2 * d_ff] to increase throughput.
        self.w1_w3 = linear_class(config.d_model, 2 * config.d_ff, bias=config.bias)
        # Final projection: [d_ff, d_model]
        self.w2 = linear_class(config.d_ff, config.d_model, bias=config.bias)

        # Internal normalization for improved gradient flow in high-capacity FFNs
        self.internal_norm = (
            RMSNorm(config.d_ff) if config.use_internal_norm else None
        )

        if hasattr(self.w2, "special_residual_init"):
            self.w2.special_residual_init(config.num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SwiGLU FFN using a combined projection.

        Args:
            x: Input tensor.

        Returns:
            The processed output tensor.
        """
        # Combined projection: [batch, seq, 2 * d_ff]
        w1_w3_out = self.w1_w3(x)

        # Split into w1 (gate) and w3 (value) paths
        w1_out, w3_out = w1_w3_out.chunk(2, dim=-1)

        # SwiGLU: SiLU(xW1) * (xW3)
        hidden = F.silu(w1_out) * w3_out

        if self.internal_norm:
            hidden = self.internal_norm(hidden)

        # Final projection
        return self.w2(hidden)
