"""
PyTorch implementation of the Gated FiLM (Feature-wise Linear Modulation) layer.
"""
import torch
from torch import nn

from src.model.layers.core.linear import Linear


class FiLMLayer(nn.Module):
    """
    A Gated FiLM (Feature-wise Linear Modulation) layer.

    This layer generates scale (gamma), shift (beta), and a gating parameter
    from a conditioning vector (e.g., from a Long-Term Memory module)
    and applies them to an input tensor via a gated residual connection.
    """

    def __init__(self, d_model: int, linear_class=Linear):
        """
        Initializes the FiLMLayer.

        Args:
            d_model: The dimensionality of the input and conditioning vectors.
            linear_class: The class to use for the linear projection.
        """
        super().__init__()
        # Project the LTM state to get gamma, beta, and gate (3 * d_model)
        self.projection = linear_class(d_model, 3 * d_model, bias=True)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, ltm_state: torch.Tensor) -> torch.Tensor:
        """
        Applies the Gated FiLM transformation.

        Args:
            x: The input tensor of shape (batch, seq_len, d_model).
            ltm_state: The conditioning tensor from the LTM of shape
                       (batch, 1, d_model).

        Returns:
            The modulated tensor of the same shape as x.
        """
        # Project LTM state and split into gamma, beta, and gate
        projected = self.projection(ltm_state)
        gamma, beta, gate_logits = projected.chunk(3, dim=-1)

        # Apply gated modulation as a residual: x + sigmoid(gate) * (gamma * x + beta)
        # We use torch.sigmoid for the gate
        gate = torch.sigmoid(gate_logits)

        modulated = gamma * x + beta
        return x + gate * modulated
