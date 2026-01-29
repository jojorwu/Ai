"""
PyTorch implementation of the FiLM (Feature-wise Linear Modulation) layer.
"""
from torch import nn

from src.model.layers.core.linear import Linear


class FiLMLayer(nn.Module):
    """
    A FiLM (Feature-wise Linear Modulation) layer.

    This layer generates scale (gamma) and shift (beta) parameters from a
    conditioning vector (e.g., from a Long-Term Memory module) and applies
    them to an input tensor.
    """

    def __init__(self, d_model: int, linear_class=Linear):
        """
        Initializes the FiLMLayer.

        Args:
            d_model: The dimensionality of the input and conditioning vectors.
            linear_class: The class to use for the linear projection.
        """
        super().__init__()
        # Project the LTM state to get gamma and beta (2 * d_model)
        self.projection = linear_class(d_model, 2 * d_model, bias=True)
        self.d_model = d_model

    def forward(self, x, ltm_state):
        """
        Applies the FiLM transformation.

        Args:
            x: The input tensor of shape (batch, seq_len, d_model).
            ltm_state: The conditioning tensor from the LTM of shape
                       (batch, 1, d_model).

        Returns:
            The modulated tensor of the same shape as x.
        """
        # Project LTM state and split into gamma and beta
        projected = self.projection(ltm_state)
        gamma, beta = projected.chunk(2, dim=-1)

        # Apply modulation
        return gamma * x + beta
