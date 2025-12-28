"""
PyTorch implementation of the Dropout layer.
"""
from torch import nn


class Dropout(nn.Module):
    """
    Dropout layer implemented as a wrapper around PyTorch's nn.Dropout.
    The `is_training` flag is handled automatically by calling `.train()` or `.eval()`
    on the parent module.
    """
    def __init__(self, probability: float):
        super().__init__()
        self.dropout = nn.Dropout(p=probability)

    def forward(self, x):
        """Forward pass."""
        return self.dropout(x)
