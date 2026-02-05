"""
PyTorch implementation of Dropout and Stochastic Depth (DropPath) layers.
"""
import torch
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


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with any number of dims
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return drop_path(x, self.drop_prob, self.training)
