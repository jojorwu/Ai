"""
PyTorch implementation of the Embedding layer.
"""
import torch.nn as nn


class Embedding(nn.Module):
    """
    Embedding layer implemented as a wrapper around PyTorch's nn.Embedding.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Expose the weight parameter for weight tying in the main model
        self.weights = self.embedding.weight

    def forward(self, x):
        """Forward pass."""
        return self.embedding(x)
