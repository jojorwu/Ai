"""
Implementation of the Embedding layer.
"""
from backend import np


class Embedding:
    """
    An Embedding layer that maps token IDs to dense vectors.
    """
    def __init__(self, vocab_size, d_model):
        self.weights = np.random.randn(vocab_size, d_model)
        self.dweights = np.zeros_like(self.weights)
        self.last_x = None

    def forward(self, x):
        """Performs the forward pass of the embedding layer."""
        self.last_x = x
        return self.weights[x]

    def backward(self, dout):
        """
        Performs the backward pass for the Embedding layer.
        """
        np.add.at(self.dweights, self.last_x, dout)
        return

    def get_trainable_params(self):
        """
        Returns the trainable parameters and their gradients.
        """
        return {'weights': (self.weights, self.dweights)}
