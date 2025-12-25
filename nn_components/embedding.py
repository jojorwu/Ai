"""
Implementation of the Embedding layer.
"""
from backend import np


class Embedding:
    """
    An Embedding layer that maps token IDs to dense vectors.
    """
    def __init__(self, vocab_size, d_model):
        self.W = np.random.randn(vocab_size, d_model)
        self.dW = np.zeros_like(self.W)
        self.last_x = None

    def forward(self, x):
        self.last_x = x
        return self.W[x]

    def backward(self, dout):
        """
        Performs the backward pass for the Embedding layer.
        """
        np.add.at(self.dW, self.last_x, dout)
        return

    def get_trainable_params(self):
        """
        Returns the trainable parameters and their gradients.
        """
        return {'W': (self.W, self.dW)}
