"""
Module containing the embedding layer.
"""
import numpy as np

class Embedding:
    """
    Layer for converting integer indices into dense vectors (embeddings).
    """
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weights = np.random.randn(vocab_size, d_model) * 0.01
        self.x_indices = None
        self.dweights = None

    def get_trainable_params(self):
        """Returns the trainable parameters of the layer."""
        return {'weights': (self.weights, self.dweights)}

    def forward(self, x):
        """
        Forward pass. Retrieves embeddings for the input indices.
        """
        self.x_indices = x
        return self.weights[x]

    def backward(self, dout):
        """
        Backward pass. Accumulates the gradient for the embedding matrix.
        """
        if self.dweights is None:
            self.dweights = np.zeros_like(self.weights)
        np.add.at(self.dweights, self.x_indices, dout)
