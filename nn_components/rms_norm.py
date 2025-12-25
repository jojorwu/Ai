"""
This module implements the Root Mean Square Normalization layer.
"""
import numpy as np

class RMSNorm:
    """
    Implementation of Root Mean Square Normalization.
    """
    def __init__(self, d_model, epsilon=1e-5):
        self.d_model = d_model
        self.epsilon = epsilon
        self.gamma = np.ones(d_model)

        self.x = None
        self.rms = None
        self.dgamma = None

    def get_trainable_params(self):
        """Returns a dictionary with trainable parameters and their gradients."""
        return {'gamma': (self.gamma, self.dgamma)}

    def forward(self, x):
        """
        Forward pass for RMSNorm.
        y = (x / sqrt(mean(x^2) + eps)) * gamma
        """
        self.x = x
        self.rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + self.epsilon)
        normalized_x = x / self.rms
        output = self.gamma * normalized_x
        return output

    def backward(self, dout):
        """
        Backward pass for RMSNorm.
        """
        normalized_x = self.x / self.rms
        dgamma = np.sum(dout * normalized_x, axis=tuple(range(dout.ndim - 1)))

        if self.dgamma is None:
            self.dgamma = dgamma
        else:
            self.dgamma += dgamma

        d_normalized_x = dout * self.gamma
        d_rms = -np.sum(d_normalized_x * self.x, axis=-1, keepdims=True) / (self.rms**2)
        dx = (d_normalized_x / self.rms) + (d_rms * self.x / (self.d_model * self.rms))
        return dx
