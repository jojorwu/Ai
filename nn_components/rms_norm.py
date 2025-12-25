"""
Implementation of the Root Mean Square Normalization (RMSNorm) layer.
"""
from backend import np


class RMSNorm:
    """
    Implements Root Mean Square Normalization.
    """
    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(d_model)
        # For backward pass
        self.normalized_x = None
        self.std_inv = None
        self.dgamma = np.zeros_like(self.gamma)

    def forward(self, x):
        """
        Performs the forward pass for RMSNorm.
        """
        self.std_inv = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        self.normalized_x = x * self.std_inv
        return self.normalized_x * self.gamma

    def backward(self, dout):
        """
        Performs the backward pass for RMSNorm.
        """
        dgamma = np.sum(dout * self.normalized_x, axis=(0, 1))
        d_normalized_x = dout * self.gamma
        dx = d_normalized_x * self.std_inv
        dx -= self.normalized_x * np.mean(d_normalized_x * self.std_inv * self.normalized_x, axis=-1, keepdims=True)
        # Store gradients
        self.dgamma += dgamma
        return dx

    def get_trainable_params(self):
        """
        Returns the trainable parameters and their gradients.
        """
        return {'gamma': (self.gamma, self.dgamma)}

    def get_children(self):
        """
        This layer has no children with trainable parameters in the conventional sense.
        """
        return {}
