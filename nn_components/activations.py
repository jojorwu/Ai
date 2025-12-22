import numpy as np

class SiLU:
    """
    Sigmoid Linear Unit (SiLU) activation function, also known as Swish.
    f(x) = x * sigmoid(x)
    """
    def __init__(self):
        self.x = None
        self.sigmoid_x = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        self.sigmoid_x = self._sigmoid(x)
        return x * self.sigmoid_x

    def backward(self, dy):
        """
        Computes the backward pass of the SiLU activation.
        d/dx(x * sig(x)) = sig(x) + x * (sig(x) * (1 - sig(x)))
                        = sig(x) * (1 + x * (1 - sig(x)))
        """
        dsigmoid_dx = self.sigmoid_x * (1 - self.sigmoid_x)
        dx = self.sigmoid_x + self.x * dsigmoid_dx
        return dy * dx

class Tanh:
    """Hyperbolic tangent activation function."""
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, dout):
        # Derivative of tanh(x) = 1 - tanh^2(x)
        return dout * (1 - self.output**2)
